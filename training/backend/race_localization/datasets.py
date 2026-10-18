"""Validated camera packages; query labels are never placed in session observations."""
from __future__ import annotations

import hashlib
import io
import json
import math
import re
import shutil
import stat
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError

from .image_identity import decode_color, rgb_hash

MAX_UPLOAD = 256 * 1024 * 1024
MAX_EXPANDED = 1024 * 1024 * 1024
MAX_FILES = 5000
MAX_PIXELS = 16_000_000
MAX_DIMENSION = 8192
SPLITS = {"reference", "train", "development", "audit"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _unique_json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate manifest field: {key}")
        result[key] = value
    return result


def _invalid_json_constant(value):
    raise ValueError(f"Manifest values must be finite; found {value}")


def _number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
        raise ValueError(f"{field} must be a finite number")
    return float(value)


def _keys(value: Any, allowed: set[str], required: set[str], field: str) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be an object")
    extra, missing = set(value) - allowed, required - set(value)
    if extra or missing:
        raise ValueError(f"{field}: unknown fields {sorted(extra)}; missing fields {sorted(missing)}")
    return value


def _identifier(value: Any, field: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", value):
        raise ValueError(f"{field} must be a safe identifier of 1 to 128 characters")
    return value


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 256:
        raise ValueError(f"{field} must be nonempty text up to 256 characters")
    return value.strip()


def _matrix(value: Any, shape: tuple[int, int], field: str) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=float)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{field} must be a numeric {shape} matrix") from exc
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"{field} must be a finite {shape} matrix")
    # Numeric conversion must not silently accept booleans or numeric strings.
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) for row in value for item in row):
        raise ValueError(f"{field} must contain numeric values")
    return result


def _transform(value: Any, field: str) -> list:
    matrix = _matrix(value, (4, 4), field)
    rotation = matrix[:3, :3]
    if (not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-6)
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-4)
            or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-4)):
        raise ValueError(f"{field} must be a rigid homogeneous transform with a proper rotation")
    return matrix.tolist()


def _calibration(value: Any) -> dict:
    value = _keys(value, {"image_width", "image_height", "K", "distortion", "T_CV"},
                  {"image_width", "image_height", "K", "distortion", "T_CV"}, "calibration")
    width, height = value["image_width"], value["image_height"]
    if (type(width) is not int or type(height) is not int or width < 1 or height < 1
            or width > MAX_DIMENSION or height > MAX_DIMENSION or width * height > MAX_PIXELS):
        raise ValueError("calibration image dimensions exceed the supported bounds")
    intrinsic = _matrix(value["K"], (3, 3), "calibration.K")
    if intrinsic[0, 0] <= 0 or intrinsic[1, 1] <= 0 or not np.array_equal(intrinsic[2], [0, 0, 1]):
        raise ValueError("calibration.K must have positive focal lengths and a [0,0,1] final row")
    if intrinsic[0, 1] != 0 or intrinsic[1, 0] != 0:
        raise ValueError("calibration.K must use canonical OpenCV intrinsics with zero skew and cross-axis terms")
    distortion = value["distortion"]
    if not isinstance(distortion, list) or len(distortion) not in {0, 4, 5, 8, 12, 14}:
        raise ValueError("calibration.distortion must have 0, 4, 5, 8, 12, or 14 OpenCV coefficients")
    return {"image_width": width, "image_height": height, "K": intrinsic.tolist(),
            "distortion": [_number(x, "distortion") for x in distortion],
            "T_CV": _transform(value["T_CV"], "calibration.T_CV")}


def _relative_path(value: Any) -> str:
    if not isinstance(value, str) or not value or "\\" in value or ":" in value or "\x00" in value:
        raise ValueError("Archive paths must be relative POSIX paths")
    parts = PurePosixPath(value).parts
    if not parts or PurePosixPath(value).is_absolute() or any(p in {"..", "."} for p in value.split("/")):
        raise ValueError("Archive path traversal is forbidden")
    if any(p.endswith((".", " ")) for p in parts):
        raise ValueError("Ambiguous Windows paths are forbidden")
    reserved = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}
    if any(p.split(".")[0].upper() in reserved for p in parts):
        raise ValueError("Reserved device paths are forbidden")
    return PurePosixPath(value).as_posix()


def _package_file(directory: Path, relative: str) -> Path:
    relative = _relative_path(relative)
    candidate = directory.joinpath(*PurePosixPath(relative).parts)
    if not candidate.resolve().is_relative_to(directory.resolve()):
        raise ValueError("Package image escapes its root")
    cursor = candidate
    while cursor != directory:
        if cursor.is_symlink():
            raise ValueError("Package symlinks are forbidden")
        cursor = cursor.parent
    if not candidate.is_file():
        raise ValueError(f"Missing image: {relative}")
    return candidate


def _image_digest(path: Path, calibration: dict) -> tuple[str, str]:
    if path.suffix.lower() not in IMAGE_SUFFIXES:
        raise ValueError(f"Unsupported camera image format: {path.name}")
    raw = path.read_bytes()
    # Read headers for all formats before OpenCV allocates a decoded pixel array.
    try:
        with Image.open(io.BytesIO(raw)) as header:
            width, height = header.size
            if width > MAX_DIMENSION or height > MAX_DIMENSION or width * height > MAX_PIXELS:
                raise ValueError("Decoded image dimensions exceed limits")
            if header.getexif().get(274, 1) != 1:
                raise ValueError("Nonidentity EXIF orientation is unsupported; orient pixels and calibration explicitly")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ValueError(f"Undecodable or oversized image: {path.name}") from exc
    try:
        image = decode_color(raw)
    except ValueError as exc:
        raise ValueError(f"Undecodable image: {path.name}") from exc
    height, width = image.shape[:2]
    if width > MAX_DIMENSION or height > MAX_DIMENSION or width * height > MAX_PIXELS:
        raise ValueError("Decoded image dimensions exceed limits")
    if (width, height) != (calibration["image_width"], calibration["image_height"]):
        raise ValueError(f"Image dimensions differ from calibration: {path.name}")
    return hashlib.sha256(raw).hexdigest(), rgb_hash(image)


def _track(value: Any) -> dict:
    value = _keys(value, {"centerline", "left_boundary", "right_boundary"}, {"centerline"}, "track")
    result = {}
    for key, rows in value.items():
        if not isinstance(rows, list) or len(rows) < 2 or len(rows) > 20_000:
            raise ValueError(f"track.{key} must contain 2 to 20000 points")
        result[key] = []
        for point in rows:
            if not isinstance(point, list) or len(point) != 3:
                raise ValueError(f"track.{key} points must have three coordinates")
            result[key].append([_number(x, f"track.{key}") for x in point])
    return result


def _evaluation_context(value: Any) -> dict:
    if value == {}:
        return {}
    value = _keys(value, {"pose_target", "orientation_metric", "world_frame", "alignment_status"},
                  {"pose_target", "orientation_metric", "world_frame", "alignment_status"}, "evaluation")
    target = value["pose_target"]
    if not isinstance(target, str) or target not in {"camera", "vehicle"}:
        raise ValueError("evaluation.pose_target must be camera or vehicle")
    orientation = "geodesic" if target == "camera" else "yaw"
    if value["orientation_metric"] != orientation:
        raise ValueError("evaluation orientation metric disagrees with pose target")
    alignment = value["alignment_status"]
    if not isinstance(alignment, str) or alignment not in {"inferred_shared", "documented_shared", "synthetic_known", "unverified"}:
        raise ValueError("Invalid evaluation.alignment_status")
    return {"pose_target": target, "orientation_metric": orientation,
            "world_frame": _identifier(value["world_frame"], "evaluation.world_frame"),
            "alignment_status": alignment}


def _bounded_metadata(value: Any, field: str) -> dict:
    """Preserve descriptive benchmark data without admitting a label side channel."""
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be an object")
    nodes = 0
    forbidden = {"ground_truth", "labels", "T_WC", "T_WV", "position_m", "yaw_rad", "reference_poses"}
    def visit(item, depth=0):
        nonlocal nodes
        nodes += 1
        if depth > 8 or nodes > 10000:
            raise ValueError(f"{field} exceeds metadata depth or item limits")
        if isinstance(item, dict):
            if len(item) > 128 or any(not isinstance(k, str) or len(k) > 128 or k in forbidden for k in item):
                raise ValueError(f"{field} contains unsupported or label metadata keys")
            for nested in item.values():
                visit(nested, depth + 1)
        elif isinstance(item, list):
            if len(item) > 2000:
                raise ValueError(f"{field} metadata array is too long")
            for nested in item:
                visit(nested, depth + 1)
        elif isinstance(item, str):
            if len(item) > 4096:
                raise ValueError(f"{field} metadata text is too long")
        elif item is None or isinstance(item, bool):
            pass
        elif not isinstance(item, (int, float)) or not math.isfinite(item):
            raise ValueError(f"{field} metadata values must be finite JSON")
    visit(value)
    if len(json.dumps(value, allow_nan=False)) > 100_000:
        raise ValueError(f"{field} exceeds 100000 character metadata limit")
    return json.loads(json.dumps(value, allow_nan=False))


def load_package_directory(directory: Path) -> dict:
    """Load a canonical camera package and separate evaluator labels from observations."""
    directory = Path(directory)
    if not directory.is_dir() or directory.is_symlink():
        raise ValueError("Package directory does not exist or is a symlink")
    directory = directory.resolve()
    paths = list(directory.rglob("*"))
    files = [p for p in paths if p.is_file()]
    if any(p.is_symlink() for p in paths) or len(files) > MAX_FILES:
        raise ValueError("Package contains symlinks or too many files")
    if sum(p.stat().st_size for p in files) > MAX_EXPANDED:
        raise ValueError("Expanded package exceeds 1 GiB")
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file() or manifest_path.stat().st_size > 32 * 1024 * 1024:
        raise ValueError("Package needs a manifest.json smaller than 32 MiB")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"),
                              object_pairs_hook=_unique_json_object, parse_constant=_invalid_json_constant)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("manifest.json must contain valid UTF-8 JSON") from exc
    manifest = _keys(manifest, {"version", "name", "source", "calibration", "sessions", "ground_truth", "evaluation", "benchmark"},
                     {"version", "name", "source", "calibration", "sessions"}, "manifest")
    if type(manifest["version"]) is not int or manifest["version"] != 1:
        raise ValueError("Only canonical manifest version 1 is supported")
    if not isinstance(manifest["source"], str) or manifest["source"] not in {"real", "synthetic"}:
        raise ValueError("source must be real or synthetic")
    name = _text(manifest["name"], "name")
    calibration = _calibration(manifest["calibration"])
    evaluation = _evaluation_context(manifest["evaluation"]) if "evaluation" in manifest else None
    camera_target = evaluation is not None and evaluation.get("pose_target") == "camera"
    benchmark = _bounded_metadata(manifest["benchmark"], "benchmark") if "benchmark" in manifest else None
    if camera_target and not np.allclose(calibration["T_CV"], np.eye(4), atol=1e-8):
        raise ValueError("Camera-target packages require identity T_CV")
    raw_sessions = manifest["sessions"]
    if not isinstance(raw_sessions, list) or not raw_sessions or len(raw_sessions) > 128:
        raise ValueError("sessions must contain 1 to 128 sessions")
    image_cache: dict[str, tuple[str, str]] = {}
    content_splits: dict[str, str] = {}
    group_splits: dict[str, str] = {}
    session_ids: set[str] = set()
    sessions = []
    digest = hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())
    for raw in raw_sessions:
        raw = _keys(raw, {"id", "name", "split", "group_id", "frames", "track", "scenario"},
                    {"id", "name", "split", "group_id", "frames"}, "session")
        sid = _identifier(raw["id"], "session.id")
        if sid in session_ids:
            raise ValueError(f"Duplicate session ID: {sid}")
        session_ids.add(sid)
        split = raw["split"]
        if not isinstance(split, str) or split not in SPLITS:
            raise ValueError("Invalid session split")
        group = _identifier(raw["group_id"], "session.group_id")
        if group in group_splits and group_splits[group] != split:
            raise ValueError(f"Session group {group} overlaps data splits")
        group_splits[group] = split
        frames_raw = raw["frames"]
        if not isinstance(frames_raw, list) or not frames_raw or len(frames_raw) > MAX_FILES:
            raise ValueError(f"Session {sid} needs a nonempty bounded frame list")
        frames, poses, frame_ids = [], {}, set()
        previous_time = -math.inf
        for frame in frames_raw:
            frame = _keys(frame, {"frame_id", "timestamp_s", "image", "T_WC"},
                          {"frame_id", "timestamp_s", "image"}, "frame")
            fid = _identifier(frame["frame_id"], "frame.frame_id")
            if fid in frame_ids:
                raise ValueError(f"Duplicate frame ID in {sid}: {fid}")
            frame_ids.add(fid)
            timestamp = _number(frame["timestamp_s"], "frame.timestamp_s")
            if timestamp < 0 or timestamp <= previous_time:
                raise ValueError(f"Session {sid} timestamps must be nonnegative and strictly increasing")
            previous_time = timestamp
            relative = _relative_path(frame["image"])
            path = _package_file(directory, relative)
            if relative not in image_cache:
                image_cache[relative] = _image_digest(path, calibration)
            encoded_hash, pixel_hash = image_cache[relative]
            if pixel_hash in content_splits and content_splits[pixel_hash] != split:
                raise ValueError(f"Image content occurs across splits: {relative}")
            content_splits[pixel_hash] = split
            digest.update(relative.encode())
            digest.update(encoded_hash.encode())
            frames.append({"frame_id": fid, "timestamp_s": timestamp, "image_path": str(path),
                           "content_hash": pixel_hash})
            if split == "reference":
                if "T_WC" not in frame:
                    raise ValueError("Reference frames require T_WC")
                poses[fid] = _transform(frame["T_WC"], f"{sid}.{fid}.T_WC")
            elif "T_WC" in frame:
                raise ValueError("Query frames must not contain pose labels")
        session = {"id": sid, "name": _text(raw["name"], "session.name"), "source": manifest["source"],
                   "split": split, "group_id": group, "original_group_id": group,
                   "calibration": calibration, "frames": frames}
        if evaluation is not None:
            session["evaluation"] = dict(evaluation)
        else:
            session["evaluation"] = {}
        if benchmark is not None:
            session["benchmark"] = _bounded_metadata(benchmark, "benchmark")
        if "scenario" in raw:
            session["scenario"] = _bounded_metadata(raw["scenario"], "scenario")
        if poses:
            session["reference_poses"] = poses
        if "track" in raw:
            session["track"] = _track(raw["track"])
        sessions.append(session)
    dataset_id = f"dataset-{digest.hexdigest()[:20]}"
    labels: dict[str, dict] = {}
    ground_truth = manifest.get("ground_truth", {})
    if not isinstance(ground_truth, dict) or set(ground_truth) - session_ids:
        raise ValueError("ground_truth must be keyed by existing session IDs")
    for session in sessions:
        sid = session["id"]
        new_sid = f"{dataset_id}--{sid}"
        if sid in ground_truth:
            if session["split"] == "reference":
                raise ValueError("Reference ground_truth is forbidden; use reference T_WC only")
            rows = ground_truth[sid]
            if not isinstance(rows, list):
                raise ValueError("ground_truth session labels must be a list")
            frame_lookup = {f["frame_id"]: f for f in session["frames"]}
            session_labels = {}
            for row in rows:
                allowed = {"frame_id", "timestamp_s", "position_m", "yaw_rad", "T_WC"} if camera_target else {"frame_id", "timestamp_s", "position_m", "yaw_rad"}
                required = allowed if camera_target else {"frame_id", "timestamp_s", "position_m", "yaw_rad"}
                row = _keys(row, allowed, required, "ground_truth row")
                fid = _identifier(row["frame_id"], "ground_truth.frame_id")
                if fid not in frame_lookup or fid in session_labels:
                    raise ValueError("Ground truth frame is unknown or duplicated")
                timestamp = _number(row["timestamp_s"], "ground_truth.timestamp_s")
                if abs(timestamp - frame_lookup[fid]["timestamp_s"]) > 1e-6:
                    raise ValueError("Ground truth timestamp does not match its camera frame")
                position = row["position_m"]
                if not isinstance(position, list) or len(position) != 3:
                    raise ValueError("Ground truth position_m needs three coordinates")
                position = [_number(x, "ground_truth.position_m") for x in position]
                label = {"timestamp_s": timestamp, "position_m": position}
                if camera_target:
                    if row["yaw_rad"] is not None:
                        raise ValueError("Camera-target ground truth yaw_rad must be null")
                    transform = _transform(row["T_WC"], "ground_truth.T_WC")
                    if not np.allclose(np.asarray(transform)[:3, 3], position, atol=1e-6, rtol=0):
                        raise ValueError("Ground truth position disagrees with T_WC translation")
                    label.update({"yaw_rad": None, "T_WC": transform})
                else:
                    label["yaw_rad"] = _number(row["yaw_rad"], "ground_truth.yaw_rad")
                session_labels[fid] = label
            labels[new_sid] = session_labels
        session["id"], session["dataset_id"] = new_sid, dataset_id
        session["group_id"] = f"{dataset_id}--{session['group_id']}"
    warnings = []
    for session in sessions:
        count = len(labels.get(session["id"], {}))
        if session["split"] != "reference" and count != len(session["frames"]):
            warnings.append(f"{session['name']}: {count}/{len(session['frames'])} frames have evaluation labels; accuracy may be unavailable or partial.")
    if manifest["source"] == "synthetic":
        warnings.append("Synthetic data validates this implementation; it does not establish real racing accuracy.")
    if camera_target:
        warnings.append("Camera pose is evaluated in native MoCap coordinates. Shared recording world origins are inferred, not certified; no query alignment is applied.")
    result = {"dataset_id": dataset_id, "name": name, "source": manifest["source"],
              "sessions": sessions, "labels": labels, "warnings": warnings}
    if evaluation is not None:
        result["evaluation"] = dict(evaluation)
    else:
        result["evaluation"] = {}
    if benchmark is not None:
        result["benchmark"] = benchmark
    return result


def import_package(zip_path: Path, destination: Path) -> dict:
    """Validate an upload before atomically moving it into a new destination directory."""
    zip_path, destination = Path(zip_path), Path(destination)
    if not zip_path.is_file() or zip_path.stat().st_size > MAX_UPLOAD:
        raise ValueError("ZIP upload exceeds 256 MiB or does not exist")
    if destination.exists():
        raise ValueError("Import destination must not already exist")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="camera-import-", dir=destination.parent) as temporary:
        root = Path(temporary)
        try:
            with zipfile.ZipFile(zip_path) as archive:
                members = archive.infolist()
                if len(members) > MAX_FILES or sum(m.file_size for m in members) > MAX_EXPANDED:
                    raise ValueError("ZIP expansion exceeds file count or 1 GiB limit")
                normalized = set()
                for member in members:
                    name = member.filename.rstrip("/")
                    relative = _relative_path(name)
                    folded = relative.casefold()
                    if folded in normalized:
                        raise ValueError("ZIP contains duplicate or case-ambiguous paths")
                    normalized.add(folded)
                    mode = member.external_attr >> 16
                    if stat.S_ISLNK(mode) or (stat.S_IFMT(mode) and not stat.S_ISREG(mode) and not stat.S_ISDIR(mode)):
                        raise ValueError("ZIP symlinks and special files are forbidden")
                    if member.flag_bits & 1:
                        raise ValueError("Encrypted ZIP entries are unsupported")
                    if member.file_size > 16 * 1024 * 1024 and member.file_size / max(1, member.compress_size) > 500:
                        raise ValueError("ZIP entry has an unsafe expansion ratio")
                    target = root.joinpath(*PurePosixPath(relative).parts)
                    if member.is_dir():
                        target.mkdir(parents=True, exist_ok=True)
                    else:
                        target.parent.mkdir(parents=True, exist_ok=True)
                        with archive.open(member) as source, target.open("wb") as output:
                            shutil.copyfileobj(source, output, 1024 * 1024)
                load_package_directory(root)
        except (zipfile.BadZipFile, RuntimeError, OSError) as exc:
            raise ValueError(f"Cannot read canonical ZIP: {exc}") from exc
        root.rename(destination)
    return load_package_directory(destination)


def create_template_zip(destination: Path) -> Path:
    """Downloadable valid example with no protected audit images or labels."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    width, height = 160, 120
    image = np.zeros((height, width, 3), np.uint8)
    for index in range(12):
        cv2.circle(image, (10 + index * 12, 25 + (index % 3) * 25), 5,
                   (50 + index * 12, 180, 230), -1)
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise ValueError("Cannot encode template image")
    manifest = {"version": 1, "name": "Camera package import example", "source": "synthetic",
                "calibration": {"image_width": width, "image_height": height,
                                "K": [[120, 0, 80], [0, 120, 60], [0, 0, 1]],
                                "distortion": [], "T_CV": np.eye(4).tolist()},
                "sessions": [{"id": "example", "name": "Replace with your recording", "split": "development",
                              "group_id": "example-group", "frames": [{"frame_id": "000001", "timestamp_s": 0.0,
                                                                          "image": "images/example.png"}]}]}
    readme = ("CANONICAL CAMERA PACKAGE v1\n\nThis tiny sample demonstrates import only, not localization accuracy.\n"
              "Replace the placeholder calibration and image with real calibrated camera data.\n"
              "Reference sessions require frame T_WC rigid transforms; query sessions forbid them.\n"
              "Optional ground_truth is a separate manifest object keyed by query session ID.\n"
              "Split complete recording groups before generating variants. Audit labels are never served by the app.\n"
              "Every frame needs a unique ID, increasing capture time in seconds, and a relative image path.\n"
              "World axes: x/y horizontal, z up. Vehicle: x forward, y left, z up.\n"
              "T_CV maps vehicle coordinates into OpenCV camera coordinates.\n")
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("manifest.json", json.dumps(manifest, indent=2))
        archive.writestr("README.txt", readme)
        archive.writestr("images/example.png", encoded.tobytes())
    return destination


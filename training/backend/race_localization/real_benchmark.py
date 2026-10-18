"""Bounded author-source TUM VI acquisition and explicit camera-pose conversion.

Only selected image payloads and exact TAR metadata ranges are transferred.
Query ground truth is used by conversion/evaluation, never by localization.
"""
from __future__ import annotations

import ast
import bisect
import hashlib
import json
import math
import tarfile
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

EVALUATION_METADATA = {"pose_target": "camera", "orientation_metric": "geodesic",
                       "world_frame": "tumvi_native_mocap", "alignment_status": "inferred_shared"}
SOURCES = {"room1": {"url": "https://cdn2.vision.in.tum.de/tumvi/exported/euroc/512_16/dataset-room1_512_16.tar", "size": 1707110400},
           "room2": {"url": "https://cdn2.vision.in.tum.de/tumvi/exported/euroc/512_16/dataset-room2_512_16.tar", "size": 1759395840}}
AUTHOR_PAGE = "https://cvg.cit.tum.de/data/datasets/visual-inertial-dataset"
AUTHOR_PAPER = "https://cvg.cit.tum.de/_media/spezial/bib/schubert2018vidataset.pdf"
INDEX_LIMIT = 8_000_000
IMAGE_LIMIT = 200_000_000
FRAME_LIMIT = 1_000_000
MAX_GT_BRACKET_NS = 25_000_000
BUNDLED_METADATA = Path(__file__).resolve().with_name("tumvi_metadata")
BUNDLED_MANIFEST_SHA256 = "cefcdcbb0c92e95d3afe4bd950304cfda777f148819128c1651359d13cb715eb"
METADATA_PATHS = {f"{recording}/{relative}" for recording in SOURCES
                  for relative in ("dso/camchain.yaml", "dso/gt_imu.csv", "dso/cam0/times.txt", "mav0/cam0/data.csv")}


def _save(path: Path, value: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def prepare_metadata(cache_dir: Path) -> dict:
    """Verify every pinned public author asset before populating an acquisition cache."""
    bundle, cache_dir = BUNDLED_METADATA, Path(cache_dir)
    manifest_path = bundle / "MANIFEST.json"
    raw_manifest = manifest_path.read_bytes()
    if hashlib.sha256(raw_manifest).hexdigest() != BUNDLED_MANIFEST_SHA256:
        raise ValueError("Bundled author metadata manifest identity mismatch")
    manifest = json.loads(raw_manifest)
    files = manifest["files"]
    if len(files) != 8 or {row["path"] for row in files} != METADATA_PATHS:
        raise ValueError("Bundled author metadata must include all eight fixed assets")
    verified = []
    for row in files:
        source = bundle / row["path"]
        if source.is_symlink() or not source.resolve().is_relative_to(bundle.resolve()):
            raise ValueError("Bundled metadata paths must remain inside the source bundle")
        raw = source.read_bytes()
        if len(raw) != row["bytes"] or hashlib.sha256(raw).hexdigest() != row["sha256"]:
            raise ValueError(f"Bundled author metadata identity mismatch: {row['path']}")
        destination = cache_dir / row["path"]
        if destination.exists() and destination.read_bytes() != raw:
            raise ValueError("Source metadata cache differs from pinned author bytes")
        verified.append((destination, raw))
    # Validate the complete bundle and existing cache before writing any asset.
    for destination, raw in verified:
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            destination.write_bytes(raw)
    return {"manifest_sha256": BUNDLED_MANIFEST_SHA256, "asset_count": 8,
            "asset_bytes": sum(row["bytes"] for row in files), "files": files}


class RangeCache:
    """Persistent byte-accounted exact-range reads with no full-archive fallback."""
    def __init__(self, directory: Path):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.ledger_path = self.directory / "download-ledger.json"
        self.ledger = json.loads(self.ledger_path.read_text()) if self.ledger_path.exists() else []
        self.totals = {kind: sum(row["bytes"] for row in self.ledger if row["category"] == kind)
                       for kind in ("metadata", "image")}

    def read(self, recording: str, offset: int, length: int, category: str = "metadata") -> bytes:
        source = SOURCES[recording]
        if offset < 0 or length <= 0 or offset + length > source["size"]:
            raise ValueError("Range is outside the fixed archive")
        if category not in {"metadata", "image"} or (category == "image" and length > FRAME_LIMIT):
            raise ValueError("Unsupported category or oversized image payload")
        file = self.directory / recording / "ranges" / f"{offset}-{length}.bin"
        if file.exists():
            data = file.read_bytes()
            known = next((row for row in reversed(self.ledger) if row.get("cache_file") == str(file.relative_to(self.directory))), None)
            if len(data) != length or known is None or hashlib.sha256(data).hexdigest() != known["sha256"]:
                raise ValueError("Range cache identity mismatch")
            return data
        for attempt in range(3):
            with self.lock:
                limit = INDEX_LIMIT if category == "metadata" else IMAGE_LIMIT
                if self.totals[category] + length + 1 > limit:
                    raise ValueError(f"{category} download budget exhausted")
                self.totals[category] += length + 1  # Reserve all bytes an oversized response could expose.
            data = b""
            try:
                request = urllib.request.Request(source["url"], headers={"Range": f"bytes={offset}-{offset + length - 1}",
                    "Accept-Encoding": "identity", "User-Agent": "SchubKI-camera-benchmark/1.0"})
                with urllib.request.urlopen(request, timeout=30) as response:
                    expected = f"bytes {offset}-{offset + length - 1}/{source['size']}"
                    if response.status != 206 or response.headers.get("Content-Range") != expected:
                        raise ValueError("Server did not honor the exact range; full-archive fallback forbidden")
                    if response.headers.get("Content-Encoding", "identity") != "identity":
                        raise ValueError("Compressed range response is unsupported")
                    data = response.read(length + 1)
                    if len(data) != length:
                        raise ValueError("Range response has an unexpected body length")
                    details = {"url": source["url"], "final_url": response.url, "status": response.status,
                               "content_range": expected, "etag": response.headers.get("ETag")}
                    with self.lock:
                        prior_etag = next((row.get("etag") for row in self.ledger
                                           if row.get("recording") == recording and row.get("etag")), None)
                    if prior_etag and details["etag"] and prior_etag != details["etag"]:
                        raise ValueError("Archive ETag changed during bounded acquisition")
            except (urllib.error.URLError, TimeoutError, OSError):
                with self.lock:
                    # A failed transport may have transferred partial bytes unavailable to us.
                    self.ledger.append({"category": category, "bytes": length + 1,
                                        "recording": recording, "offset": offset, "status": "transport uncertainty allowance"})
                    _save(self.ledger_path, self.ledger)
                if attempt == 2:
                    raise
                time.sleep(.2 * (attempt + 1))
                continue
            except Exception:
                with self.lock:
                    self.totals[category] -= length + 1 - len(data)
                    self.ledger.append({"category": category, "bytes": len(data), "recording": recording,
                                        "offset": offset, "status": "rejected response"})
                    _save(self.ledger_path, self.ledger)
                raise
            with self.lock:
                self.totals[category] -= 1
                row = {**details, "category": category, "bytes": length, "recording": recording,
                       "offset": offset, "sha256": hashlib.sha256(data).hexdigest(),
                       "cache_file": str(file.relative_to(self.directory))}
                self.ledger.append(row)
                _save(self.ledger_path, self.ledger)
                file.parent.mkdir(parents=True, exist_ok=True)
                file.write_bytes(data)
            return data
        raise RuntimeError("Unreachable range retry state")


def parse_header(header: bytes, offset: int) -> dict | None:
    if len(header) != 512:
        raise ValueError("TAR headers must be exactly 512 bytes")
    if header == bytes(512):
        return None
    info = tarfile.TarInfo.frombuf(header, "utf-8", "strict")
    if info.size < 0:
        raise ValueError("Negative TAR member size")
    return {"name": info.name, "size": info.size, "header_offset": offset,
            "data_offset": offset + 512, "type": info.type.decode("ascii"), "regular": info.isreg()}


def index_archive(recording: str, cache: RangeCache, progress_callback=None) -> dict:
    path = cache.directory / recording / "index.json"
    state = json.loads(path.read_text()) if path.exists() else {"source": SOURCES[recording], "members": [], "next_offset": 0, "complete": False}
    if state["source"] != SOURCES[recording]:
        raise ValueError("Archive index source identity changed")
    if state["complete"]:
        return state
    offset = state["next_offset"]
    while offset + 512 <= SOURCES[recording]["size"]:
        member = parse_header(cache.read(recording, offset, 512), offset)
        if member is None:
            state["complete"] = True
            _save(path, state)
            return state
        # The fixed author archives have ordinary ustar names. Reject unsupported
        # metadata extensions rather than treating their payload as image bytes.
        if member["type"] in {"L", "K", "x", "g"}:
            raise ValueError("Extended TAR headers need an explicit parser; indexing stopped safely")
        next_offset = offset + 512 + ((member["size"] + 511) // 512) * 512
        if next_offset > SOURCES[recording]["size"]:
            raise ValueError("TAR member extends beyond its declared archive")
        state["members"].append(member)
        offset = state["next_offset"] = next_offset
        if len(state["members"]) % 50 == 0:
            _save(path, state)
            if progress_callback:
                progress_callback({"stage": "index", "recording": recording, "members": len(state["members"]),
                                   "archive_offset": offset, "metadata_bytes": cache.totals["metadata"]})
    raise ValueError("TAR did not terminate within the declared archive")


def read_camera_calibration(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    text = Path(path).read_text(encoding="utf-8").split("cam1:")[0]
    lines = text.splitlines()
    row = lines.index("  T_cam_imu:")
    extrinsic = np.asarray([ast.literal_eval(lines[row + i].strip()[2:]) for i in range(1, 5)], dtype=float)
    intrinsic = ast.literal_eval(next(line.split(":", 1)[1].strip() for line in lines if line.strip().startswith("intrinsics:")))
    distortion = np.asarray(ast.literal_eval(text.split("distortion_coeffs:", 1)[1].split("distortion_model:", 1)[0].strip()), dtype=float)
    if "camera_model: pinhole" not in text or "distortion_model: equidistant" not in text:
        raise ValueError("Expected author pinhole-equidistant calibration")
    if (extrinsic.shape != (4, 4) or distortion.shape != (4,) or not np.isfinite(extrinsic).all()
            or not np.isfinite(distortion).all() or len(intrinsic) != 4 or not np.isfinite(intrinsic).all()
            or not np.allclose(extrinsic[3], [0, 0, 0, 1], atol=1e-8, rtol=0)
            or not np.allclose(extrinsic[:3, :3].T @ extrinsic[:3, :3], np.eye(3), atol=1e-6, rtol=0)
            or not np.isclose(np.linalg.det(extrinsic[:3, :3]), 1, atol=1e-6, rtol=0)):
        raise ValueError("Source calibration must contain finite intrinsic/distortion values and a rigid extrinsic")
    fx, fy, cx, cy = intrinsic
    if fx <= 0 or fy <= 0:
        raise ValueError("Source camera focal lengths must be positive")
    return np.asarray([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]]), distortion, extrinsic


def read_capture_metadata(directory: Path) -> list[dict]:
    directory = Path(directory)
    exposures = {}
    for line in (directory / "dso/cam0/times.txt").read_text().splitlines():
        if line and not line.startswith("#"):
            fields = line.split()
            timestamp = int(fields[0])
            # Parse decimal seconds using integers, avoiding epoch float rounding.
            whole, fraction = fields[1].split(".")
            if int(whole) * 1_000_000_000 + int(fraction.ljust(9, "0")) != timestamp:
                raise ValueError("DSO and EuRoC timestamps disagree")
            exposure = float(fields[2])
            if timestamp in exposures or not math.isfinite(exposure) or exposure <= 0:
                raise ValueError("Exposure metadata must be unique, finite and positive")
            exposures[timestamp] = exposure
    records = []
    for line in (directory / "mav0/cam0/data.csv").read_text().splitlines():
        if line and not line.startswith("#"):
            ns, filename = line.split(",")
            ns = int(ns)
            if filename.strip() != f"{ns}.png" or ns not in exposures:
                raise ValueError("Capture filename/exposure provenance mismatch")
            records.append({"timestamp_ns": ns, "filename": filename.strip(), "exposure_ms": exposures[ns]})
    if not records or any(b["timestamp_ns"] <= a["timestamp_ns"] for a, b in zip(records, records[1:])):
        raise ValueError("Capture metadata must be strictly ordered")
    return records


def select_window(records: list[dict]) -> list[dict]:
    origin = records[0]["timestamp_ns"]
    selected = [dict(row, timestamp_s=(row["timestamp_ns"] - origin) / 1e9) for row in records
                if 5_000_000_000 <= row["timestamp_ns"] - origin < 15_000_000_000]
    if len(selected) != 200:
        raise ValueError("Fixed [5,15) second window must contain all 200 captures")
    return selected


def read_ground_truth(path: Path) -> list[tuple]:
    rows = []
    for line in Path(path).read_text().splitlines():
        if line and not line.startswith("#"):
            fields = line.split(",")
            row = (int(fields[0]), *[float(value) for value in fields[1:]])
            if len(row) != 8 or not np.isfinite(row[1:]).all() or np.linalg.norm(row[4:]) < 1e-10:
                raise ValueError("Invalid author ground truth record")
            rows.append(row)
    if not rows or any(b[0] <= a[0] for a, b in zip(rows, rows[1:])):
        raise ValueError("Ground truth timestamps must be strictly ordered")
    return rows


def interpolate_ground_truth(rows: list[tuple], timestamp_ns: int) -> tuple[np.ndarray | None, dict]:
    times = [row[0] for row in rows]
    index = bisect.bisect_left(times, timestamp_ns)
    if index < len(rows) and rows[index][0] == timestamp_ns:
        before = after = rows[index]
        alpha = 0.0
    elif index == 0 or index == len(rows):
        return None, {"label_unavailable": "outside source ground truth; no extrapolation"}
    else:
        before, after = rows[index - 1], rows[index]
        if after[0] - before[0] > MAX_GT_BRACKET_NS:
            return None, {"source_before_ns": before[0], "source_after_ns": after[0],
                          "bracket_ns": after[0] - before[0], "label_unavailable": "source bracket exceeds 25 ms"}
        alpha = (timestamp_ns - before[0]) / (after[0] - before[0])
    qa, qb = np.asarray(before[4:], float), np.asarray(after[4:], float)
    qa /= np.linalg.norm(qa)
    qb /= np.linalg.norm(qb)
    dot = float(qa @ qb)
    if dot < 0:
        qb, dot = -qb, -dot
    if dot > .9995:
        quaternion = (1 - alpha) * qa + alpha * qb
    else:
        theta = math.acos(np.clip(dot, -1, 1))
        quaternion = (math.sin((1 - alpha) * theta) * qa + math.sin(alpha * theta) * qb) / math.sin(theta)
    w, x, y, z = quaternion / np.linalg.norm(quaternion)
    transform = np.eye(4)
    transform[:3, :3] = [[1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
                         [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
                         [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)]]
    transform[:3, 3] = (1 - alpha) * np.asarray(before[1:4]) + alpha * np.asarray(after[1:4])
    return transform, {"source_before_ns": before[0], "source_after_ns": after[0], "bracket_ns": after[0] - before[0], "alpha": alpha}


def rectify_image(raw: bytes, intrinsic: np.ndarray, distortion: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None or image.dtype != np.uint16 or image.shape != (512, 512):
        raise ValueError("Author image must decode as 512x512 uint16 monochrome")
    mx, my = cv2.fisheye.initUndistortRectifyMap(intrinsic, distortion, np.eye(3), intrinsic, (512, 512), cv2.CV_32FC1)
    rectified = cv2.remap(image, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    valid = ((mx >= 0) & (mx < 511) & (my >= 0) & (my < 511)).astype(np.uint8) * 255
    converted = np.rint(rectified.astype(np.float64) / 257.0).clip(0, 255).astype(np.uint8)
    return cv2.cvtColor(converted, cv2.COLOR_GRAY2BGR), valid


def acquire_capture(recording: str, member: dict, cache: RangeCache) -> tuple[bytes, str]:
    cached = (cache.directory / recording / "ranges" / f"{member['data_offset']}-{member['size']}.bin").exists()
    return cache.read(recording, member["data_offset"], member["size"], "image"), (
        "verified local cache of exact author image range" if cached else "new bounded exact author image range")


def prepare_tumvi(output_dir: Path, cache_dir: Path, progress_callback=None) -> dict:
    """Prepare the complete fixed window; never shorten it based on localization results."""
    output_dir, cache_dir = Path(output_dir), Path(cache_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache = RangeCache(cache_dir)
    metadata_identity = prepare_metadata(cache_dir)
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {recording: executor.submit(index_archive, recording, cache, progress_callback) for recording in SOURCES}
        indexes = {recording: future.result() for recording, future in futures.items()}
    calibration_bytes = [(cache_dir / recording / "dso/camchain.yaml").read_bytes() for recording in SOURCES]
    if calibration_bytes[0] != calibration_bytes[1]:
        raise ValueError("Fixed recordings need identical calibrated camera models")
    intrinsic, distortion, extrinsic = read_camera_calibration(cache_dir / "room1/dso/camchain.yaml")
    manifest = {"version": 1, "name": "TUM VI native camera pose [5,15) seconds, independent room recordings",
                "source": "real", "evaluation": dict(EVALUATION_METADATA),
                "benchmark": {"kind": "tumvi_camera", "version": 1,
                    "description": "Fixed native camera pose benchmark; shared origins inferred",
                    "conditions": [{"id": "native-camera", "label": "TUM VI room2 native camera [5,15) seconds",
                                    "interventions": []}],
                    "preprocessing_scope": "offline native16bit conversion and rectification excluded from replay timing",
                    "declared_window_s": {"start": 5, "end": 15, "interval": "half-open",
                                          "origin": "first capture of each original recording"},
                    "reference_stride": 5, "query_stride": 1, "source_captures_per_recording": 200,
                    "provenance": {"author_page": AUTHOR_PAGE, "author_paper": AUTHOR_PAPER,
                                   "data_license": "CC BY 4.0", "source_archives": SOURCES,
                                   "reference_group": "tumvi-official-room1", "development_group": "tumvi-official-room2",
                                   "split_policy": "Independent whole recording groups; every selected development capture retained"}},
                "calibration": {"image_width": 512, "image_height": 512, "K": intrinsic.tolist(), "distortion": [0.0] * 5,
                                "T_CV": np.eye(4).tolist()}, "sessions": [], "ground_truth": {}}
    provenance = []
    for recording, split in (("room1", "reference"), ("room2", "development")):
        records = select_window(read_capture_metadata(cache_dir / recording))
        rows = read_ground_truth(cache_dir / recording / "dso/gt_imu.csv")
        members = {row["name"]: row for row in indexes[recording]["members"]}
        frames, labels = [], []
        for frame_index, record in enumerate(records):
            member_name = f"dataset-{recording}_512_16/mav0/cam0/data/{record['filename']}"
            member = members.get(member_name)
            if member is None or not member["regular"]:
                raise ValueError(f"Missing selected original camera capture: {member_name}")
            original_header = parse_header(cache.read(recording, member["header_offset"], 512), member["header_offset"])
            if original_header != member:
                raise ValueError("Selected member index differs from its authenticated cached TAR header")
            raw, transfer = acquire_capture(recording, member, cache)
            native = cache_dir / recording / "native16" / record["filename"]
            native.parent.mkdir(exist_ok=True)
            native.write_bytes(raw)
            image, mask = rectify_image(raw, intrinsic, distortion)
            relative = f"{recording}/{record['filename']}"
            destination = output_dir / relative
            destination.parent.mkdir(exist_ok=True)
            if not cv2.imwrite(str(destination), image):
                raise ValueError("Failed to write rectified capture")
            mask_relative = f"masks/{recording}/{record['filename']}"
            mask_path = output_dir / mask_relative
            mask_path.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(mask_path), mask):
                raise ValueError("Failed to write valid-pixel mask")
            imu_pose, join = interpolate_ground_truth(rows, record["timestamp_ns"])
            camera_pose = imu_pose @ np.linalg.inv(extrinsic) if imu_pose is not None else None
            frame = {"frame_id": str(record["timestamp_ns"]), "timestamp_s": record["timestamp_s"], "image": relative}
            used = split != "reference" or frame_index % 5 == 0
            if split == "reference" and used:
                if camera_pose is None:
                    raise ValueError("Selected reference map view lacks independent pose")
                frame["T_WC"] = camera_pose.tolist()
            if used:
                frames.append(frame)
            if split == "development" and camera_pose is not None:
                labels.append({"frame_id": frame["frame_id"], "timestamp_s": frame["timestamp_s"],
                               "position_m": camera_pose[:3, 3].tolist(), "yaw_rad": None, "T_WC": camera_pose.tolist()})
            provenance.append({"recording": recording, **record, **join, "used_in_package": used,
                               "source_transfer": transfer,
                               "source_sha256": hashlib.sha256(raw).hexdigest(),
                               "rectified_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
                               "valid_pixel_mask": mask_relative, "mask_used_by_runtime": False,
                               "valid_pixel_fraction": float(np.mean(mask > 0))})
            if progress_callback and (frame_index + 1) % 25 == 0:
                progress_callback({"stage": "images", "recording": recording, "captures": frame_index + 1,
                                   "image_bytes": cache.totals["image"]})
        sid = f"tumvi-{recording}-next"
        manifest["sessions"].append({"id": sid, "name": f"TUM VI {recording} native camera [5,15) seconds",
                                     "split": split, "group_id": f"tumvi-official-{recording}", "frames": frames,
                                     "scenario": {"id": "native-camera", "label": f"TUM VI {recording} native camera [5,15) seconds",
                                                  "interventions": [], "trajectory_id": f"tumvi-official-{recording}"}})
        if split == "development":
            manifest["ground_truth"][sid] = labels
    _save(output_dir / "manifest.json", manifest)
    _save(output_dir / "conversion-provenance.json", provenance)
    _save(output_dir / "conversion-contract.json", {"evaluation": EVALUATION_METADATA, "source_K": intrinsic.tolist(),
          "bundled_metadata_manifest_sha256": metadata_identity["manifest_sha256"],
          "source_D": distortion.tolist(), "T_CI": extrinsic.tolist(), "rectification_R": np.eye(3).tolist(),
          "intensity_conversion": "round(uint16/257) -> uint8; monochrome replicated into three channels",
          "timestamp_convention": "source integer nanoseconds denote exposure midpoint; no additional offset applied",
          "pose_conversion": "T_WC=T_WI*inverse(T_CI)", "reference_stride": 5, "captures_per_recording": 200,
          "world_qualification": "Native MoCap W; shared recording origins inferred from author description, not certified. No query alignment.",
          "mask_scope": "saved diagnostic valid-pixel masks; runtime does not receive them",
          "metadata_bytes": cache.totals["metadata"], "image_bytes": cache.totals["image"],
          "source_metadata_sha256": {recording: {str(file.relative_to(cache_dir / recording)): hashlib.sha256(file.read_bytes()).hexdigest()
              for file in (cache_dir / recording).rglob("*") if file.is_file() and file.suffix in {".csv", ".yaml", ".txt"}} for recording in SOURCES}})
    (output_dir / "README.txt").write_text("TUM VI Benchmark by David Schubert, Thore Goll, Nikolaus Demmel, Vladyslav Usenko, Joerg Stueckler and Daniel Cremers (IROS 2018).\n"
        f"Author sources: {AUTHOR_PAGE}\n{AUTHOR_PAPER}\nData license: CC BY 4.0, https://creativecommons.org/licenses/by/4.0/\n"
        "Transforms: author equidistant image rectification, fixed 16-bit conversion and camera-pose conversion.\n"
        "This is a handheld camera benchmark. Shared MoCap world origins are inferred, not certified. No query-based alignment or racing-vehicle claim.\n", encoding="utf-8")
    return manifest


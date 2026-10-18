"""Camera-only, bounded-memory adapter for the official Constructor ROS 2 MCAP.

The estimator cache deliberately has no ego-pose topic decoder and no initial pose.
Initialization is a separate runner-owned artifact. T_BC maps optical-camera
coordinates into constructor0/base_link coordinates.
"""
from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from pathlib import Path
from typing import Iterator

import numpy as np

SCHEMA = "official-mcap-camera-v1"
CAMERAS = {"fl", "fr", "cl", "cr", "r", "rl", "rr"}
BODY_FRAME = "constructor0/base_link"


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _write_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(_canonical(value) + b"\n")
    temporary.replace(path)


def _stamp(stamp) -> int:
    sec, ns = int(stamp.sec), int(stamp.nanosec)
    if sec < 0 or not 0 <= ns < 1_000_000_000:
        raise ValueError("Invalid nonnegative ROS exposure timestamp")
    result = sec * 1_000_000_000 + ns
    if result > np.iinfo(np.int64).max:
        raise ValueError("Exposure timestamp exceeds signed int64")
    return result


def _iter_allowed_messages(path: Path, topics: tuple[str, ...]):
    """Filter encoded channels before constructing or invoking any decoder."""
    from mcap.reader import make_reader
    from mcap_ros2.decoder import DecoderFactory

    factory = DecoderFactory()
    decoders = {}
    with path.open("rb") as stream:
        reader = make_reader(stream)
        for schema, channel, message in reader.iter_messages(topics=topics, log_time_order=False):
            if channel.topic not in topics:
                raise ValueError("MCAP reader returned a non-whitelisted channel")
            if schema is None:
                raise ValueError(f"Missing ROS schema for {channel.topic}")
            key = (channel.message_encoding, schema.id)
            if key not in decoders:
                decoders[key] = factory.decoder_for(channel.message_encoding, schema)
            decoder = decoders[key]
            if decoder is None:
                raise ValueError(f"Unsupported ROS encoding for {channel.topic}")
            yield channel.topic, int(message.log_time), decoder(message.data)


def _finite(values, name: str) -> list[float]:
    result = [float(value) for value in values]
    if not all(math.isfinite(value) for value in result):
        raise ValueError(f"Nonfinite {name}")
    return result


def _transform(value) -> list[list[float]]:
    translation = _finite([value.translation.x, value.translation.y, value.translation.z], "translation")
    q = np.asarray(_finite([value.rotation.x, value.rotation.y, value.rotation.z, value.rotation.w], "quaternion"))
    norm = float(np.linalg.norm(q))
    if abs(norm - 1.0) > 1e-3:
        raise ValueError("Camera quaternion must be unit length")
    x, y, z, w = q / norm
    rotation = np.array([
        [1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
        [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
        [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)],
    ])
    matrix = np.eye(4)
    matrix[:3, :3], matrix[:3, 3] = rotation, translation
    return matrix.tolist()


def _intrinsics(message, expected_frame: str) -> dict:
    if message.header.frame_id.lstrip("/") != expected_frame:
        raise ValueError("CameraInfo optical frame does not match selected camera")
    k = _finite(message.k, "camera K")
    if len(k) != 9 or k[0] <= 0 or k[4] <= 0 or not np.allclose(k[6:], [0, 0, 1]):
        raise ValueError("Invalid camera intrinsics")
    width, height = int(message.width), int(message.height)
    if width < 1 or height < 1:
        raise ValueError("Invalid camera dimensions")
    distortion_model = str(message.distortion_model)
    if distortion_model not in {"plumb_bob", "rational_polynomial", ""}:
        raise ValueError(f"Unsupported camera distortion model: {distortion_model}")
    return {"K": np.asarray(k).reshape(3, 3).tolist(), "D": _finite(message.d, "camera D"),
            "width": width, "height": height, "distortion_model": distortion_model,
            "camera_frame": expected_frame}


def _record_version(versions: dict, payload: dict, stamp_ns: int) -> None:
    identity = hashlib.sha256(_canonical(payload)).hexdigest()
    if identity not in versions:
        versions[identity] = {"sha256": identity, "value": payload, "first_stamp_ns": stamp_ns,
                              "last_stamp_ns": stamp_ns, "message_count": 0}
    version = versions[identity]
    version["first_stamp_ns"] = min(stamp_ns, version["first_stamp_ns"])
    version["last_stamp_ns"] = max(stamp_ns, version["last_stamp_ns"])
    version["message_count"] += 1


def _relative(root: Path, raw: str) -> Path:
    path = (root / raw).resolve()
    if not path.is_relative_to(root) or path == root:
        raise ValueError("Cached artifact path leaves the recording cache")
    return path


def _artifact(manifest: dict, path_key: str, hash_key: str) -> Path:
    root = Path(manifest["cache_dir"]).resolve()
    path = _relative(root, manifest[path_key])
    if _hash_file(path) != manifest[hash_key]:
        raise ValueError(f"Cached artifact hash mismatch: {path_key}")
    return path


def prepare_recording(mcap_path, cache_dir, camera: str = "r") -> dict:
    """Prepare one camera without decoding ego state or retaining image arrays.

    Each selected JPEG is saved as received. A disk-backed index sorts exposure
    stamps and resolves duplicate stamps by lexicographically smallest image hash.
    Distinct calibration values are retained and make read_calibration fail closed.
    """
    if camera not in CAMERAS:
        raise ValueError(f"Unknown camera {camera!r}")
    source, root = Path(mcap_path).resolve(), Path(cache_dir).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    source_hash = _hash_file(source)
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema") != SCHEMA or manifest.get("source", {}).get("sha256") != source_hash or manifest.get("camera") != camera:
            raise ValueError("Cache belongs to a different recording, camera or adapter schema")
        _artifact(manifest, "frames_index_path", "frames_index_sha256")
        _artifact(manifest, "calibration_path", "calibration_sha256")
        _artifact(manifest, "track_bounds_path", "track_bounds_sha256")
        return manifest

    frame_dir = root / "frames"
    frame_dir.mkdir(exist_ok=True)
    image_topic = f"/constructor0/sensor/camera_{camera}/compressed_image"
    info_topic = f"/constructor0/sensor/camera_{camera}/camera_info"
    topics = (image_topic, info_topic, "/tf_static", "/constructor0/track_bounds")
    camera_frame = f"constructor0/camera_{camera}"
    intrinsics, extrinsics, borders = {}, {}, {}
    received = 0
    index_database = root / "frame_index.sqlite3"
    with sqlite3.connect(index_database) as database:
        database.execute("DROP TABLE IF EXISTS frames")
        database.execute("CREATE TABLE frames(stamp INTEGER NOT NULL, digest TEXT NOT NULL, path TEXT NOT NULL, size INTEGER NOT NULL, PRIMARY KEY(stamp,digest))")
        for topic, log_ns, message in _iter_allowed_messages(source, topics):
            if topic == image_topic:
                stamp = _stamp(message.header.stamp)
                if message.header.frame_id.lstrip("/") != camera_frame:
                    raise ValueError("Image optical frame does not match selected camera")
                jpeg = bytes(message.data)
                if not jpeg.startswith(b"\xff\xd8") or not jpeg.endswith(b"\xff\xd9"):
                    raise ValueError("Selected compressed image is not a complete JPEG")
                digest = hashlib.sha256(jpeg).hexdigest()
                relative = f"frames/{stamp}-{digest[:16]}.jpg"
                destination = root / relative
                if not destination.exists():
                    destination.write_bytes(jpeg)
                elif _hash_file(destination) != digest:
                    raise ValueError("Existing cached JPEG does not match source")
                database.execute("INSERT OR IGNORE INTO frames VALUES(?,?,?,?)", (stamp, digest, relative, len(jpeg)))
                received += 1
                if received % 256 == 0:
                    database.commit()
            elif topic == info_topic:
                _record_version(intrinsics, _intrinsics(message, camera_frame), _stamp(message.header.stamp))
            elif topic == "/tf_static":
                for item in message.transforms:
                    if item.child_frame_id.lstrip("/") != camera_frame:
                        continue
                    if item.header.frame_id.lstrip("/") != BODY_FRAME:
                        raise ValueError("Selected camera extrinsic must be a direct base_link transform")
                    payload = {"T_BC": _transform(item.transform), "body_frame": BODY_FRAME, "camera_frame": camera_frame}
                    _record_version(extrinsics, payload, _stamp(item.header.stamp))
            elif topic == "/constructor0/track_bounds":
                markers = []
                for marker in message.markers:
                    if int(marker.action) != 0 or int(marker.type) != 4:
                        continue
                    if marker.header.frame_id.lstrip("/") != "map":
                        raise ValueError("Track borders must be expressed in map")
                    pose = marker.pose
                    transform_like = type("MarkerTransform", (), {"translation": pose.position, "rotation": pose.orientation})()
                    matrix = np.asarray(_transform(transform_like))
                    points = []
                    for point in marker.points:
                        value = matrix @ np.asarray(_finite([point.x, point.y, point.z], "track point") + [1.0])
                        points.append(value[:3].tolist())
                    markers.append({"namespace": str(marker.ns), "id": int(marker.id), "frame": "map", "points": points})
                if markers:
                    markers.sort(key=lambda value: (value["namespace"], value["id"]))
                    _record_version(borders, {"markers": markers}, log_ns)
        database.commit()
        if not received:
            raise ValueError("Selected camera has no JPEG observations")
        if not intrinsics or not extrinsics:
            raise ValueError("Selected camera is missing intrinsics or direct extrinsics")
        if not borders:
            raise ValueError("Recording is missing metric map track borders")
        index_path = root / "frames.jsonl"
        temporary = index_path.with_suffix(".jsonl.tmp")
        count, first, last, previous = 0, None, None, None
        with temporary.open("wb") as output:
            for stamp, digest, relative, size in database.execute("SELECT stamp,digest,path,size FROM frames ORDER BY stamp,digest"):
                if stamp == previous:
                    continue
                previous, first, last = stamp, stamp if first is None else first, stamp
                output.write(_canonical({"timestamp_ns": stamp, "path": relative, "sha256": digest, "bytes": size}) + b"\n")
                count += 1
        temporary.replace(index_path)

    ordered = lambda values: sorted(values.values(), key=lambda value: (value["first_stamp_ns"], value["sha256"]))
    calibration = {"schema": SCHEMA, "camera": camera, "static_stable": len(intrinsics) == len(extrinsics) == 1,
                   "intrinsics_versions": ordered(intrinsics), "extrinsics_versions": ordered(extrinsics)}
    _write_json(root / "calibration.json", calibration)
    _write_json(root / "track_bounds.json", {"schema": SCHEMA, "frame": "map", "versions": ordered(borders)})
    manifest = {"schema": SCHEMA, "cache_dir": str(root), "cache_root": str(root), "camera": camera, "camera_topic": image_topic,
                "source": {"path": str(source), "bytes": source.stat().st_size, "sha256": source_hash},
                "decoder_topic_whitelist": list(topics), "pose_frame": BODY_FRAME, "world_frame": "map",
                "frame_count": count, "received_frame_count": received, "duplicate_exposure_count": received - count,
                "first_timestamp_ns": first, "last_timestamp_ns": last,
                "duplicate_policy": "lexicographically-smallest-SHA256-per-exposure-stamp",
                "calibration_static_stable": calibration["static_stable"],
                "frames_index_path": "frames.jsonl", "frames_index_sha256": _hash_file(root / "frames.jsonl"),
                "calibration_path": "calibration.json", "calibration_sha256": _hash_file(root / "calibration.json"),
                "track_bounds_path": "track_bounds.json", "track_bounds_sha256": _hash_file(root / "track_bounds.json"),
                "licence": "National AI Hackathon 2026 only; do not redistribute without Constructor Labs permission"}
    _write_json(manifest_path, manifest)
    return manifest


def read_calibration(manifest: dict) -> dict:
    payload = json.loads(_artifact(manifest, "calibration_path", "calibration_sha256").read_text(encoding="utf-8"))
    if not payload.get("static_stable") or len(payload["intrinsics_versions"]) != 1 or len(payload["extrinsics_versions"]) != 1:
        raise ValueError("Changing camera calibration recorded; static baseline requires a stable calibration")
    intrinsic = payload["intrinsics_versions"][0]
    extrinsic = payload["extrinsics_versions"][0]
    return {**intrinsic["value"], **extrinsic["value"], "static_stable": True,
            "image_width": intrinsic["value"]["width"], "image_height": intrinsic["value"]["height"],
            "distortion": intrinsic["value"]["D"],
            "intrinsics_sha256": intrinsic["sha256"], "extrinsics_sha256": extrinsic["sha256"],
            "calibration_sha256": manifest["calibration_sha256"]}


def iter_frames(manifest: dict) -> Iterator[tuple[int, Path]]:
    """Stream integrity-checked observations, optionally restricted to [start,end]."""
    root = Path(manifest["cache_dir"]).resolve()
    index = _artifact(manifest, "frames_index_path", "frames_index_sha256")
    start, end = manifest.get("start_ns"), manifest.get("end_ns")
    if any(value is not None and (isinstance(value, bool) or not isinstance(value, int)) for value in (start, end)):
        raise ValueError("Frame window must use integer nanoseconds")
    previous = None
    with index.open("r", encoding="utf-8") as stream:
        for line in stream:
            frame = json.loads(line)
            stamp = frame["timestamp_ns"]
            if isinstance(stamp, bool) or not isinstance(stamp, int) or (previous is not None and stamp <= previous):
                raise ValueError("Frame index timestamps must be unique increasing integer nanoseconds")
            previous = stamp
            if start is not None and stamp < start:
                continue
            if end is not None and stamp > end:
                break
            path = _relative(root, frame["path"])
            if path.stat().st_size != frame["bytes"] or _hash_file(path) != frame["sha256"]:
                raise ValueError("Cached JPEG integrity mismatch")
            yield stamp, path

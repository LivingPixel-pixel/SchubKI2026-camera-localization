"""Offline place retrieval preparation. No inference geometry or automatic network.

Only FrozenEncoder imports Torch, and only after a local weight hash is checked.
Labels are consumed by offline sampling/evaluation, never by the encoder/head.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import time
import uuid

import numpy as np


DATASET_SCHEMA = "schubki-place-dataset-v1"
# Explicit five-lap experiment policy (TRAIN-FIVE-KIRO-01); v1 constants below stay unchanged.
DATASET_SCHEMA_FIVE = "schubki-place-dataset-v2"
FIVE_POLICY = "train_1_5_validation_6_7_v1"
FIVE_TRAIN_LAPS, FIVE_VALIDATION_LAPS = (1, 2, 3, 4, 5), (6, 7)
FIVE_INDEXED = {"1": 2821, "2": 1676, "3": 1236, "4": 1013, "5": 1523, "6": 1205, "7": 1095}
FIVE_MAX_ROWS, FIVE_MAX_ENCODED_BYTES = 12000, 3 * 1024**3 // 2
CACHE_SCHEMA = "schubki-place-features-v1"
MODEL_SCHEMA = "schubki-place-projection-v1"
ARCHITECTURE = "mobilenet_v3_large.IMAGENET1K_V2.avgpool960"
PREPROCESSING = {
    "decode": "opencv.IMREAD_COLOR|IMREAD_IGNORE_ORIENTATION",
    "pixels": "stored_pixels_ignore_EXIF", "color": "BGR_to_RGB",
    "resize_short_edge": 232, "center_crop": [224, 224],
    "interpolation": "torchvision_tensor_bilinear", "antialias": True,
    "scale": "uint8_div_255", "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225], "augmentation": "none",
}
RECIPE = {
    "seed": 17, "positive_m": 5.0, "positive_heading_deg": 15.0,
    "negative_m": 25.0, "positives_per_anchor": 4, "negatives_per_anchor": 8,
    "max_pairs": 100000, "max_triplets": 100000, "margin": 0.2, "l2": 0.0001,
    "learning_rate": 0.001, "batch_size": 64, "max_updates": 500,
    "checkpoints": [0, 100, 250, 500], "train_laps": [1, 2],
    "validation_laps": [3], "selection": "Recall@5,Recall@1,earlier_step",
}
ROW_KEYS = {"frame_id", "image_sha256", "lap_id", "role", "body_x_m",
            "body_y_m", "body_yaw_rad"}
OBS_KEYS = {"frame_id", "camera_id", "exposure_timestamp_ns", "image_path",
            "image_sha256", "image_bytes", "calibration_sha256"}
LABEL_KEYS = ROW_KEYS | {"association"}
ASSOCIATION_KEYS = {"method", "source_row", "source_timestamp_ns", "offset_ns",
                    "maximum_offset_ns"}
DATASET_KEYS = {"schema", "stage", "complete", "images_verified", "inputs", "intervals",
    "config", "per_lap", "index_rows", "outside_fit_index_rows", "fit_images",
    "declared_encoded_bytes", "observations", "labels", "verified_encoded_bytes",
    "rejections", "ground_truth_counts", "source", "label_convention", "previously_exposed_fit_data"}
LABEL_CONVENTION = {"body_frame": "constructor0/base_link", "map_frame": "map",
    "position_units": "metres", "yaw_units": "radians", "method": "nearest_body_xy_yaw",
    "maximum_offset_ns": 50000000, "tie_break": "earlier_source_timestamp",
    "source_row": "one-based CSV record including header",
    "interval_membership": "start_inclusive_end_exclusive"}


class NotReady(ValueError):
    """The fixed recipe has insufficient valid training support."""


def canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       allow_nan=False, ensure_ascii=False) + "\n").encode("utf-8")


def digest(data):
    return hashlib.sha256(data).hexdigest()


PREPROCESSING_SHA256 = digest(canonical(PREPROCESSING))


def sha(value):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("full lowercase SHA256 required")
    return value


def exact(value, keys):
    if not isinstance(value, dict) or set(value) != set(keys):
        raise ValueError("unexpected fields: expected " + ",".join(sorted(keys)))


def integer(value, minimum, maximum):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError("integer outside finite cap")
    return value


def finite(value):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError("finite number required")
    return float(value)


def unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON field")
        result[key] = value
    return result


def parse_json(data):
    return json.loads(data, object_pairs_hook=unique,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON")))


def file_hash(path, cap):
    path = Path(path)
    if not path.is_file() or not 1 <= path.stat().st_size <= cap:
        raise ValueError("file size cap: " + path.name)
    hasher, count = hashlib.sha256(), 0
    with path.open("rb") as handle:
        while block := handle.read(min(1024 * 1024, cap + 1 - count)):
            count += len(block)
            if count > cap:
                raise ValueError("file grew beyond cap")
            hasher.update(block)
    return hasher.hexdigest()


def read_pinned(path, expected, cap=16 * 1024 * 1024):
    if file_hash(path, cap) != sha(expected):
        raise ValueError("file SHA256 mismatch: " + Path(path).name)
    with Path(path).open("rb") as handle:
        data = handle.read(cap + 1)
    if len(data) > cap or digest(data) != expected:
        raise ValueError("file changed while reading")
    return data


def local_path(root, relative):
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise ValueError("portable relative path required")
    if PurePosixPath(relative).is_absolute() or PureWindowsPath(relative).drive:
        raise ValueError("absolute path forbidden")
    if any(p in (".", "..", "") for p in relative.split("/")):
        raise ValueError("path traversal forbidden")
    root = Path(root).resolve()
    result = (root / relative).resolve()
    if result == root or not result.is_relative_to(root):
        raise ValueError("path leaves approved root")
    return result


@dataclass(frozen=True)
class Limits:
    max_rows: int = 6000
    max_encoded_bytes: int = 700 * 1024 * 1024
    max_image_bytes: int = 8 * 1024 * 1024
    max_pixels: int = 4 * 1024 * 1024
    wall_seconds: int = 1800

    def __post_init__(self):
        integer(self.max_rows, 1, 6000)
        integer(self.max_encoded_bytes, 1, 700 * 1024 * 1024)
        integer(self.max_image_bytes, 1, 8 * 1024 * 1024)
        integer(self.max_pixels, 1, 4 * 1024 * 1024)
        integer(self.wall_seconds, 1, 1800)


@dataclass(frozen=True)
class FiveLimits(Limits):
    """Explicit new-policy caps only (12000 rows, 1.5 GiB encoded); other caps unchanged."""
    max_rows: int = FIVE_MAX_ROWS
    max_encoded_bytes: int = FIVE_MAX_ENCODED_BYTES

    def __post_init__(self):
        integer(self.max_rows, 1, FIVE_MAX_ROWS)
        integer(self.max_encoded_bytes, 1, FIVE_MAX_ENCODED_BYTES)
        integer(self.max_image_bytes, 1, 8 * 1024 * 1024)
        integer(self.max_pixels, 1, 4 * 1024 * 1024)
        integer(self.wall_seconds, 1, 1800)


def expected_role(lap, policy=None):
    """Role for a lap under the old default policy or the explicit five-lap policy."""
    if policy == FIVE_POLICY:
        integer(lap, 1, 7)
        return "train" if lap in FIVE_TRAIN_LAPS else "validation"
    if policy is not None:
        raise ValueError("unknown lap policy")
    integer(lap, 1, 3)
    return "train" if lap in (1, 2) else "checkpoint_validation"


def train_laps(policy=None):
    return FIVE_TRAIN_LAPS if policy == FIVE_POLICY else (1, 2)


def check_deadline(deadline):
    if time.monotonic() >= deadline:
        raise ValueError("wall limit exceeded")


def row_check(row, policy=None):
    exact(row, ROW_KEYS)
    stamp = row["frame_id"]
    if not isinstance(stamp, str) or not stamp.startswith("fl:") or not stamp[3:].isdigit():
        raise ValueError("frame_id must be fl:<integer nanoseconds>")
    if stamp != "fl:" + str(integer(int(stamp[3:]), 0, 2**63 - 1)):
        raise ValueError("noncanonical frame_id")
    sha(row["image_sha256"])
    if type(row["lap_id"]) is not int or row["role"] != expected_role(row["lap_id"], policy):
        raise ValueError("role/lap mismatch")
    for key in ("body_x_m", "body_y_m", "body_yaw_rad"):
        finite(row[key])


def rows_check(rows, max_rows=6000, policy=None):
    if not isinstance(rows, list) or not 1 <= len(rows) <= max_rows:
        raise ValueError("row count cap")
    ids, hash_roles = set(), {}
    for row in rows:
        row_check(row, policy)
        previous_role = hash_roles.get(row["image_sha256"])
        if row["frame_id"] in ids or (previous_role is not None and previous_role != row["role"]):
            raise ValueError("duplicate observation ID or cross-role image hash")
        ids.add(row["frame_id"])
        hash_roles[row["image_sha256"]] = row["role"]


def count_map(value, cap):
    if not isinstance(value, dict) or len(value) > 64:
        raise ValueError("bounded count map required")
    for key, count in value.items():
        if not isinstance(key, str) or not key or len(key) > 80:
            raise ValueError("count map key")
        integer(count, 0, cap)


def dataset_policy(metadata):
    """None for the unchanged v1 dataset, FIVE_POLICY for the explicit v2 five-lap dataset."""
    return FIVE_POLICY if isinstance(metadata, dict) and metadata.get("schema") == DATASET_SCHEMA_FIVE else None


def dataset_metadata(directory, expected_sha256):
    """Validate exact nested producer envelope without opening labels or pixels."""
    directory = Path(directory)
    metadata = parse_json(read_pinned(directory / "dataset.json", expected_sha256))
    policy = dataset_policy(metadata)
    five = policy == FIVE_POLICY
    row_cap = FIVE_MAX_ROWS if five else 6000
    laps = tuple(range(1, 8)) if five else (1, 2, 3)
    exact(metadata, DATASET_KEYS | ({"policy", "role_counts", "lap_membership"} if five else set()))
    if metadata["schema"] not in (DATASET_SCHEMA, DATASET_SCHEMA_FIVE) or metadata["complete"] is not True or metadata["stage"] != "prepare" or metadata["images_verified"] is not True:
        raise ValueError("incomplete/wrong dataset schema")
    exact(metadata["inputs"], {"index", "calibration", "split", "baseline_split", "ground_truth"})
    for pin in metadata["inputs"].values():
        exact(pin, {"sha256", "bytes"})
        sha(pin["sha256"])
        integer(pin["bytes"], 1, 64 * 1024 * 1024)
    config_caps = {"max_index_rows": 12000, "max_gt_rows": 130000, "max_images": row_cap,
        "max_image_bytes": 8 * 1024 * 1024, "max_encoded_bytes": FIVE_MAX_ENCODED_BYTES if five else 700 * 1024 * 1024,
        "max_index_bytes": 8 * 1024 * 1024, "max_gt_bytes": 32 * 1024 * 1024,
        "max_metadata_bytes": 1024 * 1024, "max_output_bytes": 32 * 1024 * 1024,
        "max_wall_seconds": 300}
    exact(metadata["config"], config_caps)
    for key, cap in config_caps.items():
        if key == "max_wall_seconds":
            if not 0 < finite(metadata["config"][key]) <= cap:
                raise ValueError("dataset wall cap")
        else:
            integer(metadata["config"][key], 1, cap)
    count = integer(metadata["fit_images"], 1, metadata["config"]["max_images"])
    for name in ("observations", "labels"):
        pin = metadata[name]
        exact(pin, {"path", "sha256", "bytes", "rows"})
        integer(pin["rows"], 1, row_cap)
        if pin["path"] != name + ".jsonl" or pin["rows"] != count:
            raise ValueError("dataset output path/count")
        sha(pin["sha256"])
        integer(pin["bytes"], 1, 16 * 1024 * 1024)
    if metadata["label_convention"] != LABEL_CONVENTION:
        raise ValueError("BODY planar label convention mismatch")
    exact(metadata["source"], {"commit", "module_sha256"})
    commit = metadata["source"]["commit"]
    if not isinstance(commit, str) or len(commit) != 40 or any(c not in "0123456789abcdef" for c in commit):
        raise ValueError("dataset source commit")
    sha(metadata["source"]["module_sha256"])
    integer(metadata["index_rows"], count, 12000)
    integer(metadata["outside_fit_index_rows"], 0, metadata["index_rows"])
    declared = integer(metadata["declared_encoded_bytes"], 1, config_caps["max_encoded_bytes"])
    integer(metadata["verified_encoded_bytes"], 1, declared)
    exact(metadata["per_lap"], {str(lap) for lap in laps})
    exact(metadata["rejections"], {str(lap) for lap in laps})
    accepted, verified = 0, 0
    for lap in (str(lap) for lap in laps):
        stats = metadata["per_lap"][lap]
        exact(stats, {"indexed", "declared_bytes", "accepted", "rejected", "verified_bytes", "maximum_absolute_offset_ns"})
        integer(stats["indexed"], 0, row_cap)
        integer(stats["accepted"], 0, stats["indexed"])
        integer(stats["rejected"], 0, stats["indexed"])
        if stats["accepted"] + stats["rejected"] != stats["indexed"]:
            raise ValueError("per-lap accepted/rejected count")
        if five and (stats["indexed"] != FIVE_INDEXED[lap] or stats["rejected"] != 0):
            raise ValueError("five-lap per-lap membership differs from the pinned policy counts")
        integer(stats["declared_bytes"], 0, declared)
        integer(stats["verified_bytes"], 0, stats["declared_bytes"])
        integer(stats["maximum_absolute_offset_ns"], 0, 50000000)
        count_map(metadata["rejections"][lap], row_cap)
        accepted += stats["accepted"]
        verified += stats["verified_bytes"]
    if accepted != count or verified != metadata["verified_encoded_bytes"]:
        raise ValueError("per-lap total mismatch")
    count_map(metadata["ground_truth_counts"], metadata["config"]["max_gt_rows"])
    exposed = metadata["previously_exposed_fit_data"]
    exact(exposed, {"known_recording_demo_frames", "lap_id", "role", "status"})
    integer(exposed["known_recording_demo_frames"], 0, 6000)
    integer(exposed["lap_id"], 1, 1)
    if exposed["role"] != "train" or not isinstance(exposed["status"], str) or not exposed["status"]:
        raise ValueError("previously exposed fit-demo disclosure")
    if five:
        from . import place_dataset
        if (metadata["policy"] != place_dataset._policy_record()
                or metadata["role_counts"] != {"train": place_dataset.FIVE_TRAIN, "validation": place_dataset.FIVE_VALIDATION}
                or metadata["lap_membership"] != {str(lap): expected_role(lap, policy) for lap in laps}
                or count != place_dataset.FIVE_TOTAL):
            raise ValueError("five-lap policy, role counts or lap membership differ from the pinned policy")
    intervals = metadata["intervals"]
    if not isinstance(intervals, list) or len(intervals) != len(laps):
        raise ValueError("exact fit lap intervals required")
    by_lap = {}
    for interval in intervals:
        exact(interval, {"lap_id", "start_ns", "end_ns", "role"})
        lap = integer(interval["lap_id"], 1, len(laps))
        start = integer(interval["start_ns"], 0, 2**63 - 1)
        end = integer(interval["end_ns"], start + 1, 2**63 - 1)
        if lap in by_lap or interval["role"] != expected_role(lap, policy):
            raise ValueError("duplicate lap interval")
        by_lap[lap] = (start, end)
    if set(by_lap) != set(laps) or any(by_lap[lap][1] != by_lap[lap + 1][0] for lap in laps[:-1]):
        raise ValueError("nonadjacent fit intervals")
    return metadata, by_lap


def dataset_observations(directory, metadata, cache_root):
    pin = metadata["observations"]
    data = read_pinned(Path(directory) / pin["path"], pin["sha256"])
    lines = data.splitlines()
    if len(data) != pin["bytes"] or len(lines) != pin["rows"]:
        raise ValueError("observation bytes/rows mismatch")
    observations, ids = [], set()
    for line in lines:
        observation = parse_json(line)
        exact(observation, OBS_KEYS)
        timestamp = integer(observation["exposure_timestamp_ns"], 0, 2**63 - 1)
        if observation["camera_id"] != "fl" or observation["frame_id"] != f"fl:{timestamp}":
            raise ValueError("observation camera/time identity")
        if observation["frame_id"] in ids:
            raise ValueError("duplicate observation ID")
        ids.add(observation["frame_id"])
        sha(observation["image_sha256"])
        sha(observation["calibration_sha256"])
        if observation["calibration_sha256"] != metadata["inputs"]["calibration"]["sha256"]:
            raise ValueError("observation calibration identity mismatch")
        local_path(cache_root, observation["image_path"])
        integer(observation["image_bytes"], 1, metadata["config"]["max_image_bytes"])
        observations.append(observation)
    return observations


def load_dataset(directory, expected_sha256, cache_root, limits=None):
    """Read shared JSON only; image bytes belong to the explicit encode phase.

    A v1 dataset defaults to Limits(); the explicit v2 five-lap dataset defaults to FiveLimits().
    """
    directory = Path(directory)
    metadata, by_lap = dataset_metadata(directory, expected_sha256)
    policy = dataset_policy(metadata)
    limits = (FiveLimits() if policy == FIVE_POLICY else Limits()) if limits is None else limits
    observations = dataset_observations(directory, metadata, cache_root)
    pin = metadata["labels"]
    data = read_pinned(directory / pin["path"], pin["sha256"])
    lines = data.splitlines()
    if len(data) != pin["bytes"] or len(lines) != pin["rows"] or len(lines) > limits.max_rows:
        raise ValueError("labels bytes/rows/cap mismatch")
    labels = [parse_json(line) for line in lines]
    rows, encoded_bytes, per_lap = [], 0, {lap: 0 for lap in by_lap}
    lap_bytes, lap_offsets = {lap: 0 for lap in by_lap}, {lap: 0 for lap in by_lap}
    for obs, label in zip(observations, labels):
        exact(obs, OBS_KEYS)
        exact(label, LABEL_KEYS)
        row = {key: label[key] for key in ROW_KEYS}
        row_check(row, policy)
        timestamp = integer(obs["exposure_timestamp_ns"], 0, 2**63 - 1)
        if obs["camera_id"] != "fl" or obs["frame_id"] != f"fl:{timestamp}":
            raise ValueError("observation camera/time identity")
        if obs["frame_id"] != row["frame_id"] or obs["image_sha256"] != row["image_sha256"]:
            raise ValueError("observation/label identity mismatch")
        sha(obs["calibration_sha256"])
        local_path(cache_root, obs["image_path"])
        encoded_bytes += integer(obs["image_bytes"], 1, limits.max_image_bytes)
        if encoded_bytes > limits.max_encoded_bytes:
            raise ValueError("encoded byte cap")
        association = label["association"]
        exact(association, ASSOCIATION_KEYS)
        if association["method"] != "nearest_body_xy_yaw" or association["maximum_offset_ns"] != 50000000:
            raise ValueError("association convention")
        integer(association["source_row"], 2, metadata["config"]["max_gt_rows"] + 1)
        source_stamp = integer(association["source_timestamp_ns"], 0, 2**63 - 1)
        offset = integer(association["offset_ns"], -50000000, 50000000)
        start, end = by_lap[row["lap_id"]]
        if not start <= timestamp < end or not start <= source_stamp < end or source_stamp - timestamp != offset:
            raise ValueError("same-lap association/offset mismatch")
        rows.append(row)
        per_lap[row["lap_id"]] += 1
        lap_bytes[row["lap_id"]] += obs["image_bytes"]
        lap_offsets[row["lap_id"]] = max(lap_offsets[row["lap_id"]], abs(offset))
    rows_check(rows, limits.max_rows, policy)
    if metadata["verified_encoded_bytes"] != encoded_bytes:
        raise ValueError("encoded-byte manifest mismatch")
    if any(metadata["per_lap"][str(lap)]["accepted"] != count for lap, count in per_lap.items()):
        raise ValueError("per-lap observation count mismatch")
    for lap in by_lap:
        if metadata["per_lap"][str(lap)]["verified_bytes"] != lap_bytes[lap] or metadata["per_lap"][str(lap)]["maximum_absolute_offset_ns"] != lap_offsets[lap]:
            raise ValueError("per-lap bytes/offset mismatch")
    return metadata, observations, rows


def plan_dataset(directory, expected_sha256, cache_root, output, train_count=128, validation_count=64):
    """Metadata inventory and deterministic exposure selection, no labels/pixels."""
    integer(train_count, 1, 6000)
    integer(validation_count, 1, 6000)
    if train_count + validation_count > 6000:
        raise ValueError("selection row cap")
    metadata, intervals = dataset_metadata(directory, expected_sha256)
    observations = dataset_observations(directory, metadata, cache_root)
    train, validation = [], []
    for observation in sorted(observations, key=lambda o: o["exposure_timestamp_ns"]):
        stamp = observation["exposure_timestamp_ns"]
        lap = next((lap for lap, (start, end) in intervals.items() if start <= stamp < end), None)
        if lap is None:
            raise ValueError("observation outside fit1-3")
        (train if lap in (1, 2) else validation).append(observation)
    if len(train) < train_count or len(validation) < validation_count:
        raise NotReady("metadata lacks requested train/validation selection")
    # Balance lap1/lap2 prefixes so a finite smoke can support cross-lap pairs.
    per_lap = [[obs for obs in train if start <= obs["exposure_timestamp_ns"] < end]
               for lap, (start, end) in sorted(intervals.items()) if lap in (1, 2)]
    selected_train = []
    for index in range(max(map(len, per_lap))):
        for lap_rows in per_lap:
            if index < len(lap_rows) and len(selected_train) < train_count:
                selected_train.append(lap_rows[index])
    chosen = selected_train + validation[:validation_count]
    selected = [obs["frame_id"] for obs in chosen]
    output = new_output(output)
    write_json(output / "selection.json", selected)
    value = {"schema": "schubki-place-plan-v1", "complete": True,
        "dataset_sha256": sha(expected_sha256), "train_count": train_count,
        "validation_count": validation_count,
        "encoded_bytes": sum(obs["image_bytes"] for obs in chosen),
        "selection_sha256": digest(canonical(selected)), "images_read": 0, "labels_read": 0,
        "pair_support": "unknown; fixed pair rules checked only by offline fit",
        "selection": "timestamp prefixes balanced across train laps; fixed before outcome inspection"}
    write_json(output / "plan.json", value)
    return value


def normalize(values):
    values = np.asarray(values)
    if values.ndim != 2 or not np.issubdtype(values.dtype, np.floating) or not np.isfinite(values).all():
        raise ValueError("finite floating matrix required")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if (norms <= 1e-12).any() or not np.isfinite(norms).all():
        raise ValueError("zero/nonfinite vector")
    return values / norms


def validate_features(features, count, dimension=960):
    if features.dtype != np.dtype("float32") or features.shape != (count, dimension):
        raise ValueError("feature float32 shape mismatch")
    for start in range(0, count, 128):
        normalize(features[start:start + 128])


def load_numeric(path, expected_sha256, shape, max_bytes):
    """Check bounded NPY header and exact file length before np.load allocation."""
    path = Path(path)
    read_hash = file_hash(path, max_bytes)
    if read_hash != sha(expected_sha256):
        raise ValueError("numeric file hash mismatch")
    with path.open("rb") as handle:
        version = np.lib.format.read_magic(handle)
        if version == (1, 0):
            found_shape, fortran, dtype = np.lib.format.read_array_header_1_0(handle, max_header_size=4096)
        elif version == (2, 0):
            found_shape, fortran, dtype = np.lib.format.read_array_header_2_0(handle, max_header_size=4096)
        else:
            raise ValueError("unsupported NPY version")
        if found_shape != shape or fortran or dtype != np.dtype("float32") or dtype.hasobject:
            raise ValueError("numeric header shape/dtype/order mismatch")
        size = math.prod(shape) * 4
        if size > max_bytes or path.stat().st_size != handle.tell() + size:
            raise ValueError("numeric file payload size mismatch")
    result = np.load(path, allow_pickle=False, mmap_mode="r", max_header_size=4096)
    validate_features(result, shape[0], shape[1])
    return result


def encoder_identity(weights_sha256, dependencies):
    exact(dependencies, {"torch", "torchvision", "numpy", "opencv", "pillow"})
    if dependencies["torch"].split("+")[0] != "2.7.0" or dependencies["torchvision"].split("+")[0] != "0.22.0":
        raise ValueError("training Torch/torchvision version pin mismatch")
    if any(not isinstance(v, str) or not v for v in dependencies.values()):
        raise ValueError("dependency versions required")
    return {"architecture": ARCHITECTURE, "weights_sha256": sha(weights_sha256),
            "preprocessing": parse_json(canonical(PREPROCESSING)),
            "preprocessing_sha256": PREPROCESSING_SHA256, "dependencies": dict(dependencies)}


def check_encoder(value, expected_weights_sha256):
    exact(value, {"architecture", "weights_sha256", "preprocessing", "preprocessing_sha256", "dependencies"})
    if value != encoder_identity(expected_weights_sha256, value["dependencies"]):
        raise ValueError("encoder/preprocessing identity mismatch")


class FrozenEncoder:
    """Explicit CPU-only local checkpoint. Construction may never fetch weights."""
    def __init__(self, weight_path, expected_sha256, threads=1):
        integer(threads, 1, 4)
        weight_bytes = read_pinned(weight_path, expected_sha256, 40 * 1024 * 1024)
        dependencies = {key: importlib.metadata.version(package) for key, package in
            (("torch", "torch"), ("torchvision", "torchvision"), ("numpy", "numpy"),
             ("opencv", "opencv-python-headless"), ("pillow", "pillow"))}
        self.identity = encoder_identity(expected_sha256, dependencies)
        import torch
        import io
        from torchvision.models import mobilenet_v3_large
        from torchvision.transforms import functional as functional
        from torchvision.transforms import InterpolationMode
        torch.set_num_threads(threads)
        model = mobilenet_v3_large(weights=None)
        state = torch.load(io.BytesIO(weight_bytes), map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
        self.model = torch.nn.Sequential(model.features, model.avgpool, torch.nn.Flatten(1))
        self.model.requires_grad_(False)
        self.model.eval()
        self.torch, self.functional = torch, functional
        self.interpolation = InterpolationMode.BILINEAR

    def encode_rgb(self, rgb):
        if not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError("encoder takes RGB uint8 pixels only")
        image_dimensions(rgb.shape[1], rgb.shape[0], 4 * 1024 * 1024)
        torch, functional = self.torch, self.functional
        tensor = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1)
        tensor = functional.resize(tensor, [232], interpolation=self.interpolation, antialias=True)
        tensor = functional.center_crop(tensor, [224, 224])
        tensor = functional.convert_image_dtype(tensor, torch.float32)
        tensor = functional.normalize(tensor, PREPROCESSING["mean"], PREPROCESSING["std"])
        with torch.inference_mode():
            features = self.model(tensor.unsqueeze(0)).cpu().numpy().astype(np.float32, copy=False)
        validate_features(features, 1)
        return features[0]


def image_dimensions(width, height, max_pixels):
    if min(width, height) < 1 or width * height > max_pixels:
        raise ValueError("decoded pixel cap")
    # Short-edge resize can explode a very thin image despite a small input.
    resized_long_edge = int(232 * max(width, height) / min(width, height))
    if 232 * resized_long_edge > 4 * 1024 * 1024:
        raise ValueError("resized pixel cap")


def decode_rgb(data, max_pixels):
    """Header-size gate precedes OpenCV decode; no EXIF orientation correction."""
    import io
    from PIL import Image
    import cv2
    with Image.open(io.BytesIO(data)) as header:
        width, height = header.size
        image_dimensions(width, height, max_pixels)
    bgr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if bgr is None or bgr.shape[:2] != (height, width):
        raise ValueError("stored-pixel image decode failed")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def write_json(path, value):
    """Publish closed bytes without replacing an existing completion path."""
    path = Path(path)
    data = canonical(value)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    created = False
    try:
        with temporary.open("xb") as handle:
            created = True
            handle.write(data)
        os.link(temporary, path)
    finally:
        # Cleanup must not turn a successful publication into a failed write.
        try:
            if created:
                temporary.unlink(missing_ok=True)
        except OSError:
            pass


def new_output(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    return path


def failed_output(path, error):
    write_json(Path(path) / "failure.json", {"complete": False, "error": str(error),
                                              "type": type(error).__name__})


def encode_dataset(dataset_dir, dataset_sha256, cache_root, output, weights,
                   weights_sha256, source_sha256, limits=Limits(), threads=1,
                   selection=None, encoder_factory=FrozenEncoder, role_caps=(6000, 6000)):
    """Root-approved encode phase only. Fixed explicit selection precedes decoding."""
    deadline = time.monotonic() + limits.wall_seconds
    # Manifest inventory may exceed the explicitly selected encode envelope.
    _, observations, rows = load_dataset(dataset_dir, dataset_sha256, cache_root)
    if selection is not None:
        if not isinstance(selection, list) or not 1 <= len(selection) <= limits.max_rows or not all(isinstance(key, str) for key in selection) or len(selection) != len(set(selection)):
            raise ValueError("unique predeclared frame selection required")
        lookup = {obs["frame_id"]: (obs, row) for obs, row in zip(observations, rows)}
        if not selection or len(selection) > limits.max_rows or any(key not in lookup for key in selection):
            raise ValueError("selection unknown/cap")
        observations, rows = map(list, zip(*(lookup[key] for key in selection)))
    if len(rows) > limits.max_rows:
        raise ValueError("selected row cap")
    if not isinstance(role_caps, tuple) or len(role_caps) != 2:
        raise ValueError("train/validation role caps required")
    train_cap, validation_cap = (integer(cap, 1, 6000) for cap in role_caps)
    if sum(row["role"] == "train" for row in rows) > train_cap or sum(row["role"] == "checkpoint_validation" for row in rows) > validation_cap:
        raise ValueError("selected per-role row cap")
    encoded_total = sum(obs["image_bytes"] for obs in observations)
    if encoded_total > limits.max_encoded_bytes or any(obs["image_bytes"] > limits.max_image_bytes for obs in observations):
        raise ValueError("selected encoded byte cap")
    sha(source_sha256)
    output = new_output(output)
    try:
        encoder = encoder_factory(weights, weights_sha256, threads)
        check_encoder(encoder.identity, weights_sha256)
        features = np.lib.format.open_memmap(output / "features.npy", mode="w+",
                                             dtype=np.float32, shape=(len(rows), 960))
        for index, observation in enumerate(observations):
            check_deadline(deadline)
            path = local_path(cache_root, observation["image_path"])
            data = read_pinned(path, observation["image_sha256"], limits.max_image_bytes)
            if len(data) != observation["image_bytes"]:
                raise ValueError("image byte identity mismatch")
            features[index] = encoder.encode_rgb(decode_rgb(data, limits.max_pixels))
        features.flush()
        validate_features(features, len(rows))
        del features
        write_json(output / "rows.json", rows)
        check_deadline(deadline)
        metadata = {"schema": CACHE_SCHEMA, "complete": True, "count": len(rows),
            "dataset_sha256": sha(dataset_sha256), "encoder": encoder.identity,
            "source_sha256": sha(source_sha256), "limits": asdict(limits),
            "selection_sha256": digest(canonical([row["frame_id"] for row in rows])),
            "rows_sha256": file_hash(output / "rows.json", 16 * 1024 * 1024),
            "features_sha256": file_hash(output / "features.npy", 6000 * 960 * 4 + 4096)}
        write_json(output / "cache.json", metadata)
        return metadata
    except Exception as error:
        failed_output(output, error)
        raise


def load_cache(directory, expected_cache_sha256, expected_dataset_sha256,
               expected_weights_sha256, max_rows=6000):
    directory = Path(directory)
    value = parse_json(read_pinned(directory / "cache.json", expected_cache_sha256))
    exact(value, {"schema", "complete", "count", "dataset_sha256", "encoder", "source_sha256",
                  "limits", "selection_sha256", "rows_sha256", "features_sha256"})
    if value["schema"] != CACHE_SCHEMA or value["complete"] is not True:
        raise ValueError("incomplete/wrong feature cache")
    if value["dataset_sha256"] != sha(expected_dataset_sha256):
        raise ValueError("cache dataset identity mismatch")
    check_encoder(value["encoder"], expected_weights_sha256)
    sha(value["source_sha256"])
    exact(value["limits"], set(asdict(Limits())))
    Limits(**value["limits"])
    count = integer(value["count"], 1, min(integer(max_rows, 1, 6000), value["limits"]["max_rows"]))
    rows = parse_json(read_pinned(directory / "rows.json", value["rows_sha256"]))
    rows_check(rows, max_rows)
    if len(rows) != count or value["selection_sha256"] != digest(canonical([row["frame_id"] for row in rows])):
        raise ValueError("cache row selection/count identity mismatch")
    features = load_numeric(directory / "features.npy", value["features_sha256"],
                             (count, 960), 6000 * 960 * 4 + 4096)
    return value, rows, features


def heading_distance(a, b):
    return np.abs(np.arctan2(np.sin(a - b), np.cos(a - b)))


def pair_masks(anchor, candidates):
    distance = np.hypot(candidates[:, 0] - anchor[0], candidates[:, 1] - anchor[1])
    positive = (distance <= 5.0) & (heading_distance(candidates[:, 2], anchor[2]) <= math.radians(15))
    return positive, distance >= 25.0


def mine_triplets(rows, deadline=None, policy=None):
    """Existing cross-lap miner. Under FIVE_POLICY anchors are visited round-robin over laps 1-5
    BEFORE the unchanged 100000-triplet cap, so the cap cannot keep only an early-lap prefix."""
    five = policy == FIVE_POLICY
    rows_check(rows, FIVE_MAX_ROWS if five else 6000, policy)
    laps_allowed = train_laps(policy)
    indices = np.array([i for i, row in enumerate(rows) if row["lap_id"] in laps_allowed], dtype=np.int64)
    if not len(indices):
        raise NotReady("no train-lap rows")
    labels = np.array([[rows[i][key] for key in ("body_x_m", "body_y_m", "body_yaw_rad")]
                       for i in indices], dtype=np.float64)
    laps = np.array([rows[i]["lap_id"] for i in indices])
    hashes = np.array([rows[i]["image_sha256"] for i in indices])
    order = list(range(len(indices)))
    if five:
        by_lap = [[k for k in order if laps[k] == lap] for lap in laps_allowed]
        order = [group[turn] for turn in range(max(map(len, by_lap))) for group in by_lap if turn < len(group)]
    rng, triplets, eligible, pair_count = np.random.default_rng(17), [], 0, 0
    anchors_per_lap = {str(lap): 0 for lap in laps_allowed}
    for local_index in order:
        anchor_index = indices[local_index]
        if five and len(triplets) == 100000:
            break
        if deadline is not None:
            check_deadline(deadline)
        positive, negative = pair_masks(labels[local_index], labels)
        # Both positives and negatives come from the other train lap only.
        positive &= laps != laps[local_index]
        negative &= laps != laps[local_index]
        distinct_hash = hashes != hashes[local_index]
        positive &= distinct_hash
        negative &= distinct_hash
        positives, negatives = indices[positive], indices[negative]
        if not len(positives) or not len(negatives):
            continue
        eligible += 1
        anchors_per_lap[str(int(laps[local_index]))] += 1
        positives = rng.permutation(positives)[:4]
        negatives = rng.permutation(negatives)[:8]
        pair_count += len(positives) + len(negatives)
        if pair_count > 100000:
            raise ValueError("stored pair cap")
        for p in positives:
            for n in negatives:
                if len(triplets) == 100000:
                    break
                triplets.append((int(anchor_index), int(p), int(n)))
            if len(triplets) == 100000:
                break
    if not triplets:
        raise NotReady("no cross-train-lap positives/negatives under fixed 5m/15deg/25m rules")
    support = {"train_anchors": len(indices),
        "eligible_anchors": eligible, "sampled_pairs": pair_count,
        "triplets": len(triplets), "cap_reached": len(triplets) == 100000}
    if five:
        support.update(policy=FIVE_POLICY, anchor_order="lap_round_robin_before_cap",
                       eligible_anchors_per_lap=anchors_per_lap)
    return np.array(triplets, dtype=np.int64), support


def initial_projection(seed=17):
    if seed != 17:
        raise ValueError("first recipe seed is fixed at17")
    return (np.random.default_rng(seed).standard_normal((128, 960)) / math.sqrt(960)).astype(np.float32)


def project(features, projection):
    if projection.shape != (128, 960) or projection.dtype != np.float32 or not np.isfinite(projection).all():
        raise ValueError("projection float32[128,960] required")
    return normalize(np.asarray(features) @ projection.T)


def loss_gradient(projection, features, triplets):
    """Analytic normalized squared-distance hinge plus mean(W^2) gradient."""
    if triplets.ndim != 2 or triplets.shape[1] != 3 or not 1 <= len(triplets) <= 64:
        raise ValueError("batch triplet cap")
    if not np.issubdtype(triplets.dtype, np.integer) or triplets.min() < 0 or triplets.max() >= len(features):
        raise ValueError("triplet index")
    selected = features[triplets].astype(np.float64)
    weights = projection.astype(np.float64)
    raw = selected @ weights.T
    length = np.linalg.norm(raw, axis=2, keepdims=True)
    if (length <= 1e-12).any() or not np.isfinite(length).all():
        raise ValueError("zero/nonfinite projected feature")
    unit = raw / length
    a, p, n = unit[:, 0], unit[:, 1], unit[:, 2]
    hinge = 0.2 + np.sum((a - p)**2, axis=1) - np.sum((a - n)**2, axis=1)
    active = hinge > 0
    derivative = np.zeros_like(unit)
    derivative[:, 0] = 2 * (n - p)
    derivative[:, 1] = 2 * (p - a)
    derivative[:, 2] = 2 * (a - n)
    derivative *= active[:, None, None] / len(triplets)
    raw_derivative = (derivative - unit * np.sum(derivative * unit, axis=2, keepdims=True)) / length
    gradient = np.einsum("bkd,bkf->df", raw_derivative, selected)
    gradient += (2 * 1e-4 / weights.size) * weights
    loss = float(np.maximum(0, hinge).mean() + 1e-4 * np.mean(weights**2))
    if not math.isfinite(loss) or not np.isfinite(gradient).all():
        raise ValueError("nonfinite loss/gradient")
    return loss, gradient.astype(np.float32)


def adam_update(weights, gradient, first, second, step):
    integer(step, 1, 500)
    first *= 0.9
    first += 0.1 * gradient
    second *= 0.999
    second += 0.001 * gradient**2
    weights -= 0.001 * (first / (1 - 0.9**step)) / (np.sqrt(second / (1 - 0.999**step)) + 1e-8)
    if not np.isfinite(weights).all():
        raise ValueError("nonfinite Adam parameter")


def retrieval_metrics(features, rows, projection=None, deadline=None, cases=None):
    """Lap3 queries against fixed train gallery, one query at a time, stable ties."""
    rows_check(rows)
    validate_features(features, len(rows))
    gallery = np.array([i for i, row in enumerate(rows) if row["lap_id"] in (1, 2)], dtype=np.int64)
    queries = np.array([i for i, row in enumerate(rows) if row["lap_id"] == 3], dtype=np.int64)
    if not len(gallery) or not len(queries):
        raise NotReady("lap1-2 gallery and lap3 checkpoint queries required")
    labels = np.array([[row[key] for key in ("body_x_m", "body_y_m", "body_yaw_rad")]
                       for row in rows])
    embeddings = normalize(np.asarray(features)) if projection is None else project(features, projection)
    hits1, hits5, eligible, distant, no_positive_hits = 0, 0, 0, 0, 0
    for query in queries:
        if deadline is not None:
            check_deadline(deadline)
        positive, negative = pair_masks(labels[query], labels[gallery])
        supported = bool(positive.any())
        eligible += int(supported)
        rank = np.argsort(-(embeddings[gallery] @ embeddings[query]), kind="stable")[:5]
        hits1 += int(positive[rank[0]])
        hits5 += int(positive[rank].any())
        distant += int(negative[rank[0]])
        no_positive_hits += int(not supported)
        if cases is not None and (not supported or negative[rank[0]]):
            cases.append({"query_frame_id": rows[query]["frame_id"],
                "top1_gallery_frame_id": rows[gallery[rank[0]]]["frame_id"],
                "no_positive_support": not supported,
                "false_distant_top1": bool(negative[rank[0]])})
    count = len(queries)
    return {"gallery_count": len(gallery), "all_queries": count, "eligible_queries": eligible,
        "no_positive_queries": no_positive_hits, "positive_support_coverage": eligible / count,
        "recall_at_1_all": hits1 / count, "recall_at_5_all": hits5 / count,
        "recall_at_1_eligible": hits1 / eligible if eligible else None,
        "recall_at_5_eligible": hits5 / eligible if eligible else None,
        "false_distant_top1": distant, "interpretation": "retrieval_only_shadow_ranking"}


def fit_projection(cache_dir, cache_sha256, dataset_sha256, weights_sha256, output,
                   source_sha256, updates=500, wall_seconds=1800):
    integer(updates, 1, 500)
    integer(wall_seconds, 1, 1800)
    deadline = time.monotonic() + wall_seconds
    cache, rows, features = load_cache(cache_dir, cache_sha256, dataset_sha256, weights_sha256)
    triplets, support = mine_triplets(rows, deadline)
    output = new_output(output)
    try:
        weights = initial_projection()
        baseline_cases, initial_cases = [], []
        baseline = retrieval_metrics(features, rows, deadline=deadline, cases=baseline_cases)
        initial = retrieval_metrics(features, rows, weights, deadline, initial_cases)
        if not initial["eligible_queries"]:
            raise NotReady("lap3 has no positive gallery support")
        history = [{"step": 0, "loss": None, "metrics": initial}]
        best_weights, best_step, best_metrics = weights.copy(), 0, initial
        best_cases = initial_cases
        first, second = np.zeros_like(weights), np.zeros_like(weights)
        rng = np.random.default_rng(17)
        order, cursor = rng.permutation(len(triplets)), 0
        checkpoints = {0, 100, 250, 500, updates}
        save_projection(output / "checkpoint-000", weights, cache, cache_sha256, source_sha256, 0, initial)
        for step in range(1, updates + 1):
            check_deadline(deadline)
            if cursor >= len(order):
                order, cursor = rng.permutation(len(triplets)), 0
            batch = triplets[order[cursor:cursor + 64]]
            cursor += len(batch)
            loss, gradient = loss_gradient(weights, features, batch)
            adam_update(weights, gradient, first, second, step)
            if step in checkpoints:
                final_cases = []
                metrics = retrieval_metrics(features, rows, weights, deadline, final_cases)
                history.append({"step": step, "loss": loss, "metrics": metrics})
                save_projection(output / f"checkpoint-{step:03d}", weights, cache, cache_sha256,
                                source_sha256, step, metrics)
                if (metrics["recall_at_5_all"], metrics["recall_at_1_all"]) > (
                    best_metrics["recall_at_5_all"], best_metrics["recall_at_1_all"]):
                    best_weights, best_step, best_metrics = weights.copy(), step, metrics
                    best_cases = final_cases
        if np.array_equal(weights, initial_projection()):
            raise NotReady("parameters did not change")
        check_deadline(deadline)
        model = save_projection(output / "selected", best_weights, cache, cache_sha256,
                                source_sha256, best_step, best_metrics)
        _, reloaded = load_projection(output / "selected", digest(canonical(model)),
                                     weights_sha256, cache_sha256, dataset_sha256)
        if not np.array_equal(best_weights, reloaded):
            raise ValueError("selected checkpoint numeric reload mismatch")
        probe = features[:min(8, len(features))]
        reload_difference = float(np.max(np.abs(project(probe, best_weights) - project(probe, reloaded))))
        if reload_difference != 0.0:
            raise ValueError("selected checkpoint embedding reload mismatch")
        report = {"complete": True, "recipe": RECIPE, "updates_executed": updates,
            "support": support, "pretrained_960d": baseline, "initial_random_128d": initial,
            "fitted_final_128d": history[-1]["metrics"],
            "unsupported_or_false_distant_cases": {"pretrained_960d": baseline_cases,
                "initial_random_128d": initial_cases, "fitted_final_128d": final_cases,
                "selected_128d": best_cases},
            "triplets_sha256": digest(triplets.astype("<i8").tobytes()),
            "triplets_layout": "little-endian-int64[N,3]; row indices into pinned rows.json",
            "checkpoints": history, "selected_step": best_step, "selected_metrics": best_metrics,
            "last_parameter_delta_l2": float(np.linalg.norm(weights - initial_projection())),
            "reload_parameters_exact": True, "reload_embedding_max_absolute_difference": reload_difference,
            "selected_model_sha256": digest(canonical(model)),
            "cache_sha256": cache_sha256, "dataset_sha256": dataset_sha256,
            "source_sha256": sha(source_sha256), "wall_seconds_cap": wall_seconds,
            "resume": "fresh deterministic run only; no optimizer state in inference artifact",
            "claims": "offline retrieval; no official localization/geometry coverage claim"}
        check_deadline(deadline)
        write_json(output / "fit.json", report)
        return report
    except Exception as error:
        failed_output(output, error)
        raise


def save_projection(output, projection, cache, cache_sha256, source_sha256, step, metrics):
    output = new_output(output)
    # A projection is a tensor only. No labels, optimizer, lap/time inputs or GT.
    if projection.shape != (128, 960) or projection.dtype != np.float32 or not np.isfinite(projection).all():
        raise ValueError("invalid projection")
    np.save(output / "projection.npy", projection, allow_pickle=False)
    projection_sha256 = file_hash(output / "projection.npy", 128 * 960 * 4 + 4096)
    reloaded = load_numeric(output / "projection.npy", projection_sha256,
                            (128, 960), 128 * 960 * 4 + 4096)
    if not np.array_equal(projection, reloaded):
        raise ValueError("checkpoint numeric reload mismatch")
    value = {"schema": MODEL_SCHEMA, "complete": True, "shape": [128, 960],
        "projection_sha256": projection_sha256,
        "cache_sha256": sha(cache_sha256), "dataset_sha256": sha(cache["dataset_sha256"]),
        "encoder": cache["encoder"], "source_sha256": sha(source_sha256),
        "recipe": RECIPE, "step": integer(step, 0, 500), "metrics": metrics,
        "use": "shadow_retrieval_only_incomplete_geometric_candidate_coverage"}
    write_json(output / "model.json", value)
    return value


def load_projection(directory, expected_model_sha256, expected_weights_sha256,
                    expected_cache_sha256, expected_dataset_sha256):
    directory = Path(directory)
    value = parse_json(read_pinned(directory / "model.json", expected_model_sha256))
    exact(value, {"schema", "complete", "shape", "projection_sha256", "cache_sha256",
                  "dataset_sha256", "encoder", "source_sha256", "recipe", "step", "metrics", "use"})
    if value["schema"] != MODEL_SCHEMA or value["complete"] is not True or value["shape"] != [128, 960]:
        raise ValueError("incomplete/wrong projection schema")
    if value["cache_sha256"] != sha(expected_cache_sha256) or value["dataset_sha256"] != sha(expected_dataset_sha256):
        raise ValueError("projection cache/dataset identity mismatch")
    check_encoder(value["encoder"], expected_weights_sha256)
    sha(value["source_sha256"])
    if value["recipe"] != RECIPE or value["use"] != "shadow_retrieval_only_incomplete_geometric_candidate_coverage":
        raise ValueError("recipe/use identity mismatch")
    integer(value["step"], 0, 500)
    return value, load_numeric(directory / "projection.npy", value["projection_sha256"],
                               (128, 960), 128 * 960 * 4 + 4096)


def validate_projection(cache_dir, cache_sha256, dataset_sha256, weights_sha256,
                        model_dir, model_sha256, output, wall_seconds=600):
    integer(wall_seconds, 1, 1800)
    deadline = time.monotonic() + wall_seconds
    _, rows, features = load_cache(cache_dir, cache_sha256, dataset_sha256, weights_sha256)
    model, projection = load_projection(model_dir, model_sha256, weights_sha256,
                                        cache_sha256, dataset_sha256)
    output = new_output(output)
    try:
        cases = []
        result = {"complete": True, "model_sha256": model_sha256,
            "cache_sha256": cache_sha256, "dataset_sha256": dataset_sha256,
            "selected_step": model["step"],
            "pretrained_960d": retrieval_metrics(features, rows, deadline=deadline),
            "initial_random_128d": retrieval_metrics(features, rows, initial_projection(), deadline),
            "selected_128d": retrieval_metrics(features, rows, projection, deadline, cases),
            "selected_unsupported_or_false_distant_cases": cases,
            "claims": "retrieval only; all/eligible denominators; no geometry coverage claim"}
        check_deadline(deadline)
        write_json(output / "validation.json", result)
        return result
    except Exception as error:
        failed_output(output, error)
        raise

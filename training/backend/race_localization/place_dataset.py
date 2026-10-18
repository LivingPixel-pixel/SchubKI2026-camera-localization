"""Bounded training-only BODY XY/yaw dataset preparation, without image decoding.

Plan reads pinned metadata only. Explicit prepare additionally reads GT, partitions
by exact timestamp before accessing pose values, and stream-verifies fit images.
No inference, MCAP, geometry constructor, encoder or network dependency is used.
"""
from __future__ import annotations

from bisect import bisect_left
from collections import Counter
from dataclasses import asdict, dataclass
import csv
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import re
import time

SCHEMA = "schubki-place-dataset-v1"
# Experiment-only role override (TRAIN-FIVE-KIRO-01). The historical global team split is still
# verified unchanged as a source; only this explicit policy reassigns roles for this experiment.
SCHEMA_FIVE = "schubki-place-dataset-v2"
POLICY_V1 = "train_1_2_validation_3_v1"
POLICY_FIVE = "train_1_5_validation_6_7_v1"
POLICIES = {
    POLICY_V1: {"schema": SCHEMA, "laps": (1, 2, 3), "train": (1, 2), "validation_role": "checkpoint_validation"},
    POLICY_FIVE: {"schema": SCHEMA_FIVE, "laps": (1, 2, 3, 4, 5, 6, 7), "train": (1, 2, 3, 4, 5),
                  "validation_role": "validation"},
}
# Root metadata-only indexed counts, pinned as exact expectations; NOT a prepared or verified dataset.
FIVE_INDEXED_COUNTS = {"1": 2821, "2": 1676, "3": 1236, "4": 1013, "5": 1523, "6": 1205, "7": 1095}
FIVE_TOTAL, FIVE_TRAIN, FIVE_VALIDATION = 10569, 8269, 2300
FIVE_DECLARED_ENCODED_BYTES = 1_212_793_150
MAX_OFFSET_NS = 50_000_000
_SHA = re.compile(r"[0-9a-f]{64}")
_COMMIT = re.compile(r"[0-9a-f]{40}")
_INT = re.compile(r"[0-9]+")


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _exact(value, required, optional=()):
    if not isinstance(value, dict) or not set(required) <= set(value) or set(value) - set(required) - set(optional):
        raise ValueError("unknown or missing fields")


def _sha(value):
    if not isinstance(value, str) or _SHA.fullmatch(value) is None:
        raise ValueError("lowercase SHA256 required")
    return value


def _integer(value, name, minimum=0):
    if type(value) is not int or not minimum <= value <= 2**63 - 1:
        raise ValueError(f"{name}: exact bounded integer required")
    return value


def _timestamp(value):
    return _integer(value, "timestamp_ns")


def _finite(value):
    try:
        valid = not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)
    except OverflowError:
        valid = False
    if not valid:
        raise ValueError("finite number required")
    return float(value)


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON field")
        result[key] = value
    return result


def _json(data):
    return json.loads(data, object_pairs_hook=_unique,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError("nonfinite JSON")))


@dataclass(frozen=True)
class Limits:
    max_index_rows: int = 12_000
    max_gt_rows: int = 120_000
    max_images: int = 6_000
    max_image_bytes: int = 4 * 1024 * 1024
    max_encoded_bytes: int = 700 * 1024 * 1024
    max_index_bytes: int = 8 * 1024 * 1024
    max_gt_bytes: int = 32 * 1024 * 1024
    max_metadata_bytes: int = 1024 * 1024
    max_output_bytes: int = 32 * 1024 * 1024
    max_wall_seconds: float = 300.0

    def __post_init__(self):
        for name, value in asdict(self).items():
            if name == "max_wall_seconds":
                if _finite(value) <= 0:
                    raise ValueError("positive finite wall cap required")
            else:
                _integer(value, name, 1)


# Explicit new-policy row/encoded caps; old Limits() defaults (6000 rows, 700 MiB) are unchanged.
FIVE_LIMITS = Limits(max_images=12_000, max_encoded_bytes=3 * 1024**3 // 2)
FAILURE_SCHEMA = "schubki-place-dataset-failure-v1"


class PreparationFailure(ValueError):
    """Five-lap association failure with bounded structured evidence. No dataset marker or output exists."""

    def __init__(self, message, diagnostics):
        super().__init__(message)
        self.diagnostics = diagnostics


def association_diagnostics(policy, intervals, per_lap, rejections, offsets, gt_counts):
    """Bounded per-lap indexed/accepted/rejected/reason evidence; images are NOT hash-verified at this stage."""
    laps = {}
    for lap in intervals:
        key = str(lap["lap_id"])
        laps[key] = {"role": lap["role"], "indexed": per_lap[key]["indexed"], "accepted": len(offsets[key]),
                     "rejected": sum(rejections[key].values()),
                     "rejection_reasons": dict(sorted(rejections[key].items())),
                     "maximum_absolute_offset_ns": max(offsets[key]) if offsets[key] else None}
    return {"schema": FAILURE_SCHEMA, "policy": policy, "complete": False, "stage": "association",
            "images_verified": False, "dataset_written": False, "per_lap": laps,
            "accepted_total": sum(item["accepted"] for item in laps.values()),
            "indexed_total": sum(item["indexed"] for item in laps.values()),
            "ground_truth_counts": dict(sorted(gt_counts.items())), "maximum_offset_ns": MAX_OFFSET_NS,
            "contains": "association counts only; no pose values, labels, image bytes or hash checks"}


def require_complete_association(policy, intervals, per_lap, labels, rejections, offsets, gt_counts):
    """Five-lap: every pinned row associates and all laps are populated, else raise PreparationFailure."""
    missing = [str(lap["lap_id"]) for lap in intervals if not offsets[str(lap["lap_id"])]]
    if policy != POLICY_FIVE:
        if missing:
            raise ValueError("each fit lap requires a valid same-lap association")
        return
    if missing or len(labels) != FIVE_TOTAL or any(sum(r.values()) for r in rejections.values()):
        raise PreparationFailure("five-lap policy requires every pinned row to associate within 50ms; nothing is relabelled",
                                 association_diagnostics(policy, intervals, per_lap, rejections, offsets, gt_counts))


def _policy(policy, limits):
    if type(policy) is not str or policy not in POLICIES:
        raise ValueError("explicit known lap policy required")
    limits = (FIVE_LIMITS if policy == POLICY_FIVE else Limits()) if limits is None else limits
    if not isinstance(limits, Limits):
        raise ValueError("typed Limits required")
    if policy == POLICY_FIVE and (limits.max_images > FIVE_LIMITS.max_images
                                  or limits.max_encoded_bytes > FIVE_LIMITS.max_encoded_bytes):
        raise ValueError("five-lap policy cannot exceed its explicit 12000-row/1.5GiB caps")
    return POLICIES[policy], limits


@dataclass(frozen=True)
class Inputs:
    index: Path
    calibration: Path
    split: Path
    baseline_split: Path
    cache_root: Path
    index_sha256: str
    calibration_sha256: str
    split_sha256: str
    baseline_split_sha256: str

    def __post_init__(self):
        for name in ("index", "calibration", "split", "baseline_split", "cache_root"):
            object.__setattr__(self, name, Path(getattr(self, name)))
        for name in ("index_sha256", "calibration_sha256", "split_sha256", "baseline_split_sha256"):
            _sha(getattr(self, name))


class _Budget:
    def __init__(self, limits):
        self.limits = limits
        self.deadline = time.monotonic() + limits.max_wall_seconds

    def check(self):
        if time.monotonic() >= self.deadline:
            raise ValueError("wall cap exceeded")


def _snapshot(path, expected, cap, budget):
    budget.check()
    if not 1 <= path.stat().st_size <= cap:
        raise ValueError("input byte cap exceeded")
    with path.open("rb") as stream:
        raw = stream.read(cap + 1)
    budget.check()
    if not 1 <= len(raw) <= cap or hashlib.sha256(raw).hexdigest() != _sha(expected):
        raise ValueError("input hash or byte cap mismatch")
    return raw


def _local_path(root, relative):
    if not isinstance(relative, str) or not relative or "\\" in relative or ":" in relative:
        raise ValueError("normalized relative image path required")
    pure = PurePosixPath(relative)
    if pure.is_absolute() or any(p in (".", "..") for p in pure.parts) or pure.as_posix() != relative:
        raise ValueError("image path escape or noncanonical path")
    root = root.resolve()
    path = (root / relative).resolve()
    if path == root or not path.is_relative_to(root):
        raise ValueError("image path escapes cache root")
    return path


def _calibration(value):
    _exact(value, ("schema", "camera", "static_stable", "intrinsics_versions", "extrinsics_versions"))
    if value["schema"] != "official-mcap-camera-v1" or value["camera"] != "fl" or value["static_stable"] is not True:
        raise ValueError("stable fl calibration required")
    payloads = []
    for name in ("intrinsics_versions", "extrinsics_versions"):
        versions = value[name]
        if not isinstance(versions, list) or len(versions) != 1:
            raise ValueError("exactly one calibration version required")
        version = versions[0]
        _exact(version, ("sha256", "value", "first_stamp_ns", "last_stamp_ns", "message_count"))
        if hashlib.sha256(canonical(version["value"])).hexdigest() != _sha(version["sha256"]):
            raise ValueError("calibration payload hash mismatch")
        if _timestamp(version["first_stamp_ns"]) > _timestamp(version["last_stamp_ns"]):
            raise ValueError("calibration timestamp ordering")
        _integer(version["message_count"], "calibration count", 1)
        payloads.append(version["value"])
    intrinsic, extrinsic = payloads
    _exact(intrinsic, ("K", "D", "width", "height", "distortion_model", "camera_frame"))
    _exact(extrinsic, ("T_BC", "body_frame", "camera_frame"))
    if (intrinsic["camera_frame"] != "constructor0/camera_fl" or extrinsic["camera_frame"] != intrinsic["camera_frame"]
            or extrinsic["body_frame"] != "constructor0/base_link"):
        raise ValueError("calibration frame mismatch")
    _integer(intrinsic["width"], "width", 1)
    _integer(intrinsic["height"], "height", 1)
    k, d, transform = intrinsic["K"], intrinsic["D"], extrinsic["T_BC"]
    if (not isinstance(k, list) or len(k) != 3 or any(not isinstance(row, list) or len(row) != 3 for row in k)
            or not isinstance(d, list) or len(d) not in (0, 4, 5, 8, 12, 14)
            or not isinstance(transform, list) or len(transform) != 4
            or any(not isinstance(row, list) or len(row) != 4 for row in transform)):
        raise ValueError("calibration shape mismatch")
    for number in [x for row in k + transform for x in row] + d:
        _finite(number)
    if (k[0][0] <= 0 or k[1][1] <= 0 or k[2] != [0, 0, 1] or transform[3] != [0, 0, 0, 1]
            or intrinsic["distortion_model"] not in ("plumb_bob", "rational_polynomial", "")):
        raise ValueError("invalid calibration")
    rotation = [row[:3] for row in transform[:3]]
    for a in range(3):
        for b in range(3):
            if abs(sum(rotation[a][i] * rotation[b][i] for i in range(3)) - (a == b)) > 1e-5:
                raise ValueError("nonrigid calibration rotation")
    r = rotation
    determinant = (r[0][0] * (r[1][1]*r[2][2]-r[1][2]*r[2][1])
                   - r[0][1] * (r[1][0]*r[2][2]-r[1][2]*r[2][0])
                   + r[0][2] * (r[1][0]*r[2][1]-r[1][1]*r[2][0]))
    if abs(determinant - 1) > 1e-5:
        raise ValueError("improper calibration rotation")


def _intervals(split, baseline, baseline_sha, policy=POLICY_V1):
    spec = POLICIES[policy]
    _exact(split, ("schema", "timestamp_unit", "membership", "source_recording_sha256", "baseline_split_sha256",
                   "baseline_split_path", "baseline_reporting_intervals", "fit", "tune", "audit", "brackets",
                   "exact_sample", "audit_baseline_scores_already_exposed", "audit_description", "audit_reuse",
                   "hidden_organizer_recording_available", "initial_camera_id", "scalar_motion_variant_enabled"))
    if (split["schema"] != "schubki-map-membership-v1" or split["timestamp_unit"] != "integer_nanoseconds"
            or split["membership"] != "start_inclusive_end_exclusive" or split["initial_camera_id"] != "fl"
            or split["baseline_split_sha256"] != baseline_sha):
        raise ValueError("split identity or convention mismatch")
    _sha(split["source_recording_sha256"])
    previous_end = None
    for name, laps in (("fit", [1, 2, 3]), ("tune", [4, 5]), ("audit", [6, 7])):
        section = split[name]
        _exact(section, ("lap_ids", "start_ns", "end_ns"))
        start, end = _timestamp(section["start_ns"]), _timestamp(section["end_ns"])
        if (not isinstance(section["lap_ids"], list) or any(type(item) is not int for item in section["lap_ids"])
                or section["lap_ids"] != laps or start >= end or (previous_end is not None and start < previous_end)):
            raise ValueError("split lap membership or ordering mismatch")
        previous_end = end
    _exact(baseline, ("schema", "status", "ground_truth_path", "ground_truth_sha256", "criteria", "gate",
                      "definition_reason", "whole_laps", "development_laps", "reserved_laps", "excluded_prefix_end_ns",
                      "excluded_suffix_start_ns", "estimator_initializations", "lap_reseeding_allowed", "boundary_convention"))
    if baseline["schema"] != "constructor-whole-lap-split-v1" or baseline["status"] != "defined_before_evaluation":
        raise ValueError("baseline split schema mismatch")
    _sha(baseline["ground_truth_sha256"])
    laps = baseline["whole_laps"]
    if not isinstance(laps, list) or len(laps) != 7:
        raise ValueError("seven whole laps required")
    intervals = []
    previous_end = None
    for lap_id, lap in enumerate(laps, 1):
        _exact(lap, ("name", "start_ns", "end_ns", "start_gt_row", "end_gt_row", "duration_s", "travel_m", "extent_m"))
        start, end = _timestamp(lap["start_ns"]), _timestamp(lap["end_ns"])
        if lap["name"] != f"lap-{lap_id:02d}" or start >= end or (previous_end is not None and start != previous_end):
            raise ValueError("baseline lap interval mismatch")
        previous_end = end
        if lap_id in spec["laps"]:
            intervals.append({"lap_id": lap_id, "start_ns": start, "end_ns": end,
                              "role": "train" if lap_id in spec["train"] else spec["validation_role"]})
    for section, left, right in (("fit", 0, 2), ("tune", 3, 4), ("audit", 5, 6)):
        if (split[section]["start_ns"], split[section]["end_ns"]) != (laps[left]["start_ns"], laps[right]["end_ns"]):
            raise ValueError("aggregate split disagrees with whole laps")
    return intervals


def _lap(timestamp, intervals):
    return next((lap for lap in intervals if lap["start_ns"] <= timestamp < lap["end_ns"]), None)


def _plan(inputs, limits, budget, policy=POLICY_V1):
    spec = POLICIES[policy]
    snapshots = {}
    identities = {}
    for name, cap in (("index", limits.max_index_bytes), ("calibration", limits.max_metadata_bytes),
                      ("split", limits.max_metadata_bytes), ("baseline_split", limits.max_metadata_bytes)):
        raw = _snapshot(getattr(inputs, name), getattr(inputs, name + "_sha256"), cap, budget)
        snapshots[name] = raw
        identities[name] = {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
    _calibration(_json(snapshots["calibration"]))
    baseline = _json(snapshots["baseline_split"])
    intervals = _intervals(_json(snapshots["split"]), baseline, inputs.baseline_split_sha256, policy)
    observations, previous, hashes, encoded_bytes = [], None, {}, 0
    per_lap = {str(i): {"indexed": 0, "declared_bytes": 0} for i in spec["laps"]}
    outside, rows = 0, 0
    for rows, raw in enumerate(snapshots["index"].splitlines(), 1):
        budget.check()
        if rows > limits.max_index_rows:
            raise ValueError("index row cap exceeded")
        row = _json(raw)
        _exact(row, ("timestamp_ns", "path", "sha256", "bytes"))
        stamp = _timestamp(row["timestamp_ns"])
        if previous is not None and stamp <= previous:
            raise ValueError("duplicate or unordered index timestamps")
        previous = stamp
        digest, size = _sha(row["sha256"]), _integer(row["bytes"], "image bytes", 1)
        _local_path(inputs.cache_root, row["path"])
        lap = _lap(stamp, intervals)
        if lap is None:
            outside += 1
            continue
        if size > limits.max_image_bytes:
            raise ValueError("per-image declared byte cap exceeded")
        prior_role = hashes.setdefault(digest, lap["role"])
        if prior_role != lap["role"]:
            raise ValueError("duplicate image hash across roles")
        encoded_bytes += size
        if len(observations) >= limits.max_images or encoded_bytes > limits.max_encoded_bytes:
            raise ValueError("fit image or encoded byte cap exceeded")
        per_lap[str(lap["lap_id"])]["indexed"] += 1
        per_lap[str(lap["lap_id"])]["declared_bytes"] += size
        observations.append({"frame_id": f"fl:{stamp}", "camera_id": "fl", "exposure_timestamp_ns": stamp,
                             "image_path": row["path"], "image_sha256": digest, "image_bytes": size,
                             "calibration_sha256": inputs.calibration_sha256})
    if not observations or any(not counts["indexed"] for counts in per_lap.values()):
        raise ValueError("all three fit laps require observations")
    report = {"schema": spec["schema"], "stage": "plan", "complete": False, "images_verified": False,
              "inputs": identities, "intervals": intervals, "config": asdict(limits), "per_lap": per_lap,
              "index_rows": rows, "outside_fit_index_rows": outside, "fit_images": len(observations),
              "declared_encoded_bytes": encoded_bytes}
    if policy == POLICY_FIVE:
        indexed = {lap: counts["indexed"] for lap, counts in per_lap.items()}
        if indexed != FIVE_INDEXED_COUNTS or len(observations) != FIVE_TOTAL or encoded_bytes != FIVE_DECLARED_ENCODED_BYTES:
            raise ValueError("indexed five-lap membership differs from the pinned 10569-row policy counts/bytes")
        report["policy"] = _policy_record()
    return report, observations, baseline["ground_truth_sha256"]


def _policy_record():
    return {"name": POLICY_FIVE, "train_laps": [1, 2, 3, 4, 5], "validation_laps": [6, 7],
            "expected_indexed_counts": dict(FIVE_INDEXED_COUNTS), "expected_total": FIVE_TOTAL,
            "expected_train": FIVE_TRAIN, "expected_validation": FIVE_VALIDATION,
            "expected_declared_encoded_bytes": FIVE_DECLARED_ENCODED_BYTES,
            "split_role_override": "experiment-only override; historical global team split verified unchanged as source",
            "validation_exposure": "laps 6-7 baseline scores were previously exposed; reserved same-session "
                                   "validation, not pristine hidden data",
            "validation_use": "never gradients, mining, augmentation statistics or fitted preprocessing"}


def plan_dataset(inputs: Inputs, limits: Limits | None = None, *, policy: str = POLICY_V1):
    """Inventory pinned index/calibration/splits, without opening GT or images."""
    _, limits = _policy(policy, limits)
    report, _, _ = _plan(inputs, limits, _Budget(limits), policy)
    return report


def _gt_samples(raw, intervals, limits, budget):
    reader = csv.reader(io.StringIO(raw.decode("utf-8"), newline=""), strict=True)
    header = next(reader, None)
    if header not in (["timestamp_ns", "x", "y", "yaw"], ["timestamp_ns", "x", "y", "z", "yaw"]):
        raise ValueError("exact planar GT CSV header required")
    samples = {lap["lap_id"]: {} for lap in intervals}
    counts = Counter()
    for source_row, row in enumerate(reader, 2):
        budget.check()
        counts["rows"] += 1
        if counts["rows"] > limits.max_gt_rows:
            raise ValueError("GT row cap exceeded")
        if not row or _INT.fullmatch(row[0]) is None:
            raise ValueError("GT exact timestamp required")
        stamp = _timestamp(int(row[0]))
        lap = _lap(stamp, intervals)
        if lap is None:
            counts["outside_fit_pose_values_unread"] += 1
            continue
        # Only fit rows reach any pose-column access or float conversion.
        pose = None
        try:
            if len(row) != len(header):
                raise ValueError("GT column count")
            values = [float(value) for value in row[1:]]
            if not all(math.isfinite(value) for value in values):
                raise ValueError("nonfinite GT")
            x, y, yaw = values[0], values[1], values[-1]
            if not -math.pi <= yaw <= math.pi:
                raise ValueError("GT yaw outside radians range")
            pose = (x, y, yaw)
        except (ValueError, OverflowError):
            counts["invalid_pose_rows"] += 1
        group = samples[lap["lap_id"]]
        previous = group.get(stamp)
        if previous is None:
            group[stamp] = (source_row, pose)
        elif pose is None or previous[1] is None or pose != previous[1]:
            group[stamp] = (previous[0], None)
            counts["conflicting_or_invalid_duplicate_rows"] += 1
        else:
            counts["identical_duplicate_rows"] += 1
    return samples, dict(counts)


def _associate(observations, intervals, samples):
    ordered = {lap: sorted(group) for lap, group in samples.items()}
    laps = [lap["lap_id"] for lap in intervals]
    accepted, labels, rejections = [], [], {str(i): Counter() for i in laps}
    offsets = {str(i): [] for i in laps}
    for observation in observations:
        exposure = observation["exposure_timestamp_ns"]
        lap = _lap(exposure, intervals)
        lap_id = lap["lap_id"]
        stamps = ordered[lap_id]
        insertion = bisect_left(stamps, exposure)
        candidates = stamps[max(0, insertion-1):insertion+1]
        if not candidates:
            rejections[str(lap_id)]["no_same_lap_source"] += 1
            continue
        source = min(candidates, key=lambda stamp: (abs(stamp-exposure), stamp))
        source_row, pose = samples[lap_id][source]
        offset = source-exposure
        # Invalid/conflicting nearest stamps are barriers. Never seek another row.
        reason = "nearest_source_barrier" if pose is None else "nearest_offset_exceeds_50ms" if abs(offset) > MAX_OFFSET_NS else None
        if reason:
            rejections[str(lap_id)][reason] += 1
            continue
        accepted.append(observation)
        labels.append({"frame_id": observation["frame_id"], "image_sha256": observation["image_sha256"],
                       "lap_id": lap_id, "role": lap["role"], "body_x_m": pose[0], "body_y_m": pose[1],
                       "body_yaw_rad": pose[2], "association": {"method": "nearest_body_xy_yaw", "source_row": source_row,
                       "source_timestamp_ns": source, "offset_ns": offset, "maximum_offset_ns": MAX_OFFSET_NS}})
        offsets[str(lap_id)].append(abs(offset))
    return accepted, labels, rejections, offsets


def _verify_image(path, observation, budget):
    expected_bytes = observation["image_bytes"]
    budget.check()
    if path.stat().st_size != expected_bytes:
        raise ValueError("image byte size mismatch")
    digest, total = hashlib.sha256(), 0
    with path.open("rb") as stream:
        while True:
            budget.check()
            block = stream.read(min(64*1024, expected_bytes-total+1))
            if not block:
                break
            total += len(block)
            if total > expected_bytes:
                raise ValueError("image changed beyond declared size")
            digest.update(block)
    if total != expected_bytes or digest.hexdigest() != observation["image_sha256"]:
        raise ValueError("image hash or byte size mismatch")


def prepare_dataset(inputs: Inputs, *, ground_truth, ground_truth_sha256, output_dir,
                    source_commit, limits: Limits | None = None, declared_fit_demo: bool = False,
                    policy: str = POLICY_V1):
    """Explicit preparation. dataset.json is written last, only on full success.

    Completeness covers every indexed fit exposure being accepted or explicitly
    rejected. Unsupported associations remain counted rejections, never labels.
    Each fit lap must retain at least one valid example. No image is decoded.
    declared_fit_demo records caller-declared history only, never image-derived
    proof. Generic/fabricated calls do not inherit the actual recording's count.
    """
    if not isinstance(source_commit, str) or _COMMIT.fullmatch(source_commit) is None:
        raise ValueError("full source commit required")
    if type(declared_fit_demo) is not bool:
        raise ValueError("explicit boolean fit-demo declaration required")
    spec, limits = _policy(policy, limits)
    budget = _Budget(limits)
    report, observations, baseline_gt_sha = _plan(inputs, limits, budget, policy)
    if _sha(ground_truth_sha256) != baseline_gt_sha:
        raise ValueError("GT expected hash disagrees with pinned baseline split")
    raw = _snapshot(Path(ground_truth), ground_truth_sha256, limits.max_gt_bytes, budget)
    samples, gt_counts = _gt_samples(raw, report["intervals"], limits, budget)
    observations, labels, rejections, offsets = _associate(observations, report["intervals"], samples)
    require_complete_association(policy, report["intervals"], report["per_lap"], labels, rejections, offsets, gt_counts)
    output = Path(output_dir).resolve()
    if output.is_relative_to(inputs.cache_root.resolve()) or inputs.cache_root.resolve().is_relative_to(output):
        raise ValueError("output must be separate from approved cache")
    output.mkdir(parents=False, exist_ok=False)
    written_bytes, files = 0, {}
    for name, records in (("observations", observations), ("labels", labels)):
        digest, count = hashlib.sha256(), 0
        path = output / (name + ".jsonl")
        with path.open("xb") as stream:
            for record in records:
                budget.check()
                if name == "observations":
                    _verify_image(_local_path(inputs.cache_root, record["image_path"]), record, budget)
                data = canonical(record) + b"\n"
                written_bytes += len(data)
                if written_bytes > limits.max_output_bytes:
                    raise ValueError("output byte cap exceeded")
                stream.write(data)
                digest.update(data)
                count += len(data)
        files[name] = {"path": path.name, "sha256": digest.hexdigest(), "bytes": count, "rows": len(records)}
    report["inputs"]["ground_truth"] = {"sha256": ground_truth_sha256, "bytes": len(raw)}
    for lap_id, counts in report["per_lap"].items():
        counts.update({"accepted": len(offsets[lap_id]), "rejected": sum(rejections[lap_id].values()),
                       "verified_bytes": sum(row["image_bytes"] for row, label in zip(observations, labels) if str(label["lap_id"]) == lap_id),
                       "maximum_absolute_offset_ns": max(offsets[lap_id])})
    report.update({"stage": "prepare", "complete": True, "images_verified": True, **files,
                   "fit_images": len(observations), "verified_encoded_bytes": sum(row["image_bytes"] for row in observations),
                   "rejections": {key: dict(value) for key, value in rejections.items()}, "ground_truth_counts": gt_counts,
                   "source": {"commit": source_commit, "module_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
                   "label_convention": {"body_frame": "constructor0/base_link", "map_frame": "map", "position_units": "metres",
                       "yaw_units": "radians", "method": "nearest_body_xy_yaw", "maximum_offset_ns": MAX_OFFSET_NS,
                       "tie_break": "earlier_source_timestamp", "source_row": "one-based CSV record including header",
                       "interval_membership": "start_inclusive_end_exclusive"},
                   "previously_exposed_fit_data": {"known_recording_demo_frames": 20 if declared_fit_demo else 0,
                       "lap_id": 1, "role": "train", "status":
                       "declared known20-frame lap1 fit-demo history; individual membership not verified here" if declared_fit_demo else
                       "fabricated/undeclared exposure history; no actual recording membership verified"}})
    if policy == POLICY_FIVE:
        roles = Counter(label["role"] for label in labels)
        if roles != Counter({"train": FIVE_TRAIN, "validation": FIVE_VALIDATION}):
            raise ValueError("five-lap role counts differ from 8269 train / 2300 validation")
        report["role_counts"] = {"train": FIVE_TRAIN, "validation": FIVE_VALIDATION}
        report["lap_membership"] = {str(lap["lap_id"]): lap["role"] for lap in report["intervals"]}
    data = canonical(report) + b"\n"
    budget.check()
    if written_bytes + len(data) > limits.max_output_bytes:
        raise ValueError("output manifest byte cap exceeded")
    # A failed manifest write cannot leave a successfully parseable marker.
    temporary = output / ".dataset.partial"
    with temporary.open("xb") as stream:
        stream.write(data)
    budget.check()
    temporary.rename(output / "dataset.json")
    return report

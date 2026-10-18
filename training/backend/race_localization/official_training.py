"""Fit-only full-pose training reference exporter (MAP-01).

Training-only module. It reads permitted full body poses from the organizer
truth convention (/tf, header.frame_id ``map``, child_frame_id
``constructor0/base_link``, header.stamp, translation xyz, rotation xyzw) and
original camera exposure stamps, then emits S0 ``ReferenceFrame`` records for
exposures strictly inside the declared half-open fit interval.

Separation rules enforced here:

* Camera exposures and pose samples come from separate adapters.
* The exporter partitions pose samples by timestamp before any value is
  validated or used; out-of-fit samples are only counted. The /tf adapter,
  given a window, skips out-of-window transforms before touching their
  translation or rotation. Mutating out-of-fit values cannot change the export.
* Both interpolation brackets must be usable samples inside the same fit
  interval. A tuning or audit pose is never borrowed for a training exposure.
* Nothing here produces an inference input. Query exposures and their labels
  never enter the output.

Times are exact integer nanoseconds and are never converted through float
seconds. The interpolation fraction uses exact integer differences.
"""
from __future__ import annotations

import argparse
from bisect import bisect_left
from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import platform
import re
import sys
import time
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from . import map_contracts as contracts
from .map_contracts import (
    CameraCalibration, FitInterval, ReferenceFrame, camera_pose_from_body,
    pose_from_xyzw, validate_timestamp_ns,
)


EXPORT_SCHEMA = "schubki-fit-reference-export-v1"
SPLIT_SCHEMA = "schubki-map-membership-v1"
POSE_TOPIC = "/tf"
WORLD_FRAME = "map"
BODY_FRAME = "constructor0/base_link"
DEFAULT_CAMERA_ID = "fl"
# Conservative engineering setting for training interpolation. It is not the
# scorer's truth-gap rule and is declared in every export.
DEFAULT_MAX_INTERPOLATION_GAP_NS = 50_000_000
INTERPOLATION_METHOD = "linear_translation_slerp_rotation_short_arc"
GAP_DEFINITION = "right_bracket_ns_minus_left_bracket_ns"

EXPOSURE_REJECTIONS = (
    "exposure_before_fit",
    "exposure_at_or_after_fit_end",
    "exposure_duplicate_conflicting",
    "no_left_bracket_in_fit",
    "no_right_bracket_in_fit",
    "bracket_spans_invalid_pose",
    "bracket_gap_too_large",
)
IN_FIT_REJECTIONS = EXPOSURE_REJECTIONS[2:]
COUNT_KEYS = (
    "pose_samples_total",
    "pose_outside_fit_unread",
    "pose_in_fit",
    "pose_out_of_order",
    "pose_invalid_value",
    "pose_duplicate_identical",
    "pose_duplicate_conflicting_stamps",
    "pose_barrier_stamps",
    "pose_usable",
    "exposures_total",
    "exposures_out_of_order",
    "exposure_duplicate_identical",
    "exposures_in_fit",
    "references_accepted",
    "references_exact_sample",
    "references_interpolated",
    *EXPOSURE_REJECTIONS,
)

PROVENANCE_KEYS = (
    "image_source_sha256",
    "pose_source_sha256",
    "calibration_file_sha256",
    "calibration_identity",
    "split_sha256",
    "exporter_sha256",
    "map_contracts_sha256",
    "code_commit",
    "config_identity",
)
# Exact per-reference record used by report() and references_digest.
REFERENCE_RECORD_FIELDS = (
    "reference_id", "camera_id", "timestamp_ns", "image_sha256", "image_source_id",
    "source_sha256", "pose_source_sha256", "split_sha256", "calibration_id", "fit_interval",
    "translation_m", "quaternion_xyzw", "T_WB", "T_WC", "pose_bracket_ids",
    "pose_bracket_timestamps_ns", "interpolation_fraction", "bracket_gap_ns",
)
# Report fields bound by export_digest; run metadata added by the CLI is outside it.
EXPORT_DIGEST_FIELDS = (
    "schema", "config", "config_identity", "fit_interval", "calibration", "provenance", "counts",
    "in_fit_rejections", "fit_pose_digest", "references_digest",
)
DIGEST_DOMAINS = {
    "encoding": "sha256 of UTF-8 JSON with sorted keys, compact separators and no NaN or infinity",
    "references_digest": {"value": "ordered list of per-reference records, increasing timestamp_ns",
                          "fields": list(REFERENCE_RECORD_FIELDS)},
    "export_digest": {"value": "object of the listed report fields", "fields": list(EXPORT_DIGEST_FIELDS)},
    "fit_pose_digest": {"value": "usable in-fit pose samples in timestamp order",
                        "fields": ["timestamp_ns", "sample_id", "translation_m", "quaternion_xyzw"]},
}

_SHA = re.compile(r"[0-9a-f]{64}")
_COMMIT = re.compile(r"[0-9a-f]{40}")


def _sha(value: str, name: str) -> str:
    if not isinstance(value, str) or _SHA.fullmatch(value.lower()) is None:
        raise ValueError(f"{name} must be a SHA256 hex digest")
    return value.lower()


def _text(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _frozen(values: Any, name: str, shape: tuple[int, ...]) -> np.ndarray:
    """Finite float64 array with the accepted S0 immutability behavior.

    map_contracts._float_array copies bytes into a read-only buffer, so
    setflags(write=True) cannot re-enable writes on the array or any view, and
    its array type rejects shape, dtype, strides and data assignment and resize.
    """
    return contracts._float_array(values, name, shape)


def _proxy(mapping: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType(dict(mapping))


def module_sha256(module: Any) -> str:
    """SHA256 of the exact source bytes of an imported module."""
    return _sha256_file(Path(module.__file__))


@lru_cache(maxsize=1)
def code_identities() -> tuple[str, str]:
    """(exporter_sha256, map_contracts_sha256) of the loaded code, hashed once."""
    return module_sha256(sys.modules[__name__]), module_sha256(contracts)


def ros_stamp_ns(stamp: Any) -> int:
    """Exact nanoseconds from a ROS builtin_interfaces/Time (sec, nanosec)."""
    sec, nanosec = stamp.sec, stamp.nanosec
    for value in (sec, nanosec):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise ValueError("ROS stamp fields must be integers")
    sec, nanosec = int(sec), int(nanosec)
    if sec < 0 or not 0 <= nanosec < 1_000_000_000:
        raise ValueError("Invalid nonnegative ROS stamp")
    return validate_timestamp_ns(sec * 1_000_000_000 + nanosec)


# ---------------------------------------------------------------------------
# Split membership


@dataclass(frozen=True, slots=True)
class FitSplit:
    """Validated fit/tune/audit declaration from team/contracts/split.json."""

    fit: FitInterval
    tune_start_ns: int
    tune_end_ns: int
    audit_start_ns: int
    audit_end_ns: int
    split_sha256: str
    source_recording_sha256: str
    initial_camera_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.fit, FitInterval) or tuple(self.fit.lap_ids) != contracts.FIT_LAPS:
            raise ValueError("Fit split must declare laps 1-3 as one FitInterval")
        stamps = (self.fit.start_ns, self.fit.end_ns, self.tune_start_ns, self.tune_end_ns,
                  self.audit_start_ns, self.audit_end_ns)
        stamps = tuple(validate_timestamp_ns(value) for value in stamps)
        fit_start, fit_end, tune_start, tune_end, audit_start, audit_end = stamps
        if not (fit_start < fit_end <= tune_start < tune_end <= audit_start < audit_end):
            raise ValueError("Fit must end before tuning, and tuning before audit, without overlap")
        object.__setattr__(self, "tune_start_ns", tune_start)
        object.__setattr__(self, "tune_end_ns", tune_end)
        object.__setattr__(self, "audit_start_ns", audit_start)
        object.__setattr__(self, "audit_end_ns", audit_end)
        object.__setattr__(self, "split_sha256", _sha(self.split_sha256, "split SHA256"))
        object.__setattr__(self, "source_recording_sha256",
                           _sha(self.source_recording_sha256, "source recording SHA256"))
        _text(self.initial_camera_id, "initial camera ID")


def load_fit_split(path: Any) -> FitSplit:
    """Read the fixed membership file; split_sha256 hashes its exact bytes."""
    raw = Path(path).read_bytes()
    data = json.loads(raw)
    if (not isinstance(data, dict) or data.get("schema") != SPLIT_SCHEMA
            or data.get("timestamp_unit") != "integer_nanoseconds"
            or data.get("membership") != "start_inclusive_end_exclusive"):
        raise ValueError("Unsupported split schema, timestamp unit or membership convention")
    sections = {}
    for name, laps in (("fit", [1, 2, 3]), ("tune", [4, 5]), ("audit", [6, 7])):
        section = data.get(name)
        if not isinstance(section, dict) or section.get("lap_ids") != laps:
            raise ValueError(f"Split {name} section must declare laps {laps}")
        sections[name] = (validate_timestamp_ns(section.get("start_ns")),
                          validate_timestamp_ns(section.get("end_ns")))
    fit = FitInterval(sections["fit"][0], sections["fit"][1], contracts.FIT_LAPS)
    return FitSplit(fit, *sections["tune"], *sections["audit"], hashlib.sha256(raw).hexdigest(),
                    data.get("source_recording_sha256"), data.get("initial_camera_id"))


# ---------------------------------------------------------------------------
# Source records and adapters (kept separate)


@dataclass(frozen=True, slots=True)
class PoseSample:
    """One training-truth body pose sample T_WB at its source header stamp.

    Values are stored as received and validated by the exporter only when the
    sample lies inside the fit interval.
    """

    sample_id: str
    timestamp_ns: int
    translation_m: tuple
    quaternion_xyzw: tuple

    def __post_init__(self) -> None:
        _text(self.sample_id, "pose sample ID")
        object.__setattr__(self, "timestamp_ns", validate_timestamp_ns(self.timestamp_ns))
        for name in ("translation_m", "quaternion_xyzw"):
            value = getattr(self, name)
            try:
                value = tuple(value)
            except TypeError:
                value = (value,)
            object.__setattr__(self, name, value)


@dataclass(frozen=True, slots=True)
class CameraExposure:
    """One original camera exposure. Carries no pose, state or label."""

    camera_id: str
    timestamp_ns: int
    image_sha256: str
    source_id: str

    def __post_init__(self) -> None:
        _text(self.camera_id, "camera ID")
        _text(self.source_id, "image source ID")
        object.__setattr__(self, "timestamp_ns", validate_timestamp_ns(self.timestamp_ns))
        object.__setattr__(self, "image_sha256", _sha(self.image_sha256, "image SHA256"))


@dataclass(frozen=True, slots=True)
class PoseStream:
    samples: tuple[PoseSample, ...]
    counts: Mapping[str, int]
    frame_pairs: Mapping[str, int]


@dataclass(frozen=True, slots=True)
class ExposureStream:
    exposures: tuple[CameraExposure, ...]
    counts: Mapping[str, int]
    source_sha256: str
    frames_index_sha256: str
    calibration_file_sha256: str


def pose_samples_from_tf(messages: Iterable[tuple[int, Any]], *,
                         window: FitInterval | None = None) -> PoseStream:
    """Select map -> constructor0/base_link transforms from decoded /tf messages.

    ``messages`` yields (mcap_log_time_ns, decoded tf2_msgs/TFMessage). Time is
    the selected transform's header.stamp, never MCAP log time. Frame IDs are
    compared exactly, as in the organizer exporter. With a window, samples
    outside it are counted and their pose values are never read.
    """
    counts = dict.fromkeys(("messages", "transforms", "selected", "other_frame_pair",
                            "invalid_stamp", "outside_window_unread"), 0)
    pairs: dict[str, int] = {}
    samples = []
    for ordinal, (log_time_ns, message) in enumerate(messages):
        counts["messages"] += 1
        for index, item in enumerate(message.transforms):
            counts["transforms"] += 1
            parent, child = str(item.header.frame_id), str(item.child_frame_id)
            key = f"{parent}->{child}"
            pairs[key] = pairs.get(key, 0) + 1
            if parent != WORLD_FRAME or child != BODY_FRAME:
                counts["other_frame_pair"] += 1
                continue
            try:
                stamp = ros_stamp_ns(item.header.stamp)
            except ValueError:
                counts["invalid_stamp"] += 1
                continue
            if window is not None and not window.contains(stamp):
                counts["outside_window_unread"] += 1
                continue
            translation, rotation = item.transform.translation, item.transform.rotation
            samples.append(PoseSample(
                f"{POSE_TOPIC}@{int(log_time_ns)}:{ordinal}.{index}", stamp,
                (translation.x, translation.y, translation.z),
                (rotation.x, rotation.y, rotation.z, rotation.w)))
            counts["selected"] += 1
    return PoseStream(tuple(samples), _proxy(counts), _proxy(dict(sorted(pairs.items()))))


def read_tf_pose_samples(mcap_path: Any, *, expected_sha256: str,
                         window: FitInterval | None) -> PoseStream:
    """Hash-check the recording, then decode only /tf through the whitelisted reader."""
    from . import official_io

    path = Path(mcap_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    if _sha256_file(path) != _sha(expected_sha256, "expected recording SHA256"):
        raise ValueError("Pose source recording hash mismatch")
    messages = ((log_ns, message) for topic, log_ns, message
                in official_io._iter_allowed_messages(path, (POSE_TOPIC,)) if topic == POSE_TOPIC)
    return pose_samples_from_tf(messages, window=window)


def _contained(root: Path, raw: str) -> Path:
    path = (root / raw).resolve()
    if not path.is_relative_to(root) or path == root:
        raise ValueError("Cached artifact path leaves the recording cache")
    return path


def camera_exposures_from_cache(manifest: Mapping[str, Any], camera_id: str, *,
                                window: FitInterval | None, verify_images: bool = True) -> ExposureStream:
    """Exposure stamps and image identities from an official_io camera cache.

    Reads the hash-checked frames index directly and applies the half-open
    window ``start <= t < end`` independently of official_io.iter_frames,
    whose window end is closed. Image bytes are verified only inside the window.
    """
    from . import official_io

    if manifest.get("schema") != official_io.SCHEMA or manifest.get("camera") != camera_id:
        raise ValueError("Cache schema or camera does not match the requested camera")
    root = Path(manifest["cache_dir"]).resolve()
    index = _contained(root, manifest["frames_index_path"])
    index_sha = _sha256_file(index)
    if index_sha != _sha(manifest["frames_index_sha256"], "frames index SHA256"):
        raise ValueError("Cached frames index hash mismatch")
    counts = dict.fromkeys(("index_rows", "outside_window", "images_verified"), 0)
    exposures, previous = [], None
    with index.open("r", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            stamp = validate_timestamp_ns(row["timestamp_ns"])
            if previous is not None and stamp <= previous:
                raise ValueError("Frame index timestamps must be unique and increasing")
            previous = stamp
            counts["index_rows"] += 1
            if window is not None and not window.contains(stamp):
                counts["outside_window"] += 1
                continue
            digest = _sha(row["sha256"], "image SHA256")
            if verify_images:
                image = _contained(root, row["path"])
                if image.stat().st_size != row["bytes"] or _sha256_file(image) != digest:
                    raise ValueError("Cached image integrity mismatch")
                counts["images_verified"] += 1
            exposures.append(CameraExposure(camera_id, stamp, digest, str(row["path"])))
    return ExposureStream(tuple(exposures), _proxy(counts),
                          _sha(manifest["source"]["sha256"], "source SHA256"), index_sha,
                          _sha(manifest["calibration_sha256"], "calibration file SHA256"))


# ---------------------------------------------------------------------------
# Export


@dataclass(frozen=True, slots=True)
class ExportConfig:
    camera_id: str = DEFAULT_CAMERA_ID
    max_interpolation_gap_ns: int = DEFAULT_MAX_INTERPOLATION_GAP_NS
    quaternion_norm_tolerance: float = 1e-3

    def __post_init__(self) -> None:
        _text(self.camera_id, "camera ID")
        if validate_timestamp_ns(self.max_interpolation_gap_ns) < 1:
            raise ValueError("Maximum interpolation gap must be a positive integer nanosecond count")
        tolerance = self.quaternion_norm_tolerance
        if (isinstance(tolerance, bool) or not isinstance(tolerance, (int, float))
                or not 0 < float(tolerance) <= 1e-3):
            raise ValueError("Quaternion norm tolerance must be in (0, 1e-3]")
        object.__setattr__(self, "quaternion_norm_tolerance", float(tolerance))

    def as_dict(self) -> dict[str, Any]:
        return {"camera_id": self.camera_id, "max_interpolation_gap_ns": self.max_interpolation_gap_ns,
                "quaternion_norm_tolerance": self.quaternion_norm_tolerance,
                "interpolation": INTERPOLATION_METHOD, "gap_definition": GAP_DEFINITION,
                "membership": "start_inclusive_end_exclusive", "pose_frames": f"{WORLD_FRAME}->{BODY_FRAME}",
                "pose_time": "header.stamp", "quaternion_order": "xyzw",
                "output_quaternion_sign": "w_nonnegative"}

    @property
    def identity(self) -> str:
        return contracts.canonical_identity(self.as_dict())


@dataclass(frozen=True, slots=True)
class ExportProvenance:
    image_source_sha256: str
    pose_source_sha256: str
    calibration_file_sha256: str
    code_commit: str | None = None

    def __post_init__(self) -> None:
        for name in ("image_source_sha256", "pose_source_sha256", "calibration_file_sha256"):
            object.__setattr__(self, name, _sha(getattr(self, name), name))
        if self.code_commit is not None:
            if not isinstance(self.code_commit, str) or _COMMIT.fullmatch(self.code_commit.lower()) is None:
                raise ValueError("code_commit must be an exact 40-character Git commit")
            object.__setattr__(self, "code_commit", self.code_commit.lower())


@dataclass(frozen=True, slots=True)
class TrainingReference:
    """An accepted fit reference: S0 ReferenceFrame plus unrounded pose terms.

    Construction freezes both arrays and requires them to reproduce the frame's
    T_WB exactly, with fraction and gap matching the bracket timestamps.
    """

    reference: ReferenceFrame
    image_source_id: str
    translation_m: np.ndarray
    quaternion_xyzw: np.ndarray
    interpolation_fraction: float
    bracket_gap_ns: int

    def __post_init__(self) -> None:
        frame = self.reference
        if not isinstance(frame, ReferenceFrame):
            raise ValueError("Training reference requires an S0 ReferenceFrame")
        _text(self.image_source_id, "image source ID")
        translation = _frozen(self.translation_m, "translation_m", (3,))
        quaternion = _frozen(self.quaternion_xyzw, "quaternion_xyzw", (4,))
        if abs(float(np.linalg.norm(quaternion)) - 1.0) > 1e-12 or quaternion[3] < 0:
            raise ValueError("quaternion_xyzw must be unit length with w >= 0")
        left_ns, right_ns = frame.pose_bracket_timestamps_ns
        gap = validate_timestamp_ns(self.bracket_gap_ns)
        fraction = self.interpolation_fraction
        if isinstance(fraction, (bool, np.bool_)) or not isinstance(fraction, (float, int)):
            raise ValueError("Interpolation fraction must be a real number")
        expected = 0.0 if gap == 0 else (frame.timestamp_ns - left_ns) / gap
        if gap != right_ns - left_ns or float(fraction) != expected:
            raise ValueError("Gap and fraction must match the reference bracket timestamps")
        if not np.array_equal(pose_from_xyzw(translation, quaternion), frame.T_WB):
            raise ValueError("Translation and quaternion must reproduce the reference T_WB exactly")
        object.__setattr__(self, "translation_m", translation)
        object.__setattr__(self, "quaternion_xyzw", quaternion)
        object.__setattr__(self, "interpolation_fraction", float(fraction))
        object.__setattr__(self, "bracket_gap_ns", gap)


@dataclass(frozen=True, slots=True)
class TrainingExport:
    """Immutable fit-only export; construction re-derives references_digest."""

    schema: str
    config: ExportConfig
    split: FitSplit
    calibration: CameraCalibration
    references: tuple[TrainingReference, ...]
    counts: Mapping[str, int]
    in_fit_rejections: tuple[tuple[int, str], ...]
    provenance: Mapping[str, Any]
    fit_pose_digest: str
    references_digest: str

    def __post_init__(self) -> None:
        if self.schema != EXPORT_SCHEMA:
            raise ValueError("Unsupported export schema")
        config, split, calibration = self.config, self.split, self.calibration
        if (not isinstance(config, ExportConfig) or not isinstance(split, FitSplit)
                or not isinstance(calibration, CameraCalibration)):
            raise ValueError("Export requires typed configuration, split and calibration")
        if calibration.camera_id != config.camera_id:
            raise ValueError("Calibration camera does not match the configured camera")
        provenance = _validated_provenance(self.provenance, config, split, calibration.identity)
        references = tuple(self.references)
        _validate_references(references, config, split, calibration, provenance)
        counts = dict(self.counts)
        if tuple(counts) != COUNT_KEYS or any(type(value) is not int or value < 0 for value in counts.values()):
            raise ValueError("Export counts must contain exactly the declared nonnegative integer keys")
        rejections = tuple((validate_timestamp_ns(stamp), reason) for stamp, reason in self.in_fit_rejections)
        if any(reason not in IN_FIT_REJECTIONS for _, reason in rejections):
            raise ValueError("Unknown in-fit rejection reason")
        if list(rejections) != sorted(rejections) or not all(split.fit.contains(stamp) for stamp, _ in rejections):
            raise ValueError("In-fit rejections must be sorted and inside the fit interval")
        if {stamp for stamp, _ in rejections} & {item.reference.timestamp_ns for item in references}:
            raise ValueError("An exposure cannot be both accepted and rejected")
        _validate_counts(counts, references, rejections)
        if _sha(self.references_digest, "references digest") != _digest_references(references):
            raise ValueError("references_digest does not match the references")
        object.__setattr__(self, "references", references)
        object.__setattr__(self, "counts", _proxy(counts))
        object.__setattr__(self, "in_fit_rejections", rejections)
        object.__setattr__(self, "provenance", _proxy(provenance))
        object.__setattr__(self, "fit_pose_digest", _sha(self.fit_pose_digest, "fit pose digest"))

    def reference_frames(self) -> tuple[ReferenceFrame, ...]:
        return tuple(item.reference for item in self.references)

    def _digest_body(self) -> dict[str, Any]:
        fit, calibration = self.split.fit, self.calibration
        body = {
            "schema": self.schema,
            "config": self.config.as_dict(),
            "config_identity": self.config.identity,
            "fit_interval": {"start_ns": fit.start_ns, "end_ns": fit.end_ns, "lap_ids": list(fit.lap_ids),
                             "membership": "start_inclusive_end_exclusive"},
            "calibration": {"identity": calibration.identity, "camera_id": calibration.camera_id,
                            "camera_frame": calibration.camera_frame, "body_frame": calibration.body_frame,
                            "T_BC": calibration.T_BC.tolist()},
            "provenance": dict(self.provenance),
            "counts": dict(self.counts),
            "in_fit_rejections": [list(item) for item in self.in_fit_rejections],
            "fit_pose_digest": self.fit_pose_digest,
            "references_digest": self.references_digest,
        }
        assert tuple(body) == EXPORT_DIGEST_FIELDS
        return body

    @property
    def export_digest(self) -> str:
        """Identity of every export field except run metadata; see DIGEST_DOMAINS."""
        return hashlib.sha256(_canonical_bytes(self._digest_body())).hexdigest()

    def report(self, max_references: int | None = None) -> dict[str, Any]:
        selected = self.references if max_references is None else self.references[:max_references]
        body = self._digest_body()
        return {**body, "export_digest": hashlib.sha256(_canonical_bytes(body)).hexdigest(),
                "digest_domains": DIGEST_DOMAINS, "reference_count": len(self.references),
                "references": [_reference_record(item) for item in selected]}


def _validated_provenance(provenance: Any, config: ExportConfig, split: FitSplit,
                          calibration_id: str) -> dict[str, Any]:
    if not isinstance(provenance, Mapping) or sorted(provenance) != sorted(PROVENANCE_KEYS):
        raise ValueError("Export provenance must contain exactly the declared keys")
    record = {key: provenance[key] for key in PROVENANCE_KEYS}
    for key in PROVENANCE_KEYS:
        if key != "code_commit":
            record[key] = _sha(record[key], key)
    commit = record["code_commit"]
    if commit is not None and (not isinstance(commit, str) or _COMMIT.fullmatch(commit) is None):
        raise ValueError("Provenance code_commit must be None or a lowercase 40-character Git commit")
    exporter_sha256, map_contracts_sha256 = code_identities()
    expected = {"image_source_sha256": split.source_recording_sha256,
                "pose_source_sha256": split.source_recording_sha256,
                "calibration_identity": calibration_id, "split_sha256": split.split_sha256,
                "exporter_sha256": exporter_sha256, "map_contracts_sha256": map_contracts_sha256,
                "config_identity": config.identity}
    mismatched = [key for key, value in expected.items() if record[key] != value]
    if mismatched:
        raise ValueError(f"Export provenance does not match its actual values: {', '.join(mismatched)}")
    return record


def _validate_references(references: tuple[Any, ...], config: ExportConfig, split: FitSplit,
                         calibration: CameraCalibration, provenance: Mapping[str, Any]) -> None:
    calibration_id, previous = calibration.identity, None
    for item in references:
        if not isinstance(item, TrainingReference):
            raise ValueError("Export references must be TrainingReference records")
        frame = item.reference
        if (frame.camera_id != config.camera_id or frame.reference_id != f"{config.camera_id}:{frame.timestamp_ns}"
                or frame.calibration_id != calibration_id or frame.fit_interval != split.fit
                or frame.split_sha256 != split.split_sha256
                or frame.source_sha256 != provenance["image_source_sha256"]
                or frame.pose_source_sha256 != provenance["pose_source_sha256"]):
            raise ValueError("Reference camera, ID, calibration, fit interval, split or source identity "
                             "does not match the export")
        if not np.array_equal(frame.T_WC, camera_pose_from_body(frame.T_WB, calibration.T_BC)):
            raise ValueError("Reference T_WC must equal T_WB @ T_BC of the export calibration")
        if item.bracket_gap_ns > config.max_interpolation_gap_ns:
            raise ValueError("Retained bracket gap exceeds the declared maximum interpolation gap")
        if previous is not None and frame.timestamp_ns <= previous:
            raise ValueError("References must have unique increasing exposure timestamps")
        previous = frame.timestamp_ns


def _validate_counts(counts: Mapping[str, int], references: tuple[TrainingReference, ...],
                     rejections: tuple[tuple[int, str], ...]) -> None:
    exact = sum(1 for item in references if item.bracket_gap_ns == 0)
    reasons = Counter(reason for _, reason in rejections)
    expected = {"references_accepted": len(references), "references_exact_sample": exact,
                "references_interpolated": len(references) - exact,
                **{reason: reasons.get(reason, 0) for reason in IN_FIT_REJECTIONS}}
    mismatched = [key for key, value in expected.items() if counts[key] != value]
    if mismatched:
        raise ValueError(f"Counts disagree with references or rejections: {', '.join(mismatched)}")
    relations = (
        counts["exposures_in_fit"] == len(references) + len(rejections) + counts["exposure_duplicate_identical"],
        counts["exposures_total"] == (counts["exposures_in_fit"] + counts["exposure_before_fit"]
                                      + counts["exposure_at_or_after_fit_end"]),
        counts["exposures_out_of_order"] <= counts["exposures_total"],
        counts["pose_samples_total"] == counts["pose_outside_fit_unread"] + counts["pose_in_fit"],
        counts["pose_usable"] + counts["pose_barrier_stamps"] + counts["pose_duplicate_identical"]
        <= counts["pose_in_fit"],
        counts["pose_duplicate_conflicting_stamps"] <= counts["pose_barrier_stamps"],
        counts["pose_invalid_value"] <= counts["pose_in_fit"],
        counts["pose_out_of_order"] <= counts["pose_in_fit"],
    )
    if not all(relations):
        raise ValueError("Exposure or pose count totals are inconsistent")


def _reference_record(item: TrainingReference) -> dict[str, Any]:
    frame = item.reference
    fit = frame.fit_interval
    record = {"reference_id": frame.reference_id, "camera_id": frame.camera_id, "timestamp_ns": frame.timestamp_ns,
              "image_sha256": frame.image_sha256, "image_source_id": item.image_source_id,
              "source_sha256": frame.source_sha256, "pose_source_sha256": frame.pose_source_sha256,
              "split_sha256": frame.split_sha256, "calibration_id": frame.calibration_id,
              "fit_interval": {"start_ns": fit.start_ns, "end_ns": fit.end_ns, "lap_ids": list(fit.lap_ids)},
              "translation_m": item.translation_m.tolist(), "quaternion_xyzw": item.quaternion_xyzw.tolist(),
              "T_WB": frame.T_WB.tolist(), "T_WC": frame.T_WC.tolist(),
              "pose_bracket_ids": list(frame.pose_bracket_ids),
              "pose_bracket_timestamps_ns": list(frame.pose_bracket_timestamps_ns),
              "interpolation_fraction": item.interpolation_fraction, "bracket_gap_ns": item.bracket_gap_ns}
    assert tuple(record) == REFERENCE_RECORD_FIELDS
    return record


def _real(value: Any) -> float | None:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _canonical_quaternion(q: np.ndarray) -> np.ndarray:
    q = q / np.linalg.norm(q)
    for component in (q[3], q[0], q[1], q[2]):
        if component != 0:
            return -q if component < 0 else q
    raise ValueError("Zero quaternion")


def _validated_pose(sample: PoseSample, tolerance: float) -> tuple[np.ndarray, np.ndarray] | None:
    if len(sample.translation_m) != 3 or len(sample.quaternion_xyzw) != 4:
        return None
    translation = [_real(value) for value in sample.translation_m]
    quaternion = [_real(value) for value in sample.quaternion_xyzw]
    if any(value is None for value in (*translation, *quaternion)):
        return None
    q = np.asarray(quaternion, dtype=np.float64)
    if abs(float(np.linalg.norm(q)) - 1.0) > tolerance:
        return None
    return np.asarray(translation, dtype=np.float64), _canonical_quaternion(q)


def slerp_xyzw(q0: np.ndarray, q1: np.ndarray, fraction: float) -> np.ndarray:
    """Short-arc spherical interpolation of unit xyzw quaternions."""
    q0 = np.asarray(q0, dtype=np.float64) / np.linalg.norm(q0)
    q1 = np.asarray(q1, dtype=np.float64) / np.linalg.norm(q1)
    dot = float(q0 @ q1)
    if dot < 0.0:
        q1, dot = -q1, -dot
    if dot > 1.0 - 1e-12:
        result = q0 + fraction * (q1 - q0)
    else:
        theta = math.acos(min(1.0, dot))
        scale = math.sin(theta)
        result = (math.sin((1.0 - fraction) * theta) / scale) * q0 + (math.sin(fraction * theta) / scale) * q1
    return _canonical_quaternion(result)


@dataclass(slots=True)
class _Usable:
    stamps: list[int] = field(default_factory=list)
    samples: list[tuple[str, np.ndarray, np.ndarray]] = field(default_factory=list)
    barriers: list[int] = field(default_factory=list)


def _usable_fit_poses(poses: Iterable[PoseSample], fit: FitInterval, config: ExportConfig,
                      counts: dict[str, int]) -> _Usable:
    by_stamp: dict[int, list[tuple[str, tuple[np.ndarray, np.ndarray] | None]]] = {}
    seen_ids: set[str] = set()
    latest = None
    for sample in poses:
        if not isinstance(sample, PoseSample):
            raise ValueError("Poses must be PoseSample records")
        counts["pose_samples_total"] += 1
        # Membership is decided from the timestamp alone, before any value is read.
        if not fit.contains(sample.timestamp_ns):
            counts["pose_outside_fit_unread"] += 1
            continue
        counts["pose_in_fit"] += 1
        if sample.sample_id in seen_ids:
            raise ValueError(f"Duplicate pose sample ID inside fit: {sample.sample_id}")
        seen_ids.add(sample.sample_id)
        if latest is not None and sample.timestamp_ns < latest:
            counts["pose_out_of_order"] += 1
        latest = sample.timestamp_ns if latest is None else max(latest, sample.timestamp_ns)
        pose = _validated_pose(sample, config.quaternion_norm_tolerance)
        if pose is None:
            counts["pose_invalid_value"] += 1
        by_stamp.setdefault(sample.timestamp_ns, []).append((sample.sample_id, pose))
    usable = _Usable()
    for stamp in sorted(by_stamp):
        entries = sorted(by_stamp[stamp], key=lambda entry: entry[0])
        poses_at = [pose for _, pose in entries]
        if any(pose is None for pose in poses_at):
            usable.barriers.append(stamp)
            if len(entries) > 1:
                counts["pose_duplicate_conflicting_stamps"] += 1
            continue
        first = poses_at[0]
        if any(not (np.array_equal(first[0], other[0]) and np.array_equal(first[1], other[1]))
               for other in poses_at[1:]):
            usable.barriers.append(stamp)
            counts["pose_duplicate_conflicting_stamps"] += 1
            continue
        counts["pose_duplicate_identical"] += len(entries) - 1
        usable.stamps.append(stamp)
        usable.samples.append((entries[0][0], first[0], first[1]))
    counts["pose_barrier_stamps"] = len(usable.barriers)
    counts["pose_usable"] = len(usable.stamps)
    return usable


def _pose_digest(usable: _Usable) -> str:
    rows = [[stamp, sample_id, translation.tolist(), quaternion.tolist()]
            for stamp, (sample_id, translation, quaternion) in zip(usable.stamps, usable.samples)]
    return hashlib.sha256(_canonical_bytes(rows)).hexdigest()


def _fit_exposures(exposures: Iterable[CameraExposure], fit: FitInterval, camera_id: str,
                   counts: dict[str, int]) -> tuple[list[CameraExposure], list[tuple[int, str]]]:
    by_stamp: dict[int, list[CameraExposure]] = {}
    latest = None
    for exposure in exposures:
        if not isinstance(exposure, CameraExposure) or exposure.camera_id != camera_id:
            raise ValueError("Exposures must be CameraExposure records for the configured camera")
        counts["exposures_total"] += 1
        stamp = exposure.timestamp_ns
        if latest is not None and stamp < latest:
            counts["exposures_out_of_order"] += 1
        latest = stamp if latest is None else max(latest, stamp)
        if stamp < fit.start_ns:
            counts["exposure_before_fit"] += 1
        elif stamp >= fit.end_ns:
            counts["exposure_at_or_after_fit_end"] += 1
        else:
            counts["exposures_in_fit"] += 1
            by_stamp.setdefault(stamp, []).append(exposure)
    accepted, rejected = [], []
    for stamp in sorted(by_stamp):
        group = by_stamp[stamp]
        if len({(item.image_sha256, item.source_id) for item in group}) > 1:
            counts["exposure_duplicate_conflicting"] += len(group)
            rejected.extend((stamp, "exposure_duplicate_conflicting") for _ in group)
            continue
        counts["exposure_duplicate_identical"] += len(group) - 1
        accepted.append(group[0])
    return accepted, rejected


def _digest_references(references: Sequence[TrainingReference]) -> str:
    return hashlib.sha256(_canonical_bytes([_reference_record(item) for item in references])).hexdigest()


def export_fit_references(exposures: Iterable[CameraExposure], poses: Iterable[PoseSample], *,
                          split: FitSplit, calibration: CameraCalibration, provenance: ExportProvenance,
                          config: ExportConfig = ExportConfig()) -> TrainingExport:
    """Interpolate T_WB at each in-fit exposure and compose T_WC = T_WB @ T_BC.

    Exposure membership is ``start_ns <= t < end_ns``. Brackets are the nearest
    usable in-fit samples with left <= t <= right; an exact sample uses itself
    for both brackets. The gap is right minus left and must not exceed
    ``config.max_interpolation_gap_ns``. A stamp with an invalid or conflicting
    sample is a barrier that no bracket pair may span.
    """
    if not isinstance(split, FitSplit) or not isinstance(calibration, CameraCalibration):
        raise ValueError("Export requires a validated FitSplit and CameraCalibration")
    if not isinstance(provenance, ExportProvenance) or not isinstance(config, ExportConfig):
        raise ValueError("Export requires typed provenance and configuration")
    if calibration.camera_id != config.camera_id:
        raise ValueError("Calibration camera does not match the configured camera")
    if {provenance.image_source_sha256, provenance.pose_source_sha256} != {split.source_recording_sha256}:
        raise ValueError("Image and pose sources must be the split's declared recording")
    fit = split.fit
    counts = dict.fromkeys(COUNT_KEYS, 0)
    usable = _usable_fit_poses(poses, fit, config, counts)
    candidates, rejections = _fit_exposures(exposures, fit, config.camera_id, counts)
    calibration_id = calibration.identity
    references = []
    for exposure in candidates:
        t = exposure.timestamp_ns
        position = bisect_left(usable.stamps, t)
        if position < len(usable.stamps) and usable.stamps[position] == t:
            left = right = position
        else:
            left, right = position - 1, position
        if left < 0:
            rejections.append((t, "no_left_bracket_in_fit"))
            continue
        if right >= len(usable.stamps):
            rejections.append((t, "no_right_bracket_in_fit"))
            continue
        left_ns, right_ns = usable.stamps[left], usable.stamps[right]
        barrier = bisect_left(usable.barriers, left_ns)
        if barrier < len(usable.barriers) and usable.barriers[barrier] < right_ns:
            rejections.append((t, "bracket_spans_invalid_pose"))
            continue
        gap = right_ns - left_ns
        if gap > config.max_interpolation_gap_ns:
            rejections.append((t, "bracket_gap_too_large"))
            continue
        left_id, left_t, left_q = usable.samples[left]
        right_id, right_t, right_q = usable.samples[right]
        if gap == 0:
            fraction, translation, quaternion = 0.0, left_t.copy(), left_q.copy()
            counts["references_exact_sample"] += 1
        else:
            fraction = (t - left_ns) / gap
            translation = left_t + fraction * (right_t - left_t)
            quaternion = slerp_xyzw(left_q, right_q, fraction)
            counts["references_interpolated"] += 1
        T_WB = pose_from_xyzw(translation, quaternion)
        frame = ReferenceFrame(
            reference_id=f"{config.camera_id}:{t}", camera_id=config.camera_id, timestamp_ns=t,
            image_sha256=exposure.image_sha256, calibration_id=calibration_id, T_WB=T_WB,
            T_WC=camera_pose_from_body(T_WB, calibration.T_BC), fit_interval=fit,
            pose_bracket_ids=(left_id, right_id), pose_bracket_timestamps_ns=(left_ns, right_ns),
            source_sha256=provenance.image_source_sha256, pose_source_sha256=provenance.pose_source_sha256,
            split_sha256=split.split_sha256)
        references.append(TrainingReference(frame, exposure.source_id, translation, quaternion,
                                            float(fraction), gap))
    for _, reason in rejections:
        if reason != "exposure_duplicate_conflicting":
            counts[reason] += 1
    counts["references_accepted"] = len(references)
    references = tuple(references)
    record = {"image_source_sha256": provenance.image_source_sha256,
              "pose_source_sha256": provenance.pose_source_sha256,
              "calibration_file_sha256": provenance.calibration_file_sha256,
              "calibration_identity": calibration_id, "split_sha256": split.split_sha256,
              "exporter_sha256": code_identities()[0],
              "map_contracts_sha256": code_identities()[1], "code_commit": provenance.code_commit,
              "config_identity": config.identity}
    return TrainingExport(EXPORT_SCHEMA, config, split, calibration, references, _proxy(counts),
                          tuple(sorted(rejections)), _proxy(record), _pose_digest(usable),
                          _digest_references(references))


# ---------------------------------------------------------------------------
# Bounded laptop sample (coordinator-run; needs the official data environment)


def _environment() -> dict[str, Any]:
    return {"python": sys.version, "platform": platform.platform(), "machine": platform.machine(),
            "numpy": np.__version__}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export a bounded fit-only reference sample report.")
    parser.add_argument("--manifest", required=True, help="official_io camera cache manifest.json")
    parser.add_argument("--mcap", required=True, help="original recording, decoded for /tf only")
    parser.add_argument("--mcap-sha256", required=True)
    parser.add_argument("--split", required=True, help="team/contracts/split.json")
    parser.add_argument("--output", required=True, help="new report path outside tracked source")
    parser.add_argument("--camera", default=DEFAULT_CAMERA_ID)
    parser.add_argument("--max-exposures", type=int, default=None,
                        help="deterministic prefix of in-fit exposures, no outcome-based selection")
    parser.add_argument("--report-references", type=int, default=5)
    parser.add_argument("--code-commit", default=None)
    args = parser.parse_args(argv)
    output = Path(args.output).resolve()
    if output.exists() or not output.parent.is_dir():
        print(json.dumps({"error": "output must be a new file in an existing directory"}), file=sys.stderr)
        return 2
    started_wall, started = time.time_ns(), time.monotonic_ns()
    from . import official_io

    split = load_fit_split(args.split)
    expected = _sha(args.mcap_sha256, "--mcap-sha256")
    if expected != split.source_recording_sha256:
        raise ValueError("Recording hash differs from the split's declared source")
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    if manifest.get("source", {}).get("sha256") != expected:
        raise ValueError("Camera cache belongs to a different recording")
    payload = official_io.read_calibration(manifest)
    calibration = contracts.calibration_from_official(payload, args.camera)
    exposure_stream = camera_exposures_from_cache(manifest, args.camera, window=split.fit)
    exposures = exposure_stream.exposures
    if args.max_exposures is not None:
        if args.max_exposures < 1:
            raise ValueError("--max-exposures must be positive")
        exposures = exposures[:args.max_exposures]
    pose_stream = read_tf_pose_samples(args.mcap, expected_sha256=expected, window=split.fit)
    provenance = ExportProvenance(exposure_stream.source_sha256, expected,
                                  exposure_stream.calibration_file_sha256, args.code_commit)
    export = export_fit_references(exposures, pose_stream.samples, split=split, calibration=calibration,
                                   provenance=provenance, config=ExportConfig(camera_id=args.camera))
    report = export.report(max_references=args.report_references)
    report["sources"] = {
        "pose_topic": POSE_TOPIC, "pose_stream_counts": dict(pose_stream.counts),
        "tf_frame_pairs": dict(pose_stream.frame_pairs), "exposure_stream_counts": dict(exposure_stream.counts),
        "frames_index_sha256": exposure_stream.frames_index_sha256,
        "exposures_in_fit_available": len(exposure_stream.exposures), "exposures_exported_bound": len(exposures),
        "max_exposures": args.max_exposures,
    }
    report["run"] = {"argv": list(sys.argv if argv is None else argv), "environment": _environment(),
                     "started_unix_ns": started_wall, "elapsed_monotonic_ns": time.monotonic_ns() - started}
    with output.open("xb") as stream:
        stream.write(json.dumps(report, sort_keys=True, indent=1, allow_nan=False).encode("utf-8") + b"\n")
    print(json.dumps({"output": str(output), "sha256": _sha256_file(output), "export_digest": export.export_digest,
                      "references_accepted": export.counts["references_accepted"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

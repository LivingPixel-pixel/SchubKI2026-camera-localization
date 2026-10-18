"""Version 1 contracts for fit-only mapping and causal camera localization.

T_AB maps B coordinates into A. W is the metric official map, B is
constructor0/base_link (forward, left, up), and C is the optical camera
(right, down, forward). Exposure times are exact integer nanoseconds;
processing times use a separate monotonic clock. This module performs no IO,
feature extraction, inference, scoring, or asynchronous scheduling.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import math
import re
from typing import Any, Literal, Mapping

import numpy as np


FIT_LAPS = (1, 2, 3)
TUNING_LAPS = (4, 5)
AUDIT_LAPS = (6, 7)
AUDIT_BASELINE_EXPOSED = True
SCHEMA_VERSION = 1
PixelConvention = Literal["raw_distorted", "rectified"]
DistanceMetric = Literal["l2", "hamming", "cosine"]
FixStatus = Literal["accepted", "rejected"]
PoseStatus = Literal["absolute_fix", "tracked", "relative_fallback", "held", "lost"]
_INT64_MAX = (1 << 63) - 1
_PIXEL_CONVENTIONS = {"raw_distorted", "rectified"}


def validate_timestamp_ns(value: int) -> int:
    """Accept exact nonnegative signed-int64 nanoseconds, never floats/bools."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError("Timestamp must be integer nanoseconds")
    value = int(value)
    if not 0 <= value <= _INT64_MAX:
        raise ValueError("Timestamp must fit nonnegative signed int64")
    return value


def _integer(value: int, name: str, minimum: int = 0) -> int:
    result = validate_timestamp_ns(value)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _text(value: str, name: str, empty: bool = False) -> str:
    if not isinstance(value, str) or (not empty and not value.strip()):
        raise ValueError(f"{name} must be a {'possibly empty ' if empty else 'nonempty '}string")
    return value


def _sha(value: str, name: str = "SHA256") -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-fA-F]{64}", value) is None:
        raise ValueError(f"{name} must be a SHA256 hex digest")
    return value.lower()


def _finite(value: float, name: str, minimum: float | None = None) -> float:
    if isinstance(value, (bool, np.bool_, complex, np.complexfloating)) or not isinstance(value, (int, float, np.number)):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        raise ValueError(f"{name} must be finite" + (f" and >= {minimum}" if minimum is not None else ""))
    return result


class _ReadOnlyArray(np.ndarray):
    """Protect stored metadata as well as bytes through ordinary array APIs.

    NumPy views and np.asarray may expose a separate ndarray header. Changing
    that header does not change this stored array, and its bytes stay read-only.
    Like frozen dataclasses, this is an API invariant, not a security boundary
    against deliberate calls to base-class mutation internals.
    """

    def __setattr__(self, name: str, value: Any) -> None:
        if name in {"shape", "dtype", "strides", "data"}:
            raise ValueError("Contract array metadata is immutable; make an explicit copy")
        super().__setattr__(name, value)

    def resize(self, *args: Any, **kwargs: Any) -> None:
        raise ValueError("Contract arrays cannot be resized; make an explicit copy")


def _immutable(array: np.ndarray) -> np.ndarray:
    """Copy bytes and protect both their write flag and stored shape/dtype."""
    value = np.ascontiguousarray(array)
    return np.frombuffer(value.tobytes(), dtype=value.dtype).reshape(value.shape).view(_ReadOnlyArray)


def _float_array(value: Any, name: str, shape: tuple[int, ...] | None = None) -> np.ndarray:
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf" or (shape is not None and raw.shape != shape):
        raise ValueError(f"{name} must be a numeric array with shape {shape}")
    array = np.asarray(raw, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain finite values")
    return _immutable(array)


def _ids(value: Any, name: str, unique: bool = True) -> np.ndarray:
    raw = np.asarray(value)
    # An empty Python sequence has float dtype, but carries no inexact IDs.
    if raw.ndim != 1 or (raw.size and raw.dtype.kind not in "iu"):
        raise ValueError(f"{name} must be a one-dimensional integer array")
    if raw.size and (np.any(raw < 0) or np.any(raw > _INT64_MAX)):
        raise ValueError(f"{name} must contain nonnegative int64 IDs")
    array = np.asarray(raw, dtype=np.int64)
    if unique and len(np.unique(array)) != len(array):
        raise ValueError(f"{name} must be unique")
    return _immutable(array)


def _set(instance: Any, **values: Any) -> None:
    for name, value in values.items():
        object.__setattr__(instance, name, value)


def _canonical_json(metadata: Mapping[str, Any]) -> str:
    if not isinstance(metadata, Mapping):
        raise ValueError("Identity metadata must be a JSON object")
    try:
        return json.dumps(dict(metadata), sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("Identity metadata must contain finite JSON values") from exc


def canonical_identity(metadata: Mapping[str, Any]) -> str:
    """Hash canonical finite JSON metadata, including effective configuration."""
    return hashlib.sha256(_canonical_json(metadata).encode("utf-8")).hexdigest()


def validate_transform(T: Any, name: str = "transform") -> np.ndarray:
    result = _float_array(T, name, (4, 4))
    rotation = result[:3, :3]
    if (not np.allclose(result[3], [0, 0, 0, 1], atol=1e-10, rtol=0)
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-7, rtol=0)
            or not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=1e-7)):
        raise ValueError(f"{name} must be a proper rigid homogeneous transform")
    return result


def invert_transform(T_AB: Any) -> np.ndarray:
    transform = validate_transform(T_AB)
    result = np.eye(4)
    result[:3, :3] = transform[:3, :3].T
    result[:3, 3] = -result[:3, :3] @ transform[:3, 3]
    return _immutable(result)


def camera_pose_from_body(T_WB: Any, T_BC: Any) -> np.ndarray:
    return validate_transform(validate_transform(T_WB, "T_WB") @ validate_transform(T_BC, "T_BC"), "T_WC")


def body_pose_from_pnp(T_CW: Any, T_BC: Any) -> np.ndarray:
    """OpenCV PnP returns T_CW; output T_WB = inv(T_CW) @ inv(T_BC)."""
    return validate_transform(invert_transform(T_CW) @ invert_transform(T_BC), "T_WB")


def yaw_from_body_pose(T_WB: Any) -> float:
    transform = validate_transform(T_WB, "T_WB")
    return float(np.arctan2(transform[1, 0], transform[0, 0]))


def pose_from_xyzw(translation: Any, quaternion: Any) -> np.ndarray:
    """Build a pose from metres and ROS quaternion order x,y,z,w."""
    position = _float_array(translation, "translation", (3,))
    q = _float_array(quaternion, "quaternion xyzw", (4,))
    norm = float(np.linalg.norm(q))
    if abs(norm - 1.0) > 1e-3:
        raise ValueError("Quaternion must be unit length within 1e-3")
    x, y, z, w = q / norm
    result = np.eye(4)
    result[:3, :3] = [
        [1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
        [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
        [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)],
    ]
    result[:3, 3] = position
    return validate_transform(result)


@dataclass(frozen=True, slots=True)
class FitInterval:
    start_ns: int
    end_ns: int
    lap_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        start, end = validate_timestamp_ns(self.start_ns), validate_timestamp_ns(self.end_ns)
        laps = tuple(_integer(value, "lap ID", 1) for value in self.lap_ids)
        if start >= end or not laps or len(set(laps)) != len(laps) or not set(laps) <= set(FIT_LAPS):
            raise ValueError("Fit interval must be nonempty, half-open, and restricted to distinct laps 1-3")
        _set(self, start_ns=start, end_ns=end, lap_ids=laps)

    def contains(self, timestamp_ns: int) -> bool:
        return self.start_ns <= validate_timestamp_ns(timestamp_ns) < self.end_ns


@dataclass(frozen=True, slots=True)
class FeatureIdentity:
    extractor: str
    version: str
    weights_sha256: str
    descriptor_dimension: int
    descriptor_dtype: str
    distance_metric: DistanceMetric
    pixel_convention: PixelConvention
    config_json: str

    def __post_init__(self) -> None:
        _text(self.extractor, "extractor")
        _text(self.version, "extractor version")
        weights = "none" if self.weights_sha256 == "none" else _sha(self.weights_sha256, "weights identity")
        dimension = _integer(self.descriptor_dimension, "descriptor dimension", 1)
        if self.descriptor_dtype not in {"float32", "uint8"} or self.distance_metric not in {"l2", "hamming", "cosine"}:
            raise ValueError("Unsupported descriptor dtype or distance metric")
        if (self.distance_metric == "hamming") != (self.descriptor_dtype == "uint8"):
            raise ValueError("Hamming requires uint8; l2/cosine require float32 descriptors")
        if self.pixel_convention not in _PIXEL_CONVENTIONS:
            raise ValueError("Unknown feature pixel convention")
        try:
            settings = json.loads(self.config_json)
        except (TypeError, ValueError) as exc:
            raise ValueError("config_json must encode effective feature settings") from exc
        _set(self, weights_sha256=weights, descriptor_dimension=dimension, config_json=_canonical_json(settings))

    @property
    def identity(self) -> str:
        return canonical_identity({"extractor": self.extractor, "version": self.version,
            "weights_sha256": self.weights_sha256, "descriptor_dimension": self.descriptor_dimension,
            "descriptor_dtype": self.descriptor_dtype, "distance_metric": self.distance_metric,
            "pixel_convention": self.pixel_convention, "config": json.loads(self.config_json)})


@dataclass(frozen=True, slots=True)
class CameraCalibration:
    camera_id: str
    width: int
    height: int
    K: np.ndarray
    D: np.ndarray
    T_BC: np.ndarray
    distortion_model: str
    pixel_convention: PixelConvention = "raw_distorted"
    body_frame: str = "constructor0/base_link"
    camera_frame: str = ""

    def __post_init__(self) -> None:
        camera = _text(self.camera_id, "camera ID")
        width, height = _integer(self.width, "width", 1), _integer(self.height, "height", 1)
        k, d = _float_array(self.K, "K", (3, 3)), _float_array(self.D, "D")
        if (k[0, 0] <= 0 or k[1, 1] <= 0 or k[0, 1] != 0 or k[1, 0] != 0
                or not np.allclose(k[2], [0, 0, 1], atol=1e-10, rtol=0)
                or d.ndim != 1 or d.size not in {0, 4, 5, 8, 12, 14}):
            raise ValueError("Invalid pinhole camera intrinsics or OpenCV distortion coefficients")
        if self.distortion_model not in {"", "plumb_bob", "rational_polynomial"}:
            raise ValueError("Unsupported distortion model")
        if self.pixel_convention not in _PIXEL_CONVENTIONS or (self.pixel_convention == "rectified" and np.any(d != 0)):
            raise ValueError("Unknown pixel convention or nonzero rectified distortion")
        frame = self.camera_frame or f"constructor0/camera_{camera}"
        if self.body_frame != "constructor0/base_link" or frame != f"constructor0/camera_{camera}":
            raise ValueError("Calibration frame does not match the declared body/camera")
        _set(self, width=width, height=height, K=k, D=d, T_BC=validate_transform(self.T_BC, "T_BC"), camera_frame=frame)

    @property
    def identity(self) -> str:
        return canonical_identity({"camera_id": self.camera_id, "width": self.width, "height": self.height,
            "K": self.K.tolist(), "D": self.D.tolist(), "T_BC": self.T_BC.tolist(),
            "distortion_model": self.distortion_model, "pixel_convention": self.pixel_convention,
            "body_frame": self.body_frame, "camera_frame": self.camera_frame})


def calibration_from_official(payload: Mapping[str, Any], camera_id: str) -> CameraCalibration:
    """Normalize official_io.read_calibration output without retaining its dict.

    Official T_BC is camera-to-body. Legacy vision.py instead takes T_CV,
    which callers obtain with invert_transform(calibration.T_BC).
    """
    if not isinstance(payload, Mapping) or payload.get("static_stable") is not True:
        raise ValueError("Official calibration must explicitly be static and stable")
    try:
        width = payload["width"] if "width" in payload else payload["image_width"]
        height = payload["height"] if "height" in payload else payload["image_height"]
        distortion = payload["D"] if "D" in payload else payload["distortion"]
        for primary, alias in (("width", "image_width"), ("height", "image_height"), ("D", "distortion")):
            if primary in payload and alias in payload and not np.array_equal(payload[primary], payload[alias]):
                raise ValueError(f"Conflicting official calibration aliases: {primary}/{alias}")
        if "camera" in payload and payload["camera"] != camera_id:
            raise ValueError("Official camera ID mismatch")
        return CameraCalibration(camera_id, width, height, payload["K"], distortion, payload["T_BC"],
            payload["distortion_model"], "raw_distorted", payload["body_frame"], payload["camera_frame"])
    except KeyError as exc:
        raise ValueError(f"Missing official calibration field: {exc.args[0]}") from exc


def scaled_calibration(cal: CameraCalibration, width: int, height: int) -> CameraCalibration:
    """Full raw-image resize only: x'=sx*x, y'=sy*y; K'=diag(sx,sy,1)K.

    D and T_BC stay unchanged. This contract uses the OpenCV geometric pixel
    scaling convention, not a half-pixel offset. Crops, padding, rectification,
    and transforms with pixel offsets need their own calibrated operation.
    """
    if not isinstance(cal, CameraCalibration) or cal.pixel_convention != "raw_distorted":
        raise ValueError("Scaling requires raw distorted full-image calibration")
    width, height = _integer(width, "width", 1), _integer(height, "height", 1)
    k = np.diag([width / cal.width, height / cal.height, 1.0]) @ cal.K
    return replace(cal, width=width, height=height, K=k)


def require_compatible(feature_a: FeatureIdentity, feature_b: FeatureIdentity,
                       calibration_id_a: str, calibration_id_b: str) -> None:
    if not isinstance(feature_a, FeatureIdentity) or not isinstance(feature_b, FeatureIdentity):
        raise ValueError("Compatibility requires typed feature identities")
    if feature_a.identity != feature_b.identity or _sha(calibration_id_a) != _sha(calibration_id_b):
        raise ValueError("Feature or calibration identity mismatch")


@dataclass(frozen=True, slots=True)
class CameraObservation:
    camera_id: str
    timestamp_ns: int
    image: np.ndarray
    calibration_id: str

    def __post_init__(self) -> None:
        _text(self.camera_id, "camera ID")
        image = np.asarray(self.image)
        if image.dtype != np.uint8 or image.ndim not in {2, 3} or min(image.shape[:2]) < 1 or (image.ndim == 3 and image.shape[2] != 3):
            raise ValueError("Image must be nonempty uint8 grayscale or BGR pixels")
        _set(self, timestamp_ns=validate_timestamp_ns(self.timestamp_ns), image=_immutable(image), calibration_id=_sha(self.calibration_id))


@dataclass(frozen=True, slots=True)
class ReferenceFrame:
    reference_id: str
    camera_id: str
    timestamp_ns: int
    image_sha256: str
    calibration_id: str
    T_WB: np.ndarray
    T_WC: np.ndarray
    fit_interval: FitInterval
    pose_bracket_ids: tuple[str, str]
    pose_bracket_timestamps_ns: tuple[int, int]
    source_sha256: str
    pose_source_sha256: str
    split_sha256: str

    def __post_init__(self) -> None:
        _text(self.reference_id, "reference ID")
        _text(self.camera_id, "camera ID")
        timestamp = validate_timestamp_ns(self.timestamp_ns)
        ids = tuple(_text(value, "pose bracket ID") for value in self.pose_bracket_ids)
        stamps = tuple(validate_timestamp_ns(value) for value in self.pose_bracket_timestamps_ns)
        if not isinstance(self.fit_interval, FitInterval) or len(ids) != 2 or len(stamps) != 2:
            raise ValueError("Reference requires a fit interval and two pose brackets")
        if not all(self.fit_interval.contains(value) for value in (timestamp, *stamps)):
            raise ValueError("Exposure and both pose brackets must be in the same half-open fit interval")
        if not stamps[0] <= timestamp <= stamps[1] or ((stamps[0] == stamps[1]) != (ids[0] == ids[1])):
            raise ValueError("Pose brackets must enclose exposure; identical stamps require identical sample IDs")
        _set(self, timestamp_ns=timestamp, pose_bracket_ids=ids, pose_bracket_timestamps_ns=stamps,
            T_WB=validate_transform(self.T_WB, "T_WB"), T_WC=validate_transform(self.T_WC, "T_WC"),
            **{name: _sha(getattr(self, name), name) for name in
               ("image_sha256", "calibration_id", "source_sha256", "pose_source_sha256", "split_sha256")})


@dataclass(frozen=True, slots=True)
class ReferenceObservation:
    reference_id: str
    landmark_ids: np.ndarray
    pixels_xy: np.ndarray

    def __post_init__(self) -> None:
        _text(self.reference_id, "reference ID")
        ids = _ids(self.landmark_ids, "reference landmark IDs")
        pixels = _float_array(self.pixels_xy, "reference pixels", (len(ids), 2))
        _set(self, landmark_ids=ids, pixels_xy=pixels)


@dataclass(frozen=True, slots=True)
class MapProvenance:
    source_sha256: str
    split_sha256: str
    exporter_sha256: str
    code_commit: str

    def __post_init__(self) -> None:
        if not isinstance(self.code_commit, str) or re.fullmatch(r"[0-9a-fA-F]{40}", self.code_commit) is None:
            raise ValueError("Map code_commit must be an exact 40-character Git commit")
        _set(self, code_commit=self.code_commit.lower(), **{name: _sha(getattr(self, name), name)
             for name in ("source_sha256", "split_sha256", "exporter_sha256")})


@dataclass(frozen=True, slots=True)
class FrozenMap:
    schema_version: int
    map_hash: str
    landmark_ids: np.ndarray
    landmarks_xyz: np.ndarray
    descriptors: np.ndarray
    references: tuple[ReferenceFrame, ...]
    reference_observations: tuple[ReferenceObservation, ...]
    feature_identity: FeatureIdentity
    calibration: CameraCalibration
    fit_membership: tuple[FitInterval, ...]
    retrieval_index_sha256: str
    provenance: MapProvenance

    def __post_init__(self) -> None:
        if _integer(self.schema_version, "schema version", 1) != SCHEMA_VERSION:
            raise ValueError("Unsupported map schema version")
        if not isinstance(self.feature_identity, FeatureIdentity) or not isinstance(self.calibration, CameraCalibration) or not isinstance(self.provenance, MapProvenance):
            raise ValueError("Map requires typed feature, calibration, and provenance contracts")
        ids = _ids(self.landmark_ids, "map landmark IDs")
        xyz = _float_array(self.landmarks_xyz, "metric world landmarks", (len(ids), 3))
        descriptors = np.asarray(self.descriptors)
        if (not len(ids) or descriptors.shape != (len(ids), self.feature_identity.descriptor_dimension)
                or descriptors.dtype != np.dtype(self.feature_identity.descriptor_dtype) or not np.isfinite(descriptors).all()):
            raise ValueError("Map descriptors must match nonempty landmark rows, feature dimension, and exact dtype")
        references, observations, membership = tuple(self.references), tuple(self.reference_observations), tuple(self.fit_membership)
        if not references or not membership or any(not isinstance(item, FitInterval) for item in membership):
            raise ValueError("Map requires references and explicit fit membership")
        ordered = sorted(membership, key=lambda item: item.start_ns)
        if any(left.end_ns > right.start_ns for left, right in zip(ordered, ordered[1:])):
            raise ValueError("Fit membership intervals must not overlap")
        reference_ids = set()
        for reference in references:
            if not isinstance(reference, ReferenceFrame) or reference.reference_id in reference_ids:
                raise ValueError("Map references must have unique typed identities")
            reference_ids.add(reference.reference_id)
            if (reference.camera_id != self.calibration.camera_id or reference.calibration_id != self.calibration.identity
                    or reference.fit_interval not in membership or reference.source_sha256 != self.provenance.source_sha256
                    or reference.split_sha256 != self.provenance.split_sha256):
                raise ValueError("Reference camera, calibration, fit membership, or provenance mismatch")
            expected = camera_pose_from_body(reference.T_WB, self.calibration.T_BC)
            if not np.allclose(reference.T_WC, expected, atol=1e-7, rtol=0):
                raise ValueError("Reference T_WC must equal T_WB @ T_BC")
        seen, known = set(), set(ids.tolist())
        for observation in observations:
            if not isinstance(observation, ReferenceObservation) or observation.reference_id not in reference_ids or observation.reference_id in seen:
                raise ValueError("Reference observations must point to unique existing references")
            seen.add(observation.reference_id)
            if not set(observation.landmark_ids.tolist()) <= known:
                raise ValueError("Reference observation points to an unknown landmark")
            pixels = observation.pixels_xy
            if np.any(pixels < 0) or np.any(pixels[:, 0] >= self.calibration.width) or np.any(pixels[:, 1] >= self.calibration.height):
                raise ValueError("Reference pixels fall outside calibrated image dimensions")
        if self.feature_identity.pixel_convention != self.calibration.pixel_convention:
            raise ValueError("Map feature pixels and calibration conventions differ")
        _set(self, schema_version=SCHEMA_VERSION, map_hash=_sha(self.map_hash, "map hash"),
            landmark_ids=ids, landmarks_xyz=xyz, descriptors=_immutable(descriptors), references=references,
            reference_observations=observations, fit_membership=membership,
            retrieval_index_sha256=_sha(self.retrieval_index_sha256, "retrieval index hash"))


@dataclass(frozen=True, slots=True)
class FixCandidate:
    source_timestamp_ns: int
    T_WB: np.ndarray | None
    status: FixStatus
    reason: str
    landmark_ids: np.ndarray
    inlier_mask: np.ndarray
    reprojection_error_px: np.ndarray
    conditioning: float
    region_id: str
    ambiguity: float
    map_hash: str
    feature_id: str
    calibration_id: str
    latest_input_timestamp_ns: int
    completed_monotonic_ns: int

    def __post_init__(self) -> None:
        source, latest = validate_timestamp_ns(self.source_timestamp_ns), validate_timestamp_ns(self.latest_input_timestamp_ns)
        ids = _ids(self.landmark_ids, "fix landmark IDs")
        mask = np.asarray(self.inlier_mask)
        errors = _float_array(self.reprojection_error_px, "reprojection errors", (len(ids),))
        if mask.shape != (len(ids),) or mask.dtype != np.bool_ or np.any(errors < 0) or latest > source:
            raise ValueError("Fix correspondences or causal input time are invalid")
        if self.status not in {"accepted", "rejected"}:
            raise ValueError("Fix status must be accepted or rejected")
        _text(self.reason, "fix reason", empty=self.status == "accepted")
        _text(self.region_id, "region ID", empty=self.status == "rejected")
        if self.status == "accepted":
            if self.T_WB is None or np.count_nonzero(mask) < 4:
                raise ValueError("Accepted fix requires a body pose and at least four inliers")
            pose = validate_transform(self.T_WB, "fix T_WB")
        else:
            if self.T_WB is not None:
                raise ValueError("Rejected fix must not carry an accepted body pose")
            pose = None
        _set(self, source_timestamp_ns=source, latest_input_timestamp_ns=latest, T_WB=pose,
            landmark_ids=ids, inlier_mask=_immutable(mask), reprojection_error_px=errors,
            conditioning=_finite(self.conditioning, "conditioning", 0), ambiguity=_finite(self.ambiguity, "ambiguity", 0),
            completed_monotonic_ns=validate_timestamp_ns(self.completed_monotonic_ns),
            **{name: _sha(getattr(self, name), name) for name in ("map_hash", "feature_id", "calibration_id")})


@dataclass(frozen=True, slots=True)
class RelativeMotion:
    previous_timestamp_ns: int
    timestamp_ns: int
    T_Bprevious_Bcurrent: np.ndarray | None
    valid: bool
    reason: str
    source_identity: str

    def __post_init__(self) -> None:
        previous, current = validate_timestamp_ns(self.previous_timestamp_ns), validate_timestamp_ns(self.timestamp_ns)
        if previous >= current or not isinstance(self.valid, (bool, np.bool_)):
            raise ValueError("Relative motion requires increasing exposure times and boolean validity")
        _text(self.reason, "relative motion reason", empty=bool(self.valid))
        _text(self.source_identity, "relative motion source identity")
        if bool(self.valid) != (self.T_Bprevious_Bcurrent is not None):
            raise ValueError("Valid motion requires a transform; invalid motion must omit it")
        pose = validate_transform(self.T_Bprevious_Bcurrent, "T_Bprevious_Bcurrent") if self.valid else None
        _set(self, previous_timestamp_ns=previous, timestamp_ns=current, valid=bool(self.valid), T_Bprevious_Bcurrent=pose)


@dataclass(frozen=True, slots=True)
class ProcessingTiming:
    received_monotonic_ns: int
    started_monotonic_ns: int
    completed_monotonic_ns: int
    published_monotonic_ns: int
    queue_drops: int = 0

    def __post_init__(self) -> None:
        names = ("received_monotonic_ns", "started_monotonic_ns", "completed_monotonic_ns", "published_monotonic_ns")
        stamps = tuple(validate_timestamp_ns(getattr(self, name)) for name in names)
        if any(left > right for left, right in zip(stamps, stamps[1:])):
            raise ValueError("Processing times must be ordered in one monotonic clock domain")
        _set(self, **dict(zip(names, stamps)), queue_drops=_integer(self.queue_drops, "queue drops"))


@dataclass(frozen=True, slots=True)
class PoseDiagnostics:
    inlier_count: int = 0
    reprojection_rmse_px: float | None = None
    reason: str = ""

    def __post_init__(self) -> None:
        _text(self.reason, "pose reason", empty=True)
        error = None if self.reprojection_rmse_px is None else _finite(self.reprojection_rmse_px, "reprojection RMSE", 0)
        _set(self, inlier_count=_integer(self.inlier_count, "inlier count"), reprojection_rmse_px=error)


@dataclass(frozen=True, slots=True)
class PoseResult:
    timestamp_ns: int
    x_m: float
    y_m: float
    yaw_rad: float
    status: PoseStatus
    latest_input_timestamp_ns: int
    last_absolute_fix_timestamp_ns: int | None
    correction_age_ns: int | None
    timing: ProcessingTiming
    diagnostics: PoseDiagnostics

    def __post_init__(self) -> None:
        timestamp, latest = validate_timestamp_ns(self.timestamp_ns), validate_timestamp_ns(self.latest_input_timestamp_ns)
        last = None if self.last_absolute_fix_timestamp_ns is None else validate_timestamp_ns(self.last_absolute_fix_timestamp_ns)
        age = None if self.correction_age_ns is None else validate_timestamp_ns(self.correction_age_ns)
        if self.status not in {"absolute_fix", "tracked", "relative_fallback", "held", "lost"}:
            raise ValueError("Unknown pose status")
        if latest > timestamp or (last is not None and last > latest):
            raise ValueError("Pose input and absolute-fix exposure times must be causal")
        if (last is None and age is not None) or (last is not None and age != timestamp - last):
            raise ValueError("Correction age must equal output exposure minus last absolute fix, or both must be absent")
        if (self.status == "absolute_fix" and last != timestamp) or (self.status == "tracked" and last is None):
            raise ValueError("Absolute fix must be current; tracking requires an absolute fix history")
        yaw = _finite(self.yaw_rad, "yaw radians")
        if not -math.pi <= yaw <= math.pi or not isinstance(self.timing, ProcessingTiming) or not isinstance(self.diagnostics, PoseDiagnostics):
            raise ValueError("Pose requires yaw in [-pi,pi] and typed timing/diagnostics")
        _set(self, timestamp_ns=timestamp, latest_input_timestamp_ns=latest, last_absolute_fix_timestamp_ns=last,
            correction_age_ns=age, x_m=_finite(self.x_m, "x metres"), y_m=_finite(self.y_m, "y metres"), yaw_rad=yaw)

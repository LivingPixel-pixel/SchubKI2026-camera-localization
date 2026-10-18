"""Research-only image place baseline. Copies a fit reference's planar BODY pose.

Homographies verify image support, never recover metric translation or query yaw.
No query labels, progress lookup, landmarks, prediction or SE3 reconstruction.
"""
from __future__ import annotations

import base64
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Callable

import cv2
import numpy as np

from .map_contracts import CameraCalibration, CameraObservation, _immutable, validate_timestamp_ns
from .map_retrieval import QueryFeatures, SiftConfig, SiftExtractor, match_descriptors, sift_feature_identity

SCHEMA = "planar-keyframe-research-v1"
MAX_ARTIFACT_BYTES = 128 * 1024 * 1024
PIN_KEYS = {"source_sha256", "recording_sha256", "split_sha256", "label_source_sha256"}


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _hash(value):
    return hashlib.sha256(value).hexdigest()


def _sha(value):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("expected lowercase SHA256")
    return value


def _int(value, low, high):
    if type(value) is not int or not low <= value <= high:
        raise ValueError("integer outside declared bound")
    return value


def _finite(value):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError("expected finite number")
    return float(value)


def _text(value):
    if not isinstance(value, str) or not value or len(value) > 256:
        raise ValueError("invalid bounded identity")
    return value


@dataclass(frozen=True, slots=True)
class KeyframeConfig:
    max_references: int = 64
    max_keypoints: int = 2000
    max_geometry_candidates: int = 64
    min_matches: int = 12
    min_inliers: int = 12
    min_inlier_fraction: float = 0.35
    transfer_threshold_px: float = 3.0
    ransac_iterations: int = 1000
    ransac_confidence: float = 0.995
    ransac_seed: int = 17
    grid_size: int = 4
    min_grid_cells: int = 4
    min_bbox_area_fraction: float = 0.02
    ambiguity_distance_m: float = 10.0
    ambiguity_yaw_rad: float = 0.6
    max_encoded_image_bytes: int = 12 * 1024 * 1024
    max_decoded_pixels: int = 8_000_000
    max_artifact_bytes: int = MAX_ARTIFACT_BYTES

    def __post_init__(self):
        for name, low, high in (("max_references", 1, 64), ("max_keypoints", 12, 2000),
                ("max_geometry_candidates", 1, 64), ("min_matches", 12, 2000), ("min_inliers", 12, 2000),
                ("ransac_iterations", 1, 1000), ("ransac_seed", 0, 2**31-1), ("grid_size", 4, 4),
                ("min_grid_cells", 4, 16), ("max_encoded_image_bytes", 1, 12*1024*1024),
                ("max_decoded_pixels", 1, 8_000_000), ("max_artifact_bytes", 1, MAX_ARTIFACT_BYTES)):
            _int(getattr(self, name), low, high)
        for name, low, high in (("min_inlier_fraction", .35, 1.), ("transfer_threshold_px", .01, 3.),
                ("ransac_confidence", .5, .999999), ("min_bbox_area_fraction", .02, 1.),
                ("ambiguity_distance_m", .01, 10.), ("ambiguity_yaw_rad", .001, .6)):
            if not low <= _finite(getattr(self, name)) <= high:
                raise ValueError(f"invalid {name}")


@dataclass(frozen=True, slots=True)
class ReferencePose:
    reference_id: str
    image_timestamp_ns: int
    label_timestamp_ns: int
    image_sha256: str
    x: float
    y: float
    yaw: float
    lap: int
    map_frame: str
    label_source_sha256: str
    recording_sha256: str
    split_sha256: str
    calibration_id: str
    association: str = "nearest_timestamp"
    max_gap_ns: int = 50_000_000
    position_units: str = "metres"
    yaw_units: str = "radians"
    body_frame: str = "constructor0/base_link"

    def __post_init__(self):
        _text(self.reference_id)
        _text(self.map_frame)
        for name in ("image_timestamp_ns", "label_timestamp_ns"):
            validate_timestamp_ns(getattr(self, name))
        for name in ("image_sha256", "label_source_sha256", "recording_sha256", "split_sha256", "calibration_id"):
            _sha(getattr(self, name))
        for name in ("x", "y", "yaw"):
            object.__setattr__(self, name, _finite(getattr(self, name)))
        _int(self.lap, 1, 3)
        _int(self.max_gap_ns, 0, 50_000_000)
        if (abs(self.image_timestamp_ns-self.label_timestamp_ns) > self.max_gap_ns
                or self.association != "nearest_timestamp" or self.position_units != "metres"
                or self.yaw_units != "radians" or self.body_frame != "constructor0/base_link"
                or not -math.pi <= self.yaw <= math.pi):
            raise ValueError("invalid planar reference association/frame/units")


@dataclass(frozen=True, slots=True)
class KeyframeReference:
    pose: ReferencePose
    keypoints_xy: np.ndarray
    descriptors: np.ndarray

    def __post_init__(self):
        p, d = np.asarray(self.keypoints_xy), np.asarray(self.descriptors)
        if (p.dtype != np.float64 or p.ndim != 2 or p.shape[1:] != (2,) or len(p) > 2000
                or d.dtype != np.float32 or d.shape != (len(p), 128)
                or not np.isfinite(p).all() or not np.isfinite(d).all()):
            raise ValueError("invalid bounded reference features")
        object.__setattr__(self, "keypoints_xy", _immutable(p))
        object.__setattr__(self, "descriptors", _immutable(d))


@dataclass(frozen=True, slots=True)
class KeyframeMap:
    calibration: CameraCalibration
    sift_config: SiftConfig
    config: KeyframeConfig
    provenance: tuple[tuple[str, str], ...]
    fit_intervals: tuple[tuple[int, int, int], ...]
    references: tuple[KeyframeReference, ...]
    selection_sha256: str
    identity: str

    @property
    def feature_id(self):
        return sift_feature_identity(self.calibration, self.sift_config).identity


def _metadata(calibration, sift_config, config, provenance, intervals, poses):
    if not isinstance(calibration, CameraCalibration) or calibration.pixel_convention != "raw_distorted":
        raise ValueError("raw calibrated pixels required")
    if not isinstance(sift_config, SiftConfig) or sift_config.max_features > config.max_keypoints:
        raise ValueError("feature cap mismatch")
    if calibration.width*calibration.height > config.max_decoded_pixels:
        raise ValueError("decoded pixel cap")
    if set(provenance) != PIN_KEYS:
        raise ValueError("exact independent provenance pins required")
    provenance = {k: _sha(v) for k, v in provenance.items()}
    if not 1 <= len(poses) <= config.max_references:
        raise ValueError("reference count cap")
    if not isinstance(intervals, (list, tuple)) or not 1 <= len(intervals) <= 3:
        raise ValueError("bounded fit intervals required")
    normalized = []
    for lap, start, end in intervals:
        _int(lap, 1, 3)
        validate_timestamp_ns(start)
        validate_timestamp_ns(end)
        if start >= end:
            raise ValueError("invalid half-open fit interval")
        normalized.append((lap, start, end))
    if not 1 <= len(normalized) <= 3 or len({v[0] for v in normalized}) != len(normalized):
        raise ValueError("unique fit lap intervals required")
    normalized.sort(key=lambda v: v[1])
    if any(a[2] > b[1] for a, b in zip(normalized, normalized[1:])):
        raise ValueError("overlapping fit intervals")
    ids, hashes = set(), set()
    frame = poses[0].map_frame
    for pose in poses:
        if not isinstance(pose, ReferencePose) or pose.reference_id in ids or pose.image_sha256 in hashes:
            raise ValueError("duplicate or untyped reference")
        ids.add(pose.reference_id)
        hashes.add(pose.image_sha256)
        if (pose.map_frame != frame or pose.calibration_id != calibration.identity
                or any(getattr(pose, k) != provenance[k] for k in PIN_KEYS-{"source_sha256"})
                or not any(lap == pose.lap and start <= pose.image_timestamp_ns < end
                    and start <= pose.label_timestamp_ns < end for lap, start, end in normalized)):
            raise ValueError("reference outside frozen fit/provenance")
    return {"schema": SCHEMA, "calibration_id": calibration.identity,
            "feature_id": sift_feature_identity(calibration, sift_config).identity,
            "sift_config": asdict(sift_config), "config": asdict(config), "provenance": provenance,
            "fit_intervals": normalized, "selection": [asdict(p) for p in poses]}


def _make_map(calibration, sift_config, config, provenance, intervals, references):
    references = tuple(references)
    meta = _metadata(calibration, sift_config, config, provenance, intervals, [r.pose for r in references])
    for r in references:
        if (len(r.keypoints_xy) > config.max_keypoints or np.any(r.keypoints_xy < 0)
                or np.any(r.keypoints_xy[:, 0] >= calibration.width)
                or np.any(r.keypoints_xy[:, 1] >= calibration.height)):
            raise ValueError("reference feature count/pixels outside limits")
    selection = _hash(_json(meta))
    identity = _hash(_json({"metadata": meta, "arrays": [
        [_hash(r.keypoints_xy.tobytes()), _hash(r.descriptors.tobytes())] for r in references]}))
    return KeyframeMap(calibration, sift_config, config, tuple(sorted(provenance.items())),
                       tuple(tuple(v) for v in meta["fit_intervals"]), references, selection, identity)


def _decode(encoded, calibration, config):
    if type(encoded) is not bytes or not 1 <= len(encoded) <= config.max_encoded_image_bytes:
        raise ValueError("encoded image byte cap")
    image = cv2.imdecode(np.frombuffer(encoded, np.uint8), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if image is None or image.shape != (calibration.height, calibration.width, 3):
        raise ValueError("decoded image/calibration shape mismatch")
    return image


def build_keyframe_map(poses, image_loader: Callable[[str], bytes], *, calibration,
                       provenance, fit_intervals, config=None, sift_config=None):
    """Freeze all metadata before loading bytes; one verified image/decode per reference.

    SIFT's accepted detector has no internal peak-memory bound. An outer reviewed
    runtime envelope must enforce wall, memory and total decode/byte budgets.
    """
    config = KeyframeConfig() if config is None else config
    # Explicit override of accepted default bottom mask .06, recorded in identity.
    sift_config = SiftConfig(mask_bottom_fraction=0.0) if sift_config is None else sift_config
    if not isinstance(poses, (tuple, list)) or not 1 <= len(poses) <= config.max_references:
        raise ValueError("bounded explicit reference sequence required")
    poses = tuple(poses)
    provenance = dict(provenance)
    frozen = _metadata(calibration, sift_config, config, provenance, fit_intervals, poses)
    fit_intervals = tuple(tuple(v) for v in frozen["fit_intervals"])
    frozen_selection_sha = _hash(_json(frozen))
    extractor = SiftExtractor(calibration, sift_config)
    references, total = [], 0
    for pose in poses:
        encoded = image_loader(pose.reference_id)
        if type(encoded) is not bytes or not 1 <= len(encoded) <= config.max_encoded_image_bytes:
            raise ValueError("reference encoded byte cap")
        total += len(encoded)
        if total > config.max_encoded_image_bytes:
            raise ValueError("aggregate encoded reference byte cap")
        if _hash(encoded) != pose.image_sha256:
            raise ValueError("reference image SHA mismatch")
        image = _decode(encoded, calibration, config)
        features = extractor.extract(CameraObservation(calibration.camera_id, pose.image_timestamp_ns,
                                                       image, calibration.identity))
        references.append(KeyframeReference(pose, features.keypoints_xy, features.descriptors))
        del image, features, encoded
    model = _make_map(calibration, sift_config, config, provenance, fit_intervals, references)
    if model.selection_sha256 != frozen_selection_sha:
        raise ValueError("metadata changed during feature extraction")
    return model


@dataclass(frozen=True, slots=True)
class CandidateEvidence:
    reference_id: str
    matches: int
    inliers: int = 0
    inlier_fraction: float = 0.0
    median_symmetric_error_px: float | None = None
    query_grid_cells: int = 0
    reference_grid_cells: int = 0
    query_bbox_fraction: float = 0.0
    reference_bbox_fraction: float = 0.0
    qualified: bool = False
    reason: str = "insufficient_matches"


@dataclass(frozen=True, slots=True)
class ApproximateReferencePose:
    x: float
    y: float
    yaw: float
    map_frame: str
    pose_kind: str = "approximate_reference_pose"
    position_source: str = "training_reference"
    heading_source: str = "training_reference"
    yaw_approximate: bool = True
    translation_scale_observed: bool = False


@dataclass(frozen=True, slots=True)
class PlaceEstimate:
    status: str
    reason: str
    timestamp_ns: int
    query_id: str
    image_sha256: str | None
    pose: ApproximateReferencePose | None
    selected_reference_id: str | None
    map_id: str
    calibration_id: str
    feature_id: str
    source_pins: tuple[tuple[str, str], ...]
    candidates: tuple[CandidateEvidence, ...]
    timings: tuple[tuple[str, float], ...]
    candidate_coverage_complete: bool

    def report(self):
        return asdict(self)


def _ideal(points, calibration):
    if not len(points):
        return np.empty((0, 2), np.float64)
    # Iterative inversion is needed for material radial distortion near image edges.
    return cv2.undistortPointsIter(np.asarray(points).reshape(-1, 1, 2), calibration.K,
        calibration.D, None, calibration.K, (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS,
                                            50, 1e-10)).reshape(-1, 2)


def _spread(points, width, height, grid):
    cells = np.floor(points/np.array([width, height])*grid).astype(int)
    cells = np.clip(cells, 0, grid-1)
    area = float(np.prod(np.ptp(points, axis=0))/(width*height)) if len(points) else 0.0
    return len({tuple(v) for v in cells}), area


def _project(points, H):
    values = np.column_stack((points, np.ones(len(points)))) @ H.T
    if np.any(np.abs(values[:, 2]) < 1e-10):
        raise ValueError("homography point at infinity")
    return values[:, :2]/values[:, 2:]


def _verify(reference_id, query, reference, count, calibration, config):
    evidence = dict(reference_id=reference_id, matches=count)
    params = cv2.UsacParams()
    params.confidence, params.maxIterations = config.ransac_confidence, config.ransac_iterations
    params.threshold, params.randomGeneratorState = config.transfer_threshold_px, config.ransac_seed
    params.isParallel, params.sampler, params.score = False, cv2.SAMPLING_UNIFORM, cv2.SCORE_METHOD_RANSAC
    params.loMethod, params.loIterations, params.final_polisher = cv2.LOCAL_OPTIM_NULL, 0, cv2.NONE_POLISHER
    H, _ = cv2.findHomography(query, reference, params)
    if H is None or not np.isfinite(H).all() or np.linalg.cond(H) > 1e12:
        return CandidateEvidence(**evidence, reason="degenerate_homography")
    try:
        forward = np.linalg.norm(_project(query, H)-reference, axis=1)
        backward = np.linalg.norm(_project(reference, np.linalg.inv(H))-query, axis=1)
    except (ValueError, np.linalg.LinAlgError):
        return CandidateEvidence(**evidence, reason="degenerate_homography")
    # Inlier definition: BOTH forward and inverse ideal-pixel transfer <= threshold.
    mask = np.isfinite(forward) & np.isfinite(backward) & (forward <= config.transfer_threshold_px) & (backward <= config.transfer_threshold_px)
    n = int(mask.sum())
    evidence.update(inliers=n, inlier_fraction=n/count)
    if n:
        qcells, qarea = _spread(query[mask], calibration.width, calibration.height, config.grid_size)
        rcells, rarea = _spread(reference[mask], calibration.width, calibration.height, config.grid_size)
        evidence.update(median_symmetric_error_px=float(np.median(np.maximum(forward[mask], backward[mask]))),
            query_grid_cells=qcells, reference_grid_cells=rcells, query_bbox_fraction=qarea, reference_bbox_fraction=rarea)
    if n < config.min_inliers or n/count < config.min_inlier_fraction:
        return CandidateEvidence(**evidence, reason="insufficient_geometric_inliers")
    if min(qcells, rcells) < config.min_grid_cells or min(qarea, rarea) < config.min_bbox_area_fraction:
        return CandidateEvidence(**evidence, reason="insufficient_spatial_support")
    return CandidateEvidence(**evidence, qualified=True, reason="verified_2d_support")


class KeyframeLocalizer:
    """Causal current-image retrieval. No query-time-to-reference association API."""
    def __init__(self, model: KeyframeMap):
        if not isinstance(model, KeyframeMap):
            raise ValueError("typed keyframe map required")
        checked = _make_map(model.calibration, model.sift_config, model.config, dict(model.provenance),
                            model.fit_intervals, model.references)
        if checked.identity != model.identity or checked.selection_sha256 != model.selection_sha256:
            raise ValueError("map identity mismatch")
        self.model = model
        self.extractor = SiftExtractor(model.calibration, model.sift_config)
        self.last_timestamp_ns = -1

    def localize(self, *, query_id: str, timestamp_ns: int, encoded_image: bytes,
                 camera_id: str, calibration_id: str):
        start, marks = time.monotonic(), []
        model, evidence = self.model, []
        stamp = validate_timestamp_ns(timestamp_ns)
        _text(query_id)
        bounded_image = type(encoded_image) is bytes and 1 <= len(encoded_image) <= model.config.max_encoded_image_bytes
        sha = _hash(encoded_image) if bounded_image else None
        def result(reason, pose=None, selected=None, complete=True):
            return PlaceEstimate("approximate_place" if pose else "rejected", reason, stamp, query_id, sha,
                pose, selected, model.identity, model.calibration.identity, model.feature_id, model.provenance,
                tuple(evidence), tuple(marks+[("total_s", time.monotonic()-start)]), complete)
        if stamp <= self.last_timestamp_ns:
            return result("noncausal_timestamp")
        self.last_timestamp_ns = stamp
        if camera_id != model.calibration.camera_id or calibration_id != model.calibration.identity:
            return result("query_camera_calibration_mismatch", complete=False)
        if not bounded_image:
            return result("invalid_encoded_image_byte_cap", complete=False)
        if any(query_id == r.pose.reference_id or sha == r.pose.image_sha256 for r in model.references):
            return result("query_reference_identity_overlap")
        try:
            image = _decode(encoded_image, model.calibration, model.config)
            query = self.extractor.extract(CameraObservation(model.calibration.camera_id, stamp, image, model.calibration.identity))
            del image
            if (query.camera_id != model.calibration.camera_id or query.calibration_id != model.calibration.identity
                    or query.feature_id != model.feature_id or query.image_width != model.calibration.width
                    or query.image_height != model.calibration.height or query.timestamp_ns != stamp
                    or query.descriptors.dtype != np.float32 or query.descriptors.shape != (query.count, 128)
                    or query.count > model.config.max_keypoints):
                return result("invalid_query_feature_contract")
            marks.append(("extract_s", time.monotonic()-start))
            if query.count == 0:
                evidence.extend(CandidateEvidence(r.pose.reference_id, 0, reason="zero_query_features") for r in model.references)
                return result("zero_query_features")
            ranked = []
            for ref in model.references:
                matches = match_descriptors(query.descriptors, ref.descriptors, model.sift_config.ratio_test)
                ranked.append((ref, matches))
            ranked.sort(key=lambda pair: (-len(pair[1].query_indices), pair[0].pose.reference_id))
            eligible = sum(len(m.query_indices) >= model.config.min_matches for _, m in ranked)
            if eligible > model.config.max_geometry_candidates:
                evidence.extend(CandidateEvidence(r.pose.reference_id, len(m.query_indices), reason="coverage_cap") for r, m in ranked)
                return result("candidate_coverage_incomplete", complete=False)
            qideal = _ideal(query.keypoints_xy, model.calibration)
            for ref, matches in ranked:
                n = len(matches.query_indices)
                if n < model.config.min_matches:
                    evidence.append(CandidateEvidence(ref.pose.reference_id, n))
                else:
                    evidence.append(_verify(ref.pose.reference_id, qideal[matches.query_indices],
                        _ideal(ref.keypoints_xy[matches.train_indices], model.calibration), n, model.calibration, model.config))
            marks.append(("matching_geometry_s", time.monotonic()-start-marks[0][1]))
            qualified = [e for e in evidence if e.qualified]
            if not qualified:
                return result("insufficient_matches" if eligible == 0 else "no_geometrically_supported_place")
            refs = {r.pose.reference_id: r.pose for r in model.references}
            for i, a in enumerate(qualified):
                for b in qualified[i+1:]:
                    pa, pb = refs[a.reference_id], refs[b.reference_id]
                    dyaw = abs(math.atan2(math.sin(pa.yaw-pb.yaw), math.cos(pa.yaw-pb.yaw)))
                    if math.hypot(pa.x-pb.x, pa.y-pb.y) > model.config.ambiguity_distance_m or dyaw > model.config.ambiguity_yaw_rad:
                        return result("ambiguous_distinct_reference_places")
            best = min(qualified, key=lambda e: (-e.inliers, -e.inlier_fraction, e.median_symmetric_error_px, e.reference_id))
            pose = refs[best.reference_id]
            return result("verified_reference_pose_copy", ApproximateReferencePose(pose.x, pose.y, pose.yaw, pose.map_frame), pose.reference_id)
        except (ValueError, cv2.error, np.linalg.LinAlgError) as exc:
            return result(f"processing_error:{type(exc).__name__}", complete=False)


def _array(array):
    raw = array.tobytes()
    return {"shape": list(array.shape), "dtype": array.dtype.str, "sha256": _hash(raw),
            "base64": base64.b64encode(raw).decode("ascii")}


def _calibration_payload(cal):
    return {name: getattr(cal, name).tolist() if isinstance(getattr(cal, name), np.ndarray) else getattr(cal, name)
            for name in cal.__dataclass_fields__}


def write_keyframe_map(path, model):
    """New path only. Returns file SHA separately from model identity."""
    checked = _make_map(model.calibration, model.sift_config, model.config, dict(model.provenance), model.fit_intervals, model.references)
    if checked.identity != model.identity or checked.selection_sha256 != model.selection_sha256:
        raise ValueError("map identity mismatch")
    size_bound = sum(4*((r.keypoints_xy.nbytes+2)//3)+4*((r.descriptors.nbytes+2)//3)+4096 for r in model.references)+16384
    if size_bound > model.config.max_artifact_bytes:
        raise ValueError("artifact conservative size cap")
    payload = {"schema": SCHEMA, "identity": model.identity, "selection_sha256": model.selection_sha256,
        "calibration": _calibration_payload(model.calibration), "sift_config": asdict(model.sift_config),
        "config": asdict(model.config), "provenance": dict(model.provenance), "fit_intervals": model.fit_intervals,
        "references": [{"pose": asdict(r.pose), "points": _array(r.keypoints_xy), "descriptors": _array(r.descriptors)} for r in model.references]}
    data = _json(payload)
    if len(data) > model.config.max_artifact_bytes:
        raise ValueError("artifact size cap")
    with Path(path).open("xb") as handle:
        handle.write(data)
    return _hash(data)


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def read_keyframe_map(path, *, expected_file_sha256, expected_map_id, expected_selection_sha256,
                      expected_calibration_id, expected_feature_id, expected_provenance):
    """No pickle. Caller-held pins are mandatory, never taken from the artifact."""
    path = Path(path)
    if not 1 <= path.stat().st_size <= MAX_ARTIFACT_BYTES:
        raise ValueError("artifact file cap")
    with path.open("rb") as handle:
        raw = handle.read(MAX_ARTIFACT_BYTES+1)
    if len(raw) > MAX_ARTIFACT_BYTES or _hash(raw) != _sha(expected_file_sha256):
        raise ValueError("artifact file SHA mismatch")
    payload = json.loads(raw, object_pairs_hook=_unique, parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON")))
    if set(payload) != {"schema", "identity", "selection_sha256", "calibration", "sift_config", "config", "provenance", "fit_intervals", "references"} or payload["schema"] != SCHEMA:
        raise ValueError("artifact schema mismatch")
    config, sift = KeyframeConfig(**payload["config"]), SiftConfig(**payload["sift_config"])
    if len(raw) > config.max_artifact_bytes or not 1 <= len(payload["references"]) <= config.max_references:
        raise ValueError("artifact count/size cap")
    cal = CameraCalibration(**payload["calibration"])
    if (cal.identity != _sha(expected_calibration_id) or sift_feature_identity(cal, sift).identity != _sha(expected_feature_id)
            or payload["provenance"] != expected_provenance):
        raise ValueError("independent identity pin mismatch")
    def array(value, dtype, columns):
        if set(value) != {"shape", "dtype", "sha256", "base64"}:
            raise ValueError("array schema")
        shape = value["shape"]
        if not isinstance(shape, list) or len(shape) != 2 or shape[1] != columns or type(shape[1]) is not int:
            raise ValueError("array shape")
        n = _int(shape[0], 0, config.max_keypoints)
        length = n*columns*np.dtype(dtype).itemsize
        if value["dtype"] != np.dtype(dtype).str or not isinstance(value["base64"], str) or len(value["base64"]) != 4*((length+2)//3):
            raise ValueError("array dtype/encoded length")
        data = base64.b64decode(value["base64"], validate=True)
        if len(data) != length or _hash(data) != _sha(value["sha256"]):
            raise ValueError("array SHA/length mismatch")
        return np.frombuffer(data, dtype=dtype).reshape(n, columns)
    refs = []
    for value in payload["references"]:
        if set(value) != {"pose", "points", "descriptors"}:
            raise ValueError("reference schema")
        refs.append(KeyframeReference(ReferencePose(**value["pose"]), array(value["points"], "<f8", 2), array(value["descriptors"], "<f4", 128)))
    model = _make_map(cal, sift, config, expected_provenance, payload["fit_intervals"], refs)
    if (model.identity != _sha(expected_map_id) or model.identity != payload["identity"]
            or model.selection_sha256 != _sha(expected_selection_sha256) or model.selection_sha256 != payload["selection_sha256"]):
        raise ValueError("map/selection pin mismatch")
    return model

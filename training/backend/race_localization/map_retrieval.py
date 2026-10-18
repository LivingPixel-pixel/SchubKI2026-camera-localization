"""Deterministic CPU SIFT features and FrozenMap place retrieval (LOC-01).

Retrieval ranks training references of a frozen map by ratio-tested descriptor
votes. It never returns coordinates or poses: a retrieved reference is only a
hypothesis that map_localization must confirm with calibrated 2D-3D geometry.

Determinism: OpenCV SIFT keypoints are put into a documented total order
before truncation, brute-force L2 matching is exhaustive, and every ranking
tie is broken by reference ID. Repeat runs on one host and build are expected
to be identical; bitwise equality across hosts with different CPU dispatch
paths is not claimed. The effective extractor and matcher settings, image
size and dependency versions are bound into the FeatureIdentity, so a map
built with different settings is rejected instead of silently matched.
This module performs no file IO, network access or global OpenCV state change.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any

import cv2
import numpy as np

from .map_contracts import (CameraCalibration, CameraObservation, FeatureIdentity, FrozenMap,
                            _immutable, canonical_identity, validate_timestamp_ns)


SIFT_EXTRACTOR = "opencv_sift"
SIFT_ADAPTER = "race_localization.map_retrieval.SiftExtractor"
SIFT_ADAPTER_VERSION = 1
SIFT_DIMENSION = 128
KEYPOINT_ORDER = "response_desc,y_asc,x_asc,size_desc,angle_asc,octave_asc"
MATCHER = "opencv_bf_l2_knn2_ratio_unique_train_min_distance"
RETRIEVAL_METHOD = "per_reference_ratio_votes_reference_position_nms"
RETRIEVAL_SCHEMA = 1


def _frozen(value: Any, dtype: Any = None) -> np.ndarray:
    """Copy with the accepted S0 mechanism (map_contracts._immutable).

    The bytes live in an immutable buffer, so the write flag cannot be set
    again, and the stored header rejects shape, dtype, strides and resize
    changes. Accessors may therefore return stored arrays directly.
    """
    return _immutable(np.asarray(value, dtype=dtype))


def _int_setting(value: Any, name: str, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or int(value) < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _float_setting(value: Any, name: str, low: float, high: float,
                   low_open: bool = False, high_open: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if (not math.isfinite(result) or result < low or result > high
            or (low_open and result == low) or (high_open and result == high)):
        raise ValueError(f"{name} must lie in {'(' if low_open else '['}{low}, {high}{')' if high_open else ']'}")
    return result


def _sha(value: Any, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdefABCDEF" for c in value):
        raise ValueError(f"{name} must be a SHA256 hex digest")
    return value.lower()


@dataclass(frozen=True, slots=True)
class SiftConfig:
    """Effective OpenCV SIFT detector, mask and matching settings.

    max_features is enforced after masking and after the deterministic sort
    (OpenCV itself runs unbounded). mask_bottom_fraction removes the lowest
    image rows (vehicle body). ratio_test is Lowe's nearest/second-nearest
    distance ratio used by every descriptor match.
    """
    max_features: int = 2000
    n_octave_layers: int = 3
    contrast_threshold: float = 0.025
    edge_threshold: float = 10.0
    sigma: float = 1.6
    enable_precise_upscale: bool = True
    mask_bottom_fraction: float = 0.06
    ratio_test: float = 0.8

    def __post_init__(self) -> None:
        if not isinstance(self.enable_precise_upscale, (bool, np.bool_)):
            raise ValueError("enable_precise_upscale must be boolean")
        values = {
            "max_features": _int_setting(self.max_features, "max_features", 1),
            "n_octave_layers": _int_setting(self.n_octave_layers, "n_octave_layers", 1),
            "contrast_threshold": _float_setting(self.contrast_threshold, "contrast_threshold", 0.0, math.inf, low_open=True, high_open=True),
            "edge_threshold": _float_setting(self.edge_threshold, "edge_threshold", 0.0, math.inf, low_open=True, high_open=True),
            "sigma": _float_setting(self.sigma, "sigma", 0.0, math.inf, low_open=True, high_open=True),
            "enable_precise_upscale": bool(self.enable_precise_upscale),
            "mask_bottom_fraction": _float_setting(self.mask_bottom_fraction, "mask_bottom_fraction", 0.0, 1.0, high_open=True),
            "ratio_test": _float_setting(self.ratio_test, "ratio_test", 0.0, 1.0, low_open=True, high_open=True),
        }
        for name, value in values.items():
            object.__setattr__(self, name, value)


def _mask_start_row(height: int, fraction: float) -> int:
    return int(height * (1.0 - fraction))


def sift_feature_identity(calibration: CameraCalibration, config: SiftConfig | None = None) -> FeatureIdentity:
    """FeatureIdentity binding every effective SIFT, image and matching setting.

    A map builder must use this exact identity for the map's calibration so
    that MapLocalizer accepts the map. Any version or setting change yields a
    different identity and therefore an explicit incompatibility.
    """
    if not isinstance(calibration, CameraCalibration):
        raise ValueError("SIFT identity requires a typed CameraCalibration")
    config = SiftConfig() if config is None else config
    if not isinstance(config, SiftConfig):
        raise ValueError("SIFT identity requires a SiftConfig")
    settings = {
        "adapter": SIFT_ADAPTER,
        "adapter_version": SIFT_ADAPTER_VERSION,
        "detector": {
            "nfeatures": 0,
            "n_octave_layers": config.n_octave_layers,
            "contrast_threshold": config.contrast_threshold,
            "edge_threshold": config.edge_threshold,
            "sigma": config.sigma,
            "enable_precise_upscale": config.enable_precise_upscale,
        },
        "keypoint_order": KEYPOINT_ORDER,
        "max_features": config.max_features,
        "budget": "after_mask_after_sort",
        "mask": {"bottom_fraction": config.mask_bottom_fraction,
                 "masked_from_row": _mask_start_row(calibration.height, config.mask_bottom_fraction)},
        "image": {"width": calibration.width, "height": calibration.height, "resize": "none", "crop": "none",
                  "grayscale": "opencv_COLOR_BGR2GRAY_or_input_gray", "dtype": "uint8",
                  "pixel_convention": calibration.pixel_convention},
        "descriptor": {"normalization": "opencv_sift_default", "root_sift": False},
        "matching": {"matcher": MATCHER, "ratio_test": config.ratio_test},
        "dependencies": {"opencv": cv2.__version__, "numpy": np.__version__},
    }
    return FeatureIdentity(
        extractor=SIFT_EXTRACTOR, version=f"opencv-{cv2.__version__}", weights_sha256="none",
        descriptor_dimension=SIFT_DIMENSION, descriptor_dtype="float32", distance_metric="l2",
        pixel_convention=calibration.pixel_convention,
        config_json=json.dumps(settings, sort_keys=True, separators=(",", ":"), allow_nan=False))


@dataclass(frozen=True, slots=True)
class QueryFeatures:
    """Keypoints and descriptors of one query exposure, without any pose.

    Pixels use the calibration's raw distorted convention (OpenCV keypoint
    coordinates, pixel centres at integers). The descriptor dtype is kept as
    supplied so that MapLocalizer can reject a dtype or dimension mismatch
    explicitly instead of casting it away.
    """
    camera_id: str
    timestamp_ns: int
    calibration_id: str
    feature_id: str
    image_width: int
    image_height: int
    keypoints_xy: np.ndarray
    descriptors: np.ndarray
    responses: np.ndarray

    def __post_init__(self) -> None:
        if not isinstance(self.camera_id, str) or not self.camera_id.strip():
            raise ValueError("Query camera ID must be a nonempty string")
        width = _int_setting(self.image_width, "image width", 1)
        height = _int_setting(self.image_height, "image height", 1)
        raw_points = np.asarray(self.keypoints_xy)
        if raw_points.ndim != 2 or raw_points.shape[1] != 2 or (raw_points.size and raw_points.dtype.kind not in "iuf"):
            raise ValueError("Query keypoints must have shape (N, 2)")
        points = np.asarray(raw_points, dtype=np.float64)
        if not np.isfinite(points).all() or np.any(points < 0) or np.any(points[:, 0] >= width) or np.any(points[:, 1] >= height):
            raise ValueError("Query keypoints must be finite pixels inside the image")
        raw_descriptors = np.asarray(self.descriptors)
        if (raw_descriptors.ndim != 2 or raw_descriptors.shape[0] != len(points) or raw_descriptors.shape[1] < 1
                or raw_descriptors.dtype.kind not in "iuf" or not np.isfinite(raw_descriptors).all()):
            raise ValueError("Query descriptors must be a finite numeric (N, D) array")
        raw_responses = np.asarray(self.responses)
        if raw_responses.shape != (len(points),) or (raw_responses.size and raw_responses.dtype.kind not in "iuf"):
            raise ValueError("Query responses must have shape (N,)")
        responses = np.asarray(raw_responses, dtype=np.float64)
        if not np.isfinite(responses).all():
            raise ValueError("Query responses must be finite")
        for name, value in (("camera_id", self.camera_id), ("image_width", width), ("image_height", height),
                            ("timestamp_ns", validate_timestamp_ns(self.timestamp_ns)),
                            ("calibration_id", _sha(self.calibration_id, "calibration ID")),
                            ("feature_id", _sha(self.feature_id, "feature ID")),
                            ("keypoints_xy", _frozen(points)), ("descriptors", _frozen(raw_descriptors)),
                            ("responses", _frozen(responses))):
            object.__setattr__(self, name, value)

    @property
    def count(self) -> int:
        return int(len(self.keypoints_xy))


class SiftExtractor:
    """Deterministic OpenCV SIFT on full-resolution raw images of one calibration."""

    def __init__(self, calibration: CameraCalibration, config: SiftConfig | None = None):
        self.config = SiftConfig() if config is None else config
        self.identity = sift_feature_identity(calibration, self.config)
        self.calibration = calibration
        self.feature_id = self.identity.identity
        try:
            # nfeatures=0: OpenCV applies its own budget before the mask, so
            # the documented max_features budget is enforced after masking
            # and the deterministic sort instead.
            self._sift = cv2.SIFT_create(
                nfeatures=0, nOctaveLayers=self.config.n_octave_layers,
                contrastThreshold=self.config.contrast_threshold, edgeThreshold=self.config.edge_threshold,
                sigma=self.config.sigma, enable_precise_upscale=self.config.enable_precise_upscale)
        except (cv2.error, TypeError) as exc:
            raise ValueError("Installed OpenCV cannot construct the declared SIFT configuration") from exc
        mask = np.full((calibration.height, calibration.width), 255, dtype=np.uint8)
        mask[_mask_start_row(calibration.height, self.config.mask_bottom_fraction):] = 0
        self._mask = _frozen(mask)

    def extract(self, observation: CameraObservation) -> QueryFeatures:
        if not isinstance(observation, CameraObservation):
            raise ValueError("SIFT extraction requires a CameraObservation")
        calibration = self.calibration
        if observation.camera_id != calibration.camera_id or observation.calibration_id != calibration.identity:
            raise ValueError("Observation camera or calibration identity differs from the extractor calibration")
        image = np.array(observation.image, copy=True)
        if image.shape[:2] != (calibration.height, calibration.width):
            raise ValueError("Observation image size differs from the calibrated size; resizing is not declared")
        gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self._sift.detectAndCompute(gray, self._mask)
        return self._package(observation, keypoints, descriptors)

    def _package(self, observation: CameraObservation, keypoints: Any, descriptors: Any) -> QueryFeatures:
        width, height = self.calibration.width, self.calibration.height
        keypoints = tuple(keypoints or ())
        if descriptors is None or not keypoints:
            points = np.empty((0, 2), dtype=np.float64)
            values = np.empty((0, SIFT_DIMENSION), dtype=np.float32)
            responses = np.empty(0, dtype=np.float64)
        else:
            values = np.asarray(descriptors, dtype=np.float32)
            if values.shape != (len(keypoints), SIFT_DIMENSION):
                raise ValueError("OpenCV SIFT returned descriptors inconsistent with its keypoints")
            # OpenCV may emit keypoints in a thread-dependent order. The total
            # order below makes the selected set and its order reproducible.
            order = sorted(range(len(keypoints)), key=lambda index: (
                -float(keypoints[index].response), float(keypoints[index].pt[1]), float(keypoints[index].pt[0]),
                -float(keypoints[index].size), float(keypoints[index].angle), int(keypoints[index].octave)))
            points = np.array([keypoints[index].pt for index in order], dtype=np.float64).reshape(-1, 2)
            responses = np.array([keypoints[index].response for index in order], dtype=np.float64)
            values = values[np.asarray(order, dtype=np.int64)]
            inside = (np.isfinite(points).all(axis=1) & (points[:, 0] >= 0) & (points[:, 0] < width)
                      & (points[:, 1] >= 0) & (points[:, 1] < height))
            keep = np.flatnonzero(inside)[:self.config.max_features]
            points, responses, values = points[keep], responses[keep], values[keep]
        return QueryFeatures(camera_id=observation.camera_id, timestamp_ns=observation.timestamp_ns,
                             calibration_id=observation.calibration_id, feature_id=self.feature_id,
                             image_width=width, image_height=height, keypoints_xy=points,
                             descriptors=values, responses=responses)


@dataclass(frozen=True, slots=True)
class DescriptorMatches:
    """Ratio-tested matches; each train row appears at most once."""
    query_indices: np.ndarray
    train_indices: np.ndarray
    distances: np.ndarray

    def __post_init__(self) -> None:
        query, train = np.asarray(self.query_indices), np.asarray(self.train_indices)
        distances = np.asarray(self.distances)
        if (query.ndim != 1 or query.shape != train.shape or distances.shape != query.shape
                or (query.size and (query.dtype.kind not in "iu" or train.dtype.kind not in "iu"))):
            raise ValueError("Descriptor matches need equal-length integer indices and distances")
        object.__setattr__(self, "query_indices", _frozen(query, np.int64))
        object.__setattr__(self, "train_indices", _frozen(train, np.int64))
        object.__setattr__(self, "distances", _frozen(distances, np.float64))

    def __len__(self) -> int:
        return int(len(self.query_indices))


def _empty_matches() -> DescriptorMatches:
    return DescriptorMatches(_frozen(np.empty(0, np.int64)), _frozen(np.empty(0, np.int64)),
                             _frozen(np.empty(0, np.float64)))


def match_descriptors(query: np.ndarray, train: np.ndarray, ratio_test: float) -> DescriptorMatches:
    """Exhaustive L2 2-NN matching with Lowe's ratio test and unique train rows.

    A match is kept only if its best distance is strictly below ratio_test
    times the second-best distance within the same train set. When several
    queries keep the same train row, only the smallest distance survives
    (ties broken by the lower query index). Results are ordered by query index.
    """
    ratio = _float_setting(ratio_test, "ratio_test", 0.0, 1.0, low_open=True, high_open=True)
    query_values, train_values = np.asarray(query), np.asarray(train)
    if (query_values.ndim != 2 or train_values.ndim != 2 or query_values.dtype != np.float32
            or train_values.dtype != np.float32 or query_values.shape[1] != train_values.shape[1]):
        raise ValueError("L2 matching requires float32 (N, D) descriptors of equal dimension")
    if len(query_values) == 0 or len(train_values) < 2:
        return _empty_matches()
    pairs = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False).knnMatch(
        np.ascontiguousarray(query_values), np.ascontiguousarray(train_values), k=2)
    query_indices, train_indices, distances = [], [], []
    for neighbors in pairs:
        if len(neighbors) < 2:
            continue
        best, second = neighbors[0], neighbors[1]
        if best.distance < ratio * second.distance:
            query_indices.append(best.queryIdx)
            train_indices.append(best.trainIdx)
            distances.append(best.distance)
    if not query_indices:
        return _empty_matches()
    q = np.asarray(query_indices, dtype=np.int64)
    t = np.asarray(train_indices, dtype=np.int64)
    d = np.asarray(distances, dtype=np.float64)
    order = np.lexsort((q, d))
    _, first = np.unique(t[order], return_index=True)
    keep = order[first]
    keep = keep[np.argsort(q[keep], kind="stable")]
    return DescriptorMatches(_frozen(q[keep]), _frozen(t[keep]), _frozen(d[keep]))


@dataclass(frozen=True, slots=True)
class RetrievalConfig:
    """min_votes gates which places are verified; region_radius_m merges
    nearby training references into one verification region."""
    min_votes: int = 12
    region_radius_m: float = 10.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "min_votes", _int_setting(self.min_votes, "min_votes", 1))
        object.__setattr__(self, "region_radius_m", _float_setting(self.region_radius_m, "region_radius_m", 0.0, math.inf, high_open=True))


@dataclass(frozen=True, slots=True)
class RetrievalCandidate:
    """A ranked place hypothesis. Deliberately carries no coordinates."""
    rank: int
    reference_id: str
    region_id: str
    votes: int
    vote_fraction: float
    suppressed_by: str


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    """ranked lists every reference; regions are unsuppressed seeds with at
    least min_votes, in rank order. Neither contains a pose."""
    query_keypoints: int
    ranked: tuple[RetrievalCandidate, ...]
    regions: tuple[RetrievalCandidate, ...]
    index_identity: str


def region_id_for(reference_id: str) -> str:
    return f"ref:{reference_id}"


class RetrievalIndex:
    """In-memory place index derived deterministically from a FrozenMap.

    Each reference votes with the query descriptors that pass the ratio test
    against the landmarks it observes. A verification region is a seed
    reference plus every reference whose training body position lies within
    region_radius_m; its landmark set is the union of their observations.
    Training reference positions are permitted map assets and are used only
    for this grouping and suppression; they are never returned.

    identity is a derived identity over the algorithm, settings and hashed
    contents (landmark IDs, descriptors, per-reference observations,
    positions and region membership). It is a different hash domain from the
    map's declared retrieval_index_sha256 and is never compared with it.
    Cost is linear in the number of references; LOC-02 must measure it.
    """

    def __init__(self, frozen_map: FrozenMap, config: RetrievalConfig | None = None, ratio_test: float = 0.8):
        if not isinstance(frozen_map, FrozenMap):
            raise ValueError("Retrieval requires a FrozenMap")
        self.config = RetrievalConfig() if config is None else config
        if not isinstance(self.config, RetrievalConfig):
            raise ValueError("Retrieval requires a RetrievalConfig")
        if frozen_map.feature_identity.descriptor_dtype != "float32" or frozen_map.feature_identity.distance_metric != "l2":
            raise ValueError("This retrieval index supports float32 L2 descriptors only")
        self.ratio_test = _float_setting(ratio_test, "ratio_test", 0.0, 1.0, low_open=True, high_open=True)
        self._map = frozen_map
        self.feature_id = frozen_map.feature_identity.identity
        row_of = {int(value): row for row, value in enumerate(frozen_map.landmark_ids.tolist())}
        observed = {item.reference_id: item for item in frozen_map.reference_observations}
        references = sorted(frozen_map.references, key=lambda item: item.reference_id)
        self._reference_ids = tuple(item.reference_id for item in references)
        self._positions = _frozen(np.array([np.asarray(item.T_WB)[:3, 3] for item in references],
                                           dtype=np.float64).reshape(-1, 3))
        self._reference_rows: dict[str, np.ndarray] = {}
        for reference_id in self._reference_ids:
            observation = observed.get(reference_id)
            rows = (np.unique(np.array([row_of[int(value)] for value in observation.landmark_ids.tolist()], dtype=np.int64))
                    if observation is not None and len(observation.landmark_ids) else np.empty(0, np.int64))
            self._reference_rows[reference_id] = _frozen(rows)
        descriptors = np.asarray(frozen_map.descriptors)
        self._reference_descriptors = {reference_id: _frozen(descriptors[rows], np.float32)
                                       for reference_id, rows in self._reference_rows.items()}
        self._region_rows: dict[str, np.ndarray] = {}
        self._region_descriptors: dict[str, np.ndarray] = {}
        for index, reference_id in enumerate(self._reference_ids):
            distance = np.linalg.norm(self._positions - self._positions[index], axis=1)
            members = [self._reference_ids[j] for j in np.flatnonzero(distance <= self.config.region_radius_m)]
            rows = np.unique(np.concatenate([np.asarray(self._reference_rows[member]) for member in members]
                                            + [np.empty(0, np.int64)]))
            region = region_id_for(reference_id)
            self._region_rows[region] = _frozen(rows.astype(np.int64))
            self._region_descriptors[region] = _frozen(descriptors[rows.astype(np.int64)], np.float32)
        digest = hashlib.sha256()

        def feed(name: str, array: np.ndarray) -> None:
            value = np.ascontiguousarray(array)
            digest.update(f"{name}|{value.dtype.str}|{value.shape}|".encode("utf-8"))
            digest.update(value.tobytes())

        feed("landmark_ids", np.asarray(frozen_map.landmark_ids))
        feed("descriptors", descriptors)
        for index, reference_id in enumerate(self._reference_ids):
            digest.update(f"reference|{len(reference_id)}|{reference_id}|".encode("utf-8"))
            feed("reference_rows", self._reference_rows[reference_id])
            feed("reference_position", self._positions[index])
            feed("region_rows", self._region_rows[region_id_for(reference_id)])
        # Derived from the in-memory map contents actually used for ranking.
        # This is not FrozenMap.retrieval_index_sha256, which declares the
        # checksum of a serialized artifact that a loader must verify itself.
        self.content_sha256 = digest.hexdigest()
        self.identity = canonical_identity({
            "schema": RETRIEVAL_SCHEMA, "method": RETRIEVAL_METHOD, "matcher": MATCHER,
            "feature_id": self.feature_id, "ratio_test": self.ratio_test,
            "config": {"min_votes": self.config.min_votes, "region_radius_m": self.config.region_radius_m},
            "content_sha256": self.content_sha256, "declared_map_hash": frozen_map.map_hash,
        })

    @property
    def reference_count(self) -> int:
        return len(self._reference_ids)

    def region_rows(self, region_id: str) -> np.ndarray:
        """Landmark row indices (into the FrozenMap arrays) for verification."""
        if region_id not in self._region_rows:
            raise ValueError(f"Unknown retrieval region {region_id!r}")
        return self._region_rows[region_id]

    def region_descriptors(self, region_id: str) -> np.ndarray:
        if region_id not in self._region_descriptors:
            raise ValueError(f"Unknown retrieval region {region_id!r}")
        return self._region_descriptors[region_id]

    def rank(self, features: QueryFeatures) -> RetrievalResult:
        if not isinstance(features, QueryFeatures):
            raise ValueError("Retrieval requires QueryFeatures")
        if features.feature_id != self.feature_id:
            raise ValueError("Query feature identity differs from the map feature identity")
        query = np.asarray(features.descriptors)
        if query.dtype != np.float32 or query.shape[1] != self._map.feature_identity.descriptor_dimension:
            raise ValueError("Query descriptor dtype or dimension differs from the map")
        votes = []
        for reference_id in self._reference_ids:
            train = self._reference_descriptors[reference_id]
            votes.append(len(match_descriptors(query, train, self.ratio_test)) if len(train) >= 2 else 0)
        order = sorted(range(len(votes)), key=lambda index: (-votes[index], self._reference_ids[index]))
        seeds: list[int] = []
        ranked, regions = [], []
        denominator = max(features.count, 1)
        for rank, index in enumerate(order):
            suppressed = ""
            if votes[index] > 0:
                for seed in seeds:
                    if float(np.linalg.norm(self._positions[index] - self._positions[seed])) <= self.config.region_radius_m:
                        suppressed = region_id_for(self._reference_ids[seed])
                        break
                if not suppressed:
                    seeds.append(index)
            reference_id = self._reference_ids[index]
            candidate = RetrievalCandidate(rank=rank, reference_id=reference_id, region_id=region_id_for(reference_id),
                                           votes=int(votes[index]), vote_fraction=float(votes[index]) / denominator,
                                           suppressed_by=suppressed)
            ranked.append(candidate)
            if votes[index] >= self.config.min_votes and not suppressed:
                regions.append(candidate)
        return RetrievalResult(query_keypoints=features.count, ranked=tuple(ranked), regions=tuple(regions),
                               index_identity=self.identity)

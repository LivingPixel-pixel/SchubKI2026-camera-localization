"""Geometrically verified absolute body-pose fixes against a FrozenMap (LOC-01).

Pipeline for one query exposure, synchronous and deterministic on CPU:

1. Identity gates: camera, calibration identity, image size, feature identity
   and descriptor dtype/dimension must equal the frozen map. Mismatches are
   explicit rejections, never silent conversions.
2. SIFT extraction (map_retrieval.SiftExtractor) and place retrieval
   (RetrievalIndex). Retrieval only ranks references; it emits no pose.
3. For each of the top retrieved regions: ratio-tested 2D-3D matches, exact
   duplicate removal, pre-PnP shape and support gates, deterministic AP3P
   RANSAC (own seeded NumPy generator, cheirality-aware scoring) on raw
   distorted pixels with the calibrated K and D, iterative LM refinement with
   inlier reselection, then post-PnP gates.
4. Hypothesis decision: accept only if no competing pose hypothesis at a
   distinct pose has comparable support. Hypotheses are second models inside
   each verified correspondence union, the estimates of other verified
   regions (accepted or not), verified regions whose competing pose is
   unknown or not exported (bounded by their correspondences) and
   unverified regions with comparable votes.
   Comparable support is comparable_support(best) =
   max(4, ceil(ambiguity_inlier_ratio * best)), the plain ratio rule with
   the four points a pose needs. Distinct means more than
   ambiguity_distance_m or ambiguity_rotation_rad apart in body pose.

Body pose uses the full mount: T_WB = inverse(T_CW) @ inverse(T_BC).

Pre-PnP rejection criteria, in evaluation order (the first failure stops
evaluation and is the reason):

* insufficient_features: fewer query keypoints than min_query_features.
* no_retrieval_candidate: no unsuppressed reference has min_votes.
* insufficient_correspondences: fewer unique 2D-3D pairs than
  min_correspondences after removing exact duplicates (same pixel, same
  landmark ID or same 3D point). Duplicates never count as support.
* degenerate_geometry (pre-PnP): the landmark set is collinear or coincident,
  i.e. second/first singular value of centred 3D points < min_extent_ratio.
  Coplanar sets are not rejected here.
* insufficient_spatial_support (pre-PnP): matched pixels cover a convex hull
  below min_hull_area_fraction of the image or fewer than min_support_cells
  cells of a support_grid x support_grid image grid. This is the
  concentrated-support gate and is independent of 3D shape.
* pnp_failed: RANSAC or refinement produced no finite pose.

Post-PnP checks, all evaluated and listed, the first failing one in this
order becomes the reason:

* insufficient_inliers / low_inlier_ratio: inliers are correspondences with
  reprojection error <= ransac_reprojection_px and depth in
  (min_depth_m, max_depth_m).
* nonpositive_depth: correspondences that reproject within the RANSAC
  threshold but lie at depth <= min_depth_m exceed max_behind_fraction of
  the inliers (cheirality violation).
* high_reprojection_error: inlier RMSE above max_inlier_rmse_px.
* insufficient_spatial_support (post-PnP): the same image-support gate on
  inlier pixels only.
* degenerate_geometry (post-PnP): pose_conditioning of the inliers below
  min_conditioning, or collinear inliers. This is the 6-DoF observability
  gate; it catches near-collinear support that passes the shape gate.
* planar_pose_ambiguity: run only when no earlier check failed and the
  inliers are near-planar (third/first singular value < planar_ratio).
  The second planar solution is seeded by mirroring
  the plane normal about the line of sight and LM-refined. Reject if it is a
  distinct pose (rotation > ambiguity_rotation_rad or camera centre farther
  than ambiguity_distance_m) and its RMSE is within
  planar_ambiguity_rmse_factor * best RMSE + planar_ambiguity_margin_px.
  Planarity alone is not a rejection: well-perspective planes are accepted.
  A finite alternative with landmarks at depth <= min_depth_m is physically
  impossible and is ruled out (alternative_invalid). The alternative pose
  itself is not exported (only its RMSE and rotation diagnostics are), and
  the secondary search is skipped, so the cross-region decision treats this
  region as known unexported evidence (see below).
* planar_ambiguity_unresolved: the alternative could not be evaluated
  (OpenCV error, nonfinite seed, refinement or projection). The ambiguity is
  then unknown, so the check fails closed instead of accepting.
  The mirrored-seed construction is a heuristic: it is not a proof that no
  other planar minimum exists. Small distant planes are normally rejected
  earlier by the spatial-support gate.
* ambiguous_pose_hypotheses / unresolved_pose_hypotheses: run only when no
  earlier check failed. Retrieval may merge correspondences of different
  places into one region, because nearby reference positions do not prove
  that the query pose is unique, and two places can share support. A
  competing model is a pose distinct from the primary with comparable
  support on all correspondences; shared support counts and no exclusive
  support is required. A seeded AP3P search samples all unique
  correspondences. Every finite raw branch that is distinct from the primary
  and has at least half the comparable support is refined on its total
  support with quality-checked LM updates (an update must not increase the
  cost on the support it was computed from) and rescored on all
  correspondences. A verified proposal that converges to the primary
  resolves only itself and the search continues. A verified distinct one
  with comparable support is a competitor (ambiguous). A proposal whose
  refinement is unavailable or degrading keeps its raw comparable evidence,
  which makes the result unresolved unless a verified competitor is found.
  Nothing is deduplicated. The search draws exactly the samples needed to
  hit a comparable support set at ransac_confidence. A sample cap
  shortfall, raising minimal solves, unverified comparable evidence, or
  reaching secondary_refinement_limit with proposals left make the result
  unresolved (fail closed). Caps are work budgets, not completeness proofs;
  diagnostics report samples and refinements. The default cap
  (secondary_ransac_iterations=None) is secondary_iteration_bound(config):
  the worst case over every frame that passes min_inliers and
  min_inlier_ratio, about 8000 samples with defaults and typically a few
  hundred. LOC-02 must measure this cost.
Cross-region decision, after the best geometrically valid region is chosen
(most inliers, then better retrieval rank). Every verified region
contributes its exported poses to category 1; its own reason then places it
in exactly one of categories 2 to 4. Category 5 covers retrieval regions
that were never verified.

1. Exported hypotheses, with a body pose and a support count measured on
   that region's unique correspondences: every other region's primary
   estimate (whether or not it passed its own gates) and every region's
   secondary model when its search produced one. These are compared
   directly with the best pose.
2. Unrepresented evidence, UNREPRESENTED_REASONS: the region's own result
   says a competing pose may exist (2a) or does exist (2b), but that pose
   is not exported, so whether it is distinct from the global best is
   unknown.
   a. Explicit unknowns, EXPLICIT_UNKNOWN_REASONS: unresolved_pose_hypotheses
      (secondary search incomplete or its comparable evidence unverified)
      and planar_ambiguity_unresolved (the planar alternative could not be
      evaluated).
   b. Known unexported alternatives, KNOWN_UNEXPORTED_REASONS:
      planar_pose_ambiguity. The region found a distinct planar alternative
      with comparable RMSE but exports only scalar diagnostics.
   Any model of a region is supported by at most its unique
   correspondences, so the region is relevant only if correspondences >=
   comparable_support(best inliers); below that bound it provably cannot
   hold a comparable competitor and is ignored. No pose or support is
   invented for it.
3. Searched outcomes: an accepted region (reason "") and
   ambiguous_pose_hypotheses passed the planar check (no alternative left
   unresolved or comparable) and completed the secondary search, and they
   export the primary and the second model that search reported, so every
   pose they report is in category 1. The search stops at the first
   competitor; as stated above, its caps are budgets, not completeness
   proofs.
4. Plain gate failures, every remaining regional reason: only their
   exported primary, if any, counts in category 1 and they never veto on
   their own. Their planar and secondary checks did not run; that
   limitation is disclosed, not hidden.
5. Unverified retrieval regions beyond max_verified_regions: never tried
   (latency bound), represented only by retrieval votes.

* ambiguous_distant_regions: a category 1 hypothesis is distinct from the
  best and has comparable support. A nearly verified distant place
  therefore blocks a fix instead of being ignored. This known ambiguity
  takes precedence over every unknown below.
* unresolved_competing_region: a relevant category 2 region exists (its
  hidden alternative could compete even if its primary coincides with the
  best pose), or a category 5 region has comparable votes relative to the
  best region's votes. The diagnostics name that region as competitor with
  an upper-bound ratio (correspondences or votes over the best) and no
  separation or rotation, because no competing pose is known.

All numeric defaults are explicit engineering settings for synthetic checks,
not tuned racing values. LOC-02 must select them on tuning laps 4-5 only.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
import functools
import math
import time
from typing import Any, Callable

import cv2
import numpy as np

from .map_contracts import (CameraCalibration, CameraObservation, FixCandidate, FrozenMap, _immutable,
                            body_pose_from_pnp, canonical_identity, invert_transform, require_compatible,
                            validate_timestamp_ns, validate_transform)
from .map_retrieval import (QueryFeatures, RetrievalCandidate, RetrievalConfig, RetrievalIndex, RetrievalResult,
                            SiftConfig, SiftExtractor, match_descriptors)


LOCALIZER_SCHEMA = 1
PNP_METHOD = "numpy_seeded_ransac_opencv_solveP3P_AP3P_depth_scored_then_solvePnPRefineLM"
ERROR_CAP_PX = 1.0e6
SECONDARY_SEARCH_LIMIT = 100_000

REASONS = (
    "camera_mismatch", "calibration_mismatch", "image_size_mismatch", "feature_identity_mismatch",
    "descriptor_mismatch", "insufficient_features", "no_retrieval_candidate", "insufficient_correspondences",
    "degenerate_geometry", "insufficient_spatial_support", "pnp_failed", "insufficient_inliers",
    "low_inlier_ratio", "nonpositive_depth", "high_reprojection_error", "planar_pose_ambiguity",
    "planar_ambiguity_unresolved", "ambiguous_pose_hypotheses", "unresolved_pose_hypotheses",
    "ambiguous_distant_regions", "unresolved_competing_region",
)
PLANAR_CHECKS = ("not_run", "not_planar", "fronto_parallel", "alternative_invalid", "alternative_unavailable",
                 "alternative_converged", "alternative_worse", "ambiguous")
SECONDARY_CHECKS = ("not_run", "no_model", "not_distinct", "weaker", "unresolved", "ambiguous")


def _int_setting(value: Any, name: str, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or int(value) < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _float_setting(value: Any, name: str, low: float, high: float = math.inf, low_open: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or result < low or result > high or (low_open and result == low):
        raise ValueError(f"{name} must lie in {'(' if low_open else '['}{low}, {high}]")
    return result


@dataclass(frozen=True, slots=True)
class LocalizationConfig:
    min_query_features: int = 12
    min_correspondences: int = 12
    min_inliers: int = 12
    min_inlier_ratio: float = 0.25
    ransac_reprojection_px: float = 4.0
    ransac_iterations: int = 1000
    ransac_min_iterations: int = 50
    ransac_confidence: float = 0.9999
    rng_seed: int = 31415
    refinement_rounds: int = 3
    max_inlier_rmse_px: float = 2.0
    min_depth_m: float = 0.2
    max_depth_m: float = 2000.0
    max_behind_fraction: float = 0.1
    min_hull_area_fraction: float = 0.02
    support_grid: int = 4
    min_support_cells: int = 4
    min_extent_ratio: float = 1.0e-3
    min_conditioning: float = 1.0e-3
    planar_ratio: float = 0.05
    planar_ambiguity_rmse_factor: float = 1.5
    planar_ambiguity_margin_px: float = 0.5
    max_verified_regions: int = 4
    # None derives the cap from the other settings (secondary_iteration_bound).
    secondary_ransac_iterations: int | None = None
    secondary_refinement_limit: int = 1024
    ambiguity_inlier_ratio: float = 0.5
    ambiguity_distance_m: float = 2.0
    ambiguity_rotation_rad: float = 0.2
    duplicate_pixel_px: float = 1.0e-3
    duplicate_point_m: float = 1.0e-6

    def __post_init__(self) -> None:
        integers = {"min_query_features": 1, "min_correspondences": 6, "min_inliers": 4, "ransac_iterations": 1,
                    "ransac_min_iterations": 1, "rng_seed": 0, "refinement_rounds": 1, "support_grid": 1,
                    "min_support_cells": 1, "max_verified_regions": 1, "secondary_refinement_limit": 1}
        for name, minimum in integers.items():
            object.__setattr__(self, name, _int_setting(getattr(self, name), name, minimum))
        if self.secondary_ransac_iterations is not None:
            object.__setattr__(self, "secondary_ransac_iterations",
                               _int_setting(self.secondary_ransac_iterations, "secondary_ransac_iterations", 1))
        if self.min_support_cells > self.support_grid ** 2:
            raise ValueError("min_support_cells cannot exceed the number of support grid cells")
        if self.ransac_min_iterations > self.ransac_iterations:
            raise ValueError("ransac_min_iterations cannot exceed ransac_iterations")
        unit = {"min_inlier_ratio": (0.0, 1.0, False), "ransac_confidence": (0.0, 1.0, True),
                "max_behind_fraction": (0.0, 1.0, False), "min_hull_area_fraction": (0.0, 1.0, False),
                "min_extent_ratio": (0.0, 1.0, False), "min_conditioning": (0.0, 1.0, False),
                "planar_ratio": (0.0, 1.0, False), "ambiguity_inlier_ratio": (0.0, 1.0, True)}
        for name, (low, high, low_open) in unit.items():
            object.__setattr__(self, name, _float_setting(getattr(self, name), name, low, high, low_open))
        positive = ("ransac_reprojection_px", "max_inlier_rmse_px", "min_depth_m", "max_depth_m",
                    "ambiguity_distance_m", "ambiguity_rotation_rad", "duplicate_pixel_px", "duplicate_point_m")
        for name in positive:
            object.__setattr__(self, name, _float_setting(getattr(self, name), name, 0.0, low_open=True))
        for name in ("planar_ambiguity_rmse_factor", "planar_ambiguity_margin_px"):
            object.__setattr__(self, name, _float_setting(getattr(self, name), name, 0.0))
        if self.max_depth_m <= self.min_depth_m:
            raise ValueError("max_depth_m must exceed min_depth_m")

    def as_dict(self) -> dict[str, Any]:
        return {item.name: getattr(self, item.name) for item in fields(self)}


@dataclass(frozen=True, slots=True)
class RegionAttempt:
    """Scalar geometric diagnostics for one verified retrieval region."""
    region_id: str
    reference_id: str
    retrieval_rank: int
    retrieval_votes: int
    candidate_landmarks: int
    tentative_matches: int
    duplicates_removed: int
    correspondences: int
    inliers: int
    inlier_rmse_px: float | None
    hull_area_fraction: float
    support_cells: int
    extent_ratio: float
    planarity_ratio: float
    conditioning: float
    behind_low_error: int
    planar_check: str
    planar_alternative_rmse_px: float | None
    planar_alternative_rotation_rad: float | None
    secondary_check: str
    secondary_inliers: int
    secondary_exclusive_inliers: int
    secondary_separation_m: float | None
    secondary_rotation_rad: float | None
    secondary_samples: int
    secondary_refinements: int
    reason: str
    failed_checks: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class FixDiagnostics:
    """Per-attempt diagnostics. competitor_region_id names the strongest
    distinct competing hypothesis: a retrieval region ID, or that ID with the
    suffix "#secondary" for a second model found inside that region. The
    suffixed form is a label, not a retrieval region."""
    reason: str
    query_keypoints: int
    retrieval_references: int
    retrieval_regions: tuple[RetrievalCandidate, ...]
    attempts: tuple[RegionAttempt, ...]
    competitor_region_id: str
    competitor_ratio: float
    competitor_separation_m: float | None
    competitor_rotation_rad: float | None
    started_monotonic_ns: int
    extraction_ns: int
    retrieval_ns: int
    geometry_ns: int
    completed_monotonic_ns: int
    localizer_identity: str
    derived_retrieval_index_identity: str
    declared_retrieval_index_sha256: str


@dataclass(frozen=True, slots=True)
class LocalizationAttempt:
    fix: FixCandidate
    diagnostics: FixDiagnostics


@dataclass(frozen=True, slots=True)
class PoseVerification:
    """Result of geometric verification of one 2D-3D correspondence set.

    T_CW/T_WB are None unless every gate passed. estimated_T_WB is the body
    pose estimate whenever PnP produced one, including rejected estimates;
    secondary_estimated_T_WB is the strongest second model distinct from the
    primary, scored on all correspondences. Both exist only so that competing hypotheses
    can block a fix; neither is ever emitted as a fix. The planar
    alternative pose is not exported; only planar_check and its RMSE and
    rotation diagnostics are. landmark_ids,
    inlier_mask and reprojection_error_px are empty when no pose estimate was
    produced.
    """
    status: str
    reason: str
    failed_checks: tuple[str, ...]
    T_CW: np.ndarray | None
    T_WB: np.ndarray | None
    estimated_T_WB: np.ndarray | None
    landmark_ids: np.ndarray
    inlier_mask: np.ndarray
    reprojection_error_px: np.ndarray
    input_correspondences: int
    duplicates_removed: int
    correspondences: int
    inliers: int
    inlier_rmse_px: float | None
    hull_area_fraction: float
    support_cells: int
    extent_ratio: float
    planarity_ratio: float
    conditioning: float
    behind_low_error: int
    planar_check: str
    planar_alternative_rmse_px: float | None
    planar_alternative_rotation_rad: float | None
    secondary_check: str
    secondary_inliers: int
    secondary_exclusive_inliers: int
    secondary_separation_m: float | None
    secondary_rotation_rad: float | None
    secondary_estimated_T_WB: np.ndarray | None
    secondary_samples: int
    secondary_refinements: int


def _frozen(value: Any, dtype: Any = None) -> np.ndarray:
    """Accepted S0 immutable backing and protected header (map_contracts._immutable)."""
    return _immutable(np.asarray(value, dtype=dtype))


def _rotation_angle(first: np.ndarray, second: np.ndarray) -> float:
    cosine = (float(np.trace(np.asarray(first).T @ np.asarray(second))) - 1.0) / 2.0
    return float(math.acos(min(1.0, max(-1.0, cosine))))


def _distortion(calibration: CameraCalibration) -> np.ndarray | None:
    values = np.array(calibration.D, dtype=np.float64)
    return values if values.size else None


def _shape_ratios(xyz: np.ndarray) -> tuple[float, float]:
    """(second/first, third/first) singular values of centred 3D points."""
    if len(xyz) < 2:
        return 0.0, 0.0
    singular = np.linalg.svd(xyz - xyz.mean(axis=0), compute_uv=False)
    singular = np.pad(singular, (0, 3 - len(singular)))
    if not np.isfinite(singular).all() or singular[0] <= 1e-12:
        return 0.0, 0.0
    return float(singular[1] / singular[0]), float(singular[2] / singular[0])


def spatial_support(pixels_xy: np.ndarray, calibration: CameraCalibration, grid: int) -> tuple[float, int]:
    """Convex-hull area fraction of the image and occupied grid-cell count."""
    pixels = np.asarray(pixels_xy, dtype=np.float64).reshape(-1, 2)
    if not len(pixels):
        return 0.0, 0
    area = 0.0
    if len(pixels) >= 3:
        hull = cv2.convexHull(np.ascontiguousarray(pixels, dtype=np.float32).reshape(-1, 1, 2))
        area = float(cv2.contourArea(hull))
    columns = np.clip(np.floor(pixels[:, 0] * grid / calibration.width).astype(np.int64), 0, grid - 1)
    rows = np.clip(np.floor(pixels[:, 1] * grid / calibration.height).astype(np.int64), 0, grid - 1)
    cells = int(len(np.unique(rows * grid + columns)))
    return area / float(calibration.width * calibration.height), cells


def pose_conditioning(landmarks_xyz: np.ndarray, T_CW: np.ndarray, calibration: CameraCalibration) -> float:
    """Inverse condition number of the 6-DoF reprojection Jacobian.

    The Jacobian is taken with respect to a left camera-frame perturbation
    (rotation in radians, translation scaled by the median depth so both
    blocks have comparable pixel units). Collinear support has an unobservable
    rotation about its line and returns approximately zero.
    """
    transform = validate_transform(T_CW, "T_CW")
    xyz = np.asarray(landmarks_xyz, dtype=np.float64).reshape(-1, 3)
    if len(xyz) < 3:
        return 0.0
    camera = xyz @ transform[:3, :3].T + transform[:3, 3]
    if np.any(camera[:, 2] <= 0):
        return 0.0
    _, jacobian = cv2.projectPoints(np.ascontiguousarray(camera), np.zeros(3), np.zeros(3),
                                    np.array(calibration.K, dtype=np.float64), _distortion(calibration))
    matrix = np.array(jacobian[:, :6], dtype=np.float64)
    matrix[:, 3:] *= float(np.median(camera[:, 2]))
    singular = np.linalg.svd(matrix, compute_uv=False)
    if not np.isfinite(singular).all() or singular[0] <= 0:
        return 0.0
    return float(singular[-1] / singular[0])


def _residuals(xyz: np.ndarray, pixels: np.ndarray, rvec: np.ndarray, tvec: np.ndarray,
               K: np.ndarray, D: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    rotation, _ = cv2.Rodrigues(rvec)
    depth = (xyz @ rotation.T + tvec.reshape(1, 3))[:, 2]
    projected, _ = cv2.projectPoints(xyz, rvec, tvec, K, D)
    errors = np.linalg.norm(projected.reshape(-1, 2) - pixels, axis=1)
    errors = np.where(np.isfinite(errors), np.minimum(errors, ERROR_CAP_PX), ERROR_CAP_PX)
    return errors, depth


def _transform(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    rotation, _ = cv2.Rodrigues(rvec)
    result = np.eye(4)
    result[:3, :3], result[:3, 3] = rotation, np.asarray(tvec, dtype=np.float64).reshape(3)
    return result


def _deduplicate(ids: np.ndarray, xyz: np.ndarray, pixels: np.ndarray, distances: np.ndarray,
                 config: LocalizationConfig) -> np.ndarray:
    """Keep the best descriptor distance per pixel, landmark ID and 3D point."""
    order = np.lexsort((ids, distances))
    pixel_keys = np.round(pixels / config.duplicate_pixel_px).astype(np.int64)
    point_keys = np.round(xyz / config.duplicate_point_m).astype(np.int64)
    seen_ids, seen_pixels, seen_points, keep = set(), set(), set(), []
    for index in order.tolist():
        pixel_key, point_key = tuple(pixel_keys[index].tolist()), tuple(point_keys[index].tolist())
        if int(ids[index]) in seen_ids or pixel_key in seen_pixels or point_key in seen_points:
            continue
        seen_ids.add(int(ids[index]))
        seen_pixels.add(pixel_key)
        seen_points.add(point_key)
        keep.append(index)
    keep_array = np.asarray(keep, dtype=np.int64)
    return keep_array[np.argsort(ids[keep_array], kind="stable")] if len(keep_array) else keep_array


def _refine_alternative(xyz: np.ndarray, pixels: np.ndarray, K: np.ndarray, D: np.ndarray | None,
                        rvec: np.ndarray, tvec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """LM refinement of the planar alternative only (separate seam for failure-injection tests)."""
    return cv2.solvePnPRefineLM(xyz, pixels, K, D, rvec, tvec)


def _planar_alternative(xyz: np.ndarray, pixels: np.ndarray, rvec: np.ndarray, tvec: np.ndarray,
                        K: np.ndarray, D: np.ndarray | None, config: LocalizationConfig
                        ) -> tuple[str, float | None, float | None, float | None]:
    """Refine the mirrored-normal planar solution; return (state, rmse, rotation, centre distance)."""
    rotation, _ = cv2.Rodrigues(rvec)
    translation = np.asarray(tvec, dtype=np.float64).reshape(3)
    try:
        camera = xyz @ rotation.T + translation
        centroid = camera.mean(axis=0)
        if not np.isfinite(camera).all():
            return "alternative_unavailable", None, None, None
        normal = np.linalg.svd(camera - centroid)[2][2]
        centroid_norm = float(np.linalg.norm(centroid))
        if not math.isfinite(centroid_norm) or centroid_norm <= 0.0:
            return "alternative_unavailable", None, None, None
        view = centroid / centroid_norm
        mirrored = 2.0 * float(normal @ view) * view - normal
        axis = np.cross(normal, mirrored)
        sine = float(np.linalg.norm(axis))
        if not math.isfinite(sine):
            return "alternative_unavailable", None, None, None
        if sine < 1e-9:
            return "fronto_parallel", None, None, None
        angle = math.atan2(sine, float(np.clip(normal @ mirrored, -1.0, 1.0)))
        seed = (axis * (angle / sine)).reshape(3, 1)
        flip, _ = cv2.Rodrigues(seed)
        alternative_rotation = flip @ rotation
        alternative_translation = flip @ (translation - centroid) + centroid
        alternative_rvec, _ = cv2.Rodrigues(alternative_rotation)
    except (cv2.error, np.linalg.LinAlgError, FloatingPointError, ValueError):
        return "alternative_unavailable", None, None, None
    alternative_tvec = alternative_translation.reshape(3, 1).copy()
    if not (np.isfinite(alternative_rvec).all() and np.isfinite(alternative_tvec).all()):
        return "alternative_unavailable", None, None, None
    # "alternative_unavailable": the evaluation itself failed (OpenCV error or
    # nonfinite output), so ambiguity is unknown and the caller fails closed.
    # "alternative_invalid": a finite alternative that is physically
    # impossible (landmarks not in front of the camera), so it is ruled out.
    try:
        alternative_rvec, alternative_tvec = _refine_alternative(xyz, pixels, K, D, alternative_rvec, alternative_tvec)
    except cv2.error:
        return "alternative_unavailable", None, None, None
    if (alternative_rvec is None or alternative_tvec is None or not np.isfinite(alternative_rvec).all()
            or not np.isfinite(alternative_tvec).all()):
        return "alternative_unavailable", None, None, None
    try:
        alternative_rotation_matrix, _ = cv2.Rodrigues(alternative_rvec)
        depth = (xyz @ alternative_rotation_matrix.T + np.asarray(alternative_tvec).reshape(1, 3))[:, 2]
        projected, _ = cv2.projectPoints(xyz, alternative_rvec, alternative_tvec, K, D)
    except cv2.error:
        return "alternative_unavailable", None, None, None
    projected = projected.reshape(-1, 2)
    if not np.isfinite(depth).all():
        return "alternative_unavailable", None, None, None
    if np.any(depth <= config.min_depth_m):
        return "alternative_invalid", None, None, None
    if not np.isfinite(projected).all():
        return "alternative_unavailable", None, None, None
    # Finite but large errors are evidence (a clearly worse alternative), not unavailability.
    rmse = float(math.sqrt(float(np.mean(np.sum((projected - pixels) ** 2, axis=1)))))
    if not math.isfinite(rmse):
        return "alternative_unavailable", None, None, None
    refined_rotation, _ = cv2.Rodrigues(alternative_rvec)
    rotation_difference = _rotation_angle(rotation, refined_rotation)
    centre = -rotation.T @ translation
    alternative_centre = -refined_rotation.T @ np.asarray(alternative_tvec, dtype=np.float64).reshape(3)
    return "computed", rmse, rotation_difference, float(np.linalg.norm(centre - alternative_centre))


def _required_iterations(inlier_ratio: float, confidence: float, maximum: int) -> int:
    """Adaptive RANSAC bound for three-point samples."""
    if inlier_ratio >= 1.0:
        return 1
    if inlier_ratio <= 0.0 or confidence >= 1.0:
        return maximum
    all_inlier_sample = inlier_ratio ** 3
    if all_inlier_sample <= 1e-12:
        return maximum
    return max(1, min(maximum, int(math.ceil(math.log(1.0 - confidence) / math.log(1.0 - all_inlier_sample)))))


def _iterations_for_sample(probability: float, confidence: float) -> int:
    """Samples needed to draw one all-inlier minimal sample with `confidence`."""
    if probability >= 1.0:
        return 1
    if probability <= 0.0 or confidence >= 1.0:
        return 2 ** 62
    return max(1, int(math.ceil(math.log(1.0 - confidence) / math.log1p(-probability))))


def _ransac_minimal(xyz: np.ndarray, pixels: np.ndarray, K: np.ndarray, D: np.ndarray | None,
                    config: LocalizationConfig, max_iterations: int | None = None, min_iterations: int | None = None,
                    adaptive: bool = True, accept: Callable[[np.ndarray, np.ndarray], bool] | None = None
                    ) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray] | None, int]:
    """Deterministic AP3P RANSAC with cheirality-aware inlier scoring.

    Samples come from numpy.random.default_rng(config.rng_seed), so results do
    not depend on global RNG state. Every minimal solution is scored on all
    correspondences; an inlier needs error <= ransac_reprojection_px and depth
    in (min_depth_m, max_depth_m). Ties prefer the lower inlier error sum.
    Three-point samples make planar and nonplanar support equally valid.
    With adaptive=False exactly max_iterations samples are drawn. accept, if
    given, discards minimal solutions before scoring. Returns the best model
    (or None) and the number of samples whose minimal solve did not raise.
    """
    count = len(xyz)
    if count < 4:
        return None, 0
    maximum = config.ransac_iterations if max_iterations is None else max_iterations
    floor = config.ransac_min_iterations if min_iterations is None else min_iterations
    rng = np.random.default_rng(config.rng_seed)
    best: tuple[int, float, np.ndarray, np.ndarray, np.ndarray] | None = None
    required, iteration, evaluated = maximum, 0, 0
    while iteration < required:
        iteration += 1
        sample = rng.choice(count, 3, replace=False)
        try:
            solutions, rvecs, tvecs = cv2.solveP3P(np.ascontiguousarray(xyz[sample]), np.ascontiguousarray(pixels[sample]),
                                                   K, D, flags=cv2.SOLVEPNP_AP3P)
        except cv2.error:
            continue
        evaluated += 1
        for index in range(int(solutions or 0)):
            rvec = np.asarray(rvecs[index], dtype=np.float64).reshape(3, 1)
            tvec = np.asarray(tvecs[index], dtype=np.float64).reshape(3, 1)
            if not np.isfinite(rvec).all() or not np.isfinite(tvec).all():
                continue
            if accept is not None and not accept(rvec, tvec):
                continue
            errors, depth = _residuals(xyz, pixels, rvec, tvec, K, D)
            inliers = np.flatnonzero((errors <= config.ransac_reprojection_px) & (depth > config.min_depth_m)
                                     & (depth < config.max_depth_m))
            total, error_sum = len(inliers), float(np.sum(errors[inliers]))
            score = (total, -error_sum)
            if best is None or score > best[0]:
                best = (score, total, rvec.copy(), tvec.copy(), inliers)
                if adaptive:
                    # A floor keeps the error-sum tie-break meaningful when every
                    # sample is all-inlier (for example a small, distant target).
                    required = max(floor, _required_iterations(total / count, config.ransac_confidence, maximum))
    if best is None or best[1] < 4:
        return None, evaluated
    return (best[2], best[3], best[4]), evaluated


def _body_pose(rvec: np.ndarray, tvec: np.ndarray, inverse_mount: np.ndarray) -> np.ndarray:
    """Unvalidated T_WB = inverse(T_CW) @ inverse(T_BC) for internal comparisons."""
    rotation, _ = cv2.Rodrigues(rvec)
    world_camera = np.eye(4)
    world_camera[:3, :3] = rotation.T
    world_camera[:3, 3] = -rotation.T @ np.asarray(tvec, dtype=np.float64).reshape(3)
    return world_camera @ inverse_mount


def _pose_difference(first: np.ndarray, second: np.ndarray) -> tuple[float, float]:
    return (float(np.linalg.norm(np.asarray(first)[:3, 3] - np.asarray(second)[:3, 3])),
            _rotation_angle(np.asarray(first)[:3, :3], np.asarray(second)[:3, :3]))


def _distinct(separation: float, rotation: float, config: LocalizationConfig) -> bool:
    return separation > config.ambiguity_distance_m or rotation > config.ambiguity_rotation_rad


# Regional-to-global taxonomy (module docstring, cross-region decision).
# Category 2a: the region's own search could not finish.
EXPLICIT_UNKNOWN_REASONS = ("unresolved_pose_hypotheses", "planar_ambiguity_unresolved")
# Category 2b: the region knows a distinct comparable alternative but does not export its pose.
KNOWN_UNEXPORTED_REASONS = ("planar_pose_ambiguity",)
UNREPRESENTED_REASONS = EXPLICIT_UNKNOWN_REASONS + KNOWN_UNEXPORTED_REASONS


def unresolved_region_is_relevant(attempt: RegionAttempt, best_inliers: int, config: LocalizationConfig) -> bool:
    """True if a verified region holds unrepresented evidence that could hide a competitor.

    Unrepresented evidence is an explicit unknown or a known alternative whose
    pose is not exported (UNREPRESENTED_REASONS). Its unique correspondences
    bound any model it could support. Below comparable_support(best_inliers)
    it cannot hold a comparable competitor and is irrelevant; otherwise it
    must withhold the fix. Every other reason is never relevant here.
    """
    return (attempt.reason in UNREPRESENTED_REASONS
            and attempt.correspondences >= comparable_support(best_inliers, config))


def comparable_support(best_inliers: int, config: LocalizationConfig) -> int:
    """Support a competing hypothesis needs: ceil(ambiguity_inlier_ratio * best), at least 4 (a pose)."""
    return max(4, int(math.ceil(config.ambiguity_inlier_ratio * best_inliers - 1e-9)))


def _sample_probability(needed: int, total: int) -> float:
    """Probability that a random 3-sample lies inside a given set of `needed` of `total`."""
    probability = 1.0
    for offset in range(3):
        probability *= max(0, needed - offset) / max(1, total - offset)
    return probability


@functools.lru_cache(maxsize=64)
def _secondary_bound(min_inliers: int, ambiguity_inlier_ratio: float, min_inlier_ratio: float,
                     confidence: float) -> int:
    worst = 1
    for inliers in range(min_inliers, max(min_inliers + 1024, 8 * min_inliers) + 1):
        total = int(math.floor(inliers / min_inlier_ratio + 1e-9))
        needed = max(4, int(math.ceil(ambiguity_inlier_ratio * inliers - 1e-9)))
        worst = max(worst, _iterations_for_sample(_sample_probability(needed, total), confidence))
    return min(worst, SECONDARY_SEARCH_LIMIT)


def secondary_iteration_bound(config: LocalizationConfig) -> int:
    """Samples the secondary search may need for any frame that passes the earlier gates.

    It is the worst case over admissible frames (inliers >= min_inliers and
    inlier ratio >= min_inlier_ratio), capped at SECONDARY_SEARCH_LIMIT.
    The default settings give about eight thousand samples.
    """
    if config.min_inlier_ratio <= 0.0:
        return SECONDARY_SEARCH_LIMIT
    return _secondary_bound(config.min_inliers, config.ambiguity_inlier_ratio, config.min_inlier_ratio,
                            config.ransac_confidence)


def _refine_secondary(xyz: np.ndarray, pixels: np.ndarray, K: np.ndarray, D: np.ndarray | None,
                      rvec: np.ndarray, tvec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """LM refinement of secondary candidates only (separate seam for failure-injection tests)."""
    return cv2.solvePnPRefineLM(xyz, pixels, K, D, rvec, tvec)


@dataclass(frozen=True, slots=True)
class _Candidate:
    rvec: np.ndarray
    tvec: np.ndarray
    support: np.ndarray
    total: int
    exclusive: int
    separation: float
    rotation: float


def _evaluate_candidate(xyz: np.ndarray, pixels: np.ndarray, rvec: np.ndarray, tvec: np.ndarray,
                        primary_mask: np.ndarray, primary_T_WB: np.ndarray, inverse_mount: np.ndarray,
                        K: np.ndarray, D: np.ndarray | None, config: LocalizationConfig) -> _Candidate:
    errors, depth = _residuals(xyz, pixels, rvec, tvec, K, D)
    support = (errors <= config.ransac_reprojection_px) & (depth > config.min_depth_m) & (depth < config.max_depth_m)
    separation, rotation = _pose_difference(_body_pose(rvec, tvec, inverse_mount), primary_T_WB)
    return _Candidate(rvec, tvec, np.flatnonzero(support), int(np.count_nonzero(support)),
                      int(np.count_nonzero(support & ~primary_mask)), separation, rotation)


def _support_cost(xyz: np.ndarray, pixels: np.ndarray, rows: np.ndarray, rvec: np.ndarray, tvec: np.ndarray,
                  K: np.ndarray, D: np.ndarray | None) -> float:
    errors, _ = _residuals(np.ascontiguousarray(xyz[rows]), np.ascontiguousarray(pixels[rows]), rvec, tvec, K, D)
    return float(np.sum(errors ** 2))


def _refine_candidate(xyz: np.ndarray, pixels: np.ndarray, raw: _Candidate, primary_mask: np.ndarray,
                      primary_T_WB: np.ndarray, inverse_mount: np.ndarray, K: np.ndarray, D: np.ndarray | None,
                      config: LocalizationConfig) -> tuple[_Candidate, bool]:
    """LM on the candidate's total support, rescored on all correspondences.

    An update is accepted only if it is finite, keeps every correspondence it
    was computed from inside (min_depth_m, max_depth_m), and does not increase
    the squared reprojection cost on that fixed support; otherwise refinement
    stops and the last accepted candidate is kept.
    Returns (candidate, verified), where verified means at least one update
    was accepted.
    """
    current, verified = raw, False
    for _ in range(config.refinement_rounds):
        rows = current.support
        if len(rows) < 4:
            break
        try:
            new_rvec, new_tvec = _refine_secondary(np.ascontiguousarray(xyz[rows]), np.ascontiguousarray(pixels[rows]),
                                                   K, D, current.rvec.copy(), current.tvec.copy())
        except cv2.error:
            break
        if (new_rvec is None or new_tvec is None or not np.isfinite(new_rvec).all()
                or not np.isfinite(new_tvec).all()):
            break
        new_rvec = np.asarray(new_rvec, dtype=np.float64).reshape(3, 1)
        new_tvec = np.asarray(new_tvec, dtype=np.float64).reshape(3, 1)
        # Physical validation before adoption: every correspondence the
        # update was computed from must stay inside the depth range, so a
        # finite lower-cost solution behind the camera cannot replace a
        # valid candidate.
        _, new_depth = _residuals(np.ascontiguousarray(xyz[rows]), np.ascontiguousarray(pixels[rows]),
                                  new_rvec, new_tvec, K, D)
        if not (np.isfinite(new_depth).all() and np.all(new_depth > config.min_depth_m)
                and np.all(new_depth < config.max_depth_m)):
            break
        before = _support_cost(xyz, pixels, rows, current.rvec, current.tvec, K, D)
        after = _support_cost(xyz, pixels, rows, new_rvec, new_tvec, K, D)
        if not (math.isfinite(after) and after <= before * (1.0 + 1e-9) + 1e-12):
            break
        current = _evaluate_candidate(xyz, pixels, new_rvec, new_tvec, primary_mask, primary_T_WB, inverse_mount,
                                      K, D, config)
        verified = True
        if np.array_equal(current.support, rows):
            break
    return current, verified


def _secondary_model(xyz: np.ndarray, pixels: np.ndarray, primary_mask: np.ndarray, primary_T_WB: np.ndarray,
                     primary_inliers: int, calibration: CameraCalibration, K: np.ndarray, D: np.ndarray | None,
                     config: LocalizationConfig) -> tuple[str, int, int, float | None, float | None,
                                                          np.ndarray | None, int, int]:
    """Search for a second pose model that competes with the primary.

    A retrieval region may merge correspondences of several places (nearby
    reference positions do not prove the query pose is unique), and two
    places may share support (for example distant landmarks).

    Competitor: a pose distinct from the primary (ambiguity distance or
    rotation) with comparable_support on all correspondences. Shared support
    counts; exclusive support is reported but never required.

    A seeded stream of minimal samples over all unique correspondences draws
    exactly the samples needed to hit a comparable support set at
    ransac_confidence, capped by secondary_ransac_iterations (None: the
    settings-derived secondary_iteration_bound). Every finite AP3P branch that
    is distinct from the primary and has at least max(4, ceil(comparable/2))
    support is a proposal. Each proposal is refined on its total support with
    quality-checked LM updates (_refine_candidate) and then classified:
    * verified and no longer distinct: a legitimate collapse onto the
      primary; only that proposal is resolved and the search continues;
    * verified and distinct with comparable support: "ambiguous" at once;
    * verified and distinct, weaker: kept as the strongest weaker model;
    * not verified (first update unavailable or degrading) with comparable
      raw support: unverified comparable evidence, which cannot be erased
      and makes the result "unresolved" unless a verified competitor is
      found.
    Nothing is deduplicated. A sample cap shortfall, raising minimal solves,
    unverified comparable evidence, or reaching secondary_refinement_limit
    with proposals left give "unresolved" (fail closed). Caps are work
    budgets, not proofs of completeness.
    Returns (state, total, exclusive, separation, rotation, T_WB, samples, refinements).
    """
    needed = comparable_support(primary_inliers, config)
    proposal_floor = max(4, int(math.ceil(needed / 2.0 - 1e-9)))
    count = len(xyz)
    required = _iterations_for_sample(_sample_probability(needed, count), config.ransac_confidence)
    cap = (secondary_iteration_bound(config) if config.secondary_ransac_iterations is None
           else config.secondary_ransac_iterations)
    unresolved = required > cap
    budget = min(required, cap)
    inverse_mount = np.asarray(invert_transform(calibration.T_BC))
    primary_mask = np.asarray(primary_mask, dtype=bool)
    rng = np.random.default_rng(config.rng_seed)
    strongest: _Candidate | None = None
    unverified: _Candidate | None = None
    collapsed, refinements, evaluated, limit_reached = False, 0, 0, False

    def distinct(candidate: _Candidate) -> bool:
        return _distinct(candidate.separation, candidate.rotation, config)

    def result(state: str, candidate: _Candidate | None) -> tuple[str, int, int, float | None, float | None,
                                                                   np.ndarray | None, int, int]:
        if candidate is None:
            return state, 0, 0, None, None, None, evaluated, refinements
        try:
            estimate = body_pose_from_pnp(validate_transform(_transform(candidate.rvec, candidate.tvec),
                                                             "secondary T_CW"), calibration.T_BC)
        except ValueError:
            return "unresolved", candidate.total, candidate.exclusive, None, None, None, evaluated, refinements
        return (state, candidate.total, candidate.exclusive, candidate.separation, candidate.rotation, estimate,
                evaluated, refinements)

    if count < 3:
        return result("no_model", None)
    for _ in range(budget):
        if limit_reached:
            break
        sample = rng.choice(count, 3, replace=False)
        try:
            solutions, rvecs, tvecs = cv2.solveP3P(np.ascontiguousarray(xyz[sample]), np.ascontiguousarray(pixels[sample]),
                                                   K, D, flags=cv2.SOLVEPNP_AP3P)
        except cv2.error:
            continue
        evaluated += 1
        for index in range(int(solutions or 0)):
            rvec = np.asarray(rvecs[index], dtype=np.float64).reshape(3, 1)
            tvec = np.asarray(tvecs[index], dtype=np.float64).reshape(3, 1)
            if not np.isfinite(rvec).all() or not np.isfinite(tvec).all():
                continue
            raw = _evaluate_candidate(xyz, pixels, rvec, tvec, primary_mask, primary_T_WB, inverse_mount, K, D, config)
            if not distinct(raw) or raw.total < proposal_floor:
                continue
            if refinements >= config.secondary_refinement_limit:
                limit_reached = True
                break
            refinements += 1
            candidate, verified = _refine_candidate(xyz, pixels, raw, primary_mask, primary_T_WB, inverse_mount,
                                                    K, D, config)
            if not verified:
                if raw.total >= needed and (unverified is None or raw.total > unverified.total):
                    unverified = raw
                continue
            if not distinct(candidate):
                collapsed = True
                continue
            if candidate.total >= needed:
                return result("ambiguous", candidate)
            if strongest is None or candidate.total > strongest.total:
                strongest = candidate
    unresolved = unresolved or evaluated < budget or limit_reached or unverified is not None
    if unverified is not None:
        return result("unresolved", unverified)
    if strongest is not None:
        return result("unresolved" if unresolved else "weaker", strongest)
    return result("unresolved" if unresolved else ("not_distinct" if collapsed else "no_model"), None)


def verify_pose(landmark_ids: Any, landmarks_xyz: Any, pixels_xy: Any, calibration: CameraCalibration,
                config: LocalizationConfig | None = None, descriptor_distances: Any = None) -> PoseVerification:
    """Robust PnP plus every LOC-01 geometric gate for one correspondence set.

    Pixels are raw distorted coordinates in the calibration's convention.
    The returned body pose is T_WB = inverse(T_CW) @ inverse(T_BC).
    """
    config = LocalizationConfig() if config is None else config
    if not isinstance(config, LocalizationConfig) or not isinstance(calibration, CameraCalibration):
        raise ValueError("Pose verification requires a LocalizationConfig and CameraCalibration")
    ids = np.asarray(landmark_ids)
    if ids.ndim != 1 or (ids.size and ids.dtype.kind not in "iu") or (ids.size and np.any(ids < 0)):
        raise ValueError("Landmark IDs must be a one-dimensional nonnegative integer array")
    ids = ids.astype(np.int64)
    xyz = np.asarray(landmarks_xyz, dtype=np.float64).reshape(-1, 3)
    pixels = np.asarray(pixels_xy, dtype=np.float64).reshape(-1, 2)
    distances = (np.zeros(len(ids)) if descriptor_distances is None
                 else np.asarray(descriptor_distances, dtype=np.float64).reshape(-1))
    if not (len(ids) == len(xyz) == len(pixels) == len(distances)):
        raise ValueError("Correspondence arrays must have equal lengths")
    if not (np.isfinite(xyz).all() and np.isfinite(pixels).all() and np.isfinite(distances).all()):
        raise ValueError("Correspondences must be finite")
    total = int(len(ids))
    keep = _deduplicate(ids, xyz, pixels, distances, config)
    ids, xyz, pixels = ids[keep], np.ascontiguousarray(xyz[keep]), np.ascontiguousarray(pixels[keep])
    unique = int(len(keep))
    state: dict[str, Any] = {
        "input_correspondences": total, "duplicates_removed": total - unique, "correspondences": unique,
        "inliers": 0, "inlier_rmse_px": None, "hull_area_fraction": 0.0, "support_cells": 0,
        "extent_ratio": 0.0, "planarity_ratio": 0.0, "conditioning": 0.0, "behind_low_error": 0,
        "planar_check": "not_run", "planar_alternative_rmse_px": None, "planar_alternative_rotation_rad": None,
        "secondary_check": "not_run", "secondary_inliers": 0, "secondary_exclusive_inliers": 0,
        "secondary_separation_m": None, "secondary_rotation_rad": None, "secondary_estimated_T_WB": None,
        "secondary_samples": 0, "secondary_refinements": 0,
    }
    empty_ids, empty_mask, empty_errors = (_frozen(np.empty(0, np.int64)), _frozen(np.empty(0, bool)),
                                           _frozen(np.empty(0, np.float64)))

    def rejected_without_pose(reason: str) -> PoseVerification:
        return PoseVerification(status="rejected", reason=reason, failed_checks=(reason,), T_CW=None, T_WB=None,
                                estimated_T_WB=None, landmark_ids=empty_ids, inlier_mask=empty_mask,
                                reprojection_error_px=empty_errors, **state)

    if unique < config.min_correspondences:
        return rejected_without_pose("insufficient_correspondences")
    state["extent_ratio"], state["planarity_ratio"] = _shape_ratios(xyz)
    if state["extent_ratio"] < config.min_extent_ratio:
        return rejected_without_pose("degenerate_geometry")
    state["hull_area_fraction"], state["support_cells"] = spatial_support(pixels, calibration, config.support_grid)
    if state["hull_area_fraction"] < config.min_hull_area_fraction or state["support_cells"] < config.min_support_cells:
        return rejected_without_pose("insufficient_spatial_support")

    K, D = np.array(calibration.K, dtype=np.float64), _distortion(calibration)
    hypothesis, _ = _ransac_minimal(xyz, pixels, K, D, config)
    if hypothesis is None:
        return rejected_without_pose("pnp_failed")
    rvec, tvec, selected = hypothesis
    for _ in range(config.refinement_rounds):
        if len(selected) < 4:
            break
        try:
            rvec, tvec = cv2.solvePnPRefineLM(np.ascontiguousarray(xyz[selected]), np.ascontiguousarray(pixels[selected]),
                                              K, D, rvec, tvec)
        except cv2.error:
            return rejected_without_pose("pnp_failed")
        if not np.isfinite(rvec).all() or not np.isfinite(tvec).all():
            return rejected_without_pose("pnp_failed")
        errors, depth = _residuals(xyz, pixels, rvec, tvec, K, D)
        updated = np.flatnonzero((errors <= config.ransac_reprojection_px) & (depth > config.min_depth_m)
                                 & (depth < config.max_depth_m))
        if np.array_equal(updated, selected):
            break
        selected = updated
    errors, depth = _residuals(xyz, pixels, rvec, tvec, K, D)
    try:
        T_CW = validate_transform(_transform(rvec, tvec), "T_CW")
    except ValueError:
        return rejected_without_pose("pnp_failed")
    low_error = errors <= config.ransac_reprojection_px
    mask = low_error & (depth > config.min_depth_m) & (depth < config.max_depth_m)
    inliers = int(np.count_nonzero(mask))
    state["inliers"] = inliers
    state["behind_low_error"] = int(np.count_nonzero(low_error & (depth <= config.min_depth_m)))
    failed: list[str] = []
    if inliers < config.min_inliers:
        failed.append("insufficient_inliers")
    if inliers < config.min_inlier_ratio * unique:
        failed.append("low_inlier_ratio")
    if state["behind_low_error"] > config.max_behind_fraction * max(inliers, 1):
        failed.append("nonpositive_depth")
    if inliers:
        state["inlier_rmse_px"] = float(math.sqrt(float(np.mean(errors[mask] ** 2))))
    if state["inlier_rmse_px"] is None or state["inlier_rmse_px"] > config.max_inlier_rmse_px:
        failed.append("high_reprojection_error")
    state["hull_area_fraction"], state["support_cells"] = spatial_support(pixels[mask], calibration, config.support_grid)
    if state["hull_area_fraction"] < config.min_hull_area_fraction or state["support_cells"] < config.min_support_cells:
        failed.append("insufficient_spatial_support")
    state["conditioning"] = pose_conditioning(xyz[mask], T_CW, calibration) if inliers >= 3 else 0.0
    if state["conditioning"] < config.min_conditioning:
        failed.append("degenerate_geometry")
    inlier_extent, state["planarity_ratio"] = _shape_ratios(xyz[mask]) if inliers >= 3 else (0.0, 0.0)
    if inlier_extent < config.min_extent_ratio and "degenerate_geometry" not in failed:
        failed.append("degenerate_geometry")
    if failed:
        state["planar_check"] = "not_run"
    elif state["planarity_ratio"] >= config.planar_ratio:
        state["planar_check"] = "not_planar"
    else:
        inlier_xyz, inlier_pixels = np.ascontiguousarray(xyz[mask]), np.ascontiguousarray(pixels[mask])
        planar_state, rmse, rotation, distance = _planar_alternative(inlier_xyz, inlier_pixels, rvec, tvec, K, D, config)
        state["planar_alternative_rmse_px"], state["planar_alternative_rotation_rad"] = rmse, rotation
        if planar_state == "alternative_unavailable":
            state["planar_check"] = planar_state
            failed.append("planar_ambiguity_unresolved")
        elif planar_state != "computed":
            state["planar_check"] = planar_state
        elif rotation <= config.ambiguity_rotation_rad and distance <= config.ambiguity_distance_m:
            state["planar_check"] = "alternative_converged"
        elif rmse <= config.planar_ambiguity_rmse_factor * state["inlier_rmse_px"] + config.planar_ambiguity_margin_px:
            state["planar_check"] = "ambiguous"
            failed.append("planar_pose_ambiguity")
        else:
            state["planar_check"] = "alternative_worse"
    estimate = body_pose_from_pnp(T_CW, calibration.T_BC)
    if not failed:
        (state["secondary_check"], state["secondary_inliers"], state["secondary_exclusive_inliers"],
         state["secondary_separation_m"], state["secondary_rotation_rad"], state["secondary_estimated_T_WB"],
         state["secondary_samples"], state["secondary_refinements"]) = _secondary_model(
            xyz, pixels, mask, estimate, inliers, calibration, K, D, config)
        if state["secondary_check"] == "ambiguous":
            failed.append("ambiguous_pose_hypotheses")
        elif state["secondary_check"] == "unresolved":
            failed.append("unresolved_pose_hypotheses")
    accepted = not failed
    return PoseVerification(
        status="accepted" if accepted else "rejected", reason="" if accepted else failed[0],
        failed_checks=tuple(failed), T_CW=T_CW if accepted else None, T_WB=estimate if accepted else None,
        estimated_T_WB=estimate, landmark_ids=_frozen(ids), inlier_mask=_frozen(mask),
        reprojection_error_px=_frozen(errors), **state)


class MapLocalizer:
    """Synchronous absolute localizer for one FrozenMap and one calibration.

    Inputs are only the current CameraObservation (or its QueryFeatures) and
    frozen map assets. No query pose, raw state, previous output or future
    exposure enters this API.
    """

    def __init__(self, frozen_map: FrozenMap, sift_config: SiftConfig | None = None,
                 retrieval_config: RetrievalConfig | None = None, config: LocalizationConfig | None = None,
                 clock: Callable[[], int] = time.monotonic_ns):
        if not isinstance(frozen_map, FrozenMap):
            raise ValueError("MapLocalizer requires a FrozenMap")
        self.config = LocalizationConfig() if config is None else config
        if not isinstance(self.config, LocalizationConfig):
            raise ValueError("MapLocalizer requires a LocalizationConfig")
        if not callable(clock):
            raise ValueError("clock must be a callable monotonic nanosecond source")
        self.map = frozen_map
        self.calibration = frozen_map.calibration
        self.extractor = SiftExtractor(self.calibration, sift_config)
        # Query and map descriptors require identical extractor, version,
        # weights, dimension, dtype, metric, pixel convention and settings.
        require_compatible(frozen_map.feature_identity, self.extractor.identity,
                           self.calibration.identity, self.calibration.identity)
        self.index = RetrievalIndex(frozen_map, retrieval_config, self.extractor.config.ratio_test)
        self._clock = clock
        self.feature_id = frozen_map.feature_identity.identity
        self.identity = canonical_identity({
            "schema": LOCALIZER_SCHEMA, "pnp": PNP_METHOD, "map_hash": frozen_map.map_hash,
            "feature_id": self.feature_id, "calibration_id": self.calibration.identity,
            "retrieval_index": self.index.identity, "config": self.config.as_dict(),
            "opencv": cv2.__version__, "numpy": np.__version__,
        })

    def _now(self) -> int:
        return validate_timestamp_ns(self._clock())

    def localize(self, observation: CameraObservation) -> LocalizationAttempt:
        if not isinstance(observation, CameraObservation):
            raise ValueError("localize requires a CameraObservation")
        started = self._now()
        calibration = self.calibration
        if observation.camera_id != calibration.camera_id:
            return self._early(observation.timestamp_ns, "camera_mismatch", started)
        if observation.calibration_id != calibration.identity:
            return self._early(observation.timestamp_ns, "calibration_mismatch", started)
        if observation.image.shape[:2] != (calibration.height, calibration.width):
            return self._early(observation.timestamp_ns, "image_size_mismatch", started)
        features = self.extractor.extract(observation)
        extracted = self._now()
        return self._localize(features, started, extracted - started)

    def localize_features(self, features: QueryFeatures) -> LocalizationAttempt:
        if not isinstance(features, QueryFeatures):
            raise ValueError("localize_features requires QueryFeatures")
        started = self._now()
        calibration = self.calibration
        if features.camera_id != calibration.camera_id:
            return self._early(features.timestamp_ns, "camera_mismatch", started)
        if features.calibration_id != calibration.identity:
            return self._early(features.timestamp_ns, "calibration_mismatch", started)
        if (features.image_width, features.image_height) != (calibration.width, calibration.height):
            return self._early(features.timestamp_ns, "image_size_mismatch", started)
        if features.feature_id != self.feature_id:
            return self._early(features.timestamp_ns, "feature_identity_mismatch", started)
        identity = self.map.feature_identity
        if (features.descriptors.dtype != np.dtype(identity.descriptor_dtype)
                or features.descriptors.shape[1] != identity.descriptor_dimension):
            return self._early(features.timestamp_ns, "descriptor_mismatch", started)
        return self._localize(features, started, 0)

    def _fix(self, timestamp_ns: int, completed: int, verification: PoseVerification | None, region_id: str,
             ambiguity: float, reason: str) -> FixCandidate:
        accepted = not reason
        if verification is None:
            ids, mask, errors, conditioning = np.empty(0, np.int64), np.empty(0, bool), np.empty(0, np.float64), 0.0
        else:
            ids, mask = verification.landmark_ids, verification.inlier_mask
            errors, conditioning = verification.reprojection_error_px, verification.conditioning
        return FixCandidate(
            source_timestamp_ns=timestamp_ns,
            T_WB=verification.T_WB if accepted and verification is not None else None,
            status="accepted" if accepted else "rejected", reason=reason, landmark_ids=ids, inlier_mask=mask,
            reprojection_error_px=errors, conditioning=conditioning, region_id=region_id, ambiguity=ambiguity,
            map_hash=self.map.map_hash, feature_id=self.feature_id, calibration_id=self.calibration.identity,
            latest_input_timestamp_ns=timestamp_ns, completed_monotonic_ns=completed)

    def _diagnostics(self, reason: str, started: int, completed: int, extraction_ns: int = 0, retrieval_ns: int = 0,
                     geometry_ns: int = 0, query_keypoints: int = 0, retrieval: RetrievalResult | None = None,
                     attempts: tuple[RegionAttempt, ...] = (), competitor: tuple[str, float, float | None, float | None] = ("", 0.0, None, None)
                     ) -> FixDiagnostics:
        return FixDiagnostics(
            reason=reason, query_keypoints=query_keypoints, retrieval_references=self.index.reference_count,
            retrieval_regions=() if retrieval is None else retrieval.regions, attempts=attempts,
            competitor_region_id=competitor[0], competitor_ratio=competitor[1], competitor_separation_m=competitor[2],
            competitor_rotation_rad=competitor[3], started_monotonic_ns=started, extraction_ns=extraction_ns,
            retrieval_ns=retrieval_ns, geometry_ns=geometry_ns, completed_monotonic_ns=completed,
            localizer_identity=self.identity, derived_retrieval_index_identity=self.index.identity,
            # Recorded only; this declared artifact checksum is not verified here.
            declared_retrieval_index_sha256=self.map.retrieval_index_sha256)

    def _early(self, timestamp_ns: int, reason: str, started: int) -> LocalizationAttempt:
        completed = self._now()
        return LocalizationAttempt(self._fix(timestamp_ns, completed, None, "", 0.0, reason),
                                   self._diagnostics(reason, started, completed))

    def _verify_region(self, features: QueryFeatures, candidate: RetrievalCandidate) -> tuple[RegionAttempt, PoseVerification]:
        rows = self.index.region_rows(candidate.region_id)
        matches = match_descriptors(np.asarray(features.descriptors), self.index.region_descriptors(candidate.region_id),
                                    self.index.ratio_test)
        landmark_rows = rows[np.asarray(matches.train_indices, dtype=np.int64)]
        verification = verify_pose(
            np.asarray(self.map.landmark_ids)[landmark_rows], np.asarray(self.map.landmarks_xyz)[landmark_rows],
            np.asarray(features.keypoints_xy)[np.asarray(matches.query_indices, dtype=np.int64)],
            self.calibration, self.config, np.asarray(matches.distances))
        attempt = RegionAttempt(
            region_id=candidate.region_id, reference_id=candidate.reference_id, retrieval_rank=candidate.rank,
            retrieval_votes=candidate.votes, candidate_landmarks=int(len(rows)), tentative_matches=len(matches),
            duplicates_removed=verification.duplicates_removed, correspondences=verification.correspondences,
            inliers=verification.inliers, inlier_rmse_px=verification.inlier_rmse_px,
            hull_area_fraction=verification.hull_area_fraction, support_cells=verification.support_cells,
            extent_ratio=verification.extent_ratio, planarity_ratio=verification.planarity_ratio,
            conditioning=verification.conditioning, behind_low_error=verification.behind_low_error,
            planar_check=verification.planar_check, planar_alternative_rmse_px=verification.planar_alternative_rmse_px,
            planar_alternative_rotation_rad=verification.planar_alternative_rotation_rad,
            secondary_check=verification.secondary_check, secondary_inliers=verification.secondary_inliers,
            secondary_exclusive_inliers=verification.secondary_exclusive_inliers,
            secondary_separation_m=verification.secondary_separation_m,
            secondary_rotation_rad=verification.secondary_rotation_rad,
            secondary_samples=verification.secondary_samples,
            secondary_refinements=verification.secondary_refinements,
            reason=verification.reason, failed_checks=verification.failed_checks)
        return attempt, verification

    def _localize(self, features: QueryFeatures, started: int, extraction_ns: int) -> LocalizationAttempt:
        config, timestamp = self.config, features.timestamp_ns
        if features.count < config.min_query_features:
            completed = self._now()
            return LocalizationAttempt(
                self._fix(timestamp, completed, None, "", 0.0, "insufficient_features"),
                self._diagnostics("insufficient_features", started, completed, extraction_ns,
                                  query_keypoints=features.count))
        retrieval_started = self._now()
        retrieval = self.index.rank(features)
        retrieval_done = self._now()
        retrieval_ns = retrieval_done - retrieval_started
        if not retrieval.regions:
            completed = self._now()
            return LocalizationAttempt(
                self._fix(timestamp, completed, None, "", 0.0, "no_retrieval_candidate"),
                self._diagnostics("no_retrieval_candidate", started, completed, extraction_ns, retrieval_ns,
                                  query_keypoints=features.count, retrieval=retrieval))
        verified = retrieval.regions[:config.max_verified_regions]
        results = [self._verify_region(features, candidate) for candidate in verified]
        attempts = tuple(attempt for attempt, _ in results)
        valid = [(attempt, verification) for attempt, verification in results if verification.status == "accepted"]
        reason, region_id, chosen, competitor = "", "", None, ("", 0.0, None, None)
        if not valid:
            # Deliberate latency bound: regions beyond max_verified_regions are
            # not tried; the best-ranked attempt explains the rejection.
            reason, region_id, chosen = attempts[0].reason, attempts[0].region_id, results[0][1]
            first = attempts[0]
            if first.secondary_check in {"ambiguous", "unresolved"} and first.secondary_separation_m is not None:
                competitor = (f"{first.region_id}#secondary", first.secondary_inliers / max(first.inliers, 1),
                              first.secondary_separation_m, first.secondary_rotation_rad)
        else:
            best_attempt, chosen = max(valid, key=lambda item: (item[0].inliers, -item[0].retrieval_rank))
            region_id = best_attempt.region_id
            best_pose = chosen.T_WB
            needed = comparable_support(best_attempt.inliers, config)
            ambiguous = False
            # Competitors are every other primary estimate, accepted or not,
            # and every secondary model: a distant hypothesis that nearly
            # verifies must block a fix rather than be ignored.
            hypotheses = []
            for attempt, verification in results:
                if attempt.region_id != best_attempt.region_id and verification.estimated_T_WB is not None:
                    hypotheses.append((attempt.region_id, verification.estimated_T_WB, attempt.inliers))
                if verification.secondary_estimated_T_WB is not None:
                    hypotheses.append((f"{attempt.region_id}#secondary", verification.secondary_estimated_T_WB,
                                       attempt.secondary_inliers))
            for name, estimate, support in hypotheses:
                separation, rotation = _pose_difference(estimate, best_pose)
                if not _distinct(separation, rotation, config):
                    continue
                ratio = support / max(best_attempt.inliers, 1)
                if ratio > competitor[1] or not competitor[0]:
                    competitor = (name, ratio, separation, rotation)
                ambiguous = ambiguous or support >= needed
            unrepresented = [attempt for attempt in attempts if attempt.region_id != best_attempt.region_id
                             and unresolved_region_is_relevant(attempt, best_attempt.inliers, config)]
            if ambiguous:
                reason = "ambiguous_distant_regions"
            elif unrepresented:
                # A verified region with an explicit unknown or a known but
                # unexported alternative cannot certify that no competing
                # model exists. Its correspondences only bound the support.
                reason = "unresolved_competing_region"
                strongest = max(unrepresented, key=lambda item: (item.correspondences, -item.retrieval_rank))
                ratio = strongest.correspondences / max(best_attempt.inliers, 1)
                if ratio > competitor[1]:
                    competitor = (strongest.region_id, ratio, None, None)
            else:
                for candidate in retrieval.regions[config.max_verified_regions:]:
                    ratio = candidate.votes / max(best_attempt.retrieval_votes, 1)
                    if candidate.votes >= comparable_support(best_attempt.retrieval_votes, config):
                        reason = "unresolved_competing_region"
                        if ratio > competitor[1]:
                            competitor = (candidate.region_id, ratio, None, None)
                        break
        geometry_ns = self._now() - retrieval_done
        completed = self._now()
        fix = self._fix(timestamp, completed, chosen, region_id, float(competitor[1]), reason)
        return LocalizationAttempt(fix, self._diagnostics(
            reason, started, completed, extraction_ns, retrieval_ns, geometry_ns, features.count, retrieval,
            attempts, competitor))

"""Camera-only sparse mapping, geometric localization and causal tracking.

Reference camera poses anchor metric geometry. Query inference never receives
labels, a full sequence, telemetry, or ideal synthetic correspondences.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import cv2
import numpy as np

from .features import create_feature_adapter, feature_identity, subset_features


DEFAULT_CONFIG = {
    "feature_backend": "sift", "max_features": 1800, "ratio_test": 0.75,
    "ransac_px": 4.0, "min_inliers": 8, "relocalize_every": 4,
    "max_propagation_s": 0.45, "forecast_horizon_s": 0.1,
    "forecast_mode": "next_frame", "method": "tracking",
    "motion_model": "yaw", "learned_match_threshold": 0.1,
}


def wrap_angle(value: float) -> float:
    return float(np.arctan2(np.sin(value), np.cos(value)))


def rotation_distance(first: np.ndarray, second: np.ndarray) -> float:
    """SO(3) geodesic angle in radians, independent of a gravity-up axis."""
    return float(np.arccos(np.clip((np.trace(np.asarray(first).T @ np.asarray(second)) - 1.0) / 2.0, -1.0, 1.0)))


def camera_to_vehicle(camera_from_world: np.ndarray, vehicle_to_camera: np.ndarray) -> np.ndarray:
    return np.linalg.inv(camera_from_world) @ vehicle_to_camera


def pose_record(transform: np.ndarray, pose_target: str = "vehicle") -> dict:
    return {"position_m": transform[:3, 3].tolist(),
            "yaw_rad": None if pose_target == "camera" else float(np.arctan2(transform[1, 0], transform[0, 0])),
            "T_WC" if pose_target == "camera" else "T_WV": transform.tolist()}


def _configuration(config: dict) -> dict:
    result = {**DEFAULT_CONFIG, **config}
    if result["feature_backend"] not in {"sift", "orb", "xfeat_mnn", "xfeat_lighterglue"}:
        raise ValueError("Unknown feature backend")
    if result["method"] not in {"retrieval", "geometry", "tracking"}:
        raise ValueError("Unknown localization method")
    if result["forecast_mode"] not in {"next_frame", "fixed_horizon"}:
        raise ValueError("Unknown forecast mode")
    if result["motion_model"] not in {"yaw", "so3"}:
        raise ValueError("Unknown visual motion model")
    return result


def _detector(config: dict):
    if config["feature_backend"] == "sift":
        return cv2.SIFT_create(nfeatures=int(config["max_features"]), contrastThreshold=0.025)
    return cv2.ORB_create(nfeatures=int(config["max_features"]), scaleFactor=1.15,
                          nlevels=8, edgeThreshold=19, fastThreshold=12)


def _extract(image: np.ndarray, detector) -> tuple[np.ndarray, np.ndarray]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    mask = np.full(gray.shape, 255, np.uint8)
    mask[int(gray.shape[0] * 0.94):] = 0
    keypoints, descriptors = detector.detectAndCompute(gray, mask)
    points = np.asarray([keypoint.pt for keypoint in keypoints], dtype=np.float32).reshape(-1, 2)
    if descriptors is None:
        dimension = detector.descriptorSize()
        dtype = np.uint8 if detector.descriptorType() == cv2.CV_8U else np.float32
        descriptors = np.empty((0, dimension), dtype=dtype)
    return points, descriptors


def _embedding(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    tiny = cv2.resize(gray, (24, 16), interpolation=cv2.INTER_AREA).astype(np.float32).ravel()
    tiny -= np.mean(tiny)
    return tiny / max(float(np.linalg.norm(tiny)), 1e-8)


def _matches(first: np.ndarray, second: np.ndarray, config: dict) -> list[tuple[int, int, float]]:
    if len(first) < 2 or len(second) < 2:
        return []
    norm = cv2.NORM_HAMMING if config["feature_backend"] == "orb" else cv2.NORM_L2
    matcher = cv2.BFMatcher(norm)
    pairs = matcher.knnMatch(first, second, k=2)
    kept = []
    used_train = set()
    for neighbors in sorted(pairs, key=lambda pair: pair[0].distance if pair else float("inf")):
        if len(neighbors) != 2:
            continue
        best, runner_up = neighbors
        if best.distance < float(config["ratio_test"]) * runner_up.distance and best.trainIdx not in used_train:
            kept.append((best.queryIdx, best.trainIdx, float(best.distance)))
            used_train.add(best.trainIdx)
    return kept


def _projection(calibration: dict, camera_pose: np.ndarray) -> np.ndarray:
    return np.asarray(calibration["K"], dtype=float) @ np.linalg.inv(camera_pose)[:3]


def _map_digest(arrays: dict, provenance: dict) -> str:
    digest = hashlib.sha256()
    for name in sorted(arrays):
        digest.update(name.encode())
        digest.update(str(arrays[name].dtype).encode())
        digest.update(str(arrays[name].shape).encode())
        digest.update(arrays[name].tobytes())
    digest.update(json.dumps(provenance, sort_keys=True, separators=(",", ":")).encode())
    return digest.hexdigest()


def _triangulate(points_a: np.ndarray, points_b: np.ndarray, camera_a: np.ndarray,
                 camera_b: np.ndarray, calibration_a: dict, calibration_b: dict) -> tuple[np.ndarray, np.ndarray]:
    if not len(points_a):
        return np.empty((0, 3)), np.zeros(0, bool)
    k_a, k_b = np.asarray(calibration_a["K"], float), np.asarray(calibration_b["K"], float)
    d_a, d_b = np.asarray(calibration_a["distortion"], float), np.asarray(calibration_b["distortion"], float)
    pixel_a = cv2.undistortPoints(points_a.reshape(-1, 1, 2), k_a, d_a, P=k_a).reshape(-1, 2)
    pixel_b = cv2.undistortPoints(points_b.reshape(-1, 1, 2), k_b, d_b, P=k_b).reshape(-1, 2)
    projected_a, projected_b = _projection(calibration_a, camera_a), _projection(calibration_b, camera_b)
    homogeneous = cv2.triangulatePoints(projected_a, projected_b, pixel_a.T, pixel_b.T).T
    with np.errstate(divide="ignore", invalid="ignore"):
        world = homogeneous[:, :3] / homogeneous[:, 3:4]
        wa = np.c_[world, np.ones(len(world))]
        ca = wa @ np.linalg.inv(camera_a)[:3].T
        cb = wa @ np.linalg.inv(camera_b)[:3].T
        qa, qb = wa @ projected_a.T, wa @ projected_b.T
        ra = np.linalg.norm(qa[:, :2] / qa[:, 2:3] - pixel_a, axis=1)
        rb = np.linalg.norm(qb[:, :2] / qb[:, 2:3] - pixel_b, axis=1)
        rays_a = world - camera_a[:3, 3]
        rays_b = world - camera_b[:3, 3]
        cosine = np.sum(rays_a * rays_b, axis=1) / (np.linalg.norm(rays_a, axis=1) * np.linalg.norm(rays_b, axis=1))
    valid = (np.isfinite(world).all(axis=1) & (ca[:, 2] > 0.2) & (cb[:, 2] > 0.2) &
             (ca[:, 2] < 150) & (cb[:, 2] < 150) & (ra < 2.8) & (rb < 2.8) &
             (cosine < np.cos(np.deg2rad(0.35))))
    return world, valid


def build_map(reference_sessions: list[dict], output_dir: Path, config: dict) -> dict:
    """Triangulate feature matches from allowed calibrated reference images."""
    started = time.perf_counter()
    config = _configuration(config)
    if not reference_sessions or any(session.get("split") != "reference" for session in reference_sessions):
        raise ValueError("Mapping accepts reference sessions only")
    adapter = create_feature_adapter(config)
    references = []
    image_hashes = {}
    for session in reference_sessions:
        poses = session.get("reference_poses", {})
        for frame in session["frames"]:
            pose = poses.get(frame["frame_id"])
            if pose is None:
                raise ValueError("Every reference frame requires a calibrated metric T_WC")
            image = cv2.imread(str(frame["image_path"]), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"Unreadable reference image {frame['frame_id']}")
            features = adapter.extract(image)
            points, descriptors = features["keypoints"], features["descriptors"]
            image_hashes[f"{session['id']}:{frame['frame_id']}"] = hashlib.sha256(
                Path(frame["image_path"]).read_bytes()).hexdigest()
            references.append({"id": f"{session['id']}:{frame['frame_id']}",
                               "session_id": session["id"], "pose": np.asarray(pose, float),
                               "calibration": session["calibration"], "points": points,
                               "descriptors": descriptors, "features": features, "embedding": _embedding(image),
                               "landmarks": np.full(len(points), -1, np.int32)})
    if len(references) < 2:
        raise ValueError("At least two distinct reference camera views are required")
    world_points, canonical_descriptors = [], []
    pair_count, accepted_pairs = 0, 0
    for a_index, a in enumerate(references):
        distances = [np.linalg.norm(a["pose"][:3, 3] - b["pose"][:3, 3]) for b in references]
        neighbor_indices = [index for index in np.argsort(distances)
                            if index > a_index and 0.08 < distances[index] < 12.0][:3]
        for b_index in neighbor_indices:
            b = references[b_index]
            matches = adapter.match(a["features"], b["features"])
            if len(matches) < 8:
                continue
            pair_count += 1
            a_ids = np.asarray([match[0] for match in matches])
            b_ids = np.asarray([match[1] for match in matches])
            xyz, valid = _triangulate(a["points"][a_ids], b["points"][b_ids],
                                      a["pose"], b["pose"], a["calibration"], b["calibration"])
            for match_index in np.flatnonzero(valid):
                feature_a, feature_b = int(a_ids[match_index]), int(b_ids[match_index])
                existing_a, existing_b = int(a["landmarks"][feature_a]), int(b["landmarks"][feature_b])
                if existing_a >= 0 and existing_b >= 0 and existing_a != existing_b:
                    continue
                landmark = max(existing_a, existing_b)
                if landmark >= 0:
                    if np.linalg.norm(np.asarray(world_points[landmark]) - xyz[match_index]) > 1.0:
                        continue
                else:
                    landmark = len(world_points)
                    world_points.append(xyz[match_index])
                    canonical_descriptors.append(a["descriptors"][feature_a])
                a["landmarks"][feature_a] = landmark
                b["landmarks"][feature_b] = landmark
                accepted_pairs += 1
    if len(world_points) < int(config["min_inliers"]):
        raise ValueError("Reference images did not yield enough triangulated landmarks. Check overlap, calibration and metric poses.")
    offsets = [0]
    for reference in references:
        offsets.append(offsets[-1] + len(reference["points"]))
    points = np.asarray(world_points, np.float32)
    arrays = {
        "points": points, "descriptors": np.asarray(canonical_descriptors),
        "reference_ids": np.asarray([reference["id"] for reference in references]),
        "reference_poses": np.asarray([reference["pose"] for reference in references]),
        "reference_embeddings": np.asarray([reference["embedding"] for reference in references]),
        "reference_offsets": np.asarray(offsets, np.int32),
        "reference_keypoints": np.concatenate([reference["points"] for reference in references]),
        "reference_descriptors": np.concatenate([reference["descriptors"] for reference in references]),
        "reference_scores": np.concatenate([reference["features"]["scores"] for reference in references]),
        "reference_image_sizes": np.asarray([reference["features"]["image_size"] for reference in references]),
        "reference_landmarks": np.concatenate([reference["landmarks"] for reference in references]),
    }
    provenance = {"feature_backend": config["feature_backend"], "feature_identity": adapter.identity,
                  "reference_sessions": [session["id"] for session in reference_sessions],
                  "reference_image_hashes": image_hashes,
                  "reference_calibration_hashes": {
                      session["id"]: hashlib.sha256(json.dumps(session["calibration"],
                                                              sort_keys=True).encode()).hexdigest()
                      for session in reference_sessions},
                  "map_settings": {key: config[key] for key in ("max_features", "ratio_test", "learned_match_threshold")}}
    map_hash = _map_digest(arrays, provenance)
    metadata = {"id": f"map-{map_hash[:12]}", "hash": map_hash,
                "feature_backend": config["feature_backend"], "landmark_count": len(points),
                "reference_session_ids": [session["id"] for session in reference_sessions],
                "bounds": {"min": points.min(axis=0).tolist(), "max": points.max(axis=0).tolist()},
                "reference_count": len(references), "triangulation_pairs": pair_count,
                "accepted_observations": accepted_pairs, "version": 2, "feature_identity": adapter.identity,
                "build_ms": (time.perf_counter() - started) * 1000,
                "map_settings": provenance["map_settings"], "provenance": provenance}
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_dir / "map.npz", **arrays)
    (output_dir / "map.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def load_map(output_dir: Path) -> dict:
    output_dir = Path(output_dir)
    metadata = json.loads((output_dir / "map.json").read_text(encoding="utf-8"))
    with np.load(output_dir / "map.npz", allow_pickle=False) as archive:
        result = {name: archive[name].copy() for name in archive.files}
    if _map_digest(result, metadata["provenance"]) != metadata["hash"]:
        raise ValueError("Reference map integrity check failed: arrays or provenance changed")
    if (result["points"].ndim != 2 or result["points"].shape[1] != 3 or
            not np.isfinite(result["points"]).all()):
        raise ValueError("Reference map contains invalid landmarks")
    if metadata.get("version") not in {1, 2}:
        raise ValueError("Unsupported reference map version")
    if metadata.get("version") == 2:
        expected = feature_identity(metadata["feature_backend"])
        if metadata.get("feature_identity") != expected or metadata["provenance"].get("feature_identity") != expected:
            raise ValueError("Reference map feature identity differs from current source, weights or runtime")
        _validate_feature_arrays(result, expected)
    elif metadata["feature_backend"] not in {"sift", "orb"}:
        raise ValueError("Legacy reference maps support SIFT or ORB only")
    result["metadata"] = metadata
    return result


def _validate_feature_arrays(arrays: dict, identity: dict) -> None:
    references, points = len(arrays["reference_ids"]), len(arrays["reference_keypoints"])
    offsets = arrays["reference_offsets"]
    dimension = identity["descriptor_dimension"]
    dtype = np.dtype(identity["descriptor_dtype"])
    if (offsets.shape != (references + 1,) or offsets[0] != 0 or offsets[-1] != points or
            np.any(np.diff(offsets) < 0) or arrays["reference_keypoints"].shape != (points, 2) or
            arrays["reference_descriptors"].shape != (points, dimension) or
            arrays["reference_descriptors"].dtype != dtype or arrays["descriptors"].shape != (len(arrays["points"]), dimension) or
            arrays["descriptors"].dtype != dtype or arrays["reference_scores"].shape != (points,) or
            arrays["reference_landmarks"].shape != (points,) or arrays["reference_image_sizes"].shape != (references, 2)):
        raise ValueError("Reference map feature arrays have incompatible dimensions or dtype")
    if (not np.isfinite(arrays["reference_keypoints"]).all() or not np.isfinite(arrays["reference_descriptors"]).all() or
            not np.isfinite(arrays["reference_scores"]).all() or np.any(arrays["reference_image_sizes"] <= 0) or
            np.any(arrays["reference_landmarks"] < -1) or np.any(arrays["reference_landmarks"] >= len(arrays["points"]))):
        raise ValueError("Reference map contains invalid feature observations")


class Localizer:
    def __init__(self, map_data: dict, calibration: dict, config: dict):
        self.config = _configuration(config)
        if self.config["feature_backend"] != map_data["metadata"]["feature_backend"]:
            raise ValueError("Candidate feature family must match the map feature family")
        self.map = map_data
        self.calibration = calibration
        self.k = np.asarray(calibration["K"], float)
        self.distortion = np.asarray(calibration["distortion"], float)
        self.t_cv = np.asarray(calibration["T_CV"], float)
        self.pose_target = calibration.get("pose_target", "vehicle")
        if self.pose_target not in {"vehicle", "camera"}:
            raise ValueError("Unknown pose target")
        if self.pose_target == "camera" and not np.allclose(self.t_cv, np.eye(4), atol=1e-10):
            raise ValueError("Camera-target inference requires identity T_CV")
        self.adapter = create_feature_adapter(self.config)
        expected_identity = map_data["metadata"].get("feature_identity")
        if expected_identity is not None and self.adapter.identity != expected_identity:
            raise ValueError("Candidate feature identity must match the map source, weights and runtime")
        if expected_identity is None and self.config["feature_backend"] not in {"sift", "orb"}:
            raise ValueError("Learned features require a versioned map feature identity")
        self.detector = getattr(self.adapter, "detector", None)
        self.reset()

    def reset(self) -> None:
        self.previous_gray = None
        self.tracked_pixels = np.empty((0, 2), np.float32)
        self.tracked_landmarks = np.empty(0, np.int32)
        self.last_visual_pose = None
        self.last_visual_time = None
        self.last_timestamp = None
        self.velocity = np.zeros(3)
        self.yaw_rate = 0.0
        self.angular_velocity = np.zeros(3)
        self.step_count = 0
        self.last_status = "lost"

    def _solve(self, pixels: np.ndarray, landmark_ids: np.ndarray) -> dict | None:
        if len(pixels) < int(self.config["min_inliers"]):
            return None
        unique, first_indices = np.unique(landmark_ids, return_index=True)
        pixels = pixels[first_indices].astype(np.float32)
        xyz = self.map["points"][unique].astype(np.float32)
        if len(pixels) < int(self.config["min_inliers"]):
            return None
        if np.linalg.matrix_rank(xyz - np.mean(xyz, axis=0), tol=1e-5) < 2:
            return None
        cv2.setRNGSeed(31415)
        try:
            ok, rotation, translation, inliers = cv2.solvePnPRansac(
                xyz, pixels, self.k, self.distortion, iterationsCount=160,
                reprojectionError=float(self.config["ransac_px"]), confidence=0.999,
                flags=cv2.SOLVEPNP_EPNP)
        except cv2.error:
            return None
        if not ok or inliers is None or len(inliers) < int(self.config["min_inliers"]):
            return None
        selected = inliers.ravel()
        try:
            rotation, translation = cv2.solvePnPRefineLM(xyz[selected], pixels[selected],
                                                        self.k, self.distortion, rotation, translation)
        except cv2.error:
            return None
        r_cw, _ = cv2.Rodrigues(rotation)
        t_cw = np.eye(4)
        t_cw[:3, :3], t_cw[:3, 3] = r_cw, translation.ravel()
        if not np.isfinite(t_cw).all():
            return None
        depths = (xyz[selected] @ r_cw.T + translation.ravel())[:, 2]
        if np.mean(depths > 0.2) < 0.98:
            return None
        hull = cv2.convexHull(pixels[selected])
        area = float(cv2.contourArea(hull))
        if area < self.calibration["image_width"] * self.calibration["image_height"] * 0.003:
            return None
        projected, _ = cv2.projectPoints(xyz[selected], rotation, translation, self.k, self.distortion)
        residual = float(np.median(np.linalg.norm(projected.reshape(-1, 2) - pixels[selected], axis=1)))
        if residual > float(self.config["ransac_px"]):
            return None
        transform = camera_to_vehicle(t_cw, self.t_cv)
        return {"transform": transform, "pixels": pixels[selected], "landmarks": unique[selected],
                "inliers": len(selected), "residual": residual, "spread": area}

    def _map_fix(self, image: np.ndarray, timestamp_s: float) -> dict | None:
        features = self.adapter.extract(image)
        pixels = features["keypoints"]
        if len(pixels) < int(self.config["min_inliers"]):
            return None
        embedding = _embedding(image)
        scores = self.map["reference_embeddings"] @ embedding
        order = np.argsort(scores)[::-1]
        # Fast visual candidate retrieval is only a shortlist. PnP supplies
        # the metric pose and geometric verification, not the thumbnail score.
        candidate_indices = order[:min(5, len(order))]
        if self.last_visual_pose is not None and self.last_status != "lost":
            predicted_camera = self._propagate(timestamp_s) @ np.linalg.inv(self.t_cv)
            distances = np.linalg.norm(self.map["reference_poses"][:, :3, 3] - predicted_camera[:3, 3], axis=1)
            nearby = np.argsort(distances)[:3]
            candidate_indices = np.unique(np.r_[nearby, candidate_indices])
        if self.config["method"] == "retrieval":
            best = int(order[0])
            transform = self.map["reference_poses"][best] @ self.t_cv
            return {"transform": transform, "pixels": np.empty((0, 2), np.float32),
                    "landmarks": np.empty(0, np.int32), "inliers": 0,
                    "residual": None, "reference_id": str(self.map["reference_ids"][best])}
        fixes = []
        for reference_index in candidate_indices:
            offset, end = self.map["reference_offsets"][reference_index:reference_index + 2]
            associations = self.map["reference_landmarks"][offset:end]
            mapped_indices = np.flatnonzero(associations >= 0)
            if len(mapped_indices) < int(self.config["min_inliers"]):
                continue
            reference = {"keypoints": self.map["reference_keypoints"][offset:end],
                         "descriptors": self.map["reference_descriptors"][offset:end],
                         "scores": self.map.get("reference_scores", np.zeros(len(self.map["reference_keypoints"]), np.float32))[offset:end],
                         "image_size": self.map.get("reference_image_sizes", np.asarray([[self.calibration["image_width"], self.calibration["image_height"]]] * len(self.map["reference_ids"])))[reference_index]}
            matches = self.adapter.match(features, subset_features(reference, mapped_indices))
            if not matches:
                continue
            query_indices = np.asarray([match[0] for match in matches])
            reference_features = mapped_indices[np.asarray([match[1] for match in matches])]
            fix = self._solve(pixels[query_indices], associations[reference_features])
            if fix is not None:
                fix["reference_id"] = str(self.map["reference_ids"][reference_index])
                fixes.append(fix)
        if not fixes:
            return None
        return max(fixes, key=lambda fix: (fix["inliers"], -fix["residual"]))

    def _track(self, gray: np.ndarray) -> dict | None:
        if self.previous_gray is None or len(self.tracked_pixels) < int(self.config["min_inliers"]):
            return None
        previous = self.tracked_pixels.reshape(-1, 1, 2).astype(np.float32)
        next_pixels, valid, error = cv2.calcOpticalFlowPyrLK(
            self.previous_gray, gray, previous, None, winSize=(25, 25), maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        if next_pixels is None:
            return None
        back_pixels, back_valid, _ = cv2.calcOpticalFlowPyrLK(
            gray, self.previous_gray, next_pixels, None, winSize=(25, 25), maxLevel=3)
        if back_pixels is None:
            return None
        round_trip = np.linalg.norm(back_pixels.reshape(-1, 2) - previous.reshape(-1, 2), axis=1)
        current = next_pixels.reshape(-1, 2)
        h, w = gray.shape
        keep = (valid.ravel().astype(bool) & back_valid.ravel().astype(bool) &
                (round_trip < 1.5) & (error.ravel() < 32) &
                (current[:, 0] >= 0) & (current[:, 0] < w) &
                (current[:, 1] >= 0) & (current[:, 1] < h))
        return self._solve(current[keep], self.tracked_landmarks[keep])

    def _propagate(self, timestamp_s: float) -> np.ndarray:
        result = self.last_visual_pose.copy()
        dt = max(0, float(timestamp_s - self.last_visual_time))
        result[:3, 3] += self.velocity * dt
        if self.config["motion_model"] == "so3":
            rotation, _ = cv2.Rodrigues(self.angular_velocity * dt)
            result[:3, :3] = rotation @ result[:3, :3]
        else:
            angle = self.yaw_rate * dt
            c, s = np.cos(angle), np.sin(angle)
            result[:3, :3] = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]) @ result[:3, :3]
        return result

    def _plausible(self, fix: dict, timestamp_s: float) -> bool:
        if self.last_visual_pose is None or self.last_status == "lost":
            return True
        dt = timestamp_s - self.last_visual_time
        predicted = self._propagate(timestamp_s)
        distance = np.linalg.norm(fix["transform"][:3, 3] - predicted[:3, 3])
        if self.config["motion_model"] == "so3":
            angular_error = rotation_distance(predicted[:3, :3], fix["transform"][:3, :3])
            return distance <= 5.0 + 120.0 * dt and angular_error <= 0.8 + 4.0 * dt
        predicted_yaw = np.arctan2(predicted[1, 0], predicted[0, 0])
        fixed_yaw = np.arctan2(fix["transform"][1, 0], fix["transform"][0, 0])
        return distance <= 5.0 + 120.0 * dt and abs(wrap_angle(fixed_yaw - predicted_yaw)) <= 0.8 + 4.0 * dt

    def step(self, image: np.ndarray, frame_id: str, timestamp_s: float) -> dict:
        started = time.perf_counter()
        timestamp_s = float(timestamp_s)
        if not np.isfinite(timestamp_s) or (self.last_timestamp is not None and timestamp_s <= self.last_timestamp):
            raise ValueError("Causal timestamps must be finite and strictly increasing; reset between sequences")
        if image is None or image.shape[:2] != (self.calibration["image_height"], self.calibration["image_width"]):
            raise ValueError("Image dimensions must match the supplied calibration")
        # Estimate the next capture interval from observations already received.
        # An unseen timestamp or future image is never available to this object.
        observed_interval = (None if self.last_timestamp is None else
                             timestamp_s - self.last_timestamp)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
        fix, reason = None, "Visual evidence unavailable"
        use_tracking = self.config["method"] == "tracking"
        refresh = self.step_count % int(self.config["relocalize_every"]) == 0
        if use_tracking and not refresh and float(np.std(gray)) > 2.0:
            fix = self._track(gray)
            reason = "Causal KLT landmark tracking + PnP"
        if fix is None:
            fix = self._map_fix(image, timestamp_s)
            reason = ("Coarse reference retrieval" if self.config["method"] == "retrieval" else
                      "Global map matching + PnP" if self.last_status == "lost" else "Map correction + PnP")
        if fix is not None and not self._plausible(fix, timestamp_s):
            fix, reason = None, "Geometric pose rejected by temporal plausibility gate"
        if fix is not None:
            transform = fix["transform"]
            if self.last_visual_pose is not None and self.last_visual_time is not None:
                dt = timestamp_s - self.last_visual_time
                if dt > 0 and self.last_status != "lost":
                    measured_velocity = (transform[:3, 3] - self.last_visual_pose[:3, 3]) / dt
                    previous_yaw = np.arctan2(self.last_visual_pose[1, 0], self.last_visual_pose[0, 0])
                    current_yaw = np.arctan2(transform[1, 0], transform[0, 0])
                    measured_rate = wrap_angle(current_yaw - previous_yaw) / dt
                    self.velocity = 0.35 * self.velocity + 0.65 * measured_velocity
                    self.yaw_rate = float(0.35 * self.yaw_rate + 0.65 * measured_rate)
                    relative_rotation = transform[:3, :3] @ self.last_visual_pose[:3, :3].T
                    rotation_vector, _ = cv2.Rodrigues(relative_rotation)
                    self.angular_velocity = 0.35 * self.angular_velocity + 0.65 * rotation_vector.ravel() / dt
                elif self.last_status == "lost":
                    self.velocity[:] = 0
                    self.yaw_rate = 0.0
                    self.angular_velocity[:] = 0
            self.last_visual_pose, self.last_visual_time = transform.copy(), timestamp_s
            self.tracked_pixels, self.tracked_landmarks = fix["pixels"], fix["landmarks"]
            inliers, residual = fix["inliers"], fix["residual"]
            status = "localized"
            quality = "weak" if self.config["method"] == "retrieval" else "strong" if inliers >= 25 and residual < 2 else "fair"
            # Pixel residuals and inlier counts are useful diagnostics, but do
            # not define a calibrated covariance measured in meters.
            uncertainty = None
        elif (self.last_visual_pose is not None and
              timestamp_s - self.last_visual_time <= float(self.config["max_propagation_s"])):
            transform = self._propagate(timestamp_s)
            status, quality, inliers, residual = "propagated", "weak", 0, None
            uncertainty = None
            reason = "Visual match unavailable; bounded camera-derived motion propagation"
            self.tracked_pixels = np.empty((0, 2), np.float32)
            self.tracked_landmarks = np.empty(0, np.int32)
        else:
            transform = None
            status, quality, inliers, residual, uncertainty = "lost", "unavailable", 0, None, None
            reason = "No verified visual pose; propagation limit reached" if self.last_visual_pose is not None else "No geometrically verified map match"
            self.tracked_pixels = np.empty((0, 2), np.float32)
            self.tracked_landmarks = np.empty(0, np.int32)
        self.previous_gray = gray.copy()
        self.last_timestamp, self.last_status = timestamp_s, status
        self.step_count += 1
        forecast = None
        if transform is not None:
            horizon = float(self.config["forecast_horizon_s"])
            if self.config["forecast_mode"] == "next_frame" and observed_interval is not None:
                horizon = observed_interval
            target_time = timestamp_s + horizon
            forecast = {"timestamp_s": target_time, "pose": pose_record(self._propagate(target_time), self.pose_target)}
        prediction = {"frame_id": str(frame_id), "timestamp_s": timestamp_s,
                      "pose": None if transform is None else pose_record(transform, self.pose_target),
                      "forecast": forecast, "status": status,
                      "latency_ms": (time.perf_counter() - started) * 1000,
                      "inliers": int(inliers), "reprojection_error_px": residual,
                      "reason": reason, "quality": quality, "uncertainty_m": uncertainty,
                      "uncertainty_note": "Metric covariance unavailable; quality uses uncalibrated geometric diagnostics"}
        if fix is not None and "reference_id" in fix:
            prediction["reference_id"] = fix["reference_id"]
        return prediction

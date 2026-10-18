"""Causal metric planar visual odometry from a calibrated road-facing camera.

The road plane is an explicit fixed assumption in the vehicle frame. Camera
height and attitude supply metric scale. No reference poses, telemetry or
ground-truth samples are accepted by this estimator after initialization.
"""
from __future__ import annotations

import time
from numbers import Integral, Real

import cv2
import numpy as np


DEFAULT_CONFIG = {
    "max_width": 960, "max_corners": 1200, "quality_level": 0.008,
    "min_distance_px": 9, "lk_window_px": 31, "lk_levels": 4,
    "forward_backward_px": 1.0, "lk_error_max": 35.0,
    "road_z_m": 0.0, "min_ground_range_m": 2.0,
    "max_ground_range_m": 35.0, "min_image_y_fraction": 0.38,
    "mask_bottom_fraction": 0.06, "min_inliers": 12,
    "min_inlier_fraction": 0.15, "ransac_iterations": 160,
    "ransac_residual_m": 0.20, "range_residual_fraction": 0.015,
    "min_ground_spread_m": 0.8, "max_speed_mps": 60.0,
    "max_yaw_rate_radps": 3.0, "max_pair_interval_s": 0.5,
    "max_propagation_s": 0.3, "min_image_std": 3.0,
}


def _angle(value: float) -> float:
    return float(np.arctan2(np.sin(value), np.cos(value)))


def _rigid_fit(current: np.ndarray, previous: np.ndarray,
               weights: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Return R,t for previous = R @ current + t, without scale fitting."""
    if weights is None:
        weights = np.ones(len(current), dtype=float)
    weights = np.asarray(weights, float)
    weights = weights / weights.sum()
    a = np.sum(current * weights[:, None], axis=0)
    b = np.sum(previous * weights[:, None], axis=0)
    u, _, vt = np.linalg.svd((current - a).T @ ((previous - b) * weights[:, None]))
    correction = np.eye(2)
    correction[-1, -1] = np.linalg.det(vt.T @ u.T)
    rotation = vt.T @ correction @ u.T
    return rotation, b - rotation @ a


class GroundPlaneOdometry:
    """One-image-at-a-time SE(2) estimator, anchored once in the track frame.

    ``T_BC`` maps optical camera coordinates into vehicle coordinates. The
    optical convention is x right, y down, z forward. ``road_z_m`` declares
    the ground elevation in that vehicle frame, not the world-map elevation.
    """

    def __init__(self, calibration: dict, initial_pose: dict,
                 config: dict | None = None):
        if config is not None and not isinstance(config, dict):
            raise ValueError("Odometry settings must be a mapping")
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        unknown = set(self.config) - set(DEFAULT_CONFIG)
        if unknown:
            raise ValueError(f"Unknown odometry settings: {sorted(unknown)}")
        if any(isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
               for value in self.config.values()):
            raise ValueError("Odometry settings must be real numbers, not booleans or strings")
        try:
            finite = all(np.isfinite(float(value)) for value in self.config.values())
        except (OverflowError, ValueError):
            finite = False
        if not finite:
            raise ValueError("Odometry settings must be finite")
        integers = ("max_width", "max_corners", "lk_window_px", "lk_levels",
                    "min_inliers", "ransac_iterations")
        if any(not isinstance(self.config[key], Integral) for key in integers):
            raise ValueError("Odometry image sizes, feature counts, pyramid levels and iterations must be integers")
        if (self.config["min_inliers"] < 2 or
                self.config["max_corners"] < self.config["min_inliers"] or
                self.config["lk_levels"] < 0 or self.config["lk_window_px"] < 3):
            raise ValueError("Need at least two SE2 inliers, enough corners, nonnegative pyramid levels and a window of at least 3 pixels")
        if not 0 <= self.config["max_propagation_s"] <= 0.3:
            raise ValueError("Visual propagation must be bounded by 0.3 seconds")
        positive = ("max_width", "max_corners", "quality_level", "min_distance_px",
                    "lk_window_px", "forward_backward_px", "lk_error_max",
                    "min_ground_range_m", "max_ground_range_m", "min_inliers",
                    "ransac_iterations", "ransac_residual_m", "min_ground_spread_m",
                    "max_speed_mps", "max_yaw_rate_radps", "max_pair_interval_s")
        if any(self.config[key] <= 0 for key in positive):
            raise ValueError("Odometry sizes, ranges and thresholds must be positive")
        if (self.config["min_ground_range_m"] >= self.config["max_ground_range_m"] or
                not 0 < self.config["quality_level"] <= 1 or
                not 0 < self.config["min_inlier_fraction"] <= 1 or
                not 0 <= self.config["range_residual_fraction"] <= 1 or
                self.config["min_image_std"] < 0 or
                not 0 <= self.config["min_image_y_fraction"] < 1 or
                not 0 <= self.config["mask_bottom_fraction"] < 1 or
                self.config["min_image_y_fraction"] + self.config["mask_bottom_fraction"] >= 1):
            raise ValueError("Invalid range, image region or inlier fraction")
        self.width, self.height = int(calibration["image_width"]), int(calibration["image_height"])
        self.k = np.asarray(calibration["K"], dtype=float)
        self.distortion = np.asarray(calibration["distortion"], dtype=float).ravel()
        self.t_bc = np.asarray(calibration["T_BC"], dtype=float)
        if (self.width <= 0 or self.height <= 0 or self.k.shape != (3, 3) or
                self.t_bc.shape != (4, 4) or not np.isfinite(self.k).all() or
                not np.isfinite(self.distortion).all() or not np.isfinite(self.t_bc).all() or
                self.distortion.size not in (0, 4, 5, 8, 12, 14) or
                self.k[0, 0] <= 0 or self.k[1, 1] <= 0 or
                not np.allclose(self.k[2], [0, 0, 1], atol=1e-8) or
                not np.allclose(self.t_bc[3], [0, 0, 0, 1], atol=1e-8) or
                not np.allclose(self.t_bc[:3, :3].T @ self.t_bc[:3, :3], np.eye(3), atol=1e-5) or
                not np.isclose(np.linalg.det(self.t_bc[:3, :3]), 1.0, atol=1e-5)):
            raise ValueError("Invalid calibrated pinhole camera or rigid T_BC")
        if self.t_bc[2, 3] <= self.config["road_z_m"] + 0.05:
            raise ValueError("Camera centre must lie above the declared vehicle-frame road plane")
        self.pose = np.asarray([initial_pose["x"], initial_pose["y"], initial_pose["yaw"]], float)
        if not np.isfinite(self.pose).all():
            raise ValueError("Initial pose must be finite")
        self.pose[2] = _angle(self.pose[2])
        self.initial_timestamp_ns = self._timestamp(initial_pose["timestamp_ns"])
        self.last_timestamp_ns = None
        self.last_visual_timestamp_ns = None
        self.previous_gray = None
        self.previous_pixels = np.empty((0, 2), np.float32)
        self.body_velocity = np.zeros(2)
        self.yaw_rate = 0.0
        self.image_scale = min(1.0, float(self.config["max_width"]) / self.width)
        self.work_width = max(1, round(self.width * self.image_scale))
        self.work_height = max(1, round(self.height * self.image_scale))
        self.work_k = self.k.copy()
        self.work_k[0] *= self.work_width / self.width
        self.work_k[1] *= self.work_height / self.height
        self.mask = self._road_mask()

    @staticmethod
    def _timestamp(value) -> int:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise ValueError("Exposure timestamp must be an integer number of nanoseconds")
        value = int(value)
        if value < 0:
            raise ValueError("Exposure timestamp must be non-negative")
        return value

    def _ground_points(self, pixels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if not len(pixels):
            return np.empty((0, 2)), np.zeros(0, dtype=bool)
        normalized = cv2.undistortPoints(np.asarray(pixels, np.float64).reshape(-1, 1, 2),
                                         self.work_k, self.distortion).reshape(-1, 2)
        rays = np.c_[normalized, np.ones(len(normalized))] @ self.t_bc[:3, :3].T
        with np.errstate(divide="ignore", invalid="ignore"):
            scale = (float(self.config["road_z_m"]) - self.t_bc[2, 3]) / rays[:, 2]
            ground = self.t_bc[:3, 3] + rays * scale[:, None]
        distance = np.linalg.norm(ground[:, :2], axis=1)
        keep = (np.isfinite(ground).all(axis=1) & (rays[:, 2] < -1e-6) &
                (scale > 0) & (distance >= self.config["min_ground_range_m"]) &
                (distance <= self.config["max_ground_range_m"]))
        return ground[:, :2], keep

    def _road_mask(self) -> np.ndarray:
        yy, xx = np.indices((self.work_height, self.work_width))
        pixels = np.c_[xx.ravel(), yy.ravel()].astype(np.float32)
        _, keep = self._ground_points(pixels)
        mask = keep.reshape(self.work_height, self.work_width)
        mask[:int(self.work_height * self.config["min_image_y_fraction"])] = False
        mask[int(self.work_height * (1 - self.config["mask_bottom_fraction"])):] = False
        return mask.astype(np.uint8) * 255

    def _detect(self, gray: np.ndarray) -> np.ndarray:
        if not np.any(self.mask) or float(np.std(gray[self.mask > 0])) < self.config["min_image_std"]:
            return np.empty((0, 2), np.float32)
        points = cv2.goodFeaturesToTrack(gray, mask=self.mask,
            maxCorners=int(self.config["max_corners"]), qualityLevel=float(self.config["quality_level"]),
            minDistance=float(self.config["min_distance_px"]), blockSize=7)
        return np.empty((0, 2), np.float32) if points is None else points.reshape(-1, 2)

    def _motion(self, previous: np.ndarray, current: np.ndarray) -> dict | None:
        previous_ground, a = self._ground_points(previous)
        current_ground, b = self._ground_points(current)
        keep = a & b
        before, after = previous_ground[keep], current_ground[keep]
        if len(before) < int(self.config["min_inliers"]):
            return None
        thresholds = (float(self.config["ransac_residual_m"]) +
                      float(self.config["range_residual_fraction"]) *
                      np.minimum(np.linalg.norm(before, axis=1), np.linalg.norm(after, axis=1)))
        rng = np.random.default_rng(71380)
        best, best_cost = np.zeros(len(before), bool), float("inf")
        for _ in range(int(self.config["ransac_iterations"])):
            sample = rng.choice(len(before), 2, replace=False)
            if (np.linalg.norm(after[sample[0]] - after[sample[1]]) < self.config["min_ground_spread_m"] or
                    np.linalg.norm(before[sample[0]] - before[sample[1]]) < self.config["min_ground_spread_m"]):
                continue
            rotation, translation = _rigid_fit(after[sample], before[sample])
            error = np.linalg.norm(after @ rotation.T + translation - before, axis=1)
            inliers = error <= thresholds
            cost = float(np.median(error[inliers] / thresholds[inliers])) if np.any(inliers) else float("inf")
            if (int(inliers.sum()), -cost) > (int(best.sum()), -best_cost):
                best, best_cost = inliers, cost
        minimum = max(int(self.config["min_inliers"]), int(np.ceil(len(before) * self.config["min_inlier_fraction"])))
        if best.sum() < minimum:
            return None
        for _ in range(3):
            rotation, translation = _rigid_fit(after[best], before[best], 1 / thresholds[best] ** 2)
            error = np.linalg.norm(after @ rotation.T + translation - before, axis=1)
            refined = error <= thresholds
            if refined.sum() < minimum:
                return None
            if np.array_equal(refined, best):
                break
            best = refined
        rotation, translation = _rigid_fit(after[best], before[best], 1 / thresholds[best] ** 2)
        error = np.linalg.norm(after @ rotation.T + translation - before, axis=1)
        spread = np.linalg.svd(after[best] - np.mean(after[best], axis=0), compute_uv=False)
        if spread[0] / np.sqrt(best.sum()) < self.config["min_ground_spread_m"]:
            return None
        return {"translation": translation, "yaw": float(np.arctan2(rotation[1, 0], rotation[0, 0])),
                "inliers": int(best.sum()), "tracks": int(len(before)),
                "residual_m": float(np.median(error[best]))}

    def _advance(self, translation: np.ndarray, yaw: float) -> None:
        c, s = np.cos(self.pose[2]), np.sin(self.pose[2])
        self.pose[:2] += np.asarray([[c, -s], [s, c]]) @ translation
        self.pose[2] = _angle(self.pose[2] + yaw)

    def step(self, image: np.ndarray, timestamp_ns: int) -> dict:
        started = time.perf_counter()
        timestamp_ns = self._timestamp(timestamp_ns)
        if timestamp_ns < self.initial_timestamp_ns or (self.last_timestamp_ns is not None and timestamp_ns <= self.last_timestamp_ns):
            raise ValueError("Exposure timestamps must increase strictly from the supplied initial pose")
        if (not isinstance(image, np.ndarray) or image.dtype != np.uint8 or
                image.shape[:2] != (self.height, self.width) or
                image.ndim not in (2, 3) or (image.ndim == 3 and image.shape[2] != 3)):
            raise ValueError("Expected a uint8 grayscale or BGR image matching camera calibration")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
        if gray.shape != (self.work_height, self.work_width):
            gray = cv2.resize(gray, (self.work_width, self.work_height), interpolation=cv2.INTER_AREA)
        dt = None if self.last_timestamp_ns is None else (timestamp_ns - self.last_timestamp_ns) / 1_000_000_000
        motion, reason, good_tracks = None, "No previous image pair", 0
        if self.previous_gray is not None and len(self.previous_pixels) >= int(self.config["min_inliers"]):
            if dt > float(self.config["max_pair_interval_s"]):
                reason = "Image interval exceeds the configured optical-flow bound"
            elif not np.any(self.mask) or float(np.std(gray[self.mask > 0])) < self.config["min_image_std"]:
                reason = "Insufficient image contrast in the calibrated road region"
            else:
                old = self.previous_pixels.reshape(-1, 1, 2)
                lk = {"winSize": (int(self.config["lk_window_px"]),) * 2,
                      "maxLevel": int(self.config["lk_levels"]),
                      "criteria": (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)}
                current, valid, error = cv2.calcOpticalFlowPyrLK(self.previous_gray, gray, old, None, **lk)
                if current is not None:
                    backward, back_valid, _ = cv2.calcOpticalFlowPyrLK(gray, self.previous_gray, current, None, **lk)
                    if backward is not None:
                        new = current.reshape(-1, 2)
                        inside = ((new[:, 0] >= 0) & (new[:, 0] < self.work_width) &
                                  (new[:, 1] >= 0) & (new[:, 1] < self.work_height))
                        good = (valid.ravel().astype(bool) & back_valid.ravel().astype(bool) & inside &
                                np.isfinite(new).all(axis=1) &
                                (np.linalg.norm(backward.reshape(-1, 2) - self.previous_pixels, axis=1) <= self.config["forward_backward_px"]) &
                                (error.ravel() <= self.config["lk_error_max"]))
                        good_tracks = int(good.sum())
                        motion = self._motion(self.previous_pixels[good], new[good])
                        reason = "Road-plane SE2 consensus unavailable" if motion is None else "Calibrated road-plane KLT and robust metric SE2"
        if motion is not None:
            speed = np.linalg.norm(motion["translation"]) / dt
            yaw_rate = abs(motion["yaw"]) / dt
            if speed > self.config["max_speed_mps"] or yaw_rate > self.config["max_yaw_rate_radps"]:
                motion, reason = None, "Image motion rejected by speed or yaw-rate gate"
        if motion is not None:
            self._advance(motion["translation"], motion["yaw"])
            # Express displacement in the current body for the next short prediction.
            c, s = np.cos(motion["yaw"]), np.sin(motion["yaw"])
            self.body_velocity = np.asarray([[c, s], [-s, c]]) @ motion["translation"] / dt
            self.yaw_rate = motion["yaw"] / dt
            self.last_visual_timestamp_ns = timestamp_ns
            status = "tracked"
        elif self.last_timestamp_ns is None:
            status, reason = "initialized", "Supplied first pose, awaiting a visual image pair"
        elif (self.last_visual_timestamp_ns is not None and
              (timestamp_ns - self.last_visual_timestamp_ns) / 1_000_000_000 <= self.config["max_propagation_s"]):
            self._advance(self.body_velocity * dt, self.yaw_rate * dt)
            status = "propagated"
            reason += "; bounded extrapolation of previously observed visual motion"
        else:
            status = "held"
            reason += "; last finite pose held, localization unavailable"
        self.previous_gray, self.previous_pixels = gray, self._detect(gray)
        self.last_timestamp_ns = timestamp_ns
        return {"x": float(self.pose[0]), "y": float(self.pose[1]), "yaw": float(self.pose[2]),
                "status": status, "inliers": 0 if motion is None else motion["inliers"],
                "residual_m": None if motion is None else motion["residual_m"],
                "processing_ms": (time.perf_counter() - started) * 1000,
                "source_timestamp_ns": timestamp_ns, "tracked_pixels": good_tracks,
                "detected_pixels": int(len(self.previous_pixels)), "reason": reason,
                "last_visual_timestamp_ns": self.last_visual_timestamp_ns,
                "metric_scale_source": "calibrated camera mount and declared vehicle-frame road plane"}

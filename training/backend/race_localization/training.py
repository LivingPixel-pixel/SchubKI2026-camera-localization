"""Small diagnostic classifier, trained from train episodes, not an RL policy."""
from __future__ import annotations

import hashlib
import json
import math

import numpy as np

FEATURES = ["log_inliers", "reprojection_error_px", "latency_ms", "localized", "propagated", "missing_reprojection"]


def reliability_protocol(config: dict, context: dict | None = None) -> dict:
    target = (context or {}).get("pose_target", "vehicle")
    if target not in {"vehicle", "camera"}:
        raise ValueError("Unsupported reliability pose target")
    metric = "geodesic" if target == "camera" else "yaw"
    if (context or {}).get("orientation_metric", metric) != metric:
        raise ValueError("Reliability orientation protocol conflicts with pose target")
    position = _numeric(config.get("position_tolerance_m"), 1.0)
    orientation = _numeric(config.get("rotation_tolerance_deg" if target == "camera" else "heading_tolerance_deg"), 2.0)
    if position <= 0 or orientation <= 0:
        raise ValueError("Training tolerances must be positive")
    return {"version": 1, "pose_target": target, "orientation_metric": metric,
            "position_tolerance_m": position, "orientation_tolerance_deg": orientation}


def _numeric(value, fallback=0.0):
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) else fallback


def _features(record: dict) -> np.ndarray:
    residual = record.get("reprojection_error_px")
    present = isinstance(residual, (int, float)) and math.isfinite(residual)
    return np.asarray([math.log1p(max(0, _numeric(record.get("inliers")))),
                       min(1000, max(0, _numeric(residual))),
                       min(10000, max(0, _numeric(record.get("latency_ms")))),
                       float(record.get("status") == "localized"),
                       float(record.get("status") == "propagated"), float(not present)], dtype=float)


def _sigmoid(value):
    return 1.0 / (1.0 + np.exp(-np.clip(value, -40, 40)))


def train_reliability(records: list[dict], labels: dict, config: dict, *, context: dict | None = None) -> dict:
    """Fit P(error within the explicit position and orientation protocol).

    The API supplies only training records and labels. This function also rejects
    explicit nontraining provenance. Scores are uncalibrated learned quality, and
    do not change pose estimates or authorize labeling a pose correct at runtime.
    """
    if config.get("training_split", "train") != "train" or any(row.get("split", "train") != "train" for row in records):
        raise ValueError("Reliability fitting accepts training episodes only")
    from .evaluation import _pose, _rotation_error, _angle_error
    protocol = reliability_protocol(config, context)
    target = protocol["pose_target"]
    position_tolerance = protocol["position_tolerance_m"]
    orientation_tolerance = protocol["orientation_tolerance_deg"]
    features, targets, frame_ids = [], [], []
    seen = set()
    for row in records:
        fid = row.get("frame_id")
        if fid in seen:
            raise ValueError("Duplicate training record IDs")
        seen.add(fid)
        oracle = labels.get(fid)
        if oracle is None:
            continue
        oracle_pose = _pose(oracle, target)
        if oracle_pose is None:
            raise ValueError("Training oracle needs finite position and target-specific orientation labels")
        if abs(_numeric(row.get("timestamp_s"), math.inf) - _numeric(oracle.get("timestamp_s"), -math.inf)) > 1e-6:
            raise ValueError("Training labels must align to capture timestamps")
        pose = _pose(row.get("pose"), target)
        success = False
        if isinstance(pose, dict) and row.get("status") in {"localized", "propagated"}:
            try:
                position = np.asarray(pose["position_m"], dtype=float)
                angle = _rotation_error(pose, oracle_pose) if target == "camera" else _angle_error(pose["yaw_rad"], oracle_pose["yaw_rad"])
            except (TypeError, ValueError, KeyError):
                position, angle = np.full(3, np.nan), math.nan
            if position.shape == (3,) and np.isfinite(position).all() and math.isfinite(angle):
                error = float(np.linalg.norm(position - np.asarray(oracle["position_m"])))
                success = error <= position_tolerance and angle <= orientation_tolerance
        features.append(_features(row))
        targets.append(float(success))
        frame_ids.append(fid)
    if len(features) < 8:
        raise ValueError("Reliability fitting needs at least 8 timestamp-aligned labeled training predictions")
    y = np.asarray(targets)
    if np.unique(y).size != 2:
        raise ValueError("Training diagnostics need both correct and incorrect/lost examples; a one-class fit is not useful")
    x = np.asarray(features)
    mean, scale = x.mean(axis=0), x.std(axis=0)
    scale = np.where(scale < 1e-8, 1.0, scale)
    normalized = (x - mean) / scale
    design = np.column_stack([np.ones(len(x)), normalized])
    coefficients = np.zeros(design.shape[1])
    loss_history = []
    learning_rate, penalty = 0.15, 0.01
    for epoch in range(300):
        probabilities = _sigmoid(design @ coefficients)
        regularizer = coefficients.copy()
        regularizer[0] = 0
        gradient = design.T @ (probabilities - y) / len(y) + penalty * regularizer
        coefficients -= learning_rate * gradient
        if epoch % 10 == 0 or epoch == 299:
            probabilities = _sigmoid(design @ coefficients)
            loss = -np.mean(y * np.log(np.clip(probabilities, 1e-10, 1))
                            + (1 - y) * np.log(np.clip(1 - probabilities, 1e-10, 1)))
            loss_history.append({"epoch": epoch + 1, "loss": float(loss + penalty * np.sum(coefficients[1:] ** 2) / 2)})
    model = {"kind": "diagnostic_logistic_quality", "feature_names": FEATURES,
             "coefficients": coefficients.tolist(), "normalization": {"mean": mean.tolist(), "scale": scale.tolist()},
             "samples": len(y), "positive_samples": int(y.sum()), "negative_samples": int(len(y) - y.sum()),
             "position_tolerance_m": position_tolerance,
             "heading_tolerance_deg": orientation_tolerance if target == "vehicle" else None,
             "rotation_tolerance_deg": orientation_tolerance if target == "camera" else None,
             "target_protocol": protocol, "evaluation": dict(context or {}),
             "feature_hash": config.get("feature_hash"),
             "training_split": "train", "training_frame_ids": frame_ids, "loss_history": loss_history,
             "score_meaning": "probability that pose meets declared tolerances", "calibration_status": "uncalibrated",
             "limitations": ["Training diagnostic fit; validate on separate complete development episodes.",
                              "A learned quality score is not independent proof of correctness.",
                              "This model neither alters geometric poses nor learns visual descriptors."]}
    model["hash"] = hashlib.sha256(json.dumps(model, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    return model


def predict_reliability(model: dict, prediction: dict) -> float:
    if model.get("kind") != "diagnostic_logistic_quality" or model.get("feature_names") != FEATURES:
        raise ValueError("Unsupported reliability model")
    coefficients = np.asarray(model["coefficients"], dtype=float)
    mean = np.asarray(model["normalization"]["mean"], dtype=float)
    scale = np.asarray(model["normalization"]["scale"], dtype=float)
    if coefficients.shape != (len(FEATURES) + 1,) or mean.shape != (len(FEATURES),) or scale.shape != mean.shape:
        raise ValueError("Reliability model dimensions are invalid")
    if not np.isfinite(coefficients).all() or not np.isfinite(mean).all() or not np.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError("Reliability model contains invalid coefficients or scales")
    normalized = (_features(prediction) - mean) / scale
    return float(_sigmoid(coefficients[0] + normalized @ coefficients[1:]))

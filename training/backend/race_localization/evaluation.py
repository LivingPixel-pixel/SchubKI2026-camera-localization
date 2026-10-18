"""All-expected-frame scoring, separate from camera-only inference."""
from __future__ import annotations

import math
from bisect import bisect_left, bisect_right
from typing import Any

import numpy as np


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _pose(value: Any, pose_target: str = "vehicle") -> dict | None:
    if not isinstance(value, dict):
        return None
    position, yaw = value.get("position_m"), value.get("yaw_rad")
    if (not isinstance(position, (list, tuple)) or len(position) != 3
            or not all(_finite(x) for x in position)):
        return None
    if pose_target == "camera":
        raw = value.get("T_WC")
        if (not isinstance(raw, (list, tuple)) or len(raw) != 4
                or any(not isinstance(row, (list, tuple)) or len(row) != 4 for row in raw)
                or any(not _finite(item) for row in raw for item in row)):
            return None
        transform = np.asarray(raw, dtype=float)
        rotation = transform[:3, :3]
        if (not np.allclose(transform[3], [0, 0, 0, 1], atol=1e-6, rtol=0)
                or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-4, rtol=0)
                or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-4, rtol=0)
                or not np.allclose(transform[:3, 3], position, atol=1e-6, rtol=0)):
            return None
        return {"position_m": list(position), "yaw_rad": None, "T_WC": transform.tolist()}
    if not _finite(yaw):
        return None
    return {"position_m": list(position), "yaw_rad": float(yaw)}


def _percentile(values: list[float], percentile: float) -> float | None:
    return float(np.percentile(values, percentile)) if values else None


def _angle_error(predicted: float, true: float) -> float:
    delta = predicted - true
    return abs(math.atan2(math.sin(delta), math.cos(delta))) * 180 / math.pi


def _rotation_error(predicted: dict, true: dict) -> float:
    first, second = np.asarray(predicted["T_WC"])[:3, :3], np.asarray(true["T_WC"])[:3, :3]
    cosine = np.clip((np.trace(first.T @ second) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def evaluate_run(predictions: list[dict], expected_frames: list[dict], labels: dict | None,
                 config: dict, *, context: dict | None = None) -> dict:
    """Score predictions without trajectory alignment; absent oracle data stays unavailable.

    Conditional error percentiles use frames with valid predictions and labels. Coverage
    and joint success retain all expected frames. Partial labels yield no aggregate
    joint-success claim, with a separately identified labeled-only score instead.
    """
    if not expected_frames:
        raise ValueError("Evaluation needs at least one expected camera frame")
    metadata = dict(context or {})
    target = metadata.get("pose_target", "vehicle")
    if not isinstance(target, str) or target not in {"vehicle", "camera"}:
        raise ValueError("Evaluation pose_target must explicitly be vehicle or camera")
    metadata["pose_target"] = target
    metadata.setdefault("orientation_metric", "geodesic" if target == "camera" else "yaw")
    if metadata["orientation_metric"] != ("geodesic" if target == "camera" else "yaw"):
        raise ValueError("Orientation metric disagrees with the declared pose target")
    camera_target = target == "camera"
    expected_ids = [str(frame["frame_id"]) for frame in expected_frames]
    timestamps = [frame["timestamp_s"] for frame in expected_frames]
    if len(set(expected_ids)) != len(expected_ids):
        raise ValueError("Expected frame IDs must be unique")
    if not all(_finite(t) for t in timestamps) or any(b <= a for a, b in zip(timestamps, timestamps[1:])):
        raise ValueError("Expected capture times must be finite and strictly increasing")
    position_tolerance = config.get("position_tolerance_m", 1.0)
    heading_tolerance = config.get("heading_tolerance_deg", 2.0)
    rotation_tolerance = config.get("rotation_tolerance_deg", 2.0)
    if (not _finite(position_tolerance) or position_tolerance <= 0
            or not _finite(rotation_tolerance if camera_target else heading_tolerance)
            or (rotation_tolerance if camera_target else heading_tolerance) <= 0):
        raise ValueError("Evaluation tolerances must be finite and positive")
    forecast_mode = config.get("forecast_mode", "next_frame")
    forecast_horizon = config.get("forecast_horizon_s", 0.1)
    if forecast_mode not in {"next_frame", "fixed_horizon"} or not _finite(forecast_horizon) or forecast_horizon <= 0:
        raise ValueError("Forecast evaluation needs a declared mode and positive horizon")
    # Target eligibility depends on the declared protocol and independent oracle,
    # never on whether a candidate elects to emit a forecast at that frame.
    oracle_by_time: dict[float, list[dict]] = {}
    if isinstance(labels, dict):
        for row in labels.values():
            if isinstance(row, dict) and _pose(row, target) is not None and _finite(row.get("timestamp_s")):
                oracle_by_time.setdefault(float(row["timestamp_s"]), []).append(row)
    oracle_times = sorted(oracle_by_time)
    by_id: dict[str, dict] = {}
    duplicates: set[str] = set()
    unknown = []
    expected_set = set(expected_ids)
    for prediction in predictions:
        fid = str(prediction.get("frame_id", ""))
        if fid not in expected_set:
            unknown.append(fid)
        elif fid in by_id:
            duplicates.add(fid)
        else:
            by_id[fid] = prediction
    position_errors, heading_errors, planar_errors, rotation_errors, latencies = [], [], [], [], []
    forecast_position_errors, forecast_heading_errors = [], []
    forecast_rotation_errors = []
    forecast_evaluable_count = forecast_valid_count = forecast_joint_count = 0
    valid_count = localized_count = propagated_count = labeled_count = joint_count = wrong_count = 0
    recovery_count = 0
    per_frame = []
    outage_start: float | None = None
    longest_outage = 0.0
    for frame_index, (fid, timestamp) in enumerate(zip(expected_ids, timestamps)):
        prediction = by_id.get(fid)
        reason = "missing prediction" if prediction is None else prediction.get("reason", "")
        if fid in duplicates:
            prediction, reason = None, "duplicate prediction records"
        timestamp_valid = prediction is not None and _finite(prediction.get("timestamp_s")) and abs(prediction["timestamp_s"] - timestamp) <= 1e-6
        if prediction is not None and not timestamp_valid:
            reason = "prediction capture timestamp mismatch"
        status = prediction.get("status", "lost") if prediction is not None else "lost"
        if status not in {"localized", "propagated", "lost"}:
            reason, status = "invalid localization status", "lost"
        pose = _pose(prediction.get("pose"), target) if timestamp_valid and prediction else None
        valid = pose is not None and status != "lost"
        if valid:
            valid_count += 1
            localized_count += status == "localized"
            propagated_count += status == "propagated"
        else:
            pose, status = None, "lost"
        # An outage here means no valid pose; propagated coverage is separately visible.
        if not valid and outage_start is None:
            outage_start = float(timestamp)
        elif valid and outage_start is not None:
            longest_outage = max(longest_outage, timestamp - outage_start)
            recovery_count += 1
            outage_start = None
        latency = prediction.get("latency_ms") if prediction else None
        if _finite(latency) and latency >= 0:
            latencies.append(float(latency))
        oracle = labels.get(fid) if isinstance(labels, dict) else None
        oracle_pose = _pose(oracle, target)
        oracle_valid = oracle_pose is not None and _finite(oracle.get("timestamp_s")) and abs(oracle["timestamp_s"] - timestamp) <= 1e-6
        position_error = planar_error = heading_error = rotation_error = None
        success: bool | None = None
        if oracle_valid:
            labeled_count += 1
            success = False
            if valid:
                difference = np.asarray(pose["position_m"]) - np.asarray(oracle_pose["position_m"])
                position_error = float(np.linalg.norm(difference))
                position_errors.append(position_error)
                if camera_target:
                    rotation_error = _rotation_error(pose, oracle_pose)
                    rotation_errors.append(rotation_error)
                    success = position_error <= position_tolerance and rotation_error <= rotation_tolerance
                else:
                    planar_error = float(np.linalg.norm(difference[:2]))
                    heading_error = _angle_error(pose["yaw_rad"], oracle_pose["yaw_rad"])
                    planar_errors.append(planar_error)
                    heading_errors.append(heading_error)
                    success = position_error <= position_tolerance and heading_error <= heading_tolerance
                joint_count += success
                if not success and status == "localized" and prediction.get("quality") == "strong":
                    wrong_count += 1
        forecast_error = forecast_heading_error = forecast_rotation_error = None
        horizon = timestamps[frame_index] - timestamps[frame_index - 1] if forecast_mode == "next_frame" and frame_index else forecast_horizon
        expected_target = float(timestamp + horizon)
        # Accept an oracle exactly at the protocol target within timestamp roundoff.
        matching_times = oracle_times[bisect_left(oracle_times, expected_target - 1e-6):
                                      bisect_right(oracle_times, expected_target + 1e-6)]
        target_oracles = [row for time in matching_times for row in oracle_by_time[time]]
        forecast_evaluable = len(target_oracles) == 1
        forecast_valid = False
        forecast_success: bool | None = False if forecast_evaluable else None
        if forecast_evaluable:
            forecast_evaluable_count += 1
            forecast = prediction.get("forecast") if prediction and timestamp_valid else None
            forecast_pose = _pose(forecast.get("pose"), metadata["pose_target"]) if isinstance(forecast, dict) else None
            target_time = forecast.get("timestamp_s") if isinstance(forecast, dict) else None
            if forecast_pose is not None and _finite(target_time) and abs(target_time - expected_target) <= 1e-6 and target_time > timestamp:
                future = target_oracles[0]
                forecast_valid = True
                forecast_valid_count += 1
                forecast_error = float(np.linalg.norm(np.asarray(forecast_pose["position_m"]) - np.asarray(future["position_m"])))
                forecast_position_errors.append(forecast_error)
                if camera_target:
                    forecast_rotation_error = _rotation_error(forecast_pose, future)
                    forecast_rotation_errors.append(forecast_rotation_error)
                    forecast_success = forecast_error <= position_tolerance and forecast_rotation_error <= rotation_tolerance
                else:
                    forecast_heading_error = _angle_error(forecast_pose["yaw_rad"], future["yaw_rad"])
                    forecast_heading_errors.append(forecast_heading_error)
                    forecast_success = forecast_error <= position_tolerance and forecast_heading_error <= heading_tolerance
                forecast_joint_count += forecast_success
        per_frame.append({"frame_id": fid, "timestamp_s": float(timestamp), "status": status,
                          "prediction_valid": valid, "ground_truth_available": bool(oracle_valid),
                          "position_error_m": position_error, "planar_position_error_m": planar_error,
                          "heading_error_deg": heading_error, "rotation_error_deg": rotation_error, "joint_success": success,
                          "forecast_position_error_m": forecast_error, "forecast_heading_error_deg": forecast_heading_error,
                          "forecast_rotation_error_deg": forecast_rotation_error,
                          "forecast_target_timestamp_s": expected_target, "forecast_evaluable": forecast_evaluable,
                          "forecast_valid": forecast_valid, "forecast_joint_success": forecast_success,
                          "latency_ms": float(latency) if _finite(latency) and latency >= 0 else None, "reason": reason})
    trailing_outage_censored = outage_start is not None
    if outage_start is not None:
        # No next capture was observed. Report the known interval, right-censored
        # at the actual last capture, instead of inventing an extra frame period.
        longest_outage = max(longest_outage, timestamps[-1] - outage_start)
    count = len(expected_frames)
    total_latency = sum(latencies)
    complete_labels = labeled_count == count
    return {**metadata, "evaluation": metadata,
            "world_origin_qualification": "Shared recording origin is inferred from source documentation, not certified; no query alignment is applied." if metadata.get("alignment_status") == "inferred_shared" else None,
            "expected_frames": count, "valid_frames": valid_count, "labeled_frames": labeled_count,
            "ground_truth_available": labeled_count > 0, "ground_truth_complete": complete_labels,
            "label_coverage_pct": 100.0 * labeled_count / count,
            "median_position_error_m": _percentile(position_errors, 50),
            "p95_position_error_m": _percentile(position_errors, 95),
            "max_position_error_m": max(position_errors) if position_errors else None,
            "median_planar_position_error_m": _percentile(planar_errors, 50),
            "median_heading_error_deg": _percentile(heading_errors, 50),
            "p95_heading_error_deg": _percentile(heading_errors, 95),
            "max_heading_error_deg": max(heading_errors) if heading_errors else None,
            "median_rotation_error_deg": _percentile(rotation_errors, 50),
            "p95_rotation_error_deg": _percentile(rotation_errors, 95),
            "max_rotation_error_deg": max(rotation_errors) if rotation_errors else None,
            "coverage_pct": 100.0 * valid_count / count,
            "joint_success_pct": 100.0 * joint_count / count if complete_labels else None,
            "labeled_joint_success_pct": 100.0 * joint_count / labeled_count if labeled_count else None,
            "localized_pct": 100.0 * localized_count / count,
            "propagated_pct": 100.0 * propagated_count / count,
            "p95_latency_ms": _percentile(latencies, 95), "median_latency_ms": _percentile(latencies, 50),
            "throughput_fps": 1000.0 * count / total_latency if len(latencies) == count and total_latency > 0 else None,
            "longest_outage_s": float(longest_outage), "recovery_count": recovery_count,
            "trailing_outage_censored": trailing_outage_censored,
            "median_forecast_position_error_m": _percentile(forecast_position_errors, 50),
            "p95_forecast_position_error_m": _percentile(forecast_position_errors, 95),
            "max_forecast_position_error_m": max(forecast_position_errors) if forecast_position_errors else None,
            "median_forecast_heading_error_deg": _percentile(forecast_heading_errors, 50),
            "p95_forecast_heading_error_deg": _percentile(forecast_heading_errors, 95),
            "median_forecast_rotation_error_deg": _percentile(forecast_rotation_errors, 50),
            "p95_forecast_rotation_error_deg": _percentile(forecast_rotation_errors, 95),
            "max_forecast_rotation_error_deg": max(forecast_rotation_errors) if forecast_rotation_errors else None,
            "forecast_evaluable_frames": forecast_evaluable_count, "forecast_valid_frames": forecast_valid_count,
            "forecast_coverage_pct": 100.0 * forecast_valid_count / forecast_evaluable_count if forecast_evaluable_count else None,
            "forecast_joint_success_pct": 100.0 * forecast_joint_count / forecast_evaluable_count if forecast_evaluable_count else None,
            "forecast_mode": forecast_mode,
            "forecast_evaluation_scope": "exact oracle at independently derived protocol target; missing forecasts fail coverage/success",
            "wrong_confident_fixes": wrong_count if labeled_count else None,
            "duplicate_prediction_ids": sorted(duplicates), "unknown_prediction_ids": unknown,
            "position_tolerance_m": float(position_tolerance), "heading_tolerance_deg": None if camera_target else float(heading_tolerance),
            "rotation_tolerance_deg": float(rotation_tolerance) if camera_target else None,
            "accuracy_scope": "all expected frames" if complete_labels else "partial labels" if labeled_count else "unavailable",
            "throughput_scope": "sum of recorded per-frame inference latencies; excludes startup/map construction",
            "outage_definition": "no valid current pose; propagated estimates count as coverage; trailing outages end at last capture and are censored",
            "per_frame": per_frame}


def aggregate_runs(runs: list[dict], config: dict, context: dict | None = None) -> dict:
    """Pool all scored frames while preserving session-local outages/forecast joins."""
    if not runs:
        raise ValueError("No scored runs to aggregate")
    metrics = [run["metrics"] for run in runs]
    target = (context or {}).get("pose_target", "vehicle")
    rows = []
    for run, scored in zip(runs, metrics):
        if scored.get("pose_target", "vehicle") != target:
            raise ValueError("Cannot pool distinct pose targets")
        if len(scored.get("per_frame", [])) != scored.get("expected_frames"):
            raise ValueError("Aggregation requires every expected scored frame")
        for key in ("position_tolerance_m", "heading_tolerance_deg", "rotation_tolerance_deg", "forecast_mode"):
            if scored.get(key) != metrics[0].get(key):
                raise ValueError("Cannot pool runs scored under different protocols")
            if key in config and scored.get(key) is not None and scored[key] != config[key]:
                raise ValueError("Aggregate configuration disagrees with scored protocol")
        if context:
            for key, value in context.items():
                if scored.get(key) != value:
                    raise ValueError("Aggregate context disagrees with scored run")
        rows.extend({**row, "run_id": run.get("id"), "session_id": run.get("session_id")}
                    for row in scored["per_frame"])
    if not rows:
        raise ValueError("No expected scored frames to aggregate")
    count = len(rows)
    valid = sum(row.get("prediction_valid") is True for row in rows)
    labeled = sum(row.get("ground_truth_available") is True for row in rows)
    successful = sum(row.get("joint_success") is True for row in rows)
    eligible = sum(row.get("forecast_evaluable") is True for row in rows)
    forecast_valid = sum(row.get("forecast_valid") is True for row in rows)
    forecast_success = sum(row.get("forecast_joint_success") is True for row in rows)
    latencies = [row["latency_ms"] for row in rows if _finite(row.get("latency_ms")) and row["latency_ms"] >= 0]
    result = {key: value for key, value in metrics[0].items() if key != "per_frame"}
    for prefix, field in (("position", "position_error_m"), ("heading", "heading_error_deg"), ("rotation", "rotation_error_deg"),
                          ("forecast_position", "forecast_position_error_m"), ("forecast_heading", "forecast_heading_error_deg"),
                          ("forecast_rotation", "forecast_rotation_error_deg")):
        values = [float(row[field]) for row in rows if _finite(row.get(field))]
        suffix = "m" if "position" in prefix else "deg"
        result[f"median_{prefix}_error_{suffix}"] = _percentile(values, 50)
        result[f"p95_{prefix}_error_{suffix}"] = _percentile(values, 95)
        result[f"max_{prefix}_error_{suffix}"] = max(values) if values else None
    planar = [float(row["planar_position_error_m"]) for row in rows if _finite(row.get("planar_position_error_m"))]
    result.update({"expected_frames": count, "valid_frames": valid, "labeled_frames": labeled,
        "ground_truth_available": labeled > 0, "ground_truth_complete": labeled == count,
        "label_coverage_pct": 100 * labeled / count, "coverage_pct": 100 * valid / count,
        "localized_pct": 100 * sum(row.get("prediction_valid") is True and row["status"] == "localized" for row in rows) / count,
        "propagated_pct": 100 * sum(row.get("prediction_valid") is True and row["status"] == "propagated" for row in rows) / count,
        "joint_success_pct": 100 * successful / count if labeled == count else None,
        "labeled_joint_success_pct": 100 * successful / labeled if labeled else None,
        "median_planar_position_error_m": _percentile(planar, 50),
        "median_latency_ms": _percentile(latencies, 50), "p95_latency_ms": _percentile(latencies, 95),
        "throughput_fps": 1000 * count / sum(latencies) if len(latencies) == count and sum(latencies) > 0 else None,
        "longest_outage_s": max(scored["longest_outage_s"] for scored in metrics),
        "recovery_count": sum(scored["recovery_count"] for scored in metrics),
        "trailing_outage_censored": any(scored.get("trailing_outage_censored", False) for scored in metrics),
        "wrong_confident_fixes": sum(scored.get("wrong_confident_fixes") or 0 for scored in metrics) if labeled else None,
        "forecast_evaluable_frames": eligible, "forecast_valid_frames": forecast_valid,
        "forecast_coverage_pct": 100 * forecast_valid / eligible if eligible else None,
        "forecast_joint_success_pct": 100 * forecast_success / eligible if eligible else None,
        "accuracy_scope": "all expected frames" if labeled == count else "partial labels" if labeled else "unavailable",
        "duplicate_prediction_ids": [{"run_id": run.get("id"), "frame_id": fid} for run in runs for fid in run["metrics"].get("duplicate_prediction_ids", [])],
        "unknown_prediction_ids": [{"run_id": run.get("id"), "frame_id": fid} for run in runs for fid in run["metrics"].get("unknown_prediction_ids", [])],
        "aggregation_scope": "Pooled errors and all expected frames; forecasts, recoveries and outages remain session-local.",
        "per_frame": rows})
    return result


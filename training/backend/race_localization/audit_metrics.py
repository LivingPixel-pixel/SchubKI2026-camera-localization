"""Pool independently scored sessions without crossing their time boundaries."""
import numpy as np


def aggregate_audit_runs(runs: list[dict]) -> dict:
    if not runs:
        raise ValueError("No completed audit sessions to aggregate.")
    rows = [row for run in runs for row in run["metrics"]["per_frame"]]
    count = len(rows)
    metrics = [run["metrics"] for run in runs]

    def percentile(key, value):
        values = [row[key] for row in rows if row.get(key) is not None]
        return float(np.percentile(values, value)) if values else None

    valid = sum(bool(row.get("prediction_valid")) for row in rows)
    labeled = sum(bool(row.get("ground_truth_available")) for row in rows)
    successes = sum(row.get("joint_success") is True for row in rows)
    localized = sum(row.get("prediction_valid") and row["status"] == "localized" for row in rows)
    propagated = sum(row.get("prediction_valid") and row["status"] == "propagated" for row in rows)
    forecast_eligible = sum(m.get("forecast_evaluable_frames", 0) for m in metrics)
    forecast_valid = sum(m.get("forecast_valid_frames", 0) for m in metrics)
    forecast_success = sum(row.get("forecast_joint_success") is True for row in rows)
    latencies = [row["latency_ms"] for row in rows if row.get("latency_ms") is not None]
    return {
        "expected_frames": count, "valid_frames": valid, "labeled_frames": labeled,
        "ground_truth_available": labeled > 0, "ground_truth_complete": labeled == count,
        "label_coverage_pct": 100 * labeled / count,
        "median_position_error_m": percentile("position_error_m", 50),
        "p95_position_error_m": percentile("position_error_m", 95),
        "max_position_error_m": percentile("position_error_m", 100),
        "median_heading_error_deg": percentile("heading_error_deg", 50),
        "p95_heading_error_deg": percentile("heading_error_deg", 95),
        "coverage_pct": 100 * valid / count,
        "localized_pct": 100 * localized / count,
        "propagated_pct": 100 * propagated / count,
        "joint_success_pct": 100 * successes / count if labeled == count else None,
        "labeled_joint_success_pct": 100 * successes / labeled if labeled else None,
        "median_latency_ms": percentile("latency_ms", 50), "p95_latency_ms": percentile("latency_ms", 95),
        "throughput_fps": 1000 * count / sum(latencies) if len(latencies) == count and sum(latencies) > 0 else None,
        "longest_outage_s": max(m["longest_outage_s"] for m in metrics),
        "recovery_count": sum(m["recovery_count"] for m in metrics),
        "wrong_confident_fixes": sum(m.get("wrong_confident_fixes") or 0 for m in metrics) if labeled else None,
        "trailing_outage_censored": any(m.get("trailing_outage_censored", False) for m in metrics),
        "forecast_evaluable_frames": forecast_eligible, "forecast_valid_frames": forecast_valid,
        "forecast_coverage_pct": 100 * forecast_valid / forecast_eligible if forecast_eligible else None,
        "median_forecast_position_error_m": percentile("forecast_position_error_m", 50),
        "p95_forecast_position_error_m": percentile("forecast_position_error_m", 95),
        "median_forecast_heading_error_deg": percentile("forecast_heading_error_deg", 50),
        "forecast_joint_success_pct": 100 * forecast_success / forecast_eligible if forecast_eligible else None,
        "position_tolerance_m": metrics[0]["position_tolerance_m"],
        "heading_tolerance_deg": metrics[0]["heading_tolerance_deg"],
        "aggregation_scope": "Pooled per-frame errors and all expected frames across every audit session. Forecast joins and outages remain sequence-local.",
    }

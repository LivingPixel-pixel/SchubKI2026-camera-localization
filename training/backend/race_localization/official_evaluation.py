"""Offline, auditable wrappers around the unchanged Constructor starter scripts.

Ground truth belongs to initialization, structural lap selection and scoring only.
This module never feeds lap poses back into an estimator or restarts its state.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import subprocess
import sys
from pathlib import Path

import numpy as np


SUBMISSION_HEADER = ["timestamp_ns", "x", "y", "yaw"]
GT_HEADER = ["timestamp_ns", "x", "y", "z", "yaw"]
MAX_INT64 = 2**63 - 1
SCORER_MAX_GT_GAP_NS = 50_000_000


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _timestamp(value: str, line: int) -> int:
    if not re.fullmatch(r"[0-9]+", value):
        raise ValueError(f"Row {line}: timestamp_ns must be integer nanoseconds without float conversion")
    result = int(value)
    if result > MAX_INT64:
        raise ValueError(f"Row {line}: timestamp_ns exceeds signed int64")
    return result


def _read_csv(path: Path, *, ground_truth: bool = False) -> tuple[list[str], list[dict]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader, None)
        accepted = [SUBMISSION_HEADER, GT_HEADER] if ground_truth else [SUBMISSION_HEADER]
        if header not in accepted:
            raise ValueError(f"Exact CSV header required: {','.join(GT_HEADER if ground_truth else SUBMISSION_HEADER)}")
        rows = []
        for line, values in enumerate(reader, 2):
            if len(values) != len(header):
                raise ValueError(f"Row {line}: expected {len(header)} fields")
            record = {"timestamp_ns": _timestamp(values[0], line)}
            for column, value in zip(header[1:], values[1:]):
                try:
                    number = float(value)
                except ValueError as exc:
                    raise ValueError(f"Row {line}: {column} must be finite numeric data") from exc
                if not math.isfinite(number):
                    raise ValueError(f"Row {line}: {column} must be finite")
                record[column] = number
            rows.append(record)
    return header, rows


def validate_submission(csv_path, allowed_timestamps=None) -> dict:
    """Reject malformed exports before the more permissive official scorer sees them."""
    path = Path(csv_path).resolve()
    _, rows = _read_csv(path)
    if len(rows) < 2:
        raise ValueError("Submission requires at least 2 rows")
    times = [row["timestamp_ns"] for row in rows]
    if any(current <= previous for previous, current in zip(times, times[1:])):
        raise ValueError("Submission timestamps must be unique and strictly ascending")
    if allowed_timestamps is not None:
        allowed = set()
        for value in allowed_timestamps:
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
                raise ValueError("Allowed timestamps must use integer nanoseconds")
            allowed.add(int(value))
        outside = next((value for value in times if value not in allowed), None)
        if outside is not None:
            raise ValueError(f"Submission timestamp {outside} is not an allowed camera exposure")
    return {"path": str(path), "sha256": _hash_file(path), "rows": len(rows),
            "header": SUBMISSION_HEADER, "first_timestamp_ns": times[0],
            "last_timestamp_ns": times[-1], "strict_integer_nanoseconds": True,
            "allowed_timestamp_check": allowed_timestamps is not None}


def _run_script(script_path, arguments, *, stdout_path: Path, stderr_path: Path) -> dict:
    script = Path(script_path).resolve()
    before = _hash_file(script)
    command = [sys.executable, str(script), *[str(Path(value).resolve()) for value in arguments]]
    result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace",
                            cwd=script.parent, check=False)
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stdout_path.write_text(result.stdout, encoding="utf-8")
    stderr_path.write_text(result.stderr, encoding="utf-8")
    if _hash_file(script) != before:
        raise RuntimeError("Official script changed during execution")
    return {"command": command, "script_path": str(script), "script_sha256": before,
            "exit_code": result.returncode, "stdout": result.stdout, "stderr": result.stderr,
            "stdout_path": str(stdout_path), "stderr_path": str(stderr_path)}


def extract_ground_truth(mcap_path, gt_csv, script_path) -> dict:
    """Run the original extract_gt.py in a distinct offline process."""
    source, output = Path(mcap_path).resolve(), Path(gt_csv).resolve()
    if source == output:
        raise ValueError("Ground-truth output must differ from the recording")
    output.parent.mkdir(parents=True, exist_ok=True)
    report = _run_script(script_path, [source, output],
                         stdout_path=output.with_suffix(".extraction.stdout.txt"),
                         stderr_path=output.with_suffix(".extraction.stderr.txt"))
    report["source"] = {"path": str(source), "sha256": _hash_file(source)}
    if report["exit_code"] != 0:
        _write_json(output.with_suffix(".extraction.json"), report)
        raise RuntimeError(f"Official ground-truth extraction failed: {report['stderr']}")
    _, rows = _read_csv(output, ground_truth=True)
    if len(rows) < 2:
        raise ValueError("Official extraction produced fewer than 2 ground-truth poses")
    report["ground_truth"] = {"path": str(output), "sha256": _hash_file(output), "rows": len(rows)}
    _write_json(output.with_suffix(".extraction.json"), report)
    return report


def write_initial_pose(gt_csv, out_json) -> dict:
    """Export exactly the first data row, with no later-pose initialization leak."""
    source = Path(gt_csv).resolve()
    with source.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader, None)
        if header not in [SUBMISSION_HEADER, GT_HEADER]:
            raise ValueError("Ground-truth CSV has an unexpected header")
        values = next(reader, None)
    if values is None or len(values) != len(header):
        raise ValueError("Ground truth has no valid first pose")
    record = dict(zip(header, values))
    result = {"timestamp_ns": _timestamp(record["timestamp_ns"], 2)}
    for name in ("x", "y", "yaw"):
        result[name] = float(record[name])
        if not math.isfinite(result[name]):
            raise ValueError(f"First-pose {name} must be finite")
    _write_json(Path(out_json).resolve(), result)
    return result


def _ordered_ground_truth(rows: list[dict]) -> list[dict]:
    ordered = sorted(rows, key=lambda row: row["timestamp_ns"])
    unique = []
    for row in ordered:
        if unique and row["timestamp_ns"] == unique[-1]["timestamp_ns"]:
            if any(row[name] != unique[-1][name] for name in ("x", "y", "yaw")):
                raise ValueError("Conflicting ground-truth poses at one timestamp")
        else:
            unique.append(row)
    if len(unique) < 2:
        raise ValueError("Ground truth requires at least 2 distinct timestamps")
    return unique


def define_whole_lap_split(gt_csv, out_dir) -> dict:
    """Discover repeated oriented gates offline and reserve the final two full laps.

    The fixed race-scale criteria reject stationary revisits and short pit loops.
    Ambiguous or insufficient repetitions fail rather than fabricate lap labels.
    No localization outputs or scores are read by this function.
    """
    source = Path(gt_csv).resolve()
    _, raw = _read_csv(source, ground_truth=True)
    rows = _ordered_ground_truth(raw)
    times = np.array([row["timestamp_ns"] for row in rows], dtype=np.int64)
    xy = np.array([[row["x"], row["y"]] for row in rows], dtype=float)
    yaw = np.array([row["yaw"] for row in rows])
    cumulative = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
    criteria = {"gate_radius_m": 12.0, "heading_tolerance_deg": 30.0,
                "minimum_lap_duration_s": 45.0, "minimum_lap_travel_m": 1000.0,
                "minimum_lap_extent_m": 150.0, "anchor_spacing_m": 200.0,
                "maximum_lap_length_fraction_from_median": 0.35}
    # Distance-sampled anchors avoid a frame-rate-dependent search or a pit-only first gate.
    anchors = np.unique(np.searchsorted(cumulative, np.arange(0.0, cumulative[-1], 200.0)))
    best = None
    for anchor in anchors:
        distance = np.linalg.norm(xy - xy[anchor], axis=1)
        heading = np.abs(np.arctan2(np.sin(yaw - yaw[anchor]), np.cos(yaw - yaw[anchor])))
        matching = np.flatnonzero((distance <= 12.0) & (heading <= math.radians(30.0)))
        if not len(matching):
            continue
        groups = np.split(matching, np.flatnonzero(np.diff(matching) > 1) + 1)
        visits = [int(group[np.argmin(distance[group])]) for group in groups]
        accepted = []
        for visit in visits:
            # Merge only fragmented observations within one gate passage. Never
            # combine several short pit circuits into an invented longer lap.
            if accepted and int(times[visit]) - int(times[accepted[-1]]) < 5_000_000_000:
                if distance[visit] < distance[accepted[-1]]:
                    accepted[-1] = visit
            else:
                accepted.append(visit)
        laps = []
        for left, right in zip(accepted, accepted[1:]):
            if (int(times[right]) - int(times[left]) < 45_000_000_000
                    or cumulative[right] - cumulative[left] < 1000.0):
                continue
            extent = float(np.linalg.norm(np.ptp(xy[left:right + 1], axis=0)))
            if extent < 150.0:
                continue
            laps.append({"start_ns": int(times[left]), "end_ns": int(times[right]),
                         "start_gt_row": left, "end_gt_row": right,
                         "duration_s": (int(times[right]) - int(times[left])) / 1e9,
                         "travel_m": float(cumulative[right] - cumulative[left]),
                         "extent_m": extent})
        if len(laps) < 2:
            continue
        median_length = float(np.median([lap["travel_m"] for lap in laps]))
        if any(abs(lap["travel_m"] / median_length - 1) > 0.35 for lap in laps):
            continue
        # More supported complete laps wins; equal support picks the earliest gate.
        key = (len(laps), -int(times[accepted[0]]))
        if best is None or key > best[0]:
            best = (key, int(anchor), laps)
    if best is None:
        report = {"status": "insufficient_whole_laps", "ground_truth_sha256": _hash_file(source),
                  "criteria": criteria, "reason": "No repeated spatial and heading gate yielded at least 2 whole race-scale laps"}
        _write_json(Path(out_dir).resolve() / "split.json", report)
        raise ValueError(report["reason"])
    _, anchor, laps = best
    for index, lap in enumerate(laps, 1):
        lap["name"] = f"lap-{index:02d}"
    result = {"schema": "constructor-whole-lap-split-v1", "status": "defined_before_evaluation",
              "ground_truth_path": str(source), "ground_truth_sha256": _hash_file(source),
              "criteria": criteria, "gate": {"x": float(xy[anchor, 0]), "y": float(xy[anchor, 1]),
                                            "yaw": float(yaw[anchor]), "anchor_gt_row": anchor},
              "definition_reason": "Repeated spatial and heading gate visits with minimum duration, travel, extent and consistent lap lengths; final 2 complete laps reserved before model scoring",
              "whole_laps": laps, "development_laps": laps[:-2], "reserved_laps": laps[-2:],
              "excluded_prefix_end_ns": laps[0]["start_ns"], "excluded_suffix_start_ns": laps[-1]["end_ns"],
              "estimator_initializations": 1, "lap_reseeding_allowed": False,
              "boundary_convention": "closed intervals; a gate boundary pose may belong to both neighboring reports"}
    _write_json(Path(out_dir).resolve() / "split.json", result)
    return result


def _parse_metrics(stdout: str, exit_code: int) -> dict:
    patterns = {
        "poses_submitted": r"poses submitted\s+(\d+)",
        "poses_evaluated": r"poses submitted\s+\d+\s+evaluated\s+(\d+)",
        "coverage_percent": r"coverage \(1 s bins\)\s+([\d.]+)",
        "position_rmse_m": r"position RMSE\s+\[m\]\s+([^\s]+)",
        "lateral_rmse_m": r"lateral RMSE\s+\[m\]\s+([^\s]+)",
        "longitudinal_rmse_m": r"longitudinal RMSE\s+\[m\]\s+([^\s]+)",
        "yaw_rmse_deg": r"yaw RMSE\s+\[deg\]\s+([^\s]+)",
        "within_0_25_m_percent": r"within\s+0\.25 m\s+([^\s]+)",
        "within_0_50_m_percent": r"within\s+0\.50 m\s+([^\s]+)",
        "within_1_00_m_percent": r"within\s+1\.00 m\s+([^\s]+)",
    }
    metrics = {}
    for name, pattern in patterns.items():
        match = re.search(pattern, stdout)
        if match:
            try:
                value = float(match.group(1))
            except ValueError:
                value = float("nan")
            metrics[name] = (int(value) if name.startswith("poses_") else value) if math.isfinite(value) else None
    pair = re.search(r"position median/p95\s+([^\s]+)\s*/\s*([^\s]+)", stdout)
    if pair:
        for name, text in zip(("position_median_m", "position_p95_m"), pair.groups()):
            try:
                value = float(text)
            except ValueError:
                value = float("nan")
            metrics[name] = value if math.isfinite(value) else None
    complete = len(metrics) == len(patterns) + 2 and all(value is not None for value in metrics.values())
    ranked = (exit_code == 0 and complete and "NOT RANKED" not in stdout
              and metrics.get("coverage_percent", 0) >= 90.0 and metrics.get("poses_evaluated", 0) > 0)
    return {"metrics": metrics, "metrics_complete": complete, "ranked": ranked,
            "not_ranked_reason": ("coverage below threshold" if "NOT RANKED" in stdout
                                  else "scorer failed or incomplete/nonfinite metrics" if not ranked else None),
            "metric_precision": "unchanged official script printed precision; full stdout retained"}


def _scope_failures(scope: dict | None, name: str) -> list[dict]:
    if scope is None:
        return [{"scope": name, "reason": "required scorer report is missing"}]
    if scope.get("status") == "not_scored":
        return [{"scope": name, "reason": scope.get("reason", "required interval was not scored")}]
    failures = []
    if scope.get("exit_code") != 0:
        failures.append({"scope": name, "reason": f"official scorer exit code {scope.get('exit_code', 'missing')}"})
    metrics = scope.get("metrics", {})
    finite = (bool(metrics) and all(isinstance(value, (int, float)) and not isinstance(value, bool)
                                   and math.isfinite(value) for value in metrics.values()))
    if not scope.get("metrics_complete", False) or not finite:
        failures.append({"scope": name, "reason": "official scorer metrics are incomplete or nonfinite"})
    return failures


def _evaluation_failures(report: dict) -> list[dict]:
    failures = _scope_failures(report.get("full"), "full")
    split = report.get("split") or {}
    required = split.get("reserved_laps", split.get("intervals", []))
    actual = report.get("reserved_laps", [])
    for index, interval in enumerate(required):
        name = f"reserved-{index + 1:02d} ({interval.get('name', 'unnamed')})"
        scope = actual[index] if index < len(actual) else None
        failures.extend(_scope_failures(scope, name))
    return failures


def require_successful_evaluation(report: dict) -> None:
    """Fail a caller after it saves evidence, independently of ranking eligibility.

    A complete finite low-coverage NOT RANKED result executed successfully. Failed
    scripts, malformed metrics and missing required interval results did not.
    """
    failures = _evaluation_failures(report)
    if failures:
        reason = "; ".join(f"{item['scope']}: {item['reason']}" for item in failures)
        raise RuntimeError(f"Official evaluation failed: {reason}")


def _write_rows(path: Path, header: list[str], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)


def _eligible_count(gt: list[dict], predictions: list[dict]) -> int:
    times = np.array([row["timestamp_ns"] for row in gt], dtype=np.int64)
    count = 0
    for row in predictions:
        index = int(np.searchsorted(times, row["timestamp_ns"]))
        if 0 < index < len(times) and int(times[index]) - int(times[index - 1]) <= SCORER_MAX_GT_GAP_NS:
            count += 1
    return count


def evaluate_official(gt_csv, pred_csv, script_path, out_dir, split=None) -> dict:
    """Score one continuous prediction file and optional untouched final-lap slices."""
    gt_path, pred_path, script = Path(gt_csv).resolve(), Path(pred_csv).resolve(), Path(script_path).resolve()
    output = Path(out_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    validation = validate_submission(pred_path)
    gt_header, gt_raw = _read_csv(gt_path, ground_truth=True)
    gt = _ordered_ground_truth(gt_raw)
    _, predictions = _read_csv(pred_path)
    if _eligible_count(gt, predictions) == 0:
        raise ValueError("No submitted pose has an official eligible ground-truth interpolation bracket; scorer not called")
    split_data = json.loads(Path(split).read_text(encoding="utf-8")) if isinstance(split, (str, Path)) else split
    if split_data is not None:
        if split_data.get("ground_truth_sha256") not in (None, _hash_file(gt_path)):
            raise ValueError("Lap split belongs to a different ground-truth file")
        if split_data.get("lap_reseeding_allowed", False):
            raise ValueError("Lap reseeding is forbidden; predictions must remain continuous")
        intervals = split_data.get("reserved_laps", split_data.get("intervals", []))
        if not intervals:
            raise ValueError("Provided split contains no reserved intervals")
        previous_end = None
        for interval in intervals:
            start, end = interval.get("start_ns"), interval.get("end_ns")
            if (type(start) is not int or type(end) is not int or start < gt[0]["timestamp_ns"]
                    or end > gt[-1]["timestamp_ns"] or start >= end
                    or (previous_end is not None and start < previous_end)):
                raise ValueError("Reserved intervals must be ordered, non-overlapping integer bounds inside ground truth")
            previous_end = end
        _write_json(output / "split-used.json", split_data)
    else:
        intervals = []
    report = {"schema": "constructor-official-evaluation-v1", "submission_validation": validation,
              "ground_truth": {"path": str(gt_path), "sha256": _hash_file(gt_path), "rows": len(gt_raw)},
              "script_sha256": _hash_file(script), "continuous_prediction_file": str(pred_path),
              "estimator_initializations": 1, "lap_reseeding_allowed": False,
              "split": split_data, "reserved_laps": []}
    full = _run_script(script, [gt_path, pred_path], stdout_path=output / "full.stdout.txt", stderr_path=output / "full.stderr.txt")
    full.update(_parse_metrics(full["stdout"], full["exit_code"]))
    full["input_sha256"] = {"ground_truth": _hash_file(gt_path), "predictions": _hash_file(pred_path)}
    full["status"] = "failed" if _scope_failures(full, "full") else "scored"
    report["full"] = full
    times = [row["timestamp_ns"] for row in gt]
    for index, interval in enumerate(intervals, 1):
        name = f"reserved-{index:02d}"
        start, end = interval["start_ns"], interval["end_ns"]
        # Include only the closest brackets beyond a non-sample interval boundary.
        left = int(np.searchsorted(times, start, side="left"))
        if left and times[left] != start:
            left -= 1
        right = int(np.searchsorted(times, end, side="right"))
        if right < len(gt) and times[right - 1] != end:
            right += 1
        selected_gt = gt[left:right]
        selected_pred = [row for row in predictions if start <= row["timestamp_ns"] <= end]
        scope = {"name": interval.get("name", name), "start_ns": start, "end_ns": end,
                 "source_prediction_sha256": validation["sha256"], "prediction_rows": len(selected_pred),
                 "prediction_slice_only": True, "lap_reseeded": False}
        if len(selected_gt) < 2 or len(selected_pred) < 2 or _eligible_count(selected_gt, selected_pred) == 0:
            scope.update({"ranked": False, "status": "not_scored", "reason": "Insufficient predictions or eligible GT brackets; no empty scorer call"})
        else:
            sliced_gt, sliced_pred = output / f"{name}.gt.csv", output / f"{name}.pred.csv"
            _write_rows(sliced_gt, gt_header, selected_gt)
            _write_rows(sliced_pred, SUBMISSION_HEADER, selected_pred)
            score = _run_script(script, [sliced_gt, sliced_pred], stdout_path=output / f"{name}.stdout.txt", stderr_path=output / f"{name}.stderr.txt")
            score.update(_parse_metrics(score["stdout"], score["exit_code"]))
            score["input_sha256"] = {"ground_truth": _hash_file(sliced_gt), "predictions": _hash_file(sliced_pred)}
            scope.update(score)
            scope["status"] = "failed" if _scope_failures(scope, name) else "scored"
        report["reserved_laps"].append(scope)
    report["failures"] = _evaluation_failures(report)
    report["success"] = not report["failures"]
    _write_json(output / "evaluation.json", report)
    return report

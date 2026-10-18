"""Strict official CSV runner; ground truth is not an estimator input."""
from __future__ import annotations
from collections import Counter
from pathlib import Path
import csv
import hashlib
import json
import platform
import os
import subprocess
import time
import cv2
import numpy as np
from .official_io import prepare_recording, read_calibration, iter_frames
from .official_odometry import GroundPlaneOdometry


def hardware_identity() -> dict:
    """Retain the CPU product name, with an explicit generic fallback."""
    result = {"platform": platform.platform(), "processor": platform.processor(),
              "cpu_model": platform.processor() or "unavailable",
              "logical_processors": os.cpu_count(), "opencv_threads": cv2.getNumThreads(),
              "identity_source": "platform.processor fallback"}
    if platform.system() == "Windows":
        try:
            command = "Get-CimInstance Win32_Processor | Select-Object Name,NumberOfCores,NumberOfLogicalProcessors | ConvertTo-Json -Compress"
            query = subprocess.run(["powershell", "-NoProfile", "-Command", command],
                                   capture_output=True, text=True, timeout=10, check=True)
            cpu = json.loads(query.stdout)
            if isinstance(cpu, list):
                cpu = cpu[0]
            result.update(cpu_model=cpu["Name"].strip(), physical_cores=cpu["NumberOfCores"],
                          logical_processors=cpu["NumberOfLogicalProcessors"], identity_source="Win32_Processor")
        except (OSError, subprocess.SubprocessError, ValueError, KeyError, IndexError, TypeError) as error:
            result["identity_warning"] = f"CPU product query unavailable: {type(error).__name__}"
    elif platform.system() == "Linux":
        try:
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("model name"):
                    result.update(cpu_model=line.partition(":")[2].strip(), identity_source="/proc/cpuinfo")
                    break
        except OSError:
            pass
    return result


def run_baseline(mcap: Path, initial_pose: Path, output: Path, cache: Path,
                 camera: str = "fl", config: dict | None = None,
                 max_frames: int | None = None) -> dict:
    started = time.perf_counter()
    seed = json.loads(Path(initial_pose).read_text())
    if set(seed) != {"timestamp_ns", "x", "y", "yaw"}:
        raise ValueError("Initial pose must contain exactly timestamp_ns,x,y,yaw")
    if isinstance(seed["timestamp_ns"], bool) or not isinstance(seed["timestamp_ns"], int):
        raise ValueError("Initial timestamp must be an integer")
    if not np.isfinite([seed["x"], seed["y"], seed["yaw"]]).all():
        raise ValueError("Initial pose must be finite")
    cv2.setNumThreads(1)
    prepared = time.perf_counter()
    manifest = prepare_recording(Path(mcap), Path(cache), camera=camera)
    calibration = read_calibration(manifest)
    preparation_s = time.perf_counter() - prepared
    estimator = GroundPlaneOdometry(calibration, seed, config)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    diagnostic_path = output.with_suffix(".diagnostics.jsonl")
    temporary = output.with_suffix(".csv.partial")
    diagnostics_temporary = diagnostic_path.with_suffix(".partial")
    counts, latencies = Counter(), []
    skipped = 0
    previous = None
    first = last = None
    frames = iter(iter_frames(manifest))
    with temporary.open("w", newline="", encoding="utf-8") as stream, diagnostics_temporary.open("w", encoding="utf-8") as detail:
        writer = csv.writer(stream)
        writer.writerow(["timestamp_ns", "x", "y", "yaw"])
        while max_frames is None or len(latencies) < max_frames:
            frame_start = time.perf_counter()
            try:
                stamp, image_path = next(frames)
            except StopIteration:
                break
            if stamp < seed["timestamp_ns"]:
                skipped += 1
                continue
            if previous is not None and stamp <= previous:
                raise ValueError("Camera exposures must be unique and increasing")
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"Cannot decode camera observation {stamp}")
            prediction = estimator.step(image, stamp)
            values = [prediction[key] for key in ("x", "y", "yaw")]
            if not np.isfinite(values).all() or prediction["source_timestamp_ns"] > stamp:
                raise ValueError("Invalid or noncausal prediction")
            writer.writerow([str(stamp), *(format(float(value), ".10f") for value in values)])
            elapsed = (time.perf_counter() - frame_start) * 1000
            prediction["timestamp_ns"] = stamp
            prediction["decode_io_and_estimator_ms"] = elapsed
            detail.write(json.dumps(prediction, allow_nan=False) + "\n")
            counts[prediction["status"]] += 1
            latencies.append(elapsed)
            first = stamp if first is None else first
            last = previous = stamp
            if len(latencies) % 500 == 0:
                print(json.dumps({"frames": len(latencies), "states": dict(counts)}), flush=True)
    if len(latencies) < 2:
        raise ValueError("At least two post-initialization observations are required")
    temporary.replace(output)
    diagnostics_temporary.replace(diagnostic_path)
    summary = {
        "predictions": str(output.resolve()), "diagnostics": str(diagnostic_path.resolve()),
        "frames": len(latencies), "state_counts": dict(counts), "pre_initialization_frames_skipped": skipped,
        "first_timestamp_ns": first, "last_timestamp_ns": last, "camera": camera,
        "initial_pose": seed, "initialization_count": 1, "config": estimator.config,
        "input_manifest": manifest, "input_topics": manifest["decoder_topic_whitelist"],
        "csv_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "initial_pose_sha256": hashlib.sha256(Path(initial_pose).read_bytes()).hexdigest(),
        "runtime": {"mean_frame_ms": float(np.mean(latencies)), "p95_frame_ms": float(np.percentile(latencies, 95)),
                    "preparation_s": preparation_s, "total_wall_s": time.perf_counter()-started,
                    "boundary": "Cached JPEG integrity/read, decode, causal estimator and output serialization; raw MCAP preparation separately timed",
                    "hardware": hardware_identity(),
                    "gpu_used": False},
        "limitations": ["Flat local road plane and approximate mount calibration", "Uncorrected accumulated visual-odometry drift", "Held outputs are failure fallbacks, not visual localization fixes"],
    }
    output.with_suffix(".run.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    return summary

"""Causal reachable-reference selection over the unchanged frozen ViT gallery.

Only current/past image winners and a last accepted approximate BODY reference
are retained. This is a development-selected continuity pilot, not geometry.
"""
from __future__ import annotations

import csv
import hashlib
import math
import os
import time

import numpy as np

from . import vit_place_localization as base

RUN_SCHEMA = "schubki-vit-causal-run-v1"
METHOD = "causal_topk_reference_v1"
FRAME_TIMING_SCOPE = "read_hash_decode_preprocess_completed_encoder_retrieval_continuity_descriptor_csv_and_diagnostic_write"
POLICY = {"top_k": 64, "max_speed_mps": 100.0, "position_slack_m": 5.0,
          "max_yaw_rate_radps": 3.0, "yaw_slack_rad": 0.35,
          "stale_after_s": 1.0, "recovery_frames": 3, "recovery_elapsed_cap_s": 5.0}


def validate_config(value):
    base._exact(value, {"schema", "stage", "encoder", "manifest", "image_root", "cache",
                        "output_dir", "budgets", "policy"})
    if value["schema"] != RUN_SCHEMA or value["stage"] != "localize":
        raise base.IdentityError("causal localize stage required")
    policy = base._exact(value["policy"], POLICY)
    for key, expected in POLICY.items():
        actual = (base._integer(policy[key], 1, 64) if type(expected) is int
                  else base._finite(policy[key]))
        if actual != expected:
            raise base.IdentityError("frozen causal policy changed")
    inherited = {key: item for key, item in value.items() if key != "policy"}
    inherited["schema"] = base.RUN_SCHEMA
    base.validate_config(inherited)
    return value


def _pose(reference):
    return (float(reference["body_x_m"]), float(reference["body_y_m"]),
            float(reference["body_yaw_rad"]))


def reachable(previous_pose, current_pose, elapsed_s, policy=POLICY):
    """Physical pilot gate, including shortest wrapped yaw difference."""
    if not math.isfinite(elapsed_s) or elapsed_s <= 0:
        return False
    distance = math.hypot(current_pose[0] - previous_pose[0], current_pose[1] - previous_pose[1])
    angle = abs(math.atan2(math.sin(current_pose[2] - previous_pose[2]),
                           math.cos(current_pose[2] - previous_pose[2])))
    return (distance <= policy["max_speed_mps"] * elapsed_s + policy["position_slack_m"]
            and angle <= policy["max_yaw_rate_radps"] * elapsed_s + policy["yaw_slack_rad"])


def ranked_references(query, gallery):
    """Stable full ranking preserves canonical reference order on exact ties."""
    query = base.normalized(query, 1)
    scores = gallery @ query[0]
    if not np.isfinite(scores).all():
        raise base.IdentityError("invalid gallery scores")
    order = np.argsort(-scores, kind="stable")
    return order, scores


class ReachableReferenceSelector:
    """Forward-only selection; rejection never advances the trusted anchor."""

    def __init__(self, references, policy=POLICY):
        self.references = references
        self.policy = dict(policy)
        self.anchor = None
        self.chain = []
        self.last_timestamp_ns = None

    def image_lost(self, timestamp_ns):
        self._timestamp(timestamp_ns)
        self.chain.clear()

    def _timestamp(self, stamp):
        base._integer(stamp, 0, 2**63 - 1)
        if self.last_timestamp_ns is not None and stamp <= self.last_timestamp_ns:
            raise base.IdentityError("causal timestamps must strictly increase")
        self.last_timestamp_ns = stamp

    def select(self, timestamp_ns, order):
        self._timestamp(timestamp_ns)
        if len(order) == 0:
            self.chain.clear()
            return None, "topk_exhausted"
        age = None if self.anchor is None else (timestamp_ns - self.anchor[0]) / 1e9
        if self.anchor is not None and age <= self.policy["stale_after_s"]:
            self.chain.clear()
            for index in order[:self.policy["top_k"]]:
                index = int(index)
                if reachable(self.anchor[1], _pose(self.references[index]), age, self.policy):
                    self.anchor = (timestamp_ns, _pose(self.references[index]))
                    return index, "reachable_topk_reference_copy"
            return None, "topk_exhausted"

        # Startup or recovery: only consecutive current/past global winners can
        # build support. This never expands the old-anchor envelope past 5s.
        winner = int(order[0])
        current = (timestamp_ns, _pose(self.references[winner]))
        if self.chain:
            elapsed = (timestamp_ns - self.chain[-1][0]) / 1e9
            if (elapsed > self.policy["stale_after_s"]
                    or not reachable(self.chain[-1][1], current[1], elapsed, self.policy)):
                self.chain.clear()
        self.chain.append(current)
        self.chain = self.chain[-self.policy["recovery_frames"]:]
        if len(self.chain) < self.policy["recovery_frames"]:
            return None, "coherent_global_support_pending"
        if self.anchor is not None:
            envelope = min(age, self.policy["recovery_elapsed_cap_s"])
            if not reachable(self.anchor[1], current[1], envelope, self.policy):
                return None, "recovery_anchor_unreachable"
        self.anchor = current
        self.chain.clear()
        return winner, "coherent_global_reference_copy"


def _new_array_stream(path, shape, dtype):
    stream = path.open("xb")
    try:
        np.lib.format.write_array_header_1_0(stream, {"descr": np.dtype(dtype).str,
                                                     "fortran_order": False, "shape": shape})
        offset = stream.tell()
        stream.truncate(offset + math.prod(shape) * np.dtype(dtype).itemsize)
        stream.flush()
        return stream, offset
    except BaseException:
        stream.close()
        raise


def run_stage(config_path, expected_config_sha256, *, deadline_monotonic=None):
    """Run exactly one causal query pass; return counts, not scientific values."""
    started = time.monotonic()
    config_pin = base.file_pin(config_path)
    if (config_pin["sha256"] != base._sha(expected_config_sha256)
            or config_pin["bytes"] > 128 * 1024):
        raise base.IdentityError("config identity/cap mismatch")
    config = validate_config(base._json(base.read_pinned(config_pin, 128 * 1024)))
    b = config["budgets"]
    deadline = started + b["wall_seconds"]
    if deadline_monotonic is not None:
        deadline = min(deadline, base._finite(deadline_monotonic))

    def check():
        if time.monotonic() >= deadline:
            raise TimeoutError("finite stage deadline")

    manifest = base.validate_manifest(base._json(base.read_pinned(config["manifest"], b["max_manifest_bytes"])),
                                      "queries", b)
    root, output = base._absolute(config["image_root"]), base._absolute(config["output_dir"])
    if output.exists():
        raise FileExistsError("output directory must be new")
    if not root.is_dir() or not output.parent.is_dir():
        raise base.IdentityError("image root/output parent must exist")
    for row in manifest["rows"]:
        base._relative(root, row["image_path"])
    rows = manifest["rows"]
    if len(rows) * (base.DIMENSION * 4 + 1) + 1024 > min(b["max_cache_bytes"], b["max_output_bytes"]):
        raise base.IdentityError("query descriptor cache cap")
    # Identity deliberately stays bound to the unchanged encoder module.
    encoder_sha = base.encoder_identity(config["encoder"])
    features, references = base._load_cache(config, manifest, encoder_sha)
    check()
    encoder = base.FrozenEncoder(config["encoder"], b, check)
    check()
    artifacts = base.Artifacts(output, b["max_output_bytes"])
    selector = ReachableReferenceSelector(references, config["policy"])
    ready = time.monotonic()
    elapsed, predicted, lost = [], 0, 0
    descriptor_stream, descriptor_offset = _new_array_stream(output / "query_features.npy",
                                                             (len(rows), base.DIMENSION), np.float32)
    try:
        valid_stream, valid_offset = _new_array_stream(output / "query_valid.npy", (len(rows),), np.bool_)
        with descriptor_stream, valid_stream, \
                (output / "predictions.csv").open("x", encoding="ascii", newline="") as csv_stream, \
                (output / "diagnostics.jsonl").open("xb") as diagnostics, \
                (output / "frame_timings.jsonl").open("xb") as completed_timings:
            artifacts.check()
            writer = csv.writer(csv_stream, lineterminator="\n")
            writer.writerow(("timestamp_ns", "x", "y", "yaw"))
            for index, row in enumerate(rows):
                check()
                frame_start = time.monotonic()
                timing = {key: 0.0 for key in ("read_hash_s", "decode_preprocess_s", "encoder_completed_s",
                                               "retrieval_s", "continuity_s", "descriptor_write_s", "output_s", "total_s")}
                record = {"schema": base.DIAGNOSTIC_SCHEMA, "index": index, "frame_id": row["frame_id"],
                          "timestamp_ns": str(row["timestamp_ns"]), "status": "lost", "reason": "",
                          "pose": None, "reference_id": None, "cosine": None, "timing": timing}
                reading, vector = True, None
                try:
                    raw = base._read_image(row, root, b)
                    timing["read_hash_s"] = time.monotonic() - frame_start
                    reading = False
                    vector, parts = encoder.encode_batch([raw])
                    vector = base.normalized(vector, 1)
                    timing.update(parts)
                    search = time.monotonic()
                    order, scores = ranked_references(vector, features)
                    timing["retrieval_s"] = time.monotonic() - search
                    continuity = time.monotonic()
                    choice, reason = selector.select(row["timestamp_ns"], order)
                    timing["continuity_s"] = time.monotonic() - continuity
                    record["reason"] = reason
                    if choice is not None:
                        ref = references[choice]
                        pose = {"x": float(ref["body_x_m"]), "y": float(ref["body_y_m"]),
                                "yaw": float(ref["body_yaw_rad"]), "map_frame": "map",
                                "pose_kind": "approximate_reference_pose", "position_source": "training_reference",
                                "heading_source": "training_reference", "yaw_approximate": True,
                                "translation_scale_observed": False}
                        record.update(status="approximate_reference_pose", pose=pose,
                                      reference_id=ref["frame_id"], cosine=float(np.clip(scores[choice], -1, 1)))
                        predicted += 1
                    else:
                        lost += 1
                except base.ImageFailure as exc:
                    vector = None
                    lost += 1
                    record["reason"] = str(exc)
                    selector.image_lost(row["timestamp_ns"])
                    if reading:
                        timing["read_hash_s"] = time.monotonic() - frame_start
                storing = time.monotonic()
                descriptor_stream.seek(descriptor_offset + index * base.DIMENSION * 4)
                descriptor_stream.write((vector[0] if vector is not None
                                         else np.zeros(base.DIMENSION, np.float32)).tobytes(order="C"))
                valid_stream.seek(valid_offset + index)
                valid_stream.write(b"\x01" if vector is not None else b"\x00")
                descriptor_stream.flush()
                valid_stream.flush()
                timing["descriptor_write_s"] = time.monotonic() - storing
                writing = time.monotonic()
                if record["pose"] is not None:
                    pose = record["pose"]
                    artifacts.check(256)
                    writer.writerow((str(row["timestamp_ns"]), *(repr(pose[k]) for k in ("x", "y", "yaw"))))
                    csv_stream.flush()
                timing["output_s"] = time.monotonic() - writing
                timing["total_s"] = time.monotonic() - frame_start
                data = base.canonical(record)
                artifacts.check(len(data) + 256)
                diagnostics.write(data)
                diagnostics.flush()
                completed = time.monotonic() - frame_start
                elapsed.append(completed)
                complete_timing = base.canonical({"index": index, "timestamp_ns": str(row["timestamp_ns"]),
                                                  "status": record["status"], "total_s": completed,
                                                  "scope": FRAME_TIMING_SCOPE})
                artifacts.check(len(complete_timing))
                completed_timings.write(complete_timing)
                completed_timings.flush()
                artifacts.check()
            for stream in (descriptor_stream, valid_stream, csv_stream, diagnostics, completed_timings):
                stream.flush()
                os.fsync(stream.fileno())
    finally:
        descriptor_stream.close()
    check()
    names = {"predictions": "predictions.csv", "diagnostics": "diagnostics.jsonl",
             "frame_timings": "frame_timings.jsonl", "query_features": "query_features.npy",
             "query_valid": "query_valid.npy"}
    artifact_pins = {key: base.file_pin(output / name) for key, name in names.items()}
    completion = {"schema": base.COMPLETION_SCHEMA, "stage": "localize", "complete": True, "method": METHOD,
                  "camera_id": manifest["camera_id"], "calibration_id": manifest["calibration_id"],
                  "model": base._model_record(config, encoder_sha),
                  "policy": config["policy"], "policy_sha256": hashlib.sha256(base.canonical(config["policy"])).hexdigest(),
                  "inputs": {"config": config_pin, "manifest": config["manifest"],
                             "source_pins": manifest["source_pins"], "cache_completion": config["cache"]["completion"],
                             "causal_source_sha256": base.file_pin(__file__)["sha256"]},
                  "counts": {"expected": len(rows), "processed": len(rows), "predicted": predicted, "lost": lost},
                  "artifacts": artifact_pins,
                  "timing": {"startup_s": ready - started, "processing_s": time.monotonic() - ready,
                             "total_s": time.monotonic() - started,
                             "mean_frame_s": sum(elapsed) / len(elapsed) if elapsed else None,
                             "scope": "completed_per_image_processing;recorded_local_images;not_live_capture_latency"},
                  "limitations": ["development-selected physical pilot assumptions, not measured motion guarantees",
                                   "approximate fit reference BODY copies, no metric geometry recovery",
                                   "first two startup frames are lost; rejected frames emit no fabricated CSV pose",
                                   "descriptor validity means encoding succeeded, not that continuity accepted",
                                   "diagnostic timing stops before its own write; frame_timings includes it"]}
    artifacts.json("completion.json", completion)
    return {"stage": "localize", "complete": True, **completion["counts"],
            "completion_path": str(output / "completion.json")}

"""Loopback product API. Long visual jobs run in a serialized worker queue."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from copy import deepcopy
import csv
import io
import json
import os
from pathlib import Path
import platform
import hashlib
import shutil
import threading
import time

import cv2
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from . import __version__
from .schemas import AuditRequest, BenchmarkCompareRequest, CandidateRequest, MapRequest, RunRequest, SearchRequest, Settings, TrainingRequest
from .storage import Storage, artifact_hash, code_hash, new_id, now, pixel_hash, stable_hash
from .telemetry import memory_report
from .image_identity import decode_color, rgb_hash

APP_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("RACE_DATA_DIR", str(APP_ROOT / "data")))
MAX_UPLOAD_BYTES = 256 * 1024 * 1024
cv2.setNumThreads(2)


def session_dto(session: dict, store: Storage, detail=False) -> dict:
    frames = session["frames"]
    dto = {k: session[k] for k in ("id", "name", "source", "split", "dataset_id", "group_id")}
    dto.update(
        frame_count=len(frames), duration_s=frames[-1]["timestamp_s"] - frames[0]["timestamp_s"] if frames else 0,
        calibration={k: session["calibration"][k] for k in ("image_width", "image_height")},
        has_ground_truth=store.labels_path(session["id"]).exists(), locked=session["split"] == "audit",
    )
    if session.get("track"):
        dto["track"] = session["track"]
    dto["evaluation"] = deepcopy(session.get("evaluation", {}))
    dto["scenario"] = deepcopy(session.get("scenario", {}))
    if detail and session["split"] != "audit":
        dto["frames"] = [{"frame_id": f["frame_id"], "timestamp_s": f["timestamp_s"],
                          "image_url": f"/api/sessions/{session['id']}/frames/{f['frame_id']}"} for f in frames]
    return dto


def hardware() -> str:
    return f"{platform.system()} {platform.machine()} | {platform.processor() or 'CPU'} | OpenCV {cv2.__version__}"


def benchmark_candidate_title(dataset_name, backend, method):
    suffix = f" {backend} {method}"
    return dataset_name.strip()[:80 - len(suffix)] + suffix


class Service:
    def __init__(self, data_root: Path):
        self.store = Storage(data_root)
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="visual-job")
        self.audit_lock = threading.RLock()

    def submit(self, function, *, job_id=None, message="Queued") -> dict:
        identifier = job_id or new_id("job")
        self.store.put("jobs", {"id": identifier, "status": "queued", "progress": 0, "message": message})

        def progress(value, text):
            job = self.store.get("jobs", identifier)
            job.update(status="running", progress=max(0, min(1, float(value))), message=str(text))
            self.store.put("jobs", job)

        def execute():
            try:
                progress(0.01, message)
                result = function(progress)
                self.store.put("jobs", {"id": identifier, "status": "complete", "progress": 1,
                                        "message": "Complete", "result": result})
            except Exception as exc:
                self.store.put("jobs", {"id": identifier, "status": "failed", "progress": 1,
                                        "message": "Job failed", "error": str(exc)})

        self.executor.submit(execute)
        return {"job_id": identifier}

    def map_dir(self, map_info):
        return self.store.root / "maps" / map_info["id"]

    def validate_quality_model(self, model, config, identity, context):
        from .training import reliability_protocol
        content = model.get("model")
        if not isinstance(content, dict) or stable_hash(content) != model.get("hash"):
            raise ValueError("Learned quality model content changed.")
        protocol = reliability_protocol(config, context)
        if content.get("target_protocol") != protocol:
            raise ValueError("Quality model pose target or scoring tolerances are incompatible. Train a matching model.")
        if content.get("feature_hash") != stable_hash(identity):
            raise ValueError("Quality model visual feature identity is incompatible. Train a matching model.")
        if model.get("target_protocol") != content["target_protocol"] or model.get("feature_hash") != content["feature_hash"]:
            raise ValueError("Quality model protocol metadata changed.")

    def make_map(self, references, config, progress):
        from .vision import build_map
        sessions = [self.store.get("sessions", identifier) for identifier in references]
        if any(s["split"] != "reference" for s in sessions):
            raise ValueError("Maps can only be built from declared reference sessions.")
        if len({s["dataset_id"] for s in sessions}) != 1:
            raise ValueError("Reference sessions must belong to one dataset coordinate frame.")
        contexts = [s.get("evaluation", {}) for s in sessions]
        if any(context != contexts[0] for context in contexts):
            raise ValueError("Reference sessions must share the declared pose convention.")
        progress(0.15, "Extracting reference image features and triangulating landmarks")
        identifier = new_id("map")
        directory = self.store.root / "maps" / identifier
        directory.mkdir()
        start = time.perf_counter()
        info = build_map(sessions, directory, config)
        info.update(id=identifier, reference_session_ids=references, dataset_id=sessions[0]["dataset_id"],
                    feature_backend=config["feature_backend"], build_ms=(time.perf_counter() - start) * 1000,
                    created_at=now())
        from .features import feature_identity
        info.setdefault("feature_identity", feature_identity(config["feature_backend"]))
        info["feature_hash"] = stable_hash(info["feature_identity"])
        info["evaluation"] = deepcopy(contexts[0])
        info["artifact_hash"] = artifact_hash(directory)
        self.store.put("maps", info)
        return info

    def make_candidate(self, request: CandidateRequest):
        from .features import feature_identity
        map_info = self.store.get("maps", request.map_id)
        config = request.config.model_dump()
        if config["feature_backend"] != map_info["feature_backend"]:
            raise ValueError("A candidate must use the same feature family as its reference map.")
        if map_info.get("evaluation", {}).get("pose_target") == "camera":
            config["motion_model"] = "so3"
        identity = feature_identity(config["feature_backend"])
        if map_info.get("feature_identity") != identity:
            raise ValueError("Reference map and current feature assets are incompatible. Rebuild the map.")
        if request.model_id:
            model = self.store.get("models", request.model_id)
            if model["feature_backend"] != config["feature_backend"]:
                raise ValueError("The learned quality model uses a different feature family.")
            self.validate_quality_model(model, config, identity, map_info.get("evaluation", {}))
        value = {"id": new_id("candidate"), "name": request.name.strip(), "config": config,
                 "map_id": map_info["id"], "map_hash": map_info["artifact_hash"],
                 "config_hash": stable_hash(config), "code_hash": code_hash(),
                 "model_id": request.model_id, "frozen_at": None, "created_at": now()}
        value.update(feature_identity=identity, feature_hash=stable_hash(identity),
                     evaluation=deepcopy(map_info.get("evaluation", {})))
        self.store.put("candidates", value)
        return value

    def validate_candidate(self, candidate):
        from .features import feature_identity
        map_info = self.store.get("maps", candidate["map_id"])
        if stable_hash(candidate["config"]) != candidate["config_hash"]:
            raise ValueError("Candidate configuration changed. Create a new candidate.")
        if artifact_hash(self.map_dir(map_info)) != candidate["map_hash"]:
            raise ValueError("Reference map changed. Create a new map and candidate.")
        if code_hash() != candidate["code_hash"]:
            raise ValueError("Estimator code changed. Create a new candidate for this code revision.")
        current_feature = feature_identity(candidate["config"]["feature_backend"])
        if current_feature != candidate.get("feature_identity") or stable_hash(current_feature) != candidate.get("feature_hash"):
            raise ValueError("Feature assets or runtime changed. Rebuild the map and candidate.")
        if map_info.get("evaluation", {}) != candidate.get("evaluation", {}):
            raise ValueError("Candidate pose convention changed.")
        if candidate.get("model_id"):
            model = self.store.get("models", candidate["model_id"])
            self.validate_quality_model(model, candidate["config"], current_feature, candidate.get("evaluation", {}))
            if stable_hash(model["model"]) != model["hash"]:
                raise ValueError("Learned quality model content changed.")
            if candidate.get("frozen_model_hash") and candidate["frozen_model_hash"] != model["hash"]:
                raise ValueError("Learned quality model changed after freeze.")
        if candidate.get("frozen_at") and candidate.get("frozen_audit_hash") != self.audit_fingerprint(map_info["dataset_id"]):
            raise ValueError("The complete audit contract changed after candidate freeze.")
        if candidate.get("frozen_at"):
            frozen_identity = stable_hash({key: candidate.get(key) for key in
                                          ("config_hash", "map_hash", "code_hash", "feature_hash", "evaluation", "frozen_model_hash", "frozen_audit_hash")})
            if frozen_identity != candidate.get("frozen_identity"):
                raise ValueError("Frozen candidate identity changed.")
        return map_info

    def freeze(self, identifier):
        candidate = self.store.get("candidates", identifier)
        self.validate_candidate(candidate)
        if candidate.get("frozen_at"):
            return candidate
        candidate["frozen_at"] = now()
        candidate["frozen_audit_hash"] = self.audit_fingerprint(self.store.get("maps", candidate["map_id"])["dataset_id"])
        if candidate.get("model_id"):
            candidate["frozen_model_hash"] = self.store.get("models", candidate["model_id"])["hash"]
        candidate["frozen_identity"] = stable_hash({k: candidate.get(k) for k in
                                                   ("config_hash", "map_hash", "code_hash", "feature_hash", "evaluation", "frozen_model_hash", "frozen_audit_hash")})
        self.store.put("candidates", candidate)
        return candidate

    def run(self, session_id, candidate_id, progress, *, audit=False):
        from .evaluation import evaluate_run
        from .training import predict_reliability
        from .vision import Localizer, load_map
        session = self.store.get("sessions", session_id)
        candidate = self.store.get("candidates", candidate_id)
        if session["split"] == "reference":
            raise ValueError("Use a query session, not the reference recording.")
        if (session["split"] == "audit") != audit:
            raise ValueError("Final sessions require the frozen audit workflow.")
        info = self.validate_candidate(candidate)
        if session["dataset_id"] != info["dataset_id"]:
            raise ValueError("Session and map must use the same declared dataset coordinate frame.")
        context = deepcopy(session.get("evaluation", {}))
        if context != candidate.get("evaluation", {}):
            raise ValueError("Session and candidate pose conventions differ.")
        start = time.perf_counter()
        map_data = load_map(self.map_dir(info))
        runtime_calibration = {**session["calibration"], "pose_target": context.get("pose_target", "vehicle")}
        localizer = Localizer(map_data, runtime_calibration, candidate["config"])
        startup_ms = (time.perf_counter() - start) * 1000
        predictions = []
        observed_content = []
        model = self.store.get("models", candidate["model_id"])["model"] if candidate.get("model_id") else None
        frames = session["frames"]
        for index, frame in enumerate(frames):
            before = time.perf_counter()
            image = decode_color(Path(frame["image_path"]).read_bytes())
            # Hash the exact decoded buffer used by inference, not a second read.
            digest = rgb_hash(image)
            observed_content.append({"frame_id": frame["frame_id"], "timestamp_s": frame["timestamp_s"], "content_hash": digest})
            decode_ms = (time.perf_counter() - before) * 1000
            prediction = localizer.step(image, frame["frame_id"], frame["timestamp_s"])
            prediction["latency_ms"] += decode_ms
            prediction["sequence_id"] = session_id
            if model:
                prediction["learned_quality_score"] = float(predict_reliability(model, prediction))
            predictions.append(prediction)
            if index % 3 == 0:
                progress(0.1 + 0.75 * (index + 1) / len(frames), f"Localizing frame {index + 1} of {len(frames)}")
        # The estimator is finished before any query labels are loaded.
        labels = self.store.get_labels(session_id)
        metrics = evaluate_run(predictions, frames, labels, candidate["config"], context=context)
        events = []
        outage_start = None
        for row in predictions:
            if row["status"] != "localized" and outage_start is None:
                outage_start = row["timestamp_s"]
                events.append({"frame_id": row["frame_id"], "timestamp_s": row["timestamp_s"],
                               "type": row["status"], "message": row["reason"]})
            elif row["status"] == "localized" and outage_start is not None:
                events.append({"frame_id": row["frame_id"], "timestamp_s": row["timestamp_s"],
                               "type": "recovered", "message": f"Visual localization recovered after {row['timestamp_s'] - outage_start:.2f} s"})
                outage_start = None
        value = {"id": new_id("run"), "candidate_id": candidate_id, "candidate_name": candidate["name"],
                 "session_id": session_id, "session_name": session["name"], "source": session["source"],
                 "split": session["split"], "mode": "causal", "status": "complete", "metrics": metrics,
                 "evaluation": context, "scenario": deepcopy(session.get("scenario", {})),
                 "predictions": predictions, "events": events, "created_at": now(), "config": candidate["config"],
                 "map_id": candidate["map_id"], "identity": {k: candidate.get(k) for k in
                                                            ("config_hash", "map_hash", "code_hash", "feature_hash", "feature_identity", "frozen_identity")},
                 "input_hash": stable_hash({"calibration": session["calibration"], "frames": observed_content, "evaluation": context}),
                 "oracle_hash": stable_hash(labels) if labels is not None else None,
                 "runtime": {"hardware": hardware(), "map_build_ms": info["build_ms"], "startup_ms": startup_ms,
                             "total_ms": (time.perf_counter() - start) * 1000,
                             **memory_report(),
                             "timing_boundary": ("decode pre-rectified 8-bit package images, online features, pose solver and temporal update; offline native 16-bit conversion/rectification and network/playback display excluded"
                                                 if context.get("pose_target") == "camera" else
                                                 "decode, feature processing, pose solver, and temporal update; network/playback display excluded"),
                             "offline_camera_preprocessing_included": False if context.get("pose_target") == "camera" else None}}
        self.store.save_run(value)
        progress(0.98, "Saving predictions and computed evaluation")
        return value

    def bootstrap(self, progress):
        from .datasets import load_package_directory
        from .synthetic import generate_demo
        progress(0.02, "Preparing calibrated synthetic camera recordings")
        demo = self.store.root / "datasets" / "synthetic-demo"
        if not (demo / "manifest.json").exists():
            demo.mkdir(exist_ok=True)
            generate_demo(demo)
        parsed = load_package_directory(demo)
        self.store.install_dataset(parsed)
        current = code_hash()
        existing = [c for c in self.store.values("candidates") if c["code_hash"] == current and c.get("bootstrap")]
        saved_runs = self.store.values("runs")
        if all(any(c["config"]["method"] == method and any(r["candidate_id"] == c["id"] and r["status"] == "complete" for r in saved_runs)
                   for c in existing) for method in ("retrieval", "geometry", "tracking")):
            progress(1, "Ready with saved runs")
            return {"dataset_id": parsed["dataset_id"], "restored": True}
        refs = [s["id"] for s in parsed["sessions"] if s["split"] == "reference"]
        dev = next(s for s in parsed["sessions"] if s["split"] == "development")
        settings = Settings().model_dump()
        map_info = self.make_map(refs, settings, progress)
        run_ids = []
        for index, (method, title) in enumerate((("retrieval", "Reference retrieval"), ("geometry", "Geometric pose"), ("tracking", "Tracking + recovery"))):
            progress(0.3 + index * 0.2, f"Running {title.lower()} baseline")
            config = Settings(**{**settings, "method": method})
            candidate = self.make_candidate(CandidateRequest(name=title, map_id=map_info["id"], config=config))
            candidate["bootstrap"] = True
            self.store.put("candidates", candidate)
            run_ids.append(self.run(dev["id"], candidate["id"], lambda value, message: progress(0.3 + index * 0.2 + value * 0.19, message))["id"])
        return {"dataset_id": parsed["dataset_id"], "map_id": map_info["id"], "run_ids": run_ids}

    def prepare_benchmark(self, kind, progress):
        from .datasets import load_package_directory
        directory = self.store.root / "prepared" / kind
        if kind == "stress":
            from .stress import generate_stress_suite
            progress(0.02, "Rendering calibrated racing stress scenarios")
            if not (directory / "manifest.json").exists():
                generate_stress_suite(directory)
        elif kind == "tumvi":
            from .real_benchmark import prepare_tumvi
            progress(0.02, "Preparing the fixed real-camera benchmark window")
            capture_counts = {}
            def real_progress(state):
                recording = state.get("recording", "TUM VI")
                if state.get("stage") == "images":
                    capture_counts[recording] = int(state.get("captures", 0))
                    fraction = min(1, sum(capture_counts.values()) / 400)
                    progress(0.25 + 0.15 * fraction, f"{recording}: preparing capture {state.get('captures', 0)} of 200")
                else:
                    progress(0.12, f"{recording}: indexing public archive metadata")
            prepare_tumvi(directory, APP_ROOT / "data-source" / "tumvi", progress_callback=real_progress)
        else:
            raise ValueError("Unknown benchmark kind")
        parsed = load_package_directory(directory)
        self.store.install_dataset(parsed)
        references = [session["id"] for session in parsed["sessions"] if session["split"] == "reference"]
        queries = [session for session in parsed["sessions"] if session["split"] == "development"]
        config = Settings(method="tracking" if kind == "stress" else "geometry").model_dump()
        progress(0.45, "Building the SIFT reference map")
        map_info = self.make_map(references, config, lambda value, message: progress(0.45 + 0.15 * value, message))
        candidate = self.make_candidate(CandidateRequest(name=benchmark_candidate_title(parsed["name"], "SIFT", "baseline"), map_id=map_info["id"], config=Settings(**config)))
        runs = []
        for index, session in enumerate(queries):
            run = self.run(session["id"], candidate["id"],
                           lambda value, message, index=index: progress(0.6 + 0.38 * (index + value) / len(queries), message))
            runs.append(run["id"])
        return {"dataset_id": parsed["dataset_id"], "map_id": map_info["id"], "candidate_id": candidate["id"], "run_ids": runs}

    def compare_benchmark(self, dataset_id, request, progress):
        from .evaluation import aggregate_runs
        from .features import availability
        dataset = self.store.get("datasets", dataset_id)
        if not dataset.get("benchmark"):
            raise ValueError("Choose a declared benchmark dataset.")
        backends = request.feature_backends
        if len(set(backends)) != len(backends):
            raise ValueError("Choose distinct feature backends.")
        available = {item["id"] for item in availability() if item["available"]}
        if any(backend not in available for backend in backends):
            raise ValueError("A selected feature backend is unavailable. Prepare its verified local assets first.")
        sessions = [self.store.get("sessions", identifier) for identifier in dataset["session_ids"]]
        references = [session["id"] for session in sessions if session["split"] == "reference"]
        queries = [session for session in sessions if session["split"] == "development"]
        if not references or not queries:
            raise ValueError("A comparison needs reference and development recordings.")
        def input_contract():
            return [{"id": session["id"], "calibration": session["calibration"],
                     "evaluation": session.get("evaluation", {}), "reference_poses": session.get("reference_poses", {}),
                     "frames": [{"frame_id": frame["frame_id"], "timestamp_s": frame["timestamp_s"],
                                 "content_hash": pixel_hash(Path(frame["image_path"]))} for frame in session["frames"]]}
                    for session in sessions if session["split"] in ("reference", "development")]
        initial_inputs = input_contract()
        signature, comparison_code = stable_hash(initial_inputs), code_hash()
        expected_inputs = {session["id"]: stable_hash({"calibration": session["calibration"], "frames": session["frames"],
                                                     "evaluation": session["evaluation"]})
                           for session in initial_inputs if session["id"] in {query["id"] for query in queries}}
        oracle_identities = {}
        methods, failures, run_ids, map_ids = [], [], [], []
        for backend_index, backend in enumerate(backends):
            config = {**request.config.model_dump(), "feature_backend": backend}
            sub_progress = lambda value, message, backend_index=backend_index: progress((backend_index + value) / len(backends), f"{backend}: {message}")
            records = []
            try:
                map_info = self.make_map(references, config, lambda value, message: sub_progress(0.05 + 0.2 * value, message))
                candidate = self.make_candidate(CandidateRequest(name=benchmark_candidate_title(dataset["name"], backend, config["method"]), map_id=map_info["id"], config=Settings(**config)))
                for query_index, session in enumerate(queries):
                    records.append(self.run(session["id"], candidate["id"],
                                            lambda value, message, query_index=query_index: sub_progress(0.3 + 0.69 * (query_index + value) / len(queries), message)))
                    run = records[-1]
                    if run["input_hash"] != expected_inputs[session["id"]]:
                        raise ValueError("Development inputs changed during comparison.")
                    oracle = oracle_identities.setdefault(session["id"], run["oracle_hash"])
                    if oracle != run["oracle_hash"]:
                        raise ValueError("Development labels changed between compared candidates.")
                ids = [run["id"] for run in records]
                methods.append({"feature_backend": backend, "candidate_id": candidate["id"], "map_id": map_info["id"],
                                "run_ids": ids, "config": candidate["config"], "feature_identity": candidate["feature_identity"],
                                "metrics": aggregate_runs(records, candidate["config"], context=dataset.get("evaluation", {})),
                                "per_condition": [{"session_id": run["session_id"], "session_name": run["session_name"], "run_id": run["id"],
                                               "scenario": run.get("scenario", {}), "metrics": run["metrics"]} for run in records]})
                run_ids.extend(ids)
                map_ids.append(map_info["id"])
            except (ValueError, RuntimeError) as exc:
                failures.append({"feature_backend": backend, "error": str(exc), "partial_run_ids": [record["id"] for record in records]})
        if stable_hash(input_contract()) != signature or code_hash() != comparison_code:
            raise ValueError("Reference/development input contract or source changed during comparison. Results were not pooled.")
        for session_id, oracle in oracle_identities.items():
            labels = self.store.get_labels(session_id)
            if (stable_hash(labels) if labels is not None else None) != oracle:
                raise ValueError("Development labels changed during comparison. Results were not pooled.")
        comparison = {"id": new_id("comparison"), "dataset_id": dataset_id, "dataset_name": dataset["name"],
                      "source": dataset["source"], "evaluation": deepcopy(dataset.get("evaluation", {})),
                      "feature_backends": backends, "methods": methods, "failures": failures, "run_ids": run_ids,
                      "map_ids": map_ids, "code_hash": comparison_code, "input_signature": signature, "oracle_identities": oracle_identities, "created_at": now(),
                      "scope": "Fixed reference and complete development recordings. Separate feature-compatible maps. No audit data, query alignment or claimed improvement."}
        self.store.put("comparisons", comparison)
        return comparison

    def audit_contract(self, dataset_id):
        contract = []
        for session in self.store.values("sessions"):
            if session["dataset_id"] == dataset_id and session["split"] == "audit":
                contract.append({"session_id": session["id"], "original_group_id": session.get("original_group_id", session["group_id"]),
                                 "calibration": session["calibration"],
                                 "evaluation": deepcopy(session.get("evaluation", {})),
                                 "scenario": deepcopy(session.get("scenario", {})),
                                 "frames": [{"frame_id": f["frame_id"], "timestamp_s": f["timestamp_s"],
                                             "content_hash": pixel_hash(Path(f["image_path"]))} for f in session["frames"]],
                                 "labels_hash": stable_hash(self.store.get_labels(session["id"]))})
        return sorted(contract, key=lambda value: value["session_id"])

    def audit_fingerprint(self, dataset_id):
        contract = self.audit_contract(dataset_id)
        return stable_hash(contract) if contract else None

    def audit_membership(self, dataset_id):
        contract = self.audit_contract(dataset_id)
        return {"content_hashes": sorted({f["content_hash"] for s in contract for f in s["frames"]}),
                "original_group_ids": sorted({s["original_group_id"] for s in contract})}

    def execute_audit(self, record, progress):
        from .audit_metrics import aggregate_audit_runs
        sessions = [self.store.get("sessions", identifier) for identifier in record["session_ids"]]
        runs = []
        for index, session in enumerate(sessions):
            existing = [r for r in self.store.values("runs") if r.get("audit_record_id") == record["id"]
                        and r["session_id"] == session["id"] and r["candidate_id"] == record["candidate_id"]]
            run = self.store.load_run(existing[-1]["id"]) if existing else self.run(
                session["id"], record["candidate_id"],
                lambda value, message: progress((index + value) / len(sessions), message), audit=True)
            if not run.get("audit_record_id"):
                run["audit_record_id"] = record["id"]
                self.store.save_run(run)
            runs.append(run)
            record["run_ids"] = [r["id"] for r in runs]
            record["run_id"] = runs[0]["id"]
            self.store.put("audits", record)
        record["aggregate_metrics"] = aggregate_audit_runs(runs)
        record["per_session"] = [{"run_id": r["id"], "session_name": r["session_name"], "metrics":
                                  {k: v for k, v in r["metrics"].items() if k != "per_frame"}} for r in runs]
        self.store.put("audits", record)
        return {"run_id": record["run_id"], "run_ids": record["run_ids"], "candidate_id": record["candidate_id"],
                "message": "Final audit consumed and recorded for all held-out sessions."}

    def audit_status(self, candidate_id=None):
        audits = self.store.values("audits")
        frozen = [c for c in self.store.values("candidates") if c.get("frozen_at")]
        if candidate_id:
            candidate = self.store.get("candidates", candidate_id)
            dataset_id = self.store.get("maps", candidate["map_id"])["dataset_id"]
            fingerprint = self.audit_fingerprint(dataset_id)
            # Consumption belongs to the recording registry, even if its files
            # were edited later. A new fingerprint must not look like a fresh test.
            audits = [a for a in audits if a["id"] == dataset_id]
            frozen = [candidate] if candidate.get("frozen_at") else []
        if audits:
            recent = sorted(audits, key=lambda a: a["created_at"])[-1]
            job = self.store.get("jobs", recent["job_id"])
            return {"state": "consumed", "candidate_id": recent["candidate_id"], "run_id": recent.get("run_id"),
                    "message": "This final audit is reserved for its immutable candidate. A failed or interrupted run can only retry that same candidate." if job["status"] != "complete" else "This final audit has been consumed and is bound to its frozen candidate.",
                    "job_id": recent.get("job_id"), "job_status": job["status"], "run_ids": recent.get("run_ids", []),
                    "aggregate_metrics": recent.get("aggregate_metrics"), "per_session": recent.get("per_session", [])}
        return {"state": "ready" if frozen else "locked", "candidate_id": frozen[-1]["id"] if frozen else None,
                "run_id": None, "message": "Freeze a candidate before opening the final audit. Results can no longer be used as a fresh test after inspection."}


def create_app(data_root: Path | None = None, bootstrap=True) -> FastAPI:
    service = Service(data_root or DATA_ROOT)

    @asynccontextmanager
    async def lifespan(app):
        if bootstrap:
            service.submit(service.bootstrap, job_id="bootstrap", message="Preparing the visual demo")
        yield
        service.executor.shutdown(wait=True, cancel_futures=False)

    app = FastAPI(title="Camera Localization", version=__version__, lifespan=lifespan)
    app.state.service = service
    app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
                       allow_methods=["GET", "POST"], allow_headers=["Content-Type"])

    @app.middleware("http")
    async def local_origin(request, call_next):
        origin = request.headers.get("origin")
        accepted = {str(request.base_url).rstrip("/"), "http://localhost:5173", "http://127.0.0.1:5173"}
        if request.method not in ("GET", "HEAD", "OPTIONS") and origin and origin not in accepted:
            return Response(json.dumps({"detail": "Mutating requests must originate from this local application."}),
                            status_code=403, media_type="application/json")
        return await call_next(request)

    def require(kind, identifier):
        try:
            return service.store.get(kind, identifier)
        except KeyError:
            raise HTTPException(404, f"Unknown {kind.rstrip('s')}.") from None

    def validate_request(function):
        try:
            return function()
        except (ValueError, KeyError) as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.get("/api/health")
    def health():
        from .features import availability
        feature_status = availability()
        return {"status": "ok", "version": __version__, "hardware": hardware(),
                "feature_backends": [item["id"] for item in feature_status if item["available"]], "feature_status": feature_status,
                "limitations": ["Synthetic data demonstrates integration, not Yas Marina accuracy.",
                                "Causal replay is not a verified real-time racing system.",
                                "Local audit guards do not hide files from the machine owner."]}

    @app.get("/api/benchmarks")
    def benchmarks():
        return [dataset for dataset in service.store.values("datasets") if dataset.get("benchmark")]

    @app.post("/api/benchmarks/stress")
    def prepare_stress():
        return service.submit(lambda progress: service.prepare_benchmark("stress", progress), message="Preparing synthetic stress suite")

    @app.post("/api/benchmarks/tumvi")
    def prepare_real():
        return service.submit(lambda progress: service.prepare_benchmark("tumvi", progress), message="Preparing real camera benchmark")

    @app.post("/api/benchmarks/{identifier}/compare")
    def compare(identifier: str, request: BenchmarkCompareRequest):
        require("datasets", identifier)
        return service.submit(lambda progress: service.compare_benchmark(identifier, request, progress), message="Comparing visual feature candidates")

    @app.get("/api/comparisons")
    def comparisons():
        return service.store.values("comparisons")

    @app.get("/api/sessions")
    def sessions():
        return [session_dto(s, service.store) for s in service.store.values("sessions")]

    @app.get("/api/sessions/{identifier}")
    def session_detail(identifier: str):
        session = require("sessions", identifier)
        if session["split"] == "audit":
            raise HTTPException(403, "Final audit frames and labels are not available in the ordinary session API.")
        return session_dto(session, service.store, detail=True)

    @app.get("/api/sessions/{identifier}/frames/{frame_id}")
    def frame_image(identifier: str, frame_id: str):
        session = require("sessions", identifier)
        if session["split"] == "audit":
            raise HTTPException(403, "Final audit images are protected from development browsing.")
        frame = next((f for f in session["frames"] if f["frame_id"] == frame_id), None)
        if not frame:
            raise HTTPException(404, "Unknown frame.")
        return FileResponse(frame["image_path"])

    @app.post("/api/sessions/import")
    async def import_recording(file: UploadFile = File()):
        from .datasets import import_package
        identifier = new_id("upload")
        path = service.store.root / "uploads" / f"{identifier}.zip"
        size = 0
        try:
            with path.open("wb") as stream:
                while chunk := await file.read(1024 * 1024):
                    size += len(chunk)
                    if size > MAX_UPLOAD_BYTES:
                        raise HTTPException(413, "Package exceeds the 256 MiB compressed upload limit.")
                    stream.write(chunk)
            destination = service.store.root / "datasets" / identifier
            parsed = validate_request(lambda: import_package(path, destination))
            if parsed["dataset_id"] in service.store.state["datasets"]:
                if destination.resolve().is_relative_to((service.store.root / "datasets").resolve()):
                    shutil.rmtree(destination)
                raise HTTPException(409, "This dataset is already imported. Reuse its existing sessions.")
            try:
                validate_request(lambda: service.store.install_dataset(parsed))
            except HTTPException:
                if destination.resolve().is_relative_to((service.store.root / "datasets").resolve()):
                    shutil.rmtree(destination)
                raise
            return {"dataset_id": parsed["dataset_id"], "sessions": [session_dto(s, service.store) for s in parsed["sessions"]],
                    "warnings": parsed.get("warnings", [])}
        finally:
            path.unlink(missing_ok=True)
            await file.close()

    @app.get("/api/template.zip")
    def template():
        from .datasets import create_template_zip
        path = create_template_zip(service.store.root / "uploads" / "camera-package-template.zip")
        return FileResponse(path, filename="camera-package-template.zip", media_type="application/zip")

    @app.get("/api/maps")
    def maps():
        from .vision import load_map
        values = []
        for info in service.store.values("maps"):
            item = {k: v for k, v in info.items() if k not in ("path", "output_dir", "directory")}
            data = load_map(service.map_dir(info))
            points = data.get("points_xyz", data.get("points", data.get("landmarks")))
            item["points"] = points[::max(1, len(points) // 2500)].tolist() if points is not None else []
            values.append(item)
        return values

    @app.post("/api/maps/build")
    def build(request: MapRequest):
        for identifier in request.reference_session_ids:
            if require("sessions", identifier)["split"] != "reference":
                raise HTTPException(400, "Only reference sessions can enter a map.")
        return service.submit(lambda progress: service.make_map(request.reference_session_ids, request.config.model_dump(), progress),
                              message="Building a sparse visual reference map")

    @app.get("/api/jobs/{identifier}")
    def job(identifier: str):
        return require("jobs", identifier)

    @app.get("/api/candidates")
    def candidates():
        current = code_hash()
        return [{**value, "compatible_code": value["code_hash"] == current} for value in service.store.values("candidates")]

    @app.post("/api/candidates")
    def make_candidate(request: CandidateRequest):
        return validate_request(lambda: service.make_candidate(request))

    @app.post("/api/candidates/{identifier}/freeze")
    def freeze(identifier: str):
        require("candidates", identifier)
        return validate_request(lambda: service.freeze(identifier))

    @app.get("/api/runs")
    def runs():
        return service.store.values("runs")

    @app.post("/api/runs")
    def create_run(request: RunRequest):
        session = require("sessions", request.session_id)
        candidate = require("candidates", request.candidate_id)
        if session["split"] in ("audit", "reference"):
            raise HTTPException(403, "Use a training or development query session. Final queries require the audit route.")
        validate_request(lambda: service.validate_candidate(candidate))
        return service.submit(lambda progress: service.run(request.session_id, request.candidate_id, progress), message="Running causal camera localization")

    @app.get("/api/runs/{identifier}")
    def run_detail(identifier: str):
        require("runs", identifier)
        return service.store.load_run(identifier)

    @app.get("/api/runs/{identifier}/export.json")
    def export_json(identifier: str):
        data = run_detail(identifier)
        return Response(json.dumps(data, indent=2, allow_nan=False), media_type="application/json",
                        headers={"Content-Disposition": f'attachment; filename="{identifier}.json"'})

    @app.get("/api/runs/{identifier}/export.csv")
    def export_csv(identifier: str):
        data = run_detail(identifier)
        output = io.StringIO(newline="")
        fields = ["sequence_id", "frame_id", "timestamp_s", "x_m", "y_m", "z_m", "yaw_rad", "status", "latency_ms", "inliers", "reprojection_error_px", "quality", "uncertainty_m", "uncertainty_note", "learned_quality_score", "forecast_timestamp_s", "forecast_x_m", "forecast_y_m", "forecast_z_m", "forecast_yaw_rad", "pose_T_WV", "forecast_T_WV", "pose_T_WC", "forecast_T_WC", "pose_target", "world_frame", "alignment_status", "config_hash", "map_hash", "code_hash", "feature_hash", "reason"]
        writer = csv.DictWriter(output, fields)
        writer.writeheader()
        for prediction in data["predictions"]:
            pose = prediction.get("pose") or {}
            position = pose.get("position_m", [None, None, None])
            forecast = prediction.get("forecast") or {}
            future = forecast.get("pose") or {}
            future_position = future.get("position_m", [None, None, None])
            row = {k: prediction.get(k) for k in fields if k in prediction}
            row.update(x_m=position[0], y_m=position[1], z_m=position[2], yaw_rad=pose.get("yaw_rad"),
                       forecast_timestamp_s=forecast.get("timestamp_s"), forecast_x_m=future_position[0],
                       forecast_y_m=future_position[1], forecast_z_m=future_position[2], forecast_yaw_rad=future.get("yaw_rad"),
                       pose_T_WV=json.dumps(pose["T_WV"]) if "T_WV" in pose else None,
                       forecast_T_WV=json.dumps(future["T_WV"]) if "T_WV" in future else None,
                       pose_T_WC=json.dumps(pose["T_WC"]) if "T_WC" in pose else None,
                       forecast_T_WC=json.dumps(future["T_WC"]) if "T_WC" in future else None,
                       pose_target=data.get("evaluation", {}).get("pose_target", "vehicle"),
                       world_frame=data.get("evaluation", {}).get("world_frame", "metric_z_up"),
                       alignment_status=data.get("evaluation", {}).get("alignment_status", "declared"),
                       **{k: data["identity"].get(k) for k in ("config_hash", "map_hash", "code_hash", "feature_hash")})
            if row["pose_target"] == "camera":
                row["yaw_rad"] = None
                row["forecast_yaw_rad"] = None
            writer.writerow(row)
        return Response(output.getvalue(), media_type="text/csv", headers={"Content-Disposition": f'attachment; filename="{identifier}.csv"'})

    @app.post("/api/experiments/search")
    def search(request: SearchRequest):
        from .experiments import candidate_configs, rank_runs
        session = require("sessions", request.session_id)
        if session["split"] != "development":
            raise HTTPException(403, "Configuration search is restricted to development sessions.")
        require("maps", request.map_id)

        def execute(progress):
            records = []
            for index, config in enumerate(candidate_configs(request.config.model_dump(), request.count)):
                candidate = service.make_candidate(CandidateRequest(name=f"Search candidate {index + 1}", map_id=request.map_id, config=Settings(**config)))
                records.append(service.run(request.session_id, candidate["id"],
                                          lambda value, message: progress((index + value) / request.count, message)))
            ranked = rank_runs(records)
            summaries = [service.store.get("runs", run["id"]) for run in records]
            return {"runs": summaries, "selected_candidate_id": ranked[0]["candidate_id"],
                    "selection_rule": "Development joint success, coverage, tail error, then latency. No final audit information is used."}

        return service.submit(execute, message="Evaluating bounded configuration candidates")

    @app.get("/api/models")
    def models():
        return [{k: v for k, v in model.items() if k != "model"} for model in service.store.values("models")]

    @app.post("/api/models/train")
    def train(request: TrainingRequest):
        from .training import train_reliability
        session = require("sessions", request.session_id)
        if session["split"] != "train":
            raise HTTPException(403, "Learned quality training uses training sessions only.")
        if not service.store.get_labels(session["id"]):
            raise HTTPException(400, "Training requires independent pose labels for the training session.")
        candidate = require("candidates", request.candidate_id)
        validate_request(lambda: service.validate_candidate(candidate))

        def execute(progress):
            run = service.run(session["id"], candidate["id"], progress)
            model = train_reliability(run["predictions"], service.store.get_labels(session["id"]),
                                      {**candidate["config"], "feature_hash": candidate["feature_hash"]},
                                      context=session.get("evaluation", {}))
            value = {"id": new_id("quality"), "name": "Learned visual quality", "training_session_id": session["id"],
                     "samples": model.get("samples", len(run["predictions"])), "hash": stable_hash(model),
                     "feature_backend": candidate["config"]["feature_backend"], "created_at": now(), "model": model,
                     "feature_hash": model["feature_hash"], "target_protocol": model["target_protocol"],
                     "evaluation": deepcopy(session.get("evaluation", {})),
                     "limitations": "Training-label fit; uncalibrated quality score. It does not change pose geometry."}
            service.store.put("models", value)
            return {k: v for k, v in value.items() if k != "model"}

        return service.submit(execute, message="Training a small quality model on training labels")

    @app.get("/api/audit/status")
    def audit_status(candidate_id: str | None = None):
        if candidate_id:
            require("candidates", candidate_id)
        return service.audit_status(candidate_id)

    @app.post("/api/audit")
    def audit(request: AuditRequest):
        with service.audit_lock:
            candidate = require("candidates", request.candidate_id)
            if not candidate.get("frozen_at"):
                raise HTTPException(409, "Freeze the candidate before opening the final benchmark.")
            info = validate_request(lambda: service.validate_candidate(candidate))
            dataset_id = info["dataset_id"]
            fingerprint = validate_request(lambda: service.audit_fingerprint(dataset_id))
            if candidate.get("frozen_audit_hash") != fingerprint:
                raise HTTPException(409, "Audit images changed after candidate freeze.")
            membership = service.audit_membership(dataset_id)
            previous = service.store.state["audits"].get(dataset_id)
            if previous:
                if previous["candidate_id"] != candidate["id"] or previous["frozen_identity"] != candidate["frozen_identity"]:
                    raise HTTPException(409, "This final benchmark has already been consumed by another frozen candidate. Use a fresh held-out dataset.")
                job = service.store.get("jobs", previous["job_id"])
                if job["status"] == "failed":
                    previous["attempts"] = previous.get("attempts", 1) + 1
                    service.store.put("audits", previous)
                    return service.submit(lambda progress: service.execute_audit(previous, progress),
                                          job_id=previous["job_id"], message="Resuming the same immutable final audit")
                return {"job_id": previous["job_id"]}
            for consumed in service.store.values("audits"):
                if (set(consumed.get("content_hashes", [])) & set(membership["content_hashes"])
                        or set(consumed.get("original_group_ids", [])) & set(membership["original_group_ids"])):
                    raise HTTPException(409, "This held-out recording overlaps an already consumed final benchmark. A subset or renamed package is not a fresh audit.")
            final_sessions = [s for s in service.store.values("sessions") if s["dataset_id"] == dataset_id and s["split"] == "audit"]
            if not final_sessions:
                raise HTTPException(400, "This dataset has no held-out audit session.")
            # Reserve before dispatch to prevent concurrent alternative candidates.
            identifier = new_id("audit-job")
            record = {"id": dataset_id, "candidate_id": candidate["id"], "frozen_identity": candidate["frozen_identity"],
                      "audit_hash": fingerprint,
                      **membership, "attempts": 1, "run_ids": [],
                      "job_id": identifier, "run_id": None, "created_at": now(), "session_ids": [s["id"] for s in final_sessions]}
            service.store.put("audits", record)

            return service.submit(lambda progress: service.execute_audit(record, progress), job_id=identifier, message="Opening the frozen final audit")

    # A production build is served from the same loopback endpoint as the API.
    frontend = APP_ROOT / "frontend" / "dist"
    if frontend.exists():
        app.mount("/", StaticFiles(directory=frontend, html=True), name="frontend")
    else:
        @app.get("/")
        def awaiting_frontend():
            return {"message": "Build the frontend or run its development server.", "api": "/docs"}
    return app


app = create_app()

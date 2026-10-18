"""Stateless current-image ViT retrieval and approximate fit-reference BODY copies.

Two explicit, separately contained stages: a complete fit-only gallery, then
one-image-at-a-time localization. No query labels, motion, seed, previous query,
future batch, optimizer, training or implicit network access exists here.
"""
from __future__ import annotations

import csv
import gc
import hashlib
import importlib.metadata
import io
import json
import math
from pathlib import Path, PurePosixPath, PureWindowsPath
import time

import numpy as np

RUN_SCHEMA = "schubki-vit-place-run-v1"
MANIFEST_SCHEMA = "schubki-vit-place-manifest-v1"
COMPLETION_SCHEMA = "schubki-vit-place-completion-v1"
DIAGNOSTIC_SCHEMA = "schubki-vit-place-diagnostic-v1"
CHECKPOINT_SCHEMA = "schubki-full-encoder-checkpoint-v1"
DIMENSION = 768
PREPROCESSING = "torchvision.ViT_B_16_Weights.IMAGENET1K_V1.transforms;PIL_RGB;no_augmentation"
METHOD = "stateless_current_image_nearest_reference_pose"
COMMON_ROW = {"frame_id", "image_path", "image_sha256", "image_bytes"}
GALLERY_ROW = COMMON_ROW | {"lap_id", "body_x_m", "body_y_m", "body_yaw_rad"}
QUERY_ROW = COMMON_ROW | {"timestamp_ns"}
BUDGET_KEYS = {"wall_seconds", "max_manifest_bytes", "max_gallery_rows", "max_query_rows",
               "max_image_bytes", "max_pixels", "max_cache_bytes", "max_checkpoint_bytes",
               "max_output_bytes", "torch_threads"}


class IdentityError(ValueError):
    """Pinned identity or admitted membership changed. Fatal, never an image loss."""


class ImageFailure(ValueError):
    """One undecodable/unusable image. Query records it as lost; gallery is fatal."""


def canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                       allow_nan=False) + "\n").encode("ascii")


def _exact(value, keys):
    if type(value) is not dict or set(value) != set(keys):
        raise IdentityError("unexpected fields")
    return value


def _integer(value, low, high):
    if type(value) is not int or not low <= value <= high:
        raise IdentityError("exact integer outside declared bounds")
    return value


def _finite(value):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise IdentityError("finite numeric value required")
    return float(value)


def _text(value):
    if type(value) is not str or not 1 <= len(value) <= 240 or value != value.strip() or not value.isprintable():
        raise IdentityError("exact bounded printable identity required")
    return value


def _sha(value):
    if type(value) is not str or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise IdentityError("lowercase SHA256 required")
    return value


def _json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise IdentityError("duplicate JSON key")
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=pairs,
                      parse_constant=lambda _: (_ for _ in ()).throw(IdentityError("nonfinite JSON")))


def _absolute(value):
    if type(value) is not str or not Path(value).is_absolute():
        raise IdentityError("absolute local path required")
    return Path(value).resolve()


def _pin(value, cap):
    _exact(value, {"path", "sha256", "bytes"})
    _absolute(value["path"])
    _sha(value["sha256"])
    _integer(value["bytes"], 1, cap)
    return value


def file_pin(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "sha256": digest.hexdigest(), "bytes": path.stat().st_size}


def read_pinned(value, cap):
    _pin(value, cap)
    path = _absolute(value["path"])
    if path.stat().st_size != value["bytes"]:
        raise IdentityError("pinned size changed")
    with path.open("rb") as stream:
        raw = stream.read(cap + 1)
    if len(raw) != value["bytes"] or hashlib.sha256(raw).hexdigest() != value["sha256"]:
        raise IdentityError("pinned content changed")
    return raw


def _relative(root, name):
    if type(name) is not str or "\\" in name or PureWindowsPath(name).drive:
        raise IdentityError("portable relative image path required")
    portable = PurePosixPath(name)
    if portable.is_absolute() or not portable.parts or any(p in (".", "..") for p in portable.parts):
        raise IdentityError("image path escapes root")
    result = (root / name).resolve()
    if not result.is_relative_to(root):
        raise IdentityError("resolved image path escapes root")
    return result


def normalized(features, expected_rows):
    if (type(features) is not np.ndarray or features.dtype != np.float32
            or features.shape != (expected_rows, DIMENSION) or not np.isfinite(features).all()):
        raise ImageFailure("invalid_encoder_output")
    norms = np.linalg.norm(features.astype(np.float64), axis=1, keepdims=True)
    if np.any(norms <= 1e-12) or not np.isfinite(norms).all():
        raise ImageFailure("zero_encoder_output")
    return np.asarray(features / norms, dtype=np.float32)


def nearest_reference(query, gallery):
    """Exact ties retain the first row in the canonical frozen gallery order."""
    query = normalized(query, 1)
    if (type(gallery) is not np.ndarray or gallery.dtype != np.float32
            or gallery.ndim != 2 or gallery.shape[1] != DIMENSION or not len(gallery)
            or not np.isfinite(gallery).all()):
        raise IdentityError("invalid gallery features")
    cosine = gallery @ query[0]
    index = int(np.argmax(cosine))
    return index, float(np.clip(cosine[index], -1.0, 1.0))


def validate_config(value):
    _exact(value, {"schema", "stage", "encoder", "manifest", "image_root", "cache", "output_dir", "budgets"})
    if value["schema"] != RUN_SCHEMA or value["stage"] not in ("gallery", "localize"):
        raise IdentityError("unsupported run schema/stage")
    b = _exact(value["budgets"], BUDGET_KEYS)
    for key in BUDGET_KEYS - {"wall_seconds"}:
        _integer(b[key], 1, 2**40)
    if not 0 < _finite(b["wall_seconds"]) <= 14400 or b["torch_threads"] > 4:
        raise IdentityError("wall/thread budget outside finite envelope")
    _absolute(value["image_root"])
    _absolute(value["output_dir"])
    _pin(value["manifest"], b["max_manifest_bytes"])
    encoder = _exact(value["encoder"], {"kind", "backend", "device_index", "weights", "training_identity", "step"})
    if encoder["kind"] not in ("trained", "pretrained") or encoder["backend"] not in ("cpu", "directml"):
        raise IdentityError("unsupported encoder")
    _integer(encoder["device_index"], 0, 16)
    _pin(encoder["weights"], b["max_checkpoint_bytes"])
    if encoder["kind"] == "trained":
        _sha(encoder["training_identity"])
        _integer(encoder["step"], 1, 12000)
    elif encoder["training_identity"] is not None or encoder["step"] is not None:
        raise IdentityError("pretrained encoder has no training checkpoint identity")
    if value["stage"] == "gallery":
        if value["cache"] is not None:
            raise IdentityError("gallery does not consume another cache")
    else:
        cache = _exact(value["cache"], {"directory", "completion"})
        _absolute(cache["directory"])
        _pin(cache["completion"], b["max_manifest_bytes"])
    return value


def validate_manifest(value, kind, budgets):
    _exact(value, {"schema", "kind", "camera_id", "calibration_id", "source_pins", "rows"})
    if value["schema"] != MANIFEST_SCHEMA or value["kind"] != kind:
        raise IdentityError("wrong manifest stage")
    _text(value["camera_id"])
    _sha(value["calibration_id"])
    if type(value["source_pins"]) is not dict or not 1 <= len(value["source_pins"]) <= 32:
        raise IdentityError("explicit source pins required")
    for key, sha in value["source_pins"].items():
        _text(key)
        _sha(sha)
    rows = value["rows"]
    cap = budgets["max_gallery_rows" if kind == "gallery" else "max_query_rows"]
    if type(rows) is not list or not 1 <= len(rows) <= cap:
        raise IdentityError("complete finite manifest rows required")
    ids, stamps = set(), set()
    previous = None
    for row in rows:
        _exact(row, GALLERY_ROW if kind == "gallery" else QUERY_ROW)
        _text(row["frame_id"])
        _sha(row["image_sha256"])
        _integer(row["image_bytes"], 1, budgets["max_image_bytes"])
        if row["frame_id"] in ids:
            raise IdentityError("duplicate frame identity")
        ids.add(row["frame_id"])
        if kind == "gallery":
            if type(row["lap_id"]) is not int or row["lap_id"] not in (1, 2):
                raise IdentityError("gallery must contain train laps1-2 only")
            for key in ("body_x_m", "body_y_m", "body_yaw_rad"):
                _finite(row[key])
            if not -math.pi <= row["body_yaw_rad"] <= math.pi:
                raise IdentityError("gallery yaw must already be wrapped")
        else:
            stamp = _integer(row["timestamp_ns"], 0, 2**63 - 1)
            if stamp in stamps or (previous is not None and stamp <= previous):
                raise IdentityError("query timestamps must uniquely increase without repair")
            stamps.add(stamp)
            previous = stamp
    return value


class FrozenEncoder:
    """Lazy, pinned float32 inference. Never downloads or builds an optimizer."""

    def __init__(self, spec, budgets, check):
        self.spec, self.budgets, self.check = spec, budgets, check
        pin = spec["weights"]
        # Stream hash before Torch loading; do not retain the serialized checkpoint.
        actual_pin = file_pin(pin["path"])
        if actual_pin["sha256"] != pin["sha256"] or actual_pin["bytes"] != pin["bytes"]:
            raise IdentityError("model weight pin changed")
        import torch
        import torchvision.models as models
        torch.set_num_threads(budgets["torch_threads"])
        torch.backends.mha.set_fastpath_enabled(False)
        model = models.vit_b_16(weights=None)
        state = torch.load(pin["path"], map_location="cpu", weights_only=True)
        if spec["kind"] == "trained":
            if (type(state) is not dict or state.get("schema") != CHECKPOINT_SCHEMA
                    or type(state.get("step")) is not int or state["step"] != spec["step"]
                    or state.get("training_identity") != spec["training_identity"]):
                raise IdentityError("trained checkpoint schema/step/identity mismatch")
            model.heads = torch.nn.Identity()
            model.load_state_dict(state["model"], strict=True)
        else:
            model.load_state_dict(state, strict=True)
            model.heads = torch.nn.Identity()
        del state
        gc.collect()
        for layer in model.encoder.layers:
            original = layer.self_attention.forward
            def forward(*args, _original=original, **kwargs):
                kwargs["need_weights"] = True
                return _original(*args, **kwargs)
            layer.self_attention.forward = forward
        if spec["backend"] == "directml":
            import torch_directml
            self.device = torch_directml.device(spec["device_index"])
        else:
            self.device = torch.device("cpu")
        model.requires_grad_(False)
        self.model = model.to(device=self.device, dtype=torch.float32).eval()
        self.torch, self.preprocess = torch, models.ViT_B_16_Weights.IMAGENET1K_V1.transforms()
        check()

    def encode_batch(self, raw_images):
        from PIL import Image, UnidentifiedImageError
        self.check()
        started = time.monotonic()
        tensors = []
        try:
            for raw in raw_images:
                with Image.open(io.BytesIO(raw)) as image:
                    if image.width * image.height > self.budgets["max_pixels"]:
                        raise ImageFailure("decoded_pixel_cap")
                    tensors.append(self.preprocess(image.convert("RGB")))
        except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
            raise ImageFailure("image_decode_failed") from exc
        decoded = time.monotonic()
        with self.torch.no_grad():
            batch = self.torch.stack(tensors).to(self.device, dtype=self.torch.float32)
            raw = self.model(batch)
            # CPU materialization is the completion/synchronization boundary.
            features = raw.detach().cpu().numpy().copy()
        del raw, batch, tensors
        completed = time.monotonic()
        self.check()
        return normalized(features, len(raw_images)), {
            "decode_preprocess_s": decoded - started, "encoder_completed_s": completed - decoded}


def encoder_identity(spec):
    versions = {}
    for name in ("torch", "torchvision", "torch-directml", "numpy", "pillow"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "absent"
    value = {"architecture": "vit_b_16.heads_Identity.float32[768]", "preprocessing": PREPROCESSING,
             "explicit_attention": True, "encoder": spec, "dependencies": versions,
             "source_sha256": file_pin(__file__)["sha256"]}
    return hashlib.sha256(canonical(value)).hexdigest()


class Artifacts:
    def __init__(self, directory, cap):
        self.directory, self.cap = directory, cap
        directory.mkdir(parents=False, exist_ok=False)

    def check(self, additional=0):
        size = sum(p.stat().st_size for p in self.directory.iterdir() if p.is_file())
        if size + additional > self.cap:
            raise RuntimeError("output byte cap")

    def json(self, name, value):
        raw = canonical(value)
        self.check(len(raw))
        path = self.directory / name
        with path.open("xb") as stream:
            stream.write(raw)
            stream.flush()
            import os
            os.fsync(stream.fileno())
        return file_pin(path)


def _read_image(row, root, budgets):
    path = _relative(root, row["image_path"])
    try:
        return read_pinned({"path": str(path), "sha256": row["image_sha256"], "bytes": row["image_bytes"]},
                           budgets["max_image_bytes"])
    except OSError as exc:
        raise ImageFailure("image_read_failed") from exc


def _model_record(config, encoder_sha):
    spec = config["encoder"]
    return {"kind": spec["kind"], "step": spec["step"], "training_identity": spec["training_identity"],
            "weights_sha256": spec["weights"]["sha256"], "encoder_sha256": encoder_sha}


def _load_cache(config, manifest, encoder_sha):
    b, cache = config["budgets"], config["cache"]
    directory = _absolute(cache["directory"])
    if _absolute(cache["completion"]["path"]).parent != directory:
        raise IdentityError("cache completion outside directory")
    complete = _json(read_pinned(cache["completion"], b["max_manifest_bytes"]))
    if (complete.get("schema") != COMPLETION_SCHEMA or complete.get("stage") != "gallery"
            or complete.get("complete") is not True or complete.get("model") != _model_record(config, encoder_sha)):
        raise IdentityError("cache model/completion mismatch")
    if complete.get("camera_id") != manifest["camera_id"] or complete.get("calibration_id") != manifest["calibration_id"]:
        raise IdentityError("cache camera/calibration mismatch")
    _exact(complete["artifacts"], {"features", "references"})
    for artifact in complete["artifacts"].values():
        if _absolute(artifact["path"]).parent != directory:
            raise IdentityError("cache artifact outside directory")
    reference_pin, feature_pin = complete["artifacts"]["references"], complete["artifacts"]["features"]
    references = _json(read_pinned(reference_pin, b["max_cache_bytes"]))
    validate_manifest(references, "gallery", b)
    if (references["camera_id"] != manifest["camera_id"]
            or references["calibration_id"] != manifest["calibration_id"]):
        raise IdentityError("cached reference provenance mismatch")
    features_raw = read_pinned(feature_pin, b["max_cache_bytes"])
    count = len(references["rows"])
    buffer = io.BytesIO(features_raw)
    if np.lib.format.read_magic(buffer) != (1, 0):
        raise IdentityError("only bounded NPY1.0 caches are admitted")
    shape, fortran, dtype = np.lib.format.read_array_header_1_0(buffer)
    if (shape != (count, DIMENSION) or fortran or dtype != np.dtype("float32")
            or len(features_raw) - buffer.tell() != count * DIMENSION * 4):
        raise IdentityError("cache array shape/dtype/payload mismatch")
    features = np.frombuffer(features_raw, dtype=np.float32, offset=buffer.tell()).reshape(shape)
    if (features.dtype != np.float32 or features.shape != (count, DIMENSION)
            or not np.isfinite(features).all()
            or not np.allclose(np.linalg.norm(features.astype(np.float64), axis=1), 1, atol=1e-5)):
        raise IdentityError("invalid normalized gallery cache")
    if complete["counts"] != {"expected": count, "processed": count, "predicted": count, "lost": 0}:
        raise IdentityError("partial gallery cache")
    gallery_ids = {row["frame_id"] for row in references["rows"]}
    if any(row["frame_id"] in gallery_ids for row in manifest["rows"]):
        raise IdentityError("query overlaps training gallery frame identity")
    return features, references["rows"]


def run_stage(config_path, expected_config_sha256, *, deadline_monotonic=None):
    """Run one immutable stage. Returns operational counts, never numeric scores or poses."""
    started = time.monotonic()
    config_pin = file_pin(config_path)
    if config_pin["sha256"] != _sha(expected_config_sha256) or config_pin["bytes"] > 128 * 1024:
        raise IdentityError("config identity/cap mismatch")
    config = validate_config(_json(read_pinned(config_pin, 128 * 1024)))
    b, stage = config["budgets"], config["stage"]
    deadline = started + b["wall_seconds"]
    if deadline_monotonic is not None:
        deadline = min(deadline, _finite(deadline_monotonic))
    def check():
        if time.monotonic() >= deadline:
            raise TimeoutError("finite stage deadline")
    kind = "gallery" if stage == "gallery" else "queries"
    manifest = validate_manifest(_json(read_pinned(config["manifest"], b["max_manifest_bytes"])), kind, b)
    root, output = _absolute(config["image_root"]), _absolute(config["output_dir"])
    if output.exists():
        raise FileExistsError("output directory must be new")
    if not root.is_dir() or not output.parent.is_dir():
        raise IdentityError("image root/output parent must exist")
    for row in manifest["rows"]:
        _relative(root, row["image_path"])
    encoder_sha = encoder_identity(config["encoder"])
    features = references = None
    if stage == "localize":
        features, references = _load_cache(config, manifest, encoder_sha)
    check()
    encoder = FrozenEncoder(config["encoder"], b, check)
    check()
    artifacts = Artifacts(output, b["max_output_bytes"])
    ready = time.monotonic()
    artifact_pins, elapsed, predicted, lost = {}, [], 0, 0
    rows = manifest["rows"]
    if stage == "gallery":
        if len(rows) * DIMENSION * 4 + 1024 > b["max_cache_bytes"]:
            raise IdentityError("feature cache cap")
        encoded = []
        for start in range(0, len(rows), 3):
            check()
            batch_rows = rows[start:start + 3]
            raw = [_read_image(row, root, b) for row in batch_rows]
            vectors, _ = encoder.encode_batch(raw)
            encoded.append(normalized(vectors, len(batch_rows)))
        matrix = np.concatenate(encoded)
        buffer = io.BytesIO()
        np.save(buffer, matrix, allow_pickle=False)
        raw = buffer.getvalue()
        if len(raw) > b["max_cache_bytes"]:
            raise IdentityError("serialized cache cap")
        artifacts.check(len(raw))
        path = output / "features.npy"
        with path.open("xb") as stream:
            stream.write(raw)
            stream.flush()
            import os
            os.fsync(stream.fileno())
        artifact_pins["features"] = file_pin(path)
        artifact_pins["references"] = artifacts.json("references.json", manifest)
        predicted = len(rows)
    else:
        with (output / "predictions.csv").open("x", encoding="ascii", newline="") as csv_stream, \
                (output / "diagnostics.jsonl").open("xb") as diagnostics, \
                (output / "frame_timings.jsonl").open("xb") as completed_timings:
            writer = csv.writer(csv_stream, lineterminator="\n")
            writer.writerow(("timestamp_ns", "x", "y", "yaw"))
            for index, row in enumerate(rows):
                check()
                frame_start = time.monotonic()
                timing = {key: 0.0 for key in ("read_hash_s", "decode_preprocess_s", "encoder_completed_s",
                                               "retrieval_s", "output_s", "total_s")}
                record = {"schema": DIAGNOSTIC_SCHEMA, "index": index, "frame_id": row["frame_id"],
                          "timestamp_ns": str(row["timestamp_ns"]), "status": "lost", "reason": "",
                          "pose": None, "reference_id": None, "cosine": None, "timing": timing}
                reading = True
                try:
                    raw = _read_image(row, root, b)
                    timing["read_hash_s"] = time.monotonic() - frame_start
                    reading = False
                    vector, parts = encoder.encode_batch([raw])
                    timing.update(parts)
                    search = time.monotonic()
                    choice, cosine = nearest_reference(vector, features)
                    ref = references[choice]
                    timing["retrieval_s"] = time.monotonic() - search
                    pose = {"x": float(ref["body_x_m"]), "y": float(ref["body_y_m"]),
                            "yaw": float(ref["body_yaw_rad"]), "map_frame": "map",
                            "pose_kind": "approximate_reference_pose", "position_source": "training_reference",
                            "heading_source": "training_reference", "yaw_approximate": True,
                            "translation_scale_observed": False}
                    record.update(status="approximate_reference_pose", reason="nearest_fit_reference_copy",
                                  pose=pose, reference_id=ref["frame_id"], cosine=cosine)
                    predicted += 1
                except ImageFailure as exc:
                    lost += 1
                    record["reason"] = str(exc)
                    if reading:
                        timing["read_hash_s"] = time.monotonic() - frame_start
                writing = time.monotonic()
                if record["pose"] is not None:
                    pose = record["pose"]
                    artifacts.check(256)
                    writer.writerow((str(row["timestamp_ns"]), *(repr(pose[k]) for k in ("x", "y", "yaw"))))
                    csv_stream.flush()
                timing["output_s"] = time.monotonic() - writing
                timing["total_s"] = time.monotonic() - frame_start
                data = canonical(record)
                artifacts.check(len(data) + 256)
                diagnostics.write(data)
                diagnostics.flush()
                completed = time.monotonic() - frame_start
                elapsed.append(completed)
                complete_timing = canonical({"index": index, "timestamp_ns": str(row["timestamp_ns"]),
                                             "status": record["status"], "total_s": completed,
                                             "scope": "read_hash_decode_preprocess_completed_encoder_retrieval_csv_and_diagnostic_write"})
                artifacts.check(len(complete_timing))
                completed_timings.write(complete_timing)
                completed_timings.flush()
                artifacts.check()
            import os
            csv_stream.flush()
            os.fsync(csv_stream.fileno())
            diagnostics.flush()
            os.fsync(diagnostics.fileno())
            completed_timings.flush()
            os.fsync(completed_timings.fileno())
        artifact_pins = {"predictions": file_pin(output / "predictions.csv"),
                         "diagnostics": file_pin(output / "diagnostics.jsonl"),
                         "frame_timings": file_pin(output / "frame_timings.jsonl")}
    check()
    inputs = {"config": config_pin, "manifest": config["manifest"], "source_pins": manifest["source_pins"]}
    if config["cache"] is not None:
        inputs["cache_completion"] = config["cache"]["completion"]
    completion = {"schema": COMPLETION_SCHEMA, "stage": stage, "complete": True, "method": METHOD,
                  "camera_id": manifest["camera_id"], "calibration_id": manifest["calibration_id"],
                  "model": _model_record(config, encoder_sha), "inputs": inputs,
                  "counts": {"expected": len(rows), "processed": len(rows), "predicted": predicted, "lost": lost},
                  "artifacts": artifact_pins,
                  "timing": {"startup_s": ready - started, "processing_s": time.monotonic() - ready,
                             "total_s": time.monotonic() - started,
                             "mean_frame_s": sum(elapsed) / len(elapsed) if elapsed else None,
                             "scope": "completed_per_image_processing;recorded_local_images;not_live_capture_latency"},
                  "limitations": ["approximate fit reference BODY copies, no metric geometry recovery",
                                   "gallery batching is preparation, not per-query inference latency",
                                   "diagnostic timing stops before its own write; frame_timings includes it",
                                   "lost queries have diagnostics but no fabricated CSV pose"]}
    artifacts.json("completion.json", completion)
    return {"stage": stage, "complete": True, **completion["counts"], "completion_path": str(output / "completion.json")}

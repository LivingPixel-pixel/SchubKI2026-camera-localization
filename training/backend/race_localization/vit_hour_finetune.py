"""One declared DirectML hour from parent7750. All ML imports remain lazy.

Full ViT768, last two blocks/final norm only. No old optimizer continuation,
query-driven mining, automatic retries, backend switches or sampled validation.
"""
from __future__ import annotations

import gc
import hashlib
import io
import math
from pathlib import Path, PurePosixPath, PureWindowsPath
import random
import shutil
import time

from . import vit_laptop_finetune as vf

rt = vf.rt
SCHEMA = "schubki-vit-hour-config-v1"
SCHEMA_V2 = "schubki-vit-hour-config-v2"
SCHEMA_V3 = "schubki-vit-hour-config-v3"
SCHEMA_V4 = "schubki-vit-hour-config-v4"
# v5: explicit five-lap experiment (train laps1-5, validation laps6-7). Old v1-v4 are unchanged.
SCHEMA_V5 = "schubki-vit-hour-config-v5"
FIVE_POLICY = "train_1_5_validation_6_7_v1"
RAM_PROFILES = {SCHEMA: (3*rt.GIB, 4*rt.GIB, 4*rt.GIB), SCHEMA_V2: (6*rt.GIB, 8*rt.GIB, 4*rt.GIB),
                SCHEMA_V3: (12*rt.GIB, 16*rt.GIB, 2*rt.GIB), SCHEMA_V4: (12*rt.GIB, 16*rt.GIB, 2*rt.GIB),
                SCHEMA_V5: (12*rt.GIB, 16*rt.GIB, 2*rt.GIB)}
# (total, gallery/train, queries/validation, fit-feature cap). Eligible/unsupported are measured, never pinned, for v5.
OLD_COUNTS = {"total": 5733, "gallery": 4497, "queries": 1236, "fit_feature_cap": 16*1024**2,
              "train_laps": (1, 2), "query_laps": (3,)}
FIVE_COUNTS = {"total": 10569, "gallery": 8269, "queries": 2300, "fit_feature_cap": 32*1024**2,
               "train_laps": (1, 2, 3, 4, 5), "query_laps": (6, 7)}


def policy_counts(schema):
    return FIVE_COUNTS if schema == SCHEMA_V5 else OLD_COUNTS
CHECKPOINT = "schubki-vit-hour-checkpoint-v1"
POLICY = "encoder_layers_10_11_and_final_norm"
MINING = {"schema": "schubki-hour-static-mining-v1", "seed": 17,
          "max_triplets": 100000, "max_hard_repeats_per_anchor": 4}
CHECKPOINT_CAP = 640*1024**2
INFERENCE_POLICY = {"inference_batch_images": 12, "progress_every_images": 48}
SCHEDULING_POLICY = "adaptive-final-reserve-v1"
BASELINE_SOURCE_F8 = {
    "backend/race_localization/place_model.py": "0f98f6f4ca7842c5460ebb008bfd4f0c139b6b19b30ec7ebd499914193da321a",
    "backend/race_localization/vit_finetune_runtime.py": "645a134946af92c5b022e9ca975087867001d4e9b48de1fd22d37b7f8a4a207c",
    "backend/race_localization/vit_hour_finetune.py": "57951adf0fa2311abf7203652716347397f5f60a28eb68805e7092711ab43f98",
    "backend/race_localization/vit_laptop_finetune.py": "ccb56324554cee0ab6b124c0f5c9e377126ec1afb278daa2beceb7ea7c516860",
    "backend/race_localization/vit_laptop_runtime.py": "ee952e214ff3522cbff8240338455c0e2aeb5de80f13ac0d3a519bb9123aaf2e",
    "backend/requirements-vit-laptop.txt": "f1910d4053e7cea5f6f4f269f24770ce91e2ccc3a60a88764129d4564289a492",
    "backend/tools/train_vit_hour.py": "0f372ee7487700cea2720823a9a488a39f05429738bd430c3b3ff5544272a6d4",
    "tools/start_vit_hour.ps1": "8c53c33e8e25fbcd1f3ec8abcd4ed8e305c0bb1662c0c99af4205a5022ce89bb"}
BLOCK_SUFFIXES = ("ln_1.weight", "ln_1.bias", "self_attention.in_proj_weight", "self_attention.in_proj_bias",
    "self_attention.out_proj.weight", "self_attention.out_proj.bias", "ln_2.weight", "ln_2.bias",
    "mlp.0.weight", "mlp.0.bias", "mlp.3.weight", "mlp.3.bias")
TRAINABLE = tuple(sorted([f"encoder.layers.encoder_layer_{block}.{suffix}" for block in (10, 11)
                         for suffix in BLOCK_SUFFIXES] + ["encoder.ln.weight", "encoder.ln.bias"]))
REPRESENTATIVES = ("encoder.layers.encoder_layer_10.self_attention.in_proj_weight",
                   "encoder.layers.encoder_layer_11.self_attention.in_proj_weight", "encoder.ln.weight")


def sha(value):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("full lowercase SHA256 required")
    return value


def digest(value):
    return hashlib.sha256(rt.canonical(value)).hexdigest()


def read_config(path):
    path = Path(path)
    if not 0 < path.stat().st_size <= 128*1024:
        raise ValueError("hour config size cap")
    c = rt.strict(path.read_bytes())
    fields = {"schema", "backend", "device_index", "dataset", "warm_start", "warm_start_card",
              "parent", "output", "trainable_policy", "mining", "recipe", "limits"}
    if c.get("schema") == SCHEMA_V4:
        fields.add("baseline_import")
    if c.get("schema") == SCHEMA_V5:
        fields.add("lap_policy")  # No baseline_import: old lap3 baseline import is rejected for this policy.
    if set(c) != fields:
        raise ValueError("unexpected hour config fields")
    if c["schema"] == SCHEMA_V5 and c["lap_policy"] != FIVE_POLICY:
        raise ValueError("v5 requires the explicit train_1_5_validation_6_7_v1 lap policy")
    if (type(c["schema"]) is not str or c["schema"] not in RAM_PROFILES or c["backend"] != "directml"
            or type(c["device_index"]) is not int or c["device_index"] != 0):
        raise ValueError("explicit new DirectML hour policy required")
    if (c["trainable_policy"] != POLICY or type(c["mining"]) is not dict or set(c["mining"]) != set(MINING)
            or any(type(c["mining"][k]) is not type(v) or c["mining"][k] != v for k, v in MINING.items())):
        raise ValueError("frozen trainability/mining policy mismatch")
    if (set(c["parent"]) != {"step", "original_ancestor_step", "training_identity"}
            or type(c["parent"]["step"]) is not int or c["parent"]["step"] != 7750
            or type(c["parent"]["original_ancestor_step"]) is not int or c["parent"]["original_ancestor_step"] != 1125):
        raise ValueError("immediate parent7750 and original ancestry1125 required")
    sha(c["parent"]["training_identity"])
    if set(c["dataset"]) != {"directory", "cache_root", "sha256"}:
        raise ValueError("complete prepared dataset identity required")
    sha(c["dataset"]["sha256"])
    artifact_fields = [("warm_start", 512*1024**2), ("warm_start_card", 16384)]
    if c["schema"] == SCHEMA_V4:
        artifact_fields.append(("baseline_import", 2*1024**2))
    for key, maximum in artifact_fields:
        pin = c[key]
        if set(pin) != {"path", "sha256", "bytes"}:
            raise ValueError("exact parent artifact pin required")
        sha(pin["sha256"])
        vf.integer(pin["bytes"], 1, maximum)
    from_paths = [c["dataset"]["directory"], c["dataset"]["cache_root"], c["warm_start"]["path"], c["warm_start_card"]["path"]]
    if c["schema"] == SCHEMA_V4:
        imported = Path(c["baseline_import"]["path"])
        if imported.name != "bootstrap.json" or imported.parent.name != "run-native-v5":
            raise ValueError("v4 requires the explicitly pinned run-native-v5/bootstrap.json")
        from_paths.extend((str(imported), str(imported.parent)))
    for name in [c["output"], *from_paths]:
        if not isinstance(name, str) or not Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError("absolute local paths without traversal required")
        if name.startswith(("\\\\", "//")) or "://" in name:
            raise ValueError("local hour paths required")
        if any(p.is_symlink() or getattr(p, "is_junction", lambda: False)() for p in (Path(name), *Path(name).parents)):
            raise ValueError("linked hour path forbidden")
    output = Path(c["output"]).resolve()
    if any(output.is_relative_to(Path(p).resolve()) or Path(p).resolve().is_relative_to(output) for p in from_paths):
        raise ValueError("hour output must be disjoint from parent/data inputs")
    r = c["recipe"]
    expected = {"seed": 17, "learning_rate": 1e-5 if c["schema"] in (SCHEMA_V4, SCHEMA_V5) else 2e-6, "weight_decay": .01, "margin": .2,
        "save_every": 50, "validate_every": 1000, "accumulation": 1, "hard_negatives": True,
        "brightness": .15, "contrast": .15, "min_crop_scale": .9}
    if set(r) != set(expected) | {"max_updates"} or any(type(r[k]) is not type(v) or r[k] != v for k, v in expected.items()):
        raise ValueError("declared selective hour recipe changed")
    vf.integer(r["max_updates"], 50, 12000)
    if c["schema"] in (SCHEMA_V4, SCHEMA_V5) and r["max_updates"] != 1000:
        raise ValueError("v4/v5 declare exactly1000 fresh updates")
    soft, hard, reserve = RAM_PROFILES[c["schema"]]
    expected_limits = {"wall_seconds": 3600, "final_reserve_seconds": 900, "ram_soft_bytes": soft,
        "ram_hard_bytes": hard, "system_reserve_bytes": reserve, "disk_bytes": 8*rt.GIB, "threads": 4}
    if set(c["limits"]) != set(expected_limits) or any(type(c["limits"][k]) is not int or c["limits"][k] != v for k, v in expected_limits.items()):
        raise ValueError("hour envelope must match its explicit frozen schema; old policies are unchanged")
    return c


def source_identity():
    root = Path(__file__).parents[2]
    paths = [Path(__file__), Path(vf.__file__), Path(rt.__file__), Path(rt.base.__file__),
        Path(vf.__file__).with_name("place_model.py"), root/"backend/tools/train_vit_hour.py",
        root/"tools/start_vit_hour.ps1", root/"backend/requirements-vit-laptop.txt"]
    return {str(path.relative_to(root)): rt.sha_file(path, 2*1024**2) for path in paths}


def select_trainable(model):
    parameters = dict(model.named_parameters())
    if not set(TRAINABLE).issubset(parameters):
        raise ValueError("complete torchvision last-two-block/final-norm names required")
    for name, parameter in parameters.items():
        parameter.requires_grad_(name in TRAINABLE)
    names = sorted(name for name, value in parameters.items() if value.requires_grad)
    if tuple(names) != TRAINABLE:
        raise ValueError("unexpected trainable subset")
    return {"policy": POLICY, "parameters": [{"name": name, "shape": list(parameters[name].shape),
        "dtype": str(parameters[name].dtype)} for name in names], "optimizer_order": names,
        "trainable_parameters": sum(parameters[name].numel() for name in names),
        "frozen_names": sorted(set(model.state_dict())-set(names))}


def selected_optimizer(torch, model, mask, learning_rate=2e-6):
    if type(learning_rate) is not float or learning_rate not in (2e-6, 1e-5):
        raise ValueError("explicit validated hour learning rate required")
    parameters = dict(model.named_parameters())
    if tuple(mask["optimizer_order"]) != TRAINABLE or tuple(sorted(name for name, p in parameters.items() if p.requires_grad)) != TRAINABLE:
        raise ValueError("optimizer trainability/order mismatch")
    return torch.optim.AdamW([parameters[name] for name in mask["optimizer_order"]], lr=learning_rate, weight_decay=.01, foreach=False)


def fingerprints(torch, state, names=None):
    names = sorted(state) if names is None else names
    result = {}
    for name in names:
        value = state[name].detach().cpu().contiguous()
        if not bool(torch.isfinite(value).all()):
            raise ValueError("nonfinite full network state")
        result[name] = {"shape": list(value.shape), "dtype": str(value.dtype),
            "sha256": hashlib.sha256(value.numpy().tobytes()).hexdigest()}
    return result


def optimizer_fingerprint(torch, value):
    if isinstance(value, torch.Tensor):
        return fingerprints(torch, {"tensor": value})["tensor"]
    if isinstance(value, dict):
        return [[str(key), optimizer_fingerprint(torch, item)] for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))]
    if isinstance(value, (list, tuple)):
        return [optimizer_fingerprint(torch, item) for item in value]
    return vf.plain_metadata(value)


def restricted_load(torch, path):
    # The pinned stack supports mmap for its default torch.save ZIP format.
    # Unsupported formats/stacks fail closed instead of allocating a second full copy.
    return torch.load(path, map_location="cpu", weights_only=True, mmap=True)


def verify_parent(torch, config, model):
    path = rt.verify_pin(config["warm_start"], 512*1024**2)
    card_path = rt.verify_pin(config["warm_start_card"], 16384)
    card = rt.strict(card_path.read_bytes())
    fields = {"architecture", "dimension", "preprocessing", "weights", "origin_step", "new_experiment_step",
              "training_identity", "experiment_identity", "contains"}
    if (type(card) is not dict or set(card) != fields or card["architecture"] != "vit_b_16"
            or type(card["dimension"]) is not int or card["dimension"] != 768
            or card["preprocessing"] != vf.PREPROCESSING or type(card["origin_step"]) is not int or card["origin_step"] != 1125
            or type(card["new_experiment_step"]) is not int or card["new_experiment_step"] != 7750
            or card["training_identity"] != config["parent"]["training_identity"]
            or card["experiment_identity"] != card["training_identity"] or not isinstance(card["contains"], str)):
        raise ValueError("strict inference-best parent7750 card required; no relabelled1125")
    sha(card["training_identity"])
    sha(card["experiment_identity"])
    if set(card["weights"]) != {"path", "sha256", "bytes"}:
        raise ValueError("exact parent card weights pin required")
    sha(card["weights"]["sha256"])
    vf.integer(card["weights"]["bytes"], 1, 512*1024**2)
    provenance = card["weights"]["path"]
    if not isinstance(provenance, str):
        raise ValueError("parent provenance path required")
    declared = PureWindowsPath(provenance) if "\\" in provenance or PureWindowsPath(provenance).drive else PurePosixPath(provenance)
    if not declared.is_absolute() or declared.name != "inference-best.pt" or ".." in declared.parts:
        raise ValueError("recognized parent basename required; provenance is never a load route")
    if {k: card["weights"][k] for k in ("sha256", "bytes")} != {k: config["warm_start"][k] for k in ("sha256", "bytes")}:
        raise ValueError("parent card/network pin mismatch")
    state = restricted_load(torch, path)
    if (type(state) is not dict or set(state) != {"schema", "step", "training_identity", "model"}
            or state["schema"] != vf.PUBLIC_CHECKPOINT or type(state["step"]) is not int or state["step"] != 7750
            or state["training_identity"] != card["training_identity"] or type(state["model"]) is not dict
            or any(type(k) is not str or type(v) is not torch.Tensor or v.device.type != "cpu" or v.dtype != torch.float32 for k, v in state["model"].items())):
        raise ValueError("parent7750 full model-only payload mismatch; optimizer import forbidden")
    fingerprints(torch, state["model"])
    model.load_state_dict(state["model"], strict=True)
    return {"origin_step": 7750, "origin_training_identity": state["training_identity"], "original_ancestor_step": 1125,
        "model_sha256": config["warm_start"]["sha256"], "card_sha256": config["warm_start_card"]["sha256"]}


class HourDisk(rt.Disk):
    def atomic(self, name, writer, cap):
        if cap > CHECKPOINT_CAP or shutil.disk_usage(self.root).free < cap+2*rt.GIB:
            raise OSError("hour single-artifact/free-disk reserve reached")
        return super().atomic(name, writer, cap)


class PreparationProgress:
    """Small local observations, never an estimate of uncompleted work."""
    def __init__(self, disk, total=5733):
        if total not in (OLD_COUNTS["total"], FIVE_COUNTS["total"]):
            raise ValueError("pinned old5733 or five-lap10569 preparation denominator required")
        self.disk, self.started, self.total = disk, time.monotonic(), total
        self.seen, self.written, self.stage = None, None, None

    def __call__(self, stage, encoded_images, total_images):
        if type(stage) is not str or stage not in ("encoding", "retrieval", "proxy", "imported_baseline", "mining", "ready"):
            raise ValueError("unknown preparation stage")
        if type(total_images) is not int or total_images != self.total:
            raise ValueError("complete policy preparation denominator required")
        if encoded_images is not None:
            vf.integer(encoded_images, 0, total_images)
        if stage in ("encoding", "retrieval", "proxy") and encoded_images is None:
            raise ValueError("completed encoding count required for this stage")
        if stage in ("retrieval", "proxy") and encoded_images != total_images:
            raise ValueError("retrieval/proxy requires the complete encoding pass")
        if encoded_images is not None and self.seen is not None and encoded_images < self.seen:
            raise ValueError("preparation count cannot move backwards within a pass")
        self.seen = encoded_images
        if stage == self.stage:
            if encoded_images == self.written:
                return
            if (encoded_images is not None and self.written is not None and encoded_images != total_images
                    and encoded_images-self.written < INFERENCE_POLICY["progress_every_images"]):
                return
        self.disk.json("preparation.json", {"stage": stage, "encoded_images": encoded_images, "total_images": total_images,
            "elapsed_seconds": time.monotonic()-self.started, "updated_unix_seconds": time.time()})
        self.stage, self.written = stage, encoded_images


def admission_requirement(config):
    limits = config["limits"]
    ceiling = "ram_soft_bytes" if config["schema"] in (SCHEMA_V3, SCHEMA_V4, SCHEMA_V5) else "ram_hard_bytes"
    return limits[ceiling] + limits["system_reserve_bytes"]


def admission(config):
    output = Path(config["output"])
    used = rt.Disk(output, config["limits"]["disk_bytes"]).total()
    if used > 8*rt.GIB or shutil.disk_usage(output).free < max(0, 8*rt.GIB-used)+2*rt.GIB:
        raise OSError("hour requires remaining8GiB output allowance plus2GiB free disk")


def copy_parent(config, disk):
    artifacts = {}
    for key, name, maximum in (("warm_start", "inference-best.pt", 512*1024**2),
                                ("warm_start_card", "inference-best.card.json", 16384)):
        source = rt.verify_pin(config[key], maximum)
        def writer(stream, source=source):
            with source.open("rb") as incoming:
                while True:
                    block = incoming.read(1024**2)
                    if not block:
                        break
                    stream.write(block)
        pin = disk.atomic(name, writer, config[key]["bytes"])
        if {k: pin[k] for k in ("sha256", "bytes")} != {k: config[key][k] for k in ("sha256", "bytes")}:
            raise ValueError("baseline parent copy mismatch")
        rt.verify_pin(config[key], maximum)
        artifacts[name] = pin
    return artifacts


def static_triplets(np, rows, fit_indices, features, random_triplets, pair_masks, check=lambda: None, train_laps=(1, 2)):
    """One random / one descriptor negative. Never consumes validation-lap features/rows."""
    fit_indices = np.asarray(fit_indices, dtype=np.int64)
    features = np.asarray(features, dtype=np.float32)
    if len(set(map(int, fit_indices))) != len(fit_indices) or any(rows[int(i)]["lap_id"] not in train_laps for i in fit_indices):
        raise ValueError("fit-only unique embedding rows required")
    if features.shape != (len(fit_indices), 768) or not np.isfinite(features).all():
        raise ValueError("complete fit768 embeddings required")
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    if (norms <= 1e-12).any():
        raise ValueError("zero fit embedding")
    features = features/norms
    local = {int(index): position for position, index in enumerate(fit_indices)}
    labels = np.array([[rows[int(i)][k] for k in ("body_x_m", "body_y_m", "body_yaw_rad")] for i in fit_indices], dtype=np.float64)
    laps = np.array([rows[int(i)]["lap_id"] for i in fit_indices])
    hashes = [rows[int(i)]["image_sha256"] for i in fit_indices]
    rng, groups = np.random.default_rng(17), {}
    for triplet in random_triplets:
        a, p, n = map(int, triplet)
        if any(i not in local for i in (a, p, n)):
            raise ValueError("query row admitted to random mining")
        groups.setdefault(a, []).append((a, p, n))
    for anchor in groups:
        rng.shuffle(groups[anchor])
    # Round-robin anchors prevents the cap from selecting only an early prefix.
    chosen = []
    for turn in range(max(map(len, groups.values()), default=0)):
        for anchor in sorted(groups):
            if turn < len(groups[anchor]) and len(chosen) < MINING["max_triplets"]//2:
                chosen.append(groups[anchor][turn])
    cache, counts, output = {}, {}, []
    for a, p, n in chosen:
        check()
        ai, pi, ni = local[a], local[p], local[n]
        if a not in cache:
            positive, negative = pair_masks(labels[ai], labels)
            cross = laps != laps[ai]
            distinct = np.array([value != hashes[ai] for value in hashes])
            distance = np.clip(2-2*(features @ features[ai]), 0, 4)
            ranked = np.argsort(distance, kind="stable")
            cache[a] = positive & cross & distinct, negative & cross & distinct, distance, ranked
        positive, negative, distance, ranked = cache[a]
        if not positive[pi] or not negative[ni] or len({hashes[ai], hashes[pi], hashes[ni]}) != 3:
            raise ValueError("random triplet violates frozen legal identities")
        eligible = [int(i) for i in ranked if negative[int(i)] and hashes[int(i)] != hashes[pi]
                    and counts.get((a, int(i)), 0) < MINING["max_hard_repeats_per_anchor"]]
        semi = [i for i in eligible if float(distance[pi]) < float(distance[i]) < float(distance[pi])+.2]
        if not eligible:
            raise ValueError("legal descriptor pool exhausted under fixed repetition cap")
        selected = (semi or eligible)[0]
        counts[(a, selected)] = counts.get((a, selected), 0)+1
        output.extend(((a, p, n), (a, p, int(fit_indices[selected]))))
    if not output:
        raise ValueError("no legal fit-only static triplets")
    per_lap = {str(lap): len({a for a, _, _ in chosen if rows[a]["lap_id"] == lap}) for lap in train_laps}
    return np.asarray(output, dtype=np.int64), {"triplets": len(output), "random": len(output)//2,
        "descriptor": len(output)//2, "anchors": len(groups), "rules": MINING,
        **({"anchors_per_lap": per_lap} if tuple(train_laps) != (1, 2) else {}),
        "descriptor_policy": "first legal semi-hard else hardest; cross-lap;5m/15deg positives;>=25m negatives"}


def _npy(trainer, name, value, cap):
    return trainer.disk.atomic(name, lambda stream: trainer.np.save(stream, value, allow_pickle=False), cap)


def _numeric(trainer, pin, shape, dtype, cap):
    path = rt.verify_pin(pin, cap)
    with path.open("rb") as stream:
        value = trainer.np.load(stream, allow_pickle=False)
    if value.shape != shape or value.dtype != dtype or not trainer.np.isfinite(value).all():
        raise ValueError("sealed hour numeric artifact mismatch")
    return value


def _import_path(raw):
    if (type(raw) is not str or not Path(raw).is_absolute() or ".." in Path(raw).parts
            or raw.startswith(("\\\\", "//")) or "://" in raw):
        raise ValueError("absolute local baseline artifact path required")
    path = Path(raw)
    if any(p.is_symlink() or getattr(p, "is_junction", lambda: False)() for p in (path, *path.parents)):
        raise ValueError("linked baseline artifact forbidden")
    return path.resolve()


def validate_baseline_metrics(metrics):
    counts = {"all_queries": 1236, "gallery_count": 4497, "eligible_queries": 989, "unsupported_queries": 247}
    numeric = {"recall1_all", "recall5_all", "reference_copy_xy_rmse_m_proxy", "heading_normal_lateral_rmse_m_proxy",
        *{f"reference_copy_{axis}_{tail}_proxy" for axis in ("xy_m", "lateral_m", "yaw_rad") for tail in ("p95", "p99")}}
    if type(metrics) is not dict or set(metrics) != set(counts) | numeric | {"recall1_hits", "recall5_hits", "false_distant_top1", "scope"}:
        raise ValueError("complete original baseline metrics required")
    for key, expected in counts.items():
        vf.integer(metrics[key], expected, expected)
    vf.integer(metrics["recall1_hits"], 0, 989)
    vf.integer(metrics["recall5_hits"], metrics["recall1_hits"], 989)
    vf.integer(metrics["false_distant_top1"], 0, 1236)
    for key in numeric:
        vf.finite(metrics[key], 0, 1e9)
    if (metrics["recall1_all"] != metrics["recall1_hits"]/1236 or metrics["recall5_all"] != metrics["recall5_hits"]/1236
            or metrics["scope"] != "lap3 retrieval and reference-copy proxies, not official pose/coverage/runtime score"):
        raise ValueError("baseline metric ratios/scope changed")


def load_baseline_import(trainer, context):
    """Only the reviewed f8 parent bootstrap, never changed-weight final descriptors."""
    pin = trainer.config["baseline_import"]
    header_path = _import_path(pin["path"])
    if header_path.name != "bootstrap.json" or header_path.parent.name != "run-native-v5":
        raise ValueError("recognized original parent bootstrap required")
    rt.verify_pin(pin, 2*1024**2)
    header = rt.strict(header_path.read_bytes())
    if (type(header) is not dict or set(header) != {"schema", "context", "metrics", "fit_features", "validation_seconds"}
            or header["schema"] != "schubki-hour-bootstrap-v1"):
        raise ValueError("original baseline header schema mismatch")
    old = header["context"]
    if type(old) is not dict or set(old) != set(context):
        raise ValueError("original baseline context fields changed")
    source = old["source"]
    if (type(source) is not dict or len(source) != 8 or any(type(key) is not str for key in source)
            or {key.replace("\\", "/"): value for key, value in source.items()} != BASELINE_SOURCE_F8):
        raise ValueError("baseline requires exact reviewed f8 source identity")
    if old["scheduling_policy"] != "adaptive-final-reserve-v1":
        raise ValueError("original baseline scheduling provenance mismatch")
    for key in set(context)-{"config", "source", "scheduling_policy"}:
        if rt.canonical(old[key]) != rt.canonical(context[key]):
            raise ValueError("baseline inference compatibility mismatch: "+key)
    old_config = old["config"]
    expected_config = {key: value for key, value in trainer.config.items() if key != "baseline_import"}
    if type(old_config) is not dict or type(old_config.get("recipe")) is not dict:
        raise ValueError("original baseline config required")
    expected_config.update(schema=SCHEMA_V3, output=str(header_path.parent),
        recipe={**trainer.config["recipe"], "learning_rate": 2e-6, "max_updates": 12000})
    # Windows paths can retain a different slash spelling while naming the exact old output.
    if _import_path(old_config.get("output")) != header_path.parent:
        raise ValueError("original baseline output provenance mismatch")
    expected_config["output"] = old_config["output"]
    if rt.canonical(old_config) != rt.canonical(expected_config):
        raise ValueError("baseline config differs beyond declared schema/LR/target/output/import")
    validate_baseline_metrics(header["metrics"])
    vf.finite(header["validation_seconds"], 0, 3600)
    fit_pin = header["fit_features"]
    if type(fit_pin) is not dict or set(fit_pin) != {"path", "sha256", "bytes"}:
        raise ValueError("exact original fit artifact pin required")
    sha(fit_pin["sha256"])
    vf.integer(fit_pin["bytes"], 1, 16*1024**2)
    fit_path = _import_path(fit_pin["path"])
    if fit_path.parent != header_path.parent or fit_path.name != "bootstrap-fit.npy":
        raise ValueError("only adjacent original parent fit descriptors may be imported")
    fit = _numeric(trainer, fit_pin, (4497, 768), trainer.np.dtype("float32"), 16*1024**2)
    rt.verify_pin(pin, 2*1024**2)
    provenance = {"schema": "schubki-parent-baseline-import-v1", "header": pin,
        "original_fit_features": fit_pin, "original_context": old, "original_context_sha256": digest(old),
        "measurement": "imported original parent7750 measurement; no new baseline encoding"}
    return header, fit, provenance


def copy_imported_fit(trainer, pin):
    source = rt.verify_pin(pin, 16*1024**2)
    def writer(stream):
        with source.open("rb") as incoming:
            shutil.copyfileobj(incoming, stream, 1024**2)
    copied = trainer.disk.atomic("bootstrap-fit.npy", writer, 16*1024**2)
    if any(copied[key] != pin[key] for key in ("sha256", "bytes")):
        raise ValueError("imported fit copy differs from original bytes")
    rt.verify_pin(pin, 16*1024**2)
    return copied


def imported_bootstrap_record(trainer, context, original, provenance):
    copied = copy_imported_fit(trainer, original["fit_features"])
    return {"schema": "schubki-hour-bootstrap-v1", "context": context, "metrics": original["metrics"],
        "fit_features": copied, "validation_seconds": original["validation_seconds"], "imported_baseline": provenance}


def empty_loss():
    return {"count": 0, "zero_count": 0, "sum": 0., "last": None, "step": 0}


def validate_loss_summary(value, step):
    vf.integer(step, 0, 12000)
    if type(value) is not dict or set(value) != {"count", "zero_count", "sum", "last", "step"}:
        raise ValueError("bounded completed loss summary required")
    vf.integer(value["count"], step, step)
    vf.integer(value["step"], step, step)
    vf.integer(value["zero_count"], 0, step)
    vf.finite(value["sum"], 0, 5*step)
    if step:
        vf.finite(value["last"], 0, 5)
    elif value["last"] is not None:
        raise ValueError("empty loss summary must have no observation")
    return dict(value)


def observe_loss(summary, loss, step):
    vf.integer(step, 1, 12000)
    previous = validate_loss_summary(summary, step-1)
    vf.finite(loss, 0, 5)
    return validate_loss_summary({"count": step, "step": step, "sum": previous["sum"]+loss,
        "zero_count": previous["zero_count"]+int(loss == 0), "last": loss}, step)


def proxy_tails(np, features, rows, metrics, check=lambda: None, gallery_laps=(1, 2), query_laps=(3,)):
    """All-query reference-copy error tails; never an official localization score."""
    features = np.asarray(features, dtype=np.float32)
    features = features/np.linalg.norm(features, axis=1, keepdims=True)
    gallery = np.array([i for i, row in enumerate(rows) if row["lap_id"] in gallery_laps])
    queries = [i for i, row in enumerate(rows) if row["lap_id"] in query_laps]
    xy, lateral, yaw = [], [], []
    for query in queries:
        check()
        match = int(gallery[int(np.argmax(features[gallery] @ features[query]))])
        dx, dy = (rows[match][key]-rows[query][key] for key in ("body_x_m", "body_y_m"))
        heading = rows[query]["body_yaw_rad"]
        xy.append(math.hypot(dx, dy))
        lateral.append(abs(-math.sin(heading)*dx+math.cos(heading)*dy))
        angle = rows[match]["body_yaw_rad"]-heading
        yaw.append(abs(math.atan2(math.sin(angle), math.cos(angle))))
    if len(queries) != metrics["all_queries"] or not queries:
        raise ValueError("complete query denominator required for proxy tails")
    for name, values in (("xy_m", xy), ("lateral_m", lateral), ("yaw_rad", yaw)):
        metrics["reference_copy_"+name+"_p95_proxy"] = float(np.percentile(values, 95))
        metrics["reference_copy_"+name+"_p99_proxy"] = float(np.percentile(values, 99))


def _initialize_hour(t):
    torch, np = t.torch, t.np
    if t.micro != 1:
        raise ValueError("hour physical batch is exactly one triplet")
    output = Path(t.config["output"])
    if (output/"latest.pt").exists() and any(not (output/name).is_file() for name in ("bootstrap.json", "mining.json", "hour-triplets.npy")):
        raise ValueError("resume requires the exact sealed bootstrap/static list; never remine missing state")
    t.disk = HourDisk(t.config["output"], 8*rt.GIB)
    t.model, t.preprocess = vf.build_model(torch)
    t.origin = verify_parent(torch, t.config, t.model)
    t.mask = select_trainable(t.model)
    t.frozen = fingerprints(torch, t.model.state_dict(), t.mask["frozen_names"])
    t.model.to(t.device, dtype=torch.float32)
    t.step, t.best, t.pending, t.validated_step, t.durable_step = 0, None, True, -1, None
    t.loss_stats = empty_loss()
    # Bootstrap does not construct AdamW or consume sampler/RNG state.
    class ReadOnly:
        def zero_grad(self, **kwargs):
            t.model.zero_grad(set_to_none=True)
    t.optimizer = ReadOnly()
    identities = [{"row": i, **{k: row[k] for k in ("lap_id", "image_sha256", "body_x_m", "body_y_m", "body_yaw_rad")}} for i, row in enumerate(t.rows)]
    counts = policy_counts(t.config["schema"])
    t.resource_pilot = t.config["schema"] == SCHEMA_V5
    if (getattr(t, "policy", None) == FIVE_POLICY) != (t.config["schema"] == SCHEMA_V5):
        raise ValueError("mixed lap policies: v5 config requires the v2 five-lap dataset and old configs the v1 dataset")
    train, query = counts["train_laps"], counts["query_laps"]
    if (len(t.rows) != counts["total"] or sum(row["lap_id"] in train for row in t.rows) != counts["gallery"]
            or sum(row["lap_id"] in query for row in t.rows) != counts["queries"]):
        raise ValueError("exact policy bootstrap membership required (5733/4497/1236 or 10569/8269/2300)")
    context = {"config": t.config, "source": t.sources, "environment": t.env, "parent": t.origin,
        "preprocessing": vf.PREPROCESSING, "mask": t.mask, "rows_sha256": digest(identities),
        "inference_policy": INFERENCE_POLICY, "scheduling_policy": SCHEDULING_POLICY}
    if t.config["schema"] == SCHEMA_V5:
        context["lap_policy"] = FIVE_POLICY
    t.identity = digest(context)
    bootstrap_path = Path(t.config["output"])/"bootstrap.json"
    imported = load_baseline_import(t, context) if t.config["schema"] == SCHEMA_V4 else None
    reused_bootstrap = bootstrap_path.exists() or imported is not None
    preparation = PreparationProgress(t.disk, counts["total"])
    fit_indices = np.array([i for i, row in enumerate(t.rows) if row["lap_id"] in train], dtype=np.int64)
    fit_shape, fit_cap = (counts["gallery"], 768), counts["fit_feature_cap"]
    if bootstrap_path.exists():
        bootstrap = rt.strict(bootstrap_path.read_bytes())
        if bootstrap.get("context") != context:
            raise ValueError("bootstrap source/model/data/mask/backend identity changed")
        fit = _numeric(t, bootstrap["fit_features"], fit_shape, np.dtype("float32"), fit_cap)
        if imported is not None:
            original, original_fit, provenance = imported
            validate_baseline_metrics(bootstrap["metrics"])
            if (rt.canonical(bootstrap.get("imported_baseline")) != rt.canonical(provenance)
                    or rt.canonical(bootstrap["metrics"]) != rt.canonical(original["metrics"])
                    or bootstrap["validation_seconds"] != original["validation_seconds"]
                    or any(bootstrap["fit_features"][key] != original["fit_features"][key] for key in ("sha256", "bytes"))):
                raise ValueError("resealed imported baseline provenance/measurement changed")
    else:
        if any((Path(t.config["output"])/name).exists() for name in ("bootstrap-fit.npy", "latest.pt", "mining.json", "hour-triplets.npy")):
            raise ValueError("preserve partial bootstrap/mining; no automatic retry")
        if imported is not None:
            original, fit, provenance = imported
            bootstrap = imported_bootstrap_record(t, context, original, provenance)
        else:
            captured = {}
            def sink(features, metrics):
                preparation("proxy", counts["total"], counts["total"])
                proxy_tails(np, features, t.rows, metrics, t.check, train, query)
                captured["fit"] = _npy(t, "bootstrap-fit.npy", features[fit_indices].astype(np.float32), fit_cap)
            started = time.monotonic()
            metrics = vf.Trainer.validate(t, feature_sink=sink,
                inference_batch_images=INFERENCE_POLICY["inference_batch_images"], progress=preparation)
            bootstrap = {"schema": "schubki-hour-bootstrap-v1", "context": context, "metrics": metrics,
                "fit_features": captured["fit"], "validation_seconds": time.monotonic()-started}
        t.disk.json("bootstrap.json", bootstrap)
        fit = _numeric(t, bootstrap["fit_features"], fit_shape, np.dtype("float32"), fit_cap)
    del imported
    t.bootstrap, t.bootstrap_pin = bootstrap, rt.pin(bootstrap_path)
    t.validation_seconds = bootstrap["validation_seconds"]
    check_denominators(t.config["schema"], bootstrap["metrics"])
    # Preserve exact parent bytes even if this recipe is too slow to proceed.
    exported = [(output/name).exists() for name in ("inference-best.pt", "inference-best.card.json")]
    if any(exported) and not all(exported):
        raise ValueError("preserve partial inference export pair")
    if not any(exported):
        copy_parent(t.config, t.disk)
    probe = rt.strict((Path(t.config["output"])/"probe.json").read_bytes())
    t.checkpoint_seconds = probe["checkpoint_export_seconds"]
    t.iteration_forecast = probe["whole_update_seconds"]
    t.final_reserve_seconds = final_reserve(t)
    # Admission uses actual remaining time after mining/restore/durable baseline setup.
    mining_path = Path(t.config["output"])/"mining.json"
    preparation("mining", None if reused_bootstrap else counts["total"], counts["total"])
    if mining_path.exists():
        mining = rt.strict(mining_path.read_bytes())
        if mining.get("bootstrap") != t.bootstrap_pin or mining.get("rules") != MINING:
            raise ValueError("frozen mining bootstrap/policy mismatch")
        t.triplets = _numeric(t, mining["artifact"], (mining["support"]["triplets"], 3), np.dtype("int64"), 3*1024**2)
    else:
        if (Path(t.config["output"])/"hour-triplets.npy").exists():
            raise ValueError("preserve unsealed mining artifact")
        random_triplets, miner_support = t.core.mine_triplets(t.rows, t.deadline, getattr(t, "policy", None))
        t.triplets, support = static_triplets(np, t.rows, fit_indices, fit, random_triplets, t.core.pair_masks, t.check, train)
        if t.config["schema"] == SCHEMA_V5:
            support["random_miner"] = miner_support
        artifact = _npy(t, "hour-triplets.npy", t.triplets, 3*1024**2)
        mining = {"schema": "schubki-hour-mining-v1", "bootstrap": t.bootstrap_pin, "rules": MINING, "artifact": artifact, "support": support}
        t.disk.json("mining.json", mining)
    if any(int(i) < 0 or int(i) >= len(t.rows) or t.rows[int(i)]["lap_id"] not in train for i in t.triplets.flat):
        raise ValueError("non-fit index admitted to sealed triplets")
    t.triplet_sha = hashlib.sha256(t.triplets.astype("<i8").tobytes()).hexdigest()
    t.support, t.mining_pin = mining["support"], rt.pin(mining_path)
    t.identity = digest({**context, "bootstrap": t.bootstrap_pin, "mining": t.mining_pin, "triplets": t.triplet_sha})
    t.optimizer = selected_optimizer(torch, t.model, t.mask, learning_rate=t.config["recipe"]["learning_rate"])
    random.seed(17); torch.manual_seed(17)
    t.sampler, t.augmentation = vf.Sampler(len(t.triplets), 17), random.Random(18)
    del fit
    gc.collect()
    preparation("ready", None if reused_bootstrap else counts["total"], counts["total"])


def check_denominators(schema, metrics):
    """Old pinned 1236/4497/247; v5 pins 2300/8269 and only requires eligible+unsupported==2300."""
    counts = policy_counts(schema)
    if metrics.get("all_queries") != counts["queries"] or metrics.get("gallery_count") != counts["gallery"]:
        raise ValueError("validation query/gallery denominator mismatch")
    if schema != SCHEMA_V5 and metrics.get("unsupported_queries") != 247:
        raise ValueError("bootstrap validation denominator mismatch")
    if schema == SCHEMA_V5:
        vf.integer(metrics.get("eligible_queries"), 0, counts["queries"])
        if metrics["eligible_queries"]+metrics.get("unsupported_queries", -1) != counts["queries"]:
            raise ValueError("measured eligible/unsupported must partition all 2300 queries")


def selective_update(torch, model, optimizer, batches, hard=True, *, observer=None):
    model.train()
    for module in model.modules():
        if isinstance(module, (torch.nn.modules.batchnorm._BatchNorm, torch.nn.modules.dropout._DropoutNd)):
            module.eval()
    optimizer.zero_grad(set_to_none=True)
    total = 0.
    for pixels, poses in batches:
        loss = vf.triplet_loss(torch, model(pixels), poses, hard=hard)/len(batches)
        loss.backward()
        total += float(loss.detach().cpu())
    parameters = dict(model.named_parameters())
    if tuple(sorted(name for name, p in parameters.items() if p.requires_grad)) != TRAINABLE:
        raise ValueError("trainable mask changed during update")
    for name, parameter in parameters.items():
        if name in TRAINABLE:
            if parameter.grad is None or not bool(torch.isfinite(parameter.grad).all()):
                raise ValueError("missing/nonfinite selected gradient")
        elif parameter.grad is not None:
            raise ValueError("frozen prefix received a gradient")
    if observer is not None:
        observer(parameters)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    if any(not bool(torch.isfinite(parameters[name]).all()) for name in TRAINABLE):
        raise ValueError("nonfinite selected update")
    parameters[TRAINABLE[-1]].detach().reshape(-1)[0].cpu().item()
    return total


class Trainer(vf.Trainer):
    checkpoint_schema, checkpoint_cap = CHECKPOINT, CHECKPOINT_CAP

    def __init__(self, config, deadline):
        super().__init__(config, deadline, initializer=_initialize_hour, source_factory=source_identity)

    def check(self, pressure=True):
        super().check(pressure)
        if shutil.disk_usage(self.disk.root).free < 2*rt.GIB:
            raise OSError("hour free-disk reserve reached")

    def checkpoint_context(self):
        return {"hour": {"parent": self.origin, "mask": self.mask, "frozen": self.frozen,
            "bootstrap": self.bootstrap_pin, "mining": self.mining_pin, "triplets_sha256": self.triplet_sha,
            "inference_policy": INFERENCE_POLICY, "scheduling_policy": SCHEDULING_POLICY,
            "optimizer_policy": {"learning_rate": self.config["recipe"]["learning_rate"], "weight_decay": .01, "fresh_optimizer": True}},
            "hour_timing": {"validation_seconds": self.validation_seconds, "checkpoint_seconds": self.checkpoint_seconds,
                            "iteration_forecast": self.iteration_forecast}, "loss_summary": dict(self.loss_stats)}

    def validate_checkpoint(self, state):
        vf.integer(state.get("step"), 0, self.config["recipe"]["max_updates"])
        if state.get("hour") != self.checkpoint_context()["hour"]:
            raise ValueError("hour parent/mask/optimizer ordering/mining/source resume mismatch")
        if fingerprints(self.torch, state["model"], self.mask["frozen_names"]) != self.frozen:
            raise ValueError("checkpoint frozen network differs from parent7750")
        groups = state["optimizer"].get("param_groups", [])
        if len(groups) != 1 or groups[0].get("params") != list(range(len(TRAINABLE))):
            raise ValueError("selective optimizer group/order mismatch")
        if type(groups[0].get("lr")) is not float or groups[0]["lr"] != self.config["recipe"]["learning_rate"]:
            raise ValueError("checkpoint optimizer learning rate differs from declared recipe")
        self._checked_loss_snapshot = validate_loss_summary(state.get("loss_summary"), state.get("step"))
        timing = state.get("hour_timing", {})
        if set(timing) != {"validation_seconds", "checkpoint_seconds", "iteration_forecast"}:
            raise ValueError("hour measured scheduling bounds missing")
        for key, value in timing.items():
            vf.finite(value, 0, 3600)
            setattr(self, key, max(getattr(self, key), value))

    def restore(self):
        super().restore()
        self.loss_stats = dict(self._checked_loss_snapshot)

    def checkpoint(self, name="latest.pt"):
        started = time.monotonic()
        if fingerprints(self.torch, self.model.state_dict(), self.mask["frozen_names"]) != self.frozen:
            raise ValueError("frozen parent tensors changed")
        value = super().checkpoint(name)
        self.checkpoint_seconds = max(self.checkpoint_seconds, time.monotonic()-started)
        return value

    def update(self):
        started = time.monotonic()
        loss = super().update(update_fn=selective_update, return_loss=True)
        self.loss_stats = observe_loss(self.loss_stats, loss, self.step)
        if self.step % 50 == 0:
            self.write_loss()
        self.last_iteration_seconds = time.monotonic()-started
        self.iteration_forecast = max(self.iteration_forecast, self.last_iteration_seconds)

    def write_loss(self):
        summary = validate_loss_summary(self.loss_stats, self.loss_stats["step"])
        if summary["step"] > self.step:
            raise ValueError("loss observation cannot precede its completed optimizer step")
        count = summary["count"]
        self.disk.json("training-loss.json", {"schema": "schubki-hour-loss-v1", "identity": self.identity,
            "fresh_optimizer_step": summary["step"], "attempted_step": self.step, "observation_count": count,
            "mean_loss": None if not count else summary["sum"]/count, "last_loss": summary["last"],
            "zero_loss_fraction": None if not count else summary["zero_count"]/count,
            "contains": "bounded aggregate of completed already-computed triplet losses; no extra model pass"})

    def validate(self):
        if self.step == 0:
            rt.verify_pin(self.bootstrap_pin, 2*1024**2)
            metrics = self.bootstrap["metrics"]
            self.disk.json("validation-000000.json", {"step": 0, "identity": self.identity, "metrics": metrics,
                "reused_exact_bootstrap": self.bootstrap_pin})
            return metrics
        started = time.monotonic()
        schema = getattr(self, "config", {}).get("schema")
        counts = policy_counts(schema)
        preparation = PreparationProgress(self.disk, counts["total"])
        def sink(features, metrics):
            preparation("proxy", counts["total"], counts["total"])
            proxy_tails(self.np, features, self.rows, metrics, self.check, counts["train_laps"], counts["query_laps"])
        metrics = super().validate(feature_sink=sink,
            inference_batch_images=INFERENCE_POLICY["inference_batch_images"], progress=preparation)
        check_denominators(schema, metrics)
        self.validation_seconds = max(self.validation_seconds, time.monotonic()-started)
        preparation("ready", counts["total"], counts["total"])
        return metrics

    def publish(self, state):
        status = {"state": state, "attempted_update": self.step, "last_durable_update": self.durable_step,
            "pending_validation": self.pending, "selected_update": None if self.best is None else self.best["step"],
            "identity": self.identity, "origin_update": 7750, "original_ancestor_update": 1125,
            "backend": "directml", "device_name": self.env["device_name"], "private_commit_bytes": rt.private_commit_bytes(),
            "system_free_bytes": rt.available_memory(), "trainable_parameters": self.mask["trainable_parameters"],
            "last_completed_update_seconds": getattr(self, "last_update_seconds", None),
            "last_completed_iteration_seconds": getattr(self, "last_iteration_seconds", None)}
        if self.config["schema"] == SCHEMA_V5:
            # pending_validation keeps its meaning (an interrupted validation); validation_needed is separate.
            status.update(lap_policy=FIVE_POLICY, validated_step=self.validated_step,
                          validation_needed=self.step != self.validated_step,
                          resume_kind=getattr(self, "resume_kind", None))
        self.disk.json("status.json", status)

    def export(self):
        state = restricted_load(self.torch, Path(self.config["output"])/"best.pt")
        if (state.get("schema") != CHECKPOINT or state.get("identity") != self.identity or state.get("pending_validation") is not False
                or state.get("validated_step") != state.get("step") or not state.get("best") or state["best"]["step"] != state["step"]):
            raise ValueError("hour export requires fully validated selected best")
        self.validate_checkpoint(state)
        if state["step"] and tuple(state["best"]["key"]) <= vf.selection(self.bootstrap["metrics"]):
            raise ValueError("changed hour export must improve the complete parent baseline")
        if state["step"] == 0:
            copy_parent(self.config, self.disk)
        else:
            payload = {"schema": vf.PUBLIC_CHECKPOINT, "step": state["step"], "training_identity": self.identity, "model": state["model"]}
            artifact = self.disk.atomic("inference-best.pt", lambda stream: self.torch.save(payload, stream), 512*1024**2)
            self.disk.json("inference-best.card.json", {"architecture": "vit_b_16", "dimension": 768,
                "preprocessing": vf.PREPROCESSING, "weights": artifact, "origin_step": 1125,
                "new_experiment_step": state["step"], "training_identity": self.identity, "experiment_identity": self.identity,
                "contains": "model parameters only; original ancestry1125; immediate parent7750 SHA256="+self.origin["model_sha256"]+
                    "; parent identity="+self.origin["origin_training_identity"]+"; learning_rate="+str(self.config["recipe"]["learning_rate"])+
                    "; new_experiment_step is fresh optimizer updates, not old continuation"})
        self.disk.json("hour-lineage.json", {"schema": "schubki-vit-hour-lineage-v1", "parent": self.origin,
            "new_experiment_step": state["step"], "selected_baseline": state["step"] == 0,
            "identity": self.identity, "source": self.sources, **self.checkpoint_context()["hour"]})


def final_reserve(trainer):
    vf.finite(trainer.validation_seconds, 0, 3600)
    vf.finite(trainer.checkpoint_seconds, 0, 3600)
    return max(900, 1.25*trainer.validation_seconds+4*trainer.checkpoint_seconds+120)


def safe_update_estimate(trainer):
    vf.finite(trainer.iteration_forecast, 0, 3600)
    # ESTIMATE: probe timing has no real RGB preparation; later whole iterations raise the maximum.
    return 1.5*trainer.iteration_forecast+3*trainer.validation_seconds/5733


def run_admission(trainer, deadline, pilot):
    final, update = final_reserve(trainer), safe_update_estimate(trainer)
    remaining = deadline-time.monotonic()
    io_cleanup = 4*trainer.checkpoint_seconds+120
    validated = trainer.pending is False and trainer.validated_step == trainer.step
    five = getattr(trainer, "config", {}).get("schema") == SCHEMA_V5
    if pilot and five:
        # Resource-only pilot: no step-50 validation, so one final reserve (from the measured full v5 baseline V).
        required = max(0, 50-trainer.step)*update+io_cleanup+final
        plan = "resource_pilot_one_final_reserve"
    elif pilot:
        required = max(0, 50-trainer.step)*update+2*final+60
        plan = "pilot_and_separate_later_finalization"
    elif validated:
        finalize_only = remaining <= update+final
        required = io_cleanup if finalize_only else update+final
        plan = "validated_finalize_only" if finalize_only else "updates_with_final_reserve"
    elif five and remaining <= update+final:
        # v5 only: a changed, not yet fully validated state (e.g. durable step50) may only be finalized. The loop's
        # update gate declines every update; one full validation and export follow. Below the final reserve: reject.
        required = final
        plan = "unvalidated_finalize_only"
    else:
        required = update+final
        plan = "unvalidated_update_and_finalization"
    trainer.disk.json("time-admission.json", {"schema": "schubki-hour-time-admission-v1", "policy": SCHEDULING_POLICY,
        "kind": "ESTIMATE", "plan": plan, "pilot": pilot, "step": trainer.step, "remaining_seconds": remaining,
        "max_observed_validation_seconds": trainer.validation_seconds, "max_complete_checkpoint_seconds": trainer.checkpoint_seconds,
        "max_observed_iteration_seconds": trainer.iteration_forecast, "estimated_update_seconds": update,
        "final_reserve_seconds": final, "estimated_required_seconds": required, "admitted": remaining > required,
        "limit": "hard deadline always applies; no promised update count or complete training throughput"})
    if remaining <= required:
        raise ValueError("estimated post-setup pilot/finalization does not fit original remaining wall; exact parent retained")


def periodic_validation_reserve(trainer):
    config = getattr(trainer, "config", {})
    terminal = config.get("schema") in (SCHEMA_V4, SCHEMA_V5) and trainer.step+1 == config["recipe"]["max_updates"] == 1000
    return 1.25*trainer.validation_seconds+3*trainer.checkpoint_seconds if (trainer.step+1) % 1000 == 0 and not terminal else 0


def update_allowed(trainer, deadline):
    return time.monotonic()+safe_update_estimate(trainer)+periodic_validation_reserve(trainer)+final_reserve(trainer) < deadline


def record_stop(trainer, reason, target, deadline):
    if type(reason) is not str or len(reason) > 80 or not (reason in ("target_reached", "minimum_final_reserve", "update_gate",
            "memory_pause_during_update", "memory_pause_between_updates") or reason.startswith("error_")):
        raise ValueError("bounded actual stop reason required")
    trainer.write_loss()
    trainer.disk.json("stopping.json", {"schema": "schubki-hour-stop-v1", "identity": trainer.identity,
        "reason": reason, "attempted_step": trainer.step, "fresh_optimizer_step": trainer.loss_stats["step"],
        "last_durable_update": trainer.durable_step, "validated_step": trainer.validated_step,
        "pending_validation": trainer.pending, "declared_target": target, "remaining_seconds": deadline-time.monotonic(),
        "timing_kind": "ESTIMATE", "estimated_final_reserve_seconds": final_reserve(trainer),
        "estimated_update_seconds": safe_update_estimate(trainer), "max_observed_validation_seconds": trainer.validation_seconds,
        "estimated_periodic_validation_seconds": periodic_validation_reserve(trainer),
        "max_complete_checkpoint_seconds": trainer.checkpoint_seconds, "recorded_unix_seconds": time.time(),
        "contains": "actual loop/exception trigger; subsequent validation/export outcome is in observed host receipts"})


def run(config, deadline, pilot=False, resume=False):
    return vf.run(config, deadline, pilot, resume, trainer_factory=Trainer, update_allowed=update_allowed,
        admission=run_admission, stopped=record_stop)


def probe(config, deadline):
    """Actual parent/mask/native policy gate on synthetic RGB only, never real images."""
    output = Path(config["output"])
    if any((output/name).exists() for name in ("probe.json", "probe-memory.json", "bootstrap.json", "latest.pt", "probe-state.pt", "probe-network.pt")):
        raise ValueError("new hour probe only; preserve prior/partial attempts")
    disk = HourDisk(output, 8*rt.GIB)
    trace, trace_started = [], time.monotonic()
    def memory_sample(phase):
        if len(trace) >= 16:
            raise ValueError("probe memory trace sample cap")
        trace.append({"phase": phase, "elapsed_seconds": time.monotonic()-trace_started,
            "private_commit_bytes": rt.private_commit_bytes(), "system_available_bytes": rt.available_memory()})
        disk.json("probe-memory.json", {"schema": "schubki-hour-probe-memory-v1", "config_sha256": digest(config),
            "ram_soft_bytes": config["limits"]["ram_soft_bytes"], "ram_hard_bytes": config["limits"]["ram_hard_bytes"],
            "system_reserve_bytes": config["limits"]["system_reserve_bytes"], "samples": trace})
    memory_sample("start")
    vf.memory_boundary(config)
    torch, env, device = vf.backend(config)
    model, _ = vf.build_model(torch)
    origin = verify_parent(torch, config, model)
    mask = select_trainable(model)
    frozen = fingerprints(torch, model.state_dict(), mask["frozen_names"])
    model.to(device)
    optimizer = selected_optimizer(torch, model, mask, learning_rate=config["recipe"]["learning_rate"])
    memory_sample("parent_loaded")
    started = time.monotonic()
    pixels = torch.randn(3, 3, 224, 224).to(device)
    poses = torch.tensor([[0.,0.,0.], [1.,0.,0.], [30.,0.,0.]], dtype=torch.float64)
    parameters = dict(model.named_parameters())
    before = {name: parameters[name].detach().cpu().clone() for name in REPRESENTATIVES}
    def gradients(values):
        if any(not bool((values[name].grad.detach().abs() > 0).any()) for name in REPRESENTATIVES):
            raise ValueError("nonzero representative late-block/norm gradients required")
    selective_update(torch, model, optimizer, [(pixels, poses)], observer=gradients)
    whole = time.monotonic()-started
    if any(torch.equal(before[name], parameters[name].detach().cpu()) for name in REPRESENTATIVES):
        raise ValueError("representative selected parameters did not change")
    del before
    if fingerprints(torch, model.state_dict(), mask["frozen_names"]) != frozen:
        raise ValueError("probe frozen prefix changed")
    memory_sample("selected_update_verified")
    checkpoint_started = time.monotonic()
    saved = vf.cpu_tree(torch, {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "mask": mask})
    artifact = disk.atomic("probe-state.pt", lambda stream: torch.save(saved, stream), CHECKPOINT_CAP)
    del saved
    gc.collect()
    memory_sample("checkpoint_saved_cpu_copy_released")
    loaded = restricted_load(torch, artifact["path"])
    if loaded["mask"] != mask or fingerprints(torch, loaded["model"]) != fingerprints(torch, model.state_dict()):
        raise ValueError("full model/mask restricted checkpoint reload mismatch")
    memory_sample("checkpoint_mapped_verified")
    selective_update(torch, model, optimizer, [(pixels, poses)])
    expected = fingerprints(torch, model.state_dict())
    expected_optimizer = optimizer_fingerprint(torch, optimizer.state_dict())
    model.load_state_dict(loaded["model"], strict=True)
    optimizer.load_state_dict(loaded["optimizer"])
    del loaded
    gc.collect()
    selective_update(torch, model, optimizer, [(pixels, poses)])
    if (fingerprints(torch, model.state_dict()) != expected or optimizer_fingerprint(torch, optimizer.state_dict()) != expected_optimizer
            or fingerprints(torch, model.state_dict(), mask["frozen_names"]) != frozen):
        raise ValueError("selected optimizer continuation/frozen equality mismatch")
    optimizer.zero_grad(set_to_none=True)
    del optimizer, parameters, expected_optimizer
    gc.collect()
    memory_sample("continuation_verified_optimizer_released")
    with torch.no_grad():
        model.eval()
        embedding = model(pixels).detach().cpu()
    if tuple(embedding.shape) != (3, 768) or not bool(torch.isfinite(embedding).all()):
        raise ValueError("complete768-D synthetic output required")
    del embedding, pixels, poses
    probe_identity = digest({"config": config, "source": source_identity(), "environment": env, "parent": origin,
        "mask": mask, "inference_policy": INFERENCE_POLICY, "scheduling_policy": SCHEDULING_POLICY})
    payload = {"schema": vf.PUBLIC_CHECKPOINT, "step": 2, "training_identity": probe_identity, "model": vf.cpu_tree(torch, model.state_dict())}
    network = disk.atomic("probe-network.pt", lambda stream: torch.save(payload, stream), 512*1024**2)
    del payload
    gc.collect()
    memory_sample("network_saved_cpu_copy_released")
    reloaded = restricted_load(torch, network["path"])
    if set(reloaded) != {"schema", "step", "training_identity", "model"} or fingerprints(torch, reloaded["model"]) != expected:
        raise ValueError("full network-only export reload mismatch")
    del reloaded, expected
    memory_sample("full_proof_before_final_cleanup")
    model.zero_grad(set_to_none=True)
    del model
    gc.collect()
    memory_sample("full_proof_after_final_cleanup")
    vf.memory_boundary(config)
    if time.monotonic() >= deadline-60:
        raise TimeoutError("hour selective probe exhausted startup/checkpoint reserve")
    disk.json("probe.json", {"config_sha256": digest(config), "source": source_identity(), "environment": env,
        "selected_micro_triplets": 1, "parent": origin, "mask": mask, "frozen": frozen,
        "inference_policy": INFERENCE_POLICY, "scheduling_policy": SCHEDULING_POLICY,
        "whole_update_seconds": whole, "checkpoint_export_seconds": time.monotonic()-checkpoint_started,
        "checkpoint": artifact, "network_export": network, "memory_trace": rt.pin(output/"probe-memory.json"),
        "synthetic_only": True})
    return {"state": "probe_complete", "selected_micro_triplets": 1}

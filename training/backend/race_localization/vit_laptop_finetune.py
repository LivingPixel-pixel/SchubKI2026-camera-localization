"""Full ViT-B/16 explicit CPU/DirectML adaptation from a private model-only checkpoint.

Derived from the reviewed full-encoder-v1 trainer. All Torch imports are lazy.
RGB is the only network input. Labels supervise offline fit1-2 mining and
complete lap3 selection only. No downloads or silent backend substitution.
"""
from __future__ import annotations

import gc
import hashlib
import importlib.metadata
import io
import math
import os
from pathlib import Path
import random
import time

from . import vit_laptop_runtime as rt

SCHEMA = "schubki-vit-laptop-config-v1"
CHECKPOINT = "schubki-vit-laptop-checkpoint-v1"
PUBLIC_CHECKPOINT = "schubki-full-encoder-checkpoint-v1"
PREPROCESSING = "torchvision.ViT_B_16_Weights.IMAGENET1K_V1.transforms;PIL_RGB;no_augmentation"
ARCHITECTURE = "vit_b_16"
DIMENSION = 768


def finite(value, low, high):
    if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
        raise ValueError("finite numeric setting outside bounds")
    return value


def integer(value, low, high):
    if type(value) is not int:
        raise ValueError("native integer required")
    return finite(value, low, high)


def read_config(path):
    path = Path(path)
    if path.stat().st_size > 128*1024:
        raise ValueError("config size cap")
    config = rt.strict(path.read_bytes())
    if set(config) != {"schema", "dataset", "warm_start", "warm_start_card", "output", "recipe", "limits", "backend", "device_index"}:
        raise ValueError("unexpected config fields")
    if config["schema"] != SCHEMA or config["backend"] not in ("directml", "cpu"):
        raise ValueError("explicit laptop backend required")
    integer(config["device_index"], 0, 0)
    dataset = config["dataset"]
    if set(dataset) != {"directory", "sha256", "cache_root"}:
        raise ValueError("prepared dataset identity required")
    digest = dataset["sha256"]
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError("full lowercase SHA256 required")
    for value in (dataset["directory"], dataset["cache_root"], config["output"],
                  config["warm_start"]["path"], config["warm_start_card"]["path"]):
        if not Path(value).is_absolute():
            raise ValueError("absolute local paths required")
    r = config["recipe"]
    if set(r) != {"seed", "learning_rate", "weight_decay", "margin", "max_updates", "save_every", "validate_every",
                  "accumulation", "hard_negatives", "brightness", "contrast", "min_crop_scale"}:
        raise ValueError("unexpected recipe fields")
    integer(r["seed"], 0, 2**31-1)
    finite(r["learning_rate"], 1e-7, 1e-3)
    finite(r["weight_decay"], 0, .1)
    finite(r["margin"], .2, .2)
    integer(r["max_updates"], 50, 12000)
    integer(r["save_every"], 50, 50)
    integer(r["validate_every"], 1000, 1000)
    integer(r["accumulation"], 1, 16)
    if type(r["hard_negatives"]) is not bool:
        raise ValueError("hard_negatives must be boolean")
    finite(r["brightness"], 0, .3)
    finite(r["contrast"], 0, .3)
    finite(r["min_crop_scale"], .85, 1)
    limits = config["limits"]
    if set(limits) != {"wall_seconds", "final_reserve_seconds", "ram_soft_bytes", "ram_hard_bytes", "system_reserve_bytes", "disk_bytes", "threads"}:
        raise ValueError("unexpected limits")
    integer(limits["wall_seconds"], 120, 28800)
    integer(limits["final_reserve_seconds"], 30, 3600)
    if limits["final_reserve_seconds"] >= limits["wall_seconds"]:
        raise ValueError("reserve must fit wall")
    integer(limits["ram_soft_bytes"], rt.GIB, 48*rt.GIB)
    integer(limits["ram_hard_bytes"], limits["ram_soft_bytes"]+2*rt.GIB, 64*rt.GIB)
    integer(limits["system_reserve_bytes"], 4*rt.GIB if config["backend"] == "directml" else 2*rt.GIB, 4*rt.GIB if config["backend"] == "directml" else 2*rt.GIB)
    integer(limits["ram_hard_bytes"], 8*rt.GIB if config["backend"] == "directml" else 6*rt.GIB, 12*rt.GIB)
    integer(limits["disk_bytes"], 8*rt.GIB, 64*rt.GIB)
    integer(limits["threads"], 1, 8)
    for source in (dataset["directory"], dataset["cache_root"]):
        output_path, source_path = Path(config["output"]).resolve(), Path(source).resolve()
        if output_path.is_relative_to(source_path) or source_path.is_relative_to(output_path):
            raise ValueError("output and dataset/cache must be disjoint")
    return config


def source_identity():
    root = Path(__file__).parents[2]
    files = [Path(__file__), Path(rt.__file__), Path(__file__).with_name("place_model.py"),
             root / "backend/tools/train_vit_laptop.py", Path(rt.base.__file__),
             root / "tools/start_vit_laptop.ps1", root / "backend/requirements-vit-laptop.txt"]
    return {str(path.relative_to(root)): rt.sha_file(path, 2*1024**2) for path in files}


def plain_metadata(value):
    """Persist only restricted-loader-safe builtins, including string subclasses."""
    if value is None:
        return None
    if isinstance(value, str):
        return str(value)
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float) and math.isfinite(value):
        return float(value)
    if isinstance(value, (list, tuple)):
        return [plain_metadata(item) for item in value]
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        return {str(key): plain_metadata(item) for key, item in value.items()}
    raise TypeError("environment metadata must contain finite plain primitives")


PINNED = {"torch": "2.4.1", "torchvision": "0.19.1", "torch-directml": "0.2.5.dev240914",
          "numpy": "1.26.4", "pillow": "13.0.0"}


def environment(torch, config):
    import torchvision
    import sys
    versions = {name: importlib.metadata.version(name) for name in PINNED}
    if (versions != PINNED or str(torch.__version__) != "2.4.1+cpu"
            or str(torchvision.__version__) != "0.19.1+cpu" or torch.version.cuda is not None
            or sys.version_info[:2] != (3, 12)):
        raise RuntimeError("requires exact isolated Python3.12/Torch2.4.1+cpu/vision0.19.1+cpu laptop stack")
    if config["backend"] == "directml":
        import torch_directml
        name = torch_directml.device_name(config["device_index"]).rstrip("\x00")
        if not name:
            raise RuntimeError("DirectML device identity unavailable")
    else:
        name = "CPU: " + (__import__("platform").processor() or "Windows processor")
    return plain_metadata({"versions": versions, "torch": str(torch.__version__), "torchvision": str(torchvision.__version__),
        "python": sys.version, "backend": config["backend"], "device_index": config["device_index"],
        "device_name": name, "shared_memory": config["backend"] == "directml",
        "cpu_operator_fallback_possible": config["backend"] == "directml"})


def backend(config):
    import torch
    torch.set_num_threads(config["limits"]["threads"])
    env = environment(torch, config)
    torch.manual_seed(config["recipe"]["seed"])
    if config["backend"] == "directml":
        import torch_directml
        device = torch_directml.device(config["device_index"])
    else:
        device = torch.device("cpu")
    return torch, env, device


def complete(model):
    # A bounded CPU materialization observes work after the optimizer update.
    return next(model.parameters()).detach().reshape(-1)[0].cpu().item()


def memory_boundary(config, *, pressure=True):
    limits = config["limits"]
    if rt.private_commit_bytes() >= limits["ram_hard_bytes"]:
        raise MemoryError("hard process private commit exceeded")
    if pressure and (rt.private_commit_bytes() >= limits["ram_soft_bytes"]
            or rt.available_memory() < limits["system_reserve_bytes"]):
        raise MemoryPause()


def build_model(torch):
    from torchvision.models import vit_b_16, ViT_B_16_Weights
    model = vit_b_16(weights=None)
    model.heads = torch.nn.Identity()
    model.requires_grad_(True)
    torch.backends.mha.set_fastpath_enabled(False)
    for layer in model.encoder.layers:
        original = layer.self_attention.forward
        def forward(*args, _original=original, **kwargs):
            kwargs["need_weights"] = True
            return _original(*args, **kwargs)
        layer.self_attention.forward = forward
    return model, ViT_B_16_Weights.IMAGENET1K_V1.transforms()


def verify_warm_start(torch, config, model):
    path = rt.verify_pin(config["warm_start"])
    card_path = rt.verify_pin(config["warm_start_card"], 1024**2)
    card = rt.strict(card_path.read_bytes())
    required = {"schema", "architecture", "embedding_dim", "preprocessing", "origin_step", "training_identity",
                "weights_file", "weights_sha256", "weights_bytes"}
    if not required.issubset(card) or (card["schema"] != "schubki-vit-weights-card-v1"
            or card["architecture"] != ARCHITECTURE or card["embedding_dim"] != DIMENSION
            or card["preprocessing"] != "torchvision ViT_B_16_Weights.IMAGENET1K_V1.transforms; PIL RGB; no training augmentation at inference"
            or card["origin_step"] != 1125):
        raise ValueError("warm-start architecture/preprocessing/origin card mismatch")
    if card["weights_sha256"] != config["warm_start"]["sha256"] or card["weights_bytes"] != config["warm_start"]["bytes"]:
        raise ValueError("warm-start card weight pin mismatch")
    state = torch.load(path, map_location="cpu", weights_only=True)
    if (set(state) != {"schema", "step", "training_identity", "model"} or state["schema"] != PUBLIC_CHECKPOINT
            or state["step"] != 1125 or state["training_identity"] != card.get("training_identity")):
        raise ValueError("model-only warm start mismatch; optimizer import forbidden")
    if any(not isinstance(value, torch.Tensor) or value.device.type != "cpu" or not bool(torch.isfinite(value).all())
           for value in state["model"].values()):
        raise ValueError("invalid warm-start tensor")
    model.load_state_dict(state["model"], strict=True)
    return {"origin_step": state["step"], "origin_training_identity": state["training_identity"],
            "model_sha256": config["warm_start"]["sha256"]}


class Sampler:
    def __init__(self, count, seed):
        self.rng, self.order, self.cursor = random.Random(seed), list(range(count)), 0
        self.rng.shuffle(self.order)

    def next(self, count):
        result = []
        while len(result) < count:
            if self.cursor == len(self.order):
                self.rng.shuffle(self.order)
                self.cursor = 0
            take = min(count-len(result), len(self.order)-self.cursor)
            result.extend(self.order[self.cursor:self.cursor+take])
            self.cursor += take
        return result

    def state(self):
        return {"order": self.order.copy(), "cursor": self.cursor, "rng": self.rng.getstate()}

    def restore(self, state):
        if sorted(state["order"]) != list(range(len(self.order))) or not 0 <= state["cursor"] <= len(self.order):
            raise ValueError("sampler state mismatch")
        self.order, self.cursor = state["order"].copy(), state["cursor"]
        self.rng.setstate(state["rng"])


def triplet_loss(torch, raw, poses, margin=.2, hard=True):
    """Own legal positive; mine only known >=25m negatives in this physical batch."""
    if raw.ndim != 2 or raw.shape[0] % 3 or not bool(torch.isfinite(raw).all()):
        raise ValueError("invalid RGB encoder output")
    if bool((raw.square().sum(1) <= 1e-24).any()):
        raise ValueError("zero embedding")
    values = torch.nn.functional.normalize(raw, dim=1)
    vectors = values.reshape(-1, 3, values.shape[-1])
    a, p, n = vectors[:, 0], vectors[:, 1], vectors[:, 2]
    positive = (a-p).square().sum(1)
    negative = (a-n).square().sum(1)
    if hard:
        if poses is None or tuple(poses.shape) != (len(raw), 3):
            raise ValueError("offline fit labels required for legal negative mask")
        if poses.device.type != "cpu" or poses.dtype != torch.float64:
            raise ValueError("offline geometry must remain CPU float64")
        anchor_xy = poses[::3, :2]
        distances = (anchor_xy[:, None]-poses[None, :, :2]).square().sum(2)
        legal = distances >= 25.0**2
        own = torch.arange(len(a))*3+2
        if not bool(legal[torch.arange(len(a)), own].all()):
            raise ValueError("own negative violates distance boundary")
        squared = (2-2*a @ values.T).clamp_min(0)
        negative = squared.masked_fill(~legal.to(device=raw.device), float("inf")).min(1).values
    loss = (margin+positive-negative).clamp_min(0).mean()
    if not bool(torch.isfinite(loss)):
        raise ValueError("nonfinite triplet loss")
    return loss


def optimizer_update(torch, model, optimizer, batches, hard=True):
    model.train()
    for module in model.modules():
        if isinstance(module, (torch.nn.modules.batchnorm._BatchNorm, torch.nn.modules.dropout._DropoutNd)):
            module.eval()
    optimizer.zero_grad(set_to_none=True)
    total = 0.
    for pixels, poses in batches:
        loss = triplet_loss(torch, model(pixels), poses, hard=hard)/len(batches)
        loss.backward()
        total += float(loss.detach().cpu())
    gradient_flags = [torch.isfinite(parameter.grad).all() for parameter in model.parameters() if parameter.grad is not None]
    if not gradient_flags or not bool(torch.stack(gradient_flags).all()):
        raise ValueError("missing/nonfinite gradient; retain last durable checkpoint")
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    if not bool(torch.stack([torch.isfinite(parameter).all() for parameter in model.parameters()]).all()):
        raise ValueError("nonfinite updated parameter; retain last durable checkpoint")
    return total


def cpu_tree(torch, value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_tree(torch, item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_tree(torch, item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(torch, item) for item in value)
    return value


def selection(metrics):
    # Current inference uses top1. Never select primarily on top5.
    return (metrics["recall1_hits"], metrics["recall5_hits"], -metrics["false_distant_top1"])


def retrieval_metrics(np, features, rows, pair_masks, check=lambda: None, *, gallery_laps=(1, 2), query_laps=(3,)):
    features = np.asarray(features, dtype=np.float32)
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    if not np.isfinite(features).all() or (norms <= 1e-12).any() or len(features) != len(rows):
        raise ValueError("complete finite nonzero validation embeddings required")
    if set(gallery_laps) & set(query_laps):
        raise ValueError("gallery and query laps must be disjoint")
    features = features/norms
    gallery = np.array([i for i, row in enumerate(rows) if row["lap_id"] in gallery_laps])
    queries = np.array([i for i, row in enumerate(rows) if row["lap_id"] in query_laps])
    if not len(gallery) or not len(queries):
        raise ValueError("complete train gallery and validation queries required")
    labels = np.array([[row[k] for k in ("body_x_m", "body_y_m", "body_yaw_rad")] for row in rows])
    hits1 = hits5 = eligible = distant = 0
    xy_squared, lateral_squared = [], []
    for query in queries:
        check()
        positive, negative = pair_masks(labels[query], labels[gallery])
        eligible += int(positive.any())
        rank = np.argsort(-(features[gallery] @ features[query]), kind="stable")[:5]
        hits1 += int(positive[rank[0]])
        hits5 += int(positive[rank].any())
        distant += int(negative[rank[0]])
        delta = labels[gallery[rank[0]], :2]-labels[query, :2]
        xy_squared.append(float(delta @ delta))
        yaw = labels[query, 2]
        lateral_squared.append(float(-math.sin(yaw)*delta[0]+math.cos(yaw)*delta[1])**2)
    return {"all_queries": len(queries), "gallery_count": len(gallery), "eligible_queries": eligible,
        "unsupported_queries": len(queries)-eligible, "recall1_hits": hits1, "recall5_hits": hits5,
        "recall1_all": hits1/len(queries), "recall5_all": hits5/len(queries), "false_distant_top1": distant,
        "reference_copy_xy_rmse_m_proxy": math.sqrt(sum(xy_squared)/len(queries)),
        "heading_normal_lateral_rmse_m_proxy": math.sqrt(sum(lateral_squared)/len(queries)),
        "scope": ("lap3 retrieval and reference-copy proxies, not official pose/coverage/runtime score"
                  if tuple(query_laps) == (3,) else FIVE_SCOPE)}


FIVE_SCOPE = ("laps6-7 retrieval against laps1-5 gallery and reference-copy proxies, "
              "not official pose/coverage/runtime score")


class MemoryPause(Exception):
    pass


class Trainer:
    checkpoint_schema = CHECKPOINT
    checkpoint_cap = 2*rt.GIB

    def __init__(self, config, deadline, *, initializer=None, source_factory=None):
        self.config, self.deadline = config, deadline
        memory_boundary(config)
        self.torch, self.env, self.device = backend(config)
        memory_boundary(config)
        torch = self.torch
        import numpy as np
        from . import place_model as core
        self.np, self.core = np, core
        self.disk = rt.Disk(config["output"], config["limits"]["disk_bytes"])
        probe = rt.strict((Path(config["output"])/"probe.json").read_bytes())
        config_sha = hashlib.sha256(rt.canonical(config)).hexdigest()
        self.sources = (source_identity if source_factory is None else source_factory)()
        if probe["config_sha256"] != config_sha or probe["source"] != self.sources or probe["environment"] != self.env:
            raise ValueError("probe identity differs; do not change the frozen experiment")
        self.micro = probe["selected_micro_triplets"]
        integer(self.micro, 1, 4)
        d = config["dataset"]
        self.core_metadata, self.observations, self.rows = core.load_dataset(d["directory"], d["sha256"], d["cache_root"])
        self.policy = core.dataset_policy(self.core_metadata)
        self.resource_pilot = False
        # Trusted Python callers may initialize a separately validated recipe.
        # No serialized config selects factories or changes this default policy.
        if initializer is not None:
            initializer(self)
            return
        if self.policy is not None:
            raise ValueError("five-lap datasets require the explicit hour v5 policy, never the laptop default")
        self.triplets, self.support = core.mine_triplets(self.rows, deadline)
        if any(self.rows[int(i)]["lap_id"] not in (1, 2) for i in self.triplets.flat):
            raise ValueError("lap3 admitted to gradients/mining")
        self.triplet_sha = hashlib.sha256(self.triplets.astype("<i8").tobytes()).hexdigest()
        self.identity = hashlib.sha256(rt.canonical({"config": config, "source": self.sources,
            "environment": self.env, "micro_triplets": self.micro, "triplets": self.triplet_sha})).hexdigest()
        r = config["recipe"]
        random.seed(r["seed"])
        torch.manual_seed(r["seed"])
        self.model, self.preprocess = build_model(torch)
        self.origin = verify_warm_start(torch, config, self.model)
        memory_boundary(config)
        self.model.to(self.device, dtype=torch.float32)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=r["learning_rate"], weight_decay=r["weight_decay"], foreach=False)
        self.sampler = Sampler(len(self.triplets), r["seed"])
        self.augmentation = random.Random(r["seed"]+1)
        self.step, self.best, self.pending = 0, None, True
        self.validated_step, self.durable_step = -1, None

    def check(self, pressure=True):
        if time.monotonic() >= self.deadline-10:
            raise TimeoutError("final checkpoint deadline reached")
        memory_boundary(self.config, pressure=pressure)

    def lap_sets(self):
        """(train/gallery laps, validation/query laps) of the bound dataset policy; old default needs no core."""
        policy = getattr(self, "policy", None)
        if policy is None:
            return (1, 2), (3,)
        return self.core.train_laps(policy), (6, 7)

    def pixels(self, index, augment):
        from PIL import Image
        from torchvision.transforms import functional
        row, observation = self.rows[index], self.observations[index]
        if augment and row["lap_id"] not in self.lap_sets()[0]:
            raise ValueError("validation-lap augmentation forbidden")
        path = self.core.local_path(self.config["dataset"]["cache_root"], observation["image_path"])
        rt.verify_pin({"path": str(path), "sha256": observation["image_sha256"], "bytes": observation["image_bytes"]}, 8*1024**2)
        with Image.open(path) as image:
            if image.width*image.height > 4*1024**2:
                raise ValueError("decoded pixel cap")
            image = image.convert("RGB")
            if augment:
                r = self.config["recipe"]
                scale = self.augmentation.uniform(r["min_crop_scale"], 1)
                width, height = max(1, round(image.width*scale)), max(1, round(image.height*scale))
                image = functional.crop(image, self.augmentation.randrange(image.height-height+1),
                    self.augmentation.randrange(image.width-width+1), height, width)
                image = functional.adjust_brightness(image, self.augmentation.uniform(1-r["brightness"], 1+r["brightness"]))
                image = functional.adjust_contrast(image, self.augmentation.uniform(1-r["contrast"], 1+r["contrast"]))
            return self.preprocess(image)

    def update(self, *, update_fn=None, return_loss=False):
        torch = self.torch
        batches = []
        for _ in range(self.config["recipe"]["accumulation"]):
            selected = self.triplets[self.sampler.next(self.micro)]
            indices = [int(i) for i in selected.flat]
            pixels = torch.stack([self.pixels(i, True) for i in indices]).to(self.device, dtype=torch.float32)
            poses = torch.tensor([[self.rows[i][k] for k in ("body_x_m", "body_y_m", "body_yaw_rad")]
                                  for i in indices], device="cpu", dtype=torch.float64)
            batches.append((pixels, poses))
        complete(self.model)
        started = time.monotonic()
        loss = (optimizer_update if update_fn is None else update_fn)(
            torch, self.model, self.optimizer, batches, self.config["recipe"]["hard_negatives"])
        complete(self.model)
        self.step += 1
        del batches
        self.last_update_seconds = time.monotonic()-started
        self.check()  # fresh post-update memory accounting
        if return_loss:
            return loss

    def checkpoint_context(self):
        return {}

    def validate_checkpoint(self, state):
        pass

    def checkpoint(self, name="latest.pt"):
        self.optimizer.zero_grad(set_to_none=True)
        complete(self.model)
        self.check(pressure=False)
        state = {"schema": self.checkpoint_schema, "identity": self.identity, "source": self.sources, "environment": self.env,
            "step": self.step, "model": cpu_tree(self.torch, self.model.state_dict()),
            "optimizer": cpu_tree(self.torch, self.optimizer.state_dict()), "torch_rng": self.torch.get_rng_state(),
            "python_rng": random.getstate(),
            "augmentation_rng": self.augmentation.getstate(), "sampler": self.sampler.state(),
            "best": self.best, "pending_validation": self.pending, "validated_step": self.validated_step, "origin": self.origin,
            **self.checkpoint_context()}
        self.disk.atomic(name, lambda stream: self.torch.save(state, stream), self.checkpoint_cap)
        if name == "latest.pt":
            self.durable_step = self.step
        return state

    def restore(self):
        state = self.torch.load(Path(self.config["output"])/"latest.pt", map_location="cpu", weights_only=True)
        if state.get("schema") != self.checkpoint_schema or state.get("identity") != self.identity:
            raise ValueError("incompatible checkpoint source/config/data/environment identity")
        self.validate_checkpoint(state)
        self.model.load_state_dict(state["model"], strict=True)
        self.optimizer.load_state_dict(state["optimizer"])
        self.sampler.restore(state["sampler"])
        self.augmentation.setstate(state["augmentation_rng"])
        random.setstate(state["python_rng"])
        self.torch.set_rng_state(state["torch_rng"])
        self.step, self.best, self.pending = state["step"], state["best"], state["pending_validation"]
        self.validated_step, self.durable_step = state["validated_step"], self.step

    def validate(self, *, feature_sink=None, inference_batch_images=None, progress=None):
        batch_images = self.micro*3 if inference_batch_images is None else inference_batch_images
        integer(batch_images, 1, 24)
        total = len(self.rows)
        self.model.eval()
        parts = []
        torch = self.torch
        self.optimizer.zero_grad(set_to_none=True)
        gc.collect()
        if progress is not None:
            progress("encoding", 0, total)
        with torch.no_grad():
            for start in range(0, total, batch_images):
                self.check()
                end = min(start+batch_images, total)
                indices = range(start, end)
                batch = torch.stack([self.pixels(i, False) for i in indices]).to(self.device)
                parts.append(self.model(batch).detach().cpu().numpy())
                del batch
                self.check()
                if progress is not None:
                    progress("encoding", end, total)
        if progress is not None:
            progress("retrieval", total, total)
        # Old policy keeps the exact original positional call; only the five-lap policy passes lap sets.
        laps = {} if getattr(self, "policy", None) is None else dict(zip(("gallery_laps", "query_laps"), self.lap_sets()))
        if feature_sink is None:
            metrics = retrieval_metrics(self.np, self.np.concatenate(parts), self.rows, self.core.pair_masks, self.check, **laps)
        else:
            features = self.np.concatenate(parts)
            metrics = retrieval_metrics(self.np, features, self.rows, self.core.pair_masks, self.check, **laps)
            feature_sink(features, metrics)
        self.disk.json(f"validation-{self.step:06d}.json", {"step": self.step, "identity": self.identity, "metrics": metrics})
        return metrics

    def validate_durable(self):
        self.pending = True
        state = self.checkpoint()  # recover this validation if interrupted
        if self.step == 0 and not (Path(self.config["output"])/"baseline.pt").exists():
            self.disk.atomic("baseline.pt", lambda stream: self.torch.save(state, stream), self.checkpoint_cap)
        del state
        metrics = self.validate()
        improved = self.best is None or selection(metrics) > tuple(self.best["key"])
        if improved:
            self.best = {"step": self.step, "key": list(selection(metrics))}
        self.pending = False
        self.validated_step = self.step
        if improved:
            # Publish best before latest refers to it. Failed best writes leave pending validation durable.
            self.checkpoint("best.pt")
        self.checkpoint()

    def publish(self, state):
        self.disk.json("status.json", {"state": state, "attempted_update": self.step, "last_durable_update": self.durable_step,
            "pending_validation": self.pending,
            "selected_update": None if self.best is None else self.best["step"], "private_commit_bytes": rt.private_commit_bytes(),
            "backend": self.config["backend"], "device_name": self.env["device_name"],
            "system_free_bytes": rt.available_memory(), "identity": self.identity,
            "origin_update": 1125, "last_completed_update_seconds": getattr(self, "last_update_seconds", None)})

    def export(self):
        state = self.torch.load(Path(self.config["output"])/"best.pt", map_location="cpu", weights_only=True)
        if (state.get("schema") != self.checkpoint_schema or state.get("identity") != self.identity
                or state.get("pending_validation") is not False or state.get("validated_step") != state.get("step")
                or not isinstance(state.get("best"), dict) or state["best"].get("step") != state.get("step")):
            raise ValueError("export requires this experiment's fully validated selected best checkpoint")
        export_step = state["step"] or self.origin["origin_step"]
        export_identity = self.identity if state["step"] else self.origin["origin_training_identity"]
        payload = {"schema": PUBLIC_CHECKPOINT, "step": export_step, "training_identity": export_identity, "model": state["model"]}
        artifact = self.disk.atomic("inference-best.pt", lambda stream: self.torch.save(payload, stream), 512*1024**2)
        self.disk.json("inference-best.card.json", {"architecture": ARCHITECTURE, "dimension": DIMENSION,
            "preprocessing": PREPROCESSING, "weights": artifact, "origin_step": 1125, "new_experiment_step": state["step"],
            "training_identity": export_identity, "experiment_identity": self.identity,
            "contains": "model parameters only; no optimizer/RNG/labels/metrics"})


def probe(config, deadline):
    """Planned synthetic batches only; no dataset or warm-start reads."""
    torch, env, device = backend(config)
    output = Path(config["output"])
    if (output/"latest.pt").exists() or (output/"probe.json").exists():
        raise ValueError("probe must precede the frozen experiment")
    disk = rt.Disk(output, config["limits"]["disk_bytes"])
    results, selected = [], None
    for count in (1, 2, 4):
        if time.monotonic() >= deadline-60:
            raise TimeoutError("probe deadline")
        model = optimizer = pixels = poses = None
        try:
            memory_boundary(config)
            torch.manual_seed(config["recipe"]["seed"])
            model, _ = build_model(torch)
            model.to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=.01, foreach=False)
            pixels = torch.randn(count*3, 3, 224, 224).to(device)
            poses = torch.tensor([[float(i*100+offset), 0., 0.] for i in range(count) for offset in (0, 1, 30)], dtype=torch.float64)
            complete(model)
            started = time.monotonic()
            optimizer_update(torch, model, optimizer, [(pixels, poses)], config["recipe"]["hard_negatives"])
            complete(model)
            elapsed = time.monotonic()-started
            saved = cpu_tree(torch, {"model": model.state_dict(), "optimizer": optimizer.state_dict()})
            artifact = disk.atomic("probe-roundtrip.pt", lambda stream: torch.save(saved, stream), 2*rt.GIB)
            del saved
            loaded = torch.load(artifact["path"], map_location="cpu", weights_only=True)
            for key, value in model.state_dict().items():
                if not torch.equal(value.cpu(), loaded["model"][key]):
                    raise ValueError("probe save/load differs")
            model.load_state_dict(loaded["model"], strict=True)
            optimizer.load_state_dict(loaded["optimizer"])
            del loaded
            optimizer_update(torch, model, optimizer, [(pixels, poses)], config["recipe"]["hard_negatives"])
            complete(model)
            memory_boundary(config)
            (output/"probe-roundtrip.pt").unlink()
            results.append({"micro_triplets": count, "passed": True, "completed_update_seconds": elapsed,
                "private_commit_bytes": rt.private_commit_bytes(), "system_free_bytes": rt.available_memory()})
            selected = count
        except (MemoryError, MemoryPause):
            results.append({"micro_triplets": count, "passed": False, "reason": "laptop_memory_boundary"})
            break
        finally:
            del model, optimizer, pixels, poses
            gc.collect()
    if selected is None:
        raise RuntimeError("no physical batch passed the explicit laptop backend probe")
    result = {"config_sha256": hashlib.sha256(rt.canonical(config)).hexdigest(), "source": source_identity(),
              "environment": env, "selected_micro_triplets": selected, "planned_trials": results}
    disk.json("probe.json", result)
    return {"state": "probe_complete", "selected_micro_triplets": selected}


def run(config, deadline, pilot=False, resume=False, *, trainer_factory=None, update_allowed=None, admission=None, stopped=None):
    if not pilot and not resume:
        raise ValueError("long training must resume a completed50-update pilot")
    if not resume and (Path(config["output"])/"latest.pt").exists():
        raise ValueError("existing experiment requires explicit --resume")
    trainer = (Trainer if trainer_factory is None else trainer_factory)(config, deadline)
    target = min(50, config["recipe"]["max_updates"]) if pilot else config["recipe"]["max_updates"]
    if resume:
        trainer.restore()
    if not pilot:
        pilot_record = rt.strict((Path(config["output"])/"pilot.json").read_bytes())
        if getattr(trainer, "resource_pilot", False):
            # The original resource-only pilot50 provenance is always required exactly.
            expected = {"identity": trainer.identity, "kind": "resource_only", "completed_update": 50,
                        "durable_update": 50, "validated_step": 0}
            if type(pilot_record) is not dict or any(pilot_record.get(key) != value for key, value in expected.items()):
                raise ValueError("matching durable resource-only50 pilot provenance required; never remine or rebaseline")
            # The restored latest.pt (same identity, checked by restore) may be the initial handoff or a later
            # durable state, including an interrupted validation that is recovered once below. No auto retry.
            step, validated = trainer.step, trainer.validated_step
            if (type(step) is not int or not 50 <= step <= target or type(trainer.pending) is not bool
                    or trainer.durable_step != step or type(validated) is not int
                    or not (validated == 0 or 50 <= validated <= step)):
                raise ValueError("restored state is not a compatible durable continuation of the resource pilot")
            trainer.resume_kind = ("initial_resource_handoff" if (step, validated, trainer.pending) == (50, 0, False)
                                   else "later_durable_resume")
        elif pilot_record.get("identity") != trainer.identity or pilot_record.get("validated_update") != 50:
            raise ValueError("matching complete50-update pilot required")
    updating = False
    stop_notified = False
    def notify_stop(reason):
        nonlocal stop_notified
        if stopped is not None:
            stop_notified = True
            stopped(trainer, reason, target, deadline)
    try:
        trainer.publish("starting")
        if trainer.pending:
            trainer.validate_durable()
        if admission is not None:
            admission(trainer, deadline, pilot)
        stop_reason = "target_reached"
        resource = pilot and getattr(trainer, "resource_pilot", False)
        while trainer.step < target:
            if time.monotonic() >= deadline-config["limits"]["final_reserve_seconds"]:
                stop_reason = "minimum_final_reserve"
                break
            if update_allowed is not None and not update_allowed(trainer, deadline):
                stop_reason = "update_gate"
                break
            trainer.check()
            updating = True
            trainer.update()
            updating = False
            if not resource and (trainer.step % config["recipe"]["validate_every"] == 0 or trainer.step == target):
                trainer.validate_durable()
            elif trainer.step % config["recipe"]["save_every"] == 0:
                trainer.checkpoint()
            if trainer.step % 10 == 0:
                trainer.publish("training")
        notify_stop(stop_reason)
        if resource:
            # Resource-only pilot: no step-50 validation and no learned-weight export; the parent stays selected.
            if trainer.step != 50 or trainer.durable_step != 50 or trainer.validated_step != 0 or trainer.pending is not False:
                trainer.publish("pilot_incomplete")
                raise RuntimeError("resource pilot did not seal a durable50 state")
            trainer.disk.json("pilot.json", {"identity": trainer.identity, "kind": "resource_only",
                "completed_update": 50, "durable_update": 50, "validated_step": 0,
                "parent_baseline": getattr(trainer, "bootstrap_pin", None),
                "contains": "resource measurement only; not validated50; parent remains selected"})
            trainer.publish("pilot_complete")
            return {"state": "pilot_complete", "update": trainer.step}
        if trainer.step and trainer.validated_step != trainer.step:
            trainer.validate_durable()
        trainer.export()
        if pilot:
            if trainer.step != 50 or trainer.validated_step != 50:
                trainer.publish("pilot_incomplete")
                raise RuntimeError("pilot did not complete50updates and full validation")
            trainer.disk.json("pilot.json", {"identity": trainer.identity, "validated_update": 50})
        trainer.publish("pilot_complete" if pilot else "complete")
        return {"state": "pilot_complete" if pilot else "complete", "update": trainer.step}
    except MemoryPause:
        # A post-update pressure event marks updating: do not serialize consumed/partial state.
        if not updating:
            trainer.checkpoint()
        if not stop_notified:
            notify_stop("memory_pause_during_update" if updating else "memory_pause_between_updates")
        trainer.publish("paused_memory")
        return {"state": "paused_memory", "update": trainer.step}
    except BaseException as error:
        if not stop_notified:
            try:
                notify_stop("error_"+type(error).__name__)
            except BaseException as diagnostic:
                if hasattr(error, "add_note"):
                    error.add_note("stop diagnostic failed: "+type(diagnostic).__name__)
        trainer.publish("failed_keep_last_durable")
        raise

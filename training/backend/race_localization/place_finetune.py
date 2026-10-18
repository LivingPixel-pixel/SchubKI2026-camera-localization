"""Full-backbone place retrieval fine-tuning (TEAM-TRAIN-02).

Laps 1-2 supply gradients, mining and augmentation. Lap 3 only selects
checkpoints. The network sees RGB pixels only; labels drive offline sampling
and metrics. Nothing here downloads: pretrained weights are a pinned local file.
Numeric metrics go to local files only; console and status output stay sanitized.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

import numpy as np

from race_localization import place_model as pm


PACK_SCHEMA = "schubki-place-pixel-pack-v1"
CHECKPOINT_SCHEMA = "schubki-place-finetune-checkpoint-v2"
PROVENANCE_SCHEMA = "schubki-place-training-provenance-v1"
VALIDATION_SCHEMA = "schubki-place-validation-v1"
SELECTION_ORDER = ["recall_at_5_all", "recall_at_1_all", "earlier_step"]
STATUS_SCHEMA = "schubki-place-finetune-status-v1"
ARCHITECTURE = "mobilenet_v3_large.IMAGENET1K_V2.full_finetune.avgpool960.l2"
PACK_EDGE = 232
CROP = 224
# Eval crop of a packed tile equals place_model resize232 + center_crop224.
PACK_PREPROCESSING = {**pm.PREPROCESSING, "pack": "resize_short_edge232_then_center_crop232x232_uint8_CHW"}
# Public torchvision ImageNet checkpoints (fetch explicitly; the trainer never downloads).
BACKBONES = {
    "mobilenet_v3_large": {"dim": 960, "weights": "MobileNet_V3_Large_Weights.IMAGENET1K_V2",
                           "url": "https://download.pytorch.org/models/mobilenet_v3_large-5c1a4163.pth"},
    "resnet50": {"dim": 2048, "weights": "ResNet50_Weights.IMAGENET1K_V2",
                 "url": "https://download.pytorch.org/models/resnet50-11ad3fa6.pth"},
    "efficientnet_v2_s": {"dim": 1280, "weights": "EfficientNet_V2_S_Weights.IMAGENET1K_V1",
                          "url": "https://download.pytorch.org/models/efficientnet_v2_s-dd5fe13b.pth"},
}
AUGMENTATIONS = ("basic", "strong")
# Settings that a resume must match exactly. Budgets may change between runs.
COMPATIBLE_KEYS = ("dataset_sha256", "pack_sha256", "weights_sha256", "batch_pairs",
                   "accumulation", "learning_rate", "weight_decay", "warmup_updates",
                   "max_updates", "validate_every", "margin", "seed", "jitter_px",
                   "color_jitter", "device", "backbone", "input_width", "augment")


def pack_preprocessing(input_width=CROP):
    """224 keeps the exact v1 identity; wider crops keep the full 232 short-edge resize."""
    if input_width == CROP:
        return PACK_PREPROCESSING
    return {**PACK_PREPROCESSING, "center_crop": [CROP, input_width],
            "pack": f"resize_short_edge232_then_center_crop232x{input_width + 8}_uint8_CHW"}


def architecture(backbone, input_width):
    base = f"{backbone}.{BACKBONES[backbone]['weights'].split('.')[-1]}.full_finetune.avgpool{BACKBONES[backbone]['dim']}.l2"
    return base if input_width == CROP else f"{base}.input224x{input_width}"


def build_features(torch, backbone):
    """Feature extractor without classifier; spatial mean pooling happens in Embedder."""
    import torchvision.models as models
    if backbone == "mobilenet_v3_large":
        return models.mobilenet_v3_large(weights=None), lambda network: network.features
    if backbone == "resnet50":
        return models.resnet50(weights=None), lambda network: torch.nn.Sequential(*list(network.children())[:-2])
    if backbone == "efficientnet_v2_s":
        return models.efficientnet_v2_s(weights=None), lambda network: network.features
    raise ValueError(f"unknown backbone {backbone!r}")


@dataclass(frozen=True)
class Config:
    pack_dir: str
    pack_sha256: str
    dataset_sha256: str
    weights: str
    weights_sha256: str
    output: str
    batch_pairs: int = 8
    accumulation: int = 2
    learning_rate: float = 3e-5
    weight_decay: float = 1e-4
    warmup_updates: int = 100
    max_updates: int = 3000
    validate_every: int = 250
    margin: float = 0.2
    seed: int = 17
    jitter_px: int = 4
    color_jitter: float = 0.2
    device: str = "cuda"
    wall_seconds: int = 14400
    soft_ram_gib: float = 10.0
    hard_ram_gib: float = 12.0
    vram_gib: float = 1.3
    threads: int = 4
    backbone: str = "mobilenet_v3_large"
    input_width: int = CROP
    augment: str = "basic"

    def __post_init__(self):
        if self.backbone not in BACKBONES or self.augment not in AUGMENTATIONS:
            raise ValueError(f"backbone in {sorted(BACKBONES)}, augment in {AUGMENTATIONS}")
        pm.integer(self.input_width, CROP, 1024)
        if self.input_width % 8:
            raise ValueError("input_width must be a multiple of 8")
        for name in ("pack_sha256", "dataset_sha256", "weights_sha256"):
            pm.sha(getattr(self, name))
        pm.integer(self.batch_pairs, 2, 256)
        pm.integer(self.accumulation, 1, 64)
        pm.integer(self.warmup_updates, 0, 100000)
        pm.integer(self.max_updates, 1, 1000000)
        pm.integer(self.validate_every, 1, 1000000)
        pm.integer(self.seed, 0, 2**31 - 1)
        pm.integer(self.jitter_px, 0, (PACK_EDGE - CROP) // 2)
        pm.integer(self.wall_seconds, 1, 7 * 86400)
        pm.integer(self.threads, 1, 64)
        for name in ("learning_rate", "weight_decay", "margin", "color_jitter",
                     "soft_ram_gib", "hard_ram_gib", "vram_gib"):
            if pm.finite(getattr(self, name)) < 0:
                raise ValueError(f"{name}: nonnegative value required")
        if not 0 < self.soft_ram_gib < self.hard_ram_gib or self.vram_gib <= 0 or self.learning_rate <= 0:
            raise ValueError("memory caps or learning rate")
        if self.color_jitter >= 1:
            raise ValueError("color_jitter must be below 1")
        if self.device not in ("cuda", "cpu"):
            raise ValueError("device must be cuda or cpu")

    @classmethod
    def load(cls, path):
        value = pm.parse_json(Path(path).read_bytes())
        names = {field.name for field in fields(cls)}
        if not isinstance(value, dict) or set(value) - names:
            raise ValueError("unknown config fields")
        return cls(**value)

    def compatible(self):
        return {key: getattr(self, key) for key in COMPATIBLE_KEYS}


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def source_sha():
    return file_sha(Path(__file__))


def source_snapshot():
    """Pin the actual checkout and executed source bytes, including local changes."""
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                            capture_output=True, text=True, timeout=5, check=True)
    commit = result.stdout.strip()
    if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
        raise ValueError("full actual source checkout commit required")
    paths = ("backend/race_localization/place_finetune.py", "backend/race_localization/place_model.py",
             "backend/tools/train_place_finetune.py")
    return {"checkout_commit": commit, "files": {name: file_sha(root / name) for name in paths}}


def training_provenance(config):
    return {"schema": PROVENANCE_SCHEMA, "source": source_snapshot(),
            "base_weights_sha256": config.weights_sha256, "dataset_sha256": config.dataset_sha256,
            "pack_sha256": config.pack_sha256, "recipe_sha256": pm.digest(pm.canonical(config.compatible()))}


def model_state_sha256(torch, state):
    """Incrementally hash tensors without cloning the entire network on the CPU."""
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        tensor = value.detach().to("cpu").contiguous()
        header = pm.canonical({"name": name, "dtype": str(tensor.dtype), "shape": list(tensor.shape)})
        digest.update(len(header).to_bytes(8, "little"))
        digest.update(header)
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().data)
    return digest.hexdigest()


def validation_record(step, metrics, model_sha256, provenance):
    return {"schema": VALIDATION_SCHEMA, "step": step, "selection_order": SELECTION_ORDER.copy(),
            "model_sha256": pm.sha(model_sha256), "provenance_sha256": pm.digest(pm.canonical(provenance)),
            "metrics": metrics, "metrics_sha256": pm.digest(pm.canonical(metrics))}


def selection_key(record):
    metrics = record["metrics"]
    return [metrics["recall_at_5_all"], metrics["recall_at_1_all"], -record["step"]]


def checkpoint_metadata(payload):
    """Reject older ambiguous continuity and verify linked source/selection metadata."""
    if not isinstance(payload, dict) or payload.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError("checkpoint v2 with pending-validation and provenance state required; old resume/export unsupported")
    required = {"step", "model", "optimizer", "numpy_rng", "torch_rng", "cuda_rng", "compatible",
                "source_sha256", "provenance", "pending_validation", "validation", "best"}
    if not required <= payload.keys() or type(payload["pending_validation"]) is not bool:
        raise ValueError("incomplete checkpoint validation/provenance state")
    step = pm.integer(payload["step"], 0, 1000000)
    settings, provenance = payload["compatible"], payload["provenance"]
    pm.exact(settings, set(COMPATIBLE_KEYS))
    pm.integer(settings["validate_every"], 1, 1000000)
    if step > pm.integer(settings["max_updates"], 1, 1000000):
        raise ValueError("checkpoint step exceeds the declared recipe")
    pm.exact(provenance, {"schema", "source", "base_weights_sha256", "dataset_sha256", "pack_sha256", "recipe_sha256"})
    if provenance["schema"] != PROVENANCE_SCHEMA:
        raise ValueError("unknown training provenance")
    source = provenance["source"]
    pm.exact(source, {"checkout_commit", "files"})
    commit = source["checkout_commit"]
    if not isinstance(commit, str) or len(commit) != 40 or any(c not in "0123456789abcdef" for c in commit):
        raise ValueError("checkpoint source commit")
    pm.exact(source["files"], {"backend/race_localization/place_finetune.py",
                             "backend/race_localization/place_model.py", "backend/tools/train_place_finetune.py"})
    for value in source["files"].values():
        pm.sha(value)
    if source["files"]["backend/race_localization/place_finetune.py"] != payload["source_sha256"]:
        raise ValueError("checkpoint trainer source identity mismatch")
    for provenance_key, setting_key in (("base_weights_sha256", "weights_sha256"),
                                         ("dataset_sha256", "dataset_sha256"), ("pack_sha256", "pack_sha256")):
        if pm.sha(provenance[provenance_key]) != settings[setting_key]:
            raise ValueError("checkpoint base/dataset/pack provenance mismatch")
    if provenance["recipe_sha256"] != pm.digest(pm.canonical(settings)):
        raise ValueError("checkpoint recipe provenance mismatch")

    def check_validation(record):
        pm.exact(record, {"schema", "step", "selection_order", "model_sha256", "provenance_sha256", "metrics", "metrics_sha256"})
        pm.integer(record["step"], 0, step)
        if (record["schema"] != VALIDATION_SCHEMA or record["selection_order"] != SELECTION_ORDER
                or record["provenance_sha256"] != pm.digest(pm.canonical(provenance))
                or record["metrics_sha256"] != pm.digest(pm.canonical(record["metrics"]))):
            raise ValueError("checkpoint validation provenance mismatch")
        pm.sha(record["model_sha256"])
        metrics = record["metrics"]
        pm.exact(metrics, {"gallery_count", "all_queries", "eligible_queries", "no_positive_queries",
                           "positive_support_coverage", "recall_at_1_all", "recall_at_5_all",
                           "recall_at_1_eligible", "recall_at_5_eligible", "false_distant_top1", "interpretation"})
        count = pm.integer(metrics["all_queries"], 1, 6000)
        eligible = pm.integer(metrics["eligible_queries"], 0, count)
        pm.integer(metrics["gallery_count"], 1, 6000)
        pm.integer(metrics["false_distant_top1"], 0, count)
        if (metrics["interpretation"] != "retrieval_only_shadow_ranking"
                or metrics["no_positive_queries"] != count - eligible
                or metrics["positive_support_coverage"] != eligible / count):
            raise ValueError("checkpoint validation protocol/count mismatch")
        for key in ("recall_at_5_all", "recall_at_1_all"):
            if not 0 <= pm.finite(metrics[key]) <= eligible / count:
                raise ValueError("invalid validation selection metric")
        if metrics["recall_at_1_all"] > metrics["recall_at_5_all"]:
            raise ValueError("invalid validation recall ordering")

    validation, best = payload["validation"], payload["best"]
    if validation is not None:
        check_validation(validation)
    if best is not None:
        pm.exact(best, {"step", "key", "validation"})
        pm.integer(best["step"], 0, step)
        check_validation(best["validation"])
        if best["step"] != best["validation"]["step"] or best["key"] != selection_key(best["validation"]):
            raise ValueError("checkpoint best-selection metadata mismatch")
        if validation is None or best["step"] > validation["step"]:
            raise ValueError("checkpoint best is newer than completed validation")
        if (tuple(best["key"]) < tuple(selection_key(validation))
                or (best["step"] == validation["step"] and best["validation"] != validation)):
            raise ValueError("checkpoint best does not cover completed validation")
    if (validation is None or best is None) and not (step == 0 and payload["pending_validation"]):
        raise ValueError("checkpoint lacks completed baseline validation")
    if (not payload["pending_validation"] and (step % settings["validate_every"] == 0 or step == settings["max_updates"])
            and (validation is None or validation["step"] != step)):
        raise ValueError("checkpoint omits required validation")
    return provenance


def atomic_bytes(path, data):
    """Replace path only after the full payload is on disk."""
    path = Path(path)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_json(path, value):
    atomic_bytes(path, pm.canonical(value))


def pack_tile(rgb, input_width=CROP):
    """RGB uint8 HWC to uint8 CHW 232x(input_width+8) tile, place_model resize semantics."""
    import torch
    from torchvision.transforms import InterpolationMode
    from torchvision.transforms import functional
    if not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("RGB uint8 pixels required")
    tensor = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1)
    tensor = functional.resize(tensor, [PACK_EDGE], interpolation=InterpolationMode.BILINEAR, antialias=True)
    height, width = tensor.shape[1:]
    tile_width = input_width + (PACK_EDGE - CROP)
    # Offsets keep the later center crop identical to a direct center crop of the resized image.
    top = int(round((height - CROP) / 2.0)) - (PACK_EDGE - CROP) // 2
    left = int(round((width - input_width) / 2.0)) - (PACK_EDGE - CROP) // 2
    if top < 0 or left < 0 or top + PACK_EDGE > height or left + tile_width > width:
        raise ValueError("image too small for the pack tile")
    return tensor[:, top:top + PACK_EDGE, left:left + tile_width].contiguous().numpy()


def build_pack(dataset_dir, dataset_sha256, cache_root, output, wall_seconds=3600,
               decode=pm.decode_rgb, input_width=CROP):
    """Decode every fit1-3 image once into a pinned uint8 pixel pack."""
    deadline = time.monotonic() + wall_seconds
    _, observations, rows = pm.load_dataset(dataset_dir, dataset_sha256, cache_root)
    output = pm.new_output(output)
    try:
        pixels = np.lib.format.open_memmap(output / "pixels.npy", mode="w+", dtype=np.uint8,
                                           shape=(len(rows), 3, PACK_EDGE, input_width + PACK_EDGE - CROP))
        for index, observation in enumerate(observations):
            pm.check_deadline(deadline)
            path = pm.local_path(cache_root, observation["image_path"])
            data = pm.read_pinned(path, observation["image_sha256"], 8 * 1024 * 1024)
            if len(data) != observation["image_bytes"]:
                raise ValueError("image byte identity mismatch")
            pixels[index] = pack_tile(decode(data, 4 * 1024 * 1024), input_width)
        pixels.flush()
        del pixels
        pm.write_json(output / "rows.json", rows)
        manifest = {"schema": PACK_SCHEMA, "complete": True, "count": len(rows),
                    "dataset_sha256": pm.sha(dataset_sha256), "preprocessing": pack_preprocessing(input_width),
                    "pixels_sha256": file_sha(output / "pixels.npy"),
                    "rows_sha256": file_sha(output / "rows.json"), "source_sha256": source_sha()}
        pm.write_json(output / "pack.json", manifest)
        return {"complete": True, "count": len(rows), "pack_sha256": file_sha(output / "pack.json")}
    except Exception as error:
        pm.failed_output(output, error)
        raise


def load_pack(pack_dir, pack_sha256, dataset_sha256, input_width=CROP):
    pack_dir = Path(pack_dir)
    manifest = pm.parse_json(pm.read_pinned(pack_dir / "pack.json", pack_sha256))
    if manifest.get("schema") != PACK_SCHEMA or manifest.get("complete") is not True:
        raise ValueError("incomplete or wrong pixel pack")
    if manifest["dataset_sha256"] != dataset_sha256 or manifest["preprocessing"] != pack_preprocessing(input_width):
        raise ValueError("pack dataset or preprocessing identity mismatch")
    rows = pm.parse_json(pm.read_pinned(pack_dir / "rows.json", manifest["rows_sha256"]))
    pm.rows_check(rows)
    if len(rows) != manifest["count"] or file_sha(pack_dir / "pixels.npy") != manifest["pixels_sha256"]:
        raise ValueError("pack rows or pixels mismatch")
    pixels = np.load(pack_dir / "pixels.npy", mmap_mode="r", allow_pickle=False)
    if pixels.dtype != np.uint8 or pixels.shape != (len(rows), 3, PACK_EDGE, input_width + PACK_EDGE - CROP):
        raise ValueError("pack pixel shape")
    return pixels, rows


def positive_lists(rows):
    """Cross-train-lap positives per train anchor under the fixed 5m/15deg rule."""
    train = np.array([i for i, row in enumerate(rows) if row["lap_id"] in (1, 2)], dtype=np.int64)
    labels = np.array([[rows[i][k] for k in ("body_x_m", "body_y_m", "body_yaw_rad")] for i in train])
    laps = np.array([rows[i]["lap_id"] for i in train])
    hashes = np.array([rows[i]["image_sha256"] for i in train])
    anchors, positives = [], []
    for local, index in enumerate(train):
        positive, _ = pm.pair_masks(labels[local], labels)
        positive &= (laps != laps[local]) & (hashes != hashes[local])
        if positive.any():
            anchors.append(int(index))
            positives.append(train[positive])
    if not anchors:
        raise pm.NotReady("no cross-train-lap positives under the fixed 5m/15deg rule")
    return np.array(anchors, dtype=np.int64), positives


def pose_array(rows):
    return np.array([[row[k] for k in ("body_x_m", "body_y_m", "body_yaw_rad")] for row in rows],
                    dtype=np.float64)


def sample_batch(rng, anchors, positives, count):
    chosen = rng.choice(len(anchors), size=count, replace=len(anchors) < count)
    anchor = anchors[chosen]
    positive = np.array([rng.choice(positives[i]) for i in chosen], dtype=np.int64)
    return anchor, positive


def batch_hard_loss(torch, embeddings, poses, margin):
    """Symmetric triplet hinge with the hardest in-batch negative at >=25 m.

    embeddings is [2B, D] ordered (anchors, positives). Pairs between 5 and
    25 m are ignored. Anchors without any valid negative do not contribute.
    """
    count = embeddings.shape[0] // 2
    distance = torch.cdist(poses[:, :2], poses[:, :2])
    negative = distance >= 25.0
    squared = (2.0 - 2.0 * embeddings @ embeddings.T).clamp_min(0.0)
    partner = torch.cat([torch.arange(count, 2 * count), torch.arange(0, count)]).to(embeddings.device)
    positive_distance = squared[torch.arange(2 * count, device=embeddings.device), partner]
    masked = squared.masked_fill(~negative, float("inf"))
    hardest, _ = masked.min(dim=1)
    valid = torch.isfinite(hardest)
    if not bool(valid.any()):
        return None, 0
    hinge = (margin + positive_distance - hardest).clamp_min(0.0)
    return hinge[valid].mean(), int(valid.sum())


class Trainer:
    def __init__(self, config: Config, model_factory=None):
        import torch
        self.torch = torch
        self.config = config
        if config.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable; no silent CPU fallback")
        self.device = torch.device(config.device)
        torch.set_num_threads(config.threads)
        if config.device == "cuda":
            total = torch.cuda.get_device_properties(0).total_memory
            torch.cuda.set_per_process_memory_fraction(min(1.0, config.vram_gib * 2**30 / total), 0)
        weight_bytes = pm.read_pinned(config.weights, config.weights_sha256, 256 * 1024 * 1024)
        self.model = (model_factory or self.pretrained)(weight_bytes).to(self.device)
        self.pixels, self.rows = load_pack(config.pack_dir, config.pack_sha256, config.dataset_sha256,
                                           config.input_width)
        self.poses = pose_array(self.rows)
        self.anchors, self.positives = positive_lists(self.rows)
        self.mean = torch.tensor(pm.PREPROCESSING["mean"], device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor(pm.PREPROCESSING["std"], device=self.device).view(1, 3, 1, 1)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.learning_rate,
                                           weight_decay=config.weight_decay)
        self.provenance = training_provenance(config)
        self.pending_validation, self.validation = False, None

    def pretrained(self, weight_bytes):
        import io
        torch = self.torch
        network, features = build_features(torch, self.config.backbone)
        network.load_state_dict(torch.load(io.BytesIO(weight_bytes), map_location="cpu", weights_only=True), strict=True)
        return Embedder(torch, features(network))

    def tensors(self, indices, train):
        torch = self.torch
        batch = torch.from_numpy(np.stack([self.pixels[i] for i in indices])).to(self.device)
        margin, width = (PACK_EDGE - CROP) // 2, self.config.input_width
        if train and self.config.jitter_px:
            crops = []
            for tile in batch:
                dy, dx = (int(v) for v in torch.randint(-self.config.jitter_px, self.config.jitter_px + 1, (2,)))
                crops.append(tile[:, margin + dy:margin + dy + CROP, margin + dx:margin + dx + width])
            batch = torch.stack(crops)
        else:
            batch = batch[:, :, margin:margin + CROP, margin:margin + width]
        batch = batch.float() / 255.0
        if train and self.config.color_jitter:
            strength = self.config.color_jitter
            shape = (batch.shape[0], 1, 1, 1)
            brightness = 1 + (torch.rand(shape, device=self.device) * 2 - 1) * strength
            contrast = 1 + (torch.rand(shape, device=self.device) * 2 - 1) * strength
            saturation = 1 + (torch.rand(shape, device=self.device) * 2 - 1) * strength
            batch = batch * brightness
            gray = (0.299 * batch[:, 0:1] + 0.587 * batch[:, 1:2] + 0.114 * batch[:, 2:3])
            batch = (batch - gray.mean(dim=(2, 3), keepdim=True)) * contrast + gray.mean(dim=(2, 3), keepdim=True)
            gray = (0.299 * batch[:, 0:1] + 0.587 * batch[:, 1:2] + 0.114 * batch[:, 2:3])
            batch = (batch - gray) * saturation + gray
            batch = batch.clamp(0.0, 1.0)
        if train and self.config.augment == "strong":
            batch = self.strong_augment(batch)
        return (batch - self.mean) / self.std

    def strong_augment(self, batch):
        """Lighting and motion robustness; never flips or rotates (keeps handedness/viewpoint)."""
        torch = self.torch
        shape = (batch.shape[0], 1, 1, 1)
        gamma = torch.exp((torch.rand(shape, device=self.device) * 2 - 1) * 0.4)
        batch = batch.clamp_min(1e-6) ** gamma
        exposure = torch.exp((torch.rand(shape, device=self.device) * 2 - 1) * 0.5)
        batch = batch * exposure
        blurred = []
        for image in batch:
            if float(torch.rand(())) < 0.3:  # horizontal motion blur at race speed
                size = int(torch.randint(1, 5, ())) * 2 + 1
                kernel = torch.full((3, 1, 1, size), 1.0 / size, device=self.device)
                image = torch.nn.functional.conv2d(image.unsqueeze(0), kernel, padding=(0, size // 2), groups=3)[0]
            blurred.append(image)
        batch = torch.stack(blurred)
        noise = torch.rand(shape, device=self.device) * 0.02
        batch = batch + torch.randn_like(batch) * noise
        return batch.clamp(0.0, 1.0)

    def learning_rate(self, step):
        warmup = self.config.warmup_updates
        return self.config.learning_rate * (min(1.0, step / warmup) if warmup else 1.0)

    def update(self, rng, step):
        torch = self.torch
        self.model.train()
        for group in self.optimizer.param_groups:
            group["lr"] = self.learning_rate(step)
        self.optimizer.zero_grad(set_to_none=True)
        losses, contributing = [], 0
        for _ in range(self.config.accumulation):
            anchor, positive = sample_batch(rng, self.anchors, self.positives, self.config.batch_pairs)
            indices = np.concatenate([anchor, positive])
            embeddings = self.model(self.tensors(indices, train=True))
            poses = torch.from_numpy(self.poses[indices]).to(self.device, torch.float32)
            loss, valid = batch_hard_loss(torch, embeddings, poses, self.config.margin)
            if loss is None:
                continue
            (loss / self.config.accumulation).backward()
            losses.append(float(loss.detach()))
            contributing += valid
        if not losses:
            raise pm.NotReady("batch without valid negatives")
        for parameter in self.model.parameters():
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                raise ValueError("nonfinite gradient; last durable checkpoint kept")
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        return float(np.mean(losses)), contributing

    def embed_all(self, deadline=None, batch=32):
        torch = self.torch
        self.model.eval()
        output = None
        with torch.no_grad():
            for start in range(0, len(self.rows), batch):
                if deadline is not None:
                    pm.check_deadline(deadline)
                indices = np.arange(start, min(start + batch, len(self.rows)))
                values = self.model(self.tensors(indices, train=False)).float().cpu().numpy()
                if output is None:
                    output = np.empty((len(self.rows), values.shape[1]), dtype=np.float32)
                output[indices] = values
        return output

    def validate(self, deadline=None):
        return retrieval_metrics(self.embed_all(deadline), self.rows, deadline=deadline)

    def checkpoint(self, step, rng, best):
        torch = self.torch
        return {"schema": CHECKPOINT_SCHEMA, "step": step, "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(), "numpy_rng": rng.bit_generator.state,
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state() if self.config.device == "cuda" else None,
                "compatible": self.config.compatible(),
                "source_sha256": self.provenance["source"]["files"]["backend/race_localization/place_finetune.py"],
                "best": best,
                "provenance": self.provenance, "pending_validation": self.pending_validation,
                "validation": self.validation}

    def save(self, path, payload):
        import io
        buffer = io.BytesIO()
        self.torch.save(payload, buffer)
        atomic_bytes(path, buffer.getvalue())

    def restore(self, path):
        torch = self.torch
        payload = torch.load(path, map_location="cpu", weights_only=True)
        provenance = checkpoint_metadata(payload)
        if payload["compatible"] != self.config.compatible():
            changed = sorted(k for k in COMPATIBLE_KEYS if payload["compatible"].get(k) != getattr(self.config, k))
            raise ValueError(f"incompatible resume settings: {changed}")
        if provenance != self.provenance:
            raise ValueError("training source/base/data identity changed since checkpoint; resume rejected")
        if (payload["validation"] is not None and payload["validation"]["step"] == payload["step"]
                and payload["validation"]["model_sha256"] != model_state_sha256(torch, payload["model"])):
            raise ValueError("checkpoint model differs from its completed validation")
        self.model.load_state_dict(payload["model"], strict=True)
        self.optimizer.load_state_dict(payload["optimizer"])
        rng = np.random.default_rng()
        rng.bit_generator.state = payload["numpy_rng"]
        torch.set_rng_state(payload["torch_rng"])
        if payload["cuda_rng"] is not None:
            torch.cuda.set_rng_state(payload["cuda_rng"])
        self.pending_validation, self.validation = payload["pending_validation"], payload["validation"]
        return payload["step"], rng, payload["best"]


def retrieval_metrics(features, rows, deadline=None):
    """place_model.retrieval_metrics for any embedding width; 960-D delegates to it unchanged."""
    features = np.asarray(features)
    if features.ndim == 2 and features.shape[1] == 960:
        return pm.retrieval_metrics(features, rows, deadline=deadline)
    pm.rows_check(rows)
    if features.ndim != 2 or features.shape[0] != len(rows) or features.dtype != np.float32:
        raise ValueError("float32[N,D] features required")
    gallery = np.array([i for i, row in enumerate(rows) if row["lap_id"] in (1, 2)], dtype=np.int64)
    queries = np.array([i for i, row in enumerate(rows) if row["lap_id"] == 3], dtype=np.int64)
    if not len(gallery) or not len(queries):
        raise pm.NotReady("lap1-2 gallery and lap3 checkpoint queries required")
    labels = np.array([[row[key] for key in ("body_x_m", "body_y_m", "body_yaw_rad")] for row in rows])
    embeddings = pm.normalize(features)
    hits1 = hits5 = eligible = distant = no_positive = 0
    for query in queries:
        if deadline is not None:
            pm.check_deadline(deadline)
        positive, negative = pm.pair_masks(labels[query], labels[gallery])
        supported = bool(positive.any())
        eligible += int(supported)
        rank = np.argsort(-(embeddings[gallery] @ embeddings[query]), kind="stable")[:5]
        hits1 += int(positive[rank[0]])
        hits5 += int(positive[rank].any())
        distant += int(negative[rank[0]])
        no_positive += int(not supported)
    count = len(queries)
    return {"gallery_count": len(gallery), "all_queries": count, "eligible_queries": eligible,
            "no_positive_queries": no_positive, "positive_support_coverage": eligible / count,
            "recall_at_1_all": hits1 / count, "recall_at_5_all": hits5 / count,
            "recall_at_1_eligible": hits1 / eligible if eligible else None,
            "recall_at_5_eligible": hits5 / eligible if eligible else None,
            "false_distant_top1": distant, "interpretation": "retrieval_only_shadow_ranking"}


def Embedder(torch, features):
    class _Embedder(torch.nn.Module):
        """Pretrained features, global average pool, L2. Step 0 equals the frozen baseline."""
        def __init__(self):
            super().__init__()
            self.features = features

        def train(self, mode=True):
            super().train(mode)
            # Small batches: keep pretrained BatchNorm statistics fixed, train affine terms.
            for module in self.modules():
                if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                    module.eval()
            return self

        def forward(self, pixels):
            pooled = self.features(pixels).mean(dim=(2, 3))
            return torch.nn.functional.normalize(pooled, dim=1)
    return _Embedder()


def resident_gib():
    """Process memory charge in GiB, or None when unavailable.

    Windows: private commit (PagefileUsage), the quantity a Job memory cap limits.
    Linux: resident set size (VmRSS). The status field keeps the name resident_gib.
    """
    try:
        if sys.platform == "win32":
            import ctypes
            from ctypes import wintypes

            class Counters(ctypes.Structure):
                _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
                    (name, ctypes.c_size_t) for name in (
                        "PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                        "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage",
                        "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")]
            counters = Counters()
            counters.cb = ctypes.sizeof(counters)
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel32.GetCurrentProcess.restype = wintypes.HANDLE
            query = kernel32.K32GetProcessMemoryInfo
            query.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
            query.restype = wintypes.BOOL
            if not query(kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
                return None
            return counters.PagefileUsage / 2**30
        with open("/proc/self/status", encoding="ascii") as status:
            for line in status:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 2**20
    except (OSError, ValueError, AttributeError):
        return None
    return None


def run(config: Config, resume=False, model_factory=None, alert=None):
    """One bounded job. Returns the sanitized final status (no metrics)."""
    started = time.monotonic()
    deadline = started + config.wall_seconds
    output = Path(config.output)
    checkpoints = output / "checkpoints"
    if resume:
        if not (checkpoints / "latest.pt").is_file():
            raise ValueError("resume requested but no durable latest checkpoint exists")
    else:
        output.mkdir(parents=True, exist_ok=False)
        checkpoints.mkdir()
        atomic_json(output / "config.json", asdict(config))
    status = {"schema": STATUS_SCHEMA, "state": "starting", "pid": os.getpid(),
              "source_sha256": source_sha(), "last_durable_update": None, "updates_done": None,
              "max_updates": config.max_updates, "started_unix": time.time(), "message": ""}

    def publish(**changes):
        status.update(changes, updated_unix=time.time(), wall_s=round(time.monotonic() - started, 1),
                      resident_gib=resident_gib())
        atomic_json(output / "status.json", status)

    publish()
    try:
        trainer = Trainer(config, model_factory)
        torch = trainer.torch
        if resume:
            step, rng, best = trainer.restore(checkpoints / "latest.pt")
        else:
            step, rng, best = 0, np.random.default_rng(config.seed), None
            torch.manual_seed(config.seed)
        publish(state="running", last_durable_update=step if resume else None, updates_done=step)
        metrics_log = output / "metrics.jsonl"

        def durable(current):
            trainer.save(checkpoints / "latest.pt", trainer.checkpoint(current, rng, best))
            publish(last_durable_update=current, updates_done=current)

        def validate_and_select(current):
            nonlocal best
            trainer.pending_validation = True
            durable(current)  # replay this exact model/RNG/best state after an interruption
            metrics = trainer.validate(deadline)
            completed = validation_record(current, metrics, model_state_sha256(torch, trainer.model.state_dict()),
                                          trainer.provenance)
            with metrics_log.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"step": current, "metrics": metrics}, sort_keys=True) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            trainer.validation, trainer.pending_validation = completed, False
            key = selection_key(completed)
            if best is None or tuple(key) > tuple(best["key"]):
                best = {"step": current, "key": key, "validation": completed}
                trainer.save(checkpoints / "best.pt", trainer.checkpoint(current, rng, best))
            durable(current)  # refresh same-step selection state

        def check_hard_cap():
            # The in-memory update is discarded; latest.pt stays the last coherent state.
            memory = resident_gib()
            if memory is not None and memory >= config.hard_ram_gib:
                raise MemoryError("hard RAM cap exceeded; last durable checkpoint kept")

        if trainer.pending_validation or (step == 0 and best is None):
            validate_and_select(step)
            check_hard_cap()  # baseline or replay of the exact interrupted validation state
        while step < config.max_updates:
            if time.monotonic() >= deadline:
                durable(step)
                publish(state="wall_cap", message="wall budget reached; resume from last durable update")
                return finish(status, alert)
            memory = resident_gib()
            if memory is not None and memory >= config.soft_ram_gib:
                durable(step)
                publish(state="paused_memory", message="soft RAM threshold reached after a durable save")
                return finish(status, alert)
            trainer.update(rng, step + 1)
            step += 1
            check_hard_cap()  # fresh sample: the pre-update one is below the soft threshold
            if step % config.validate_every == 0 or step == config.max_updates:
                validate_and_select(step)
                check_hard_cap()
            elif step % 50 == 0:
                durable(step)
        publish(state="completed", message="max updates reached",
                selected_step=None if best is None else best["step"])
        return finish(status, alert)
    except Exception as error:
        publish(state="failed", message=f"{type(error).__name__}: {error}"[:500])
        finish(status, alert)
        raise


def finish(status, alert):
    if alert is not None:
        try:
            alert(status)
        except Exception:
            pass
    return dict(status)


MODEL_SCHEMA = "schubki-place-finetune-model-v1"
MODEL_FILE = "place_finetune_v1.pt"


def export_inference(checkpoint_path, output_dir, base_weights_sha256, source_commit):
    """Network parameters only: no optimizer, RNG, metrics, images, poses or labels."""
    import io
    import torch
    pm.sha(base_weights_sha256)
    if (not isinstance(source_commit, str) or len(source_commit) != 40
            or any(character not in "0123456789abcdef" for character in source_commit)):
        raise ValueError("full source commit required")
    checkpoint_sha = file_sha(checkpoint_path)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if file_sha(checkpoint_path) != checkpoint_sha:
        raise ValueError("checkpoint changed during export")
    provenance = checkpoint_metadata(payload)
    if base_weights_sha256 != provenance["base_weights_sha256"]:
        raise ValueError("export base weights differ from checkpoint provenance")
    if source_commit != provenance["source"]["checkout_commit"]:
        raise ValueError("export source commit differs from checkpoint provenance")
    validation, best = payload["validation"], payload["best"]
    if (payload["pending_validation"] or validation is None or best is None
            or validation["step"] != payload["step"] or best["step"] != payload["step"]
            or best["validation"] != validation):
        raise ValueError("export requires a completed, selected best checkpoint")
    if validation["model_sha256"] != model_state_sha256(torch, payload["model"]):
        raise ValueError("export model differs from its completed validation")
    # Owning compact storage prevents tensor views serializing unrelated checkpoint bytes.
    state = {key: value.detach().to(device="cpu", dtype=torch.float32, copy=True,
                                   memory_format=torch.contiguous_format)
             for key, value in payload["model"].items() if key.startswith("features.")}
    if not state or len(state) != len(payload["model"]):
        raise ValueError("unexpected parameters outside the feature network")
    settings = payload["compatible"]
    backbone, input_width = settings["backbone"], settings["input_width"]
    output = pm.new_output(output_dir)
    buffer = io.BytesIO()
    torch.save(state, buffer)
    atomic_bytes(output / MODEL_FILE, buffer.getvalue())
    card = {"schema": MODEL_SCHEMA, "architecture": architecture(backbone, input_width),
            "backbone": backbone, "input_width": input_width, "embedding_dim": BACKBONES[backbone]["dim"],
            "augment": settings.get("augment", "basic"),
            "normalization": "l2", "preprocessing": encoder_preprocessing(input_width), "weights_file": MODEL_FILE,
            "weights_sha256": file_sha(output / MODEL_FILE), "weights_bytes": (output / MODEL_FILE).stat().st_size,
            "training_step": payload["step"], "base_weights_sha256": base_weights_sha256,
            "trainer_source_sha256": payload["source_sha256"], "source_commit": source_commit,
            "training_provenance": provenance, "checkpoint_sha256": checkpoint_sha,
            "selection": {"step": best["step"], "order": SELECTION_ORDER.copy(),
                          "validation_sha256": pm.digest(pm.canonical(validation)),
                          "validated_model_sha256": validation["model_sha256"],
                          "interpretation": "retrieval_only_shadow_ranking"},
            "contains": "network parameters only; no images, poses, labels, optimizer state or metrics",
            "training_scope": "laps 1-2 gradients, lap 3 checkpoint selection; laps 4-7 unused",
            "loader": "race_localization.place_finetune.FinetunedEncoder"}
    pm.write_json(output / "model.json", card)
    return {"weights_sha256": card["weights_sha256"], "weights_bytes": card["weights_bytes"],
            "training_step": card["training_step"]}


def encoder_preprocessing(input_width=CROP):
    return pm.PREPROCESSING if input_width == CROP else {**pm.PREPROCESSING, "center_crop": [CROP, input_width]}


class FinetunedEncoder:
    """Drop-in for place_model.FrozenEncoder.encode_rgb with exported fine-tuned weights.

    Loads only a local file whose SHA256 matches. 232 short-edge resize, then a
    224 x input_width center crop (224 x 224 is the frozen-baseline path).
    Returns one L2-normalized float32[embedding_dim] vector per image.
    """
    def __init__(self, weight_path, expected_sha256, device="cpu", threads=1,
                 backbone="mobilenet_v3_large", input_width=CROP):
        import io
        import torch
        from torchvision.transforms import InterpolationMode
        from torchvision.transforms import functional
        pm.integer(threads, 1, 64)
        pm.integer(input_width, CROP, 1024)
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable; no silent CPU fallback")
        weight_bytes = pm.read_pinned(weight_path, expected_sha256, 256 * 1024 * 1024)
        torch.set_num_threads(threads)
        network, features = build_features(torch, backbone)
        self.model = Embedder(torch, features(network))
        self.input_width, self.dim = input_width, BACKBONES[backbone]["dim"]
        self.model.load_state_dict(torch.load(io.BytesIO(weight_bytes), map_location="cpu", weights_only=True), strict=True)
        self.model.requires_grad_(False)
        self.model.eval()
        self.device = torch.device(device)
        self.model.to(self.device)
        self.torch, self.functional = torch, functional
        self.interpolation = InterpolationMode.BILINEAR
        self.mean = torch.tensor(pm.PREPROCESSING["mean"]).view(3, 1, 1)
        self.std = torch.tensor(pm.PREPROCESSING["std"]).view(3, 1, 1)
        self.identity = {"architecture": architecture(backbone, input_width), "weights_sha256": pm.sha(expected_sha256),
                         "preprocessing": encoder_preprocessing(input_width)}

    @classmethod
    def from_card(cls, model_dir, device="cpu", threads=1):
        """Load an exported model directory; v1 cards without backbone/width use the v1 values."""
        card = json.loads((Path(model_dir) / "model.json").read_text(encoding="utf-8"))
        encoder = cls(Path(model_dir) / card["weights_file"], card["weights_sha256"], device, threads,
                      card.get("backbone", "mobilenet_v3_large"), card.get("input_width", CROP))
        if card["preprocessing"] != encoder.identity["preprocessing"]:
            raise ValueError("model card preprocessing differs from the encoder path")
        return encoder, card

    def tensor(self, rgb):
        if not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError("encoder takes RGB uint8 pixels only")
        pm.image_dimensions(rgb.shape[1], rgb.shape[0], 4 * 1024 * 1024)
        torch, functional = self.torch, self.functional
        tensor = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1)
        tensor = functional.resize(tensor, [PACK_EDGE], interpolation=self.interpolation, antialias=True)
        tensor = functional.center_crop(tensor, [CROP, self.input_width])
        tensor = functional.convert_image_dtype(tensor, torch.float32)
        return (tensor - self.mean) / self.std

    def encode_batch(self, images):
        torch = self.torch
        batch = torch.stack([self.tensor(rgb) for rgb in images]).to(self.device)
        with torch.inference_mode():
            features = self.model(batch).float().cpu().numpy()
        pm.validate_features(features, len(images), self.dim)
        return features

    def encode_rgb(self, rgb):
        return self.encode_batch([rgb])[0]

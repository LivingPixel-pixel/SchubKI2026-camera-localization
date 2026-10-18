"""Explicit local-only adapters for exported CNN and headless ViT place networks.

Card inspection is Torch-free. Only load_encoder constructs a network; every
production loader verifies bounded local weight bytes before restricted loading.
"""
from __future__ import annotations

import importlib.metadata
import io
from pathlib import Path, PurePosixPath, PureWindowsPath

import numpy as np

from . import place_model as pm

VIT_PREPROCESSING = "torchvision.ViT_B_16_Weights.IMAGENET1K_V1.transforms;PIL_RGB;no_augmentation"
SHARED_PREPROCESSING = "torchvision ViT_B_16_Weights.IMAGENET1K_V1.transforms; PIL RGB; no training augmentation at inference"
PUBLIC_CHECKPOINT = "schubki-full-encoder-checkpoint-v1"
CARD_NAMES = ("model.json", "model-card.json", "inference-best.card.json")


def runtime_versions():
    versions = {}
    for package in ("torch", "torchvision", "numpy", "Pillow", "opencv-python", "opencv-python-headless",
                    "opencv-contrib-python", "opencv-contrib-python-headless"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "absent"
    return versions


def read_card(model_dir):
    """Accept one unambiguous known card, or an explicitly named card file."""
    source = Path(model_dir).resolve()
    if source.is_dir():
        cards = [source / name for name in CARD_NAMES if (source / name).is_file()]
        if len(cards) != 1:
            raise ValueError("model directory needs exactly one known card; specify the card file explicitly")
        source = cards[0]
    pm.file_hash(source, 1024 * 1024)
    raw = source.read_bytes()
    if not 0 < len(raw) <= 1024 * 1024:
        raise ValueError("model card byte cap")
    card = pm.parse_json(raw)
    if not isinstance(card, dict):
        raise ValueError("model card must be an object")
    checkpoint = None
    if card.get("schema") == "schubki-place-finetune-model-v1":
        from .place_finetune import BACKBONES, architecture, encoder_preprocessing
        backbone, width = card.get("backbone", "mobilenet_v3_large"), card.get("input_width", 224)
        pm.integer(width, 224, 1024)
        if backbone not in BACKBONES:
            raise ValueError("unsupported CNN backbone")
        architecture_name, dimension = architecture(backbone, width), BACKBONES[backbone]["dim"]
        preprocessing = encoder_preprocessing(width)
        if (card.get("architecture") != architecture_name or card.get("embedding_dim") != dimension
                or card.get("preprocessing") != preprocessing or card.get("normalization") != "l2"):
            raise ValueError("CNN card architecture/dimension/preprocessing mismatch")
        kind, name, pin, size = "cnn", card["weights_file"], card["weights_sha256"], card["weights_bytes"]
        options = {"backbone": backbone, "input_width": width}
    elif card.get("schema") == "schubki-vit-weights-card-v1":
        if (card.get("architecture") != "vit_b_16" or card.get("embedding_dim") != 768
                or card.get("preprocessing") != SHARED_PREPROCESSING or card.get("origin_step") != 1125
                or card.get("checkpoint_schema", PUBLIC_CHECKPOINT) != PUBLIC_CHECKPOINT):
            raise ValueError("shared ViT card architecture/preprocessing/origin mismatch")
        kind, name, pin, size = "vit", card["weights_file"], card["weights_sha256"], card["weights_bytes"]
        checkpoint = {"schema": PUBLIC_CHECKPOINT, "step": 1125, "training_identity": card["training_identity"]}
        architecture_name, dimension, preprocessing, options = "vit_b_16", 768, VIT_PREPROCESSING, {}
    elif source.name == "inference-best.card.json" and "schema" not in card:
        required = {"architecture", "dimension", "preprocessing", "weights", "origin_step", "new_experiment_step",
                    "training_identity", "experiment_identity", "contains"}
        pm.exact(card, required)
        if (card["architecture"] != "vit_b_16" or card["dimension"] != 768
                or card["preprocessing"] != VIT_PREPROCESSING or card["origin_step"] != 1125):
            raise ValueError("inference-best ViT card architecture/preprocessing mismatch")
        step = pm.integer(card["new_experiment_step"], 0, 12000)
        pm.exact(card["weights"], {"path", "sha256", "bytes"})
        provenance = card["weights"]["path"]
        if not isinstance(provenance, str):
            raise ValueError("inference-best provenance path must be a string")
        declared = PureWindowsPath(provenance) if "\\" in provenance or PureWindowsPath(provenance).drive else PurePosixPath(provenance)
        if not declared.is_absolute() or declared.name != "inference-best.pt" or ".." in declared.parts:
            raise ValueError("inference-best provenance must name the recognized artifact without traversal")
        # The old absolute path is provenance only. Load exactly the adjacent
        # recognized basename, never a card-selected external file.
        kind, name, pin, size = "vit", "inference-best.pt", card["weights"]["sha256"], card["weights"]["bytes"]
        checkpoint = {"schema": PUBLIC_CHECKPOINT, "step": step or 1125, "training_identity": card["training_identity"]}
        architecture_name, dimension, preprocessing, options = "vit_b_16", 768, VIT_PREPROCESSING, {}
    else:
        raise ValueError("unrecognized model card schema; no implicit encoder conversion")
    if not isinstance(name, str) or Path(name).name != name or "/" in name or "\\" in name or name in ("", ".", ".."):
        raise ValueError("model weights_file must be an adjacent basename")
    pm.sha(pin)
    pm.integer(size, 1, (512 if kind == "vit" else 256) * 1024**2)
    if type(dimension) is not int or "training_identity" not in card and kind == "vit":
        raise ValueError("model dimension/training identity missing")
    identity = {"schema": "schubki-place-encoder-v2", "kind": kind, "architecture": architecture_name,
                "embedding_dim": dimension, "preprocessing": preprocessing, "normalization": "l2",
                "pixel_decode": "PIL_RGB_stored_pixels_ignore_EXIF" if kind == "vit" else pm.PREPROCESSING["decode"],
                "weights_sha256": pin, "weights_bytes": size, "card_sha256": pm.digest(raw),
                "checkpoint": checkpoint, "dependencies": runtime_versions(),
                "adapter_sha256": pm.file_hash(__file__, 1024**2),
                "cnn_loader_sha256": pm.file_hash(Path(__file__).with_name("place_finetune.py"), 2 * 1024**2) if kind == "cnn" else None}
    return {"identity": identity, "weight_path": source.parent / name, "options": options, "card": card}


def read_weight_bytes(spec):
    identity = spec["identity"]
    cap = (512 if identity["kind"] == "vit" else 256) * 1024**2
    data = pm.read_pinned(spec["weight_path"], identity["weights_sha256"], cap)
    if len(data) != identity["weights_bytes"]:
        raise ValueError("model weight size mismatch")
    return data


class ViTEncoder:
    def __init__(self, spec, device, threads):
        # No pretrained weights or transform downloads occur here.
        import torch
        from torchvision.models import vit_b_16, ViT_B_16_Weights
        identity = spec["identity"]
        data = read_weight_bytes(spec)
        payload = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
        validate_vit_checkpoint(payload, identity["checkpoint"],
            lambda value: type(value) is torch.Tensor and value.device.type == "cpu" and bool(torch.isfinite(value).all()))
        pm.integer(threads, 1, 64)
        torch.set_num_threads(threads)
        if str(device).startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable; no CPU fallback")
        model = vit_b_16(weights=None)
        model.heads = torch.nn.Identity()
        model.load_state_dict(payload["model"], strict=True)
        model.requires_grad_(False)
        self.device, self.torch = torch.device(device), torch
        self.model = model.to(device=self.device, dtype=torch.float32).eval()
        self.preprocess = ViT_B_16_Weights.IMAGENET1K_V1.transforms()

    def encode_batch(self, images):
        from PIL import Image
        tensors = [self.preprocess(Image.fromarray(rgb, mode="RGB")) for rgb in images]
        with self.torch.inference_mode():
            result = self.model(self.torch.stack(tensors).to(self.device, dtype=self.torch.float32))
            result = result.float().cpu().numpy()
        return pm.normalize(result).astype(np.float32)


def validate_vit_checkpoint(payload, expected, tensor_validator):
    """Header checks are pure; fixtures inject a scalar validator, never Torch."""
    pm.exact(payload, {"schema", "step", "training_identity", "model"})
    if (type(payload) is not dict or type(payload["step"]) is not int
            or pm.canonical({key: payload[key] for key in ("schema", "step", "training_identity")}) != pm.canonical(expected)
            or type(payload["model"]) is not dict or not payload["model"]):
        raise ValueError("ViT four-key checkpoint identity mismatch")
    if any(not isinstance(key, str) or not tensor_validator(value) for key, value in payload["model"].items()):
        raise ValueError("invalid ViT network tensor")


class Encoder:
    def __init__(self, implementation, identity):
        self.implementation, self.identity = implementation, identity

    def encode_batch(self, images):
        for rgb in images:
            if not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
                raise ValueError("encoder takes RGB uint8 pixels only")
            pm.image_dimensions(rgb.shape[1], rgb.shape[0], 4 * 1024**2)
        result = np.asarray(self.implementation.encode_batch(images))
        pm.validate_features(result, len(images), self.identity["embedding_dim"])
        if result.dtype != np.float32 or not np.allclose(np.linalg.norm(result, axis=1), 1, atol=1e-5):
            raise ValueError("encoder must return normalized float32 embeddings")
        return result

    def encode_rgb(self, rgb):
        return self.encode_batch([rgb])[0]


def load_encoder(model_dir, device="cpu", threads=1, *, factory=None):
    spec = read_card(model_dir)
    if factory is not None:
        implementation = factory(spec, device, threads)
    elif spec["identity"]["kind"] == "vit":
        implementation = ViTEncoder(spec, device, threads)
    else:
        from .place_finetune import FinetunedEncoder
        identity = spec["identity"]
        if spec["weight_path"].stat().st_size != identity["weights_bytes"]:
            raise ValueError("CNN weight size mismatch")
        implementation = FinetunedEncoder(spec["weight_path"], identity["weights_sha256"], device, threads, **spec["options"])
    return Encoder(implementation, spec["identity"]), spec["identity"]

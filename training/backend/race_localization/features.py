"""Local feature adapters with explicit, verified pretrained CPU identities.

Optional dependencies are imported only when constructing a learned adapter.
No import, discovery, extraction or matching operation downloads anything.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import io
import json
import copy
from pathlib import Path

import cv2
import numpy as np

PRETRAINED_ROOT = Path(__file__).with_name("pretrained")
VERIFICATION_PATH = Path(__file__).resolve().parents[2] / "evidence" / "feature-install" / "smoke_verification.json"
XFEAT_REVISION = "e92685f57f8318b18725c5c8c0bd28c7fe188d9a"
BACKENDS = {"sift": "SIFT + ratio matching", "orb": "ORB + ratio matching",
            "xfeat_mnn": "Pretrained XFeat + mutual nearest neighbors",
            "xfeat_lighterglue": "Pretrained XFeat + pretrained LighterGlue"}
MNN_COSINE_THRESHOLD = 0.82


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stable_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _assets() -> dict:
    path = PRETRAINED_ROOT / "assets_manifest.json"
    if not path.is_file():
        raise ValueError("Pinned XFeat assets are absent; run tools/prepare_features.py explicitly")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("revision") != XFEAT_REVISION or manifest.get("license") != "Apache-2.0":
        raise ValueError("Pretrained asset revision or license identity changed")
    required = {"vendor/xfeat/model.py", "vendor/xfeat/interpolator.py", "vendor/xfeat/xfeat.py",
                "LICENSE-XFeat.txt", "weights/xfeat.pt", "weights/xfeat-lighterglue.pt"}
    if set(manifest.get("files", {})) != required:
        raise ValueError("Incomplete pinned feature asset manifest")
    for relative, item in manifest["files"].items():
        asset = PRETRAINED_ROOT / relative
        if not asset.is_file() or _hash_file(asset) != item["sha256"]:
            raise ValueError(f"Pretrained asset integrity check failed: {relative}")
    return manifest


def feature_identity(backend: str) -> dict:
    if backend not in BACKENDS:
        raise ValueError(f"Unknown feature backend {backend}")
    common = {"backend": backend, "adapter_version": 1, "original_pixel_coordinates": True,
              "mask_bottom_fraction": 0.06, "opencv_version": cv2.__version__, "numpy_version": np.__version__}
    if backend in {"sift", "orb"}:
        return {**common, "extractor": backend, "matcher": "opencv_knn_ratio_unique_train",
                "descriptor_dimension": 128 if backend == "sift" else 32,
                "descriptor_dtype": "float32" if backend == "sift" else "uint8", "weights": {}}
    manifest = _assets()
    weights = {"extractor": manifest["files"]["weights/xfeat.pt"]["sha256"]}
    if backend == "xfeat_lighterglue":
        weights["matcher"] = manifest["files"]["weights/xfeat-lighterglue.pt"]["sha256"]
    return {**common, "extractor": "XFeat sparse", "matcher": "mutual_nearest_neighbors" if backend == "xfeat_mnn" else "LighterGlue",
            "descriptor_dimension": 64, "descriptor_dtype": "float32", "revision": XFEAT_REVISION,
            "license": manifest["license"], "asset_manifest_sha256": _hash_file(PRETRAINED_ROOT / "assets_manifest.json"),
            "source_sha256": {name: item["sha256"] for name, item in manifest["files"].items() if name.endswith(".py")},
            "weights": weights, "device": "cpu", "precision": "float32", "torch_threads": 2,
            "torch_version": _version("torch"), "tqdm_version": _version("tqdm"),
            "kornia_version": _version("kornia") if backend == "xfeat_lighterglue" else None,
            "matching_settings": {"min_cosine_similarity": MNN_COSINE_THRESHOLD} if backend == "xfeat_mnn" else
                {"descriptor_dim": 96, "n_layers": 6, "num_heads": 1, "depth_confidence": -1,
                 "width_confidence": 0.95, "flash": False, "mixed_precision": False}}


def availability() -> list[dict]:
    """Fast discovery without importing Torch or constructing networks.

    A learned backend is advertised only after a real local smoke verification
    bound to its current source, weight and installed dependency identity.
    """
    result = []
    for backend, label in BACKENDS.items():
        identity, available, reason = None, False, ""
        try:
            identity = feature_identity(backend)
            if backend in {"sift", "orb"}:
                available, reason = True, "OpenCV CPU backend is installed"
            else:
                required = ["torch", "tqdm"] + (["kornia"] if backend == "xfeat_lighterglue" else [])
                missing = [name for name in required if _version(name) is None]
                if missing:
                    reason = "Missing optional CPU dependencies: " + ", ".join(missing)
                else:
                    verification = VERIFICATION_PATH
                    records = json.loads(verification.read_text(encoding="utf-8")) if verification.is_file() else {}
                    record = records.get(backend, {})
                    available = record.get("identity_hash") == _stable_hash(identity) and record.get("status") == "passed"
                    reason = "Pinned CPU extraction and matching verified locally" if available else "Dependencies and pinned assets present; actual CPU smoke verification is pending"
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            reason = str(exc)
        result.append({"id": backend, "label": label, "available": available, "reason": reason, "identity": identity})
    return result


def _empty(dimension: int, dtype, width: int, height: int) -> dict:
    return {"keypoints": np.empty((0, 2), np.float32), "descriptors": np.empty((0, dimension), dtype),
            "scores": np.empty(0, np.float32), "image_size": np.asarray([width, height], np.int32)}


def subset_features(features: dict, indices: np.ndarray) -> dict:
    return {name: value[indices] if name in {"keypoints", "descriptors", "scores"} else value
            for name, value in features.items()}


class ClassicalAdapter:
    def __init__(self, config: dict):
        self.config = config
        self.backend = config["feature_backend"]
        self.identity = feature_identity(self.backend)
        self.detector = (cv2.SIFT_create(nfeatures=int(config["max_features"]), contrastThreshold=0.025)
                         if self.backend == "sift" else cv2.ORB_create(nfeatures=int(config["max_features"]),
                         scaleFactor=1.15, nlevels=8, edgeThreshold=19, fastThreshold=12))

    def extract(self, image: np.ndarray) -> dict:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        mask = np.full(gray.shape, 255, np.uint8)
        mask[int(gray.shape[0] * 0.94):] = 0
        keypoints, descriptors = self.detector.detectAndCompute(gray, mask)
        if descriptors is None:
            return _empty(self.detector.descriptorSize(), np.uint8 if self.backend == "orb" else np.float32,
                          gray.shape[1], gray.shape[0])
        return {"keypoints": np.asarray([keypoint.pt for keypoint in keypoints], np.float32).reshape(-1, 2),
                "descriptors": descriptors, "scores": np.asarray([keypoint.response for keypoint in keypoints], np.float32),
                "image_size": np.asarray([gray.shape[1], gray.shape[0]], np.int32)}

    def match(self, first: dict, second: dict) -> list[tuple[int, int, float]]:
        if len(first["descriptors"]) < 2 or len(second["descriptors"]) < 2:
            return []
        norm = cv2.NORM_HAMMING if self.backend == "orb" else cv2.NORM_L2
        pairs = cv2.BFMatcher(norm).knnMatch(first["descriptors"], second["descriptors"], k=2)
        kept, used_train = [], set()
        for neighbors in sorted(pairs, key=lambda pair: pair[0].distance if pair else float("inf")):
            if len(neighbors) != 2:
                continue
            best, runner_up = neighbors
            if best.distance < float(self.config["ratio_test"]) * runner_up.distance and best.trainIdx not in used_train:
                kept.append((best.queryIdx, best.trainIdx, float(best.distance)))
                used_train.add(best.trainIdx)
        return kept


def strict_checkpoint_load(module, state: dict) -> None:
    """Reject incomplete checkpoints instead of retaining random parameters."""
    if not isinstance(state, dict) or not state:
        raise ValueError("Checkpoint must contain a non-empty tensor state dictionary")
    expected = set(module.state_dict())
    missing, unexpected = sorted(expected - set(state)), sorted(set(state) - expected)
    if missing or unexpected:
        raise ValueError(f"Pretrained checkpoint strict loading failed: missing={missing}, unexpected={unexpected}")
    try:
        module.load_state_dict(state, strict=True)
    except (RuntimeError, TypeError) as exc:
        raise ValueError(f"Pretrained checkpoint strict loading failed: {exc}") from exc


class LearnedAdapter:
    def __init__(self, config: dict):
        self.config = config
        self.backend = config["feature_backend"]
        self.identity = feature_identity(self.backend)
        required = ["torch", "tqdm"] + (["kornia"] if self.backend == "xfeat_lighterglue" else [])
        missing = [name for name in required if _version(name) is None]
        if missing:
            raise ValueError("Missing optional CPU dependencies: " + ", ".join(missing))
        import torch
        from .pretrained.vendor.xfeat.xfeat import XFeat
        self.torch = torch
        torch.set_num_threads(2)
        # Passing an already safely loaded dictionary avoids the upstream path
        # loader. This adapter never invokes torch.hub or any demo helper.
        state = self._checkpoint("xfeat.pt", self.identity["weights"]["extractor"])
        self.extractor = XFeat(weights=state, top_k=int(config["max_features"]))
        strict_checkpoint_load(self.extractor.net, state)
        self.extractor.dev = torch.device("cpu")
        self.extractor.net.to("cpu").eval()
        self.matcher = None
        if self.backend == "xfeat_lighterglue":
            from kornia.feature.lightglue import LightGlue
            matcher_config = {"input_dim": 64, "descriptor_dim": 96, "add_scale_ori": False,
                              "add_laf": False, "scale_coef": 1.0, "n_layers": 6, "num_heads": 1,
                              "flash": False, "mp": False, "depth_confidence": -1, "width_confidence": 0.95,
                              "filter_threshold": float(config.get("learned_match_threshold", 0.1)), "weights": None}
            self.matcher = LightGlue(features=None, **matcher_config).to("cpu").eval()
            state = self._checkpoint("xfeat-lighterglue.pt", self.identity["weights"]["matcher"])
            # The official checkpoint exports the whole training pipeline.
            # Its extractor parameters equal xfeat.pt, while BatchNorm running
            # statistics differ. Validate that namespace completely, then keep
            # the published inference extractor, matching the official wrapper.
            extractor_state = {name.removeprefix("extractor.model.net."): tensor for name, tensor in state.items()
                               if name.startswith("extractor.model.net.")}
            strict_checkpoint_load(copy.deepcopy(self.extractor.net), extractor_state)
            if any(not torch.equal(parameter.detach(), extractor_state[name])
                   for name, parameter in self.extractor.net.named_parameters()):
                raise ValueError("LighterGlue checkpoint extractor parameters differ from the pinned XFeat model")
            renamed = {}
            for name, tensor in state.items():
                if name.startswith("extractor.model.net."):
                    continue
                if not name.startswith("matcher."):
                    raise ValueError(f"Unexpected LighterGlue checkpoint namespace: {name}")
                name = name.removeprefix("matcher.")
                for index in range(6):
                    name = name.replace(f"self_attn.{index}", f"transformers.{index}.self_attn")
                    name = name.replace(f"cross_attn.{index}", f"transformers.{index}.cross_attn")
                renamed[name] = tensor
            # Kornia registers this deterministic, non-learned configuration
            # buffer. The official checkpoint predates its serialization.
            if "confidence_thresholds" not in renamed:
                renamed["confidence_thresholds"] = self.matcher.confidence_thresholds.detach().clone()
            strict_checkpoint_load(self.matcher, renamed)

    def _checkpoint(self, filename: str, expected_sha256: str) -> dict:
        # Hash and deserialize exactly the same immutable byte buffer. A file
        # replaced between discovery and loading cannot acquire the old identity.
        content = (PRETRAINED_ROOT / "weights" / filename).read_bytes()
        if hashlib.sha256(content).hexdigest() != expected_sha256:
            raise ValueError(f"Pretrained checkpoint changed before loading: {filename}")
        state = self.torch.load(io.BytesIO(content), map_location="cpu", weights_only=True)
        if not isinstance(state, dict) or any(not self.torch.is_tensor(value) for value in state.values()):
            raise ValueError("Pretrained checkpoint must contain tensor values only")
        return state

    def extract(self, image: np.ndarray) -> dict:
        height, width = image.shape[:2]
        if min(height, width) < 32:
            raise ValueError("XFeat requires image dimensions of at least 32 pixels")
        if float(np.std(image)) <= 2.0:
            return _empty(64, np.float32, width, height)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image.ndim == 3 else np.repeat(image[..., None], 3, axis=2)
        tensor = self.torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1)[None].float() / 255.0
        with self.torch.inference_mode():
            output = self.extractor.detectAndCompute(tensor, top_k=int(self.config["max_features"]))[0]
        points = output["keypoints"].cpu().numpy().astype(np.float32)
        descriptors = output["descriptors"].cpu().numpy().astype(np.float32)
        scores = output["scores"].cpu().numpy().astype(np.float32)
        # The official extractor already restores coordinates after its /32
        # resize. Keep all geometry and LightGlue sizes in original pixels.
        valid = (np.isfinite(points).all(axis=1) & np.isfinite(descriptors).all(axis=1) & np.isfinite(scores) &
                 (points[:, 0] >= 0) & (points[:, 0] < width) & (points[:, 1] >= 0) & (points[:, 1] < int(height * 0.94)))
        return {"keypoints": points[valid], "descriptors": descriptors[valid], "scores": scores[valid],
                "image_size": np.asarray([width, height], np.int32)}

    def match(self, first: dict, second: dict) -> list[tuple[int, int, float]]:
        if not len(first["keypoints"]) or not len(second["keypoints"]):
            return []
        torch = self.torch
        with torch.inference_mode():
            if self.matcher is None:
                descriptors_a = torch.from_numpy(np.ascontiguousarray(first["descriptors"]))
                descriptors_b = torch.from_numpy(np.ascontiguousarray(second["descriptors"]))
                similarity = descriptors_a @ descriptors_b.T
                scores, destinations = similarity.max(dim=1)
                sources = similarity.argmax(dim=0)
                query = torch.arange(len(descriptors_a))
                valid = (sources[destinations] == query) & (scores >= MNN_COSINE_THRESHOLD)
                return [(int(a), int(b), float(1.0 - score)) for a, b, score in
                        zip(query[valid].tolist(), destinations[valid].tolist(), scores[valid].tolist())]
            data = {}
            for key, features in (("image0", first), ("image1", second)):
                data[key] = {name: torch.from_numpy(np.ascontiguousarray(features[name])).float()[None]
                             for name in ("keypoints", "descriptors", "image_size")}
            result = self.matcher(data)
            indices = result["matches"][0].cpu().numpy()
            scores = result["scores"][0].cpu().numpy()
            return [(int(pair[0]), int(pair[1]), float(1.0 - score)) for pair, score in zip(indices, scores)]


def create_feature_adapter(config: dict):
    backend = config.get("feature_backend", "sift")
    if backend not in BACKENDS:
        raise ValueError(f"Unknown feature backend {backend}")
    return ClassicalAdapter(config) if backend in {"sift", "orb"} else LearnedAdapter(config)


def verify_pretrained(backends=("xfeat_mnn", "xfeat_lighterglue")) -> dict:
    """Explicitly run local official extraction/matching and record readiness."""
    rng = np.random.default_rng(9147)
    image = rng.integers(0, 256, (240, 320, 3), np.uint8)
    for index in range(24):
        cv2.circle(image, (20 + index * 11 % 280, 20 + index * 29 % 200), 7, (255, 255, 255), 2)
    destination = VERIFICATION_PATH
    records = json.loads(destination.read_text(encoding="utf-8")) if destination.is_file() else {}
    for backend in backends:
        adapter = create_feature_adapter({"feature_backend": backend, "max_features": 512, "ratio_test": 0.75,
                                          "learned_match_threshold": 0.1})
        features = adapter.extract(image)
        matches = adapter.match(features, features)
        if len(features["keypoints"]) < 16 or len(matches) < 16:
            raise ValueError(f"Actual pinned CPU smoke verification failed for {backend}")
        records[backend] = {"identity_hash": _stable_hash(adapter.identity), "status": "passed",
                            "features": len(features["keypoints"]), "matches": len(matches),
                            "checks": ["Official pretrained state dictionaries strictly loaded", "CPU extraction", "Actual feature self matching"]}
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    return records

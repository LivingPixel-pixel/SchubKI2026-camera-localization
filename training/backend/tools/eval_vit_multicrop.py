"""Wide-view ViT place retrieval: 5 horizontal crops plus shift-aligned matching.

Frozen ViT-B/16 inference-only weights (for example the step-7750 release). No
training, no downloads. RGB is the only network input. Labels are used only by
the lap-3 retrieval metric, which reuses vit_finetune.retrieval_metrics rules
(pair_masks 5m/15deg positives, >=25m distant errors, gallery laps 1-2).

extract:  cache per-crop CLS embeddings for every prepared observation.
evaluate: lap-3 R1/R5/distant for centre crop, crop mean and shift-aligned matching.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"backend"))

from race_localization import place_model as core  # noqa: E402

CROPS = 5
CENTER = 2
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def crop_tensor(image):
    """Resize(256) like ViT_B_16 IMAGENET1K_V1, then 5 crops spanning the full width.

    The middle crop equals torchvision's CenterCrop(224), so slot 2 reproduces the
    single-crop descriptor exactly.
    """
    import torch
    from torchvision.transforms import InterpolationMode
    from torchvision.transforms import functional as F
    image = F.resize(image, 256, interpolation=InterpolationMode.BILINEAR, antialias=True)
    t = F.normalize(F.to_tensor(image), MEAN, STD)
    h, w = t.shape[1:]
    y, center = int(round((h-224)/2.0)), int(round((w-224)/2.0))
    xs = [0, center//2, center, center+(w-224-center)//2, w-224]
    return torch.stack([t[:, y:y+224, x:x+224] for x in xs])


def encode_cls(model, x):
    import torch
    n = x.shape[0]
    x = model._process_input(x)
    x = torch.cat([model.class_token.expand(n, -1, -1), x], 1)
    return model.encoder(x)[:, 0]


def load_model(weights):
    import torch
    from torchvision.models import vit_b_16
    model = vit_b_16(weights=None)
    model.heads = torch.nn.Identity()
    state = torch.load(weights, map_location="cpu", weights_only=True)
    model.load_state_dict(state["model"] if "model" in state else state, strict=True)
    return model.eval().cuda()


class Frames:
    """Module-level so Windows DataLoader workers can pickle it."""

    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        from PIL import Image
        with Image.open(self.paths[i]) as image:
            return crop_tensor(image.convert("RGB"))


def extract(args):
    import numpy as np
    import torch

    _, observations, rows = core.load_dataset(args.dataset_dir, args.dataset_sha256, args.cache_root)
    paths = [core.local_path(args.cache_root, o["image_path"]) for o in observations]
    model = load_model(args.weights)
    loader = torch.utils.data.DataLoader(Frames(paths), batch_size=16, num_workers=args.workers, pin_memory=True)
    parts, started = [], time.time()
    with torch.no_grad():
        for b, crops in enumerate(loader):
            n = crops.shape[0]
            cls = encode_cls(model, crops.reshape(-1, 3, 224, 224).cuda(non_blocking=True))
            parts.append(cls.reshape(n, CROPS, -1).cpu().numpy())
            if b % 50 == 0:
                print(f"batch {b} elapsed {time.time()-started:.1f}s", flush=True)
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    np.save(work/"cls.npy", np.concatenate(parts).astype(np.float32))
    print(f"saved {work/'cls.npy'} in {time.time()-started:.1f}s", flush=True)


def aligned_similarity(torch, q, g, max_shift):
    """Max over horizontal shifts s of mean_k cos(q[k], g[k+s]); q,g are [N,5,D] unit vectors."""
    pair = torch.einsum("qkd,gld->qgkl", q, g)
    best = None
    for s in range(-max_shift, max_shift+1):
        ks = [k for k in range(CROPS) if 0 <= k+s < CROPS]
        value = torch.stack([pair[:, :, k, k+s] for k in ks], -1).mean(-1)
        best = value if best is None else torch.maximum(best, value)
    return best


def metrics_from_similarity(np, similarity, rows, gallery, queries):
    labels = np.array([[row[k] for k in ("body_x_m", "body_y_m", "body_yaw_rad")] for row in rows])
    hits1 = hits5 = distant = 0
    for qi, query in enumerate(queries):
        positive, negative = core.pair_masks(labels[query], labels[gallery])
        rank = np.argsort(-similarity[qi], kind="stable")[:5]
        hits1 += int(positive[rank[0]])
        hits5 += int(positive[rank].any())
        distant += int(negative[rank[0]])
    return {"all_queries": len(queries), "gallery_count": len(gallery), "recall1_hits": hits1,
            "recall5_hits": hits5, "false_distant_top1": distant}


def evaluate(args):
    import numpy as np
    import torch
    _, _, rows = core.load_dataset(args.dataset_dir, args.dataset_sha256, args.cache_root)
    cls = np.load(Path(args.work)/"cls.npy")
    if len(cls) != len(rows):
        raise ValueError("feature cache does not match the prepared dataset")
    laps = np.array([row["lap_id"] for row in rows])
    gallery, queries = np.where(np.isin(laps, (1, 2)))[0], np.where(laps == 3)[0]
    c = torch.nn.functional.normalize(torch.tensor(cls, device="cuda"), dim=-1)
    q, g = c[queries], c[gallery]
    mean = torch.nn.functional.normalize(c.mean(1), dim=1)
    variants = {
        "center_crop": q[:, CENTER] @ g[:, CENTER].T,
        "crop_mean": mean[queries] @ mean[gallery].T,
        **{f"aligned_shift{s}": aligned_similarity(torch, q, g, s) for s in (1, 2)},
    }
    result = {}
    for name, similarity in variants.items():
        result[name] = metrics_from_similarity(np, similarity.cpu().numpy(), rows, gallery, queries)
        m = result[name]
        print(f"{name}: R1 {m['recall1_hits']} R5 {m['recall5_hits']} distant {m['false_distant_top1']}", flush=True)
    (Path(args.work)/"multicrop_lap3_metrics.json").write_text(json.dumps(result, indent=1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("extract", "evaluate"):
        p = sub.add_parser(name)
        p.add_argument("--dataset-dir", required=True)
        p.add_argument("--dataset-sha256", required=True)
        p.add_argument("--cache-root", required=True)
        p.add_argument("--work", required=True, help="local private output directory, never inside Git")
        if name == "extract":
            p.add_argument("--weights", required=True, help="inference-only ViT-B/16 checkpoint")
            p.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    extract(args) if args.command == "extract" else evaluate(args)


if __name__ == "__main__":
    main()

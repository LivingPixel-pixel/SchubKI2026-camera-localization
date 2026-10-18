"""Causal camera-only localizer (submission path).

ViT-B/16 place descriptors (fine-tuned parent step 7750) + causal retrieval filter.
Per frame t uses only frames <= t. Gallery: training laps 1-5 (front-left camera).

  python localizer/kiro_localize.py validate          # laps 6-7, official scorer
  python localizer/kiro_localize.py test --mcap TEST.mcap [--initial-pose x,y,yaw] --output poses.csv
"""
from __future__ import annotations
import argparse, io, json, math, os, platform, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DEFAULTS = {
    "weights": os.environ.get("SCHUBKI_WEIGHTS", str(ROOT / "models/vit7750/inference-best.pt")),
    "gallery": os.environ.get("SCHUBKI_GALLERY", str(ROOT / "models/vit7750/gallery-laps1-5.npz")),
}
TOPIC = "/constructor0/sensor/camera_fl/compressed_image"


def device():
    import torch
    try:
        import torch_directml
        return torch_directml.device(0)
    except Exception:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(weights):
    import torch
    from torchvision.models import vit_b_16, ViT_B_16_Weights
    torch.backends.mha.set_fastpath_enabled(False)
    model = vit_b_16(weights=None)
    model.heads = torch.nn.Identity()
    for layer in model.encoder.layers:  # need_weights=True avoids the CPU-only fused MHA op on DirectML
        original = layer.self_attention.forward
        def forward(*args, _original=original, **kwargs):
            kwargs['need_weights'] = True
            return _original(*args, **kwargs)
        layer.self_attention.forward = forward
    payload = torch.load(weights, map_location="cpu", weights_only=True)
    model.load_state_dict(payload["model"] if "model" in payload else payload, strict=True)
    model.eval()
    dev = device()
    model.to(dev)
    return model, ViT_B_16_Weights.IMAGENET1K_V1.transforms(), dev


def encode(model, tf, dev, images, batch=12, log=None):
    """images: iterable of JPEG bytes. Returns L2-normalised float32 [N,768] and per-image seconds."""
    import torch
    from PIL import Image
    out, buf, times = [], [], []
    def flush():
        started = time.perf_counter()
        x = torch.stack([tf(Image.open(io.BytesIO(b)).convert("RGB")) for b in buf]).to(dev)
        with torch.no_grad():
            f = model(x).detach().cpu().numpy().astype(np.float32)
        times.append((time.perf_counter() - started) / len(buf))
        out.append(f); buf.clear()
        if log: log(sum(len(o) for o in out))
    for b in images:
        buf.append(b)
        if len(buf) == batch:
            flush()
    if buf:
        flush()
    f = np.concatenate(out)
    return f / np.linalg.norm(f, axis=1, keepdims=True), float(np.mean(times))


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class CausalFilter:
    """Retrieval + constant-velocity gating. State update uses only current and past frames."""
    def __init__(self, gallery_xyyaw, k=30, gate=25.0, alpha=0.75, neighbour=6.0, relocalize_after=8):
        self.g, self.k, self.gate, self.alpha, self.nb, self.reloc = gallery_xyyaw, k, gate, alpha, neighbour, relocalize_after
        self.p = None; self.v = np.zeros(2); self.yaw = None; self.t = None; self.misses = 0

    def init(self, x, y, yaw, t):
        self.p, self.yaw, self.t = np.array([x, y], float), yaw, t

    def step(self, sims, t):
        top = np.argpartition(-sims, self.k)[:self.k]
        top = top[np.argsort(-sims[top])]
        if self.p is None:  # no first pose: global top-1
            c = top[0]; self.init(self.g[c, 0], self.g[c, 1], self.g[c, 2], t)
            return self.p.copy(), self.yaw, "global"
        dt = max(1e-3, (t - self.t) * 1e-9)
        pred = self.p + self.v * dt
        d = np.hypot(self.g[top, 0] - pred[0], self.g[top, 1] - pred[1])
        ok = d < self.gate + np.linalg.norm(self.v) * dt
        mode = "match"
        if ok.any():
            cand = top[ok]; s = sims[cand] - 0.004 * d[ok]
            c = cand[int(np.argmax(s))]
            near = top[np.hypot(self.g[top, 0] - self.g[c, 0], self.g[top, 1] - self.g[c, 1]) < self.nb]
            w = np.exp((sims[near] - sims[near].max()) / 0.02)
            meas = (self.g[near, :2] * w[:, None]).sum(0) / w.sum()
            myaw = math.atan2((np.sin(self.g[near, 2]) * w).sum(), (np.cos(self.g[near, 2]) * w).sum())
            self.misses = 0
        elif self.misses >= self.reloc:
            c = top[0]; meas = self.g[c, :2].copy(); myaw = self.g[c, 2]; mode = "relocalize"; self.misses = 0
            self.v[:] = 0; pred = meas
        else:
            self.misses += 1
            new = pred
            self.t = t; self.p = new
            return new.copy(), self.yaw, "predict"
        new = self.alpha * meas + (1 - self.alpha) * pred
        nv = (new - self.p) / dt
        self.v = 0.6 * nv + 0.4 * self.v if mode == "match" else np.zeros(2)
        self.yaw = wrap(self.yaw + self.alpha * wrap(myaw - self.yaw)) if mode == "match" else myaw
        self.p, self.t = new, t
        return new.copy(), self.yaw, mode


def run_filter(gallery_feat, gallery_pose, query_feat, stamps, initial=None, **kw):
    f = CausalFilter(gallery_pose, **kw)
    if initial is not None:
        f.init(*initial, stamps[0])
    rows, modes, started = [], {}, time.perf_counter()
    for i, t in enumerate(stamps):
        sims = gallery_feat @ query_feat[i]
        p, yaw, mode = f.step(sims, t)
        if i == 0 and initial is not None:
            p, yaw = np.array(initial[:2]), initial[2]
        modes[mode] = modes.get(mode, 0) + 1
        rows.append((int(t), float(p[0]), float(p[1]), float(yaw)))
    return rows, modes, (time.perf_counter() - started) / max(1, len(stamps))


def write_csv(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as s:
        s.write("timestamp_ns,x,y,yaw\n")
        for t, x, y, yaw in rows:
            s.write(f"{t},{x:.4f},{y:.4f},{yaw:.6f}\n")


def hardware():
    return {"platform": platform.platform(), "processor": platform.processor(), "python": sys.version.split()[0]}


def mcap_images(path, topic=TOPIC):
    from mcap.reader import make_reader
    with open(path, "rb") as stream:
        for _schema, _channel, message in make_reader(stream).iter_messages(topics=[topic]):
            raw = message.data
            yield parse_compressed(raw)


def parse_compressed(raw):
    """Minimal CDR decode of sensor_msgs/CompressedImage: header.stamp, frame_id, format, data."""
    import struct
    o = 4
    sec, nsec = struct.unpack_from("<iI", raw, o); o += 8
    n = struct.unpack_from("<I", raw, o)[0]; o += 4 + n
    o = (o + 3) & ~3
    n = struct.unpack_from("<I", raw, o)[0]; o += 4 + n
    o = (o + 3) & ~3
    n = struct.unpack_from("<I", raw, o)[0]; o += 4
    return sec * 1_000_000_000 + nsec, raw[o:o + n]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=("test", "validate"))
    ap.add_argument("--mcap"); ap.add_argument("--initial-pose"); ap.add_argument("--output", default=str(ROOT / "results/test_poses.csv"))
    ap.add_argument("--weights", default=DEFAULTS["weights"]); ap.add_argument("--gallery", default=DEFAULTS["gallery"])
    ap.add_argument("--prepared"); ap.add_argument("--frames"); ap.add_argument("--gt"); ap.add_argument("--scorer")
    ap.add_argument("--query-cache")
    a = ap.parse_args(argv)
    g = np.load(a.gallery)
    gfeat, gpose = g["features"].astype(np.float32), g["poses"].astype(np.float64)
    gfeat /= np.linalg.norm(gfeat, axis=1, keepdims=True)
    model = None
    report = {"hardware": hardware(), "gallery_images": int(len(gfeat))}
    if a.mode == "test":
        cache = Path(a.query_cache) if a.query_cache else None
        dev = "cached"
        if cache and cache.exists():
            z = np.load(cache); qfeat, stamps, enc = z["features"], [int(t) for t in z["stamps"]], float(z["encode_seconds"])
        else:
            stamps, blobs = zip(*mcap_images(a.mcap))
            order = np.argsort(stamps); stamps = [int(stamps[i]) for i in order]; blobs = [blobs[i] for i in order]
            model, tf, dev = load_model(a.weights)
            qfeat, enc = encode(model, tf, dev, blobs, log=lambda n: print(f"encoded {n}/{len(blobs)}", flush=True) if n % 120 == 0 else None)
            if cache:
                np.savez(cache, features=qfeat, stamps=np.array(stamps, dtype=np.int64), encode_seconds=enc)
        init = tuple(map(float, a.initial_pose.split(","))) if a.initial_pose else None
        rows, modes, filt = run_filter(gfeat, gpose, qfeat, stamps, init)
        write_csv(a.output, rows)
        report.update(frames=len(rows), modes=modes, encode_ms_per_frame=enc * 1e3, filter_ms_per_frame=filt * 1e3,
                      initial_pose=init, device=str(dev))
        Path(a.output).with_suffix(".run.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))
        return 0
    # validate: laps 6/7 from the prepared training dataset (labels used ONLY for scoring and lap first pose)
    prepared = Path(a.prepared)
    obs = [json.loads(l) for l in open(prepared / "observations.jsonl")]
    lab = [json.loads(l) for l in open(prepared / "labels.jsonl")]
    results = {}
    cache = Path(a.query_cache) if a.query_cache else None
    if cache and cache.exists():
        z = np.load(cache); qall, enc = z["features"], float(z["encode_seconds"])
    else:
        idx = [i for i, l in enumerate(lab) if l["lap_id"] in (6, 7)]
        model, tf, dev = load_model(a.weights)
        blobs = (open(Path(a.frames) / obs[i]["image_path"].split("/")[-1], "rb").read() for i in idx)
        qall, enc = encode(model, tf, dev, blobs, log=lambda n: print(f"encoded {n}/{len(idx)}", flush=True) if n % 240 == 0 else None)
        if cache:
            np.savez(cache, features=qall, encode_seconds=enc)
    idx = [i for i, l in enumerate(lab) if l["lap_id"] in (6, 7)]
    for lap in (6, 7):
        sel = [k for k, i in enumerate(idx) if lab[i]["lap_id"] == lap]
        stamps = [obs[idx[k]]["exposure_timestamp_ns"] for k in sel]
        first = lab[idx[sel[0]]]
        rows, modes, filt = run_filter(gfeat, gpose, qall[sel], stamps, (first["body_x_m"], first["body_y_m"], first["body_yaw_rad"]))
        out = ROOT / f"results/validation_lap{lap}_poses.csv"
        write_csv(out, rows)
        results[lap] = {"frames": len(rows), "modes": modes, "filter_ms_per_frame": filt * 1e3, "csv": str(out.relative_to(ROOT))}
    report.update(validation=results, encode_ms_per_frame=enc * 1e3)
    (ROOT / "results/validation.run.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


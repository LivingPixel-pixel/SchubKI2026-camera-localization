import json, sys, numpy as np, subprocess
sys.path.insert(0, "localizer")
import track_odometry as to, kiro_localize as kl
P, GT, SC, MAP = sys.argv[1:5]
L = to.Line(to.racing_line(MAP))
g = np.load("models/vit7750/gallery-laps1-5.npz"); gf = g["features"] / np.linalg.norm(g["features"], axis=1, keepdims=True)
gs = L.project(g["poses"][:, :2]); np.save("cache/gallery_s.npy", gs)
obs = [json.loads(l) for l in open(P + "/observations.jsonl")]; lab = [json.loads(l) for l in open(P + "/labels.jsonl")]
idx = [i for i, l in enumerate(lab) if l["lap_id"] in (6, 7)]
q = np.load("cache/val67.npz")["features"]
gt = np.genfromtxt(GT, delimiter=",", names=True)
t = gt["timestamp_ns"].astype(float); x, y = gt["x"], gt["y"]; dt = np.gradient(t) * 1e-9
tw = np.c_[t, np.hypot(np.gradient(x) / dt, np.gradient(y) / dt), np.zeros_like(t), np.zeros_like(t)]
for lap in (6, 7):
    sel = [k for k, i in enumerate(idx) if lab[i]["lap_id"] == lap]
    st = [obs[idx[k]]["exposure_timestamp_ns"] for k in sel]
    tr = to.TrackOdometry(L, gs, tw); rows = []
    for k, ts in zip(sel, st):
        p, yw = tr.step(gf @ q[k], ts); rows.append((ts, p[0], p[1], yw))
    out = f"cache/to_lap{lap}.csv"; kl.write_csv(out, rows)
    r = subprocess.run([sys.executable, SC, f"cache/gt_lap{lap}.csv", out], capture_output=True, text=True).stdout
    print(lap, [l for l in r.splitlines() if "RMSE" in l or "within 1" in l])

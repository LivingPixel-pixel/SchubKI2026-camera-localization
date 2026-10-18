import json, sys, numpy as np, subprocess
sys.path.insert(0, "localizer")
import motion_tracker as mt, kiro_localize as kl
P = sys.argv[1]; GT = sys.argv[2]; SC = sys.argv[3]
g = np.load("models/vit7750/gallery-laps1-5.npz"); gf = g["features"] / np.linalg.norm(g["features"], axis=1, keepdims=True); gp = g["poses"]
obs = [json.loads(l) for l in open(P + "/observations.jsonl")]; lab = [json.loads(l) for l in open(P + "/labels.jsonl")]
idx = [i for i, l in enumerate(lab) if l["lap_id"] in (6, 7)]
q = np.load("cache/val67.npz")["features"]
gt = np.genfromtxt(GT, delimiter=",", names=True)
# twist proxy from GT differentiation (training run has no twist topic): body-frame vx, vy, yaw rate at 100 Hz-ish
t = gt["timestamp_ns"].astype(float); x, y, yaw = gt["x"], gt["y"], np.unwrap(gt["yaw"])
dt = np.gradient(t) * 1e-9; vxw, vyw, wz = np.gradient(x) / dt, np.gradient(y) / dt, np.gradient(yaw) / dt
c, s = np.cos(yaw), np.sin(yaw)
tw = np.c_[t, c * vxw + s * vyw, -s * vxw + c * vyw, wz]
for lap in (6, 7):
    sel = [k for k, i in enumerate(idx) if lab[i]["lap_id"] == lap]
    st = [obs[idx[k]]["exposure_timestamp_ns"] for k in sel]
    tr = mt.Tracker(gp, tw); rows = []
    for k, ts in zip(sel, st):
        p, yw, n = tr.step(gf @ q[k], ts); rows.append((ts, p[0], p[1], yw))
    out = f"cache/mt_lap{lap}.csv"; kl.write_csv(out, rows)
    r = subprocess.run([sys.executable, SC, f"cache/gt_lap{lap}.csv", out], capture_output=True, text=True).stdout
    print(lap, [l for l in r.splitlines() if "RMSE" in l or "within 1" in l])

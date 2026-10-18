import json, sys, time, numpy as np
sys.path.insert(0, "localizer")
import track_odometry as to, kiro_localize as kl
L = to.Line(to.racing_line(sys.argv[1]))
g = np.load("models/vit7750/gallery-laps1-5.npz"); gf = g["features"] / np.linalg.norm(g["features"], axis=1, keepdims=True)
gs = np.load("cache/gallery_s.npy")
z = np.load("cache/test.npz"); q, st = z["features"], [int(v) for v in z["stamps"]]
tw = np.load("cache/test_twist.npy"); tw = tw[np.argsort(tw[:, 0])]
tr = to.TrackOdometry(L, gs, tw); rows = []; t0 = time.perf_counter()
for k, ts in enumerate(st):
    p, yw = tr.step(gf @ q[k], ts); rows.append((ts, float(p[0]), float(p[1]), yw, tr.confidence))
kl.write_csv("results/test_poses_scorer_format.csv", [r[:4] for r in rows])
with open("results/test_poses.csv", "w", newline="") as f:
    f.write("timestamp,x,y,yaw,confidence,status\n")
    for t_, x_, y_, a_, c_ in rows:
        f.write(f"{t_},{x_:.4f},{y_:.4f},{a_:.6f},{min(1.0, c_ / 0.2):.3f},{'ok' if c_ >= 0.05 else 'lost'}\n")
import os; r = json.load(open("results/test_poses.run.json" if os.path.exists("results/test_poses.run.json") else "results/test_poses_retrieval_only.run.json")); r.update(method="ViT-B/16 retrieval votes start offset s0; car-state speed integrates arc length; pose on racing line (track_odometry.py)", tracker_ms_per_frame=(time.perf_counter()-t0)/len(st)*1e3, frames=len(rows))
json.dump(r, open("results/test_poses.run.json", "w"), indent=2); print(r["tracker_ms_per_frame"], rows[0], rows[-1])

"""Validation scoring (official evaluate.py) and trajectory plots on the track map."""
import json, subprocess, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
MAP = Path(sys.argv[1]); GT = Path(sys.argv[2]); SCORER = Path(sys.argv[3])

def borders():
    from mcap.reader import make_reader
    from mcap_ros2.decoder import DecoderFactory
    out = {}
    with open(MAP, "rb") as s:
        for sch, ch, m, msg in make_reader(s, decoder_factories=[DecoderFactory()]).iter_decoded_messages():
            if sch.name == "nav_msgs/msg/Path" and ch.topic not in out:
                out[ch.topic.split("/")[-1]] = np.array([[p.pose.position.x, p.pose.position.y] for p in msg.poses])
    return out

def load(path):
    d = np.genfromtxt(path, delimiter=",", names=True)
    return d

B = borders()
gt = load(GT)
def draw_map(ax):
    for k in ("track_left", "track_right"):
        ax.plot(*B[k].T, color="#9aa3ad", lw=0.8)
    for k in ("pitlane_left", "pitlane_right"):
        ax.plot(*B[k].T, color="#d0d5da", lw=0.6)
    ax.set_aspect("equal"); ax.axis("off")

scores = {}
for lap in (6, 7):
    sub = RES / f"validation_lap{lap}_poses.csv"
    if not sub.exists():
        continue
    p0 = load(sub); lines = open(GT).read().splitlines()
    keep = [lines[0]] + [l for l in lines[1:] if p0["timestamp_ns"][0] - 1e8 <= int(l.split(",")[0]) <= p0["timestamp_ns"][-1] + 1e8]
    crop = RES.parent / "cache" / f"gt_lap{lap}.csv"; crop.write_text("\n".join(keep) + "\n")
    r = subprocess.run([sys.executable, str(SCORER), str(crop), str(sub)], capture_output=True, text=True)
    (RES / f"validation_lap{lap}_score.txt").write_text(r.stdout + r.stderr)
    scores[lap] = r.stdout
    p = load(sub)
    tg = gt["timestamp_ns"]; i = np.clip(np.searchsorted(tg, p["timestamp_ns"]), 1, len(tg) - 1)
    gx, gy = gt["x"][i], gt["y"][i]
    err = np.hypot(p["x"] - gx, p["y"] - gy)
    m = (tg >= p["timestamp_ns"][0]) & (tg <= p["timestamp_ns"][-1])
    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
    draw_map(ax)
    ax.plot(gt["x"][m], gt["y"][m], color="black", lw=1.2, label="ground truth")
    sc = ax.scatter(p["x"], p["y"], c=np.clip(err, 0, 5), cmap="turbo", s=4, label="estimate (causal)")
    fig.colorbar(sc, ax=ax, shrink=0.6, label="position error [m] (clipped at 5)")
    ax.legend(loc="lower left"); ax.set_title(f"Validation lap {lap}: causal camera-only estimate vs ground truth")
    fig.savefig(RES / f"trajectory_lap{lap}.png", bbox_inches="tight"); plt.close(fig)
    np.save(RES.parent / "cache" / f"err_lap{lap}.npy", err)

test = RES / "test_poses.csv"
if test.exists():
    p = load(test)
    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
    draw_map(ax)
    ax.plot(p["x"], p["y"], color="#e4572e", lw=1.5, label="hidden test run: estimate")
    ax.legend(loc="lower left"); ax.set_title("Hidden test run (no ground truth): causal estimate on the track map")
    fig.savefig(RES / "trajectory_test.png", bbox_inches="tight"); plt.close(fig)
print(json.dumps(scores, indent=1))

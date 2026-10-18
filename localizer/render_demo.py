"""Render the jury demo: 7 onboard cameras + live causal estimate on the track map (local data, local output).

  python localizer/render_demo.py test TEST.mcap map.mcap results/test_poses.csv OUT_DIR
  python localizer/render_demo.py val  map.mcap gt.csv results/validation_lap7_poses_motion_tracker.csv OUT_DIR [title]
Frames are written as PNG; encode with ffmpeg. Camera images are partner data: keep camera renders local.
"""
import io, math, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parent))
import kiro_localize as kl
import track_odometry as to

BG, PANEL, FG, ACC, GT_C, MUTED = "#0b1220", "#121b2e", "#e6edf7", "#ff6b35", "#3ddc97", "#6b7a90"
CAMS = ["cl", "fl", "fr", "cr", "rl", "r", "rr"]
LABEL = {"cl": "centre-left", "fl": "front-left", "fr": "front-right", "cr": "centre-right", "rl": "rear-left", "r": "rear", "rr": "rear-right"}


def borders(map_mcap):
    from mcap.reader import make_reader
    from mcap_ros2.decoder import DecoderFactory
    out = {}
    with open(map_mcap, "rb") as s:
        for sch, ch, _m, msg in make_reader(s, decoder_factories=[DecoderFactory()]).iter_decoded_messages():
            if sch.name == "nav_msgs/msg/Path":
                out[ch.topic.split("/")[-1]] = np.array([[p.pose.position.x, p.pose.position.y] for p in msg.poses])
    return out


def draw_track(ax, B):
    ax.set_facecolor(PANEL)
    for k in ("track_left", "track_right"):
        ax.plot(*B[k].T, color="#33415c", lw=1.4, zorder=1)
    for k in ("pitlane_left", "pitlane_right"):
        ax.plot(*B[k].T, color="#222d42", lw=0.8, zorder=1)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("#1f2a40")


def car(ax, x, y, yaw, color, size=18):
    ax.annotate("", xy=(x + size * math.cos(yaw), y + size * math.sin(yaw)), xytext=(x, y),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=2.2, mutation_scale=16), zorder=6)
    ax.scatter([x], [y], s=60, color=color, edgecolor="white", linewidth=1.2, zorder=7)


def camera_frames(mcap, stamps):
    from mcap.reader import make_reader
    want = {f"/constructor0/sensor/camera_{c}/compressed_image": c for c in CAMS}
    per = {c: ([], []) for c in CAMS}
    with open(mcap, "rb") as s:
        for _a, ch, msg in make_reader(s).iter_messages(topics=list(want)):
            t, b = kl.parse_compressed(msg.data); per[want[ch.topic]][0].append(t); per[want[ch.topic]][1].append(b)
    idx = {}
    for c, (ts, bs) in per.items():
        ts = np.array(ts); o = np.argsort(ts); ts = ts[o]; bs = [bs[i] for i in o]
        j = np.clip(np.searchsorted(ts, stamps), 0, len(ts) - 1); idx[c] = (j, bs)
    return idx


def thumb(b, w=480):
    im = Image.open(io.BytesIO(b)).convert("RGB"); h = int(im.height * w / im.width)
    return np.asarray(im.resize((w, h), Image.BILINEAR))


def main():
    mode = sys.argv[1]
    if mode == "test":
        mcap, map_mcap, poses, out = sys.argv[2:6]; gt = None; title = "Hidden test run | daylight | causal camera-only estimate"
    else:
        map_mcap, gt_csv, poses, out = sys.argv[2:6]; mcap = None
        title = sys.argv[6] if len(sys.argv) > 6 else "Validation lap | estimate vs ground truth"
        g = np.genfromtxt(gt_csv, delimiter=",", names=True); gt = (g["timestamp_ns"].astype(float), g["x"], g["y"])
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    B = borders(map_mcap)
    p = np.genfromtxt(poses, delimiter=",", names=True)
    st, X, Y, YAW = p["timestamp_ns"].astype(np.int64), p["x"], p["y"], p["yaw"]
    cams = camera_frames(mcap, st) if mcap else None
    tw = None
    if mode == "test":
        try:
            tw = np.load(Path(poses).parents[1] / "cache/test_twist.npy")
        except Exception:
            tw = None
    err = None
    if gt is not None:
        i = np.clip(np.searchsorted(gt[0], st), 1, len(gt[0]) - 1); gx, gy = gt[1][i], gt[2][i]; err = np.hypot(X - gx, Y - gy)
    plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": FG})
    for k in range(len(st)):
        fig = plt.figure(figsize=(19.2, 10.8), dpi=100); fig.patch.set_facecolor(BG)
        fig.text(0.02, 0.955, "SchubKI 2026", color=ACC, fontsize=20, weight="bold")
        fig.text(0.14, 0.955, title, color=FG, fontsize=20)
        t = (st[k] - st[0]) * 1e-9
        fig.text(0.98, 0.955, f"t = {t:5.1f} s   frame {k + 1}/{len(st)}", color=MUTED, fontsize=16, ha="right")
        if cams:
            boxes = {"fl": [0.015, 0.47, 0.44, 0.44], "cl": [0.015, 0.25, 0.19, 0.18], "fr": [0.21, 0.25, 0.19, 0.18],
                     "cr": [0.405, 0.25, 0.19, 0.18], "rl": [0.015, 0.04, 0.19, 0.18], "r": [0.21, 0.04, 0.19, 0.18], "rr": [0.405, 0.04, 0.19, 0.18]}
            for name, box in boxes.items():
                ax = fig.add_axes(box)
                j, bs = cams[name]; ax.imshow(thumb(bs[j[k]], 960 if name == "fl" else 400), aspect="auto"); ax.set_xticks([]); ax.set_yticks([])
                for s in ax.spines.values():
                    s.set_color(ACC if name == "fl" else "#1f2a40"); s.set_linewidth(3 if name == "fl" else 1)
                ax.set_title(LABEL[name] + ("  (model input)" if name == "fl" else ""), color=ACC if name == "fl" else MUTED, fontsize=11, loc="left")
            hud = fig.add_axes([0.465, 0.47, 0.13, 0.44]); hud.set_facecolor(PANEL); hud.set_xticks([]); hud.set_yticks([])
            for s in hud.spines.values():
                s.set_color("#1f2a40")
            v = None
            if tw is not None:
                jj = min(np.searchsorted(tw[:, 0], st[k]), len(tw) - 1); v = math.hypot(tw[jj, 1], tw[jj, 2]) * 3.6
            lines = [("speed", f"{v:5.0f} km/h" if v is not None else "n/a"), ("x, y", f"{X[k]:7.1f}, {Y[k]:7.1f} m"),
                     ("heading", f"{math.degrees(YAW[k]):6.1f} deg"), ("runtime", "75 ms/frame"), ("gallery", "8,269 imgs")]
            for n, (a, b) in enumerate(lines):
                hud.text(0.06, 0.85 - n * 0.18, a, color=MUTED, fontsize=11, transform=hud.transAxes)
                hud.text(0.06, 0.77 - n * 0.18, b, color=FG, fontsize=14, weight="bold", transform=hud.transAxes)
            mapax = fig.add_axes([0.61, 0.05, 0.38, 0.86])
        else:
            mapax = fig.add_axes([0.03, 0.05, 0.55, 0.86])
            side = fig.add_axes([0.62, 0.09, 0.35, 0.36]); side.set_facecolor(PANEL)
            side.plot((st[:k + 1] - st[0]) * 1e-9, err[:k + 1], color=ACC, lw=1.4)
            side.set_xlim(0, (st[-1] - st[0]) * 1e-9); side.set_ylim(0, max(6, float(np.percentile(err, 99))))
            side.set_xlabel("time [s]", color=MUTED); side.set_ylabel("position error [m]", color=MUTED); side.tick_params(colors=MUTED)
            for s in side.spines.values():
                s.set_color("#1f2a40")
            e = err[:k + 1]
            stats = [("position RMSE so far", f"{math.sqrt(float(np.mean(e ** 2))):.2f} m"), ("current error", f"{e[-1]:.2f} m"),
                     ("within 1 m", f"{100 * float(np.mean(e < 1)):.0f} %")]
            for n, (a, b) in enumerate(stats):
                fig.text(0.62 + n * 0.125, 0.78, a, color=MUTED, fontsize=13); fig.text(0.62 + n * 0.125, 0.72, b, color=FG, fontsize=24, weight="bold")
            fig.text(0.62, 0.62, "Causal: each pose uses only images and speed up to its own timestamp.", color=MUTED, fontsize=13)
            fig.text(0.62, 0.58, "Ground truth is shown for scoring only, never used by the estimator.", color=MUTED, fontsize=13)
        draw_track(mapax, B)
        lo = max(0, k - 400)
        if gt is not None:
            m = (gt[0] >= st[0]) & (gt[0] <= st[k]); mapax.plot(gt[1][m], gt[2][m], color=GT_C, lw=2.2, alpha=0.9, zorder=3, label="ground truth")
        mapax.plot(X[lo:k + 1], Y[lo:k + 1], color=ACC, lw=2.6, zorder=4, label="estimate")
        car(mapax, X[k], Y[k], YAW[k], ACC)
        leg = mapax.legend(loc="lower right", facecolor=PANEL, edgecolor="#1f2a40", labelcolor=FG, fontsize=12)
        # inset zoom around the car
        z = mapax.inset_axes([0.02, 0.62, 0.36, 0.36]); draw_track(z, B)
        if gt is not None:
            z.plot(gt[1][m], gt[2][m], color=GT_C, lw=2.5)
        z.plot(X[lo:k + 1], Y[lo:k + 1], color=ACC, lw=2.5); car(z, X[k], Y[k], YAW[k], ACC, size=8)
        z.set_xlim(X[k] - 60, X[k] + 60); z.set_ylim(Y[k] - 60, Y[k] + 60)
        fig.savefig(out / f"f{k:05d}.png", facecolor=BG); plt.close(fig)
        if k % 100 == 0:
            print(k, flush=True)


if __name__ == "__main__":
    main()

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch
import matplotlib.image as mpimg
from pathlib import Path
R = Path("results"); OUT = Path("docs/submission"); OUT.mkdir(parents=True, exist_ok=True)
DARK, ACC, GREY = "#0f1b2d", "#e4572e", "#5b6b7f"

def slide(title, sub=None):
    fig = plt.figure(figsize=(13.33, 7.5), dpi=110); fig.patch.set_facecolor("white")
    fig.add_artist(plt.Rectangle((0, 0.9), 1, 0.1, color=DARK, transform=fig.transFigure))
    fig.text(0.04, 0.935, title, color="white", fontsize=26, weight="bold", va="center")
    if sub: fig.text(0.96, 0.935, sub, color="#c9d3df", fontsize=13, va="center", ha="right")
    fig.text(0.04, 0.025, "SchubKI 2026 | Constructor Labs: camera-based localization on a race track", color=GREY, fontsize=10)
    return fig

def bullets(fig, items, x=0.05, y=0.82, size=17, gap=0.075, w=None):
    for it in items:
        fig.text(x, y, "\u2022 " + it, fontsize=size, color=DARK, va="top", wrap=True); y -= gap
    return y

pages = []
# 1 title
f = slide("Camera-only localization at race pace", "1/6")
f.text(0.05, 0.75, "Learned place recognition + causal track-constrained odometry", fontsize=22, color=ACC, weight="bold")
bullets(f, ["Input: front camera images, track map, car-state speed. No GNSS, LiDAR, IMU or ground truth at inference.",
            "ViT-B/16 place descriptors fine-tuned on training laps 1-5 (8,269 images).",
            "Causal: the pose at frame t uses only data up to t. 10 Hz output, 100 % coverage.",
            "Validation laps 6-7 (official scorer): position RMSE 1.97-2.64 m, lateral RMSE 1.26-1.35 m.",
            "Real-time on a laptop iGPU: 75 ms/frame encode + <1 ms tracking (Ryzen AI 7 350, Radeon 860M)."], y=0.64, size=16, gap=0.085)
pages.append(f)
# 2 system diagram
f = slide("System", "2/6"); ax = f.add_axes([0.03, 0.08, 0.94, 0.78]); ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 6)
def box(x, y, w, h, t, c="#e8eef5"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05", fc=c, ec=DARK, lw=1.5)); ax.text(x + w/2, y + h/2, t, ha="center", va="center", fontsize=12.5, color=DARK, wrap=True)
def arrow(a, b): ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="->", lw=2, color=DARK))
box(0.1, 4.2, 2.0, 1.1, "Front camera\nJPEG, 10 Hz"); box(2.8, 4.2, 2.3, 1.1, "ViT-B/16 encoder\n(fine-tuned, 768-D)", "#fde3da")
box(5.8, 4.2, 2.0, 1.1, "Cosine top-k vs\n8,269-image gallery"); box(8.2, 4.2, 1.7, 1.1, "Vote start\noffset s0 (causal)", "#fde3da")
box(0.1, 1.6, 2.0, 1.1, "Car-state speed\n(twist, 100 Hz)"); box(2.8, 1.6, 2.3, 1.1, "Integrate arc length\ns(t) = s0 + \u222bv dt")
box(5.8, 1.6, 2.0, 1.1, "Racing line from\ntrack map (s \u2192 x, y, yaw)"); box(8.2, 1.6, 1.7, 1.1, "Pose CSV\nt, x, y, yaw", "#d9f2e3")
arrow((2.1, 4.75), (2.8, 4.75)); arrow((5.1, 4.75), (5.8, 4.75)); arrow((7.8, 4.75), (8.2, 4.75))
arrow((2.1, 2.15), (2.8, 2.15)); arrow((5.1, 2.15), (5.8, 2.15)); arrow((7.8, 2.15), (8.2, 2.15)); arrow((9.05, 4.2), (9.05, 2.7))
ax.text(0.1, 0.6, "Training (offline): triplet fine-tuning of the last two ViT blocks on laps 1-5, cross-lap positives <=5 m/15\u00b0, negatives >=25 m.\nGallery = training-lap descriptors with their poses. Laps 6-7 held out for validation.", fontsize=12, color=GREY)
pages.append(f)
# 3 results
f = slide("Validation results (official scorer, laps 6-7)", "3/6")
rows = [("Method", "Lap", "Pos RMSE", "Lateral", "Yaw deg", "<1 m"),
        ("Track odometry (submitted)", "6", "2.64 m", "1.35 m", "2.72", "19 %"), ("", "7", "1.97 m", "1.26 m", "2.29", "14 %"),
        ("Multi-hypothesis tracker", "6", "1.79 m", "0.63 m", "0.79", "69 %"), ("", "7", "1.83 m", "0.83 m", "1.02", "63 %"),
        ("Retrieval filter, no motion", "6", "2.45 m", "0.82 m", "1.07", "53 %"), ("", "7", "2.91 m", "0.84 m", "1.31", "56 %"),
        ("Earlier team baseline", "4-5", "15.0 m", "1.54 m", "1.37", "39 %")]
y = 0.83
for i, r in enumerate(rows):
    for x, c in zip((0.04, 0.27, 0.32, 0.40, 0.47, 0.54), r):
        f.text(x, y, c, fontsize=13, weight="bold" if i == 0 else None, color=DARK)
    y -= 0.06
f.text(0.04, y - 0.02, "Coverage 100 % (1 s bins) on both laps. Speed for laps 6-7 is a GT-derived\nproxy (training run has no twist topic); the hidden test uses real car-state twist.\nLaps 6-7 were seen in earlier baselines: reserved validation, not pristine.", fontsize=11.5, color=GREY, va="top")
ax = f.add_axes([0.6, 0.06, 0.39, 0.82]); ax.imshow(mpimg.imread(R / "trajectory_lap7.png")); ax.axis("off")
pages.append(f)
# 4 hidden test
f = slide("Hidden test run: daylight, up to 68 m/s", "4/6")
ax = f.add_axes([0.02, 0.06, 0.48, 0.82]); ax.imshow(mpimg.imread(R / "trajectory_test.png")); ax.axis("off")
bullets(f, ["739 front-camera frames, 74 s, one full lap (2,964 m integrated speed).",
            "Training data is night only. Daylight retrieval alone jumps between track sections.",
            "Fix: retrieval only votes the start offset; speed integration carries continuity; hysteresis prevents flips.",
            "One correction jump in the first seconds while votes accumulate; then a continuous lap.",
            "No first pose was used. 739/739 frames emitted (coverage 100 %)."], x=0.52, y=0.82, size=14, gap=0.12)
pages.append(f)
# 5 runtime
f = slide("Runtime and engineering", "5/6")
bullets(f, ["Laptop: AMD Ryzen AI 7 350, Radeon 860M iGPU (DirectML), 32 GB RAM, Windows 11.",
            "ViT-B/16 encode 75 ms/frame (batch 12) + tracker 0.6 ms/frame: about 13 Hz, faster than the 10 Hz camera.",
            "One run script: localizer/kiro_localize.py (encode + gallery) and localizer/track_odometry.py (causal pose).",
            "Fine-tuning pipeline with RAM/time guards, durable checkpoints and full held-out validation (backend/).",
            "Weights (343 MB) as a GitHub release asset; data stays local per partner licence."], size=16, gap=0.11)
pages.append(f)
# 6 failures / lessons
f = slide("Failure cases and lessons learned", "6/6")
bullets(f, ["Night-to-day domain shift is the main failure: daylight descriptors match wrong but similar straights.",
            "Repetitive long straights cause longitudinal error (1.1-2.3 m RMSE dominates position error).",
            "Track constraint pins lateral error to the racing-line offset; true lateral deviation is not observed.",
            "Planned, not done: lighting augmentation, lateral offset from lane borders, multi-camera descriptors.",
            "Lesson: a time budget must be checked before expensive steps; our last 1-hour training run spent its\nbudget on baseline encoding and pair mining and produced 0 new updates, so the submitted model is step 7750."], size=15, gap=0.12)
pages.append(f)
with PdfPages(OUT / "slides.pdf") as pdf:
    for i, f in enumerate(pages):
        pdf.savefig(f); f.savefig(OUT / f"slide{i+1}.png"); plt.close(f)
print("ok")


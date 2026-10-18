"""Rebuild the retrieval gallery from the raw training run (laps 1-5 only).

  python <dataset>/tools/extract_gt.py TRAIN_DIR gt.csv        # official starter-kit ground-truth export
  python localizer/build_gallery.py TRAIN.mcap gt.csv models/vit7750/gallery-laps1-5.npz
Front-left images in [lap1 start, lap5 end) are paired with the nearest gt.csv pose within 50 ms
(training labels only), then encoded with the fine-tuned ViT. Laps 6-7 are excluded.
"""
import os, sys, numpy as np
from mcap.reader import make_reader
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kiro_localize as kl
START, END = 1791918162834135962, 1791918989712705596   # laps 1-5 of c0_2026-10-13_22-53-42
LAP_STARTS = [1791918162834135962, 1791918444945299229, 1791918612521911091, 1791918736149655196, 1791918837410638975]
path, gt_csv, out = sys.argv[1:4]
g = np.genfromtxt(gt_csv, delimiter=",", names=True)
gt = np.c_[g["timestamp_ns"].astype(float), g["x"], g["y"], g["yaw"]]; gt = gt[np.argsort(gt[:, 0])]
stamps, blobs = [], []
with open(path, "rb") as s:
    for _a, _b, msg in make_reader(s).iter_messages(topics=[kl.TOPIC], start_time=START - 10**9, end_time=END + 10**9):
        t, b = kl.parse_compressed(msg.data)
        if START <= t < END:
            stamps.append(t); blobs.append(b)
order = np.argsort(stamps); stamps = np.array(stamps)[order]; blobs = [blobs[i] for i in order]
i = np.clip(np.searchsorted(gt[:, 0], stamps), 1, len(gt) - 1)
i = np.where(np.abs(gt[i - 1, 0] - stamps) < np.abs(gt[i, 0] - stamps), i - 1, i)
keep = np.abs(gt[i, 0] - stamps.astype(float)) <= 5e7
model, tf, dev = kl.load_model(kl.DEFAULTS["weights"])
feat, _ = kl.encode(model, tf, dev, [b for b, k in zip(blobs, keep) if k])
st = stamps[keep]
np.savez(out, features=feat, poses=gt[i[keep], 1:4], stamps=st, laps=np.searchsorted(LAP_STARTS, st, side="right"))
print(feat.shape)

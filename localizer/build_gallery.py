"""Rebuild the retrieval gallery from the raw training mcap (laps 1-5 only).

  python localizer/build_gallery.py TRAIN.mcap models/vit7750/gallery-laps1-5.npz
Front-left images in [lap1 start, lap5 end) are paired with the nearest /constructor0/state_estimation
pose within 50 ms (training labels only), then encoded with the fine-tuned ViT. Every 1st image is kept.
"""
import sys, numpy as np
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory
sys.path.insert(0, __import__("os").path.dirname(__file__))
import kiro_localize as kl
START, END = 1791918162834135962, 1791918989712705596   # laps 1-5 of c0_2026-10-13_22-53-42 (lap 6-7 held out)
LAP_STARTS = [1791918162834135962, 1791918444945299229, 1791918612521911091, 1791918736149655196, 1791918837410638975]
path, out = sys.argv[1], sys.argv[2]
gt = []
with open(path, "rb") as s:
    for _a, _b, _c, m in make_reader(s, decoder_factories=[DecoderFactory()]).iter_decoded_messages(topics=["/constructor0/state_estimation"]):
        st = m.header.stamp.sec * 10**9 + m.header.stamp.nanosec
        if START - 10**8 <= st <= END + 10**8:
            q = m.pose.pose.orientation if hasattr(m, "pose") and hasattr(m.pose, "pose") else None
            if q is None:
                continue
            yaw = np.arctan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
            gt.append((st, m.pose.pose.position.x, m.pose.pose.position.y, yaw))
gt = np.array(sorted(gt)); stamps, blobs = [], []
with open(path, "rb") as s:
    for _a, _b, msg in make_reader(s).iter_messages(topics=[kl.TOPIC], start_time=START, end_time=END):
        t, b = kl.parse_compressed(msg.data)
        if START <= t < END:
            stamps.append(t); blobs.append(b)
i = np.clip(np.searchsorted(gt[:, 0], stamps), 1, len(gt) - 1)
i = np.where(np.abs(gt[i - 1, 0] - stamps) < np.abs(gt[i, 0] - stamps), i - 1, i)
keep = np.abs(gt[i, 0] - np.array(stamps, float)) <= 5e7
model, tf, dev = kl.load_model(kl.DEFAULTS["weights"])
feat, _ = kl.encode(model, tf, dev, [b for b, k in zip(blobs, keep) if k])
poses = gt[i[keep], 1:4]; st = np.array(stamps)[keep]
laps = np.searchsorted(LAP_STARTS, st, side="right")
np.savez(out, features=feat, poses=poses, stamps=st, laps=laps); print(feat.shape)

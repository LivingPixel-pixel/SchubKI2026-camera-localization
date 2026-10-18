"""Extract car-state twist (body vx, vy, yaw rate) from a run mcap -> .npy [t_ns, vx, vy, wz]."""
import sys, numpy as np
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory
rows = []
with open(sys.argv[1], "rb") as s:
    for _a, _b, _c, m in make_reader(s, decoder_factories=[DecoderFactory()]).iter_decoded_messages(topics=["/constructor0/validation/twist"]):
        rows.append((m.header.stamp.sec * 10**9 + m.header.stamp.nanosec, m.twist.linear.x, m.twist.linear.y, m.twist.angular.z))
a = np.array(rows, dtype=float); np.save(sys.argv[2], a[np.argsort(a[:, 0])]); print(a.shape)

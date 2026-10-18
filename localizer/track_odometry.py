"""Track-constrained causal odometry: s(t) = s0 + integral(speed), s0 voted from ViT retrieval.

Prediction uses only car-state speed (/constructor0/validation/twist, allowed car state data);
the start offset s0 is a causal running weighted mode of (s_retrieved - travelled) over frames <= t.
Pose = racing line (from the track map in the dataset mcap) at s, heading = line tangent.
"""
from __future__ import annotations
import math
import numpy as np


def racing_line(map_mcap, topic="/constructor0/map_ar/racing_line"):
    from mcap.reader import make_reader
    from mcap_ros2.decoder import DecoderFactory
    with open(map_mcap, "rb") as s:
        for _sch, _ch, _m, msg in make_reader(s, decoder_factories=[DecoderFactory()]).iter_decoded_messages(topics=[topic]):
            return np.array([[p.pose.position.x, p.pose.position.y] for p in msg.poses])
    raise ValueError("racing line missing")


class Line:
    def __init__(self, xy):
        xy = np.asarray(xy, float)
        if np.hypot(*(xy[-1] - xy[0])) < 1e-6:
            xy = xy[:-1]
        self.p = xy; d = np.roll(xy, -1, 0) - xy
        self.len = np.hypot(d[:, 0], d[:, 1]); self.u = d / np.maximum(self.len, 1e-9)[:, None]
        self.s = np.r_[0, np.cumsum(self.len[:-1])]; self.lap = float(self.len.sum())

    def project(self, xy):
        xy = np.atleast_2d(xy); out = np.empty(len(xy))
        for i, q in enumerate(xy):
            rel = q - self.p; t = np.clip((rel * self.u).sum(1), 0, self.len)
            k = int(np.argmin(np.hypot(*(rel - t[:, None] * self.u).T)))
            out[i] = (self.s[k] + t[k]) % self.lap
        return out

    def at(self, s):
        s = s % self.lap; k = int(np.searchsorted(self.s, s, side="right") - 1)
        xy = self.p[k] + (s - self.s[k]) * self.u[k]
        return xy, math.atan2(self.u[k, 1], self.u[k, 0])


class TrackOdometry:
    def __init__(self, line, gallery_s, twist, k=5, bin_m=4.0, decay=0.995, switch_ratio=1.5):
        self.L, self.gs, self.tw, self.k, self.bin, self.decay = line, gallery_s, twist, k, bin_m, decay
        self.ratio, self.cur = switch_ratio, None
        self.hist = np.zeros(int(math.ceil(line.lap / bin_m))); self.travel = 0.0; self.t = None; self.j = 0

    def _advance(self, t):
        tw = self.tw
        if self.t is None:
            self.t = t; return
        prev = self.t
        while self.j < len(tw) and tw[self.j, 0] <= prev:
            self.j += 1
        while self.j < len(tw) and tw[self.j, 0] <= t:
            self.travel += (tw[self.j, 0] - prev) * 1e-9 * math.hypot(tw[self.j, 1], tw[self.j, 2]); prev = tw[self.j, 0]; self.j += 1
        if self.j > 0 and prev < t:
            self.travel += (t - prev) * 1e-9 * math.hypot(tw[self.j - 1, 1], tw[self.j - 1, 2])
        self.t = t

    def step(self, sims, t):
        self._advance(t)
        top = np.argpartition(-sims, self.k)[:self.k]
        self.hist *= self.decay
        for c in top:
            off = (self.gs[c] - self.travel) % self.L.lap
            b = int(off // self.bin) % len(self.hist)
            w = math.exp((sims[c] - sims[top].max()) / 0.02)
            for db, f in ((-1, .5), (0, 1.), (1, .5)):
                self.hist[(b + db) % len(self.hist)] += w * f
        b = int(np.argmax(self.hist))
        if self.cur is not None and self.hist[b] < self.ratio * self.hist[self.cur]:
            b = self.cur  # hysteresis: keep the current start offset unless clearly outvoted
        self.cur = b
        nb = [(b + d) % len(self.hist) for d in (-1, 0, 1)]
        centres = np.array([(x + .5) * self.bin for x in nb]); w = self.hist[nb]
        ang = centres / self.L.lap * 2 * math.pi
        s0 = (math.atan2((np.sin(ang) * w).sum(), (np.cos(ang) * w).sum()) / (2 * math.pi) * self.L.lap) % self.L.lap
        xy, yaw = self.L.at(s0 + self.travel)
        return xy, yaw

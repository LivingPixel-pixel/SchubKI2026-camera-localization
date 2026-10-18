"""Causal multi-hypothesis tracker for runs with car-state twist (allowed car state data).

Prediction: integrate /constructor0/validation/twist (body vx, vy, yaw rate) between image stamps.
Update: ViT retrieval candidates gated around each hypothesis (position + heading).
Hypotheses are seeded from distinct global top retrievals (no first pose needed) and scored causally;
the best-scoring hypothesis is emitted at every frame. Uses only data up to the current image stamp.
"""
from __future__ import annotations
import math
import numpy as np


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class Hyp:
    def __init__(self, x, y, yaw):
        self.p = np.array([x, y], float); self.yaw = float(yaw); self.score = 0.0; self.age = 0


class Tracker:
    def __init__(self, gpose, twist, k=40, gate=12.0, yaw_gate=math.radians(35), a_pos=0.35, a_yaw=0.15,
                 n_hyp=12, reseed_every=30, ema=0.97, switch_margin=0.01):
        self.g, self.tw = gpose, twist
        self.k, self.gate, self.yaw_gate, self.a_pos, self.a_yaw = k, gate, yaw_gate, a_pos, a_yaw
        self.n_hyp, self.reseed_every, self.ema, self.margin = n_hyp, reseed_every, ema, switch_margin
        self.cur = None
        self.h, self.t, self.frame, self.j = [], None, 0, 0

    def _seed(self, top, sims, count):
        out = []
        for c in top:
            if all(math.hypot(self.g[c, 0] - o.p[0], self.g[c, 1] - o.p[1]) > 40 for o in self.h + out):
                n = Hyp(*self.g[c]); n.score = float(sims[c]) - 0.05; out.append(n)
            if len(out) >= count:
                break
        return out

    def _propagate(self, t):
        """Integrate twist samples with stamps in (self.t, t]. Causal: never reads samples after t."""
        tw, j = self.tw, self.j
        while j < len(tw) and tw[j, 0] <= self.t:
            j += 1
        prev = self.t
        while j < len(tw) and tw[j, 0] <= t:
            dt = (tw[j, 0] - prev) * 1e-9; prev = tw[j, 0]
            vx, vy, wz = tw[j, 1], tw[j, 2], tw[j, 3]
            for h in self.h:
                c, s = math.cos(h.yaw), math.sin(h.yaw)
                h.p += dt * np.array([c * vx - s * vy, s * vx + c * vy]); h.yaw = wrap(h.yaw + dt * wz)
            j += 1
        if j > 0 and prev < t:  # hold last sample (zero-order) up to the image stamp
            dt = (t - prev) * 1e-9; vx, vy, wz = tw[j - 1, 1], tw[j - 1, 2], tw[j - 1, 3]
            for h in self.h:
                c, s = math.cos(h.yaw), math.sin(h.yaw)
                h.p += dt * np.array([c * vx - s * vy, s * vx + c * vy]); h.yaw = wrap(h.yaw + dt * wz)
        self.j = max(0, j - 1)

    def step(self, sims, t):
        top = np.argpartition(-sims, self.k)[:self.k]; top = top[np.argsort(-sims[top])]
        if not self.h:
            self.h = self._seed(top, sims, self.n_hyp)
        else:
            self._propagate(t)
        for h in self.h:
            d = np.hypot(self.g[top, 0] - h.p[0], self.g[top, 1] - h.p[1])
            dy = np.abs(wrap(self.g[top, 2] - h.yaw))
            ok = (d < self.gate) & (dy < self.yaw_gate)
            if ok.any():
                cand = top[ok]; c = cand[int(np.argmax(sims[cand] - 0.003 * d[ok]))]
                h.p += self.a_pos * (self.g[c, :2] - h.p); h.yaw = wrap(h.yaw + self.a_yaw * wrap(self.g[c, 2] - h.yaw))
                obs = float(sims[c])
            else:
                obs = float(sims[top[-1]]) - 0.05
            off = float(np.min(np.hypot(self.g[::4, 0] - h.p[0], self.g[::4, 1] - h.p[1])))  # off-track penalty
            obs -= 0.01 * max(0.0, off - 8.0)
            h.score = self.ema * h.score + (1 - self.ema) * obs if h.age else obs; h.age += 1
        self.frame += 1
        if self.frame % self.reseed_every == 0 and len(self.h) > 1:
            self.h.sort(key=lambda h: -h.score)
            self.h = self.h[:self.n_hyp - 2] + self._seed(top, sims, 2)
        self.t = t
        best = max(self.h, key=lambda h: h.score)
        if self.cur in self.h and best is not self.cur and best.score < self.cur.score + self.margin:
            best = self.cur
        self.cur = best
        return best.p.copy(), best.yaw, len(self.h)

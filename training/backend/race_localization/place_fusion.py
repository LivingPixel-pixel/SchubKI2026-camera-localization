"""Causal pose fusion from predicted body motion and fine-tuned place fixes.

The unchanged GroundPlaneOdometry supplies frame-to-frame motion. Each image is
embedded by the fine-tuned place encoder and matched against a frozen place map
(training-run embeddings with training-only poses). Gated matches pull the fused
pose toward the matched place. Every output uses only the current and earlier
images, the supplied first pose and the frozen map. No telemetry is decoded here;
experimental state deltas require the caller's separate permission/clock context.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, fields
import json
import math
from pathlib import Path

import numpy as np

from race_localization import place_model as pm

MAP_SCHEMA = "schubki-place-map-v1"
RECORD_SCHEMA = "schubki-place-fusion-record-v1"


def wrap(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def body_delta(previous, current):
    """Motion between two poses expressed in the previous body frame."""
    c, s = math.cos(previous[2]), math.sin(previous[2])
    dx, dy = current[0] - previous[0], current[1] - previous[1]
    return np.array([c * dx + s * dy, -s * dx + c * dy, wrap(current[2] - previous[2])])


def compose(pose, delta):
    c, s = math.cos(pose[2]), math.sin(pose[2])
    return np.array([pose[0] + c * delta[0] - s * delta[1], pose[1] + s * delta[0] + c * delta[1],
                     wrap(pose[2] + delta[2])])


@dataclass(frozen=True)
class FusionConfig:
    top_k: int = 5
    cluster_m: float = 15.0
    temperature: float = 0.02
    min_similarity: float = 0.5
    gate_m: float = 20.0
    gate_yaw_deg: float = 30.0
    gain: float = 0.2
    yaw_gain: float = 0.2
    lost_s: float = 3.0
    recover_count: int = 3
    recover_m: float = 10.0
    # v2. "vo": odometry motion (v1). "constant_velocity": body velocity and yaw rate
    # learned from the fused track. "cv_vo_yaw": learned translation, odometry heading change.
    motion: str = "vo"  # "state" is an experimental, explicitly authorized supplied delta
    velocity_alpha: float = 0.3
    max_speed_mps: float = 80.0
    # v3 (defaults keep v1/v2 behavior). Fix weight rises linearly from 0 at sim_zero to 1 at
    # sim_full. The position gate starts at gate_m right after a fix and widens by
    # gate_growth_mps per second without one, up to gate_max_m.
    sim_zero: float = 0.0
    sim_full: float = 0.0
    gate_growth_mps: float = 0.0
    gate_max_m: float = 0.0
    speed_scale: float = 1.0  # "state" mode only; fitted on laps 1-2

    def __post_init__(self):
        pm.integer(self.top_k, 1, 64)
        pm.integer(self.recover_count, 1, 100)
        if self.sim_full < self.sim_zero or self.sim_full > 1:
            raise ValueError("need sim_zero <= sim_full <= 1")
        if self.gate_growth_mps > 0 and self.gate_max_m < self.gate_m:
            raise ValueError("gate_max_m must be at least gate_m when the gate grows")
        for name in ("cluster_m", "temperature", "min_similarity", "gate_m", "gate_yaw_deg",
                     "gain", "yaw_gain", "lost_s", "recover_m", "velocity_alpha", "max_speed_mps",
                     "sim_zero", "sim_full", "gate_growth_mps", "gate_max_m", "speed_scale"):
            if pm.finite(getattr(self, name)) < 0:
                raise ValueError(f"{name}: nonnegative value required")
        if self.temperature <= 0 or self.gain > 1 or self.yaw_gain > 1 or self.min_similarity > 1:
            raise ValueError("temperature must be positive; gains and similarity at most 1")
        if self.motion not in ("vo", "constant_velocity", "cv_vo_yaw", "state") or self.velocity_alpha > 1:
            raise ValueError("motion must be vo, constant_velocity, cv_vo_yaw or state; velocity_alpha at most 1")
        if not 0.5 <= self.speed_scale <= 1.5:
            raise ValueError("speed_scale must be within 0.5-1.5")

    @property
    def uses_odometry(self):
        return self.motion in ("vo", "cv_vo_yaw")

    @property
    def uses_state_motion(self):
        return self.motion == "state"

    @classmethod
    def from_dict(cls, value):
        names = {field.name for field in fields(cls)}
        if not isinstance(value, dict) or set(value) - names:
            raise ValueError("unknown fusion config fields")
        return cls(**value)


class PlaceMap:
    """Frozen place map: L2 float32[N,D] embeddings (D from the encoder) and float64[N,3] poses."""
    def __init__(self, embeddings, poses, identity):
        embeddings = np.asarray(embeddings, dtype=np.float32)
        poses = np.asarray(poses, dtype=np.float64)
        if embeddings.ndim != 2 or not 1 <= embeddings.shape[1] <= 8192 or poses.shape != (len(embeddings), 3) or not len(poses):
            raise ValueError("place map shape mismatch")
        pm.validate_features(embeddings, len(embeddings), embeddings.shape[1])
        if not np.isfinite(poses).all():
            raise ValueError("nonfinite map pose")
        norms = np.linalg.norm(embeddings, axis=1)
        # Saved maps are already unit length; renormalizing again would perturb the last bit.
        self.embeddings = embeddings if np.abs(norms - 1).max() <= 1e-5 else pm.normalize(embeddings).astype(np.float32)
        self.poses = poses
        self.identity = identity

    def save(self, directory):
        directory = pm.new_output(directory)
        np.save(directory / "embeddings.npy", self.embeddings)
        np.save(directory / "poses.npy", self.poses)
        identity = {**self.identity, "schema": MAP_SCHEMA, "count": len(self.poses),
                    "embeddings_sha256": pm.file_hash(directory / "embeddings.npy", 1 << 30),
                    "poses_sha256": pm.file_hash(directory / "poses.npy", 1 << 30)}
        pm.write_json(directory / "map.json", identity)
        return identity

    @classmethod
    def load(cls, directory, expected_model_sha256=None):
        directory = Path(directory)
        identity = pm.parse_json((directory / "map.json").read_bytes())
        if identity.get("schema") != MAP_SCHEMA:
            raise ValueError("unknown place map schema")
        if expected_model_sha256 is not None and identity.get("model_weights_sha256") != expected_model_sha256:
            raise ValueError("place map was built with a different encoder")
        for name in ("embeddings", "poses"):
            if pm.file_hash(directory / f"{name}.npy", 1 << 30) != identity[f"{name}_sha256"]:
                raise ValueError(f"place map {name} hash mismatch")
        embeddings = np.load(directory / "embeddings.npy", allow_pickle=False)
        poses = np.load(directory / "poses.npy", allow_pickle=False)
        return cls(embeddings, poses, identity)

    def candidate(self, embedding, config: FusionConfig):
        """Similarity-weighted pose of the top matches that agree with the best one."""
        query = np.asarray(embedding, dtype=np.float32)
        if query.shape != (self.embeddings.shape[1],) or not np.isfinite(query).all():
            raise ValueError("query embedding size differs from the place map")
        norm = np.linalg.norm(query)
        if not math.isfinite(norm) or norm <= 1e-12:
            raise ValueError("query embedding must have a finite nonzero norm")
        similarity = self.embeddings @ (query / norm)
        k = min(config.top_k, len(similarity))
        top = np.argpartition(-similarity, k - 1)[:k]
        top = top[np.argsort(-similarity[top], kind="stable")]
        best = self.poses[top[0]]
        keep = top[np.hypot(self.poses[top, 0] - best[0], self.poses[top, 1] - best[1]) <= config.cluster_m]
        weights = np.exp((similarity[keep] - similarity[keep].max()) / config.temperature)
        weights /= weights.sum()
        x, y = weights @ self.poses[keep, 0], weights @ self.poses[keep, 1]
        yaw = math.atan2(weights @ np.sin(self.poses[keep, 2]), weights @ np.cos(self.poses[keep, 2]))
        return {"pose": np.array([x, y, yaw]), "similarity": float(similarity[top[0]]),
                "support": int(len(keep)), "map_index": int(top[0])}


class PlaceFusion:
    """Fused pose state. step() sees only the current frame's VO delta and embedding."""
    def __init__(self, seed, place_map: PlaceMap | None, config: FusionConfig):
        if type(seed["timestamp_ns"]) is not int or not 0 <= seed["timestamp_ns"] < 2**63:
            raise ValueError("seed timestamp must be integer nanoseconds")
        self.pose = np.array([seed["x"], seed["y"], seed["yaw"]], dtype=np.float64)
        if not np.isfinite(self.pose).all():
            raise ValueError("finite seed required")
        self.map, self.config = place_map, config
        self.last_timestamp_ns = int(seed["timestamp_ns"])
        self.last_fix_ns = int(seed["timestamp_ns"])
        self.recent = []  # (candidate pose, fused pose) while lost
        self.velocity = np.zeros(3)  # body forward, body left [m/s], yaw rate [rad/s]
        self._has_observation = False

    def predict(self, dt, vo_delta):
        mode = self.config.motion
        vo_delta = np.asarray(vo_delta, dtype=np.float64)
        if mode == "vo":
            return vo_delta
        if mode == "state":
            return np.array([vo_delta[0] * self.config.speed_scale, vo_delta[1] * self.config.speed_scale, vo_delta[2]])
        learned = self.velocity * dt
        return learned if mode == "constant_velocity" else np.array([learned[0], learned[1], vo_delta[2]])

    def step(self, timestamp_ns, vo_delta, embedding):
        if (type(timestamp_ns) is not int or not 0 <= timestamp_ns < 2**63
                or timestamp_ns < self.last_timestamp_ns
                or (self._has_observation and timestamp_ns == self.last_timestamp_ns)):
            raise ValueError("exposure timestamps must increase after the initial seed")
        if timestamp_ns == self.last_timestamp_ns:
            # The supplied pose is the observation at this exact time, not a visual hypothesis.
            self._has_observation = True
            return {"x": float(self.pose[0]), "y": float(self.pose[1]), "yaw": float(self.pose[2]), "status": "seed"}
        previous = self.pose.copy()
        previous_stamp, previous_fix = self.last_timestamp_ns, self.last_fix_ns
        previous_velocity, previous_observed = self.velocity.copy(), self._has_observation
        try:
            delta = np.asarray(vo_delta, dtype=np.float64)
            if delta.shape != (3,) or not np.isfinite(delta).all():
                raise ValueError("motion delta must be a finite three-component vector")
            candidate = self.map.candidate(embedding, self.config) if (
                self.map is not None and embedding is not None and self.config.gain > 0) else None
            dt = (timestamp_ns - previous_stamp) / 1e9
            predicted = compose(previous, self.predict(dt, delta))
            if not np.isfinite(predicted).all():
                raise ValueError("predicted pose must be finite")
            self.last_timestamp_ns, self.pose = timestamp_ns, predicted
            result = self._correct(timestamp_ns, candidate)
            if result["status"] != "relocalized":
                # A relocalization jump is not motion.
                measured = body_delta(previous, self.pose) / dt
                if not np.isfinite(measured).all():
                    raise ValueError("measured motion must be finite")
                speed = math.hypot(measured[0], measured[1])
                if speed > self.config.max_speed_mps:
                    measured[:2] *= self.config.max_speed_mps / speed
                alpha = self.config.velocity_alpha
                self.velocity = (1 - alpha) * self.velocity + alpha * measured
            self._has_observation = True
            return result
        except (TypeError, ValueError, OverflowError, FloatingPointError):
            self.pose, self.last_timestamp_ns, self.last_fix_ns = previous, previous_stamp, previous_fix
            self.velocity, self._has_observation = previous_velocity, previous_observed
            self.recent = []  # A rejected observation cannot bridge recovery evidence.
            raise

    def _weight(self, similarity):
        low, high = self.config.sim_zero, self.config.sim_full
        if high <= low:
            return 1.0
        return min(1.0, max(0.0, (similarity - low) / (high - low)))

    def _correct(self, timestamp_ns, candidate):
        since_fix = (timestamp_ns - self.last_fix_ns) / 1e9
        lost = since_fix > self.config.lost_s
        status = "lost" if lost and self.map is not None and self.config.gain > 0 else "odometry"
        detail = {}
        if candidate is None:
            self.recent = []
        else:
            target = candidate["pose"]
            distance = float(np.hypot(*(target[:2] - self.pose[:2])))
            heading = abs(math.degrees(wrap(target[2] - self.pose[2])))
            detail = {"similarity": round(candidate["similarity"], 4), "support": candidate["support"],
                      "distance_m": round(distance, 3), "heading_deg": round(heading, 2)}
            weight = self._weight(candidate["similarity"])
            confident = candidate["similarity"] >= self.config.min_similarity and weight > 0
            gate = self.config.gate_m
            if self.config.gate_growth_mps > 0:
                gate = min(self.config.gate_max_m, gate + self.config.gate_growth_mps * since_fix)
            if confident and distance <= gate and heading <= self.config.gate_yaw_deg:
                self.pose[:2] += weight * self.config.gain * (target[:2] - self.pose[:2])
                self.pose[2] = wrap(self.pose[2] + weight * self.config.yaw_gain * wrap(target[2] - self.pose[2]))
                self.last_fix_ns, self.recent, status = timestamp_ns, [], "fixed"
            elif lost and confident:
                self.recent = (self.recent + [(target.copy(), self.pose.copy())])[-self.config.recover_count:]
                if len(self.recent) == self.config.recover_count and self._consistent():
                    self.pose = target.copy()
                    self.last_fix_ns, self.recent, status = timestamp_ns, [], "relocalized"
                else:
                    status = "lost"
            elif lost:
                self.recent = []
                status = "lost"
            else:
                self.recent = []
        return {"x": float(self.pose[0]), "y": float(self.pose[1]), "yaw": float(self.pose[2]),
                "status": status, **detail}

    def _consistent(self):
        """Successive candidates must move like the odometry did between them.

        Compared in body frames, so a wrong lost heading cannot hide a correct match.
        """
        for (c0, f0), (c1, f1) in zip(self.recent, self.recent[1:]):
            mismatch = body_delta(c0, c1)[:2] - body_delta(f0, f1)[:2]
            if np.hypot(*mismatch) > self.config.recover_m:
                return False
        return True


def replay(seed, place_map, config, timestamps, deltas, embeddings):
    """Deterministic offline replay of a causal record; identical maths to live runs."""
    fusion = PlaceFusion(seed, place_map, config)
    rows = []
    for index, stamp in enumerate(timestamps):
        rows.append((int(stamp), fusion.step(int(stamp), deltas[index], None if embeddings is None else embeddings[index])))
    return rows


def write_csv(path, rows):
    path = Path(path)
    temporary = path.with_suffix(".csv.partial")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        stream.write("timestamp_ns,x,y,yaw\n")
        for stamp, row in rows:
            stream.write(f"{stamp},{row['x']:.10f},{row['y']:.10f},{row['yaw']:.10f}\n")
    temporary.replace(path)


def config_json(config):
    return json.loads(json.dumps(asdict(config)))

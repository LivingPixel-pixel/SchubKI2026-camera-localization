"""Pure causal planar SE(2) predictor (TEAM-MOTION-01).

``Predictor(seed, policy).step(...)`` consumes arrival prefixes only: the
motion samples that arrived since the previous image record, plus an optional
current-image visual approximation. It integrates a stable constant-twist
SE(2) model with the latest available sample held piecewise constant, never
interpolates towards a future sample, never rewrites an emitted interval and
never reseeds from truth. There is no IO, clock, image or model access.

Output states follow the shared ``causal_planar_v1`` wire: ``predicted``,
``approximate_corrected``, ``held`` and ``lost``. The organizer seed is anchor
provenance, not a fifth state. Every status still describes an approximate
process; ``predicted`` is not a geometrically measured fix, and an accepted
visual correction remains a copied training-reference approximation.

Semantics follow coordinator clarification 6232540283 and R1 repair 6232930062
(M1 exact seed output, M2 source-after-image admission) on issue 9. All policy
values are caller supplied; example values in tests and docs are UNMEASURED
proposals. Real telemetry use stays blocked until the coordinator verifies
sign, BODY reference point, units, clocks and arrival order.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
import re

from .motion_projection import MotionSample

__all__ = [
    "ANCHOR_KINDS", "ArrivalContext", "ArrivalContextUndeclared", "NotReady", "PlanarEstimate", "PlanarPolicy",
    "PlanarSeed", "Predictor", "SCHEMA", "STATUSES", "VisualReference", "WIRE_FIELDS", "advance", "wrap_angle",
]

SCHEMA = "causal_planar_v1"
STATUSES = ("predicted", "approximate_corrected", "held", "lost")
ANCHOR_KINDS = ("organizer_seed", "approximate_reference_pose")
WIRE_FIELDS = (
    "schema", "timestamp_ns", "image_log_ns", "image_seq", "x", "y", "yaw", "map_frame", "status",
    "last_anchor_ns", "anchor_age_ns", "anchor_kind", "latest_source_ns", "latest_log_ns",
    "latest_input_seq", "motion_age_ns", "visual_reference", "reason", "source_pins",
)
LOG_CLOCKS = ("comparable_to_image_log", "not_comparable")
_INT64_MAX = 2**63 - 1
_TAU = 2.0 * math.pi
_NS_PER_S = 1_000_000_000
_HEX64 = re.compile(r"[0-9a-f]{64}")
_MAX_PINS = 16
_MAX_REASON = 160


class NotReady(RuntimeError):
    """A required prerequisite is absent; no pose is fabricated."""


class ArrivalContextUndeclared(NotReady):
    """No explicit arrival/clock declaration exists, so no ordering or integration runs."""


def _ns(value, name: str) -> int:
    if type(value) is not int or not 0 <= value <= _INT64_MAX:
        raise ValueError(f"{name} must be an exact int in [0, 2**63-1]")
    return value


def _real(value, name: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite real number")
    return float(value)


def _text(value, name: str, limit: int) -> str:
    if type(value) is not str or not 1 <= len(value) <= limit or not value.isprintable():
        raise ValueError(f"{name} must be printable text of 1 to {limit} characters")
    return value


def _yaw(value, name: str) -> float:
    value = _real(value, name)
    if not -math.pi <= value <= math.pi:
        raise ValueError(f"{name} must already lie in [-pi, pi]")
    return value


def wrap_angle(angle: float) -> float:
    """Wrap an angle to [-pi, pi]."""
    return math.remainder(angle, _TAU)


def _sinc(x: float) -> float:
    if abs(x) < 1e-4:
        x2 = x * x
        return 1.0 - x2 / 6.0 + x2 * x2 / 120.0
    return math.sin(x) / x


def advance(x: float, y: float, yaw: float, v_mps: float, wz_radps: float, dt_ns: int) -> tuple[float, float, float]:
    """Exact constant-twist SE(2) step: chord v*dt*sinc(a/2) along yaw+a/2."""
    dt = dt_ns / _NS_PER_S
    a = wz_radps * dt
    travel = v_mps * dt * _sinc(0.5 * a)
    heading = yaw + 0.5 * a
    return x + travel * math.cos(heading), y + travel * math.sin(heading), wrap_angle(yaw + a)


@dataclass(frozen=True, slots=True)
class PlanarSeed:
    """The single permitted organizer seed: BODY pose in the map frame."""

    timestamp_ns: int
    x: float
    y: float
    yaw: float
    map_frame: str

    def __post_init__(self):
        _ns(self.timestamp_ns, "seed timestamp_ns")
        object.__setattr__(self, "x", _real(self.x, "seed x"))
        object.__setattr__(self, "y", _real(self.y, "seed y"))
        object.__setattr__(self, "yaw", _yaw(self.yaw, "seed yaw"))
        _text(self.map_frame, "seed map_frame", 80)


@dataclass(frozen=True, slots=True)
class ArrivalContext:
    """Explicit caller declaration of arrival order and clock comparability.

    Fabricated declarations are test inputs only. Production requires the
    coordinator's validated convention check of the actual recording.
    """

    declaration_ref: str
    sequence_scope: str
    source_clock: str
    log_clock: str

    def __post_init__(self):
        _text(self.declaration_ref, "declaration_ref", 160)
        if self.sequence_scope != "recording_wide":
            raise ValueError("sequence_scope must be 'recording_wide'; per-topic counters are not global order")
        if self.source_clock != "comparable_to_image_exposure":
            raise ValueError("source_clock must declare comparability to image exposure before integration")
        if self.log_clock not in LOG_CLOCKS:
            raise ValueError(f"log_clock must be one of {LOG_CLOCKS}")

    @property
    def log_comparable(self) -> bool:
        return self.log_clock == "comparable_to_image_log"


@dataclass(frozen=True, slots=True)
class PlanarPolicy:
    """Caller-supplied bounds. No field has a default racing value."""

    max_sample_age_ns: int
    blind_horizon_ns: int
    max_pending_samples: int
    max_abs_speed_mps: float
    max_abs_yaw_rate_radps: float
    innovation_gate_m: float
    innovation_gate_rad: float
    correction_gain: float
    expected_map_id: str
    arrival_context: ArrivalContext | None

    def __post_init__(self):
        if _ns(self.max_sample_age_ns, "max_sample_age_ns") == 0:
            raise ValueError("max_sample_age_ns must be positive")
        _ns(self.blind_horizon_ns, "blind_horizon_ns")
        if type(self.max_pending_samples) is not int or not 1 <= self.max_pending_samples <= 100_000:
            raise ValueError("max_pending_samples must be an int in [1, 100000]")
        for name in ("max_abs_speed_mps", "max_abs_yaw_rate_radps", "innovation_gate_m"):
            if _real(getattr(self, name), name) <= 0.0:
                raise ValueError(f"{name} must be positive")
            object.__setattr__(self, name, float(getattr(self, name)))
        gate_rad = _real(self.innovation_gate_rad, "innovation_gate_rad")
        if not 0.0 < gate_rad <= math.pi:
            raise ValueError("innovation_gate_rad must lie in (0, pi]")
        gain = _real(self.correction_gain, "correction_gain")
        if not 0.0 < gain < 1.0:
            raise ValueError("correction_gain must lie strictly inside (0, 1)")
        object.__setattr__(self, "innovation_gate_rad", gate_rad)
        object.__setattr__(self, "correction_gain", gain)
        _text(self.expected_map_id, "expected_map_id", 128)
        if self.arrival_context is not None and type(self.arrival_context) is not ArrivalContext:
            raise ValueError("arrival_context must be an ArrivalContext or None")


@dataclass(frozen=True, slots=True)
class VisualReference:
    """Coarse provenance of an accepted approximate visual anchor."""

    reference_id: str
    pose_kind: str = "approximate_reference_pose"
    position_source: str = "training_reference"
    heading_source: str = "training_reference"
    yaw_approximate: bool = True
    translation_scale_observed: bool = False

    def __post_init__(self):
        _text(self.reference_id, "reference_id", 120)
        if (self.pose_kind != "approximate_reference_pose" or self.position_source != "training_reference"
                or self.heading_source != "training_reference" or self.yaw_approximate is not True
                or self.translation_scale_observed is not False):
            raise ValueError("visual reference provenance flags must describe an approximate reference copy")

    def wire(self) -> dict:
        return {"reference_id": self.reference_id, "pose_kind": self.pose_kind,
                "position_source": self.position_source, "heading_source": self.heading_source,
                "yaw_approximate": self.yaw_approximate,
                "translation_scale_observed": self.translation_scale_observed}


@dataclass(frozen=True, slots=True)
class PlanarEstimate:
    """Immutable per-image output plus non-wire diagnostics."""

    timestamp_ns: int
    image_log_ns: int
    image_seq: int
    x: float
    y: float
    yaw: float
    map_frame: str
    status: str
    last_anchor_ns: int
    anchor_age_ns: int
    anchor_kind: str
    latest_source_ns: int | None
    latest_log_ns: int | None
    latest_input_seq: int | None
    motion_age_ns: int | None
    visual_reference: VisualReference | None
    reason: str
    source_pins: tuple[tuple[str, str], ...]
    coverage_break_ns: int | None
    chain_intact: bool
    correction: str
    sample_rejections: tuple[tuple[int, str], ...]
    schema: str = SCHEMA

    def wire(self) -> dict:
        """Native primitive record with exactly the causal_planar_v1 fields."""
        return {
            "schema": self.schema, "timestamp_ns": self.timestamp_ns, "image_log_ns": self.image_log_ns,
            "image_seq": self.image_seq, "x": self.x, "y": self.y, "yaw": self.yaw,
            "map_frame": self.map_frame, "status": self.status, "last_anchor_ns": self.last_anchor_ns,
            "anchor_age_ns": self.anchor_age_ns, "anchor_kind": self.anchor_kind,
            "latest_source_ns": self.latest_source_ns, "latest_log_ns": self.latest_log_ns,
            "latest_input_seq": self.latest_input_seq, "motion_age_ns": self.motion_age_ns,
            "visual_reference": None if self.visual_reference is None else self.visual_reference.wire(),
            "reason": self.reason, "source_pins": dict(self.source_pins),
        }


@dataclass(frozen=True, slots=True)
class _State:
    last_image_ns: int | None
    last_seq: int | None
    last_log_ns: int | None
    last_source_ns: int | None
    pose: tuple[float, float, float]
    intact: bool
    break_ns: int | None
    active: MotionSample | None
    used: MotionSample | None
    anchor_ns: int
    anchor_kind: str
    visual: VisualReference | None


def _pins(source_pins) -> tuple[tuple[str, str], ...]:
    if source_pins is None:
        return ()
    if not isinstance(source_pins, dict) or len(source_pins) > _MAX_PINS:
        raise ValueError("source_pins must be a dict of at most 16 entries")
    for name, digest in source_pins.items():
        _text(name, "source pin name", 80)
        if type(digest) is not str or _HEX64.fullmatch(digest) is None:
            raise ValueError("source pin values must be lowercase SHA256 hex")
    return tuple(sorted(source_pins.items()))


class Predictor:
    """Stateful causal driver; every step is validated before it commits."""

    __slots__ = ("_seed", "_policy", "_pins", "_state")

    def __init__(self, seed: PlanarSeed | None, policy: PlanarPolicy, *, source_pins: dict | None = None):
        if seed is None:
            raise NotReady("missing_initial_seed")
        if type(seed) is not PlanarSeed:
            raise TypeError("seed must be a PlanarSeed")
        if type(policy) is not PlanarPolicy:
            raise TypeError("policy must be a PlanarPolicy")
        self._seed = seed
        self._policy = policy
        self._pins = _pins(source_pins)
        self._state = _State(None, None, None, None, (seed.x, seed.y, seed.yaw), True, None, None, None,
                             seed.timestamp_ns, "organizer_seed", None)

    @property
    def seed(self) -> PlanarSeed:
        return self._seed

    @property
    def policy(self) -> PlanarPolicy:
        return self._policy

    def step(self, image_ns: int, image_log_ns: int, image_seq: int, samples, place_estimate=None) -> PlanarEstimate:
        """Emit the estimate for one image record.

        ``samples`` are the MotionSamples that arrived after the previous image
        record and before this one, in recording-wide arrival order. A sample
        whose source time is after the image is rejected at admission and never
        retained. ``place_estimate`` is an optional current-image approximate
        visual result; it is ignored at the exact seed image.
        """
        st, policy, seed = self._state, self._policy, self._seed
        context = policy.arrival_context
        if context is None:
            raise ArrivalContextUndeclared("arrival_context_undeclared")
        _ns(image_ns, "image_ns")
        _ns(image_log_ns, "image_log_ns")
        _ns(image_seq, "image_seq")
        if image_ns < seed.timestamp_ns:
            raise ValueError("image_before_seed")
        if st.last_image_ns is not None and image_ns <= st.last_image_ns:
            raise ValueError("image_time_not_increasing")
        if st.last_seq is not None and image_seq <= st.last_seq:
            raise ValueError("image_sequence_not_increasing")
        if context.log_comparable and st.last_log_ns is not None and image_log_ns < st.last_log_ns:
            raise ValueError("image_log_reversed")
        if not isinstance(samples, (tuple, list)):
            raise TypeError("samples must be a tuple or list of MotionSample in arrival order")
        start = seed.timestamp_ns if st.last_image_ns is None else st.last_image_ns
        last_seq, last_log, last_source = st.last_seq, st.last_log_ns, st.last_source_ns
        # Admitted samples newer than the last emitted image; all lie inside (start, image_ns].
        active, pending, rejections = st.active, [], []
        for sample in samples:
            if type(sample) is not MotionSample:
                raise TypeError("samples must contain only MotionSample")
            # Arrival order advances for every structurally ordered record. The source
            # watermark advances only for an admissible source time (not after the image),
            # even if the values are then rejected, so a rejection cannot hide a reversal.
            reason = self._arrival_rejection(sample, image_log_ns, image_seq, last_seq, last_log)
            if reason is None:
                last_seq = sample.seq
                if context.log_comparable:
                    last_log = sample.log_ns
                reason = self._source_rejection(sample, image_ns, last_source)
                if reason is None:
                    last_source = sample.source_ns
                    reason = self._value_rejection(sample, start, len(pending))
            if reason is not None:
                rejections.append((sample.seq, reason))
                continue
            if sample.source_ns <= start:
                active = sample  # late arrival: applies from the last emitted image onward only
            else:
                pending.append(sample)

        # Coverage of the positive interval [start, image_ns] from the last emitted pose.
        max_age = policy.max_sample_age_ns
        integrated, covered, break_ns = st.pose, st.intact, st.break_ns
        if st.intact:
            x, y, yaw = st.pose
            u = start
            while True:
                while pending and pending[0].source_ns <= u:
                    active = pending.pop(0)
                if u >= image_ns:
                    break
                seg_end = image_ns if active is None else min(image_ns, active.source_ns + max_age)
                if pending:
                    seg_end = min(seg_end, pending[0].source_ns)
                # seg_end <= u means the active sample's age would exceed max_sample_age.
                if active is None or seg_end <= u:
                    covered, break_ns = False, u
                    break
                x, y, yaw = advance(x, y, yaw, active.v_mps, active.wz_radps, seg_end - u)
                u = seg_end
            if covered:
                integrated = (x, y, yaw)
        while pending:
            active = pending.pop(0)  # every admitted source is <= image_ns
        fresh = active is not None and image_ns - active.source_ns <= max_age

        # Pre-correction estimate: lost by anchor age, then held, then predicted.
        anchor_ns, anchor_kind, visual = st.anchor_ns, st.anchor_kind, st.visual
        anchor_age = image_ns - anchor_ns
        at_seed = st.last_image_ns is None and image_ns == seed.timestamp_ns
        if at_seed:
            pose, intact, used = st.pose, True, active if fresh else None
            status, reason = ("predicted" if fresh else "held"), "seed_anchor"
        elif covered and anchor_age <= policy.blind_horizon_ns:
            pose, intact, used = integrated, True, active
            status, reason = "predicted", "motion_fresh"
        else:
            # Lost and held both keep the last emitted pose; neither resumes a broken chain.
            pose, intact, used = st.pose, False, st.used
            if anchor_age > policy.blind_horizon_ns:
                status, reason = "lost", "blind_horizon_exceeded"
            elif st.intact and not covered:
                # A fresh end sample cannot hide an earlier coverage gap.
                status = "held"
                reason = "motion_missing" if active is None else ("motion_coverage_gap" if fresh else "motion_stale")
            elif fresh:
                status, reason = "held", "chain_broken_awaiting_correction"
            else:
                status, reason = "held", "motion_missing" if active is None else "motion_stale"

        correction = "none"
        if place_estimate is not None and at_seed:
            # The exact seed output is never altered by a visual result (non-wire diagnostic only).
            correction = "ignored:seed_image"
        elif place_estimate is not None:
            reference, why = self._reference(place_estimate, image_ns)
            if reference is None:
                correction = f"rejected:{why}"
            else:
                ref_visual, rx, ry, ryaw = reference
                x, y, yaw = pose
                dx, dy, dyaw = rx - x, ry - y, wrap_angle(ryaw - yaw)
                if math.hypot(dx, dy) > policy.innovation_gate_m:
                    correction = "rejected:innovation_gate_distance"
                elif abs(dyaw) > policy.innovation_gate_rad:
                    correction = "rejected:innovation_gate_yaw"
                elif not fresh:
                    correction = "rejected:requires_fresh_motion"
                else:
                    g = policy.correction_gain
                    pose = (x + g * dx, y + g * dy, wrap_angle(yaw + g * dyaw))
                    reason = "approximate_reference_blended" if status == "predicted" else f"recovered_from_{status}"
                    correction, status = "applied", "approximate_corrected"
                    intact, used, break_ns = True, active, None
                    anchor_ns, anchor_kind, visual = image_ns, "approximate_reference_pose", ref_visual
        if correction.startswith("rejected:"):
            reason = f"{reason};correction_{correction}"
        reason = reason[:_MAX_REASON]
        if intact:
            break_ns = None

        estimate = PlanarEstimate(
            image_ns, image_log_ns, image_seq, pose[0], pose[1], pose[2], seed.map_frame, status, anchor_ns,
            image_ns - anchor_ns, anchor_kind,
            None if used is None else used.source_ns, None if used is None else used.log_ns,
            None if used is None else used.seq, None if used is None else image_ns - used.source_ns,
            visual, reason, self._pins, break_ns, intact, correction, tuple(rejections))
        self._state = replace(
            st, last_image_ns=image_ns, last_seq=image_seq, last_log_ns=image_log_ns if context.log_comparable
            else None, last_source_ns=last_source, pose=pose, intact=intact, break_ns=break_ns, active=active,
            used=used, anchor_ns=anchor_ns, anchor_kind=anchor_kind, visual=visual)
        return estimate

    def _arrival_rejection(self, s, image_log_ns, image_seq, last_seq, last_log) -> str | None:
        if last_seq is not None and s.seq <= last_seq:
            return "arrival_sequence_reversed"
        if s.seq == image_seq:
            return "arrival_sequence_reused"
        if s.seq > image_seq:
            return "arrival_after_image"
        if self._policy.arrival_context.log_comparable:
            if s.log_ns > image_log_ns:
                return "arrival_after_image"
            if last_log is not None and s.log_ns < last_log:
                return "arrival_log_reversed"
        return None

    @staticmethod
    def _source_rejection(s, image_ns, last_source) -> str | None:
        if s.source_ns > image_ns:
            return "source_after_image"  # never retained for a later image
        if last_source is not None and s.source_ns == last_source:
            return "duplicate_source_time"
        if last_source is not None and s.source_ns < last_source:
            return "reversed_source_time"
        return None

    def _value_rejection(self, s, start, pending_count) -> str | None:
        policy = self._policy
        if abs(s.v_mps) > policy.max_abs_speed_mps or abs(s.wz_radps) > policy.max_abs_yaw_rate_radps:
            return "signal_out_of_bounds"
        if s.source_ns > start and pending_count >= policy.max_pending_samples:
            return "pending_capacity"
        return None

    def _reference(self, estimate, image_ns: int):
        """Validate a current-image approximate place estimate; never a metric fix."""
        if getattr(estimate, "status", None) != "approximate_place":
            return None, "visual_not_approximate_place"
        pose = getattr(estimate, "pose", None)
        if pose is None:
            return None, "visual_pose_missing"
        try:
            visual = VisualReference(
                getattr(estimate, "selected_reference_id", None), getattr(pose, "pose_kind", None),
                getattr(pose, "position_source", None), getattr(pose, "heading_source", None),
                getattr(pose, "yaw_approximate", None), getattr(pose, "translation_scale_observed", None))
        except ValueError:
            return None, "invalid_reference_provenance"
        if getattr(estimate, "map_id", None) != self._policy.expected_map_id:
            return None, "wrong_map"
        if getattr(pose, "map_frame", None) != self._seed.map_frame:
            return None, "wrong_map_frame"
        stamp = getattr(estimate, "timestamp_ns", None)
        if type(stamp) is not int:
            return None, "invalid_reference_time"
        if stamp < image_ns:
            return None, "delayed_correction"
        if stamp > image_ns:
            return None, "future_correction"
        try:
            rx, ry = _real(getattr(pose, "x", None), "x"), _real(getattr(pose, "y", None), "y")
            ryaw = _yaw(getattr(pose, "yaw", None), "yaw")
        except ValueError:
            return None, "invalid_reference_pose"
        return (visual, rx, ry, ryaw), None

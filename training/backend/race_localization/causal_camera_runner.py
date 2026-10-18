"""Bounded synchronous causal camera event loop with injected IO (TEAM-RUNNER-01).

Composes the accepted Motion predictor (``causal_se2.Predictor``) with an
injected current-image visual callback compatible with
``KeyframeLocalizer.localize``. Every input arrives through one caller-supplied
event iterator in physical recording order; every output leaves through
injected sinks. The loop performs no file, MCAP, image-decoding, label, clock
or network access of its own, imports no exporter and never sorts, looks ahead
or reseeds.

Exactly one predictor is built from the single permitted seed. Pre-seed images
are skipped without vision; every eligible image from the seed through the
frozen output end gets exactly one visual call and one predictor step, and only
images inside the frozen output suffix emit a ``causal_planar_v1`` wire row.
Any order, identity, budget, callback, estimator or sink fault stops the run:
the already emitted prefix stays, the failure sink is told, and finalization
never happens. A sink return is not proof of flush, fsync or durable storage.

Source-only. Real conventions, the raw MCAP adapter, the exporter and timing
measurement remain root-owned and unverified here.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any, Callable, Iterable
import unicodedata

from .causal_se2 import NotReady, PlanarPolicy, PlanarSeed, Predictor
from .keyframe_localization import PlaceEstimate
from .motion_projection import MotionSample

__all__ = [
    "DIAGNOSTIC_SCHEMA", "FAILURE_SCHEMA", "SUMMARY_SCHEMA", "CausalCameraRunner", "ExpectedImages",
    "ImageEvent", "ProjectionRejected", "RunIdentity", "RunSummary", "RunnerFailure", "RunnerLimits",
    "UnsupportedSource", "admit_identifier",
]

DIAGNOSTIC_SCHEMA = "causal_camera_runner_diag_v1"
FAILURE_SCHEMA = "causal_camera_runner_failure_v1"
SUMMARY_SCHEMA = "causal_camera_runner_summary_v1"
_INT64_MAX = 2**63 - 1
_CODE = re.compile(r"[a-z][a-z0-9_]{0,63}")
_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,79}")
_HEX64 = re.compile(r"[0-9a-f]{64}")
_MAX_REASON = 160
_MAX_MESSAGE = 1000
_MAX_REFERENCES = 64
_MAX_PINS = 16
_VISUAL_STATUSES = ("approximate_place", "rejected")


def _ns(value, name: str) -> int:
    if type(value) is not int or not 0 <= value <= _INT64_MAX:
        raise ValueError(f"{name} must be an exact int in [0, 2**63-1]")
    return value


def admit_identifier(value, name: str, limit: int) -> str:
    """Admit an exact identifier: nonempty, bounded, unpadded, no Cc/Cs. Never normalized."""
    if type(value) is not str or not 1 <= len(value) <= limit:
        raise ValueError(f"{name} must be a str of 1 to {limit} characters")
    if value != value.strip():
        raise ValueError(f"{name} must not have leading or trailing whitespace; it is never stripped")
    if any(unicodedata.category(ch) in ("Cc", "Cs") for ch in value):
        raise ValueError(f"{name} must not contain control or surrogate characters")
    return value


def _code(value, name: str) -> str:
    if type(value) is not str or _CODE.fullmatch(value) is None:
        raise ValueError(f"{name} must match [a-z][a-z0-9_]{{0,63}}")
    return value


def _positive(value, name: str) -> int:
    if type(value) is not int or not 1 <= value <= _INT64_MAX:
        raise ValueError(f"{name} must be an explicit positive int")
    return value


# Events ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class ProjectionRejected:
    """An individual motion record the adapter could not project; recorded, not fatal."""

    log_ns: int
    seq: int
    code: str

    def __post_init__(self):
        _ns(self.log_ns, "log_ns")
        _ns(self.seq, "seq")
        _code(self.code, "code")


@dataclass(frozen=True, slots=True)
class UnsupportedSource:
    """Unsupported schema or encapsulation for the motion source; always fatal."""

    log_ns: int
    seq: int
    code: str

    def __post_init__(self):
        _ns(self.log_ns, "log_ns")
        _ns(self.seq, "seq")
        _code(self.code, "code")


@dataclass(frozen=True, slots=True)
class ImageEvent:
    """One pinned-camera image record: original exposure, arrival identity and bytes only."""

    exposure_ns: int
    log_ns: int
    seq: int
    camera_id: str
    encoded_image: bytes

    def __post_init__(self):
        _ns(self.exposure_ns, "exposure_ns")
        _ns(self.log_ns, "log_ns")
        _ns(self.seq, "seq")
        if type(self.camera_id) is not str:
            raise ValueError("camera_id must be a str")
        if type(self.encoded_image) is not bytes:
            raise ValueError("encoded_image must be immutable bytes")


_EVENT_TYPES = (MotionSample, ProjectionRejected, UnsupportedSource, ImageEvent)


# Configuration --------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class RunIdentity:
    run_id: str
    camera_id: str
    calibration_id: str
    map_id: str
    feature_id: str
    map_frame: str
    convention_ref: str
    reference_ids: tuple[str, ...]
    source_pins: tuple[tuple[str, str], ...]

    def __post_init__(self):
        if type(self.run_id) is not str or _RUN_ID.fullmatch(self.run_id) is None:
            raise ValueError("run_id must match [A-Za-z0-9][A-Za-z0-9._-]{0,79}")
        admit_identifier(self.camera_id, "camera_id", 32)
        for name in ("calibration_id", "map_id", "feature_id"):
            admit_identifier(getattr(self, name), name, 128)
        admit_identifier(self.map_frame, "map_frame", 80)
        admit_identifier(self.convention_ref, "convention_ref", 160)
        refs = self.reference_ids
        if type(refs) is not tuple or not 1 <= len(refs) <= _MAX_REFERENCES or len(set(refs)) != len(refs):
            raise ValueError("reference_ids must be a tuple of 1 to 64 unique identifiers")
        for ref in refs:
            admit_identifier(ref, "reference_id", 120)
        pins = self.source_pins
        if type(pins) is not tuple or len(pins) > _MAX_PINS or len({name for name, _ in pins}) != len(pins):
            raise ValueError("source_pins must be a tuple of at most 16 unique (name, sha256) pairs")
        for item in pins:
            if type(item) is not tuple or len(item) != 2:
                raise ValueError("source_pins entries must be (name, sha256) tuples")
            admit_identifier(item[0], "source pin name", 80)
            if type(item[1]) is not str or _HEX64.fullmatch(item[1]) is None:
                raise ValueError("source pin values must be lowercase SHA256 hex")


@dataclass(frozen=True, slots=True)
class RunnerLimits:
    """Explicit finite budgets; no field has a default."""

    max_events: int
    max_images: int
    max_pending_inputs: int
    max_encoded_image_bytes: int
    max_expected_outputs: int
    max_diagnostic_bytes: int

    def __post_init__(self):
        for name in self.__dataclass_fields__:
            _positive(getattr(self, name), name)


@dataclass(frozen=True, slots=True)
class ExpectedImages:
    """Frozen processed-prefix manifest from the seed; output is its suffix from output_start_ns."""

    prefix_ns: tuple[int, ...]
    output_start_ns: int

    def __post_init__(self):
        prefix = self.prefix_ns
        if type(prefix) is not tuple or not prefix:
            raise ValueError("prefix_ns must be a nonempty tuple")
        for index, stamp in enumerate(prefix):
            _ns(stamp, "prefix_ns")
            if index and stamp <= prefix[index - 1]:
                raise ValueError("prefix_ns must strictly increase; it is never sorted")
        _ns(self.output_start_ns, "output_start_ns")
        if self.output_start_ns not in prefix:
            raise ValueError("output_start_ns must be an item of prefix_ns")

    @property
    def output_index(self) -> int:
        return self.prefix_ns.index(self.output_start_ns)

    @property
    def output_ns(self) -> tuple[int, ...]:
        return self.prefix_ns[self.output_index:]


@dataclass(frozen=True, slots=True)
class RunSummary:
    run_id: str
    events_consumed: int
    pre_seed_images_skipped: int
    motion_samples: int
    projection_rejections: int
    images_processed: int
    rows_emitted: int
    status_counts: tuple[tuple[str, int], ...]
    last_diagnostic_sink_returned_ns: int

    def record(self) -> dict:
        return {
            "schema": SUMMARY_SCHEMA, "run_id": self.run_id, "events_consumed": str(self.events_consumed),
            "pre_seed_images_skipped": str(self.pre_seed_images_skipped),
            "motion_samples": str(self.motion_samples), "projection_rejections": str(self.projection_rejections),
            "images_processed": str(self.images_processed), "rows_emitted": str(self.rows_emitted),
            "status_counts": {name: str(count) for name, count in self.status_counts},
            "last_diagnostic_sink_returned_ns": str(self.last_diagnostic_sink_returned_ns),
            "durability": "sink_returns_only_not_flush_or_fsync_proof",
        }


class RunnerFailure(RuntimeError):
    """A stopped run. ``code`` is stable; ``original`` keeps the causing error."""

    def __init__(self, code: str, message: str, original: BaseException | None = None):
        super().__init__(f"{code}: {message}")
        self.code = code
        self.original = original
        self.failure_sink_error: BaseException | None = None


class _Stop(Exception):
    def __init__(self, code: str, message: str, original: BaseException | None = None):
        super().__init__(message)
        self.code, self.message, self.original = code, message, original


def _canonical(record: dict) -> str:
    return json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def _pairs(values) -> list[list[str]]:
    return [[str(seq), code] for seq, code in values]


# Runner ---------------------------------------------------------------------------

class CausalCameraRunner:
    """One synchronous run over one physical-order event iterator."""

    def __init__(self, seed: PlanarSeed | None, policy: PlanarPolicy, *, identity: RunIdentity,
                 limits: RunnerLimits, expected: ExpectedImages, localize: Callable[..., PlaceEstimate],
                 wire_sink: Callable[[dict], Any], diagnostic_sink: Callable[[str], Any],
                 finalize_sink: Callable[[dict], Any], failure_sink: Callable[[dict], Any],
                 clock: Callable[[], int]):
        if seed is None:
            raise NotReady("missing_initial_seed")
        if type(seed) is not PlanarSeed or type(policy) is not PlanarPolicy:
            raise TypeError("seed and policy must be the accepted PlanarSeed and PlanarPolicy")
        if (type(identity) is not RunIdentity or type(limits) is not RunnerLimits
                or type(expected) is not ExpectedImages):
            raise TypeError("identity, limits and expected must be RunIdentity, RunnerLimits and ExpectedImages")
        for name, sink in (("localize", localize), ("wire_sink", wire_sink), ("diagnostic_sink", diagnostic_sink),
                           ("finalize_sink", finalize_sink), ("failure_sink", failure_sink), ("clock", clock)):
            if not callable(sink):
                raise TypeError(f"{name} must be callable")
        admit_identifier(seed.map_frame, "seed map_frame", 80)
        if seed.map_frame != identity.map_frame:
            raise ValueError("seed map_frame must equal the pinned map_frame exactly")
        if policy.expected_map_id != identity.map_id:
            raise ValueError("policy expected_map_id must equal the pinned map_id")
        context = policy.arrival_context
        if context is None or not context.log_comparable:
            raise ValueError("a declared comparable_to_image_log arrival context is required; no fallback")
        if context.declaration_ref != identity.convention_ref:
            raise ValueError("arrival context declaration_ref must equal the pinned convention_ref")
        if expected.prefix_ns[0] < seed.timestamp_ns:
            raise ValueError("the expected prefix must start at or after the seed")
        if len(expected.prefix_ns) > limits.max_images:
            raise ValueError("expected prefix exceeds max_images")
        if len(expected.output_ns) > limits.max_expected_outputs:
            raise ValueError("expected output exceeds max_expected_outputs")
        self._predictor = Predictor(seed, policy, source_pins=dict(identity.source_pins))
        self._seed, self._identity, self._limits, self._expected = seed, identity, limits, expected
        self._localize, self._wire_sink, self._diagnostic_sink = localize, wire_sink, diagnostic_sink
        self._finalize_sink, self._failure_sink, self._clock = finalize_sink, failure_sink, clock
        self._state = "ready"
        self._events = self._images = self._pre_seed = self._motion = self._rejected = 0
        self._processed = self._emitted = self._diag_bytes = 0
        self._status_counts = {status: 0 for status in ("predicted", "approximate_corrected", "held", "lost")}
        self._last_clock: int | None = None
        self._last_diag_return: int | None = None

    @property
    def state(self) -> str:
        return self._state

    def _now(self) -> int:
        try:
            value = self._clock()
        except Exception as exc:
            raise _Stop("clock_fault", f"{type(exc).__name__}: {exc}", exc) from exc
        if type(value) is not int or not 0 <= value <= _INT64_MAX:
            raise _Stop("clock_fault", "clock must return an exact nonnegative int")
        if self._last_clock is not None and value < self._last_clock:
            raise _Stop("clock_fault", "clock must not decrease")
        self._last_clock = value
        return value

    def run(self, events: Iterable[Any]) -> RunSummary:
        if self._state != "ready":
            raise RuntimeError("a runner instance runs once")
        self._state = "running"
        try:
            summary = self._loop(events)
        except _Stop as stop:
            self._fail(stop.code, stop.message, stop.original)
        except Exception as exc:  # any unexpected fault still stops the run explicitly
            self._fail("internal_error", f"{type(exc).__name__}: {exc}", exc)
        self._state = "complete"
        return summary

    def _fail(self, code: str, message: str, original: BaseException | None):
        self._state = "failed"
        failure = RunnerFailure(code, message, original)
        payload = {"schema": FAILURE_SCHEMA, "run_id": self._identity.run_id, "code": code,
                   "message": message[:_MAX_MESSAGE], "events_consumed": str(self._events),
                   "images_processed": str(self._processed), "rows_emitted": str(self._emitted)}
        try:
            self._failure_sink(payload)
        except Exception as sink_error:  # the original failure is never replaced
            failure.failure_sink_error = sink_error
        raise failure from original

    def _loop(self, events) -> RunSummary:
        prefix, output_index = self._expected.prefix_ns, self._expected.output_index
        limits, identity = self._limits, self._identity
        try:
            iterator = iter(events)
        except TypeError as exc:
            raise _Stop("event_source_error", "events must be iterable", exc) from exc
        last_seq = last_log = None
        batch: list[MotionSample] = []
        projection: list[tuple[int, str]] = []
        while self._processed < len(prefix):
            try:
                event = next(iterator)
            except StopIteration:
                raise _Stop("missing_expected_image",
                            f"event source ended before expected image {prefix[self._processed]}") from None
            except Exception as exc:
                raise _Stop("event_source_error", f"{type(exc).__name__}: {exc}", exc) from exc
            supplied = self._now()
            self._events += 1
            if self._events > limits.max_events:
                raise _Stop("budget_events", f"more than {limits.max_events} events")
            if type(event) not in _EVENT_TYPES:
                raise _Stop("unknown_event_type", f"unsupported event type {type(event).__name__}")
            if last_seq is not None and event.seq <= last_seq:
                raise _Stop("sequence_order_reversed", f"event seq {event.seq} does not follow {last_seq}")
            if last_log is not None and event.log_ns < last_log:
                raise _Stop("log_order_reversed", f"event log_ns {event.log_ns} precedes {last_log}")
            last_seq, last_log = event.seq, event.log_ns
            if type(event) is UnsupportedSource:
                raise _Stop("unsupported_source", f"unsupported motion source: {event.code}")
            if type(event) in (MotionSample, ProjectionRejected):
                if len(batch) + len(projection) >= limits.max_pending_inputs:
                    raise _Stop("budget_pending_inputs", f"more than {limits.max_pending_inputs} pending inputs")
                if type(event) is MotionSample:
                    batch.append(event)
                    self._motion += 1
                else:
                    projection.append((event.seq, event.code))
                    self._rejected += 1
                continue
            self._images += 1
            if self._images > limits.max_images:
                raise _Stop("budget_images", f"more than {limits.max_images} image events")
            if len(event.encoded_image) > limits.max_encoded_image_bytes:
                raise _Stop("budget_image_bytes", f"image {event.exposure_ns} exceeds the encoded byte cap")
            if event.camera_id != identity.camera_id:
                raise _Stop("camera_mismatch", "image camera_id differs from the pinned camera")
            stamp = event.exposure_ns
            if self._processed == 0 and stamp < self._seed.timestamp_ns:
                self._pre_seed += 1  # skipped: no vision, no step, no row; the batch is kept
                continue
            self._check_order(stamp, prefix)
            self._process(event, supplied, batch, projection, self._processed >= output_index)
            batch, projection = [], []
        summary = RunSummary(identity.run_id, self._events, self._pre_seed, self._motion, self._rejected,
                             self._processed, self._emitted, tuple(self._status_counts.items()),
                             self._last_diag_return)
        if self._processed != len(prefix) or self._emitted != len(self._expected.output_ns):
            raise _Stop("count_mismatch", "processed or emitted count differs from the frozen manifests")
        try:
            self._finalize_sink(summary.record())
        except Exception as exc:
            raise _Stop("finalize_sink_error", f"{type(exc).__name__}: {exc}", exc) from exc
        return summary

    def _check_order(self, stamp: int, prefix: tuple[int, ...]):
        if self._processed:
            previous = prefix[self._processed - 1]
            if stamp == previous:
                raise _Stop("duplicate_image", f"image {stamp} was already processed")
            if stamp < previous:
                raise _Stop("image_order_reversed", f"image {stamp} precedes processed image {previous}")
        expected = prefix[self._processed]
        if stamp == expected:
            return
        if stamp in prefix:
            raise _Stop("missing_expected_image", f"expected image {expected} is absent before {stamp}")
        raise _Stop("extra_image", f"image {stamp} is not in the frozen processed-prefix manifest")

    def _process(self, event: ImageEvent, supplied: int, batch, projection, in_output: bool):
        identity, stamp = self._identity, event.exposure_ns
        query_id = f"{identity.camera_id}:{stamp}"
        visual_started = self._now()
        try:
            place = self._localize(query_id=query_id, timestamp_ns=stamp, encoded_image=event.encoded_image,
                                   camera_id=identity.camera_id, calibration_id=identity.calibration_id)
        except Exception as exc:
            raise _Stop("visual_error", f"{type(exc).__name__}: {exc}", exc) from exc
        visual_completed = self._now()
        self._check_place(place, stamp, query_id)
        try:
            estimate = self._predictor.step(stamp, event.log_ns, event.seq, batch, place)
        except Exception as exc:
            raise _Stop("predictor_error", f"{type(exc).__name__}: {exc}", exc) from exc
        predictor_completed = self._now()
        output_position = None
        wire_returned = None
        if in_output:
            output_position = self._emitted
            if self._expected.output_ns[output_position] != stamp:
                raise _Stop("count_mismatch", "emitted image differs from the next expected output image")
            try:
                self._wire_sink(estimate.wire())
            except Exception as exc:
                raise _Stop("wire_sink_error", f"{type(exc).__name__}: {exc}", exc) from exc
            # Count the successful sink return before any further fallible operation (r1 R3).
            self._emitted += 1
            self._status_counts[estimate.status] += 1
            wire_returned = self._now()
        prefix_index = self._processed
        self._processed += 1
        record = {
            "schema": DIAGNOSTIC_SCHEMA, "run_id": identity.run_id, "prefix_index": str(prefix_index),
            "output_index": None if output_position is None else str(output_position),
            "timestamp_ns": str(stamp), "image_log_ns": str(event.log_ns), "image_seq": str(event.seq),
            "status": estimate.status, "wire_emitted": in_output, "correction": estimate.correction,
            "coverage_break_ns": None if estimate.coverage_break_ns is None else str(estimate.coverage_break_ns),
            "projection_rejections": _pairs(projection), "sample_rejections": _pairs(estimate.sample_rejections),
            "visual": {"status": place.status, "reason": place.reason[:_MAX_REASON],
                       "candidate_count": str(len(place.candidates)),
                       "coverage_complete": place.candidate_coverage_complete,
                       "selected_reference_id": place.selected_reference_id},
            "stages_ns": {"event_supplied": str(supplied), "visual_started": str(visual_started),
                          "visual_completed": str(visual_completed),
                          "predictor_completed": str(predictor_completed),
                          "wire_sink_returned": None if wire_returned is None else str(wire_returned)},
        }
        text = _canonical(record)
        self._diag_bytes += len(text.encode("ascii")) + 1
        if self._diag_bytes > self._limits.max_diagnostic_bytes:
            raise _Stop("budget_diagnostic_bytes", "diagnostics exceed max_diagnostic_bytes")
        try:
            self._diagnostic_sink(text)
        except Exception as exc:
            raise _Stop("diagnostic_sink_error", f"{type(exc).__name__}: {exc}", exc) from exc
        self._last_diag_return = self._now()

    def _check_place(self, place, stamp: int, query_id: str):
        identity = self._identity
        if type(place) is not PlaceEstimate:
            raise _Stop("visual_result_type", "visual callback must return a PlaceEstimate")
        if type(place.status) is not str or place.status not in _VISUAL_STATUSES:
            raise _Stop("visual_result_status", "unknown visual status")
        # Exact native integer before any equality (r1 R1): 1000.0 or True never stand in for an exposure.
        if type(place.timestamp_ns) is not int or not 0 <= place.timestamp_ns <= _INT64_MAX:
            raise _Stop("visual_result_type", "visual timestamp_ns must be an exact native int in range")
        if (place.timestamp_ns != stamp or place.query_id != query_id
                or place.calibration_id != identity.calibration_id or place.map_id != identity.map_id
                or place.feature_id != identity.feature_id):
            raise _Stop("visual_identity_mismatch", "visual result identity differs from the current image or pins")
        if type(place.reason) is not str or type(place.candidate_coverage_complete) is not bool:
            raise _Stop("visual_result_type", "visual reason or coverage flag has the wrong type")
        # Complete payload admission before output (r1 R4): only the bounded count is ever used.
        if type(place.candidates) is not tuple or len(place.candidates) > _MAX_REFERENCES:
            raise _Stop("visual_result_type", "visual candidates must be a native tuple of at most 64 entries")
        # Bounded reference identity for every visual status (r1 R2), checked before any row or diagnostic.
        selected = place.selected_reference_id
        if place.status == "rejected":
            if selected is not None:
                raise _Stop("visual_reference_invalid", "a rejected visual result must have no selected reference")
        elif type(selected) is not str or selected not in identity.reference_ids:
            raise _Stop("visual_reference_unknown", "selected reference is not an exact pinned reference identity")

"""Finite multiwindow fit-only training preparation adapter (PREP-01).

Training-only and coordinator-run. It prepares one declared map sector from
exact predeclared half-open exposure windows inside the accepted fit interval
and returns the accepted MAP-01 ``TrainingExport`` in process, together with a
deterministic geometry-only keyframe/overlay selection, length- and
hash-verified encoded image bytes and separate JSON-safe diagnostics. It never
decodes an image, extracts features, builds a map, reads query/audit data or
scores anything.

Rules enforced here:

* Every budget is finite and checked before the work it bounds: decoded /tf
  messages, transform headers, frames-index bytes and rows, candidate
  exposures, keyframes and decodes, selected encoded image bytes and one shared
  monotonic deadline. The /tf iterator is never advanced once a limit is
  reached, and pose values, image bytes and decodes are never touched past a
  limit.
* /tf header order is never inferred from MCAP log order. Header coverage is
  complete only when the finite source is observed exhausted within budget. No
  unread-data timing assumption is approved, so a first header beyond a window,
  or a capped prefix in which every target bracket happens to be present, is
  not a completion certificate. Reaching a cap yields an explicit incomplete
  result, never a partial export.
* Pose values are read only by the accepted ``pose_samples_from_tf`` and only
  inside per-window pose ranges ``[start - gap, end + gap)`` clipped to the
  accepted fit, where ``gap`` is the accepted exporter's maximum interpolation
  gap. Values between windows, before/after the fit and in tuning/audit time
  stay unread. Interpolation is done only by ``export_fit_references``.
* Every in-window exposure must become an accepted reference. Any missing
  bracket, oversized gap, barrier or conflict makes the result incomplete.
* Recording identity is either a fresh full-file SHA256 or an explicitly
  approved reused acquisition identity with exact declared checks; the two
  are reported differently and hashing work is counted.

Times are exact integer nanoseconds throughout.
"""
from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass, fields
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Callable, Iterable, Iterator, Mapping, Protocol, Sequence

from . import map_contracts as contracts
from . import official_io
from . import official_training as training
from .map_contracts import FitInterval, validate_timestamp_ns


POLICY_SCHEMA = "schubki-prep-sector-policy-v1"
RESULT_SCHEMA = "schubki-prep-sector-result-v1"
MiB = 1024 * 1024
HASH_CHUNK_BYTES = 4 * MiB

COMPLETE = "complete"
INCOMPLETE = "incomplete"
INCOMPLETE_REASONS = (
    "deadline_exhausted",
    "index_unavailable",
    "index_bytes_cap_exhausted",
    "index_hash_mismatch",
    "index_row_cap_exhausted",
    "index_row_malformed",
    "index_order_violation",
    "candidate_cap_exhausted",
    "window_without_exposures",
    "recording_unavailable",
    "recording_identity_mismatch",
    "tf_message_cap_exhausted",
    "tf_header_cap_exhausted",
    "tf_message_malformed",
    "tf_header_malformed",
    "tf_header_reread_mismatch",
    "target_bracket_incomplete",
    "insufficient_window_overlap",
    "overlay_candidates_insufficient",
    "selected_bytes_cap_exhausted",
    "image_unavailable",
    "image_integrity_mismatch",
)

COVERAGE_SOURCE_EXHAUSTED = "source_exhausted_within_budget"
REGION_DEFINITION = "body_T_WB_xy_euclidean_distance_to_anchor_at_most_radius"
KEYFRAME_RULE = "per_window_in_region_xy_arc_length_uniform_targets_nearest_then_earliest"
OVERLAY_RULE = "non_keyframe_in_region_nearest_anchor_then_earliest"
POSE_SCOPE_RULE = "per_window_start_minus_gap_to_end_plus_gap_clipped_to_fit_merged"


def _int(value: Any, name: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer of at least {minimum}")
    return int(value)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@lru_cache(maxsize=1)
def module_identity() -> str:
    """SHA256 of this module's exact source bytes."""
    return training.module_sha256(sys.modules[__name__])


# ---------------------------------------------------------------------------
# Declared policy


@dataclass(frozen=True, slots=True)
class PrepWindow:
    """One predeclared half-open training exposure window ``start <= t < end``."""

    window_id: str
    start_ns: int
    end_ns: int

    def __post_init__(self) -> None:
        training._text(self.window_id, "window ID")
        start, end = validate_timestamp_ns(self.start_ns), validate_timestamp_ns(self.end_ns)
        if start >= end:
            raise ValueError("Window must be a nonempty half-open interval")
        object.__setattr__(self, "start_ns", start)
        object.__setattr__(self, "end_ns", end)

    def contains(self, timestamp_ns: int) -> bool:
        return self.start_ns <= validate_timestamp_ns(timestamp_ns) < self.end_ns

    def as_dict(self) -> dict[str, Any]:
        return {"window_id": self.window_id, "start_ns": self.start_ns, "end_ns": self.end_ns}


@dataclass(frozen=True, slots=True)
class ApprovedRecordingReuse:
    """Explicitly approved reuse of a previously verified recording SHA256.

    The declared checks are exact: the declared SHA must equal the accepted
    split's recording, the file size from stat must equal ``size_bytes``, and
    the camera cache manifest must declare the same SHA and byte size. No byte
    of the recording is hashed.
    """

    sha256: str
    size_bytes: int
    approval_ref: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "sha256", training._sha(self.sha256, "reused recording SHA256"))
        object.__setattr__(self, "size_bytes", _int(self.size_bytes, "reused recording size", 1))
        training._text(self.approval_ref, "recording reuse approval reference")

    def as_dict(self) -> dict[str, Any]:
        return {"sha256": self.sha256, "size_bytes": self.size_bytes, "approval_ref": self.approval_ref}


@dataclass(frozen=True, slots=True)
class SectorPolicy:
    """Exact windows, sector geometry and finite engineering budgets.

    Defaults are the brief's first-sector gate budgets; they are engineering
    limits, not score targets. Header, index-row and index-byte caps have no
    default and must be declared.
    """

    split_sha256: str
    windows: tuple[PrepWindow, ...]
    anchor_xy_m: tuple[float, float]
    max_transform_headers: int
    max_index_rows: int
    max_index_bytes: int
    camera_id: str = training.DEFAULT_CAMERA_ID
    region_radius_m: float = 50.0
    max_window_duration_ns: int = 6_000_000_000
    max_candidate_exposures: int = 120
    max_keyframes_per_window: int = 23
    max_keyframes_total: int = 46
    min_keyframes_per_window: int = 2
    overlay_decodes: int = 2
    max_decodes_total: int = 48
    max_selected_encoded_bytes: int = 12 * MiB
    max_tf_messages: int = 50_000
    deadline_ns: int = 300_000_000_000
    recording_reuse: ApprovedRecordingReuse | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "split_sha256", training._sha(self.split_sha256, "split SHA256"))
        training._text(self.camera_id, "camera ID")
        try:
            windows = tuple(self.windows)
            anchor = tuple(self.anchor_xy_m)
        except TypeError as exc:
            raise ValueError("Windows and anchor must be sequences") from exc
        if not windows or any(not isinstance(item, PrepWindow) for item in windows):
            raise ValueError("Policy needs at least one PrepWindow")
        if len({item.window_id for item in windows}) != len(windows):
            raise ValueError("Window IDs must be unique")
        if any(a.end_ns > b.start_ns for a, b in zip(windows, windows[1:])):
            raise ValueError("Windows must be declared in time order without overlap")
        if len(anchor) != 2:
            raise ValueError("Anchor must be (x, y) metres")
        anchor = tuple(contracts._finite(value, "anchor coordinate") for value in anchor)
        radius = contracts._finite(self.region_radius_m, "region radius")
        if radius <= 0:
            raise ValueError("Region radius must be positive")
        for name in ("max_transform_headers", "max_index_rows", "max_index_bytes", "max_window_duration_ns",
                     "max_candidate_exposures", "max_keyframes_per_window", "max_keyframes_total",
                     "min_keyframes_per_window", "max_decodes_total", "max_selected_encoded_bytes",
                     "max_tf_messages", "deadline_ns"):
            object.__setattr__(self, name, _int(getattr(self, name), name, 1))
        object.__setattr__(self, "overlay_decodes", _int(self.overlay_decodes, "overlay_decodes", 0))
        if any(item.end_ns - item.start_ns > self.max_window_duration_ns for item in windows):
            raise ValueError("A declared window is longer than the maximum window duration")
        if self.min_keyframes_per_window > self.max_keyframes_per_window:
            raise ValueError("Minimum keyframes per window exceeds the per-window cap")
        if len(windows) * self.max_keyframes_per_window > self.max_keyframes_total:
            raise ValueError("Per-window keyframe caps exceed the total keyframe cap")
        if self.max_keyframes_total + self.overlay_decodes > self.max_decodes_total:
            raise ValueError("Keyframe and overlay decodes exceed the total decode budget")
        if self.recording_reuse is not None and not isinstance(self.recording_reuse, ApprovedRecordingReuse):
            raise ValueError("Recording reuse must be an ApprovedRecordingReuse or None")
        object.__setattr__(self, "windows", windows)
        object.__setattr__(self, "anchor_xy_m", anchor)
        object.__setattr__(self, "region_radius_m", radius)

    def as_dict(self) -> dict[str, Any]:
        record: dict[str, Any] = {}
        for item in fields(self):
            value = getattr(self, item.name)
            if item.name == "windows":
                value = [window.as_dict() for window in value]
            elif item.name == "anchor_xy_m":
                value = list(value)
            elif item.name == "recording_reuse":
                value = None if value is None else value.as_dict()
            record[item.name] = value
        return {"schema": POLICY_SCHEMA, **record}

    @property
    def identity(self) -> str:
        return contracts.canonical_identity(self.as_dict())


def policy_from_packet(data: Mapping[str, Any]) -> SectorPolicy:
    """Strict inverse of ``SectorPolicy.as_dict``: every key exactly once, no defaults."""
    if not isinstance(data, Mapping) or data.get("schema") != POLICY_SCHEMA:
        raise ValueError("Unsupported sector policy packet schema")
    expected = {"schema", *(item.name for item in fields(SectorPolicy))}
    if set(data) != expected:
        missing, extra = sorted(expected - set(data)), sorted(set(data) - expected)
        raise ValueError(f"Policy packet keys differ: missing {missing}, unexpected {extra}")
    values = {key: value for key, value in data.items() if key != "schema"}
    windows = values["windows"]
    if not isinstance(windows, list) or any(not isinstance(item, dict) or set(item) != {"window_id", "start_ns", "end_ns"}
                                            for item in windows):
        raise ValueError("Packet windows must be objects with window_id, start_ns and end_ns")
    values["windows"] = tuple(PrepWindow(**item) for item in windows)
    reuse = values["recording_reuse"]
    if reuse is not None:
        if not isinstance(reuse, dict) or set(reuse) != {"sha256", "size_bytes", "approval_ref"}:
            raise ValueError("Packet recording_reuse must be null or an exact reuse object")
        values["recording_reuse"] = ApprovedRecordingReuse(**reuse)
    if not isinstance(values["anchor_xy_m"], list):
        raise ValueError("Packet anchor_xy_m must be a list")
    return SectorPolicy(**values)


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON object key {key!r}")
        result[key] = value
    return result


def _reject_constant(name: str) -> Any:
    raise ValueError(f"Non-finite JSON constant {name} is not allowed")


def strict_json_loads(raw: bytes | str) -> Any:
    """``json.loads`` that rejects duplicate object keys at every depth and NaN/Infinity.

    Ordinary ``json.loads`` keeps the last duplicate silently, which would let a
    packet declare one budget and enforce another.
    """
    return json.loads(raw, object_pairs_hook=_unique_object, parse_constant=_reject_constant)


def load_policy_packet(path: Any) -> tuple[SectorPolicy, str]:
    """Read a frozen coordinator packet once; parse and hash that same byte snapshot."""
    raw = Path(path).read_bytes()
    return policy_from_packet(strict_json_loads(raw)), _sha256_bytes(raw)


class Deadline:
    """Shared monotonic deadline for preparation and the later in-process build.

    Checked at operation boundaries; a single blocking source pull is not
    interrupted. ``clock`` returns integer monotonic nanoseconds.
    """

    __slots__ = ("budget_ns", "start_ns", "_clock")

    def __init__(self, budget_ns: int, clock: Callable[[], int] = time.monotonic_ns,
                 start_ns: int | None = None):
        self.budget_ns = _int(budget_ns, "deadline budget", 1)
        self._clock = clock
        self.start_ns = self.now_ns() if start_ns is None else validate_timestamp_ns(start_ns)

    def now_ns(self) -> int:
        return validate_timestamp_ns(self._clock())

    def elapsed_ns(self) -> int:
        return self.now_ns() - self.start_ns

    def remaining_ns(self) -> int:
        return self.budget_ns - self.elapsed_ns()

    def expired(self) -> bool:
        return self.elapsed_ns() >= self.budget_ns


# ---------------------------------------------------------------------------
# Sources


class ImageSource(Protocol):
    def size(self, source_id: str) -> int: ...

    def read(self, source_id: str, length: int) -> bytes: ...


class CacheImageSource:
    """Encoded image bytes from an official_io camera cache, contained to its root."""

    def __init__(self, manifest: Mapping[str, Any]):
        self._root = Path(manifest["cache_dir"]).resolve()

    def size(self, source_id: str) -> int:
        return training._contained(self._root, source_id).stat().st_size

    def read(self, source_id: str, length: int) -> bytes:
        """Read at most ``length`` bytes; the open file must have exactly that size before and after."""
        with training._contained(self._root, source_id).open("rb") as stream:
            if os.fstat(stream.fileno()).st_size != length:
                raise ValueError("Image size differs from the declared index length")
            data = stream.read(length)
            if os.fstat(stream.fileno()).st_size != length:
                raise ValueError("Image size changed while reading")
        return data


def official_tf_messages(recording_path: Any) -> Iterator[tuple[int, Any]]:
    """Production /tf source through the accepted whitelisted reader (needs mcap).

    Each advance decodes one /tf message. Coordinator laptop only.
    """
    messages = official_io._iter_allowed_messages(Path(recording_path).resolve(), (training.POSE_TOPIC,))
    try:
        for topic, log_ns, message in messages:
            if topic != training.POSE_TOPIC:
                raise ValueError("Reader returned a non-/tf message")
            yield log_ns, message
    finally:
        messages.close()


# ---------------------------------------------------------------------------
# Results


@dataclass(frozen=True, slots=True)
class SectorSelection:
    """Deterministic geometry-only choice made before any decode or feature."""

    keyframes: tuple[tuple[str, tuple[str, ...]], ...]
    overlays: tuple[str, ...]
    selected_encoded_bytes: int

    def keyframe_ids(self) -> tuple[str, ...]:
        return tuple(reference_id for _, ids in self.keyframes for reference_id in ids)

    def as_dict(self) -> dict[str, Any]:
        return {"keyframes": {window: list(ids) for window, ids in self.keyframes},
                "overlays": list(self.overlays), "selected_encoded_bytes": self.selected_encoded_bytes}


@dataclass(frozen=True, slots=True)
class VerifiedImage:
    """Encoded bytes whose length and SHA256 matched the hash-checked index. Not decoded."""

    reference_id: str
    source_id: str
    sha256: str
    data: bytes

    def __post_init__(self) -> None:
        if not isinstance(self.data, bytes) or _sha256_bytes(self.data) != self.sha256:
            raise ValueError("Verified image bytes must match their SHA256")


@dataclass(frozen=True, slots=True)
class PreparationResult:
    """Complete results carry the accepted TrainingExport; incomplete ones carry none."""

    status: str
    reasons: tuple[str, ...]
    export: training.TrainingExport | None
    selection: SectorSelection | None
    images: tuple[VerifiedImage, ...]
    diagnostics_json: str

    def __post_init__(self) -> None:
        reasons = tuple(self.reasons)
        if any(reason not in INCOMPLETE_REASONS for reason in reasons) or len(set(reasons)) != len(reasons):
            raise ValueError("Unknown or repeated incomplete reason")
        if self.status == COMPLETE:
            if (reasons or not isinstance(self.export, training.TrainingExport)
                    or not isinstance(self.selection, SectorSelection)):
                raise ValueError("A complete result needs an export, a selection and no reasons")
        elif self.status == INCOMPLETE:
            if not reasons or self.export is not None or self.selection is not None or self.images:
                raise ValueError("An incomplete result needs reasons and exposes no export, selection or images")
        else:
            raise ValueError("Unknown preparation status")
        json.loads(self.diagnostics_json)
        object.__setattr__(self, "reasons", reasons)
        object.__setattr__(self, "images", tuple(self.images))

    @property
    def complete(self) -> bool:
        return self.status == COMPLETE

    @property
    def diagnostics(self) -> dict[str, Any]:
        """A fresh JSON-safe copy; the stored diagnostics cannot be mutated."""
        return json.loads(self.diagnostics_json)


class _Incomplete(Exception):
    def __init__(self, reason: str, **detail: Any):
        if reason not in INCOMPLETE_REASONS:
            raise ValueError(f"Undeclared incomplete reason {reason}")
        super().__init__(reason)
        self.reason, self.detail = reason, detail


def _check_deadline(deadline: Deadline, stage: str) -> None:
    if deadline.expired():
        raise _Incomplete("deadline_exhausted", stage=stage)


# ---------------------------------------------------------------------------
# Pose scope and coverage


def pose_ranges(windows: Sequence[PrepWindow], fit: FitInterval, gap_ns: int) -> tuple[FitInterval, ...]:
    """Header ranges that can affect an accepted reference in any window.

    An accepted bracket pair for t has left >= t - gap and right <= left + gap,
    so [start - gap, end + gap) per window, clipped to the fit, contains both
    brackets and any barrier or duplicate between them.
    """
    gap = validate_timestamp_ns(gap_ns)
    spans = sorted((max(fit.start_ns, item.start_ns - gap), min(fit.end_ns, item.end_ns + gap)) for item in windows)
    merged: list[list[int]] = []
    for start, end in spans:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return tuple(FitInterval(start, end, fit.lap_ids) for start, end in merged)


def _range_index(ranges: Sequence[FitInterval], starts: Sequence[int], stamp: int) -> int | None:
    position = bisect_right(starts, stamp) - 1
    if position >= 0 and stamp < ranges[position].end_ns:
        return position
    return None


class _HeaderInspectedOutOfScope:
    """Stands in for a read message with no selected header in any pose range.

    It keeps the accepted adapter's ordinal (message position) and log time, so
    sample IDs equal those of one pass over the full read sequence, without
    retaining messages that cannot contribute samples.
    """

    __slots__ = ()
    transforms: tuple = ()


_OUT_OF_SCOPE = _HeaderInspectedOutOfScope()


@dataclass(slots=True)
class _TfScan:
    entries: list[tuple[int, Any]]
    counts: dict[str, int]
    coverage_basis: str


def _scan_tf(source: Iterable[tuple[int, Any]], ranges: tuple[FitInterval, ...], policy: SectorPolicy,
             deadline: Deadline, diagnostics: dict[str, Any]) -> _TfScan:
    """Budgeted header pass over the whole finite source. Never touches transform values.

    Limits are checked before each advance, so the iterator is never advanced
    past a cap or the deadline. Header order is only counted, never used to
    stop: completion requires observing the source exhausted.
    """
    counts = dict.fromkeys(("messages_read", "headers_inspected", "other_frame_pair", "selected_headers",
                            "selected_in_pose_scope", "selected_outside_pose_scope", "selected_header_decreases",
                            "log_time_decreases", "messages_retained", "messages_out_of_scope"), 0)
    diagnostics["tf"] = {"counts": counts, "source_exhausted": False, "coverage_basis": None,
                         "pose_ranges": [[item.start_ns, item.end_ns] for item in ranges]}
    starts = [item.start_ns for item in ranges]
    entries: list[tuple[int, Any]] = []
    iterator = iter(source)
    latest_log = latest_header = None
    try:
        while True:
            _check_deadline(deadline, "tf_scan")
            if counts["messages_read"] >= policy.max_tf_messages:
                raise _Incomplete("tf_message_cap_exhausted", messages_read=counts["messages_read"])
            try:
                item = next(iterator)
            except StopIteration:
                diagnostics["tf"]["source_exhausted"] = True
                diagnostics["tf"]["coverage_basis"] = COVERAGE_SOURCE_EXHAUSTED
                return _TfScan(entries, counts, COVERAGE_SOURCE_EXHAUSTED)
            counts["messages_read"] += 1
            try:
                log_ns, message = item
                log_ns = validate_timestamp_ns(log_ns)
                transforms = message.transforms
                header_count = len(transforms)
            except (TypeError, ValueError, AttributeError) as exc:
                raise _Incomplete("tf_message_malformed", message=counts["messages_read"]) from exc
            if counts["headers_inspected"] + header_count > policy.max_transform_headers:
                raise _Incomplete("tf_header_cap_exhausted", headers_inspected=counts["headers_inspected"],
                                  next_message_headers=header_count)
            if latest_log is not None and log_ns < latest_log:
                counts["log_time_decreases"] += 1
            latest_log = log_ns if latest_log is None else max(latest_log, log_ns)
            in_scope = False
            for transform in transforms:
                counts["headers_inspected"] += 1
                try:
                    parent, child = str(transform.header.frame_id), str(transform.child_frame_id)
                except AttributeError as exc:
                    raise _Incomplete("tf_message_malformed", message=counts["messages_read"]) from exc
                if parent != training.WORLD_FRAME or child != training.BODY_FRAME:
                    counts["other_frame_pair"] += 1
                    continue
                counts["selected_headers"] += 1
                try:
                    stamp = training.ros_stamp_ns(transform.header.stamp)
                except (ValueError, AttributeError, TypeError) as exc:
                    raise _Incomplete("tf_header_malformed", message=counts["messages_read"]) from exc
                if latest_header is not None and stamp < latest_header:
                    counts["selected_header_decreases"] += 1
                latest_header = stamp if latest_header is None else max(latest_header, stamp)
                if _range_index(ranges, starts, stamp) is None:
                    counts["selected_outside_pose_scope"] += 1
                else:
                    counts["selected_in_pose_scope"] += 1
                    in_scope = True
            entries.append((log_ns, message if in_scope else _OUT_OF_SCOPE))
            counts["messages_retained" if in_scope else "messages_out_of_scope"] += 1
    finally:
        close = getattr(iterator, "close", None)
        if callable(close):
            close()


# ---------------------------------------------------------------------------
# Index and recording


@dataclass(slots=True)
class _IndexScan:
    exposures: list[training.CameraExposure]
    declared_bytes: dict[int, int]
    window_of: dict[int, str]
    index_sha256: str


def _scan_index(manifest: Mapping[str, Any], policy: SectorPolicy, deadline: Deadline,
                diagnostics: dict[str, Any]) -> _IndexScan:
    """Byte cap before hashing; full hash; every row validated within the row cap."""
    if not all(isinstance(manifest.get(key), str) for key in ("cache_dir", "frames_index_path")):
        raise ValueError("Camera cache manifest must declare cache_dir and frames_index_path")
    root = Path(manifest["cache_dir"]).resolve()
    index = training._contained(root, manifest["frames_index_path"])
    expected = training._sha(manifest.get("frames_index_sha256"), "frames index SHA256")
    counts = dict.fromkeys(("index_file_bytes", "index_bytes_read", "index_rows_declared", "index_rows_parsed",
                            "rows_before_first_window", "rows_between_windows", "rows_after_last_window",
                            "candidate_exposures"), 0)
    per_window = {item.window_id: {"rows": 0, "first_ns": None, "last_ns": None} for item in policy.windows}
    diagnostics["index"] = {"counts": counts, "windows": per_window, "frames_index_sha256": None}
    _check_deadline(deadline, "index")
    try:
        size = index.stat().st_size
    except OSError as exc:
        raise _Incomplete("index_unavailable") from exc
    counts["index_file_bytes"] = size
    if size > policy.max_index_bytes:
        raise _Incomplete("index_bytes_cap_exhausted", index_file_bytes=size)
    try:
        with index.open("rb") as stream:
            data = stream.read(policy.max_index_bytes + 1)
    except OSError as exc:
        raise _Incomplete("index_unavailable") from exc
    counts["index_bytes_read"] = len(data)
    if len(data) > policy.max_index_bytes:
        raise _Incomplete("index_bytes_cap_exhausted", index_bytes_read=len(data))
    digest = _sha256_bytes(data)
    if digest != expected:
        raise _Incomplete("index_hash_mismatch")
    diagnostics["index"]["frames_index_sha256"] = digest
    if data and not data.endswith(b"\n"):
        raise _Incomplete("index_row_malformed", row="final row lacks a newline")
    lines = data.split(b"\n")[:-1] if data else []
    counts["index_rows_declared"] = len(lines)
    if len(lines) > policy.max_index_rows:
        raise _Incomplete("index_row_cap_exhausted", index_rows=len(lines))
    windows = policy.windows
    starts = [item.start_ns for item in windows]
    exposures: list[training.CameraExposure] = []
    declared_bytes: dict[int, int] = {}
    window_of: dict[int, str] = {}
    previous = None
    for number, line in enumerate(lines, 1):
        _check_deadline(deadline, "index")
        try:
            row = strict_json_loads(line)
            if not isinstance(row, dict) or set(row) != {"timestamp_ns", "path", "sha256", "bytes"}:
                raise ValueError("Index row keys differ")
            stamp = validate_timestamp_ns(row["timestamp_ns"])
            path = training._text(row["path"], "image path")
            image_sha = training._sha(row["sha256"], "image SHA256")
            length = _int(row["bytes"], "image bytes", 1)
        except (ValueError, TypeError) as exc:
            raise _Incomplete("index_row_malformed", row=number) from exc
        counts["index_rows_parsed"] += 1
        if previous is not None and stamp <= previous:
            raise _Incomplete("index_order_violation", row=number)
        previous = stamp
        position = bisect_right(starts, stamp) - 1
        if position < 0:
            counts["rows_before_first_window"] += 1
            continue
        window = windows[position]
        if stamp >= window.end_ns:
            counts["rows_after_last_window" if position == len(windows) - 1 else "rows_between_windows"] += 1
            continue
        if len(exposures) >= policy.max_candidate_exposures:
            raise _Incomplete("candidate_cap_exhausted", candidates=len(exposures), row=number)
        exposures.append(training.CameraExposure(policy.camera_id, stamp, image_sha, path))
        declared_bytes[stamp], window_of[stamp] = length, window.window_id
        counts["candidate_exposures"] += 1
        record = per_window[window.window_id]
        record["rows"] += 1
        record["first_ns"] = stamp if record["first_ns"] is None else record["first_ns"]
        record["last_ns"] = stamp
    empty = [window_id for window_id, record in per_window.items() if record["rows"] == 0]
    if empty:
        raise _Incomplete("window_without_exposures", windows=empty)
    return _IndexScan(exposures, declared_bytes, window_of, digest)


def _recording_identity(path: Path, expected: str, reuse: ApprovedRecordingReuse | None,
                        manifest_source: Mapping[str, Any], deadline: Deadline,
                        diagnostics: dict[str, Any]) -> dict[str, Any]:
    record: dict[str, Any] = {"expected_sha256": expected, "method": None, "bytes_hashed": 0,
                              "approval_ref": None, "checks": [], "size_bytes": None, "sha256": None}
    diagnostics["recording"] = record
    _check_deadline(deadline, "recording")
    try:
        size = path.stat().st_size if path.is_file() else None
    except OSError:
        size = None
    if size is None:
        raise _Incomplete("recording_unavailable")
    record["size_bytes"] = size
    if reuse is not None:
        record.update(method="approved_reused_acquisition_identity", approval_ref=reuse.approval_ref,
                      checks=["declared_sha256_equals_split_recording", "stat_size_equals_declared",
                              "manifest_sha256_equals_declared", "manifest_bytes_equals_declared"])
        if (reuse.sha256 != expected or size != reuse.size_bytes or manifest_source.get("sha256") != reuse.sha256
                or manifest_source.get("bytes") != reuse.size_bytes):
            raise _Incomplete("recording_identity_mismatch", method=record["method"])
        record["sha256"] = reuse.sha256
        return record
    record.update(method="fresh_full_file_sha256", checks=["full_file_sha256_equals_split_recording"])
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            while True:
                _check_deadline(deadline, "recording_hash")
                block = stream.read(HASH_CHUNK_BYTES)
                if not block:
                    break
                digest.update(block)
                record["bytes_hashed"] += len(block)
    except OSError as exc:
        raise _Incomplete("recording_unavailable") from exc
    if record["bytes_hashed"] != size or digest.hexdigest() != expected:
        raise _Incomplete("recording_identity_mismatch", method=record["method"])
    record["sha256"] = expected
    return record


# ---------------------------------------------------------------------------
# Geometry-only selection


def _arc_length_pick(points: Sequence[tuple[float, float]], count: int) -> list[int]:
    """Indices nearest to ``count`` uniform XY arc-length targets; ties take the earlier index."""
    if len(points) <= count:
        return list(range(len(points)))
    arc = [0.0]
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        arc.append(arc[-1] + math.hypot(x1 - x0, y1 - y0))
    total = arc[-1]
    if count == 1 or total == 0.0:
        return [0]
    chosen: list[int] = []
    for step in range(count):
        target = total * step / (count - 1)
        position = bisect_left(arc, target)
        options = [index for index in (position - 1, position) if 0 <= index < len(arc)]
        pick = min(options, key=lambda index: (abs(arc[index] - target), index))
        if pick not in chosen:
            chosen.append(pick)
    return sorted(chosen)


def select_sector(references: Sequence[training.TrainingReference], window_of: Mapping[int, str],
                  policy: SectorPolicy) -> tuple[dict[str, Any], tuple[tuple[str, tuple[str, ...]], ...],
                                                 tuple[str, ...]]:
    """Deterministic keyframe/overlay choice from training body poses only."""
    ax, ay = policy.anchor_xy_m
    per_window: dict[str, list[tuple[int, str, float, float, float]]] = {item.window_id: [] for item in policy.windows}
    for item in references:
        frame = item.reference
        x, y = float(frame.T_WB[0, 3]), float(frame.T_WB[1, 3])
        distance = math.hypot(x - ax, y - ay)
        if distance <= policy.region_radius_m:
            per_window[window_of[frame.timestamp_ns]].append((frame.timestamp_ns, frame.reference_id, x, y, distance))
    keyframes, summary, chosen = [], {}, set()
    for window in policy.windows:
        rows = sorted(per_window[window.window_id])
        picks = _arc_length_pick([(row[2], row[3]) for row in rows], policy.max_keyframes_per_window)
        ids = tuple(rows[index][1] for index in picks)
        chosen.update(ids)
        keyframes.append((window.window_id, ids))
        summary[window.window_id] = {"in_region": len(rows), "keyframes": len(ids)}
    remaining = sorted((row[4], row[0], row[1]) for rows in per_window.values() for row in rows if row[1] not in chosen)
    overlays = tuple(row[2] for row in remaining[:policy.overlay_decodes])
    return summary, tuple(keyframes), overlays


# ---------------------------------------------------------------------------
# Entry point


def prepare_sector(*, policy: SectorPolicy, split: training.FitSplit, calibration: contracts.CameraCalibration,
                   manifest: Mapping[str, Any], recording_path: Any, deadline: Deadline,
                   tf_source: Callable[[Path], Iterable[tuple[int, Any]]] = official_tf_messages,
                   image_source: ImageSource | None = None, code_commit: str | None = None) -> PreparationResult:
    """Prepare one declared sector; returns an explicit complete or incomplete result.

    Declared-input inconsistencies raise ValueError before any budgeted work.
    Data-dependent limits, gaps, malformed rows/messages and integrity
    mismatches return ``status == "incomplete"`` with no TrainingExport. The
    exporter always uses the accepted default ExportConfig for the policy
    camera, so its interpolation gap and quaternion tolerance are not loosened.
    The caller keeps ``deadline`` for the later in-process build.
    """
    if not isinstance(policy, SectorPolicy) or not isinstance(split, training.FitSplit):
        raise ValueError("Preparation needs a SectorPolicy and an accepted FitSplit")
    if not isinstance(calibration, contracts.CameraCalibration) or not isinstance(deadline, Deadline):
        raise ValueError("Preparation needs a CameraCalibration and a shared Deadline")
    if not isinstance(manifest, Mapping):
        raise ValueError("Camera cache manifest must be a mapping")
    config = training.ExportConfig(camera_id=policy.camera_id)
    if policy.split_sha256 != split.split_sha256:
        raise ValueError("Policy is bound to a different accepted split")
    fit = split.fit
    if any(item.start_ns < fit.start_ns or item.end_ns > fit.end_ns for item in policy.windows):
        raise ValueError("Every declared window must lie inside the accepted fit interval")
    if not policy.camera_id == config.camera_id == calibration.camera_id == manifest.get("camera"):
        raise ValueError("Policy, export, calibration and camera cache must name the same camera")
    if manifest.get("schema") != official_io.SCHEMA:
        raise ValueError("Unsupported camera cache schema")
    source = manifest.get("source")
    if not isinstance(source, Mapping) or source.get("sha256") != split.source_recording_sha256:
        raise ValueError("Camera cache belongs to a different recording than the accepted split")
    if deadline.budget_ns != policy.deadline_ns:
        raise ValueError("Shared deadline budget differs from the declared policy deadline")
    calibration_file_sha = training._sha(manifest.get("calibration_sha256"), "calibration file SHA256")
    code_commit = training.ExportProvenance(split.source_recording_sha256, split.source_recording_sha256,
                                            calibration_file_sha, code_commit).code_commit
    images_from = CacheImageSource(manifest) if image_source is None else image_source
    exporter_sha, contracts_sha = training.code_identities()
    ranges = pose_ranges(policy.windows, fit, config.max_interpolation_gap_ns)
    diagnostics: dict[str, Any] = {
        "schema": RESULT_SCHEMA, "status": None, "reasons": [], "stage": "inputs",
        "policy": policy.as_dict(), "policy_identity": policy.identity,
        "identities": {"bounded_training_sha256": module_identity(), "exporter_sha256": exporter_sha,
                       "map_contracts_sha256": contracts_sha, "split_sha256": split.split_sha256,
                       "fit_interval": [fit.start_ns, fit.end_ns, list(fit.lap_ids)],
                       "source_recording_sha256": split.source_recording_sha256,
                       "calibration_identity": calibration.identity,
                       "calibration_file_sha256": calibration_file_sha,
                       "export_config": config.as_dict(), "export_config_identity": config.identity,
                       "code_commit": code_commit},
        "rules": {"pose_scope": POSE_SCOPE_RULE, "region": REGION_DEFINITION, "keyframes": KEYFRAME_RULE,
                  "overlays": OVERLAY_RULE, "window_membership": "start_inclusive_end_exclusive"},
        "decodes": {"performed": 0, "reserved_keyframes": policy.max_keyframes_total,
                    "reserved_overlays": policy.overlay_decodes, "max_total": policy.max_decodes_total},
    }
    try:
        diagnostics["stage"] = "index"
        index = _scan_index(manifest, policy, deadline, diagnostics)
        diagnostics["stage"] = "recording"
        recording = _recording_identity(Path(recording_path), split.source_recording_sha256,
                                        policy.recording_reuse, source, deadline, diagnostics)
        diagnostics["stage"] = "tf_scan"
        scan = _scan_tf(tf_source(Path(recording_path)), ranges, policy, deadline, diagnostics)
        diagnostics["stage"] = "pose_samples"
        _check_deadline(deadline, "pose_samples")
        streams = [training.pose_samples_from_tf(iter(scan.entries), window=item) for item in ranges]
        diagnostics["tf"]["adapter_counts"] = [dict(stream.counts) for stream in streams]
        diagnostics["tf"]["adapter_counts_scope"] = ("one accepted pose_samples_from_tf pass per pose range over all "
                                                     "read messages; messages without an in-scope selected header "
                                                     "are passed as empty stand-ins, so adapter transform counts "
                                                     "cover retained messages only")
        samples = tuple(sample for stream in streams for sample in stream.samples)
        if len(samples) != scan.counts["selected_in_pose_scope"]:
            raise _Incomplete("tf_header_reread_mismatch", scanned=scan.counts["selected_in_pose_scope"],
                              adapter=len(samples))
        diagnostics["stage"] = "export"
        _check_deadline(deadline, "export")
        provenance = training.ExportProvenance(source["sha256"], recording["sha256"], calibration_file_sha,
                                               code_commit)
        export = training.export_fit_references(index.exposures, samples, split=split, calibration=calibration,
                                                provenance=provenance, config=config)
        diagnostics["export"] = {"counts": dict(export.counts),
                                 "in_fit_rejections": [list(item) for item in export.in_fit_rejections],
                                 "targets": len(index.exposures), "references_accepted": len(export.references)}
        if export.in_fit_rejections or len(export.references) != len(index.exposures):
            raise _Incomplete("target_bracket_incomplete", rejected=len(export.in_fit_rejections))
        diagnostics["export"].update(export_digest=export.export_digest,
                                     references_digest=export.references_digest,
                                     fit_pose_digest=export.fit_pose_digest)
        diagnostics["stage"] = "selection"
        _check_deadline(deadline, "selection")
        summary, keyframes, overlays = select_sector(export.references, index.window_of, policy)
        diagnostics["selection"] = {"windows": summary, "keyframes": {w: list(ids) for w, ids in keyframes},
                                    "overlays": list(overlays)}
        thin = [window for window, ids in keyframes if len(ids) < policy.min_keyframes_per_window]
        if thin:
            raise _Incomplete("insufficient_window_overlap", windows=thin)
        if len(overlays) < policy.overlay_decodes:
            raise _Incomplete("overlay_candidates_insufficient", available=len(overlays))
        by_id = {item.reference.reference_id: item for item in export.references}
        selected = sorted({*(rid for _, ids in keyframes for rid in ids), *overlays},
                          key=lambda rid: by_id[rid].reference.timestamp_ns)
        selected_bytes = sum(index.declared_bytes[by_id[rid].reference.timestamp_ns] for rid in selected)
        diagnostics["selection"]["selected_encoded_bytes"] = selected_bytes
        if selected_bytes > policy.max_selected_encoded_bytes:
            raise _Incomplete("selected_bytes_cap_exhausted", selected_encoded_bytes=selected_bytes)
        diagnostics["stage"] = "images"
        images = _verify_images([by_id[rid] for rid in selected], index.declared_bytes, images_from, policy,
                                deadline, diagnostics)
        diagnostics["images"]["unselected_references_bytes_unread"] = len(export.references) - len(selected)
        _check_deadline(deadline, "complete")
    except _Incomplete as incomplete:
        diagnostics.update(status=INCOMPLETE, reasons=[incomplete.reason], incomplete_detail=incomplete.detail,
                           deadline=_deadline_record(deadline))
        return PreparationResult(INCOMPLETE, (incomplete.reason,), None, None, (), _dumps(diagnostics))
    diagnostics.update(status=COMPLETE, stage="complete", deadline=_deadline_record(deadline))
    selection = SectorSelection(keyframes, overlays, selected_bytes)
    return PreparationResult(COMPLETE, (), export, selection, images, _dumps(diagnostics))


def _verify_images(selected: Sequence[training.TrainingReference], declared: Mapping[int, int],
                   source: ImageSource, policy: SectorPolicy, deadline: Deadline,
                   diagnostics: dict[str, Any]) -> tuple[VerifiedImage, ...]:
    """Stat, then bounded read, then length and SHA256 against the index. No decode."""
    record = {"verified": 0, "bytes_read": 0, "records": []}
    diagnostics["images"] = record
    images = []
    for item in selected:
        _check_deadline(deadline, "images")
        frame = item.reference
        length = declared[frame.timestamp_ns]
        if record["bytes_read"] + length > policy.max_selected_encoded_bytes:
            raise _Incomplete("selected_bytes_cap_exhausted", bytes_read=record["bytes_read"])
        try:
            size = source.size(item.image_source_id)
        except (OSError, ValueError) as exc:
            raise _Incomplete("image_unavailable", reference_id=frame.reference_id) from exc
        if size != length:
            raise _Incomplete("image_integrity_mismatch", reference_id=frame.reference_id, check="size")
        try:
            data = source.read(item.image_source_id, length)
        except OSError as exc:
            raise _Incomplete("image_unavailable", reference_id=frame.reference_id) from exc
        except ValueError as exc:
            raise _Incomplete("image_integrity_mismatch", reference_id=frame.reference_id, check="size") from exc
        if not isinstance(data, bytes):
            raise _Incomplete("image_integrity_mismatch", reference_id=frame.reference_id, check="type")
        record["bytes_read"] += len(data)
        if len(data) != length or _sha256_bytes(data) != frame.image_sha256:
            raise _Incomplete("image_integrity_mismatch", reference_id=frame.reference_id, check="sha256")
        images.append(VerifiedImage(frame.reference_id, item.image_source_id, frame.image_sha256, data))
        record["verified"] += 1
        record["records"].append({"reference_id": frame.reference_id, "source_id": item.image_source_id,
                                  "sha256": frame.image_sha256, "bytes": length})
    return tuple(images)


def _deadline_record(deadline: Deadline) -> dict[str, Any]:
    elapsed = deadline.elapsed_ns()
    return {"budget_ns": deadline.budget_ns, "elapsed_ns": elapsed, "remaining_ns": deadline.budget_ns - elapsed,
            "clock": "monotonic_ns", "shared_with_build": True}


def _dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)

"""TEAM-EVAL-01: strict causal_planar_v1 validation and write-once CSV/status export.

Standard library only. This module independently implements the frozen shared
wire in team/teammates/WIRE_V1.md. It imports no motion, vision or demo source,
copies no official scorer and runs no estimator, recording or live runner.

Three caller-named outputs are produced for one run:

* CSV ``timestamp_ns,x,y,yaw``: exact original integer exposure nanoseconds and
  positional (never scientific) shortest-roundtrip metres/radians, LF, ASCII.
* status sidecar (JSON Lines): a header, one exact canonical wire record per
  row and a trailer. Every integer is a canonical decimal string.
* completion manifest: staged exclusively as ``<name>.staging`` only after both
  files are durably closed and re-hashed, itself written/flushed/fsynced/closed,
  then published under the final name without replacing anything. Absence of
  the final name means the run is not complete.

Nothing is imputed, reseeded, wrapped, repaired or rewritten. Rejected or
interrupted runs keep their partial files and a failure trailer and never
receive a completion manifest. Presence coverage includes held/lost rows and is
not fresh tracking or accuracy credit.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import unicodedata
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

WIRE_SCHEMA = "causal_planar_v1"
EXPORT_SCHEMA = "causal_csv_status_v1"
SIDECAR_SCHEMA = EXPORT_SCHEMA + "/status"
COMPLETION_SCHEMA = EXPORT_SCHEMA + "/completion"
INT64_MAX = (1 << 63) - 1
CSV_HEADER = ("timestamp_ns", "x", "y", "yaw")
CSV_HEADER_LINE = ",".join(CSV_HEADER) + "\n"

STATUSES = ("predicted", "approximate_corrected", "held", "lost")
STALE_STATUSES = frozenset({"held", "lost"})
ANCHOR_KINDS = ("organizer_seed", "approximate_reference_pose")

FIELDS = (
    "schema", "timestamp_ns", "image_log_ns", "image_seq", "x", "y", "yaw", "map_frame",
    "status", "last_anchor_ns", "anchor_age_ns", "anchor_kind", "latest_source_ns",
    "latest_log_ns", "latest_input_seq", "motion_age_ns", "visual_reference", "reason",
    "source_pins",
)
_FIELD_SET = frozenset(FIELDS)
NS_FIELDS = ("timestamp_ns", "image_log_ns", "image_seq", "last_anchor_ns", "anchor_age_ns")
NULLABLE_NS_FIELDS = ("latest_source_ns", "latest_log_ns", "latest_input_seq", "motion_age_ns")
MOTION_FIELDS = ("latest_source_ns", "latest_log_ns", "latest_input_seq")
POSE_FIELDS = ("x", "y", "yaw")

VISUAL_REFERENCE_FIXED: Mapping[str, Any] = {
    "pose_kind": "approximate_reference_pose",
    "position_source": "training_reference",
    "heading_source": "training_reference",
    "yaw_approximate": True,
    "translation_scale_observed": False,
}
VISUAL_REFERENCE_FIELDS = ("reference_id", *VISUAL_REFERENCE_FIXED)
CLOCK_DOMAIN_KEYS = ("exposure", "arrival_log", "motion_source")

MAP_FRAME_MAX = 80
REASON_MAX = 160
REFERENCE_ID_MAX = 120
PIN_NAME_MAX = 80
MAX_SOURCE_PINS = 16
CLOCK_LABEL_MAX = 80
FAILURE_MESSAGE_MAX = 1000
COMPLETION_STAGING_SUFFIX = ".staging"  # staged manifest beside the final completion name, same parent

COMPLETION_NOTES = (
    "presence_rows includes held and lost rows; it is not fresh tracking or accuracy credit",
    "predicted and approximate_corrected describe an approximate planar process, not measured metric fixes",
    "this export makes no official score, timing, latency or real-data claim",
)

_CANONICAL_DECIMAL = re.compile(r"0|[1-9][0-9]{0,18}")
_CSV_FLOAT = re.compile(r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,79}")
_FAIL_CODE = re.compile(r"[a-z][a-z0-9_]{0,39}")
_FORBIDDEN_CATEGORIES = frozenset({"Cc", "Cs"})


class CausalExportError(ValueError):
    """Explicit rejection with a stable machine-readable ``code``."""

    def __init__(self, code: str, message: str, *, index: int | None = None, field: str | None = None,
                 details: Mapping[str, Any] | None = None) -> None:
        self.code = code
        self.message = message
        self.index = index
        self.field = field
        self.details = dict(details or {})
        where = "" if index is None else f"record {index}: "
        where += "" if field is None else f"{field}: "
        super().__init__(f"[{code}] {where}{message}")


# Primitive checks ----------------------------------------------------------------------

def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def _pretty_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, indent=2, ensure_ascii=True, allow_nan=False) + "\n"


def _reject_constant(name: str) -> Any:
    raise CausalExportError("json", f"nonstandard JSON constant {name} is not allowed")


def _unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CausalExportError("json", f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def strict_json_loads(text: str) -> Any:
    """Parse JSON rejecting NaN/Infinity constants and duplicate object keys."""
    if type(text) is not str:
        raise CausalExportError("type", "JSON input must be str")
    try:
        return json.loads(text, parse_constant=_reject_constant, object_pairs_hook=_unique_pairs)
    except CausalExportError:
        raise
    except (ValueError, RecursionError) as exc:
        raise CausalExportError("json", f"malformed JSON: {exc}") from None


def _ns(value: Any, field: str, index: int | None = None) -> int:
    if type(value) is not int:
        raise CausalExportError("type", f"must be a native non-bool int, got {type(value).__name__}",
                                index=index, field=field)
    if not 0 <= value <= INT64_MAX:
        raise CausalExportError("range", "must be within [0, 2^63-1]", index=index, field=field)
    return value


def parse_decimal_ns(text: Any, field: str = "timestamp_ns", index: int | None = None) -> int:
    """Parse one canonical decimal string ``0|[1-9][0-9]*`` within signed int64."""
    if type(text) is not str:
        raise CausalExportError("type", f"must be a canonical decimal string, got {type(text).__name__}",
                                index=index, field=field)
    if _CANONICAL_DECIMAL.fullmatch(text) is None:
        raise CausalExportError("decimal", f"{text[:40]!r} is not a canonical decimal 0|[1-9][0-9]*",
                                index=index, field=field)
    value = int(text)
    if value > INT64_MAX:
        raise CausalExportError("range", "must be within [0, 2^63-1]", index=index, field=field)
    return value


def format_decimal_ns(value: Any, field: str = "timestamp_ns") -> str:
    return str(_ns(value, field))


def _pose(value: Any, field: str, index: int | None) -> float:
    if type(value) is float:
        number = value
    elif type(value) is int:
        try:
            number = float(value)
        except OverflowError:
            raise CausalExportError("nonfinite", "integer pose value overflows float", index=index,
                                    field=field) from None
        if int(number) != value:
            raise CausalExportError("pose", "integer pose value is not exactly representable as float",
                                    index=index, field=field)
    else:
        raise CausalExportError("type", f"must be a finite float, got {type(value).__name__}",
                                index=index, field=field)
    if not math.isfinite(number):
        raise CausalExportError("nonfinite", "must be finite", index=index, field=field)
    return number


def _text(value: Any, field: str, max_length: int, index: int | None = None, *, identifier: bool = False,
          code: str = "text") -> str:
    if type(value) is not str:
        raise CausalExportError("type", f"must be str, got {type(value).__name__}", index=index, field=field)
    if not value.strip():
        raise CausalExportError(code, "must be nonempty", index=index, field=field)
    if len(value) > max_length:
        raise CausalExportError(code, f"must be at most {max_length} characters", index=index, field=field)
    if identifier and value != value.strip():
        raise CausalExportError(code, "must not have leading or trailing whitespace", index=index, field=field)
    for character in value:
        if unicodedata.category(character) in _FORBIDDEN_CATEGORIES:
            raise CausalExportError(code, f"contains control or surrogate character U+{ord(character):04X}",
                                    index=index, field=field)
    return value


def _pose_bits(record: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(float.hex(record[field]) for field in POSE_FIELDS)


def _copy_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Independent copy of a normalized record; its only nested values are the two small objects."""
    copied = dict(record)
    copied["visual_reference"] = None if record["visual_reference"] is None else dict(record["visual_reference"])
    copied["source_pins"] = dict(record["source_pins"])
    return copied


def _motion(record: Mapping[str, Any]) -> tuple[int, int, int] | None:
    values = tuple(record[field] for field in MOTION_FIELDS)
    return None if values[0] is None else values  # type: ignore[return-value]


def format_csv_float(value: Any) -> str:
    """Positional shortest-roundtrip text of a finite float; never scientific notation."""
    if type(value) is not float or not math.isfinite(value):
        raise CausalExportError("nonfinite", "CSV pose values must be finite floats")
    text = format(Decimal(repr(value)), "f")
    parsed = float(text)
    if parsed != value or math.copysign(1.0, parsed) != math.copysign(1.0, value):
        raise CausalExportError("pose", f"float {value!r} does not roundtrip through {text!r}")
    return text


def expected_timestamps_sha256(expected: Sequence[int]) -> str:
    """Same byte formula as the accepted EVAL-01 expected_timestamps_sha256."""
    return hashlib.sha256(("\n".join(str(value) for value in expected) + "\n").encode("ascii")).hexdigest()


def normalize_expected_timestamps(values: Iterable[Any]) -> tuple[int, ...]:
    """Exact expected image exposure list, already strictly increasing; never sorted silently."""
    if isinstance(values, (str, bytes, bytearray, Mapping)):
        raise CausalExportError("expected", "expected timestamps must be a sequence of native ints")
    stamps = tuple(_ns(value, "expected_timestamps", index) for index, value in enumerate(values))
    if not stamps:
        raise CausalExportError("expected", "expected-frame list is empty")
    for index in range(1, len(stamps)):
        if stamps[index] <= stamps[index - 1]:
            kind = "duplicated" if stamps[index] == stamps[index - 1] else "out of order"
            raise CausalExportError("expected", f"expected timestamp {stamps[index]} at position {index} is {kind}")
    return stamps


# Record validation ---------------------------------------------------------------------

def _visual_reference(value: Any, index: int | None) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise CausalExportError("type", "must be null or an object", index=index, field="visual_reference")
    keys = set(value)
    if keys != set(VISUAL_REFERENCE_FIELDS):
        raise CausalExportError("visual_reference",
                                f"exact fields required; missing {sorted(set(VISUAL_REFERENCE_FIELDS) - keys)}, "
                                f"unknown {sorted(str(key) for key in keys - set(VISUAL_REFERENCE_FIELDS))}",
                                index=index, field="visual_reference")
    reference_id = _text(value["reference_id"], "visual_reference.reference_id", REFERENCE_ID_MAX, index,
                         identifier=True, code="visual_reference")
    for key, fixed in VISUAL_REFERENCE_FIXED.items():
        item = value[key]
        valid = item is fixed if isinstance(fixed, bool) else (type(item) is str and item == fixed)
        if not valid:
            raise CausalExportError("visual_reference", f"{key} must be exactly {fixed!r}", index=index,
                                    field="visual_reference")
    return {"reference_id": reference_id, **VISUAL_REFERENCE_FIXED}


def _source_pins(value: Any, index: int | None) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise CausalExportError("type", "must be an object", index=index, field="source_pins")
    if len(value) > MAX_SOURCE_PINS:
        raise CausalExportError("source_pins", f"at most {MAX_SOURCE_PINS} pins allowed", index=index,
                                field="source_pins")
    pins: dict[str, str] = {}
    for name in value:
        _text(name, "source_pins name", PIN_NAME_MAX, index, identifier=True, code="source_pins")
        digest = value[name]
        if type(digest) is not str or _SHA256.fullmatch(digest) is None:
            raise CausalExportError("source_pins", f"pin {name!r} must be lowercase 64-hex SHA256", index=index,
                                    field="source_pins")
        pins[name] = digest
    return dict(sorted(pins.items()))


def validate_record(record: Any, index: int | None = None) -> dict[str, Any]:
    """Validate one native normalized record and return an independent normalized copy."""
    if not isinstance(record, Mapping):
        raise CausalExportError("type", f"record must be a mapping, got {type(record).__name__}", index=index)
    if any(type(key) is not str for key in record):
        raise CausalExportError("field_set", "record keys must be str", index=index)
    keys = set(record)
    if keys != _FIELD_SET:
        raise CausalExportError("field_set", f"exact wire fields required; missing {sorted(_FIELD_SET - keys)}, "
                                f"unknown {sorted(keys - _FIELD_SET)}", index=index)
    if type(record["schema"]) is not str or record["schema"] != WIRE_SCHEMA:
        raise CausalExportError("schema", f"must be exactly {WIRE_SCHEMA!r}", index=index, field="schema")
    out: dict[str, Any] = {"schema": WIRE_SCHEMA}
    for field in NS_FIELDS:
        out[field] = _ns(record[field], field, index)
    for field in NULLABLE_NS_FIELDS:
        value = record[field]
        out[field] = None if value is None else _ns(value, field, index)
    for field in POSE_FIELDS:
        out[field] = _pose(record[field], field, index)
    if not -math.pi <= out["yaw"] <= math.pi:
        raise CausalExportError("yaw_range", "must lie in [-pi, pi]; supplied yaw is never wrapped", index=index,
                                field="yaw")
    out["map_frame"] = _text(record["map_frame"], "map_frame", MAP_FRAME_MAX, index, identifier=True)
    status = record["status"]
    if type(status) is not str or status not in STATUSES:
        raise CausalExportError("status", f"unknown status {status!r}", index=index, field="status")
    out["status"] = status
    anchor_kind = record["anchor_kind"]
    if type(anchor_kind) is not str or anchor_kind not in ANCHOR_KINDS:
        raise CausalExportError("anchor", f"unknown anchor_kind {anchor_kind!r}", index=index, field="anchor_kind")
    out["anchor_kind"] = anchor_kind
    out["visual_reference"] = _visual_reference(record["visual_reference"], index)
    out["reason"] = _text(record["reason"], "reason", REASON_MAX, index)
    out["source_pins"] = _source_pins(record["source_pins"], index)

    stamp = out["timestamp_ns"]
    if out["last_anchor_ns"] > stamp:
        raise CausalExportError("future_provenance", "last_anchor_ns is after timestamp_ns", index=index,
                                field="last_anchor_ns")
    if out["anchor_age_ns"] != stamp - out["last_anchor_ns"]:
        raise CausalExportError("age", "anchor_age_ns must equal timestamp_ns - last_anchor_ns", index=index,
                                field="anchor_age_ns")
    present = [out[field] is not None for field in MOTION_FIELDS]
    if not any(present):
        # WIRE_V1 makes the motion fields nullable together for every status; a producer's stricter
        # fresh-motion policy belongs to that producer, not to this generic exporter.
        if out["motion_age_ns"] is not None:
            raise CausalExportError("age", "motion_age_ns must be null without used motion", index=index,
                                    field="motion_age_ns")
    elif not all(present):
        raise CausalExportError("motion", "latest_source_ns, latest_log_ns and latest_input_seq are null together",
                                index=index)
    else:
        if out["latest_source_ns"] > stamp:
            raise CausalExportError("future_provenance", "motion source time is after the image", index=index,
                                    field="latest_source_ns")
        if out["latest_log_ns"] > out["image_log_ns"]:
            raise CausalExportError("future_provenance", "motion arrived after the image", index=index,
                                    field="latest_log_ns")
        if out["latest_input_seq"] > out["image_seq"]:
            raise CausalExportError("future_provenance", "motion arrival sequence is after the image", index=index,
                                    field="latest_input_seq")
        if out["motion_age_ns"] is None or out["motion_age_ns"] != stamp - out["latest_source_ns"]:
            raise CausalExportError("age", "motion_age_ns must equal timestamp_ns - latest_source_ns", index=index,
                                    field="motion_age_ns")
    if anchor_kind == "organizer_seed" and out["visual_reference"] is not None:
        raise CausalExportError("visual_reference", "organizer_seed anchor has null visual_reference", index=index,
                                field="visual_reference")
    if anchor_kind == "approximate_reference_pose" and out["visual_reference"] is None:
        raise CausalExportError("visual_reference", "approximate reference anchor requires its provenance object",
                                index=index, field="visual_reference")
    if status == "approximate_corrected" and (anchor_kind != "approximate_reference_pose" or
                                              out["last_anchor_ns"] != stamp):
        raise CausalExportError("anchor", "approximate_corrected is a current-image approximate reference anchor "
                                "(anchor_age_ns 0); delayed corrections are out of scope", index=index,
                                field="status")
    return {field: out[field] for field in FIELDS}


def _wire_from_normalized(record: Mapping[str, Any]) -> dict[str, Any]:
    wire = dict(record)
    for field in NS_FIELDS:
        wire[field] = str(record[field])
    for field in NULLABLE_NS_FIELDS:
        wire[field] = None if record[field] is None else str(record[field])
    wire["visual_reference"] = None if record["visual_reference"] is None else dict(record["visual_reference"])
    wire["source_pins"] = dict(record["source_pins"])
    return wire


def record_to_wire(record: Any, index: int | None = None) -> dict[str, Any]:
    """Native record to the JSON wire object; every ns/sequence becomes a canonical decimal string."""
    return _wire_from_normalized(validate_record(record, index))


def record_from_wire(obj: Any, index: int | None = None) -> dict[str, Any]:
    """Parse a decoded JSON wire object into a validated native record."""
    if not isinstance(obj, Mapping):
        raise CausalExportError("type", f"wire record must be an object, got {type(obj).__name__}", index=index)
    keys = set(obj)
    if keys != _FIELD_SET:
        raise CausalExportError("field_set", f"exact wire fields required; missing {sorted(_FIELD_SET - keys)}, "
                                f"unknown {sorted(str(key) for key in keys - _FIELD_SET)}", index=index)
    native = dict(obj)
    for field in NS_FIELDS:
        native[field] = parse_decimal_ns(obj[field], field, index)
    for field in NULLABLE_NS_FIELDS:
        native[field] = None if obj[field] is None else parse_decimal_ns(obj[field], field, index)
    return validate_record(native, index)


def dumps_wire(record: Any) -> str:
    return canonical_json(record_to_wire(record))


def loads_wire(text: str, index: int | None = None) -> dict[str, Any]:
    return record_from_wire(strict_json_loads(text), index)


# Run validation ------------------------------------------------------------------------

def classify_timestamps(observed: Sequence[int], expected: Sequence[int]) -> dict[str, int]:
    """Whole-list diagnostic counts: duplicate, out_of_order, extra and missing rows."""
    expected_set = set(expected)
    seen: set[int] = set()
    running: int | None = None
    counts = {"duplicate": 0, "out_of_order": 0, "extra": 0, "missing": 0}
    for stamp in observed:
        if stamp in seen:
            counts["duplicate"] += 1
            continue
        seen.add(stamp)
        if running is not None and stamp < running:
            counts["out_of_order"] += 1
        running = stamp if running is None else max(running, stamp)
        if stamp not in expected_set:
            counts["extra"] += 1
    counts["missing"] = len(expected_set - seen)
    return counts


class RunValidator:
    """Streaming causal-run checks. Record i is judged only from records 0..i and the expected list."""

    def __init__(self, expected_timestamps: Iterable[Any], *, map_frame: str) -> None:
        self.expected = normalize_expected_timestamps(expected_timestamps)
        self._position = {stamp: position for position, stamp in enumerate(self.expected)}
        self.map_frame = _text(map_frame, "map_frame", MAP_FRAME_MAX, identifier=True)
        self._previous: dict[str, Any] | None = None
        self._last_motion: tuple[int, int, int] | None = None
        self._seed_anchor_ns: int | None = None
        self._visual_anchor_seen = False
        self._count = 0
        self._status_counts = dict.fromkeys(STATUSES, 0)
        self._anchor_counts = dict.fromkeys(ANCHOR_KINDS, 0)
        self._without_motion = 0
        self._max_anchor_age: int | None = None
        self._max_motion_age: int | None = None
        self._rejection: CausalExportError | None = None
        self._finished = False

    @property
    def count(self) -> int:
        return self._count

    def add(self, record: Any) -> dict[str, Any]:
        if self._finished:
            raise CausalExportError("state", "run validation already finished")
        if self._rejection is not None:
            raise CausalExportError("state", f"run already rejected: {self._rejection}")
        index = self._count
        try:
            normalized = validate_record(record, index)
            self._check_position(normalized, index)
            self._check_continuity(normalized, index)
        except CausalExportError as exc:
            self._rejection = exc
            raise
        # The retained history is a private snapshot; callers receive a separate copy they may mutate.
        self._accept(_copy_record(normalized))
        return normalized

    def _check_position(self, record: Mapping[str, Any], index: int) -> None:
        stamp = record["timestamp_ns"]
        position = self._position.get(stamp)
        previous = self._previous
        if position is not None and position < index:
            raise CausalExportError("duplicate", f"timestamp_ns {stamp} was already emitted", index=index,
                                    field="timestamp_ns")
        if index >= len(self.expected):
            raise CausalExportError("extra", f"more records than the {len(self.expected)} expected images",
                                    index=index, field="timestamp_ns")
        if previous is not None and stamp < previous["timestamp_ns"]:
            raise CausalExportError("out_of_order", f"timestamp_ns {stamp} precedes the previous record",
                                    index=index, field="timestamp_ns")
        if position is None:
            raise CausalExportError("extra", f"timestamp_ns {stamp} is not an expected image", index=index,
                                    field="timestamp_ns")
        if position > index:
            raise CausalExportError("missing", f"expected image {self.expected[index]} at position {index} is "
                                    f"absent (got expected position {position}; missing or reordered)",
                                    index=index, field="timestamp_ns")
        if record["map_frame"] != self.map_frame:
            raise CausalExportError("map_frame", f"{record['map_frame']!r} differs from run map_frame "
                                    f"{self.map_frame!r}", index=index, field="map_frame")
        if previous is not None:
            if record["image_seq"] <= previous["image_seq"]:
                raise CausalExportError("sequence", "image_seq must strictly increase", index=index,
                                        field="image_seq")
            if record["image_log_ns"] < previous["image_log_ns"]:
                raise CausalExportError("sequence", "image_log_ns must not decrease", index=index,
                                        field="image_log_ns")

    def _check_continuity(self, record: Mapping[str, Any], index: int) -> None:
        previous = self._previous
        kind, status = record["anchor_kind"], record["status"]
        if kind == "organizer_seed":
            if self._visual_anchor_seen:
                raise CausalExportError("anchor", "organizer_seed anchor after a visual anchor would be reseeding",
                                        index=index, field="anchor_kind")
            if self._seed_anchor_ns is not None and record["last_anchor_ns"] != self._seed_anchor_ns:
                raise CausalExportError("anchor", "the single organizer seed anchor time changed (reseeding)",
                                        index=index, field="last_anchor_ns")
        elif status != "approximate_corrected":
            if (previous is None or previous["anchor_kind"] != kind or
                    previous["last_anchor_ns"] != record["last_anchor_ns"] or
                    previous["visual_reference"] != record["visual_reference"]):
                raise CausalExportError("anchor", "approximate reference anchor must come from an emitted current-"
                                        "image correction and be preserved unchanged until the next one",
                                        index=index, field="visual_reference")
        motion = _motion(record)
        if status in STALE_STATUSES and previous is not None:
            if _pose_bits(record) != _pose_bits(previous):
                raise CausalExportError("stale_pose", f"{status} must retain the previous emitted pose exactly",
                                        index=index, field="x,y,yaw")
            if motion != _motion(previous):
                raise CausalExportError("provenance", f"{status} must not use new motion provenance", index=index,
                                        field="latest_input_seq")
        if motion is not None and self._last_motion is not None:
            last_source, last_log, last_seq = self._last_motion
            source, log, seq = motion
            if seq < last_seq or source < last_source or log < last_log:
                raise CausalExportError("motion", "used motion provenance moved backwards", index=index,
                                        field="latest_input_seq")
            if seq == last_seq and (source, log) != (last_source, last_log):
                raise CausalExportError("motion", "same motion arrival sequence with a different source/log time",
                                        index=index, field="latest_input_seq")

    def _accept(self, record: dict[str, Any]) -> None:
        self._status_counts[record["status"]] += 1
        self._anchor_counts[record["anchor_kind"]] += 1
        age = record["anchor_age_ns"]
        self._max_anchor_age = age if self._max_anchor_age is None else max(self._max_anchor_age, age)
        motion = _motion(record)
        if motion is None:
            self._without_motion += 1
        else:
            motion_age = record["motion_age_ns"]
            self._max_motion_age = motion_age if self._max_motion_age is None else max(self._max_motion_age,
                                                                                         motion_age)
            self._last_motion = motion
        if record["anchor_kind"] == "organizer_seed" and self._seed_anchor_ns is None:
            self._seed_anchor_ns = record["last_anchor_ns"]
        if record["anchor_kind"] == "approximate_reference_pose":
            self._visual_anchor_seen = True
        self._previous = record
        self._count += 1

    def summary(self) -> dict[str, Any]:
        return {
            "expected_count": len(self.expected),
            "presence_rows": self._count,
            "status": dict(self._status_counts),
            "anchor_kind": dict(self._anchor_counts),
            "rows_without_motion": self._without_motion,
            "max_anchor_age_ns": self._max_anchor_age,
            "max_motion_age_ns": self._max_motion_age,
        }

    def finish(self) -> dict[str, Any]:
        if self._rejection is not None:
            raise CausalExportError("state", f"run already rejected: {self._rejection}")
        if self._count < len(self.expected):
            missing = self.expected[self._count]
            raise CausalExportError("missing", f"{len(self.expected) - self._count} expected images have no record; "
                                    f"first missing timestamp_ns {missing}", index=self._count,
                                    details={"missing": len(self.expected) - self._count})
        self._finished = True
        return self.summary()


def validate_records(records: Iterable[Any], expected_timestamps: Iterable[Any], *, map_frame: str) -> dict[str, Any]:
    """Batch validation with whole-list duplicate/out_of_order/extra/missing diagnostics."""
    expected = normalize_expected_timestamps(expected_timestamps)
    normalized = [validate_record(record, index) for index, record in enumerate(records)]
    counts = classify_timestamps([record["timestamp_ns"] for record in normalized], expected)
    for code in ("duplicate", "out_of_order", "extra", "missing"):
        if counts[code]:
            raise CausalExportError(code, f"timestamp sequence differs from the expected images: {counts}",
                                    details=counts)
    validator = RunValidator(expected, map_frame=map_frame)
    for record in normalized:
        validator.add(record)
    return validator.finish()


def _string_counts(summary: Mapping[str, Any]) -> dict[str, Any]:
    def text(value: int | None) -> str | None:
        return None if value is None else str(value)
    return {
        "presence_rows": str(summary["presence_rows"]),
        "status": {key: str(value) for key, value in summary["status"].items()},
        "anchor_kind": {key: str(value) for key, value in summary["anchor_kind"].items()},
        "rows_without_motion": str(summary["rows_without_motion"]),
        "max_anchor_age_ns": text(summary["max_anchor_age_ns"]),
        "max_motion_age_ns": text(summary["max_motion_age_ns"]),
    }


# Write-once export ---------------------------------------------------------------------

def _run_id(value: Any) -> str:
    if type(value) is not str or _RUN_ID.fullmatch(value) is None:
        raise CausalExportError("text", "run_id must match [A-Za-z0-9][A-Za-z0-9._-]{0,79}", field="run_id")
    return value


def _clock_domains(value: Any) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != set(CLOCK_DOMAIN_KEYS):
        raise CausalExportError("text", f"clock_domains must have exactly {list(CLOCK_DOMAIN_KEYS)}",
                                field="clock_domains")
    return {key: _text(value[key], f"clock_domains.{key}", CLOCK_LABEL_MAX, identifier=True)
            for key in CLOCK_DOMAIN_KEYS}


def _output_paths(named: Mapping[str, Any]) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    seen: dict[str, str] = {}
    for role, value in named.items():
        if not isinstance(value, (str, os.PathLike)) or not isinstance(os.fspath(value), str):
            raise CausalExportError("output_path", f"{role} must be a str path", field=role)
        path = Path(os.path.abspath(os.fspath(value)))
        if not path.name:
            raise CausalExportError("output_path", f"{role} must name a file", field=role)
        key = os.path.normcase(str(path))
        if key in seen:
            raise CausalExportError("output_path", f"{role} and {seen[key]} are the same path", field=role)
        seen[key] = role
        if not path.parent.is_dir():
            raise CausalExportError("output_path", f"parent directory of {role} must already exist", field=role)
        paths[role] = path
    for role, path in paths.items():
        if os.path.lexists(path):
            raise CausalExportError("output_exists", f"{role} {path} already exists; outputs are never replaced",
                                    field=role)
    return paths


def _create_exclusive(path: Path):
    try:
        return open(path, "xb")
    except FileExistsError:
        raise CausalExportError("output_exists", f"{path} already exists; outputs are never replaced") from None


def _fsync_directory(path: Path) -> None:
    """Best effort directory-entry durability on POSIX; Windows cannot open directories."""
    if os.name == "nt":
        return
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def _publish_no_replace(staging: Path, final: Path) -> bool:
    """Give the closed, synced staging document its final name; never replaces an existing name.

    POSIX uses a hard link (atomic, fails if ``final`` exists, needs hard-link support on the output
    filesystem) and returns True because the staging name remains as a second link. Windows uses
    os.rename, which raises instead of replacing an existing destination, and returns False.
    """
    try:
        if os.name == "nt":
            os.rename(staging, final)
            return False
        os.link(staging, final)
        return True
    except FileExistsError:
        raise CausalExportError("output_exists", f"{final} already exists; completion is never replaced") from None


def _file_identity(path: Path) -> tuple[str, int, bytes]:
    data = path.read_bytes()
    return hashlib.sha256(data).hexdigest(), len(data), data


def _failure_code(exc: BaseException) -> str:
    if isinstance(exc, CausalExportError):
        return exc.code
    return "io" if isinstance(exc, OSError) else "exception"


def _csv_row(record: Mapping[str, Any]) -> str:
    return (f"{record['timestamp_ns']},{format_csv_float(record['x'])},{format_csv_float(record['y'])},"
            f"{format_csv_float(record['yaw'])}\n")


class CausalCsvStatusExporter:
    """Write-once streaming export of one causal run. Never auto-completes."""

    def __init__(self, *, csv_path: str | os.PathLike, status_path: str | os.PathLike,
                 completion_path: str | os.PathLike, run_id: str, map_frame: str,
                 expected_timestamps: Iterable[Any], clock_domains: Mapping[str, str]) -> None:
        self.run_id = _run_id(run_id)
        self.clock_domains = _clock_domains(clock_domains)
        self._validator = RunValidator(expected_timestamps, map_frame=map_frame)
        named: dict[str, Any] = {"csv_path": csv_path, "status_path": status_path, "completion_path": completion_path}
        if isinstance(completion_path, (str, os.PathLike)) and isinstance(os.fspath(completion_path), str):
            named["completion_staging_path"] = os.fspath(completion_path) + COMPLETION_STAGING_SUFFIX
        paths = _output_paths(named)
        self.csv_path, self.status_path = paths["csv_path"], paths["status_path"]
        self.completion_path = paths["completion_path"]
        self.completion_staging_path = paths["completion_staging_path"]
        self.expected_sha256 = expected_timestamps_sha256(self._validator.expected)
        self.state = "open"
        self.failure: dict[str, str] | None = None
        self._csv = None
        self._status = _create_exclusive(self.status_path)
        header = {
            "kind": "header", "schema": SIDECAR_SCHEMA, "wire_schema": WIRE_SCHEMA, "run_id": self.run_id,
            "map_frame": self._validator.map_frame, "clock_domains": self.clock_domains,
            "expected_count": str(len(self._validator.expected)),
            "expected_timestamps_sha256": self.expected_sha256, "csv_name": self.csv_path.name,
        }
        try:
            self._status.write((canonical_json(header) + "\n").encode("ascii"))
            self._status.flush()
            self._csv = _create_exclusive(self.csv_path)
            self._csv.write(CSV_HEADER_LINE.encode("ascii"))
            self._csv.flush()
        except BaseException as exc:
            self._fail(_failure_code(exc), f"{type(exc).__name__}: {exc}")
            raise

    @property
    def records_written(self) -> int:
        return self._validator.count

    def __enter__(self) -> "CausalCsvStatusExporter":
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        if self.state == "open":
            if exc is None:
                self._fail("not_completed", "exporter closed without complete(); run is not complete")
            else:
                self._fail(_failure_code(exc), f"{type(exc).__name__}: {exc}")
        return False

    def _require_open(self) -> None:
        if self.state != "open":
            raise CausalExportError("state", f"exporter is {self.state}; no further writes are accepted")

    def write(self, record: Any) -> None:
        """Validate and append one record; any rejection fails the run and keeps the prefix."""
        self._require_open()
        try:
            normalized = self._validator.add(record)
            index = self._validator.count - 1
            status_line = canonical_json({"kind": "record", "index": str(index),
                                          "record": _wire_from_normalized(normalized)}) + "\n"
            csv_line = _csv_row(normalized)
            self._status.write(status_line.encode("ascii"))
            self._status.flush()
            self._csv.write(csv_line.encode("ascii"))
            self._csv.flush()
        except BaseException as exc:
            self._fail(_failure_code(exc), f"{type(exc).__name__}: {exc}")
            raise

    def fail(self, reason: str, *, code: str = "run_failed") -> None:
        """Record an explicit run failure such as a missing seed (NotReady) or missing pose."""
        self._require_open()
        if type(code) is not str or _FAIL_CODE.fullmatch(code) is None:
            raise CausalExportError("text", "failure code must match [a-z][a-z0-9_]{0,39}", field="code")
        self._fail(code, _text(reason, "reason", REASON_MAX))

    def _close_files(self) -> None:
        """Best-effort cleanup for failure paths only: tries every handle and never raises."""
        for handle in (self._status, self._csv):
            if handle is None or handle.closed:
                continue
            try:
                handle.flush()
                os.fsync(handle.fileno())
            except (OSError, ValueError):
                pass
            try:
                handle.close()
            except (OSError, ValueError):
                pass

    def _finalize_files(self) -> None:
        """Required durable close for successful completion.

        Every owned handle is flushed, fsynced and closed. The first failure propagates only after
        closing has been attempted for all handles. Page-cache bytes left readable after a failed
        sync are never treated as durable evidence.
        """
        primary: BaseException | None = None
        for handle in (self._status, self._csv):
            try:
                handle.flush()
                os.fsync(handle.fileno())
            except BaseException as exc:  # noqa: B036 - re-raised below after all handles are closed
                primary = primary or exc
            try:
                handle.close()
            except BaseException as exc:  # noqa: B036 - re-raised below after all handles are closed
                primary = primary or exc
        if primary is not None:
            raise primary

    def _write_staging(self, data: bytes) -> None:
        """Required exclusive write, flush, fsync and close of the staged manifest; failures keep the file."""
        handle = _create_exclusive(self.completion_staging_path)
        primary: BaseException | None = None
        try:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        except BaseException as exc:  # noqa: B036 - re-raised below after the close attempt
            primary = exc
        try:
            handle.close()
        except BaseException as exc:  # noqa: B036 - re-raised below
            primary = primary or exc
        if primary is not None:
            raise primary

    def _abort(self, exc: BaseException) -> None:
        self.state = "failed"
        self.failure = {"code": _failure_code(exc), "message": f"{type(exc).__name__}: {exc}"[:FAILURE_MESSAGE_MAX]}
        self._close_files()

    def _fail(self, code: str, message: str) -> None:
        if self.state != "open":
            return
        self.state = "failed"
        self.failure = {"code": str(code), "message": message[:FAILURE_MESSAGE_MAX]}
        trailer = {"kind": "failed", "code": self.failure["code"], "message": self.failure["message"],
                   "records": str(self._validator.count)}
        try:
            if self._status is not None and not self._status.closed:
                self._status.write((canonical_json(trailer) + "\n").encode("ascii"))
        except OSError:
            pass
        finally:
            self._close_files()

    def complete(self) -> dict[str, Any]:
        """Durably close both files, re-hash them, stage the manifest durably, then publish it no-replace.

        Any required flush/fsync/close/write/publication failure propagates, leaves the final completion
        name absent and retains every partial artifact, including a staged manifest.
        """
        self._require_open()
        try:
            summary = self._validator.finish()
        except CausalExportError as exc:
            self._fail(exc.code, exc.message)
            raise
        try:
            end = {"kind": "end", "records": str(summary["presence_rows"]),
                   "status_counts": {key: str(value) for key, value in summary["status"].items()}}
            self._status.write((canonical_json(end) + "\n").encode("ascii"))
            self._finalize_files()
            _fsync_directory(self.csv_path.parent)
            _fsync_directory(self.status_path.parent)
            csv_sha, csv_bytes, _ = _file_identity(self.csv_path)
            status_sha, status_bytes, _ = _file_identity(self.status_path)
            manifest = {
                "schema": COMPLETION_SCHEMA, "complete": True, "run_id": self.run_id, "wire_schema": WIRE_SCHEMA,
                "map_frame": self._validator.map_frame, "clock_domains": self.clock_domains,
                "expected_count": str(len(self._validator.expected)),
                "expected_timestamps_sha256": self.expected_sha256,
                "csv": {"name": self.csv_path.name, "sha256": csv_sha, "bytes": str(csv_bytes),
                        "rows": str(summary["presence_rows"])},
                "status": {"name": self.status_path.name, "sha256": status_sha, "bytes": str(status_bytes),
                           "records": str(summary["presence_rows"])},
                "counts": _string_counts(summary),
                "notes": list(COMPLETION_NOTES),
            }
            self._write_staging(_pretty_json(manifest).encode("ascii"))
            _fsync_directory(self.completion_staging_path.parent)
            staging_linked = _publish_no_replace(self.completion_staging_path, self.completion_path)
        except BaseException as exc:  # noqa: B036 - recorded, then re-raised unchanged
            # Without the final name the run stays incomplete; any staged manifest is retained as evidence.
            self._abort(exc)
            raise
        # Published. Nothing after this point may turn a reload-verifiable success into an in-memory failure.
        self.state = "complete"
        if staging_linked:
            try:
                os.unlink(self.completion_staging_path)  # the second hard link only; the final name stays
            except OSError:
                pass
        try:
            _fsync_directory(self.completion_path.parent)
        except OSError:
            pass
        return manifest


def export_records(records: Iterable[Any], **options: Any) -> dict[str, Any]:
    """Export a finite record iterable; returns the completion manifest or raises after keeping evidence."""
    with CausalCsvStatusExporter(**options) as exporter:
        for record in records:
            exporter.write(record)
        return exporter.complete()


# Independent verification of written artifacts ----------------------------------------

def read_exact_csv(path: str | os.PathLike) -> list[tuple[int, float, float, float]]:
    """Strictly read this exporter's CSV dialect without any float timestamp conversion."""
    try:
        text = Path(path).read_bytes().decode("ascii")
    except UnicodeDecodeError:
        raise CausalExportError("csv_format", "CSV must be ASCII") from None
    if not text.startswith(CSV_HEADER_LINE):
        raise CausalExportError("csv_format", "exact header line timestamp_ns,x,y,yaw followed by LF required")
    if not text.endswith("\n") or "\r" in text:
        raise CausalExportError("csv_format", "rows must be LF-terminated without CR")
    rows: list[tuple[int, float, float, float]] = []
    for index, line in enumerate(text[len(CSV_HEADER_LINE):-1].split("\n") if text != CSV_HEADER_LINE else []):
        tokens = line.split(",")
        if len(tokens) != 4:
            raise CausalExportError("csv_format", "row must have exactly four fields", index=index)
        stamp = parse_decimal_ns(tokens[0], "timestamp_ns", index)
        values = []
        for field, token in zip(POSE_FIELDS, tokens[1:], strict=True):
            if _CSV_FLOAT.fullmatch(token) is None:
                raise CausalExportError("csv_format", f"{token[:40]!r} is not a positional decimal", index=index,
                                        field=field)
            values.append(_pose(float(token), field, index))
        rows.append((stamp, values[0], values[1], values[2]))
    return rows


def _expect(condition: bool, code: str, message: str) -> None:
    if not condition:
        raise CausalExportError(code, message)


def _read_manifest(path: Path) -> dict[str, Any]:
    if not os.path.lexists(path):
        raise CausalExportError("incomplete", f"completion manifest {path.name} is absent; run is not complete")
    try:
        manifest = strict_json_loads(path.read_bytes().decode("ascii"))
    except UnicodeDecodeError:
        raise CausalExportError("completion", "completion manifest must be ASCII JSON") from None
    except CausalExportError as exc:
        raise CausalExportError("completion", f"completion manifest is malformed: {exc.message}") from None
    keys = {"schema", "complete", "run_id", "wire_schema", "map_frame", "clock_domains", "expected_count",
            "expected_timestamps_sha256", "csv", "status", "counts", "notes"}
    _expect(isinstance(manifest, dict) and set(manifest) == keys, "completion", "manifest fields differ")
    _expect(manifest["schema"] == COMPLETION_SCHEMA and manifest["complete"] is True and
            manifest["wire_schema"] == WIRE_SCHEMA and manifest["notes"] == list(COMPLETION_NOTES),
            "completion", "manifest schema/complete/notes differ")
    for role, count_key in (("csv", "rows"), ("status", "records")):
        item = manifest[role]
        _expect(isinstance(item, dict) and set(item) == {"name", "sha256", "bytes", count_key}, "completion",
                f"manifest {role} entry fields differ")
    return manifest


def verify_export(*, csv_path: str | os.PathLike, status_path: str | os.PathLike,
                  completion_path: str | os.PathLike, run_id: str, map_frame: str,
                  expected_timestamps: Iterable[Any]) -> dict[str, Any]:
    """Re-derive and cross-check a completed export from its bytes; raises on any mismatch."""
    expected = normalize_expected_timestamps(expected_timestamps)
    run_id = _run_id(run_id)
    map_frame = _text(map_frame, "map_frame", MAP_FRAME_MAX, identifier=True)
    csv_file, status_file = Path(csv_path), Path(status_path)
    manifest = _read_manifest(Path(completion_path))
    _expect(manifest["run_id"] == run_id, "identity_mismatch", "manifest run_id differs from the requested run")
    _expect(manifest["map_frame"] == map_frame, "identity_mismatch", "manifest map_frame differs")
    _expect(manifest["csv"]["name"] == csv_file.name and manifest["status"]["name"] == status_file.name,
            "identity_mismatch", "manifest output names differ from the supplied paths")
    _expect(manifest["expected_count"] == str(len(expected)) and
            manifest["expected_timestamps_sha256"] == expected_timestamps_sha256(expected),
            "identity_mismatch", "manifest expected-image identity differs")
    contents = {}
    for role, path in (("csv", csv_file), ("status", status_file)):
        _expect(path.is_file(), "hash_mismatch", f"{role} file is missing")
        digest, size, data = _file_identity(path)
        _expect(manifest[role]["sha256"] == digest and manifest[role]["bytes"] == str(size), "hash_mismatch",
                f"{role} bytes differ from the completion manifest")
        contents[role] = data

    try:
        status_text = contents["status"].decode("ascii")
    except UnicodeDecodeError:
        raise CausalExportError("sidecar", "status sidecar must be ASCII") from None
    _expect(status_text.endswith("\n"), "sidecar", "status sidecar must end with LF")
    lines = status_text[:-1].split("\n")
    _expect(len(lines) >= 2, "sidecar", "status sidecar needs a header and a trailer")
    objects = []
    for number, line in enumerate(lines):
        obj = strict_json_loads(line)
        _expect(isinstance(obj, dict) and canonical_json(obj) == line, "sidecar",
                f"sidecar line {number} is not canonical JSON")
        objects.append(obj)
    header, trailer = objects[0], objects[-1]
    header_expected = {
        "kind": "header", "schema": SIDECAR_SCHEMA, "wire_schema": WIRE_SCHEMA, "run_id": manifest["run_id"],
        "map_frame": manifest["map_frame"], "clock_domains": manifest["clock_domains"],
        "expected_count": manifest["expected_count"],
        "expected_timestamps_sha256": manifest["expected_timestamps_sha256"], "csv_name": csv_file.name,
    }
    _expect(header == header_expected, "identity_mismatch", "sidecar header IDs differ from the manifest")
    _clock_domains(header["clock_domains"])
    if trailer.get("kind") == "failed":
        raise CausalExportError("incomplete", "sidecar records a failed run")
    validator = RunValidator(expected, map_frame=map_frame)
    records = []
    for number, obj in enumerate(objects[1:-1]):
        _expect(set(obj) == {"kind", "index", "record"} and obj["kind"] == "record" and obj["index"] == str(number),
                "sidecar", f"sidecar record line {number + 1} envelope differs")
        records.append(validator.add(record_from_wire(obj["record"], number)))
    summary = validator.finish()
    counts = _string_counts(summary)
    _expect(set(trailer) == {"kind", "records", "status_counts"} and trailer["kind"] == "end", "sidecar",
            "sidecar trailer differs")
    _expect(trailer["records"] == counts["presence_rows"] and trailer["status_counts"] == counts["status"],
            "count_mismatch", "sidecar trailer counts differ from its records")
    _expect(manifest["counts"] == counts and manifest["csv"]["rows"] == counts["presence_rows"] and
            manifest["status"]["records"] == counts["presence_rows"], "count_mismatch",
            "manifest counts differ from the re-derived records")

    rows = read_exact_csv(csv_file)
    _expect(len(rows) == len(records), "count_mismatch", "CSV row count differs from the sidecar")
    csv_lines = contents["csv"].decode("ascii")[len(CSV_HEADER_LINE):].split("\n")
    for index, record in enumerate(records):
        if csv_lines[index] + "\n" != _csv_row(record):
            raise CausalExportError("csv_mismatch", "CSV row differs from the sidecar record", index=index)
    return {"verified": True, "run_id": run_id, "csv_sha256": manifest["csv"]["sha256"],
            "status_sha256": manifest["status"]["sha256"], "counts": counts}

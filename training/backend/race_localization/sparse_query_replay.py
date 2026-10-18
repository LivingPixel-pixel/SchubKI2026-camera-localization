"""SPARSE-01: label-free sparse query replay with sealed Phase A evidence.

Phase A replays a frozen list of label-free query rows through one absolute
localizer and seals what happened before any evaluator sees ground truth:

1. Rows are validated before any evidence directory or query exists. The row
   schema is exactly query_id,timestamp_ns,camera_id,image_path,image_sha256,
   image_bytes. Timestamps are exact, strictly increasing, unique integer
   nanoseconds. Any extra key (a lap label or a pose, for example) is rejected.
2. Every row produces exactly one write-once attempt record with status
   accepted, rejected or error. Byte count and SHA256 checks precede
   IMREAD_COLOR (BGR) decoding, and decoded dimensions must equal the checked
   calibration before localize. Errors keep their stage and carry no pose.
   Only decode and localize errors and ordinary rejections (features,
   retrieval, geometry, ambiguity) are query outcomes; every other error stage,
   explicit setup faults surfacing mid-run and returned setup-fault reasons
   (camera, calibration, size, feature or descriptor mismatch) are integrity
   failures that make Phase A ok=false.
3. Every returned fix is checked against the query identity (source and
   latest input exposure, pinned map hash, feature and calibration identity,
   monotonic completion window); a mismatch is an identity_check error and
   withholds the pose. An accepted full body pose is converted through
   map_contracts.yaw_from_body_pose into timestamp_ns,x,y,yaw.
4. The accepted-only CSV, the per-attempt records, the configuration and the
   summary go through the accepted EVAL-01 EvidenceWriter, which then seals.
   A separate evaluator must verify the seal before it touches ground truth.

There is no ground-truth, lap-label, scorer or query-selection input anywhere
in this module. Accepted-fix coverage is not accuracy, and cached replay is
not live capture-to-publication latency. This is not a continuous runner: no
held, tracked or fallback pose is ever emitted.

The core is injection-based (image source, decoder, localizer, clock). The
production adapter imports the unaccepted LOC and MAP modules lazily, only when
explicitly invoked, and their absence is an honest ProductionUnavailable error.
It loads the query calibration independently, calls
metric_map.load_map_artifact with caller-checked pins, constructs
map_localization.MapLocalizer on the loaded FrozenMap and calls
localize(CameraObservation); localize_features is never used.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, fields, is_dataclass
import hashlib
import importlib
import io
import math
from pathlib import Path
import re
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from . import race_measurement as rm
from .map_contracts import (CameraCalibration, CameraObservation, FitInterval, FixCandidate,
                            validate_timestamp_ns, validate_transform, yaw_from_body_pose)


SCHEMA = "schubki-sparse-phase-a-v1"
ATTEMPT_SCHEMA = "schubki-sparse-attempt-v1"
ROW_FIELDS = ("query_id", "timestamp_ns", "camera_id", "image_path", "image_sha256", "image_bytes")
ROW_FORMATS = {".json": "json", ".csv": "csv"}
CLOCK_DOMAIN = "time.monotonic_ns"
DECODE_FLAG = "IMREAD_COLOR"
CONFIG_NAME = "phase-a-config.json"
COMPONENTS_NAME = "phase-a-components.json"
SUMMARY_NAME = "phase-a.json"
ACCEPTED_CSV = "accepted.csv"
STATUSES = ("accepted", "rejected", "error")
ERROR_STAGES = ("read", "size", "hash", "decode", "observation", "dimension", "localize", "deferred_setup",
                "identity_check", "serialization", "internal")
# Integrity failures make Phase A unusable (ok=false). Only decode and localize
# errors are ordinary per-query outcomes of correctly frozen inputs and setup.
INTEGRITY_STAGES = frozenset({"read", "size", "hash", "observation", "dimension", "deferred_setup", "identity_check",
                              "serialization", "internal"})
# Returned rejection reasons that the replay's own setup checks make impossible
# for a correct binding (camera, calibration identity, decoded dimensions,
# localize() never taking features). They are integrity faults, never R.
SETUP_FAULT_REASONS = ("camera_mismatch", "calibration_mismatch", "image_size_mismatch",
                       "feature_identity_mismatch", "descriptor_mismatch")
# Optional runtime bindings: ReplaySetup field -> attribute of returned diagnostics.
DIAGNOSTIC_BINDINGS = {"expected_localizer_identity": "localizer_identity",
                       "expected_derived_index_identity": "derived_retrieval_index_identity",
                       "expected_declared_index_sha256": "declared_retrieval_index_sha256"}
TIMING_KEYS = ("read_start_ns", "read_end_ns", "hash_end_ns", "decode_end_ns", "observation_end_ns",
               "localize_start_ns", "localize_end_ns")
LOCALIZER_STAGES = ("extraction", "retrieval", "geometry")
# Localizer stages that ran, by rejection reason. This mirrors the stage order
# of the LOC-01 MapLocalizer.localize candidate (identity gates, extraction,
# retrieval, geometry); recheck it against the accepted LOC revision.
_STAGES_BY_REASON = {
    "camera_mismatch": (), "calibration_mismatch": (), "image_size_mismatch": (),
    "insufficient_features": ("extraction",),
    "no_retrieval_candidate": ("extraction", "retrieval"),
}
CSV_FORMAT = ".10f"
_ROW_INT_TEXT = re.compile(r"[0-9]+")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_MESSAGE_LIMIT = 2000


class QueryRowError(ValueError):
    """The frozen query-row input is malformed; no Phase A evidence is created."""


class ProductionUnavailable(RuntimeError):
    """An accepted production component (LOC or MAP) is absent in this source."""


class SetupFailure(RuntimeError):
    """A pinned production identity does not match the loaded component."""


# Explicit setup faults surfacing during an attempt are integrity, not query outcomes.
DEFERRED_SETUP_ERRORS = (SetupFailure, ProductionUnavailable, ImportError)


# Rows ---------------------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class QueryRow:
    """One label-free inference row. No pose, lap or evaluator field exists."""
    query_id: str
    timestamp_ns: int
    camera_id: str
    image_path: str
    image_sha256: str
    image_bytes: int

    def __post_init__(self) -> None:
        for name in ("query_id", "camera_id", "image_path"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip() or "\x00" in value:
                raise QueryRowError(f"{name} must be a nonempty string without NUL")
        if not isinstance(self.image_sha256, str) or _SHA256.fullmatch(self.image_sha256) is None:
            raise QueryRowError("image_sha256 must be a lowercase SHA256 hex digest")
        try:
            stamp = rm.strict_int_ns(self.timestamp_ns, "timestamp_ns")
            size = rm.strict_int_ns(self.image_bytes, "image_bytes")
        except ValueError as exc:
            raise QueryRowError(str(exc)) from exc
        if size < 1:
            raise QueryRowError("image_bytes must be positive")
        object.__setattr__(self, "timestamp_ns", stamp)
        object.__setattr__(self, "image_bytes", size)


@dataclass(frozen=True, slots=True)
class QueryRows:
    rows: tuple[QueryRow, ...]
    source_sha256: str
    source_format: str
    expected_timestamps_sha256: str

    @property
    def timestamps(self) -> tuple[int, ...]:
        return tuple(row.timestamp_ns for row in self.rows)


def _rows_from_json(text: str) -> list[QueryRow]:
    try:
        data = rm.strict_json_loads(text)
    except ValueError as exc:
        raise QueryRowError(f"Query rows are not strict JSON: {exc}") from exc
    if not isinstance(data, list):
        raise QueryRowError("Query-row JSON must be an array of row objects")
    rows = []
    for index, item in enumerate(data):
        if not isinstance(item, dict) or set(item) != set(ROW_FIELDS):
            keys = sorted(item) if isinstance(item, dict) else type(item).__name__
            raise QueryRowError(f"Row {index} must have exactly the keys {','.join(ROW_FIELDS)}; found {keys}")
        try:
            rows.append(QueryRow(**{name: item[name] for name in ROW_FIELDS}))
        except QueryRowError as exc:
            raise QueryRowError(f"Row {index}: {exc}") from exc
    return rows


def _rows_from_csv(text: str) -> list[QueryRow]:
    if text.startswith("\ufeff"):
        raise QueryRowError("Query-row CSV must not start with a UTF-8 BOM")
    reader = csv.reader(io.StringIO(text, newline=""), strict=True)
    try:
        header = next(reader, None)
        if header is None or tuple(header) != ROW_FIELDS:
            raise QueryRowError(f"Query-row CSV header must be exactly {','.join(ROW_FIELDS)}")
        rows = []
        for line, values in enumerate(reader, 2):
            if len(values) != len(ROW_FIELDS):
                raise QueryRowError(f"CSV line {line}: expected {len(ROW_FIELDS)} fields, found {len(values)}")
            item = dict(zip(ROW_FIELDS, values))
            for name in ("timestamp_ns", "image_bytes"):
                if _ROW_INT_TEXT.fullmatch(item[name]) is None:
                    raise QueryRowError(f"CSV line {line}: {name} must be unsigned integer text")
                item[name] = int(item[name])
            try:
                rows.append(QueryRow(**item))
            except QueryRowError as exc:
                raise QueryRowError(f"CSV line {line}: {exc}") from exc
    except csv.Error as exc:
        raise QueryRowError(f"Query-row CSV is malformed: {exc}") from exc
    return rows


def parse_query_rows(raw: bytes, source_format: str, expected_count: int) -> QueryRows:
    """Strictly validate frozen label-free rows in their given order."""
    if source_format not in ROW_FORMATS.values():
        raise QueryRowError(f"Unknown query-row format {source_format!r}")
    if isinstance(expected_count, bool) or not isinstance(expected_count, int) or expected_count < 1:
        raise QueryRowError("expected_count must be a positive integer declared by the caller")
    try:
        text = bytes(raw).decode("utf-8")
    except UnicodeDecodeError as exc:
        raise QueryRowError(f"Query rows are not UTF-8: {exc.reason}") from exc
    rows = _rows_from_json(text) if source_format == "json" else _rows_from_csv(text)
    if len(rows) != expected_count:
        raise QueryRowError(f"Declared {expected_count} queries but the row file has {len(rows)}")
    stamps = [row.timestamp_ns for row in rows]
    try:
        expected = rm.normalize_expected_timestamps(stamps)
    except ValueError as exc:
        raise QueryRowError(str(exc)) from exc
    if tuple(stamps) != expected:
        # normalize_expected_timestamps sorts silently; disordered input is rejected here.
        raise QueryRowError("Query rows must be in strictly increasing timestamp_ns order")
    ids = [row.query_id for row in rows]
    if len(set(ids)) != len(ids):
        raise QueryRowError("query_id values must be unique")
    return QueryRows(tuple(rows), rm.sha256_bytes(bytes(raw)), source_format, rm.expected_timestamps_sha256(expected))


def load_query_rows(path: str | Path, expected_count: int) -> QueryRows:
    source = Path(path)
    source_format = ROW_FORMATS.get(source.suffix.lower())
    if source_format is None:
        raise QueryRowError("Query rows must be a .json array or a .csv file")
    try:
        raw = source.read_bytes()
    except OSError as exc:
        raise QueryRowError(f"Cannot read query rows: {exc}") from exc
    return parse_query_rows(raw, source_format, expected_count)


# Injectable components ----------------------------------------------------------------------

@dataclass(frozen=True)
class FileImageSource:
    """Bounded file reader. With a root, relative and absolute paths must stay inside it."""
    root: Path | None = None

    def resolve(self, row: QueryRow) -> Path:
        raw = Path(row.image_path)
        if self.root is None:
            return raw
        root = Path(self.root).resolve()
        path = (raw if raw.is_absolute() else root / raw).resolve()
        if not path.is_relative_to(root) or path == root:
            raise ValueError("image_path leaves the declared image root")
        return path

    def __call__(self, row: QueryRow) -> bytes:
        with self.resolve(row).open("rb") as stream:
            # Read at most one byte beyond the declaration so oversize files are bounded.
            return stream.read(row.image_bytes + 1)

    def identity(self) -> dict:
        return {"kind": "file", "root": None if self.root is None else str(Path(self.root).resolve()),
                "bounded_read": "image_bytes + 1"}


class ImreadColorDecoder:
    """cv2.imdecode(..., IMREAD_COLOR): 8-bit three-channel BGR. OpenCV is imported lazily."""

    def __call__(self, data: bytes) -> np.ndarray:
        import cv2  # noqa: PLC0415

        image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("cv2.imdecode with IMREAD_COLOR returned no image")
        return image

    def identity(self) -> dict:
        import cv2  # noqa: PLC0415

        return {"kind": "cv2.imdecode", "flag": DECODE_FLAG, "flag_value": int(cv2.IMREAD_COLOR),
                "color_order": "BGR", "opencv": cv2.__version__}


@dataclass(frozen=True)
class ReplaySetup:
    """Loaded components for one Phase A run.

    localizer.localize(CameraObservation) must return an object whose .fix is a
    FixCandidate and whose .diagnostics is a dataclass. calibration is the
    independently loaded and checked query calibration. identity is JSON-safe.
    The optional expected_* diagnostic bindings are checked against every
    returned diagnostics object when declared; undeclared ones are recorded as
    not_declared, never as checked.
    """
    localizer: Any
    calibration: CameraCalibration
    identity: Mapping[str, Any]
    expected_map_hash: str
    expected_feature_id: str | None = None
    expected_localizer_identity: str | None = None
    expected_derived_index_identity: str | None = None
    expected_declared_index_sha256: str | None = None


@dataclass(frozen=True)
class PhaseAResult:
    ok: bool
    directory: Path
    manifest_sha256: str
    counts: Mapping[str, int]
    summary: Mapping[str, Any]


# JSON conversion ----------------------------------------------------------------------------

def to_jsonable(value: Any, where: str = "value") -> Any:
    """Finite JSON form through dataclasses.fields, tolist() and item(); tuples become lists.

    Raises TypeError for unsupported objects and ValueError for nonfinite numbers.
    """
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, np.generic):
        return to_jsonable(value.item(), where)
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"Nonfinite number at {where}")
        return float(value)
    if isinstance(value, np.ndarray):
        return to_jsonable(value.tolist(), where)
    if is_dataclass(value) and not isinstance(value, type):
        return {item.name: to_jsonable(getattr(value, item.name), f"{where}.{item.name}") for item in fields(value)}
    if isinstance(value, Mapping):
        result = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"Non-string mapping key at {where}")
            result[key] = to_jsonable(item, f"{where}.{key}")
        return result
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item, f"{where}[{index}]") for index, item in enumerate(value)]
    raise TypeError(f"Unsupported {type(value).__name__} at {where}")


def _message(exc: BaseException) -> str:
    text = str(exc)
    return text if len(text) <= _MESSAGE_LIMIT else text[:_MESSAGE_LIMIT] + "...[truncated]"


def _failure(stage: str, exc: BaseException) -> dict:
    return {"stage": stage, "error_type": type(exc).__name__, "message": _message(exc)}


def _sha_file_or_none(path: Any) -> str | None:
    try:
        return rm.sha256_file(path) if path else None
    except OSError:
        return None


# Attempt construction -----------------------------------------------------------------------

class _AttemptError(Exception):
    def __init__(self, stage: str, cause: BaseException, component_stage: str | None = None):
        super().__init__(stage)
        self.stage = stage
        self.cause = cause
        self.component_stage = component_stage


def _component_error(stage: str, exc: BaseException) -> _AttemptError:
    if isinstance(exc, DEFERRED_SETUP_ERRORS):
        return _AttemptError("deferred_setup", exc, stage)
    return _AttemptError(stage, exc)


def _base_record(index: int, row: QueryRow) -> dict:
    return {"schema": ATTEMPT_SCHEMA, "query_index": index, "query_id": row.query_id,
            "timestamp_ns": row.timestamp_ns, "camera_id": row.camera_id, "image_path": row.image_path,
            "image_sha256": row.image_sha256, "image_bytes": row.image_bytes, "cached_replay": True}


def _timing(clock_domain: str) -> dict:
    return {"clock_domain": clock_domain, **{key: None for key in TIMING_KEYS}}


def _finish_timing(timing: dict) -> dict:
    timing["not_executed"] = [key for key in TIMING_KEYS if timing[key] is None]
    return timing


def _localizer_stage_timing(reason: str, diagnostics: Any) -> dict:
    ran = _STAGES_BY_REASON.get(reason, LOCALIZER_STAGES) if reason else LOCALIZER_STAGES
    result = {}
    for stage in LOCALIZER_STAGES:
        value = getattr(diagnostics, f"{stage}_ns", None)
        result[f"{stage}_ran"] = stage in ran
        result[f"{stage}_ns"] = value if stage in ran and value is not None else None
    result["basis"] = "stages inferred from the fix reason; unexecuted stages are unavailable, not zero work"
    return result


def _check_fix(row: QueryRow, result: Any, setup: ReplaySetup, timing: dict) -> FixCandidate:
    """Identity and causality of a returned fix. Any mismatch withholds the pose."""
    fix = getattr(result, "fix", None)
    if not isinstance(fix, FixCandidate):
        raise TypeError(f"localize returned {type(result).__name__} without a FixCandidate .fix")
    diagnostics = getattr(result, "diagnostics", None)
    if not is_dataclass(diagnostics) or isinstance(diagnostics, type):
        raise TypeError("localize result .diagnostics must be a dataclass instance")
    problems = []
    for name in ("source_timestamp_ns", "latest_input_timestamp_ns"):
        value = getattr(fix, name)
        if type(value) is not int or value != row.timestamp_ns:
            problems.append(f"{name} {value!r} != query timestamp_ns {row.timestamp_ns}")
    if fix.reason in SETUP_FAULT_REASONS:
        problems.append(f"returned reason {fix.reason} is a setup or integrity fault, not a query outcome")
    if fix.calibration_id != setup.calibration.identity:
        problems.append("calibration_id differs from the checked query calibration")
    if fix.map_hash != setup.expected_map_hash:
        problems.append("map_hash differs from the pinned map content hash")
    if setup.expected_feature_id is not None and fix.feature_id != setup.expected_feature_id:
        problems.append("feature_id differs from the pinned feature identity")
    for name, attribute in DIAGNOSTIC_BINDINGS.items():
        expected = getattr(setup, name)
        if expected is not None and getattr(diagnostics, attribute, None) != expected:
            problems.append(f"diagnostics {attribute} differs from the checked runtime binding {name}")
    start, end = timing["localize_start_ns"], timing["localize_end_ns"]
    started = getattr(diagnostics, "started_monotonic_ns", None)
    completed = getattr(diagnostics, "completed_monotonic_ns", None)
    if (started is None) != (completed is None):
        problems.append("diagnostics must report both started_monotonic_ns and completed_monotonic_ns or neither")
    # localize start <= diagnostic start <= fix completion <= diagnostic completion <= localize end.
    chain = [start, *([started] if started is not None else []), fix.completed_monotonic_ns,
             *([completed] if completed is not None else []), end]
    if (any(isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) for value in chain)
            or any(left > right for left, right in zip(chain, chain[1:]))):
        problems.append("localizer monotonic stamps are not ordered inside the replay localize window "
                        "(clock domain or order mismatch)")
    ordered = [timing[key] for key in TIMING_KEYS if timing[key] is not None]
    if any(left > right for left, right in zip(ordered, ordered[1:])):
        problems.append("replay monotonic stamps are out of order")
    if fix.status == "accepted" and fix.T_WB is None:
        problems.append("accepted fix without a body pose")
    if problems:
        raise ValueError("; ".join(problems))
    return fix


def _fix_record(fix: FixCandidate) -> dict:
    record = {item.name: getattr(fix, item.name) for item in fields(fix) if item.name != "T_WB"}
    record["correspondence_count"] = int(len(fix.landmark_ids))
    record["inlier_count"] = int(np.count_nonzero(fix.inlier_mask))
    return record


def _pose(fix: FixCandidate) -> dict:
    T_WB = validate_transform(fix.T_WB, "fix T_WB")
    x, y, yaw = float(T_WB[0, 3]), float(T_WB[1, 3]), yaw_from_body_pose(T_WB)
    if not all(math.isfinite(value) for value in (x, y, yaw)):
        raise ValueError("Accepted pose is not finite")
    return {"T_WB": T_WB.tolist(), "x": x, "y": y, "yaw": yaw, "yaw_source": "map_contracts.yaw_from_body_pose"}


def _returned_summary(result: Any) -> dict:
    """Returned details for an integrity error: fix fields and diagnostics, never the pose."""
    fix = getattr(result, "fix", None)
    if not isinstance(fix, FixCandidate):
        return {"returned_type": type(result).__name__}
    summary = {"returned_type": type(result).__name__, "status": fix.status, "reason": fix.reason,
               "source_timestamp_ns": fix.source_timestamp_ns,
               "latest_input_timestamp_ns": fix.latest_input_timestamp_ns, "map_hash": fix.map_hash,
               "feature_id": fix.feature_id, "calibration_id": fix.calibration_id, "region_id": fix.region_id,
               "inlier_count": int(np.count_nonzero(fix.inlier_mask)), "correspondence_count": int(len(fix.landmark_ids)),
               "completed_monotonic_ns": fix.completed_monotonic_ns,
               "pose": "withheld" if fix.status == "accepted" else "none"}
    try:
        summary["diagnostics"] = to_jsonable(getattr(result, "diagnostics", None), "returned.diagnostics")
    except (TypeError, ValueError) as exc:
        summary["diagnostics"] = {"unserializable": type(exc).__name__, "message": _message(exc)}
    return summary


class _Replay:
    def __init__(self, setup: ReplaySetup, image_source: Callable, decoder: Callable,
                 clock: Callable[[], int], clock_domain: str):
        self.setup, self.image_source, self.decoder = setup, image_source, decoder
        self.clock, self.clock_domain = clock, clock_domain

    def now(self) -> int:
        return validate_timestamp_ns(self.clock())

    def attempt(self, index: int, row: QueryRow) -> tuple[dict, tuple | None]:
        """Return the attempt record and, for accepted fixes, its CSV values."""
        record, timing = _base_record(index, row), _timing(self.clock_domain)
        result = None
        try:
            timing["read_start_ns"] = self.now()
            try:
                data = self.image_source(row)
                if not isinstance(data, (bytes, bytearray, memoryview)):
                    raise TypeError(f"image source returned {type(data).__name__}, not bytes")
                data = bytes(data)
            except Exception as exc:  # noqa: BLE001  (every read failure is recorded)
                raise _component_error("read", exc) from exc
            finally:
                timing["read_end_ns"] = self.now()
            if len(data) != row.image_bytes:
                relation = "more than" if len(data) > row.image_bytes else "fewer than"
                raise _AttemptError("size", ValueError(f"read {relation} the declared {row.image_bytes} bytes"))
            digest = hashlib.sha256(data).hexdigest()
            timing["hash_end_ns"] = self.now()
            if digest != row.image_sha256:
                raise _AttemptError("hash", ValueError(f"image SHA256 {digest} differs from the declared digest"))
            try:
                image = self.decoder(data)
            except Exception as exc:  # noqa: BLE001
                raise _component_error("decode", exc) from exc
            finally:
                timing["decode_end_ns"] = self.now()
            calibration = self.setup.calibration
            try:
                if row.camera_id != calibration.camera_id:
                    raise _AttemptError("observation", ValueError(
                        "row camera_id does not match the checked query calibration"))
                if not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
                    raise _AttemptError("observation", ValueError("decoded image must be 8-bit three-channel BGR"))
                if image.shape[:2] != (calibration.height, calibration.width):
                    raise _AttemptError("dimension", ValueError(
                        f"decoded image is {image.shape[1]}x{image.shape[0]} but the checked calibration is "
                        f"{calibration.width}x{calibration.height}"))
                observation = CameraObservation(row.camera_id, row.timestamp_ns, image, calibration.identity)
            except _AttemptError:
                raise
            except Exception as exc:  # noqa: BLE001
                raise _AttemptError("observation", exc) from exc
            finally:
                timing["observation_end_ns"] = self.now()
            timing["localize_start_ns"] = self.now()
            try:
                result = self.setup.localizer.localize(observation)
            except Exception as exc:  # noqa: BLE001
                raise _component_error("localize", exc) from exc
            finally:
                timing["localize_end_ns"] = self.now()
            try:
                fix = _check_fix(row, result, self.setup, timing)
                pose = _pose(fix) if fix.status == "accepted" else None
            except Exception as exc:  # noqa: BLE001
                raise _AttemptError("identity_check", exc) from exc
        except _AttemptError as error:
            failure = _failure(error.stage, error.cause)
            if error.component_stage is not None:
                failure["component_stage"] = error.component_stage
            record.update(status="error", error=failure, timing=_finish_timing(timing))
            if error.stage == "identity_check" and result is not None:
                record["returned"] = _returned_summary(result)
            return record, None
        record.update(status=fix.status, fix=_fix_record(fix), diagnostics=result.diagnostics,
                      localizer_stage_timing=_localizer_stage_timing(fix.reason, result.diagnostics),
                      timing=_finish_timing(timing))
        if pose is None:
            return record, None
        record["pose"] = pose
        return record, (row.timestamp_ns, pose["x"], pose["y"], pose["yaw"])


def _serialization_fallback(record: dict, exc: BaseException) -> dict:
    keep = ("schema", "query_index", "query_id", "timestamp_ns", "camera_id", "image_path", "image_sha256",
            "image_bytes", "cached_replay")
    fallback = {key: record[key] for key in keep}
    fallback.update(status="error", error=_failure("serialization", exc),
                    returned_status=record.get("status") if record.get("status") in STATUSES else None)
    return fallback


def accepted_csv_text(rows: Sequence[tuple[int, float, float, float]]) -> str:
    lines = [",".join(rm.SUBMISSION_HEADER)]
    for stamp, x, y, yaw in rows:
        lines.append(f"{stamp},{format(x, CSV_FORMAT)},{format(y, CSV_FORMAT)},{format(yaw, CSV_FORMAT)}")
    return "\n".join(lines) + "\n"


def _stats(values: list[int]) -> dict:
    """Integer min/max and the conventional median (race_measurement.percentile, NumPy linear)."""
    if not values:
        return {"count": 0, "min_ns": None, "median_ns": None, "max_ns": None}
    ordered = sorted(values)
    return {"count": len(ordered), "min_ns": ordered[0], "median_ns": rm.percentile(ordered, 50),
            "max_ns": ordered[-1]}


def _timing_summary(records: list[dict], clock_domain: str) -> dict:
    spans = {"read": ("read_start_ns", "read_end_ns"), "hash": ("read_end_ns", "hash_end_ns"),
             "decode": ("hash_end_ns", "decode_end_ns"), "localize": ("localize_start_ns", "localize_end_ns")}
    result = {"clock_domain": clock_domain, "cached_replay": True,
              "note": "Cached file replay in one monotonic domain; not live capture-to-publication latency"}
    for name, (left, right) in spans.items():
        values = [item["timing"][right] - item["timing"][left] for item in records
                  if isinstance(item.get("timing"), dict) and item["timing"].get(left) is not None
                  and item["timing"].get(right) is not None]
        result[name] = _stats(values)
    return result


def _module_identity() -> dict:
    here = Path(__file__).resolve().parent
    return {name: _sha_file_or_none(here / name) for name in
            ("sparse_query_replay.py", "race_measurement.py", "map_contracts.py")}


def _binding_status(setup: ReplaySetup | None) -> dict:
    """Declared diagnostic bindings are checked per attempt; others are explicitly not declared."""
    if setup is None:
        return {name: {"status": "not_available"} for name in DIAGNOSTIC_BINDINGS}
    result = {}
    for name, attribute in DIAGNOSTIC_BINDINGS.items():
        value = getattr(setup, name)
        result[name] = ({"status": "checked", "value": value, "diagnostics_attribute": attribute} if value is not None
                        else {"status": "not_declared", "diagnostics_attribute": attribute})
    return result


def _component_identity(component: Any) -> Any:
    describe = getattr(component, "identity", None)
    if callable(describe):
        return describe()
    return {"kind": "injected", "type": type(component).__name__}


def run_phase_a(rows: QueryRows, out_dir: str | Path, setup_factory: Callable[[], ReplaySetup], *,
                image_source: Callable[[QueryRow], bytes], decoder: Callable[[bytes], np.ndarray] | None = None,
                clock: Callable[[], int] = time.monotonic_ns, clock_domain: str = CLOCK_DOMAIN,
                run_metadata: Mapping[str, Any] | None = None,
                environment: Callable[[], Mapping[str, Any]] = rm.environment_identity) -> PhaseAResult:
    """Replay every validated row once and seal Phase A evidence.

    Raises before creating evidence for invalid arguments or a reused directory.
    After the directory exists, every outcome is recorded and sealed.
    """
    if not isinstance(rows, QueryRows) or not rows.rows:
        raise TypeError("run_phase_a requires validated QueryRows")
    if not callable(setup_factory) or not callable(image_source) or not callable(clock):
        raise TypeError("setup_factory, image_source and clock must be callable")
    if not isinstance(clock_domain, str) or not clock_domain:
        raise ValueError("clock_domain must name the monotonic clock")
    decoder = ImreadColorDecoder() if decoder is None else decoder
    metadata = to_jsonable(dict(run_metadata or {}), "run_metadata")
    rm.canonical_json(metadata)
    count = len(rows.rows)
    width = max(3, len(str(count - 1)))
    started = validate_timestamp_ns(clock())
    writer = rm.EvidenceWriter(out_dir)
    setup_failures: list[dict] = []
    integrity_failures: list[dict] = []
    evidence_failures: list[dict] = []
    config = {
        "schema": SCHEMA + "/config", "phase": "A", "attempt_schema": ATTEMPT_SCHEMA,
        "rows": {"sha256": rows.source_sha256, "format": rows.source_format, "count": count,
                 "expected_timestamps_sha256": rows.expected_timestamps_sha256,
                 "first_timestamp_ns": rows.rows[0].timestamp_ns, "last_timestamp_ns": rows.rows[-1].timestamp_ns,
                 "fields": list(ROW_FIELDS)},
        "expected_count": count, "decode_flag": DECODE_FLAG, "clock_domain": clock_domain,
        "cached_replay": True, "continuous_stream": False, "ground_truth_inputs": "none",
        "image_source": None, "decoder": None, "run_metadata": metadata,
        "source_sha256": _module_identity(), "environment": None,
    }
    for key, component in (("image_source", image_source), ("decoder", decoder)):
        try:
            config[key] = to_jsonable(_component_identity(component), key)
        except Exception as exc:  # noqa: BLE001
            setup_failures.append(_failure("setup", exc))
    try:
        config["environment"] = to_jsonable(dict(environment()), "environment")
    except Exception as exc:  # noqa: BLE001
        setup_failures.append(_failure("setup", exc))
    writer.write_json(CONFIG_NAME, config)

    setup = None
    try:
        setup = setup_factory()
        if not isinstance(setup, ReplaySetup) or not isinstance(setup.calibration, CameraCalibration):
            raise TypeError("setup_factory must return a ReplaySetup with a CameraCalibration")
        if not callable(getattr(setup.localizer, "localize", None)):
            raise TypeError("ReplaySetup.localizer must provide localize(CameraObservation)")
        optional = ("expected_feature_id", *DIAGNOSTIC_BINDINGS)
        for name in ("expected_map_hash", *(item for item in optional if getattr(setup, item) is not None)):
            if not isinstance(getattr(setup, name), str) or _SHA256.fullmatch(getattr(setup, name)) is None:
                raise ValueError(f"ReplaySetup.{name} must be a lowercase SHA256 hex digest")
        cameras = sorted({row.camera_id for row in rows.rows})
        if cameras != [setup.calibration.camera_id]:
            raise SetupFailure(f"Row camera IDs {cameras} do not match the checked query calibration "
                               f"camera {setup.calibration.camera_id!r}")
        components = {"schema": SCHEMA + "/components", "identity": to_jsonable(dict(setup.identity), "identity"),
                      "calibration": {"identity": setup.calibration.identity, "camera_id": setup.calibration.camera_id,
                                      "width": setup.calibration.width, "height": setup.calibration.height},
                      "expected_map_hash": setup.expected_map_hash, "expected_feature_id": setup.expected_feature_id,
                      "diagnostic_bindings": _binding_status(setup)}
        writer.write_json(COMPONENTS_NAME, components)
    except Exception as exc:  # noqa: BLE001  (setup failures are sealed evidence)
        setup_failures.append(_failure("setup", exc))
        setup = None

    records: list[dict] = []
    accepted: list[tuple] = []
    attempt_files: list[str] = []
    if setup is not None and not setup_failures:
        replay = _Replay(setup, image_source, decoder, clock, clock_domain)
        for index, row in enumerate(rows.rows):
            name = f"attempt-{index:0{width}d}.json"
            try:
                record, csv_values = replay.attempt(index, row)
            except Exception as exc:  # noqa: BLE001  (an unexpected bug still leaves a record)
                record, csv_values = _base_record(index, row), None
                record.update(status="error", error=_failure("internal", exc))
            try:
                payload = to_jsonable(record, name)
                rm.canonical_json(payload)
            except (TypeError, ValueError) as exc:
                payload, csv_values = _serialization_fallback(record, exc), None
            try:
                writer.write_json(name, payload)
            except Exception as exc:  # noqa: BLE001
                evidence_failures.append({"query_index": index, **_failure("evidence", exc)})
                continue
            records.append(payload)
            attempt_files.append(name)
            if payload["status"] == "error" and payload["error"]["stage"] in INTEGRITY_STAGES:
                integrity_failures.append({"query_index": index, "query_id": row.query_id, **payload["error"]})
            if csv_values is not None and payload["status"] == "accepted":
                accepted.append(csv_values)

    accepted_stamps = [item[0] for item in accepted]
    validation: dict
    try:
        writer.write_text(ACCEPTED_CSV, accepted_csv_text(accepted))
        if accepted:
            report = rm.validate_predictions(writer.path(ACCEPTED_CSV), accepted_stamps)
            sealed_sha = writer.files()[ACCEPTED_CSV]["sha256"]
            validation = {"status": "validated", "report": report,
                          "ok": bool(report["ok"]) and report["sha256"] == sealed_sha}
        else:
            validation = {"status": "not_applicable_k0", "ok": None,
                          "note": "K=0 is a valid outcome: header-only CSV; validate_predictions needs a nonempty list"}
    except Exception as exc:  # noqa: BLE001
        validation = {"status": "failed", "ok": False, "error": _failure("accepted_csv", exc)}

    status_counts = {status: sum(1 for item in records if item["status"] == status) for status in STATUSES}
    k, r, e = status_counts["accepted"], status_counts["rejected"], status_counts["error"]
    missing = count - len(records)
    stages = {stage: sum(1 for item in records if item["status"] == "error" and item["error"]["stage"] == stage)
              for stage in ERROR_STAGES}
    ok = (not setup_failures and not integrity_failures and not evidence_failures and missing == 0
          and validation["ok"] is not False)
    files = writer.files()
    summary = {
        "schema": SCHEMA, "phase": "A", "ok": ok,
        "counts": {"expected": count, "records": len(records), "accepted": k, "rejected": r, "error": e,
                   "missing": missing},
        "fractions": {"completeness": (k + r + e) / count, "non_error_execution": (k + r) / count,
                      "accepted_fix_coverage": k / count},
        "fraction_notes": {"completeness": "(K+R+E)/N attempts recorded",
                           "non_error_execution": "(K+R)/N localizer returned a checked fix",
                           "accepted_fix_coverage": "K/N accepted fixes; not accuracy"},
        "error_stages": stages, "integrity_stages": sorted(INTEGRITY_STAGES),
        "setup_fault_reasons": list(SETUP_FAULT_REASONS), "diagnostic_bindings": _binding_status(setup),
        "setup_failures": setup_failures, "integrity_failures": integrity_failures,
        "evidence_failures": evidence_failures, "accepted_csv": {"name": ACCEPTED_CSV, "rows": k,
            "sha256": files.get(ACCEPTED_CSV, {}).get("sha256"), "header": list(rm.SUBMISSION_HEADER),
            "format": CSV_FORMAT}, "validation": validation,
        "attempt_files": attempt_files, "config_sha256": files[CONFIG_NAME]["sha256"],
        "components_sha256": files.get(COMPONENTS_NAME, {}).get("sha256"),
        "expected_timestamps_sha256": rows.expected_timestamps_sha256, "rows_sha256": rows.source_sha256,
        "timing": _timing_summary(records, clock_domain), "run_started_monotonic_ns": started,
        "run_completed_monotonic_ns": validate_timestamp_ns(clock()),
        "cached_replay": True, "continuous_stream": False, "ground_truth_inputs": "none",
        "evaluation": "not performed; a separate evaluator must verify this seal before accessing ground truth",
    }
    writer.write_json(SUMMARY_NAME, summary)
    writer.seal()
    manifest_sha = rm.sha256_file(writer.directory / rm.EvidenceWriter.MANIFEST)
    return PhaseAResult(ok, writer.directory, manifest_sha, dict(summary["counts"]), summary)


def verify_sealed_phase_a(directory: str | Path) -> list[str]:
    """Problems with a sealed Phase A directory; an unsealed directory is incomplete."""
    try:
        return rm.verify_evidence(directory)
    except FileNotFoundError:
        return ["evidence-manifest.json: missing; Phase A is not sealed and is incomplete"]


# Production adapter (laptop only, after accepted LOC/MAP and coordinator freeze) ------------

PRODUCTION_MODULES = ("map_localization", "map_retrieval", "metric_map")
# MAP-02 loader and LOC-01 identity entry points named by the coordinator reconciliation.
PRODUCTION_MAP_LOADER = "load_map_artifact"
PRODUCTION_FEATURE_IDENTITY = "sift_feature_identity"
_REQUIRED_PIN_HASHES = ("calibration_file_sha256", "expected_map_hash", "expected_source_sha256",
                        "expected_split_sha256", "expected_exporter_sha256")
_OPTIONAL_PIN_HASHES = ("expected_manifest_sha256", "expected_arrays_sha256", "expected_calibration_identity",
                        "expected_localizer_identity")


@dataclass(frozen=True)
class ProductionPins:
    """Frozen identities required before production replay. All are checked, not trusted.

    expected_fit_membership comes from the caller's own checked split, never
    from artifact metadata. expected_source_sha256 is the checked recording
    hash. Optional manifest/arrays hashes bind independently approved files.
    """
    camera_id: str
    calibration_path: Path
    calibration_file_sha256: str
    map_path: Path
    expected_map_hash: str
    expected_fit_membership: tuple[FitInterval, ...]
    expected_source_sha256: str
    expected_split_sha256: str
    expected_exporter_sha256: str
    component_sha256: Mapping[str, str]
    expected_manifest_sha256: str | None = None
    expected_arrays_sha256: str | None = None
    expected_calibration_identity: str | None = None
    expected_localizer_identity: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.camera_id, str) or not self.camera_id:
            raise ValueError("camera_id is required")
        for name in _REQUIRED_PIN_HASHES + _OPTIONAL_PIN_HASHES:
            value = getattr(self, name)
            if (value is not None or name in _REQUIRED_PIN_HASHES) and (
                    not isinstance(value, str) or _SHA256.fullmatch(value) is None):
                raise ValueError(f"{name} must be a lowercase SHA256 hex digest")
        membership = tuple(self.expected_fit_membership)
        if not membership or any(not isinstance(item, FitInterval) for item in membership):
            raise ValueError("expected_fit_membership must list the frozen map's FitIntervals")
        pins = dict(self.component_sha256)
        if set(pins) != set(PRODUCTION_MODULES) or any(
                not isinstance(value, str) or _SHA256.fullmatch(value) is None for value in pins.values()):
            raise ValueError(f"component_sha256 must pin exactly {', '.join(PRODUCTION_MODULES)}")
        object.__setattr__(self, "calibration_path", Path(self.calibration_path))
        object.__setattr__(self, "map_path", Path(self.map_path))
        object.__setattr__(self, "expected_fit_membership", membership)
        object.__setattr__(self, "component_sha256", pins)


def fit_membership_from_json(value: Any) -> tuple[FitInterval, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError("Fit membership must be a nonempty JSON array")
    result = []
    for item in value:
        if not isinstance(item, dict) or set(item) != {"start_ns", "end_ns", "lap_ids"} or not isinstance(item["lap_ids"], list):
            raise ValueError("Each fit interval needs exactly start_ns, end_ns and a lap_ids array")
        result.append(FitInterval(rm.strict_int_ns(item["start_ns"], "start_ns"), rm.strict_int_ns(item["end_ns"], "end_ns"),
                                  tuple(rm.strict_int_ns(lap, "lap id") for lap in item["lap_ids"])))
    return tuple(result)


def load_query_calibration(path: str | Path, file_sha256: str, camera_id: str) -> CameraCalibration:
    """Independently load the official cached calibration file, never the map's copy."""
    from . import official_io  # noqa: PLC0415  (existing helper, imported only for production)
    from .map_contracts import calibration_from_official  # noqa: PLC0415

    source = Path(path).resolve()
    payload = official_io.read_calibration({"cache_dir": str(source.parent), "calibration_path": source.name,
                                            "calibration_sha256": file_sha256})
    return calibration_from_official(payload, camera_id)


def _artifact_identity(path: Path) -> dict:
    target = Path(path).resolve()
    if target.is_file():
        return {"kind": "file", "path": str(target), "sha256": rm.sha256_file(target), "bytes": target.stat().st_size}
    if target.is_dir():
        files = {item.relative_to(target).as_posix(): rm.sha256_file(item)
                 for item in sorted(target.rglob("*")) if item.is_file()}
        return {"kind": "directory", "path": str(target), "files": files}
    raise FileNotFoundError(f"Map artifact not found: {target}")


def _fit_key(membership: Sequence[FitInterval]) -> list:
    return sorted([item.start_ns, item.end_ns, list(item.lap_ids)] for item in membership)


def production_setup_factory(pins: ProductionPins, *, clock: Callable[[], int] = time.monotonic_ns,
                             importer: Callable[[str], Any] | None = None) -> Callable[[], ReplaySetup]:
    """Lazy production binding. Nothing is imported until the returned factory runs."""
    if not isinstance(pins, ProductionPins):
        raise TypeError("production_setup_factory requires ProductionPins")
    load = importer or (lambda name: importlib.import_module(f".{name}", __package__))

    def build() -> ReplaySetup:
        calibration = load_query_calibration(pins.calibration_path, pins.calibration_file_sha256, pins.camera_id)
        if pins.expected_calibration_identity is not None and calibration.identity != pins.expected_calibration_identity:
            raise SetupFailure("Query calibration identity differs from the pinned identity")
        modules = {}
        for name in PRODUCTION_MODULES:
            try:
                modules[name] = load(name)
            except ImportError as exc:
                raise ProductionUnavailable(f"Accepted production module race_localization.{name} is not available: "
                                            f"{exc}. Production replay is gated on accepted LOC/MAP.") from exc
        sources = {}
        for name, module in modules.items():
            source = getattr(module, "__file__", None)
            digest = _sha_file_or_none(source)
            if digest != pins.component_sha256[name]:
                raise SetupFailure(f"race_localization.{name} source SHA256 {digest} differs from the pinned accepted blob")
            sources[name] = {"path": str(source), "sha256": digest}
        localizer_type = getattr(modules["map_localization"], "MapLocalizer", None)
        loader = getattr(modules["metric_map"], PRODUCTION_MAP_LOADER, None)
        feature_identity = getattr(modules["map_retrieval"], PRODUCTION_FEATURE_IDENTITY, None)
        if localizer_type is None or not callable(loader) or not callable(feature_identity):
            raise ProductionUnavailable(f"MapLocalizer, metric_map.{PRODUCTION_MAP_LOADER} or "
                                        f"map_retrieval.{PRODUCTION_FEATURE_IDENTITY} is not available")
        # The expected feature identity derives from the checked query calibration, not the map.
        expected_feature = feature_identity(calibration, None)
        artifact = _artifact_identity(pins.map_path)
        loaded = loader(pins.map_path, expected_feature_identity=expected_feature, expected_calibration=calibration,
                        expected_source_sha256=pins.expected_source_sha256,
                        expected_split_sha256=pins.expected_split_sha256,
                        expected_fit_membership=pins.expected_fit_membership,
                        expected_exporter_sha256=pins.expected_exporter_sha256,
                        expected_map_hash=pins.expected_map_hash)
        from .map_contracts import FrozenMap  # noqa: PLC0415

        if _artifact_identity(pins.map_path) != artifact:
            raise SetupFailure("Map artifact bytes changed while the map was being loaded")
        frozen_map = getattr(loaded, "frozen_map", None)
        if not isinstance(frozen_map, FrozenMap):
            raise SetupFailure("Map loader did not return a loaded artifact with a FrozenMap")
        loaded_files = {"manifest_sha256": getattr(loaded, "manifest_sha256", None),
                        "arrays_sha256": getattr(loaded, "arrays_sha256", None)}
        hashed = artifact.get("files", {})
        names = {"manifest_sha256": getattr(modules["metric_map"], "MANIFEST_NAME", "manifest.json"),
                 "arrays_sha256": getattr(modules["metric_map"], "ARRAYS_NAME", "arrays.npz")}
        for key, name in names.items():
            if loaded_files[key] is None or hashed.get(name) != loaded_files[key]:
                raise SetupFailure(f"Loaded {key} differs from the {name} bytes hashed by the replay")
            pinned = getattr(pins, f"expected_{key}")
            if pinned is not None and loaded_files[key] != pinned:
                raise SetupFailure(f"Loaded {key} differs from the independently approved file hash")
        if frozen_map.feature_identity.identity != expected_feature.identity:
            raise SetupFailure("Map feature identity differs from the identity derived from the checked calibration")
        if frozen_map.map_hash != pins.expected_map_hash:
            raise SetupFailure("Loaded map content hash differs from the pinned map hash")
        if _fit_key(frozen_map.fit_membership) != _fit_key(pins.expected_fit_membership):
            raise SetupFailure("Loaded map fit membership differs from the pinned fit membership")
        if frozen_map.calibration.identity != calibration.identity:
            raise SetupFailure("Map calibration differs from the independently loaded query calibration")
        provenance = frozen_map.provenance
        if (provenance.source_sha256, provenance.split_sha256, provenance.exporter_sha256) != (
                pins.expected_source_sha256, pins.expected_split_sha256, pins.expected_exporter_sha256):
            raise SetupFailure("Map provenance differs from the pinned recording, split or exporter hash")
        # Same monotonic clock as the replay, so localizer stamps share its domain.
        localizer = localizer_type(frozen_map, clock=clock)
        localizer_identity = getattr(localizer, "identity", None)
        if pins.expected_localizer_identity is not None and localizer_identity != pins.expected_localizer_identity:
            raise SetupFailure("Localizer identity differs from the pinned localizer identity")
        # Runtime identities captured after construction bind every returned diagnostics object.
        # The derived index identity is distinct from FrozenMap.retrieval_index_sha256 (declared).
        derived_index_identity = getattr(getattr(localizer, "index", None), "identity", None)
        for label, value in (("MapLocalizer.identity", localizer_identity),
                             ("MapLocalizer.index.identity", derived_index_identity)):
            if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
                raise SetupFailure(f"{label} is not an exposed SHA256 runtime identity")
        config = getattr(localizer, "config", None)
        build_json = getattr(loaded, "build_json", None)
        identity = {
            "kind": "production", "map_artifact": artifact, "map_hash": frozen_map.map_hash,
            "loaded_file_sha256": loaded_files,
            "build_sha256": rm.sha256_bytes(build_json.encode("utf-8")) if isinstance(build_json, str) else None,
            "loader_pins": {"expected_source_sha256": pins.expected_source_sha256,
                            "expected_split_sha256": pins.expected_split_sha256,
                            "expected_exporter_sha256": pins.expected_exporter_sha256,
                            "expected_feature_identity": expected_feature.identity,
                            "fit_membership_source": "caller-checked pins, not artifact metadata"},
            "fit_membership": _fit_key(frozen_map.fit_membership),
            "feature_id": frozen_map.feature_identity.identity, "retrieval_index_sha256": frozen_map.retrieval_index_sha256,
            "map_provenance": to_jsonable(frozen_map.provenance, "map_provenance"),
            "calibration_file_sha256": pins.calibration_file_sha256, "calibration_identity": calibration.identity,
            "localizer_identity": localizer_identity, "derived_retrieval_index_identity": derived_index_identity,
            "localizer_config": config.as_dict() if callable(getattr(config, "as_dict", None)) else None,
            "component_sources": sources, "entry_point": "MapLocalizer.localize(CameraObservation)",
        }
        return ReplaySetup(localizer, calibration, to_jsonable(identity, "identity"), frozen_map.map_hash,
                           frozen_map.feature_identity.identity, expected_localizer_identity=localizer_identity,
                           expected_derived_index_identity=derived_index_identity,
                           expected_declared_index_sha256=frozen_map.retrieval_index_sha256)

    return build

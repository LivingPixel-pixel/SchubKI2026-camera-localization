"""EVAL-01 measurement harness: strict outputs, candidate identity, scoring and timing.

Standard library only, so fabricated checks run without the image stack.

Information boundary: evaluator ground truth enters only the scoring functions,
which slice it for brackets and pass it to the unchanged official scorer. Nothing
here feeds an estimator. Exposure stamps are exact integer nanoseconds in the
recording domain; processing stamps use one declared monotonic clock domain and
are never subtracted from exposure stamps.

Coverage terms are kept apart: finite-output continuity counts every expected
camera exposure with a finite row (held poses included), absolute-fix coverage
comes from the candidate's own status diagnostics, and accuracy comes only from
the official scorer's printed metrics. No composite score is computed.
"""
from __future__ import annotations

from bisect import bisect_left, bisect_right
import csv
from dataclasses import asdict, dataclass, fields
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time
from typing import Any, Callable, Iterable, Mapping, Sequence


SCHEMA = "schubki-race-measurement-v1"
SUBMISSION_HEADER = ("timestamp_ns", "x", "y", "yaw")
GT_HEADER = ("timestamp_ns", "x", "y", "z", "yaw")
INT64_MAX = (1 << 63) - 1
OFFICIAL_SCORER_SHA256 = "3a14b712f21304040d52fccd0b6624970c7aec9b9b521c0e2c33c6b6cfcd3f8e"
OFFICIAL_EXTRACTOR_SHA256 = "eba5903bc574b2a265c671e1221993d61d0885210c4a8b2f07f6929ebd388e8e"
SCORER_MAX_GT_GAP_NS = 50_000_000
RANKING_COVERAGE_PERCENT = 90.0
RUNTIME_REFERENCE_MS = 100.0
BENCHMARK_RESOURCE = "benchmark-cpu"
AUDIT_RESOURCE = "validation-audit"
# Printed full-run values of the preserved original baseline (EVAL-01 brief).
BASELINE_REFERENCE_METRICS = {"position_rmse_m": 612.896, "lateral_rmse_m": 354.487,
                              "yaw_rmse_deg": 98.608, "coverage_percent": 100.0}
MEMBERSHIPS = ("full", "fit", "tune", "audit")
INTERVAL_CONVENTIONS = ("closed", "half_open")
YAW_CONVENTIONS = ("wrapped_pi", "finite")
# The original runner prints yaw with 10 decimals, so a wrapped value within 5e-11 of
# pi is written as 3.1415926536 > math.pi. Allow only that printed-rounding margin.
YAW_WRAP_TOLERANCE_RAD = 1e-9
SAMPLE_LIMIT = 20

# Exactly the printed labels and capture groups of official_evaluation._parse_metrics.
# The coordinator confirmed these twelve fields against the unchanged evaluate.py; the
# 90 % threshold on the coverage line is a constant and NOT RANKED is a status.
SCORER_METRIC_PATTERNS = {
    "poses_submitted": r"poses submitted\s+(\d+)",
    "poses_evaluated": r"poses submitted\s+\d+\s+evaluated\s+(\d+)",
    "coverage_percent": r"coverage \(1 s bins\)\s+([\d.]+)",
    "position_rmse_m": r"position RMSE\s+\[m\]\s+([^\s]+)",
    "lateral_rmse_m": r"lateral RMSE\s+\[m\]\s+([^\s]+)",
    "longitudinal_rmse_m": r"longitudinal RMSE\s+\[m\]\s+([^\s]+)",
    "yaw_rmse_deg": r"yaw RMSE\s+\[deg\]\s+([^\s]+)",
    "within_0_25_m_percent": r"within\s+0\.25 m\s+([^\s]+)",
    "within_0_50_m_percent": r"within\s+0\.50 m\s+([^\s]+)",
    "within_1_00_m_percent": r"within\s+1\.00 m\s+([^\s]+)",
}
SCORER_PAIR_PATTERN = r"position median/p95\s+([^\s]+)\s*/\s*([^\s]+)"
SCORER_PAIR_METRICS = ("position_median_m", "position_p95_m")
REQUIRED_SCORER_METRICS = (*SCORER_METRIC_PATTERNS, *SCORER_PAIR_METRICS)
INTEGER_SCORER_METRICS = ("poses_submitted", "poses_evaluated")
LOWER_IS_BETTER = ("position_rmse_m", "lateral_rmse_m", "longitudinal_rmse_m", "yaw_rmse_deg",
                   "position_median_m", "position_p95_m")

# Candidate status vocabularies map onto disjoint measurement categories.
STATUS_CATEGORIES = ("absolute_fix", "map_tracked", "relative_visual", "fallback", "held", "lost", "seed")
LOCALIZED_CATEGORIES = ("absolute_fix", "map_tracked")
STATUS_VOCABULARIES = {
    "contract-v1": {"absolute_fix": "absolute_fix", "tracked": "map_tracked",
                    "relative_fallback": "fallback", "held": "held", "lost": "lost"},
    "official-baseline-v1": {"initialized": "seed", "tracked": "relative_visual",
                             "propagated": "fallback", "held": "held"},
}

_DECIMAL = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?")
_NONFINITE = re.compile(r"[+-]?(?:nan|inf|infinity)", re.IGNORECASE)
_DIGITS = re.compile(r"[0-9]+")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_SHA1 = re.compile(r"[0-9a-f]{40}")
_SAFE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")


class MeasurementFailure(RuntimeError):
    """Raised after evidence is saved when a measurement must not be used."""

    def __init__(self, message: str, failures: Sequence[str], evidence_dir: str | None = None):
        super().__init__(message + ": " + "; ".join(failures))
        self.failures = list(failures)
        self.evidence_dir = evidence_dir


# Identity and evidence ---------------------------------------------------------------------

def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def _pretty_json(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n"


def strict_json_loads(text: str) -> Any:
    def reject(token: str) -> Any:
        raise ValueError(f"Nonfinite JSON constant {token} is not permitted")
    return json.loads(text, parse_constant=reject)


def strict_int_ns(value: Any, name: str = "timestamp_ns") -> int:
    """Exact nonnegative signed-int64 nanoseconds; floats, bools and strings are rejected."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be integer nanoseconds, not {type(value).__name__}")
    if not 0 <= value <= INT64_MAX:
        raise ValueError(f"{name} must fit nonnegative signed int64")
    return int(value)


def _sha_or_raise(value: Any, name: str, allow_none_marker: bool = False) -> str:
    if allow_none_marker and value == "none":
        return "none"
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA256 hex digest" + (" or 'none'" if allow_none_marker else ""))
    return value


class EvidenceWriter:
    """Write-once evidence directory. Files use exclusive creation and are hashed."""

    MANIFEST = "evidence-manifest.json"

    def __init__(self, directory: str | Path):
        path = Path(directory).resolve()
        if path.exists() and (not path.is_dir() or any(path.iterdir())):
            raise FileExistsError(f"Evidence directory must be new or empty, never reused: {path}")
        path.mkdir(parents=True, exist_ok=True)
        self.directory = path
        self._files: dict[str, dict] = {}
        self._nested: list[str] = []
        self._sealed = False

    def nested_directory(self, name: str) -> Path:
        """Reserve a subdirectory for a separately sealed evidence writer."""
        if self._sealed:
            raise RuntimeError("Evidence is sealed and immutable")
        target = self.path(name)
        if target.exists() or name in self._nested:
            raise FileExistsError(f"Nested evidence directory already exists: {target}")
        self._nested.append(name)
        return target

    def path(self, name: str) -> Path:
        if _SAFE_NAME.fullmatch(name) is None or name == self.MANIFEST:
            raise ValueError(f"Unsafe or reserved evidence file name: {name!r}")
        return self.directory / name

    def write_bytes(self, name: str, data: bytes) -> dict:
        if self._sealed:
            raise RuntimeError("Evidence is sealed and immutable")
        target = self.path(name)
        with target.open("xb") as stream:
            stream.write(data)
        record = {"path": str(target), "sha256": sha256_bytes(data), "bytes": len(data)}
        self._files[name] = record
        return record

    def write_text(self, name: str, text: str) -> dict:
        return self.write_bytes(name, text.encode("utf-8"))

    def write_json(self, name: str, value: Any) -> dict:
        return self.write_text(name, _pretty_json(value))

    def files(self) -> dict:
        return {name: dict(record) for name, record in sorted(self._files.items())}

    def seal(self) -> dict:
        for name in self._nested:
            nested_manifest = self.directory / name / self.MANIFEST
            if not nested_manifest.is_file():
                raise RuntimeError(f"Nested evidence {name} was not sealed")
            self._files[name] = {"path": str(self.directory / name), "nested": True,
                                 "manifest_sha256": sha256_file(nested_manifest)}
        manifest = {"schema": SCHEMA + "/evidence-manifest", "directory": str(self.directory), "files": self.files()}
        with (self.directory / self.MANIFEST).open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(_pretty_json(manifest))
        self._sealed = True
        return manifest


def verify_evidence(directory: str | Path) -> list[str]:
    """Re-hash a sealed evidence directory; any mutation, removal or addition is reported."""
    root = Path(directory).resolve()
    manifest = strict_json_loads((root / EvidenceWriter.MANIFEST).read_text(encoding="utf-8"))
    problems = []
    listed = manifest["files"]
    for name, record in listed.items():
        target = root / name
        if record.get("nested"):
            nested_manifest = target / EvidenceWriter.MANIFEST
            if not nested_manifest.is_file():
                problems.append(f"{name}: nested manifest missing")
            elif sha256_file(nested_manifest) != record["manifest_sha256"]:
                problems.append(f"{name}: nested manifest SHA256 changed")
            else:
                problems.extend(f"{name}/{item}" for item in verify_evidence(target))
        elif not target.is_file():
            problems.append(f"{name}: missing")
        elif sha256_file(target) != record["sha256"]:
            problems.append(f"{name}: SHA256 changed")
    for target in root.iterdir():
        if target.name != EvidenceWriter.MANIFEST and target.name not in listed:
            problems.append(f"{target.name}: not listed in sealed manifest")
    return problems


def environment_identity() -> dict:
    """Interpreter, platform, CPU count, thread limits and load where the OS exposes it."""
    try:
        load = list(os.getloadavg())
    except (AttributeError, OSError):
        load = None
    threads = {name: os.environ.get(name) for name in
               ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
                "OPENCV_FOR_THREADS_NUM", "CUDA_VISIBLE_DEVICES")}
    return {"python_executable": sys.executable, "python_version": platform.python_version(),
            "implementation": platform.python_implementation(), "platform": platform.platform(),
            "machine": platform.machine(), "processor": platform.processor() or "unavailable",
            "logical_processors": os.cpu_count(), "thread_environment": threads,
            "load_average_1_5_15": load, "load_average_note": None if load is not None else "os.getloadavg unavailable on this OS"}


# Candidate identity -----------------------------------------------------------------------

IDENTITY_HASH_FIELDS = ("config_sha256", "calibration_identity", "input_recording_sha256", "split_sha256",
                        "scorer_sha256", "extractor_sha256", "dependencies_sha256")
IDENTITY_OPTIONAL_HASH_FIELDS = ("map_sha256", "feature_identity", "weights_sha256")


def build_candidate_identity(fields_: Mapping[str, Any], *, none_reasons: Mapping[str, str] | None = None,
                             expected_scorer_sha256: str = OFFICIAL_SCORER_SHA256,
                             expected_extractor_sha256: str = OFFICIAL_EXTRACTOR_SHA256) -> dict:
    """Bind source, config, map, features, weights, calibration, data, split and helpers.

    map/feature/weights may be the literal 'none' only with an explicit reason, as for
    the original visual-odometry baseline that has no map. The returned candidate_id is
    the SHA256 of the canonical identity, so any field change produces a new candidate.
    """
    reasons = dict(none_reasons or {})
    required = {"source_commit", "source_tree_clean", "camera_id", *IDENTITY_HASH_FIELDS, *IDENTITY_OPTIONAL_HASH_FIELDS}
    missing = sorted(required - set(fields_))
    unknown = sorted(set(fields_) - required - {"label"})
    if missing or unknown:
        raise ValueError(f"Candidate identity fields missing {missing} or unknown {unknown}")
    commit = fields_["source_commit"]
    if not isinstance(commit, str) or _SHA1.fullmatch(commit) is None:
        raise ValueError("source_commit must be the exact 40-character lowercase commit SHA")
    if fields_["source_tree_clean"] is not True:
        raise ValueError("A dirty source tree cannot be bound to a commit identity")
    camera = fields_["camera_id"]
    if not isinstance(camera, str) or not camera.strip():
        raise ValueError("camera_id must be a nonempty string")
    identity: dict[str, Any] = {"schema": SCHEMA + "/candidate", "source_commit": commit,
                                "source_tree_clean": True, "camera_id": camera}
    if "label" in fields_:
        identity["label"] = str(fields_["label"])
    for name in IDENTITY_HASH_FIELDS:
        identity[name] = _sha_or_raise(fields_[name], name)
    for name in IDENTITY_OPTIONAL_HASH_FIELDS:
        identity[name] = _sha_or_raise(fields_[name], name, allow_none_marker=True)
        if identity[name] == "none" and not str(reasons.get(name, "")).strip():
            raise ValueError(f"{name}='none' requires an explicit reason")
    stray = sorted(name for name in reasons if identity.get(name) != "none")
    if stray:
        raise ValueError(f"none_reasons given for fields that are not 'none': {stray}")
    if identity["scorer_sha256"] != expected_scorer_sha256:
        raise ValueError("Candidate scorer identity differs from the unchanged official scorer")
    if identity["extractor_sha256"] != expected_extractor_sha256:
        raise ValueError("Candidate extractor identity differs from the unchanged official exporter")
    identity["none_reasons"] = {name: reasons[name] for name in sorted(reasons)}
    identity["candidate_id"] = sha256_bytes(canonical_json(identity).encode("ascii"))
    return identity


_IDENTITY_FIELD_NAMES = ("source_commit", "source_tree_clean", "camera_id", "label",
                         *IDENTITY_HASH_FIELDS, *IDENTITY_OPTIONAL_HASH_FIELDS)
_IDENTITY_DERIVED_KEYS = ("schema", "none_reasons", "candidate_id")


def verify_candidate_identity(identity: Any, files: Mapping[str, str | Path] | None = None, *,
                              expected_scorer_sha256: str = OFFICIAL_SCORER_SHA256,
                              expected_extractor_sha256: str = OFFICIAL_EXTRACTOR_SHA256) -> list[str]:
    """Validate the complete identity contract, then compare file-backed fields with bytes.

    The identity is rebuilt canonically with build_candidate_identity. A self-hashed
    partial, extended or altered record therefore fails even when its candidate_id is
    consistent with its own content.
    """
    if not isinstance(identity, Mapping):
        return ["candidate identity must be a mapping"]
    problems = []
    unknown = sorted(set(identity) - set(_IDENTITY_FIELD_NAMES) - set(_IDENTITY_DERIVED_KEYS))
    if unknown:
        problems.append(f"identity has unknown keys {unknown}")
    if identity.get("schema") != SCHEMA + "/candidate":
        problems.append("identity schema is missing or not the candidate schema")
    reasons = identity.get("none_reasons")
    if not isinstance(reasons, Mapping) or not all(isinstance(key, str) and isinstance(value, str)
                                                    for key, value in reasons.items()):
        problems.append("identity none_reasons must be a mapping of field names to reason strings")
        reasons = {}
    try:
        rebuilt = build_candidate_identity({name: identity[name] for name in _IDENTITY_FIELD_NAMES if name in identity},
                                           none_reasons=reasons, expected_scorer_sha256=expected_scorer_sha256,
                                           expected_extractor_sha256=expected_extractor_sha256)
    except ValueError as error:
        problems.append(f"identity contract invalid: {error}")
        rebuilt = None
    if rebuilt is not None:
        if identity.get("candidate_id") != rebuilt["candidate_id"]:
            problems.append("candidate_id does not match the canonical identity content")
        elif not unknown and dict(identity) != rebuilt:
            problems.append("identity differs from its canonical rebuild")
    for name, path in (files or {}).items():
        if name not in (*IDENTITY_HASH_FIELDS, *IDENTITY_OPTIONAL_HASH_FIELDS):
            problems.append(f"{name}: not a file-backed identity field")
        elif not Path(path).is_file():
            problems.append(f"{name}: file missing at {path}")
        elif sha256_file(path) != identity.get(name):
            problems.append(f"{name}: SHA256 of {path} differs from identity")
    return problems


# Strict CSV validation --------------------------------------------------------------------

def normalize_expected_timestamps(values: Iterable[Any]) -> tuple[int, ...]:
    """Strict, unique, sorted expected exposure list."""
    stamps = [strict_int_ns(value, "expected timestamp") for value in values]
    ordered = sorted(stamps)
    duplicate = next((left for left, right in zip(ordered, ordered[1:]) if left == right), None)
    if duplicate is not None:
        raise ValueError(f"Expected timestamp {duplicate} is duplicated in the expected-frame list")
    if not ordered:
        raise ValueError("Expected-frame list is empty")
    return tuple(ordered)


def expected_timestamps_sha256(expected: Sequence[int]) -> str:
    return sha256_bytes(("\n".join(str(value) for value in expected) + "\n").encode("ascii"))


def read_expected_timestamps(path: str | Path) -> tuple[int, ...]:
    """Read a JSON integer array or one integer per line; floats are rejected."""
    source = Path(path)
    text = source.read_text(encoding="utf-8")
    if source.suffix.lower() == ".json":
        values = strict_json_loads(text)
        if not isinstance(values, list):
            raise ValueError("Expected-timestamp JSON must be an array of integers")
        return normalize_expected_timestamps(values)
    values = []
    for line_number, line in enumerate(text.splitlines(), 1):
        if _DIGITS.fullmatch(line) is None:
            raise ValueError(f"Expected-timestamp line {line_number} is not integer nanoseconds")
        values.append(int(line))
    return normalize_expected_timestamps(values)


def expected_from_baseline_run(run_json: str | Path, iter_frames: Callable | None = None) -> dict:
    """Expected outputs of the original runner: every cached exposure at or after the seed.

    The original runner skips exposures earlier than the seed and emits one row for
    each later exposure. iter_frames defaults to official_io.iter_frames, which needs
    the image stack and the laptop cache and re-hashes every cached JPEG: a full cache
    read that belongs under the benchmark-cpu resource.
    """
    run = strict_json_loads(Path(run_json).read_text(encoding="utf-8"))
    seed = strict_int_ns(run["initial_pose"]["timestamp_ns"], "initial pose timestamp")
    if iter_frames is None:
        from .official_io import iter_frames  # noqa: PLC0415  (optional heavy dependency)
    stamps = [strict_int_ns(stamp, "cached exposure") for stamp, _ in iter_frames(run["input_manifest"])]
    expected = normalize_expected_timestamps(stamp for stamp in stamps if stamp >= seed)
    return {"expected": expected, "rule": "official runner: all cached exposures with stamp >= seed timestamp",
            "seed_timestamp_ns": seed, "cached_exposures": len(stamps), "skipped_before_seed": len(stamps) - len(expected),
            "run_json_sha256": sha256_file(run_json)}


def _parse_number(token: str) -> tuple[str, float | None]:
    if _NONFINITE.fullmatch(token):
        return "nonfinite", None
    if _DECIMAL.fullmatch(token) is None:
        return "malformed", None
    value = float(token)
    return ("ok", value) if math.isfinite(value) else ("nonfinite", None)


def _read_pose_csv(path: Path, headers: Sequence[Sequence[str]], yaw_convention: str) -> tuple[dict, list[tuple]]:
    """Parse every line, keeping raw tokens; collect all problems instead of stopping."""
    if yaw_convention not in YAW_CONVENTIONS:
        raise ValueError(f"yaw convention must be one of {YAW_CONVENTIONS}")
    issues = {"header": None, "malformed": [], "nonfinite": [], "yaw_out_of_range": [], "decode_error": None}
    rows: list[tuple] = []
    try:
        text = path.read_bytes().decode("utf-8")
    except UnicodeDecodeError as error:
        issues["decode_error"] = f"not UTF-8: {error.reason} at byte {error.start}"
        return issues, rows
    reader = csv.reader(text.splitlines(keepends=True))
    header = next(reader, None)
    issues["header"] = header
    if header is None or tuple(header) not in {tuple(option) for option in headers}:
        return issues, rows
    yaw_index = list(header).index("yaw")
    for line, values in enumerate(reader, 2):
        if len(values) != len(header):
            issues["malformed"].append({"line": line, "reason": f"expected {len(header)} fields, found {len(values)}"})
            continue
        token = values[0]
        if _DIGITS.fullmatch(token) is None:
            issues["malformed"].append({"line": line, "reason": "timestamp_ns is not unsigned integer text"})
            continue
        stamp = int(token)
        if stamp > INT64_MAX:
            issues["malformed"].append({"line": line, "reason": "timestamp_ns exceeds signed int64"})
            continue
        numbers, bad = [], None
        for column, item in zip(header[1:], values[1:], strict=True):
            kind, number = _parse_number(item)
            if kind != "ok":
                bad = (kind, column)
                break
            numbers.append(number)
        if bad is not None:
            target = issues["nonfinite"] if bad[0] == "nonfinite" else issues["malformed"]
            target.append({"line": line, "timestamp_ns": stamp, "reason": f"{bad[1]} is {bad[0]}"})
            continue
        yaw = numbers[yaw_index - 1]
        if yaw_convention == "wrapped_pi" and not abs(yaw) <= math.pi + YAW_WRAP_TOLERANCE_RAD:
            issues["yaw_out_of_range"].append({"line": line, "timestamp_ns": stamp, "yaw": yaw})
            continue
        rows.append((line, stamp, tuple(values), tuple(numbers)))
    return issues, rows


def _bounded(items: list) -> list:
    return items[:SAMPLE_LIMIT]


def _validate_predictions(path: str | Path, expected: Sequence[int], *, yaw_convention: str) -> tuple[dict, list[tuple]]:
    source = Path(path).resolve()
    expected_set = set(expected)
    issues, rows = _read_pose_csv(source, [SUBMISSION_HEADER], yaw_convention)
    failures = []
    if issues["decode_error"]:
        failures.append(f"CSV {issues['decode_error']}")
    elif issues["header"] is None:
        failures.append("CSV is empty; exact header timestamp_ns,x,y,yaw required")
    elif tuple(issues["header"]) != SUBMISSION_HEADER:
        bom = issues["header"] and issues["header"][0].startswith("\ufeff")
        failures.append("Exact CSV header timestamp_ns,x,y,yaw required" + (" (UTF-8 BOM found)" if bom else ""))
    seen: set[int] = set()
    duplicates, order_violations, extra, unique_rows = [], [], [], []
    latest = None
    for line, stamp, raw, numbers in rows:
        if stamp in seen:
            duplicates.append({"line": line, "timestamp_ns": stamp})
            continue
        seen.add(stamp)
        if latest is not None and stamp < latest:
            order_violations.append({"line": line, "timestamp_ns": stamp, "after_timestamp_ns": latest})
        latest = stamp if latest is None else max(latest, stamp)
        if stamp not in expected_set:
            extra.append({"line": line, "timestamp_ns": stamp})
        unique_rows.append((line, stamp, raw, numbers))
    # A malformed or nonfinite row at an expected exposure still leaves that exposure without output.
    invalid_stamps = {item["timestamp_ns"] for item in issues["malformed"] + issues["nonfinite"] + issues["yaw_out_of_range"]
                      if "timestamp_ns" in item}
    missing = [stamp for stamp in expected if stamp not in seen]
    missing_without_any_row = [stamp for stamp in missing if stamp not in invalid_stamps]
    finite_frames = len(expected) - len(missing)
    checks = {"header": not failures, "malformed_rows": len(issues["malformed"]), "nonfinite_rows": len(issues["nonfinite"]),
              "yaw_out_of_range_rows": len(issues["yaw_out_of_range"]), "duplicate_rows": len(duplicates),
              "order_violations": len(order_violations), "extra_rows": len(extra), "missing_expected": len(missing)}
    for key, label in (("malformed_rows", "malformed rows"), ("nonfinite_rows", "nonfinite rows"),
                       ("yaw_out_of_range_rows", "yaw outside [-pi, pi]"), ("duplicate_rows", "duplicate timestamps"),
                       ("order_violations", "rows out of exposure order"), ("extra_rows", "rows at unexpected timestamps"),
                       ("missing_expected", "expected exposures without a valid finite row")):
        if checks[key]:
            failures.append(f"{checks[key]} {label}")
    report = {
        "path": str(source), "sha256": sha256_file(source), "bytes": source.stat().st_size,
        "header": issues["header"], "yaw_convention": yaw_convention,
        "expected_frames": len(expected), "expected_timestamps_sha256": expected_timestamps_sha256(expected),
        "data_rows_parsed_valid": len(rows), "unique_valid_rows": len(unique_rows),
        "finite_output_frames": finite_frames, "finite_output_fraction": finite_frames / len(expected),
        "finite_output_note": "Continuity only: held or fallback poses count here; this is not accurate localization",
        "checks": checks,
        "samples": {"malformed": _bounded(issues["malformed"]), "nonfinite": _bounded(issues["nonfinite"]),
                    "yaw_out_of_range": _bounded(issues["yaw_out_of_range"]), "duplicates": _bounded(duplicates),
                    "order_violations": _bounded(order_violations), "extra": _bounded(extra),
                    "missing": _bounded(missing), "missing_without_any_row": _bounded(missing_without_any_row)},
        "failures": failures, "ok": not failures,
    }
    return report, unique_rows


def validate_predictions(path: str | Path, expected_timestamps: Iterable[Any], *, yaw_convention: str = "wrapped_pi") -> dict:
    """Check exact header, integer stamps, finite values, order and complete per-frame output.

    All problems are counted; nothing stops at the first error. The official scorer
    sorts rows and bins coverage per second, so it cannot establish this itself.
    """
    report, _ = _validate_predictions(path, normalize_expected_timestamps(expected_timestamps), yaw_convention=yaw_convention)
    return report


# Status coverage ---------------------------------------------------------------------------

def read_jsonl(path: str | Path) -> tuple[list[dict], list[dict]]:
    """Return (records, problems). Nonfinite constants and non-object lines are problems."""
    records, problems = [], []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        try:
            value = strict_json_loads(line)
        except ValueError as error:
            problems.append({"line": line_number, "reason": f"invalid JSON: {error}"})
            continue
        if not isinstance(value, dict):
            problems.append({"line": line_number, "reason": "line is not a JSON object"})
            continue
        records.append(value)
    return records, problems


def summarize_statuses(records: Iterable[Mapping[str, Any]], expected_timestamps: Iterable[Any], vocabulary: str) -> dict:
    """Count every expected exposure by candidate status category.

    Missing, duplicate, extra, malformed and unknown-status records are failures. The
    categories stay disjoint: absolute fixes, map tracking, relative visual motion,
    fallback, held, lost and seed. Only absolute_fix and map_tracked count as localized
    states; none of these counts is an accuracy measurement.
    """
    if vocabulary not in STATUS_VOCABULARIES:
        raise ValueError(f"Unknown status vocabulary {vocabulary!r}; declare one of {sorted(STATUS_VOCABULARIES)}")
    mapping = STATUS_VOCABULARIES[vocabulary]
    expected = normalize_expected_timestamps(expected_timestamps)
    expected_set = set(expected)
    seen: dict[int, str] = {}
    malformed, duplicates, extra, unknown = [], [], [], []
    for index, record in enumerate(records):
        try:
            stamp = strict_int_ns(record.get("timestamp_ns"))
        except ValueError as error:
            malformed.append({"record": index, "reason": str(error)})
            continue
        status = record.get("status")
        if stamp in seen:
            duplicates.append(stamp)
            continue
        if stamp not in expected_set:
            extra.append(stamp)
            continue
        if not isinstance(status, str) or status not in mapping:
            unknown.append({"timestamp_ns": stamp, "status": status})
            continue
        seen[stamp] = mapping[status]
    unknown_stamps = {item["timestamp_ns"] for item in unknown}
    missing = [stamp for stamp in expected if stamp not in seen and stamp not in unknown_stamps]
    counts = {category: 0 for category in STATUS_CATEGORIES}
    for category in seen.values():
        counts[category] += 1
    total = len(expected)
    failures = []
    for items, label in ((malformed, "malformed status records"), (duplicates, "duplicate status timestamps"),
                         (extra, "status records at unexpected timestamps"), (unknown, "unknown statuses"),
                         (missing, "expected exposures without a status record")):
        if items:
            failures.append(f"{len(items)} {label}")
    localized = sum(counts[name] for name in LOCALIZED_CATEGORIES)
    return {"vocabulary": vocabulary, "mapping": dict(mapping), "expected_frames": total,
            "category_counts": counts, "category_fractions": {name: counts[name] / total for name in STATUS_CATEGORIES},
            "absolute_fix_frames": counts["absolute_fix"], "absolute_fix_fraction": counts["absolute_fix"] / total,
            "localized_state_frames": localized, "localized_state_fraction": localized / total,
            "unlocalized_state_frames": total - localized,
            "note": "Status coverage comes from candidate diagnostics; accuracy is only the official scorer output",
            "samples": {"malformed": _bounded(malformed), "duplicates": _bounded(duplicates), "extra": _bounded(extra),
                        "unknown": _bounded(unknown), "missing": _bounded(missing)},
            "failures": failures, "ok": not failures}


# Official scorer invocation ----------------------------------------------------------------

def parse_scorer_stdout(stdout: str) -> dict:
    """Parse every required printed metric; missing, repeated or nonfinite values are reported."""
    metrics: dict[str, Any] = {}
    printed: dict[str, str] = {}
    missing, duplicated, invalid = [], [], []

    def accept(name: str, token: str) -> None:
        printed[name] = token
        if name in INTEGER_SCORER_METRICS:
            metrics[name] = int(token) if _DIGITS.fullmatch(token) else None
        else:
            kind, value = _parse_number(token)
            metrics[name] = value if kind == "ok" else None
        if metrics[name] is None:
            invalid.append(name)

    for name, pattern in SCORER_METRIC_PATTERNS.items():
        found = re.findall(pattern, stdout)
        if not found:
            missing.append(name)
        elif len(found) > 1:
            duplicated.append(name)
        else:
            accept(name, found[0])
    pairs = re.findall(SCORER_PAIR_PATTERN, stdout)
    if not pairs:
        missing.extend(SCORER_PAIR_METRICS)
    elif len(pairs) > 1:
        duplicated.extend(SCORER_PAIR_METRICS)
    else:
        for name, token in zip(SCORER_PAIR_METRICS, pairs[0], strict=True):
            accept(name, token)
    return {"metrics": metrics, "printed": printed, "missing": missing, "duplicated": duplicated,
            "nonfinite_or_unparseable": invalid, "not_ranked_printed": "NOT RANKED" in stdout,
            "metrics_complete": not (missing or duplicated or invalid),
            "metric_precision": "official printed precision; raw stdout retained byte-exact"}


def scorer_failures(scope: Mapping[str, Any], *, expected_submitted: int | None = None) -> list[str]:
    """Every reason a scorer scope must not be used as a measurement."""
    if scope.get("status") == "not_scored":
        return [f"required scope was not scored: {scope.get('reason', 'no reason recorded')}"]
    failures = list(scope.get("custody_failures", []))
    if scope.get("status") == "refused":
        return failures
    if scope.get("timed_out"):
        failures.append("official scorer timed out")
    if scope.get("launch_error"):
        failures.append(f"official scorer could not start: {scope['launch_error']}")
    if scope.get("exit_code") != 0:
        failures.append(f"official scorer exit code {scope.get('exit_code', 'missing')}")
    parsed = scope.get("parsed") or {}
    if parsed.get("missing"):
        failures.append("missing required metrics: " + ",".join(parsed["missing"]))
    if parsed.get("duplicated"):
        failures.append("ambiguous repeated metrics: " + ",".join(parsed["duplicated"]))
    if parsed.get("nonfinite_or_unparseable"):
        failures.append("nonfinite or unparseable metrics: " + ",".join(parsed["nonfinite_or_unparseable"]))
    if parsed.get("not_ranked_printed"):
        failures.append("official scorer printed NOT RANKED")
    metrics = parsed.get("metrics") or {}
    coverage = metrics.get("coverage_percent")
    if isinstance(coverage, float) and coverage < RANKING_COVERAGE_PERCENT and not parsed.get("not_ranked_printed"):
        failures.append("coverage below ranking threshold without NOT RANKED; scorer state inconsistent")
    submitted, evaluated = metrics.get("poses_submitted"), metrics.get("poses_evaluated")
    if expected_submitted is not None and isinstance(submitted, int) and submitted != expected_submitted:
        failures.append(f"scorer counted {submitted} submitted poses, file has {expected_submitted}")
    if isinstance(evaluated, int) and (evaluated == 0 or (isinstance(submitted, int) and evaluated > submitted)):
        failures.append(f"scorer evaluated {evaluated} of {submitted} submitted poses")
    return failures


def run_official_scorer(scorer: str | Path, gt_csv: str | Path, pred_csv: str | Path, evidence: EvidenceWriter,
                        name: str, *, expected_scorer_sha256: str = OFFICIAL_SCORER_SHA256,
                        expected_submitted: int | None = None, python: str = sys.executable,
                        timeout_s: float = 1800.0) -> dict:
    """Run the unchanged scorer in its own process; save raw output before interpreting it."""
    script, gt_path, pred_path = Path(scorer).resolve(), Path(gt_csv).resolve(), Path(pred_csv).resolve()
    command = [python, str(script), str(gt_path), str(pred_path)]
    scope: dict[str, Any] = {"name": name, "command": command, "cwd": str(script.parent),
                             "expected_scorer_sha256": expected_scorer_sha256, "custody_failures": []}
    before = {"scorer": sha256_file(script), "ground_truth": sha256_file(gt_path), "predictions": sha256_file(pred_path)}
    scope["input_sha256"] = before
    scope["script_sha256"] = before["scorer"]
    if before["scorer"] != expected_scorer_sha256:
        scope.update(status="refused", exit_code=None, parsed=None)
        scope["custody_failures"].append("scorer SHA256 differs from the unchanged official scorer; not executed")
        scope["failures"] = scorer_failures(scope)
        evidence.write_json(f"{name}.scorer.json", scope)
        return scope
    started = time.monotonic_ns()
    stdout, stderr = b"", b""
    try:
        result = subprocess.run(command, cwd=script.parent, capture_output=True, timeout=timeout_s, check=False)
        stdout, stderr, scope["exit_code"] = result.stdout, result.stderr, result.returncode
    except subprocess.TimeoutExpired as error:
        stdout, stderr = error.stdout or b"", error.stderr or b""
        scope.update(exit_code=None, timed_out=True)
    except OSError as error:
        scope.update(exit_code=None, launch_error=f"{type(error).__name__}: {error}")
    scope["elapsed_monotonic_ns"] = time.monotonic_ns() - started
    scope["stdout_file"] = evidence.write_bytes(f"{name}.stdout.txt", stdout)
    scope["stderr_file"] = evidence.write_bytes(f"{name}.stderr.txt", stderr)
    after = {"scorer": sha256_file(script), "ground_truth": sha256_file(gt_path), "predictions": sha256_file(pred_path)}
    for key in before:
        if after[key] != before[key]:
            scope["custody_failures"].append(f"{key} bytes changed during scoring")
    try:
        stdout.decode("utf-8")
        scope["stdout_utf8_valid"] = True
    except UnicodeDecodeError:
        scope["stdout_utf8_valid"] = False  # parsed with replacement characters; raw bytes kept
    text = stdout.decode("utf-8", errors="replace")
    scope["stdout"] = text
    scope["stderr"] = stderr.decode("utf-8", errors="replace")
    scope["parsed"] = parse_scorer_stdout(text)
    scope["failures"] = scorer_failures(scope, expected_submitted=expected_submitted)
    scope["status"] = "failed" if scope["failures"] else "scored"
    evidence.write_json(f"{name}.scorer.json", {key: value for key, value in scope.items() if key not in ("stdout", "stderr")})
    return scope


# Interval scoring --------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class ScoringInterval:
    """A scored slice of one continuous prediction stream. No reseeding is implied."""

    name: str
    start_ns: int
    end_ns: int
    convention: str
    membership: str

    def __post_init__(self) -> None:
        if _SAFE_NAME.fullmatch(self.name) is None:
            raise ValueError(f"Unsafe interval name {self.name!r}")
        start, end = strict_int_ns(self.start_ns, "interval start"), strict_int_ns(self.end_ns, "interval end")
        if start >= end:
            raise ValueError("Interval start must precede its end")
        if self.convention not in INTERVAL_CONVENTIONS:
            raise ValueError(f"Interval convention must be one of {INTERVAL_CONVENTIONS}; no default is assumed")
        if self.membership not in MEMBERSHIPS[1:]:
            raise ValueError("Interval membership must be fit, tune or audit")

    def contains(self, stamp: int) -> bool:
        return self.start_ns <= stamp <= self.end_ns if self.convention == "closed" else self.start_ns <= stamp < self.end_ns


def intervals_from_split(split: Mapping[str, Any], scopes: Sequence[str]) -> list[ScoringInterval]:
    """Build intervals from the team membership split or the original baseline split.

    team/contracts/split.json fit/tune/audit are half-open. The original baseline
    reserved_laps are closed and are the final two laps, so they are audit membership.
    """
    result = []
    for scope in scopes:
        if scope in ("fit", "tune", "audit"):
            if split.get("schema") != "schubki-map-membership-v1" or split.get("membership") != "start_inclusive_end_exclusive":
                raise ValueError("fit/tune/audit scopes require the half-open schubki-map-membership-v1 split")
            block = split[scope]
            laps = "-".join(str(lap) for lap in block["lap_ids"])
            result.append(ScoringInterval(f"{scope}-laps-{laps}", block["start_ns"], block["end_ns"], "half_open", scope))
        elif scope == "baseline_reserved":
            if split.get("lap_reseeding_allowed", False):
                raise ValueError("Lap reseeding is forbidden")
            for index, lap in enumerate(split["reserved_laps"], 1):
                result.append(ScoringInterval(f"baseline-reserved-{index:02d}", lap["start_ns"], lap["end_ns"], "closed", "audit"))
        else:
            raise ValueError(f"Unknown scope {scope!r}")
    return result


def _ordered_ground_truth(rows: list[tuple]) -> list[tuple]:
    ordered = sorted(rows, key=lambda row: row[1])
    unique: list[tuple] = []
    for row in ordered:
        if unique and row[1] == unique[-1][1]:
            if row[3] != unique[-1][3]:
                raise ValueError(f"Conflicting ground-truth poses at timestamp {row[1]}")
            continue
        unique.append(row)
    return unique


def _eligible_count(gt_times: Sequence[int], stamps: Iterable[int]) -> int:
    """Predictions with an official interpolation bracket (same rule as official_evaluation)."""
    count = 0
    for stamp in stamps:
        index = bisect_left(gt_times, stamp)
        if 0 < index < len(gt_times) and gt_times[index] - gt_times[index - 1] <= SCORER_MAX_GT_GAP_NS:
            count += 1
    return count


def _require_files(**paths: str | Path) -> None:
    """Reject absent inputs before an evidence directory is created."""
    absent = [f"{name}={path}" for name, path in paths.items() if not Path(path).is_file()]
    if absent:
        raise FileNotFoundError(f"Required input files are missing: {', '.join(absent)}")


def _write_csv(evidence: EvidenceWriter, name: str, header: Sequence[str], rows: Iterable[tuple]) -> Path:
    lines = [",".join(header)] + [",".join(row[2]) for row in rows]
    evidence.write_text(name, "\n".join(lines) + "\n")
    return evidence.path(name)


def load_split(path: str | Path) -> tuple[dict, dict]:
    """Read a split file once; return (parsed split, split_source with the SHA256 of those bytes).

    Scopes and reserved bounds must be derived from the returned dictionary, so the
    hash that is checked against candidate_identity.split_sha256 is the hash of exactly
    the bytes that defined them.
    """
    source = Path(path).resolve()
    data = source.read_bytes()
    split = strict_json_loads(data.decode("utf-8"))
    if not isinstance(split, dict):
        raise ValueError("Split file must contain a JSON object")
    return split, {"path": str(source), "sha256": sha256_bytes(data), "bytes": len(data)}


def reserved_intervals_from_split(split: Mapping[str, Any]) -> list[ScoringInterval]:
    """Reserved audit membership declared by a split.

    The team split's audit block (laps 6-7, half-open) or the original baseline's
    reserved_laps (the same final two laps, closed). A split declaring neither fails.
    """
    if split.get("schema") == "schubki-map-membership-v1":
        return intervals_from_split(split, ["audit"])
    if split.get("reserved_laps"):
        return intervals_from_split(split, ["baseline_reserved"])
    raise ValueError("Split declares no reserved audit intervals")


def gt_slice_bounds(gt_times: Sequence[int], interval: ScoringInterval) -> tuple[int, int]:
    """Half-open GT row range a scorer slice needs for this interval.

    Under the official left-search rule a prediction at t uses GT rows index-1 and index,
    where index = bisect_left(gt_times, t). The slice therefore always keeps the GT row
    before the interval start, including when the start equals a GT sample, and the
    first GT row at or after the interval end. These bracket rows are evaluator-only.
    """
    left = bisect_left(gt_times, interval.start_ns)
    if left > 0:
        left -= 1
    right = bisect_right(gt_times, interval.end_ns)
    if right < len(gt_times) and (right == 0 or gt_times[right - 1] != interval.end_ns):
        right += 1
    return left, right


REPRODUCTION_EXCEPTION = ("old-baseline reproduction: re-scores the preserved original baseline CSV, whose full and "
                          "reserved-lap scores are already exposed; no new candidate, no identity binding, no inference")


def score_candidate(gt_csv: str | Path, pred_csv: str | Path, scorer: str | Path, evidence_dir: str | Path, *,
                    expected_timestamps: Iterable[Any], candidate_identity: Mapping[str, Any] | None,
                    reserved_intervals: Sequence[ScoringInterval], split_source: Mapping[str, Any],
                    intervals: Sequence[ScoringInterval] = (),
                    score_full: bool = False, identity_files: Mapping[str, str | Path] | None = None,
                    status_records: Iterable[Mapping[str, Any]] | None = None, status_vocabulary: str | None = None,
                    audit_authorization: Mapping[str, Any] | None = None,
                    expected_scorer_sha256: str = OFFICIAL_SCORER_SHA256,
                    expected_extractor_sha256: str = OFFICIAL_EXTRACTOR_SHA256, yaw_convention: str = "wrapped_pi",
                    python: str = sys.executable, timeout_s: float = 1800.0, label: str = "candidate") -> dict:
    """Validate one continuous candidate stream, score only the requested scopes, seal evidence.

    No scope is implicit. The full stream is scored only when score_full is True, and
    each interval is a slice of the same continuous predictions (no reseeding).
    reserved_intervals must declare the audit membership (laps 6-7). Any requested scope
    containing a reserved exposure, including a full stream, needs a recorded coordinator
    decision and a validation-audit claim fingerprint; otherwise the scorer is not called.
    split_source records the path and SHA256 of the exact split bytes the intervals and
    reserved bounds came from (see load_split); a hash different from
    candidate_identity.split_sha256 is a recorded failure before any scorer call.
    A complete candidate identity is required; None or a malformed identity is a
    recorded failure. Call require_success afterwards; the sealed evidence already exists.
    """
    reserved = list(reserved_intervals)
    if not reserved or any(interval.membership != "audit" for interval in reserved):
        raise ValueError("reserved_intervals must declare at least one audit-membership interval")
    return _score_stream(gt_csv, pred_csv, scorer, evidence_dir, expected_timestamps=expected_timestamps,
                         candidate_identity=candidate_identity, reserved_intervals=reserved, split_source=split_source,
                         intervals=intervals,
                         score_full=score_full, identity_files=identity_files, status_records=status_records,
                         status_vocabulary=status_vocabulary, audit_authorization=audit_authorization,
                         expected_scorer_sha256=expected_scorer_sha256, expected_extractor_sha256=expected_extractor_sha256,
                         yaw_convention=yaw_convention, python=python, timeout_s=timeout_s, label=label, exception=None)


def _audit_authorized(authorization: Mapping[str, Any] | None) -> bool:
    return bool(authorization is not None and str(authorization.get("decision_ref", "")).strip()
                and authorization.get("resource") == AUDIT_RESOURCE
                and _SHA256.fullmatch(str(authorization.get("claim_fingerprint_sha256", ""))) is not None)


def _score_stream(gt_csv: str | Path, pred_csv: str | Path, scorer: str | Path, evidence_dir: str | Path, *,
                  expected_timestamps: Iterable[Any], candidate_identity: Mapping[str, Any] | None,
                  reserved_intervals: Sequence[ScoringInterval], split_source: Mapping[str, Any] | None,
                  intervals: Sequence[ScoringInterval], score_full: bool,
                  identity_files: Mapping[str, str | Path] | None, status_records: Iterable[Mapping[str, Any]] | None,
                  status_vocabulary: str | None, audit_authorization: Mapping[str, Any] | None,
                  expected_scorer_sha256: str, expected_extractor_sha256: str, yaw_convention: str, python: str,
                  timeout_s: float, label: str, exception: str | None) -> dict:
    names = [interval.name for interval in intervals]
    if "full" in names or len(set(names)) != len(names):
        raise ValueError("Interval names must be unique and must not be 'full'")
    if not score_full and not intervals:
        raise ValueError("Request at least one scope: score_full or an interval; no scope is implicit")
    if status_records is not None and status_vocabulary not in STATUS_VOCABULARIES:
        raise ValueError(f"Status records require a declared vocabulary from {sorted(STATUS_VOCABULARIES)}")
    if yaw_convention not in YAW_CONVENTIONS:
        raise ValueError(f"yaw convention must be one of {YAW_CONVENTIONS}")
    _require_files(gt_csv=gt_csv, pred_csv=pred_csv, scorer=scorer)
    expected = normalize_expected_timestamps(expected_timestamps)
    evidence = EvidenceWriter(evidence_dir)
    report: dict[str, Any] = {
        "schema": SCHEMA + "/score", "label": label, "evidence_dir": str(evidence.directory),
        "candidate_identity": dict(candidate_identity) if candidate_identity is not None else None,
        "identity_exception": exception,
        "requested_scopes": (["full"] if score_full else []) + names,
        "reserved_intervals": [asdict(interval) for interval in reserved_intervals],
        "split_source": dict(split_source) if isinstance(split_source, Mapping) else None,
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "environment": environment_identity(),
        "information_boundary": ("Ground truth is read only here for bracket slicing and by the unchanged scorer; "
                                 "never an estimator input"),
        "continuous_stream": True, "lap_reseeding": False, "scopes": [], "failures": [],
    }
    failures: list[str] = report["failures"]
    if exception is None:
        if candidate_identity is None:
            failures.append("identity: a complete candidate identity is required for a bound candidate score")
        else:
            failures.extend(f"identity: {item}" for item in verify_candidate_identity(
                candidate_identity, identity_files, expected_scorer_sha256=expected_scorer_sha256,
                expected_extractor_sha256=expected_extractor_sha256))
            report["identity_files_verified"] = sorted(identity_files or {})
        if candidate_identity is None and identity_files:
            failures.append("identity: files supplied without a candidate identity")
        split_sha = split_source.get("sha256") if isinstance(split_source, Mapping) else None
        if not isinstance(split_sha, str) or _SHA256.fullmatch(split_sha) is None \
                or not isinstance(split_source.get("path"), str):
            failures.append("split: split_source must record the path and SHA256 of the split bytes actually used")
        elif isinstance(candidate_identity, Mapping) and candidate_identity.get("split_sha256") != split_sha:
            failures.append(f"identity: split file SHA256 {split_sha} at {split_source['path']} differs from "
                            "candidate_identity.split_sha256; scopes and reserved bounds are not the bound split")
    validation, rows = _validate_predictions(pred_csv, expected, yaw_convention=yaw_convention)
    report["prediction_validation"] = validation
    evidence.write_json("prediction-validation.json", validation)
    failures.extend(f"predictions: {item}" for item in validation["failures"])
    if status_records is not None:
        statuses = summarize_statuses(status_records, expected, status_vocabulary)
        report["status_coverage"] = statuses
        failures.extend(f"statuses: {item}" for item in statuses["failures"])
    else:
        report["status_coverage"] = None
    report["disclosures"] = []
    if score_full:
        report["disclosures"].append("Full-run scores include map-fit laps 1-3 and cannot establish generalization")

    def reserved_count(contains: Callable[[int], bool]) -> int:
        return sum(1 for stamp in expected if contains(stamp) and any(item.contains(stamp) for item in reserved_intervals))

    touching = {}
    if score_full:
        touching["full"] = reserved_count(lambda stamp: True)
    for interval in intervals:
        touching[interval.name] = reserved_count(interval.contains)
    report["reserved_frames_by_scope"] = touching
    audit_scopes = sorted({name for name, count in touching.items() if count}
                          | {interval.name for interval in intervals if interval.membership == "audit"})
    report["audit_scopes"] = audit_scopes
    report["audit_authorization"] = dict(audit_authorization) if audit_authorization else None
    if exception is not None:
        report["disclosures"].append(f"Authorized exception: {exception}")
    elif audit_scopes:
        report["disclosures"].append("Audit laps 6-7 are reserved same-session validation; their baseline scores are already exposed")
        if not _audit_authorized(audit_authorization):
            failures.append(f"scopes {audit_scopes} include reserved audit exposures without a recorded coordinator "
                            "decision and validation-audit resource")
    gt_path = Path(gt_csv).resolve()
    gt_issues, gt_rows = _read_pose_csv(gt_path, [SUBMISSION_HEADER, GT_HEADER], "finite")
    gt_problems = ([gt_issues["decode_error"]] if gt_issues["decode_error"] else []) + \
        (["ground-truth header invalid"] if not gt_issues["decode_error"] and (gt_issues["header"] is None or
         tuple(gt_issues["header"]) not in (SUBMISSION_HEADER, GT_HEADER)) else []) + \
        [f"ground-truth line {item['line']}: {item['reason']}" for item in gt_issues["malformed"] + gt_issues["nonfinite"]]
    gt: list[tuple] = []
    if not gt_problems:
        try:
            gt = _ordered_ground_truth(gt_rows)
        except ValueError as error:
            gt_problems.append(str(error))
        if len(gt) < 2:
            gt_problems.append("ground truth requires at least 2 distinct timestamps")
    report["ground_truth"] = {"path": str(gt_path), "sha256": sha256_file(gt_path), "rows": len(gt_rows),
                              "distinct_rows": len(gt), "problems": _bounded(gt_problems)}
    failures.extend(f"ground truth: {item}" for item in gt_problems)
    gt_times = [row[1] for row in gt]
    if failures:
        report["scorer_called"] = False
        report["scorer_not_called_reason"] = "preconditions failed; no partial measurement is produced"
    else:
        report["scorer_called"] = True
        if score_full:
            eligible = _eligible_count(gt_times, (row[1] for row in rows))
            if eligible == 0:
                full = {"name": "full", "status": "not_scored", "reason": "no prediction has an eligible ground-truth bracket"}
                full["failures"] = scorer_failures(full)
            else:
                full = run_official_scorer(scorer, gt_path, pred_csv, evidence, "full",
                                           expected_scorer_sha256=expected_scorer_sha256,
                                           expected_submitted=len(rows), python=python, timeout_s=timeout_s)
            full.update(membership="full", convention=None, prediction_rows=len(rows), eligible_predictions=eligible,
                        eligible_against_full_gt=eligible, reserved_frames=touching["full"],
                        source_prediction_sha256=validation["sha256"])
            _record_bracket_rule(full)
            report["scopes"].append(full)
        for interval in intervals:
            selected = [row for row in rows if interval.contains(row[1])]
            scope: dict[str, Any] = {"name": interval.name, "membership": interval.membership, "convention": interval.convention,
                                     "start_ns": interval.start_ns, "end_ns": interval.end_ns, "prediction_rows": len(selected),
                                     "reserved_frames": touching[interval.name],
                                     "source_prediction_sha256": validation["sha256"], "prediction_slice_only": True,
                                     "lap_reseeded": False}
            if not gt_times or interval.start_ns < gt_times[0] or interval.end_ns > gt_times[-1]:
                scope.update(status="not_scored", reason="interval lies outside ground-truth coverage")
            else:
                left, right = gt_slice_bounds(gt_times, interval)
                selected_gt = gt[left:right]
                eligible = _eligible_count([row[1] for row in selected_gt], (row[1] for row in selected))
                against_full = _eligible_count(gt_times, (row[1] for row in selected))
                scope.update(eligible_predictions=eligible, eligible_against_full_gt=against_full,
                             gt_slice_rows=len(selected_gt), gt_slice_first_ns=selected_gt[0][1] if selected_gt else None,
                             gt_slice_last_ns=selected_gt[-1][1] if selected_gt else None)
                if eligible != against_full:
                    scope.update(status="not_scored", reason=(f"GT slice keeps {eligible} official brackets but the full "
                                                              f"truth gives {against_full}; slice would drop predictions"))
                elif len(selected_gt) < 2 or len(selected) < 2 or eligible == 0:
                    scope.update(status="not_scored", reason="insufficient predictions or eligible ground-truth brackets")
                else:
                    gt_slice = _write_csv(evidence, f"{interval.name}.gt.csv", gt_issues["header"], selected_gt)
                    pred_slice = _write_csv(evidence, f"{interval.name}.pred.csv", SUBMISSION_HEADER, selected)
                    scope.update(run_official_scorer(scorer, gt_slice, pred_slice, evidence, interval.name,
                                                     expected_scorer_sha256=expected_scorer_sha256,
                                                     expected_submitted=len(selected), python=python, timeout_s=timeout_s))
                    _record_bracket_rule(scope)
            if scope.get("status") == "not_scored":
                scope["failures"] = scorer_failures(scope)
            report["scopes"].append(scope)
        for scope in report["scopes"]:
            failures.extend(f"{scope['name']}: {item}" for item in scope["failures"])
    report["ok"] = not failures
    report["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    report["evidence_files"] = evidence.files()
    evidence.write_json("score-report.json", _without_text(report))
    report["evidence_manifest"] = evidence.seal()
    return report


def _record_bracket_rule(scope: dict) -> None:
    """Diagnostic only: does the scorer's evaluated count equal the official bracket rule?"""
    evaluated = ((scope.get("parsed") or {}).get("metrics") or {}).get("poses_evaluated")
    scope["evaluated_matches_bracket_rule"] = None if evaluated is None else evaluated == scope.get("eligible_against_full_gt")


def _without_text(report: Mapping[str, Any]) -> dict:
    """Drop duplicated raw stdout/stderr from JSON; byte-exact copies are separate files."""
    result = dict(report)
    result["scopes"] = [{key: value for key, value in scope.items() if key not in ("stdout", "stderr")}
                        for scope in report.get("scopes", [])]
    return result


def require_success(report: Mapping[str, Any], what: str = "Measurement") -> None:
    if not report.get("ok"):
        raise MeasurementFailure(f"{what} failed", report.get("failures") or ["no success flag recorded"],
                                 report.get("evidence_dir"))


# Baseline reproduction and comparison ----------------------------------------------------

PRESERVED_METRICS_PATH = ("official_evaluation", "full", "metrics")


def preserved_full_scope(preserved: Any) -> dict:
    """Return preserved['official_evaluation']['full'] (layout confirmed by the coordinator).

    Any other structure, a missing metrics dictionary or a missing required metric fails.
    """
    node: Any = preserved
    for key in PRESERVED_METRICS_PATH[:-1]:
        if not isinstance(node, Mapping) or key not in node:
            raise ValueError(f"Preserved report lacks {'.'.join(PRESERVED_METRICS_PATH)} (missing {key!r})")
        node = node[key]
    metrics = node.get("metrics") if isinstance(node, Mapping) else None
    if not isinstance(metrics, Mapping):
        raise ValueError(f"Preserved report lacks a {'.'.join(PRESERVED_METRICS_PATH)} dictionary")
    absent = [name for name in REQUIRED_SCORER_METRICS if name not in metrics]
    if absent:
        raise ValueError(f"Preserved metrics lack required fields {absent}")
    return dict(node)


def check_reproduction(full_scope: Mapping[str, Any], preserved_metrics: Mapping[str, Any], *,
                       preserved_stdout: str | None = None,
                       reference: Mapping[str, float] = BASELINE_REFERENCE_METRICS) -> dict:
    """Every required printed metric must equal the preserved value and the stated references."""
    parsed = (full_scope.get("parsed") or {}).get("metrics") or {}
    mismatches, failures = [], list(full_scope.get("failures") or [])
    for name in REQUIRED_SCORER_METRICS:
        old, new = preserved_metrics.get(name), parsed.get(name)
        if old is None or new is None or old != new:
            mismatches.append({"metric": name, "preserved": old, "rescored": new})
    for name, value in reference.items():
        if parsed.get(name) != value:
            failures.append(f"{name} rescored {parsed.get(name)} differs from reference {value}")
    if mismatches:
        failures.append(f"{len(mismatches)} printed metrics differ from preserved evidence")
    # The preserved report decoded stdout in text mode; raw Windows bytes carry CRLF.
    stdout = None if full_scope.get("stdout") is None else full_scope["stdout"].replace("\r\n", "\n")
    preserved_stdout = None if preserved_stdout is None else preserved_stdout.replace("\r\n", "\n")
    identical = None if preserved_stdout is None or stdout is None else stdout == preserved_stdout
    diff = []
    if identical is False:
        old_lines, new_lines = preserved_stdout.splitlines(), stdout.splitlines()
        for index in range(max(len(old_lines), len(new_lines))):
            left = old_lines[index] if index < len(old_lines) else None
            right = new_lines[index] if index < len(new_lines) else None
            if left != right:
                diff.append({"line": index + 1, "preserved": left, "rescored": right})
    return {"ok": not failures, "failures": failures, "mismatches": mismatches, "reference": dict(reference),
            "stdout_identical": identical, "stdout_differences": _bounded(diff),
            "stdout_note": ("Metric equality is the criterion; stdout identity (CRLF normalized to LF) is recorded "
                            "because the scorer may print paths")}


def reproduce_baseline(gt_csv: str | Path, pred_csv: str | Path, scorer: str | Path, preserved_report: str | Path,
                       evidence_dir: str | Path, *, expected_timestamps: Iterable[Any] | None = None,
                       expected_rule: str | None = None, expected_scorer_sha256: str = OFFICIAL_SCORER_SHA256,
                       python: str = sys.executable, timeout_s: float = 1800.0) -> dict:
    """Re-score preserved baseline predictions with the unchanged scorer; no inference is run.

    This is the one labelled exception to the candidate rules: the full continuous run of
    the original baseline (whose full and reserved-lap scores are already exposed) is
    re-scored without a candidate identity or audit decision. No separate audit slice is
    scored. Without an expected-frame list, per-frame completeness is explicitly unverified.
    The preserved inputs are only read; reproduction.json and the nested score
    evidence are sealed before the result is returned, including on failure.
    """
    if expected_timestamps is not None:
        expected_timestamps = normalize_expected_timestamps(expected_timestamps)
    _require_files(gt_csv=gt_csv, pred_csv=pred_csv, scorer=scorer)
    outer = EvidenceWriter(evidence_dir)
    preserved_path = Path(preserved_report).resolve()
    result: dict[str, Any] = {"schema": SCHEMA + "/baseline-reproduction", "inference_rerun": False,
                              "audit_slices_scored": False, "failures": [],
                              "preserved_report": {"path": str(preserved_path), "metrics_at": ".".join(PRESERVED_METRICS_PATH)}}
    failures: list[str] = result["failures"]
    try:
        result["preserved_report"]["sha256"] = sha256_file(preserved_path)
        preserved_full = preserved_full_scope(strict_json_loads(preserved_path.read_text(encoding="utf-8")))
    except (OSError, ValueError) as error:
        failures.append(f"preserved report unusable: {error}")
        preserved_full = None
    completeness = expected_timestamps is not None
    if preserved_full is not None and expected_timestamps is None:
        _, rows = _read_pose_csv(Path(pred_csv), [SUBMISSION_HEADER], "wrapped_pi")
        expected_timestamps = sorted({row[1] for row in rows})
        if not expected_timestamps:
            failures.append("preserved predictions contain no valid rows and no expected-frame list was supplied")
    result["completeness_verified"] = completeness
    result["expected_rule"] = expected_rule if completeness else "not supplied; rows checked only against themselves"
    result["identity_exception"] = REPRODUCTION_EXCEPTION
    if not failures:
        score = _score_stream(gt_csv, pred_csv, scorer, outer.nested_directory("score"),
                              expected_timestamps=expected_timestamps, candidate_identity=None, reserved_intervals=(),
                              split_source=None, intervals=(), score_full=True, identity_files=None, status_records=None,
                              status_vocabulary=None, audit_authorization=None,
                              expected_scorer_sha256=expected_scorer_sha256,
                              expected_extractor_sha256=OFFICIAL_EXTRACTOR_SHA256, yaw_convention="wrapped_pi",
                              python=python, timeout_s=timeout_s, label="baseline-reproduction",
                              exception=REPRODUCTION_EXCEPTION)
        full = next((scope for scope in score["scopes"] if scope["name"] == "full"), {"failures": ["full scope missing"]})
        check = check_reproduction(full, preserved_full["metrics"], preserved_stdout=preserved_full.get("stdout"))
        preserved_inputs = preserved_full.get("input_sha256") or {}
        actual_inputs = {"ground_truth": score["ground_truth"]["sha256"], "predictions": score["prediction_validation"]["sha256"]}
        for key, value in actual_inputs.items():
            if key in preserved_inputs and preserved_inputs[key] != value:
                failures.append(f"preserved {key} SHA256 differs from the preserved report")
        if preserved_full.get("script_sha256") not in (None, full.get("script_sha256")):
            failures.append("preserved scorer SHA256 differs from the scorer used now")
        scope_failures = set(full.get("failures") or [])
        failures.extend([*score["failures"], *(item for item in check["failures"] if item not in scope_failures)])
        result.update(preserved_metrics=preserved_full["metrics"], rescored_metrics=(full.get("parsed") or {}).get("metrics"),
                      comparison=check, input_sha256=actual_inputs, preserved_input_sha256=preserved_inputs,
                      score_evidence_dir=score["evidence_dir"], score_ok=score["ok"])
    result["ok"] = not failures
    outer.write_json("reproduction.json", result)
    result["evidence_dir"] = str(outer.directory)
    result["evidence_manifest"] = outer.seal()
    return result


def _identity_bindings(report: Mapping[str, Any]) -> dict:
    identity = report.get("candidate_identity") or {}
    return {"input_recording_sha256": identity.get("input_recording_sha256"), "split_sha256": identity.get("split_sha256"),
            "scorer_sha256": identity.get("scorer_sha256"), "extractor_sha256": identity.get("extractor_sha256"),
            "ground_truth_sha256": (report.get("ground_truth") or {}).get("sha256"),
            "expected_timestamps_sha256": (report.get("prediction_validation") or {}).get("expected_timestamps_sha256"),
            "reserved_intervals": report.get("reserved_intervals"),
            "scopes": [[scope.get("name"), scope.get("membership"), scope.get("convention"), scope.get("start_ns"), scope.get("end_ns")]
                       for scope in report.get("scopes", [])]}


def compare_candidates(reference: Mapping[str, Any], candidate: Mapping[str, Any], *,
                       expected_scorer_sha256: str = OFFICIAL_SCORER_SHA256,
                       expected_extractor_sha256: str = OFFICIAL_EXTRACTOR_SHA256) -> dict:
    """Side-by-side official metrics for two successful, identically scoped score reports.

    Both reports need a complete, canonically valid candidate identity; an unbound
    old-baseline reproduction is not comparable. Deltas are candidate minus reference for
    printed official metrics only. No composite score is formed; lower is better for
    RMSE/median/p95, higher for coverage/within.
    """
    failures = []
    for name, report in (("reference", reference), ("candidate", candidate)):
        if not report.get("ok"):
            failures.append(f"{name} score report is not successful")
        if report.get("identity_exception"):
            failures.append(f"{name} is an unbound exception report, not a candidate measurement")
        if not report.get("candidate_identity"):
            failures.append(f"{name} has no candidate identity")
        else:
            failures.extend(f"{name} identity: {item}" for item in verify_candidate_identity(
                report["candidate_identity"], expected_scorer_sha256=expected_scorer_sha256,
                expected_extractor_sha256=expected_extractor_sha256))
    left, right = _identity_bindings(reference), _identity_bindings(candidate)
    for key in left:
        if left[key] != right[key] or left[key] in (None, []):
            failures.append(f"incompatible or missing {key}")
    if (reference.get("candidate_identity") or {}).get("candidate_id") == (candidate.get("candidate_identity") or {}).get("candidate_id"):
        failures.append("reference and candidate have the same candidate_id")
    scopes = []
    if not failures:
        for ref_scope, cand_scope in zip(reference["scopes"], candidate["scopes"], strict=True):
            ref_metrics, cand_metrics = ref_scope["parsed"]["metrics"], cand_scope["parsed"]["metrics"]
            scopes.append({"name": ref_scope["name"], "membership": ref_scope["membership"],
                           "reference": ref_metrics, "candidate": cand_metrics,
                           "delta_candidate_minus_reference": {name: cand_metrics[name] - ref_metrics[name]
                                                               for name in REQUIRED_SCORER_METRICS},
                           "lower_is_better": list(LOWER_IS_BETTER)})
    continuity = {name: {"finite_output_fraction": (report.get("prediction_validation") or {}).get("finite_output_fraction"),
                         "absolute_fix_fraction": (report.get("status_coverage") or {}).get("absolute_fix_fraction"),
                         "localized_state_fraction": (report.get("status_coverage") or {}).get("localized_state_fraction")}
                  for name, report in (("reference", reference), ("candidate", candidate))}
    disclosures = sorted({item for report in (reference, candidate) for item in report.get("disclosures", [])})
    return {"schema": SCHEMA + "/comparison", "ok": not failures, "failures": failures,
            "reference_candidate_id": (reference.get("candidate_identity") or {}).get("candidate_id"),
            "candidate_candidate_id": (candidate.get("candidate_identity") or {}).get("candidate_id"),
            "scopes": scopes, "continuity_and_fix_coverage": continuity, "disclosures": disclosures,
            "composite_score": None, "composite_note": "No composite official score is invented"}


# Timing ------------------------------------------------------------------------------------

_TIMING_FIELDS = ("received_monotonic_ns", "started_monotonic_ns", "decoded_monotonic_ns",
                  "completed_monotonic_ns", "published_monotonic_ns")


@dataclass(frozen=True, slots=True)
class FrameTiming:
    """One camera exposure, including failures. All *_monotonic_ns share clock_domain.

    received: frame available to the pipeline (cached read start or capture delivery)
    started: dequeued for processing; decoded: image decoded and preprocessed
    completed: estimator finished, including synchronized accelerator work
    published: CSV row written. capture_monotonic_ns exists only for live capture.
    """

    exposure_ns: int
    clock_domain: str
    status: str
    received_monotonic_ns: int
    started_monotonic_ns: int
    decoded_monotonic_ns: int
    completed_monotonic_ns: int
    published_monotonic_ns: int
    queue_drops: int = 0
    capture_monotonic_ns: int | None = None
    last_absolute_fix_exposure_ns: int | None = None
    accelerator: str = "none"
    accelerator_synchronized: bool = False
    failure: str = ""

    def __post_init__(self) -> None:
        strict_int_ns(self.exposure_ns, "exposure_ns")
        if not isinstance(self.clock_domain, str) or not self.clock_domain.strip():
            raise ValueError("clock_domain must be declared")
        if not isinstance(self.status, str) or not self.status:
            raise ValueError("status must be a nonempty string")
        stamps = [strict_int_ns(getattr(self, name), name) for name in _TIMING_FIELDS]
        if any(left > right for left, right in zip(stamps, stamps[1:])):
            raise ValueError("received <= started <= decoded <= completed <= published is required in one monotonic domain")
        strict_int_ns(self.queue_drops, "queue_drops")
        if self.capture_monotonic_ns is not None and strict_int_ns(self.capture_monotonic_ns, "capture") > stamps[0]:
            raise ValueError("capture must precede receipt")
        if self.last_absolute_fix_exposure_ns is not None and \
                strict_int_ns(self.last_absolute_fix_exposure_ns, "last fix exposure") > self.exposure_ns:
            raise ValueError("last absolute fix exposure must not be later than the output exposure")
        if not isinstance(self.accelerator, str) or not isinstance(self.accelerator_synchronized, bool):
            raise ValueError("accelerator must be a string and accelerator_synchronized a boolean")
        if not isinstance(self.failure, str):
            raise ValueError("failure must be a string")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "FrameTiming":
        names = {item.name for item in fields(cls)}
        unknown = sorted(set(value) - names)
        if unknown:
            raise ValueError(f"unknown timing fields {unknown}")
        return cls(**value)


class TimingRecorder:
    """Collect FrameTiming records with one monotonic clock; injectable for tests."""

    def __init__(self, clock_domain: str = "time.monotonic_ns", clock: Callable[[], int] = time.monotonic_ns):
        self.clock_domain, self.clock = clock_domain, clock
        self.records: list[FrameTiming] = []

    def frame(self, exposure_ns: int, *, capture_monotonic_ns: int | None = None, queue_drops: int = 0) -> "_FrameTimer":
        return _FrameTimer(self, strict_int_ns(exposure_ns, "exposure_ns"), capture_monotonic_ns, queue_drops)

    def write_jsonl(self, evidence: EvidenceWriter, name: str = "timing.jsonl") -> dict:
        return evidence.write_text(name, "".join(canonical_json(asdict(record)) + "\n" for record in self.records))


class _FrameTimer:
    def __init__(self, recorder: TimingRecorder, exposure_ns: int, capture: int | None, drops: int):
        self.recorder, self.exposure_ns, self.capture, self.drops = recorder, exposure_ns, capture, drops
        self.marks = {"received_monotonic_ns": recorder.clock()}
        self.accelerator, self.synchronized = "none", False

    def started(self) -> None:
        self.marks["started_monotonic_ns"] = self.recorder.clock()

    def decoded(self) -> None:
        self.marks["decoded_monotonic_ns"] = self.recorder.clock()

    def completed(self, *, accelerator: str = "none", synchronized: bool = False) -> None:
        """Call only after accelerator work is synchronized; enqueue time is not completion."""
        self.marks["completed_monotonic_ns"] = self.recorder.clock()
        self.accelerator, self.synchronized = accelerator, synchronized

    def published(self, status: str, *, last_absolute_fix_exposure_ns: int | None = None, failure: str = "") -> FrameTiming:
        self.marks["published_monotonic_ns"] = self.recorder.clock()
        missing = [name for name in _TIMING_FIELDS if name not in self.marks]
        if missing:
            raise ValueError(f"frame timing marks missing: {missing}")
        record = FrameTiming(exposure_ns=self.exposure_ns, clock_domain=self.recorder.clock_domain, status=status,
                             queue_drops=self.drops, capture_monotonic_ns=self.capture,
                             last_absolute_fix_exposure_ns=last_absolute_fix_exposure_ns, accelerator=self.accelerator,
                             accelerator_synchronized=self.synchronized, failure=failure, **self.marks)
        self.recorder.records.append(record)
        return record


def percentile(values: Sequence[float], q: float) -> float | None:
    """Linear interpolation between closest ranks (NumPy's default method)."""
    if not values:
        return None
    data = sorted(values)
    position = (len(data) - 1) * q / 100.0
    low, high = math.floor(position), math.ceil(position)
    return float(data[low]) if low == high else data[low] + (data[high] - data[low]) * (position - low)


def _stats_ms(values_ns: Sequence[int], threshold_ms: float | None = None) -> dict:
    if not values_ns:
        return {"count": 0, "mean_ms": None, "median_ms": None, "p95_ms": None, "max_ms": None}
    milliseconds = [value / 1e6 for value in values_ns]
    result = {"count": len(values_ns), "mean_ms": sum(values_ns) / len(values_ns) / 1e6,
              "median_ms": percentile(milliseconds, 50), "p95_ms": percentile(milliseconds, 95), "max_ms": max(milliseconds)}
    if threshold_ms is not None:
        result["over_threshold_fraction"] = sum(value > threshold_ms for value in milliseconds) / len(milliseconds)
        result["threshold_ms"] = threshold_ms
    return result


def summarize_timing(records: Iterable[Any], expected_timestamps: Iterable[Any], *, clock_domain: str, boundary: str,
                     run: Mapping[str, Any], resource: Mapping[str, Any] | None,
                     require_controlled: bool = True) -> dict:
    """Per-stage and total timing over every expected exposure, including failed frames.

    boundary 'cached_replay' measures frame availability to CSV publication and makes
    no live-latency claim. 'capture_to_publication' requires capture stamps in the same
    monotonic domain. run holds started/prepared/first_frame/finished monotonic stamps.
    """
    if boundary not in ("cached_replay", "capture_to_publication"):
        raise ValueError("boundary must be cached_replay or capture_to_publication")
    expected = normalize_expected_timestamps(expected_timestamps)
    expected_set = set(expected)
    failures, invalid, frames = [], [], []
    for index, raw in enumerate(records):
        try:
            record = raw if isinstance(raw, FrameTiming) else FrameTiming.from_mapping(raw)
        except (TypeError, ValueError) as error:
            invalid.append({"record": index, "reason": str(error)})
            continue
        frames.append(record)
    seen, duplicates, extra, domains = set(), [], [], set()
    previous_exposure = previous_publication = None
    order_violations = 0
    kept = []
    for record in frames:
        domains.add(record.clock_domain)
        if record.exposure_ns in seen:
            duplicates.append(record.exposure_ns)
            continue
        seen.add(record.exposure_ns)
        if record.exposure_ns not in expected_set:
            extra.append(record.exposure_ns)
        if previous_exposure is not None and (record.exposure_ns < previous_exposure
                                              or record.published_monotonic_ns < previous_publication):
            order_violations += 1
        previous_exposure, previous_publication = record.exposure_ns, record.published_monotonic_ns
        kept.append(record)
    missing = [stamp for stamp in expected if stamp not in seen]
    run_fields = ("started_monotonic_ns", "prepared_monotonic_ns", "first_frame_monotonic_ns", "finished_monotonic_ns")
    try:
        run_stamps = [strict_int_ns(run.get(name), f"run {name}") for name in run_fields]
        if any(left > right for left, right in zip(run_stamps, run_stamps[1:])):
            raise ValueError("run stamps must be ordered started <= prepared <= first_frame <= finished")
        if run.get("clock_domain") != clock_domain:
            raise ValueError("run clock_domain differs from the declared domain")
    except ValueError as error:
        failures.append(f"run boundaries invalid: {error}")
        run_stamps = None
    if domains - {clock_domain}:
        failures.append(f"records use clock domains {sorted(domains)} instead of only {clock_domain!r}")
    if run_stamps and kept and (min(r.received_monotonic_ns for r in kept) < run_stamps[2]
                                or max(r.published_monotonic_ns for r in kept) > run_stamps[3]):
        failures.append("frame stamps fall outside the run's first_frame..finished window")
    unsynced = [r.exposure_ns for r in kept if r.accelerator != "none" and not r.accelerator_synchronized]
    with_capture = [r for r in kept if r.capture_monotonic_ns is not None]
    if boundary == "cached_replay" and with_capture:
        failures.append("capture stamps present in a cached replay; declare capture_to_publication instead")
    if boundary == "capture_to_publication" and len(with_capture) != len(kept):
        failures.append(f"{len(kept) - len(with_capture)} frames lack capture stamps for capture_to_publication")
    for items, label in ((invalid, "invalid timing records"), (duplicates, "duplicate timing exposures"),
                         (extra, "timing records at unexpected exposures"), (missing, "expected exposures without timing"),
                         (unsynced, "frames with unsynchronized accelerator work")):
        if items:
            failures.append(f"{len(items)} {label}")
    if order_violations:
        failures.append(f"{order_violations} records out of exposure or publication order")
    controlled = not benchmark_resource_problems(resource)
    if require_controlled and not controlled:
        failures.append("timing was not recorded under an exclusive benchmark-cpu claim fingerprint")
    stages = {
        "queue_wait": [r.started_monotonic_ns - r.received_monotonic_ns for r in kept],
        "decode_preprocess": [r.decoded_monotonic_ns - r.started_monotonic_ns for r in kept],
        "estimator_completed": [r.completed_monotonic_ns - r.decoded_monotonic_ns for r in kept],
        "output_serialization": [r.published_monotonic_ns - r.completed_monotonic_ns for r in kept],
        "processing": [r.published_monotonic_ns - r.started_monotonic_ns for r in kept],
    }
    total = [r.published_monotonic_ns - r.received_monotonic_ns for r in kept]
    capture = [r.published_monotonic_ns - r.capture_monotonic_ns for r in with_capture]
    ages = [r.exposure_ns - r.last_absolute_fix_exposure_ns for r in kept if r.last_absolute_fix_exposure_ns is not None]
    total_stats = _stats_ms(total, RUNTIME_REFERENCE_MS)
    statuses: dict[str, int] = {}
    for record in kept:
        statuses[record.status] = statuses.get(record.status, 0) + 1
    summary = {
        "schema": SCHEMA + "/timing", "clock_domain": clock_domain, "boundary": boundary,
        "boundary_definition": ("cached frame availability to CSV row publication; excludes capture, transport and raw MCAP extraction"
                                if boundary == "cached_replay" else "capture stamp to CSV row publication in one monotonic domain"),
        "live_latency_measured": boundary == "capture_to_publication",
        "expected_frames": len(expected), "timed_frames": len(kept), "failed_frames": sum(bool(r.failure) for r in kept),
        "status_counts": statuses,
        "stages": {name: _stats_ms(values) for name, values in stages.items()},
        "total_frame": total_stats,
        "capture_to_publication": _stats_ms(capture, RUNTIME_REFERENCE_MS) if boundary == "capture_to_publication" else None,
        "queue_drops": sum(r.queue_drops for r in kept),
        "correction_age": {**_stats_ms(ages), "frames_without_absolute_fix": len(kept) - len(ages)},
        "accelerators": sorted({r.accelerator for r in kept}),
        "run": ({"preparation_ms": (run_stamps[1] - run_stamps[0]) / 1e6, "cold_load_to_first_frame_ms": (run_stamps[2] - run_stamps[1]) / 1e6,
                 "frame_loop_wall_ms": (run_stamps[3] - run_stamps[2]) / 1e6, "total_wall_ms": (run_stamps[3] - run_stamps[0]) / 1e6}
                if run_stamps else None),
        "runtime_reference": {"mean_ms_reference": RUNTIME_REFERENCE_MS,
                              "mean_total_frame_within_reference": (total_stats["mean_ms"] is not None
                                                                    and total_stats["mean_ms"] <= RUNTIME_REFERENCE_MS),
                              "note": "100 ms mean is the official full-runtime-points reference, not a measured guarantee"},
        "controlled": controlled, "resource": dict(resource) if isinstance(resource, Mapping) else None,
        "samples": {"invalid": _bounded(invalid), "duplicates": _bounded(duplicates), "extra": _bounded(extra),
                    "missing": _bounded(missing), "unsynchronized_accelerator": _bounded(unsynced)},
        "failures": failures, "ok": not failures,
    }
    return summary


def benchmark_resource_problems(resource: Any) -> list[str]:
    """Why resource metadata is not a usable benchmark-cpu claim record (advisory, not a lock)."""
    if not isinstance(resource, Mapping):
        return ["no benchmark-cpu claim record supplied"]
    problems = []
    if resource.get("name") != BENCHMARK_RESOURCE:
        problems.append(f"resource name {resource.get('name')!r} is not {BENCHMARK_RESOURCE!r}")
    if _SHA256.fullmatch(str(resource.get("claim_fingerprint_sha256", ""))) is None:
        problems.append("claim_fingerprint_sha256 is not a lowercase SHA256 hex digest")
    return problems


def run_timed_command(command: Sequence[str], evidence: EvidenceWriter, *, resource: Mapping[str, Any] | None,
                      timeout_s: float, cwd: str | Path | None = None) -> dict:
    """Run one candidate process (no shell), recording wall time, exit status and raw output.

    The benchmark-cpu claim record is validated before anything is launched. A missing,
    malformed or wrong-resource record saves an explicit refused, not-executed result and
    starts no process. The fingerprint is advisory; it does not prove a live lock, and
    the coordinator keeps actual resource custody and machine quieting.
    """
    if not command:
        raise ValueError("command is required")
    base: dict[str, Any] = {"command": list(command), "cwd": str(Path(cwd).resolve()) if cwd else os.getcwd(),
                            "clock_domain": "time.monotonic_ns", "resource": dict(resource) if isinstance(resource, Mapping) else None,
                            "note": "A resource record does not stop other processes; the coordinator must quiet the machine"}
    problems = benchmark_resource_problems(resource)
    if problems:
        # No environment probe either: platform.processor() may itself spawn `uname -p`.
        record = dict(base, status="refused", executed=False, exit_code=None,
                      failures=[f"not executed: {item}" for item in problems], ok=False)
        evidence.write_json("command.json", record)
        return record
    environment_before = environment_identity()
    started = time.monotonic_ns()
    record = dict(base, status="executed", executed=True, started_monotonic_ns=started)
    stdout = stderr = b""
    try:
        result = subprocess.run(list(command), cwd=cwd, capture_output=True, timeout=timeout_s, check=False)
        stdout, stderr, record["exit_code"] = result.stdout, result.stderr, result.returncode
    except subprocess.TimeoutExpired as error:
        stdout, stderr = error.stdout or b"", error.stderr or b""
        record.update(exit_code=None, timed_out=True)
    except OSError as error:
        record.update(exit_code=None, launch_error=f"{type(error).__name__}: {error}")
    record["finished_monotonic_ns"] = time.monotonic_ns()
    record["wall_ms"] = (record["finished_monotonic_ns"] - started) / 1e6
    record["stdout_file"] = evidence.write_bytes("command.stdout.txt", stdout)
    record["stderr_file"] = evidence.write_bytes("command.stderr.txt", stderr)
    record["environment_before"] = environment_before
    record["environment_after"] = environment_identity()
    failures = []
    if record.get("exit_code") != 0:
        failures.append(f"candidate command exit {record.get('exit_code')}"
                        + (" (timed out)" if record.get("timed_out") else "")
                        + (f" ({record['launch_error']})" if record.get("launch_error") else ""))
    record["failures"], record["ok"] = failures, not failures
    evidence.write_json("command.json", record)
    return record

"""Bounded development search and candidate identity helpers; never select from audit."""
from __future__ import annotations

import copy
import hashlib
import json
import math

from .schemas import Settings


def canonical_hash(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def candidate_configs(base: dict, count: int) -> list[dict]:
    """Return a deterministic, bounded set of genuine parameter variations."""
    if type(count) is not int or not 1 <= count <= 8:
        raise ValueError("Development search count must be between 1 and 8")
    settings = Settings.model_validate(base)
    variations = [{}, {"ratio_test": 0.70, "ransac_px": 3.0},
                  {"ratio_test": 0.80, "ransac_px": 5.0},
                  {"max_features": min(5000, max(300, int(settings.max_features * 1.4))), "relocalize_every": 2},
                  {"min_inliers": min(100, max(6, settings.min_inliers + 4)), "max_propagation_s": 0.25},
                  {"ratio_test": 0.72, "relocalize_every": 1},
                  {"max_features": max(300, int(settings.max_features * 0.75)), "ransac_px": 3.5},
                  {"ratio_test": 0.78, "max_propagation_s": 0.60, "min_inliers": max(6, settings.min_inliers)}]
    result = []
    for patch in variations[:count]:
        candidate = copy.deepcopy(base)
        candidate.update(patch)
        Settings.model_validate(candidate)
        result.append(candidate)
    return result


def _metric(metrics: dict, key: str, fallback: float) -> float:
    value = metrics.get(key)
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) else fallback


def rank_runs(runs: list[dict]) -> list[dict]:
    """Rank labeled development runs by success, coverage, tail error, then latency.

    A fixed scorecard ordering is useful before official thresholds exist. It is not
    the official race score and must not be used to repeatedly tune on final tests.
    """
    if any(run.get("split") not in {None, "development"} for run in runs):
        raise ValueError("Candidate selection is restricted to development runs")
    for run in runs:
        if run.get("metrics", {}).get("joint_success_pct") is None:
            raise ValueError("Candidate selection needs complete labeled development results")
    return sorted(runs, key=lambda run: (
        -_metric(run.get("metrics", {}), "joint_success_pct", -1),
        -_metric(run.get("metrics", {}), "coverage_pct", -1),
        _metric(run.get("metrics", {}), "p95_position_error_m", math.inf),
        _metric(run.get("metrics", {}), "p95_heading_error_deg", math.inf),
        _metric(run.get("metrics", {}), "p95_latency_ms", math.inf),
        str(run.get("id", ""))))


def freeze_identity(config: dict, map_hash: str, code_hash: str, model_hash: str | None = None) -> dict:
    if not map_hash or not code_hash:
        raise ValueError("Freezing requires map and code hashes")
    identity = {"config_hash": canonical_hash(config), "map_hash": map_hash,
                "code_hash": code_hash, "model_hash": model_hash}
    return {**identity, "hash": canonical_hash(identity)}


def verify_frozen_identity(identity: dict, config: dict, map_hash: str, code_hash: str,
                           model_hash: str | None = None) -> None:
    if identity != freeze_identity(config, map_hash, code_hash, model_hash):
        raise ValueError("Frozen candidate changed; audit is forbidden")

"""Advisory pretrained place-embedding ranking (TEAM-VISION-01). Not a pose estimator.

Inputs are caller-supplied float32[960] embeddings: one query and at most 64
references with unique IDs, all produced by one explicit frozen encoder
identity (place_model.encoder_identity). The adapter ranks every reference by
cosine similarity and may report an advisory top-K shortlist. It performs no
feature extraction, runs no encoder, reads no labels, poses, timestamps or
images, and never maps time to position.

Optional verification delegates exactly once to an injected, existing
KeyframeLocalizer.localize and returns its PlaceEstimate object unchanged
beside the ranking. The shortlist never selects, removes or reorders the
references that the verifier checks: the verifier keeps its own complete
reference coverage and ambiguity evidence. No speed or accuracy claim follows.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from typing import Any, Callable, Mapping

import numpy as np

from .keyframe_localization import PlaceEstimate
from .map_contracts import _immutable
from .place_model import canonical, check_encoder, encoder_identity, parse_json

SCHEMA = "pretrained-place-ranking-v1"
EMBEDDING_DIMENSION = 960
MAX_REFERENCES = 64
MAX_REFERENCE_ID_CHARS = 120
INTERPRETATION = "advisory_embedding_ranking_not_pose"
COVERAGE = "all_references_ranked"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _encoder(identity: Any, expected_weights_sha256: Any) -> tuple[str, str]:
    """Validate an exact accepted encoder identity against a caller-held weights pin.

    identity is a place_model.encoder_identity mapping or JSON object text.
    It must equal, in canonical bytes, the identity that place_model rebuilds
    from expected_weights_sha256 and the stated dependencies, so aliases
    such as 1 for true or 232.0 for 232 are rejected. Returns (canonical JSON
    text, SHA256 of those canonical bytes).
    """
    try:
        if isinstance(identity, str):
            value = parse_json(identity)
        elif isinstance(identity, Mapping):
            value = json.loads(json.dumps(dict(identity), sort_keys=True, allow_nan=False))
        else:
            raise ValueError("encoder identity must be a mapping or JSON object text")
        if not isinstance(value, dict):
            raise ValueError("encoder identity must be a JSON object")
        check_encoder(value, expected_weights_sha256)
        if canonical(value) != canonical(encoder_identity(expected_weights_sha256, value["dependencies"])):
            raise ValueError("encoder identity is not in canonical accepted form")
    except (ValueError, TypeError, KeyError, AttributeError) as exc:
        raise ValueError(f"invalid encoder identity: {exc}") from exc
    text = canonical(value)
    return text.decode("utf-8"), _sha256(text)


def _exact_float32(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    if not isinstance(value, np.ndarray) or isinstance(value, np.ma.MaskedArray):
        raise ValueError(f"{name} must be an unmasked NumPy array")
    if value.dtype != np.dtype("float32") or value.shape != shape:
        raise ValueError(f"{name} must be float32 with shape {shape}")
    if not np.isfinite(value).all():
        raise ValueError(f"{name} must be finite")
    return value


def _unit(vector: np.ndarray, name: str) -> list[float]:
    """Unit float64 copy with an exactly rounded norm (math.fsum), independent of BLAS order.

    Every finite nonzero float32 vector is accepted regardless of magnitude:
    in float64 the squares and their sum stay finite and positive from the
    smallest float32 subnormal (2**-149) to the float32 maximum, so only an
    exact all-zero vector (including negative zeros) has a zero norm.
    """
    values = vector.astype(np.float64).tolist()
    norm = math.sqrt(math.fsum(v * v for v in values))
    if not math.isfinite(norm) or norm == 0.0:
        raise ValueError(f"{name} must be a finite nonzero vector")
    return [v / norm for v in values]


def _reference_id(value: Any) -> str:
    if not isinstance(value, str) or not value or len(value) > MAX_REFERENCE_ID_CHARS or value != value.strip():
        raise ValueError(f"reference IDs must be nonempty strings of at most {MAX_REFERENCE_ID_CHARS} characters")
    return value


@dataclass(frozen=True, slots=True, eq=False)
class ReferenceEmbeddings:
    """Frozen reference embeddings. Carries IDs and vectors only, no pose or label.

    encoder_identity accepts a mapping or JSON object text and is stored as
    canonical JSON text, so the stored identity cannot be mutated through
    nested dicts. expected_weights_sha256 is the caller's independently held
    weights pin; the identity must name exactly those weights. vectors are
    stored as an immutable copy; the caller's array is untouched.
    """
    reference_ids: tuple[str, ...]
    vectors: np.ndarray
    encoder_identity: Mapping[str, Any] | str
    expected_weights_sha256: str
    encoder_sha256: str = field(init=False)
    reference_set_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        ids = tuple(self.reference_ids) if isinstance(self.reference_ids, (tuple, list)) else None
        if ids is None or not 1 <= len(ids) <= MAX_REFERENCES:
            raise ValueError(f"between 1 and {MAX_REFERENCES} reference IDs required")
        ids = tuple(_reference_id(value) for value in ids)
        if len(set(ids)) != len(ids):
            raise ValueError("reference IDs must be unique")
        vectors = _exact_float32(self.vectors, (len(ids), EMBEDDING_DIMENSION), "reference vectors")
        for row, reference_id in zip(vectors, ids, strict=True):
            _unit(row, f"reference vector {reference_id}")
        encoder_json, encoder_sha = _encoder(self.encoder_identity, self.expected_weights_sha256)
        frozen = _immutable(vectors)
        digest = hashlib.sha256()
        digest.update(json.dumps(list(ids), separators=(",", ":")).encode("utf-8"))
        digest.update(frozen.tobytes())
        object.__setattr__(self, "reference_ids", ids)
        object.__setattr__(self, "vectors", frozen)
        object.__setattr__(self, "encoder_identity", encoder_json)
        object.__setattr__(self, "encoder_sha256", encoder_sha)
        object.__setattr__(self, "reference_set_sha256", digest.hexdigest())


@dataclass(frozen=True, slots=True)
class RankedReference:
    """One rank: reference ID and cosine only. No coordinates or pose."""
    rank: int
    reference_id: str
    cosine: float


@dataclass(frozen=True, slots=True)
class PlaceRanking:
    schema: str
    interpretation: str
    coverage: str
    encoder_sha256: str
    reference_set_sha256: str
    query_vector_sha256: str
    reference_count: int
    ranked: tuple[RankedReference, ...]
    top_k: int | None
    shortlist: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class VerifiedRanking:
    """Advisory ranking beside the verifier's unchanged PlaceEstimate."""
    ranking: PlaceRanking
    estimate: PlaceEstimate


def rank_references(query_vector: np.ndarray, query_encoder_identity: Mapping[str, Any] | str,
                    references: ReferenceEmbeddings, top_k: int | None = None) -> PlaceRanking:
    """Rank every reference by cosine similarity, ties broken by reference ID.

    The query must come from exactly the same encoder identity as the
    references, checked against the references' caller-held weights pin.
    top_k only selects an advisory prefix of the full ranking.
    """
    if not isinstance(references, ReferenceEmbeddings):
        raise ValueError("references must be ReferenceEmbeddings")
    query = _exact_float32(query_vector, (EMBEDDING_DIMENSION,), "query vector")
    _, query_encoder_sha = _encoder(query_encoder_identity, references.expected_weights_sha256)
    if query_encoder_sha != references.encoder_sha256:
        raise ValueError("query and reference encoder identities differ")
    count = len(references.reference_ids)
    if top_k is not None and (type(top_k) is not int or not 1 <= top_k <= count):
        raise ValueError(f"top_k must be None or an integer in [1, {count}]")
    unit_query = _unit(query, "query vector")
    scored = []
    for reference_id, row in zip(references.reference_ids, references.vectors, strict=True):
        unit_row = _unit(row, f"reference vector {reference_id}")
        cosine = math.fsum(a * b for a, b in zip(unit_query, unit_row, strict=True))
        scored.append((min(1.0, max(-1.0, cosine)), reference_id))
    scored.sort(key=lambda item: (-item[0], item[1]))
    ranked = tuple(RankedReference(rank, reference_id, cosine)
                   for rank, (cosine, reference_id) in enumerate(scored, start=1))
    shortlist = tuple(item.reference_id for item in ranked[:top_k]) if top_k is not None else ()
    return PlaceRanking(schema=SCHEMA, interpretation=INTERPRETATION, coverage=COVERAGE,
                        encoder_sha256=references.encoder_sha256,
                        reference_set_sha256=references.reference_set_sha256,
                        query_vector_sha256=_sha256(np.ascontiguousarray(query).tobytes()),
                        reference_count=count, ranked=ranked, top_k=top_k, shortlist=shortlist)


def rank_and_verify(query_vector: np.ndarray, query_encoder_identity: Mapping[str, Any] | str,
                    references: ReferenceEmbeddings, verifier: Callable[..., PlaceEstimate],
                    verifier_arguments: Mapping[str, Any], top_k: int | None = None) -> VerifiedRanking:
    """Rank, then call the injected verifier exactly once with the caller's arguments.

    verifier is an existing KeyframeLocalizer.localize (or a test double with
    the same contract). It receives verifier_arguments unchanged and nothing
    from the ranking, so its complete reference coverage and ambiguity checks
    are unaffected. Invalid embeddings reject (ValueError) before the verifier
    is called. Exceptions raised by the verifier itself propagate unchanged,
    for example KeyframeLocalizer's own timestamp or query ID validation.
    The returned estimate is the verifier's PlaceEstimate object itself.
    """
    if not callable(verifier):
        raise ValueError("verifier must be callable")
    if not isinstance(verifier_arguments, Mapping) or not all(isinstance(key, str) for key in verifier_arguments):
        raise ValueError("verifier_arguments must be a mapping of keyword names")
    ranking = rank_references(query_vector, query_encoder_identity, references, top_k)
    estimate = verifier(**dict(verifier_arguments))
    if not isinstance(estimate, PlaceEstimate):
        raise ValueError("verifier must return a PlaceEstimate")
    return VerifiedRanking(ranking=ranking, estimate=estimate)

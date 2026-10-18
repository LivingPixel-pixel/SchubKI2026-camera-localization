"""Pinned encode-only chunks and exact-once canonical feature-cache assembly.

Uses accepted place_model APIs unchanged. Plan/assembly never open pixels or
weights, construct Torch, mine pairs or fit a model. Execution is root-gated.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import time

import numpy as np

from . import place_model as p


CORE_SHA256 = "0f98f6f4ca7842c5460ebb008bfd4f0c139b6b19b30ec7ebd499914193da321a"
PLAN_SCHEMA = "schubki-place-cache-plan-v1"
MAX_ROWS, MAX_CHUNKS = 6000, 64
MAX_CHUNK_ROWS, MAX_CHUNK_BYTES = 192, 24 * 1024 * 1024
PLAN_KEYS = {"schema", "complete", "dataset_sha256", "existing_cache_sha256",
    "weights_sha256", "core_source_sha256", "implementation_sha256", "encoder",
    "dataset_rows_sha256", "limits", "count", "role_counts", "canonical_frame_ids",
    "reused_frame_ids", "chunks"}


@dataclass(frozen=True)
class Inputs:
    dataset: Path
    dataset_sha256: str
    cache_root: Path
    existing_cache: Path
    existing_cache_sha256: str
    weights_sha256: str
    core_source_sha256: str
    implementation_sha256: str

    def verify_sources(self):
        for value in (self.dataset_sha256, self.existing_cache_sha256,
                      self.weights_sha256, self.core_source_sha256, self.implementation_sha256):
            p.sha(value)
        if self.core_source_sha256 != CORE_SHA256 or p.file_hash(p.__file__, 2 * 1024 * 1024) != CORE_SHA256:
            raise ValueError("accepted core source identity mismatch")
        if p.file_hash(__file__, 2 * 1024 * 1024) != self.implementation_sha256:
            raise ValueError("cache implementation source identity mismatch")


def role_counts(rows):
    return {role: sum(row["role"] == role for row in rows)
            for role in ("train", "checkpoint_validation")}


def _close_features(features):
    # Accepted loaders return owned read-only mappings, not shared file writers.
    mapping = getattr(features, "_mmap", None)
    if mapping is not None:
        mapping.close()


class _Mappings:
    """Own transferred mappings independently of retained exception frames."""

    def __init__(self):
        self.pending = []

    def __enter__(self):
        return self

    def hold(self, features):
        self.pending.append(features)

    def transfer(self):
        self.pending.clear()

    def close(self, preserve_error=False):
        pending, self.pending = self.pending, []
        first_error = None
        for features in reversed(pending):
            try:
                _close_features(features)
            except BaseException as error:
                if first_error is None:
                    first_error = error
        if first_error is not None and not preserve_error:
            raise first_error

    def __exit__(self, error_type, error, traceback):
        if self.pending:
            self.close(preserve_error=error_type is not None)


def checked_cache(inputs, directory, cache_sha256, max_rows=MAX_ROWS, encoder=None):
    directory = Path(directory)
    if (directory / "failure.json").exists():
        raise ValueError("failed input cache is not reusable")
    with _Mappings() as owned:
        metadata, rows, features = p.load_cache(directory, cache_sha256,
            inputs.dataset_sha256, inputs.weights_sha256, max_rows=max_rows)
        owned.hold(features)
        if metadata["source_sha256"] != inputs.core_source_sha256:
            raise ValueError("cache core source identity mismatch")
        if encoder is not None and metadata["encoder"] != encoder:
            raise ValueError("cache encoder/dependency identity mismatch")
        owned.transfer()
        return metadata, rows, features


def context(inputs):
    inputs.verify_sources()
    _, observations, rows = p.load_dataset(inputs.dataset, inputs.dataset_sha256, inputs.cache_root)
    if not 1 <= len(rows) <= MAX_ROWS or any(count == 0 for count in role_counts(rows).values()):
        raise ValueError("full dataset requires both training and checkpoint-validation roles")
    lookup = {row["frame_id"]: (index, row, obs)
              for index, (row, obs) in enumerate(zip(rows, observations))}
    with _Mappings() as owned:
        old, old_rows, old_features = checked_cache(inputs, inputs.existing_cache,
                                                   inputs.existing_cache_sha256)
        owned.hold(old_features)
        for row in old_rows:
            expected = lookup.get(row["frame_id"])
            if expected is None or p.canonical(row) != p.canonical(expected[1]):
                raise ValueError("existing cache row does not match prepared dataset")
        owned.transfer()
        return observations, rows, lookup, old, old_rows, old_features


def make_plan(inputs, loaded, chunk_rows, chunk_bytes):
    p.integer(chunk_rows, 1, MAX_CHUNK_ROWS)
    p.integer(chunk_bytes, 1, MAX_CHUNK_BYTES)
    observations, rows, _, old, old_rows, _ = loaded
    reused = {row["frame_id"] for row in old_rows}
    selections, groups, current, encoded_bytes = [], [], [], 0
    for row, observation in zip(rows, observations):
        if row["frame_id"] in reused:
            continue
        size = observation["image_bytes"]
        if size > chunk_bytes:
            raise ValueError("single remaining image exceeds chunk encoded-byte cap")
        if current and (len(current) == chunk_rows or encoded_bytes + size > chunk_bytes):
            groups.append((current, encoded_bytes))
            current, encoded_bytes = [], 0
        current.append(row)
        encoded_bytes += size
    if current:
        groups.append((current, encoded_bytes))
    if len(groups) > MAX_CHUNKS:
        raise ValueError("new chunk count exceeds64")
    chunks = []
    for index, (group, byte_count) in enumerate(groups, 1):
        selection = [row["frame_id"] for row in group]
        selections.append(selection)
        chunks.append({"index": index, "selection_path": f"chunk-{index:04d}.json",
            "selection_sha256": p.digest(p.canonical(selection)), "count": len(group),
            "encoded_bytes": byte_count, "role_counts": role_counts(group)})
    value = {"schema": PLAN_SCHEMA, "complete": True,
        "dataset_sha256": inputs.dataset_sha256, "existing_cache_sha256": inputs.existing_cache_sha256,
        "weights_sha256": inputs.weights_sha256, "core_source_sha256": inputs.core_source_sha256,
        "implementation_sha256": inputs.implementation_sha256, "encoder": old["encoder"],
        "dataset_rows_sha256": p.digest(p.canonical(rows)),
        "limits": {"max_rows": MAX_ROWS, "max_chunks": MAX_CHUNKS,
                   "chunk_rows": chunk_rows, "chunk_encoded_bytes": chunk_bytes},
        "count": len(rows), "role_counts": role_counts(rows),
        "canonical_frame_ids": [row["frame_id"] for row in rows],
        "reused_frame_ids": [row["frame_id"] for row in rows if row["frame_id"] in reused],
        "chunks": chunks}
    return value, selections


def mark_failure(output, error):
    try:
        p.failed_output(output, error)
    except OSError:
        pass


def bounded_json(path, value, byte_cap):
    if len(p.canonical(value)) > byte_cap:
        raise ValueError("output metadata byte cap")
    p.write_json(path, value)


def output_path(inputs, output, additional_inputs=()):
    result = Path(output).resolve()
    for directory in (inputs.dataset, inputs.cache_root, inputs.existing_cache, *additional_inputs):
        protected = Path(directory).resolve()
        if result == protected or result.is_relative_to(protected):
            raise ValueError("output must remain outside input directories")
    return result


def plan_cache(inputs, output, chunk_rows=MAX_CHUNK_ROWS, chunk_bytes=MAX_CHUNK_BYTES):
    """Freeze metadata-only selections after verifying all existing cache bytes."""
    with _Mappings() as owned:
        loaded = context(inputs)
        owned.hold(loaded[-1])
        value, selections = make_plan(inputs, loaded, chunk_rows, chunk_bytes)
    output = p.new_output(output_path(inputs, output))
    try:
        for chunk, selection in zip(value["chunks"], selections):
            bounded_json(output / chunk["selection_path"], selection, 256 * 1024)
        bounded_json(output / "plan.json", value, 2 * 1024 * 1024)
        return value
    except Exception as error:
        mark_failure(output, error)
        raise


def load_plan(inputs, directory, expected_sha256):
    directory = Path(directory)
    if (directory / "failure.json").exists():
        raise ValueError("failed plan is not reusable")
    value = p.parse_json(p.read_pinned(directory / "plan.json", expected_sha256, 2 * 1024 * 1024))
    p.exact(value, PLAN_KEYS)
    p.exact(value["limits"], {"max_rows", "max_chunks", "chunk_rows", "chunk_encoded_bytes"})
    chunk_rows = p.integer(value["limits"]["chunk_rows"], 1, MAX_CHUNK_ROWS)
    chunk_bytes = p.integer(value["limits"]["chunk_encoded_bytes"], 1, MAX_CHUNK_BYTES)
    with _Mappings() as owned:
        loaded = context(inputs)
        owned.hold(loaded[-1])
        expected, selections = make_plan(inputs, loaded, chunk_rows, chunk_bytes)
        if p.canonical(value) != p.canonical(expected):
            raise ValueError("plan/input/membership/source identity mismatch")
        for chunk, selection in zip(value["chunks"], selections):
            data = p.read_pinned(directory / chunk["selection_path"], chunk["selection_sha256"], 256 * 1024)
            if data != p.canonical(selection):
                raise ValueError("noncanonical or changed chunk selection")
        owned.transfer()
        return value, selections, loaded


def encode_chunk(inputs, plan_directory, plan_sha256, index, output, weights,
                 wall_seconds=300, threads=1, encoder_factory=p.FrozenEncoder):
    """Run one explicit core encode call. Timings return to the CLI log only."""
    started = time.perf_counter_ns()
    p.integer(wall_seconds, 1, 300)
    p.integer(threads, 1, 4)
    with _Mappings() as owned:
        plan, selections, loaded = load_plan(inputs, plan_directory, plan_sha256)
        owned.hold(loaded[-1])
        p.integer(index, 1, len(selections))
        chunk, selection = plan["chunks"][index - 1], selections[index - 1]
        old = loaded[3]
        output = output_path(inputs, output, (plan_directory,))
        measurements = {"encoder_initialization_ns": 0, "rgb_encoding_calls_ns": 0,
                        "rgb_encoding_call_count": 0}

        class ObservedEncoder:
            def __init__(self, weight_path, expected, num_threads):
                before = time.perf_counter_ns()
                self.encoder = encoder_factory(weight_path, expected, num_threads)
                measurements["encoder_initialization_ns"] = time.perf_counter_ns() - before
                self.identity = self.encoder.identity
                if self.identity != plan["encoder"]:
                    raise ValueError("new chunk encoder/dependency identity differs from existing cache")

            def encode_rgb(self, rgb):
                before = time.perf_counter_ns()
                try:
                    return self.encoder.encode_rgb(rgb)
                finally:
                    measurements["rgb_encoding_calls_ns"] += time.perf_counter_ns() - before
                    measurements["rgb_encoding_call_count"] += 1

        limits = p.Limits(max_rows=chunk["count"], max_encoded_bytes=chunk["encoded_bytes"],
            max_image_bytes=max(loaded[2][frame_id][2]["image_bytes"] for frame_id in selection),
            max_pixels=old["limits"]["max_pixels"], wall_seconds=wall_seconds)
    cache = p.encode_dataset(inputs.dataset, inputs.dataset_sha256, inputs.cache_root,
        output, weights, inputs.weights_sha256, inputs.core_source_sha256, limits,
        threads, selection, encoder_factory=ObservedEncoder,
        role_caps=(MAX_CHUNK_ROWS, MAX_CHUNK_ROWS))
    total = time.perf_counter_ns() - started
    measurements["total_encode_ns"] = total
    measurements["unseparated_residual_ns"] = total - measurements["encoder_initialization_ns"] - measurements["rgb_encoding_calls_ns"]
    return {"complete": True, "chunk_index": index, "count": chunk["count"],
        "plan_sha256": plan_sha256, "cache_sha256": p.digest(p.canonical(cache)),
        "core_source_sha256": inputs.core_source_sha256,
        "implementation_sha256": inputs.implementation_sha256,
        "timings": measurements, "timing_scope": {
            "encoder_initialization_ns": "factory constructor only",
            "rgb_encoding_calls_ns": "sum of encode_rgb calls, including preprocessing and forward",
            "total_encode_ns": "plan/input validation through successful core cache publication",
            "unseparated_residual_ns": "validation, hashes, image decode, writes and wrapper overhead; not per-phase measurements"}}


def checked_chunk_descriptors(descriptors, expected_count):
    if not isinstance(descriptors, list) or len(descriptors) != expected_count or len(descriptors) > MAX_CHUNKS:
        raise ValueError("explicit complete chunk list required")
    seen, hashes, paths = set(), set(), set()
    for item in descriptors:
        p.exact(item, {"index", "directory", "cache_sha256"})
        index = p.integer(item["index"], 1, expected_count)
        p.sha(item["cache_sha256"])
        if not isinstance(item["directory"], str) or not 1 <= len(item["directory"]) <= 4096:
            raise ValueError("bounded chunk directory required")
        resolved = str(Path(item["directory"]).resolve()).casefold()
        if index in seen or item["cache_sha256"] in hashes or resolved in paths:
            raise ValueError("duplicate chunk index/hash/path")
        seen.add(index)
        hashes.add(item["cache_sha256"])
        paths.add(resolved)
    if seen != set(range(1, expected_count + 1)):
        raise ValueError("missing or unknown planned chunk")


def assemble_cache(inputs, plan_directory, plan_sha256, descriptors, output, wall_seconds=300):
    """Copy verified row/feature pairs by ID. Completion manifest publishes last."""
    p.integer(wall_seconds, 1, 300)
    deadline = time.monotonic() + wall_seconds
    with _Mappings() as owned:
        plan, selections, loaded = load_plan(inputs, plan_directory, plan_sha256)
        owned.hold(loaded[-1])
        observations, rows, lookup, old, old_rows, old_features = loaded
        checked_chunk_descriptors(descriptors, len(selections))
        sources = [("existing", inputs.existing_cache_sha256, old, old_rows, old_features)]
        for item in descriptors:
            p.check_deadline(deadline)
            index = item["index"]
            metadata, chunk_rows, features = checked_cache(inputs, item["directory"],
                item["cache_sha256"], max_rows=MAX_CHUNK_ROWS, encoder=plan["encoder"])
            owned.hold(features)
            if {row["frame_id"] for row in chunk_rows} != set(selections[index - 1]) or len(chunk_rows) != len(selections[index - 1]):
                raise ValueError("chunk membership differs from plan")
            sources.append((index, item["cache_sha256"], metadata, chunk_rows, features))
        seen = set()
        for _, _, _, source_rows, _ in sources:
            for row in source_rows:
                expected = lookup.get(row["frame_id"])
                if expected is None or p.canonical(row) != p.canonical(expected[1]):
                    raise ValueError("chunk row/label/hash differs from prepared dataset")
                if row["frame_id"] in seen:
                    raise ValueError("duplicate feature row across cache inputs")
                seen.add(row["frame_id"])
        if seen != set(lookup):
            raise ValueError("incomplete exact-once dataset coverage")
        output = p.new_output(output_path(inputs, output,
            (plan_directory, *(item["directory"] for item in descriptors))))
        try:
            features = np.lib.format.open_memmap(output / "features.npy", mode="w+",
                dtype=np.float32, shape=(len(rows), 960))
            owned.hold(features)
            for _, _, _, source_rows, source_features in sources:
                for source_index, row in enumerate(source_rows):
                    p.check_deadline(deadline)
                    features[lookup[row["frame_id"]][0]] = source_features[source_index]
            features.flush()
            p.validate_features(features, len(rows))
            owned.close()
            del features
            bounded_json(output / "rows.json", rows, 16 * 1024 * 1024)
            provenance = {"schema": "schubki-place-cache-assembly-v1", "complete": True,
                "plan_sha256": p.sha(plan_sha256), "dataset_sha256": inputs.dataset_sha256,
                "implementation_sha256": inputs.implementation_sha256,
                "core_source_sha256": inputs.core_source_sha256, "count": len(rows),
                "inputs": [{"chunk_index": index, "cache_sha256": cache_sha,
                    "rows_sha256": metadata["rows_sha256"], "features_sha256": metadata["features_sha256"],
                    "count": len(source_rows)} for index, cache_sha, metadata, source_rows, _ in sources],
                "copy": "raw float32 row copied by verified frame_id into prepared-dataset order"}
            bounded_json(output / "assembly.json", provenance, 1024 * 1024)
            limits = p.Limits(max_rows=len(rows),
                max_encoded_bytes=sum(obs["image_bytes"] for obs in observations),
                max_image_bytes=max(obs["image_bytes"] for obs in observations),
                max_pixels=old["limits"]["max_pixels"], wall_seconds=wall_seconds)
            metadata = {"schema": p.CACHE_SCHEMA, "complete": True, "count": len(rows),
                "dataset_sha256": inputs.dataset_sha256, "encoder": plan["encoder"],
                "source_sha256": inputs.core_source_sha256, "limits": asdict(limits),
                "selection_sha256": p.digest(p.canonical(plan["canonical_frame_ids"])),
                "rows_sha256": p.file_hash(output / "rows.json", 16 * 1024 * 1024),
                "features_sha256": p.file_hash(output / "features.npy", MAX_ROWS * 960 * 4 + 4096)}
            p.check_deadline(deadline)
            bounded_json(output / "cache.json", metadata, 16 * 1024 * 1024)
            return metadata
        except Exception as error:
            mark_failure(output, error)
            raise

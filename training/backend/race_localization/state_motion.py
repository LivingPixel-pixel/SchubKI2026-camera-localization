"""Experimental speed/yaw-rate projection, requiring a separately approved local context.

Only the narrow motion_projection fields are decoded. Integration requires both
source-time and recorded-arrival prefixes and complete fresh coverage. This module
does not establish permission, clock conventions or actual device behavior.
"""
from __future__ import annotations

import math

import numpy as np

from race_localization import motion_projection as mp

SCHEMA = "schubki-state-motion-v1"
_FIELDS = {"source_ns", "log_ns", "v", "wz", "schema_sha256", "recording_sha256", "rejected"}


class MotionCoverageError(ValueError):
    """The admitted arrival prefix cannot cover the whole requested interval."""


def _sha_pin(value, label):
    if type(value) is not str or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{label} must be a lowercase SHA256 pin")
    return value


def _ns(value, label):
    if type(value) is not int or not 0 <= value < 2**63:
        raise ValueError(f"{label} must be nonnegative integer nanoseconds")
    return value


def _file_sha(path):
    import hashlib
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def extract(mcap_path, output, expected_schema_sha256=None):
    """Project every state_estimation record to (source_ns, log_ns, v, wz); returns the schema pin."""
    _sha_pin(expected_schema_sha256, "expected schema")
    from mcap.reader import make_reader
    rows, rejected, schema_pin = [], 0, None
    with open(mcap_path, "rb") as stream:
        reader = make_reader(stream)
        for seq, (schema, channel, message) in enumerate(reader.iter_messages(topics=[mp.TOPIC])):
            pin = mp.schema_sha256(bytes(schema.data))
            if schema_pin is None:
                schema_pin = pin
                if expected_schema_sha256 is not None and pin != expected_schema_sha256:
                    raise ValueError("state_estimation schema differs from the pinned schema")
            try:
                sample = mp.project_cdr(channel.topic, schema.name, pin, bytes(message.data), message.log_time, seq,
                                        expected_schema_sha256=schema_pin, schema_encoding=schema.encoding,
                                        message_encoding=channel.message_encoding)
            except mp.MotionProjectionError:
                rejected += 1
                continue
            rows.append((sample.source_ns, sample.log_ns, sample.v_mps, sample.wz_radps))
    if not rows:
        raise ValueError("no state_estimation motion samples")
    order = np.argsort(np.array([r[0] for r in rows], dtype=np.int64), kind="stable")
    np.savez(output, source_ns=np.array([r[0] for r in rows], np.int64)[order],
             log_ns=np.array([r[1] for r in rows], np.int64)[order],
             v=np.array([r[2] for r in rows], np.float64)[order], wz=np.array([r[3] for r in rows], np.float64)[order],
             schema_sha256=np.array(schema_pin), rejected=np.array(rejected),
             recording_sha256=np.array(_file_sha(mcap_path)))
    return {"samples": len(rows), "rejected": rejected, "schema_sha256": schema_pin}


class StateMotion:
    """Validated projected samples; local permission and clock pins belong to the caller."""
    def __init__(self, path):
        archive = np.load(path, allow_pickle=False)
        if not hasattr(archive, "files"):
            raise ValueError("state motion must be an NPZ archive")
        with archive as data:
            if set(data.files) != _FIELDS or len(data.files) != len(_FIELDS):
                raise ValueError("state motion archive fields differ from the pinned format")
            self.t, self.log_ns = data["source_ns"], data["log_ns"]
            self.v, self.wz = data["v"], data["wz"]
            for name, values in (("source_ns", self.t), ("log_ns", self.log_ns)):
                if values.ndim != 1 or values.dtype != np.dtype("int64") or np.any(values < 0):
                    raise ValueError(f"{name} must be a nonnegative int64 vector")
            if (not len(self.t) or self.log_ns.shape != self.t.shape
                    or np.any(self.t[1:] <= self.t[:-1])):
                raise ValueError("motion timestamps must be aligned and source timestamps unique increasing")
            for name, values in (("v", self.v), ("wz", self.wz)):
                if values.shape != self.t.shape or values.dtype != np.dtype("float64") or not np.isfinite(values).all():
                    raise ValueError(f"{name} must be an aligned finite float64 vector")
            for name in ("schema_sha256", "recording_sha256"):
                value = data[name]
                if value.shape != () or value.dtype.kind != "U":
                    raise ValueError(f"{name} must be one text pin")
                setattr(self, name, _sha_pin(value.item(), name))
            rejected = data["rejected"]
            if rejected.shape != () or rejected.dtype != np.dtype("int64") or rejected.item() < 0:
                raise ValueError("rejected count must be one nonnegative int64")
            self.rejected = int(rejected.item())
        for values in (self.t, self.log_ns, self.v, self.wz):
            values.setflags(write=False)

    def delta(self, start_ns, end_ns, max_hold_s=0.5, *, available_log_ns=None):
        """Integrate only admitted samples; stale or missing coverage raises, never fabricates zero motion."""
        _ns(start_ns, "start_ns")
        _ns(end_ns, "end_ns")
        if end_ns <= start_ns:
            return np.zeros(3)
        _ns(available_log_ns, "available_log_ns")
        if (type(max_hold_s) not in (int, float) or not math.isfinite(max_hold_s)
                or max_hold_s <= 0 or not math.isfinite(max_hold_s * 1e9)):
            raise ValueError("max_hold_s must be finite and positive")
        hold_ns = max_hold_s * 1e9
        first = int(np.searchsorted(self.t, start_ns, side="right"))
        last = int(np.searchsorted(self.t, end_ns, side="right"))
        held = first - 1
        while held >= 0 and self.log_ns[held] > available_log_ns:
            held -= 1
        x = y = yaw = 0.0

        def integrate(a, b, index):
            nonlocal x, y, yaw
            if b <= a:
                return
            if index < 0 or b - int(self.t[index]) > hold_ns:
                raise MotionCoverageError("motion interval has uncovered or stale arrival-prefix samples")
            dt = (b - a) / 1e9
            mid = yaw + 0.5 * self.wz[index] * dt
            x += self.v[index] * dt * math.cos(mid)
            y += self.v[index] * dt * math.sin(mid)
            yaw += self.wz[index] * dt

        cursor = start_ns
        for index in range(first, last):
            if self.log_ns[index] > available_log_ns:
                continue
            stamp = int(self.t[index])
            integrate(cursor, stamp, held)
            cursor, held = stamp, index
        integrate(cursor, end_ns, held)
        delta = np.array([x, y, yaw])
        if not np.isfinite(delta).all():
            raise ValueError("integrated motion must be finite")
        return delta

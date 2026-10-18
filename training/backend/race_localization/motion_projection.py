"""Narrow required-prefix CDR projector for scalar motion (TEAM-MOTION-01).

Only five values leave this module: the header source stamp, the record's
arrival log time, its arrival sequence, the scalar speed ``v`` and the yaw rate
``wz``. The projector walks the CDR v1 (XCDR1 plain) layout by offset and
unpacks exactly the header stamp, the frame_id length and the two permitted
float64 signals. Pose, accuracy, Frenet, angular x/y, linear and wheel-rate
bytes, the frame_id text and every trailing byte stay opaque. No complete
StateEstimation object is created. Only the literal four-byte headers
00 00 00 00 (CDR_BE) and 00 01 00 00 (CDR_LE) are supported in this first
version; anything else raises UnsupportedEncapsulation.

Declared units are metres/second and radians/second. Their actual sign, BODY
reference point/frame and header-versus-log clock relation are UNVERIFIED and
remain a coordinator fit-check prerequisite. This is a source interface only;
it does not admit any recording schema or authorize real telemetry use.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import re
import struct

__all__ = [
    "MAX_FRAME_ID_BYTES", "MAX_PAYLOAD_BYTES", "MESSAGE_ENCODING", "MESSAGE_TYPE",
    "MotionProjectionError", "MotionSample", "SCHEMA_ENCODING", "TOPIC", "UnsupportedEncapsulation",
    "project_cdr", "required_prefix_bytes", "schema_sha256",
]

TOPIC = "/constructor0/state_estimation"
MESSAGE_TYPE = "sd_localization_msgs/msg/StateEstimation"
SCHEMA_ENCODING = "ros2msg"
MESSAGE_ENCODING = "cdr"
MAX_PAYLOAD_BYTES = 16 * 1024
# CDR string length field value, which counts the terminating NUL.
MAX_FRAME_ID_BYTES = 128
_INT64_MAX = 2**63 - 1
_NS_PER_S = 1_000_000_000
_SHA256 = re.compile(r"[0-9a-f]{64}")
# First-version subset only: literal CDR_BE and CDR_LE headers with zero options.
_SUPPORTED_HEADERS = {b"\x00\x00\x00\x00": ">", b"\x00\x01\x00\x00": "<"}
_ENCAPSULATION_BYTES = 4
# Required prefix after the header string (sizes in bytes):
# int32 se_status, int32 se_state | 12 float64 pose/accuracy | float64 wx, wy |
# float64 wz (READ) | 7 float64 vx/vy/vz/wheel rates | float64 v (READ).
_SKIP_STATUS_STATE = 8
_SKIP_POSE_ACCURACY = 12 * 8
_SKIP_WX_WY = 2 * 8
_SKIP_LINEAR_WHEELS = 7 * 8


class MotionProjectionError(ValueError):
    """Explicit projection rejection with a stable machine-readable reason."""

    def __init__(self, reason: str, detail: str = ""):
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason


class UnsupportedEncapsulation(MotionProjectionError):
    """Outside the first-version subset; not a claim that the bytes are malformed CDR."""


def _exact_ns(value, name: str) -> int:
    if type(value) is not int or not 0 <= value <= _INT64_MAX:
        raise MotionProjectionError("invalid_integer", f"{name} must be an exact int in [0, 2**63-1]")
    return value


def _finite_float(value, name: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise MotionProjectionError("invalid_signal", f"{name} must be a finite float")
    return value


@dataclass(frozen=True, slots=True)
class MotionSample:
    """Immutable permitted motion observation; nothing else is retained."""

    source_ns: int
    log_ns: int
    seq: int
    v_mps: float
    wz_radps: float

    def __post_init__(self):
        _exact_ns(self.source_ns, "source_ns")
        _exact_ns(self.log_ns, "log_ns")
        _exact_ns(self.seq, "seq")
        _finite_float(self.v_mps, "v_mps")
        _finite_float(self.wz_radps, "wz_radps")


def schema_sha256(schema_bytes: bytes) -> str:
    """Hash raw schema bytes for an independently supplied pin comparison."""
    if type(schema_bytes) is not bytes:
        raise TypeError("schema bytes must be bytes")
    return hashlib.sha256(schema_bytes).hexdigest()


def _align(offset: int, size: int) -> int:
    return offset + (-offset % size)


def _layout(frame_id_length: int) -> tuple[int, int, int]:
    """Return body offsets of wz, v and the required prefix end."""
    offset = _align(12 + frame_id_length, 4) + _SKIP_STATUS_STATE
    offset = _align(offset, 8) + _SKIP_POSE_ACCURACY + _SKIP_WX_WY
    wz_offset = offset
    v_offset = wz_offset + 8 + _SKIP_LINEAR_WHEELS
    return wz_offset, v_offset, v_offset + 8


def required_prefix_bytes(frame_id_length: int) -> int:
    """Total payload bytes required for a frame_id length field value."""
    if type(frame_id_length) is not int or not 1 <= frame_id_length <= MAX_FRAME_ID_BYTES:
        raise MotionProjectionError("header_string_bounds", "frame_id length outside [1, 128]")
    return _ENCAPSULATION_BYTES + _layout(frame_id_length)[2]


def project_cdr(topic: str, message_type: str, schema_sha: str, payload: bytes, log_ns: int, seq: int, *,
                expected_schema_sha256: str, schema_encoding: str, message_encoding: str) -> MotionSample:
    """Project one CDR v1 record onto MotionSample(source, log, seq, v, wz).

    The source time is the message header stamp. ``log_ns`` and ``seq`` are the
    caller's arrival provenance and are never used as measurement time.
    """
    if topic != TOPIC:
        raise MotionProjectionError("unsupported_topic")
    if message_type != MESSAGE_TYPE:
        raise MotionProjectionError("unsupported_message_type")
    if schema_encoding != SCHEMA_ENCODING or message_encoding != MESSAGE_ENCODING:
        raise MotionProjectionError("unsupported_encoding")
    if type(expected_schema_sha256) is not str or _SHA256.fullmatch(expected_schema_sha256) is None:
        raise MotionProjectionError("invalid_schema_pin", "expected pin must be lowercase SHA256 hex")
    if type(schema_sha) is not str or _SHA256.fullmatch(schema_sha) is None:
        raise MotionProjectionError("invalid_schema_identity", "schema identity must be lowercase SHA256 hex")
    if schema_sha != expected_schema_sha256:
        raise MotionProjectionError("schema_identity_mismatch")
    _exact_ns(log_ns, "log_ns")
    _exact_ns(seq, "seq")
    if type(payload) is not bytes:
        raise MotionProjectionError("invalid_payload_type", "payload must be immutable bytes")
    if not _ENCAPSULATION_BYTES <= len(payload) <= MAX_PAYLOAD_BYTES:
        raise MotionProjectionError("payload_bounds")
    endian = _SUPPORTED_HEADERS.get(payload[:4])
    if endian is None:
        raise UnsupportedEncapsulation("unsupported_encapsulation", payload[:4].hex())
    body = memoryview(payload)[_ENCAPSULATION_BYTES:]
    if len(body) < 12:
        raise MotionProjectionError("truncated_required_prefix", "header")
    sec, nanosec, frame_id_length = struct.unpack_from(endian + "iII", body, 0)
    if sec < 0 or nanosec >= _NS_PER_S:
        raise MotionProjectionError("invalid_header_stamp")
    if not 1 <= frame_id_length <= MAX_FRAME_ID_BYTES:
        raise MotionProjectionError("header_string_bounds")
    if len(body) < 12 + frame_id_length:
        raise MotionProjectionError("truncated_required_prefix", "header string")
    # Structural terminator check only; the frame_id text is never decoded or exposed.
    if body[12 + frame_id_length - 1] != 0:
        raise MotionProjectionError("header_string_unterminated")
    wz_offset, v_offset, end = _layout(frame_id_length)
    if len(body) < end:
        raise MotionProjectionError("truncated_required_prefix", "motion signals")
    (wz,) = struct.unpack_from(endian + "d", body, wz_offset)
    (v,) = struct.unpack_from(endian + "d", body, v_offset)
    if not (math.isfinite(wz) and math.isfinite(v)):
        raise MotionProjectionError("nonfinite_signal")
    return MotionSample(sec * _NS_PER_S + nanosec, log_ns, seq, float(v), float(wz))

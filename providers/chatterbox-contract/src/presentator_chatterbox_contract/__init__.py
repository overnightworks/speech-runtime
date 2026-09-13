"""The narrow binary protocol shared by the Chatterbox parent and provider."""

import struct
from dataclasses import dataclass
from enum import IntEnum
from io import BytesIO
from typing import BinaryIO

CHATTERBOX_SAMPLE_RATE = 24_000
CHATTERBOX_REVISION = "5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18"
CHATTERBOX_ARTIFACTS = (
    "ve.pt",
    "t3_mtl23ls_v3.safetensors",
    "s3gen.pt",
    "grapheme_mtl_merged_expanded_v1.json",
    "conds.pt",
    "Cangjie5_TC.json",
)
MAX_FRAME_LENGTH = 1_048_585
FRAME_OVERHEAD = 9
MIN_PCM_BYTES = 2
MAX_PCM_BYTES = 1_048_576
MIN_SYNTHESIS_BYTES = 2
MAX_SYNTHESIS_BYTES = 16_385
_HEADER = struct.Struct("!IBQ")
_LENGTH = struct.Struct("!I")


class FrameKind(IntEnum):
    """The seven directions and terminal states of the provider protocol."""

    READY = 1
    SYNTHESIZE = 2
    CANCEL = 3
    PCM = 4
    COMPLETE = 5
    CANCELLED = 6
    FAILED = 7


class ProviderFailure(IntEnum):
    """The sole sanitized provider failure vocabulary."""

    PROVIDER_FAILURE = 1


class ProtocolError(ValueError):
    """A malformed private provider frame."""


@dataclass(frozen=True, slots=True)
class Frame:
    """One framed provider message after its length prefix."""

    kind: FrameKind
    request_id: int
    payload: bytes


class FrameDecoder:
    """Incrementally decode complete frames without blocking on partial input."""

    def __init__(self) -> None:
        """Start with no partial frame bytes."""
        self._buffer = bytearray()

    def feed(self, data: bytes) -> None:
        """Append bytes after rejecting an invalid announced length promptly."""
        self._buffer.extend(data)
        self._validate_announced_length()

    def next_frame(self) -> Frame | None:
        """Return one complete frame, retaining any partial or following frame."""
        if len(self._buffer) < _LENGTH.size:
            return None
        (length,) = _LENGTH.unpack_from(self._buffer)
        _validate_frame_length(length)
        total = _LENGTH.size + length
        if len(self._buffer) < total:
            return None
        encoded = bytes(self._buffer[:total])
        del self._buffer[:total]
        return read_frame(BytesIO(encoded))

    def finish(self) -> None:
        """Reject EOF when it leaves an incomplete frame."""
        if self._buffer:
            raise ProtocolError

    def _validate_announced_length(self) -> None:
        if len(self._buffer) >= _LENGTH.size:
            (length,) = _LENGTH.unpack_from(self._buffer)
            _validate_frame_length(length)


def write_frame(stream: BinaryIO | None, frame: Frame) -> None:
    """Validate and write one complete network-order frame."""
    if stream is None:
        raise OSError
    _validate(frame)
    stream.write(
        _HEADER.pack(FRAME_OVERHEAD + len(frame.payload), frame.kind, frame.request_id)
    )
    stream.write(frame.payload)
    stream.flush()


def read_frame(stream: BinaryIO | None) -> Frame:
    """Read and validate one complete network-order frame."""
    if stream is None:
        raise EOFError
    length, raw_kind, request_id = _HEADER.unpack(_read_exact(stream, _HEADER.size))
    _validate_frame_length(length)
    try:
        kind = FrameKind(raw_kind)
    except ValueError as error:
        raise ProtocolError from error
    frame = Frame(kind, request_id, _read_exact(stream, length - FRAME_OVERHEAD))
    _validate(frame)
    return frame


def _read_exact(stream: BinaryIO, size: int) -> bytes:
    value = bytearray()
    while len(value) < size:
        chunk = stream.read(size - len(value))
        if not chunk:
            raise EOFError
        value.extend(chunk)
    return bytes(value)


def _validate_frame_length(length: int) -> None:
    if length < FRAME_OVERHEAD or length > MAX_FRAME_LENGTH:
        raise ProtocolError


def _validate(frame: Frame) -> None:
    if frame.request_id < 0 or frame.request_id >= 1 << 64:
        raise ProtocolError
    payload = frame.payload
    if frame.kind is FrameKind.READY:
        valid = frame.request_id == 0 and payload == b"\x00\x01\x00\x00]\xc0"
    elif frame.kind is FrameKind.SYNTHESIZE:
        valid = frame.request_id != 0 and _valid_synthesis(payload)
    elif frame.kind in {FrameKind.CANCEL, FrameKind.COMPLETE, FrameKind.CANCELLED}:
        valid = frame.request_id != 0 and not payload
    elif frame.kind is FrameKind.PCM:
        valid = frame.request_id != 0 and _valid_pcm(payload)
    else:
        valid = payload == bytes([ProviderFailure.PROVIDER_FAILURE])
    if not valid:
        raise ProtocolError


def _valid_synthesis(payload: bytes) -> bool:
    if not MIN_SYNTHESIS_BYTES <= len(payload) <= MAX_SYNTHESIS_BYTES:
        return False
    if payload[0] not in {1, 2}:
        return False
    try:
        payload[1:].decode("utf-8")
    except UnicodeDecodeError:
        return False
    return bool(payload[1:])


def _valid_pcm(payload: bytes) -> bool:
    return MIN_PCM_BYTES <= len(payload) <= MAX_PCM_BYTES and len(payload) % 2 == 0

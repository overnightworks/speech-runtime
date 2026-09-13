"""The narrow binary protocol and worker loop shared by local providers."""

import os
import queue
import struct
import sys
import threading
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from enum import IntEnum, StrEnum
from io import BytesIO
from pathlib import Path
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
QWEN_SAMPLE_RATE = 24_000
QWEN_MODEL_ID = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
QWEN_REVISION = "85e237c12c027371202489a0ec509ded67b5e4b5"
QWEN_ARTIFACTS = (
    ".gitattributes",
    "README.md",
    "config.json",
    "generation_config.json",
    "merges.txt",
    "model.safetensors",
    "preprocessor_config.json",
    "speech_tokenizer/config.json",
    "speech_tokenizer/configuration.json",
    "speech_tokenizer/model.safetensors",
    "speech_tokenizer/preprocessor_config.json",
    "tokenizer_config.json",
    "vocab.json",
)
VOXCPM_MODEL_ID = "openbmb/VoxCPM2"
VOXCPM_REVISION = "32279effe8c19989596f05d353d1447f51d9e915"
VOXCPM_ARTIFACTS = (
    ".gitattributes",
    "README.md",
    "audiovae.pth",
    "config.json",
    "model.safetensors",
    "special_tokens_map.json",
    "tokenization_voxcpm2.py",
    "tokenizer.json",
    "tokenizer_config.json",
)
VOXCPM_SAMPLE_RATE = 48_000
MAGPIE_MODEL_ID = "nvidia/magpie_tts_multilingual_357m"
MAGPIE_MODEL_REVISION = "5023df68bd3f5b5ce6d666a50979bc501af145cc"
MAGPIE_MODEL_FILENAME = "magpie_tts_multilingual_357m.nemo"
MAGPIE_CODEC_ID = "nvidia/nemo-nano-codec-22khz-1.89kbps-21.5fps"
MAGPIE_CODEC_REVISION = "fc00890b604aa2de298d2641ffc6c5f6caf8c4d7"
MAGPIE_CODEC_FILENAME = "nemo-nano-codec-22khz-1.89kbps-21.5fps.nemo"
MAGPIE_SAMPLE_RATE = 22_050


class QwenSpeaker(StrEnum):
    """The closed custom-voice values shared across the process boundary."""

    VIVIAN = "Vivian"
    SERENA = "Serena"
    UNCLE_FU = "Uncle_Fu"
    DYLAN = "Dylan"
    ERIC = "Eric"
    RYAN = "Ryan"
    AIDEN = "Aiden"
    ONO_ANNA = "Ono_Anna"
    SOHEE = "Sohee"


class MagpieSpeaker(StrEnum):
    """The fixed Magpie voices accepted across the process boundary."""

    ARIA = "Aria"
    JASON = "Jason"
    JOHN = "John"
    LEO = "Leo"
    SOFIA = "Sofia"


MAX_FRAME_LENGTH = 1_048_585
FRAME_OVERHEAD = 9
MIN_PCM_BYTES = 2
MAX_PCM_BYTES = 1_048_576
MIN_SYNTHESIS_BYTES = 2
MAX_SYNTHESIS_BYTES = 16_385
_HEADER = struct.Struct("!IBQ")
_LENGTH = struct.Struct("!I")
_READY = struct.Struct("!HI")


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
        valid = frame.request_id == 0 and _ready_sample_rate(payload) is not None
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


@dataclass(frozen=True, slots=True)
class ProviderFunctions:
    """Provider callbacks behind the shared command loop."""

    load: Callable[[str, Path], object]
    synthesize: Callable[[object, str, str], Iterable[bytes]]
    prepare: Callable[[object], None] | None = None


def serve_provider(
    *,
    device: str,
    cache: str,
    protocol_stdout: int,
    sample_rate: int,
    functions: ProviderFunctions,
) -> int:
    """Load once, optionally prepare, then process serialized commands."""
    stream = os.fdopen(protocol_stdout, "wb", buffering=0)
    try:
        model = functions.load(device, Path(cache))
        if functions.prepare is not None:
            functions.prepare(model)
        write_frame(stream, Frame(FrameKind.READY, 0, ready_payload(sample_rate)))
    except Exception:
        _failed(stream, 0)
        stream.close()
        return 1
    commands: queue.Queue[Frame | BaseException] = queue.Queue()
    reader_stopped = threading.Event()
    reader = threading.Thread(
        target=_read_commands,
        args=(commands, reader_stopped),
        name="speech-provider-input",
        daemon=True,
    )
    reader.start()
    try:
        return serve_commands(commands, stream, functions.synthesize, model)
    finally:
        if reader_stopped.is_set():
            reader.join()
        stream.close()


def _read_commands(
    commands: queue.Queue[Frame | BaseException], stopped: threading.Event
) -> None:
    try:
        while True:
            commands.put(read_frame(sys.stdin.buffer))
    except (EOFError, ProtocolError, OSError) as error:
        stopped.set()
        commands.put(error)


def serve_commands(
    commands: queue.Queue[Frame | BaseException],
    stream: BinaryIO,
    synthesize: Callable[[object, str, str], Iterable[bytes]],
    model: object,
) -> int:
    """Serve validated commands until input or synthesis fails."""
    previous_terminal = 0
    while True:
        command = commands.get()
        if isinstance(command, BaseException):
            _failed(stream, 0)
            return 1
        if command.kind is FrameKind.CANCEL and command.request_id == previous_terminal:
            continue
        if (
            command.kind is not FrameKind.SYNTHESIZE
            or command.request_id <= previous_terminal
        ):
            _failed(stream, 0)
            return 1
        request_id = command.request_id
        language = {1: "de", 2: "en"}[command.payload[0]]
        text = command.payload[1:].decode("utf-8")
        iterator: Iterable[bytes] | None = None
        try:
            cancelled = False
            iterator = iter(synthesize(model, text, language))
            for pcm in iterator:
                cancelled = _consume_cancellation(
                    commands, request_id, previous_terminal
                )
                if cancelled:
                    break
                write_frame(stream, Frame(FrameKind.PCM, request_id, pcm))
            if not cancelled:
                cancelled = _consume_cancellation(
                    commands, request_id, previous_terminal
                )
            terminal = FrameKind.CANCELLED if cancelled else FrameKind.COMPLETE
        except Exception:
            return _fail_request(stream, request_id, iterator)
        try:
            _close_iterator(iterator)
        except Exception:
            return _fail_request(stream, request_id, None)
        write_frame(stream, Frame(terminal, request_id, b""))
        previous_terminal = request_id


def _consume_cancellation(
    commands: queue.Queue[Frame | BaseException],
    request_id: int,
    previous_terminal: int,
) -> bool:
    cancelled = False
    while True:
        try:
            command = commands.get_nowait()
        except queue.Empty:
            return cancelled
        if isinstance(command, BaseException):
            raise command
        if command.kind is not FrameKind.CANCEL:
            raise ProtocolError
        if command.request_id == previous_terminal:
            continue
        if command.request_id != request_id:
            raise ProtocolError
        cancelled = True


def ready_payload(sample_rate: int) -> bytes:
    """Encode the version-one worker declaration for one native PCM rate."""
    if sample_rate <= 0 or sample_rate >= 1 << 32:
        raise ValueError
    return _READY.pack(1, sample_rate)


def _ready_sample_rate(payload: bytes) -> int | None:
    if len(payload) != _READY.size:
        return None
    version, sample_rate = _READY.unpack(payload)
    return sample_rate if version == 1 and sample_rate > 0 else None


def ready_sample_rate(payload: bytes) -> int:
    """Return a validated version-one provider rate."""
    sample_rate = _ready_sample_rate(payload)
    if sample_rate is None:
        raise ProtocolError
    return sample_rate


def _close_iterator(iterator: Iterable[bytes] | None) -> None:
    close = getattr(iterator, "close", None)
    if callable(close):
        close()


def _fail_request(
    stream: BinaryIO, request_id: int, iterator: Iterable[bytes] | None
) -> int:
    try:
        _close_iterator(iterator)
    except Exception:
        _failed(stream, request_id)
        return 1
    _failed(stream, request_id)
    return 1


def _failed(stream: BinaryIO, request_id: int) -> None:
    try:
        write_frame(
            stream,
            Frame(
                FrameKind.FAILED, request_id, bytes([ProviderFailure.PROVIDER_FAILURE])
            ),
        )
    except (OSError, ProtocolError):
        pass

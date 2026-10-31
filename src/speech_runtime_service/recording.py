"""A recording, decoded incrementally into the 16 kHz mono PCM the engine hears."""

from __future__ import annotations

import io
from itertools import chain
from typing import TYPE_CHECKING, Final

import av
from av.error import FFmpegError

from speech_runtime_service.hearing import HEARING_SAMPLE_RATE
from speech_runtime_service.pcm import BYTES_PER_SAMPLE

if TYPE_CHECKING:
    from av.container import InputContainer

_PCM_FORMAT: Final = "s16"
_MONO: Final = "mono"


class RecordingTooLongError(Exception):
    """The recording is longer than the configured length."""


class UndecodableRecordingError(Exception):
    """The bytes are not audio PyAV can read, or hold no sound."""


def decode_recording(recording: bytes, max_seconds: float) -> bytes:
    """16-bit PCM of `recording`, decoding only as far as `max_seconds` allow.

    Raises `RecordingTooLongError` the moment the decoded audio passes the
    length, and `UndecodableRecordingError` for anything that is not sound.
    """
    limit = int(max_seconds * HEARING_SAMPLE_RATE) * BYTES_PER_SAMPLE
    try:
        with av.open(io.BytesIO(recording)) as container:
            pcm = _decode_within(container, limit)
    except FFmpegError, IndexError:
        raise UndecodableRecordingError from None
    if not pcm:
        raise UndecodableRecordingError
    return bytes(pcm)


def _decode_within(container: InputContainer, limit: int) -> bytearray:
    resampler = av.AudioResampler(
        format=_PCM_FORMAT, layout=_MONO, rate=HEARING_SAMPLE_RATE
    )
    pcm = bytearray()
    for frame in chain(container.decode(container.streams.audio[0]), [None]):
        for resampled in resampler.resample(frame):
            pcm += resampled.to_ndarray().tobytes()
            if len(pcm) > limit:
                raise RecordingTooLongError
    return pcm

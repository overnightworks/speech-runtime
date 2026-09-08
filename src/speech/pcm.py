"""16-bit PCM helpers for the wire: a streaming WAV header and energy."""

from __future__ import annotations

import math
import struct
from typing import Final

BITS_PER_SAMPLE: Final = 16
BYTES_PER_SAMPLE: Final = BITS_PER_SAMPLE // 8
CHANNELS: Final = 1
PCM_FORMAT: Final = 1
WAV_HEADER_BYTES: Final = 44
# Streaming WAV: the length is not known when the first byte leaves.
STREAMING_DATA_SIZE: Final = 0xFFFFFFFF
SILENCE_RMS: Final = 0.015


def wav_header(sample_rate: int, data_bytes: int = STREAMING_DATA_SIZE) -> bytes:
    """A 44-byte PCM WAV header for 16-bit mono at `sample_rate`."""
    byte_rate = sample_rate * CHANNELS * BYTES_PER_SAMPLE
    block_align = CHANNELS * BYTES_PER_SAMPLE
    riff_size = (36 + data_bytes) & 0xFFFFFFFF
    return struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        riff_size,
        b"WAVE",
        b"fmt ",
        16,
        PCM_FORMAT,
        CHANNELS,
        sample_rate,
        byte_rate,
        block_align,
        BITS_PER_SAMPLE,
        b"data",
        data_bytes & 0xFFFFFFFF,
    )


def pcm_from_wav(data: bytes) -> tuple[int, bytes]:
    """Sample rate and PCM payload of a 16-bit mono WAV, ignoring a streaming size."""
    if len(data) < WAV_HEADER_BYTES or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        message = "not a WAV body"
        raise ValueError(message)
    rate = int.from_bytes(data[24:28], "little")
    position = 12
    while position + 8 <= len(data):
        chunk_id = data[position : position + 4]
        chunk_size = int.from_bytes(data[position + 4 : position + 8], "little")
        payload_at = position + 8
        if chunk_id == b"data":
            return rate, data[payload_at:]
        if chunk_size == STREAMING_DATA_SIZE:
            message = "WAV chunk before data has no length"
            raise ValueError(message)
        position = payload_at + chunk_size
    message = "WAV has no data chunk"
    raise ValueError(message)


def rms(pcm: bytes) -> float:
    """Root-mean-square of 16-bit samples, scaled to 0..1."""
    if len(pcm) < BYTES_PER_SAMPLE:
        return 0.0
    count = len(pcm) // BYTES_PER_SAMPLE
    total = 0.0
    for index in range(count):
        sample = int.from_bytes(
            pcm[index * BYTES_PER_SAMPLE : (index + 1) * BYTES_PER_SAMPLE],
            "little",
            signed=True,
        )
        total += sample * sample
    return math.sqrt(total / count) / 32768.0


def is_silence(pcm: bytes, threshold: float = SILENCE_RMS) -> bool:
    """True when the PCM is quieter than the threshold."""
    return rms(pcm) < threshold

"""The 24 kHz to 16 kHz resampler keeps its state across frame boundaries."""

import itertools
import struct

import pytest

from speech_runtime_service.pcm import Resampler
from tests.support.fakes_support import sine_pcm

INPUT_RATE = 24_000
OUTPUT_RATE = 16_000


def _samples(pcm: bytes) -> list[int]:
    return list(struct.unpack(f"<{len(pcm) // 2}h", pcm))


def _in_frames(pcm: bytes, sizes: list[int]) -> bytes:
    resampler = Resampler()
    out = b""
    position = 0
    for size in sizes:
        out += resampler.feed(pcm[position : position + size])
        position += size
    return out + resampler.feed(pcm[position:])


def test_a_tone_split_across_frames_is_resampled_as_if_it_came_in_one() -> None:
    tone = sine_pcm(0.5, INPUT_RATE)

    whole = Resampler().feed(tone)

    assert _in_frames(tone, [2, 6, 1000, 2]) == whole


@pytest.mark.parametrize("split", [2, 4, 6, 2400])
def test_any_split_of_the_input_gives_the_same_output(split: int) -> None:
    tone = sine_pcm(0.2, INPUT_RATE)

    assert _in_frames(tone, [split]) == Resampler().feed(tone)


def test_two_thirds_as_many_samples_leave_as_came_in() -> None:
    out = Resampler().feed(sine_pcm(1.0, INPUT_RATE))

    assert abs(len(out) // 2 - OUTPUT_RATE) <= 1


def test_a_constant_level_stays_that_level() -> None:
    level = 1234
    out = Resampler().feed(struct.pack("<300h", *[level] * 300))

    assert set(_samples(out)) == {level}


def test_a_tone_keeps_its_pitch() -> None:
    out = _samples(Resampler().feed(sine_pcm(1.0, INPUT_RATE, freq=440.0)))

    rising = sum(1 for a, b in itertools.pairwise(out) if a < 0 <= b)
    assert abs(rising - 440) <= 1

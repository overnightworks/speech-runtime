"""A recording becomes 16 kHz mono PCM, or is refused as too long or no sound."""

import pytest

from speech_runtime_service.hearing import HEARING_SAMPLE_RATE
from speech_runtime_service.pcm import BYTES_PER_SAMPLE
from speech_runtime_service.recording import (
    RecordingTooLongError,
    UndecodableRecordingError,
    decode_recording,
)
from tests.support.hear_support import recording

NO_LIMIT_SECONDS = 60.0


def test_a_recording_is_decoded_to_the_rate_the_engine_hears() -> None:
    pcm = decode_recording(recording(seconds=1.0), NO_LIMIT_SECONDS)

    assert len(pcm) == HEARING_SAMPLE_RATE * BYTES_PER_SAMPLE


def test_a_recording_longer_than_the_limit_is_refused_not_cut() -> None:
    with pytest.raises(RecordingTooLongError):
        decode_recording(recording(seconds=1.0), 0.5)


def test_a_recording_exactly_as_long_as_the_limit_is_decoded() -> None:
    pcm = decode_recording(recording(seconds=1.0), 1.0)

    assert len(pcm) == HEARING_SAMPLE_RATE * BYTES_PER_SAMPLE


@pytest.mark.parametrize(
    "bytes_sent",
    [b"", b"not audio at all" * 64, recording(seconds=0.5)[:20]],
    ids=["empty", "garbage", "cut-header"],
)
def test_bytes_that_are_not_sound_are_refused(bytes_sent: bytes) -> None:
    with pytest.raises(UndecodableRecordingError):
        decode_recording(bytes_sent, NO_LIMIT_SECONDS)

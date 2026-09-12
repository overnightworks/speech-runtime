"""A hearing session emits partials and a second utterance without a new model."""

import pytest

from speech.config import HEAR_SAMPLE_RATE
from speech.hearing import (
    MAX_FRAME_SECONDS,
    MAX_UTTERANCE_SECONDS,
    FrameTooLargeError,
    HearingSession,
)
from speech.pcm import BYTES_PER_SAMPLE
from tests.conftest import sine_pcm


def _silence(seconds: float) -> bytes:
    return bytes(int(seconds * HEAR_SAMPLE_RATE * BYTES_PER_SAMPLE))


def test_session_partial_then_final_then_a_second_utterance() -> None:
    def transcribe(pcm: bytes) -> str:
        duration = len(pcm) / (HEAR_SAMPLE_RATE * BYTES_PER_SAMPLE)
        if duration < 0.2:
            return ""
        if duration < 0.9:
            return "eins"
        return "eins zwei"

    session = HearingSession(transcribe, HEAR_SAMPLE_RATE)
    session.add_pcm(sine_pcm(0.5))
    partial = session.poll()
    assert partial == {"text": "eins", "final": False}

    session.add_pcm(_silence(0.8))
    final = session.poll()
    assert final == {"text": "eins zwei", "final": True}

    session.add_pcm(sine_pcm(0.5))
    second = session.poll()
    assert second == {"text": "eins", "final": False}


def test_session_skips_silence_until_speech_and_keeps_its_context() -> None:
    transcribed: list[bytes] = []

    def transcribe(pcm: bytes) -> str:
        transcribed.append(pcm)
        return "eins"

    session = HearingSession(transcribe, HEAR_SAMPLE_RATE)
    silence = _silence(0.5)
    speech = sine_pcm(0.5)
    session.add_pcm(silence)

    assert session.poll() is None
    assert transcribed == []

    session.add_pcm(speech)

    assert session.poll() == {"text": "eins", "final": False}
    assert transcribed == [silence + speech]


def test_session_rejects_a_frame_over_the_size_bound() -> None:
    session = HearingSession(lambda _pcm: "x", HEAR_SAMPLE_RATE)
    too_big = sine_pcm(MAX_FRAME_SECONDS + 0.1)

    with pytest.raises(FrameTooLargeError):
        session.add_pcm(too_big)


def test_session_forces_a_final_at_the_duration_cap() -> None:
    session = HearingSession(lambda _pcm: "lang", HEAR_SAMPLE_RATE)
    frame = sine_pcm(MAX_FRAME_SECONDS)
    for _ in range(int(MAX_UTTERANCE_SECONDS / MAX_FRAME_SECONDS)):
        session.add_pcm(frame)

    event = session.poll()

    assert event == {"text": "lang", "final": True}
    session.add_pcm(sine_pcm(0.5))
    second = session.poll()
    assert second == {"text": "lang", "final": False}


def test_session_resets_silent_audio_at_the_duration_cap() -> None:
    transcribed: list[bytes] = []

    def transcribe(pcm: bytes) -> str:
        transcribed.append(pcm)
        return "eins"

    session = HearingSession(transcribe, HEAR_SAMPLE_RATE)
    silent_frame = _silence(MAX_FRAME_SECONDS)
    for _ in range(int(MAX_UTTERANCE_SECONDS / MAX_FRAME_SECONDS)):
        session.add_pcm(silent_frame)

    assert session.poll() is None
    assert transcribed == []

    speech = sine_pcm(0.5)
    session.add_pcm(speech)

    assert session.poll() == {"text": "eins", "final": False}
    assert transcribed == [speech]

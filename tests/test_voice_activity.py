"""Silero over a session's audio: the shipped model, window by window."""

from pathlib import Path

import numpy as np
import onnxruntime
import pytest
from faster_whisper.vad import get_vad_model

from speech_service.hearing_session import HearingSession, SpeechStart, TurnEnd
from speech_service.pcm import pcm_from_wav
from speech_service.voice_activity import (
    SAMPLE_RATE,
    START_THRESHOLD,
    WINDOW_SAMPLES,
    ModelShapeError,
    SileroDetectors,
)

MIN_VOICED_SECONDS = 0.5
LEAD_MS = 900
SPEECH = Path(__file__).parent / "fixtures" / "speech" / "nein_danke.wav"


@pytest.fixture(scope="module")
def detectors() -> SileroDetectors:
    loaded = SileroDetectors()
    loaded.load()
    return loaded


def _pcm(samples: np.ndarray) -> bytes:
    return np.clip(samples, -32768, 32767).astype("<i2").tobytes()


def _noise(seconds: float, level: float, seed: int = 1) -> bytes:
    generator = np.random.default_rng(seed)
    return _pcm(generator.standard_normal(int(seconds * SAMPLE_RATE)) * level)


def _speech() -> bytes:
    rate, pcm = pcm_from_wav(SPEECH.read_bytes())
    assert rate == SAMPLE_RATE
    return pcm


def _in_chunks(stream_of: SileroDetectors, pcm: bytes, size: int) -> list[float]:
    stream = stream_of.open()
    out: list[float] = []
    for start in range(0, len(pcm), size):
        out += stream.probabilities(pcm[start : start + size])
    return out


def test_a_split_of_the_audio_gives_the_probabilities_of_the_library_run(
    detectors: SileroDetectors,
) -> None:
    pcm = _noise(1.0, 3000) + _speech() + _noise(0.5, 100, seed=2)
    pcm = pcm[: len(pcm) // (2 * WINDOW_SAMPLES) * 2 * WINDOW_SAMPLES]
    samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
    # The library zeroes the last 64 samples of its input, so its last window differs.
    whole = get_vad_model()(samples.copy())[:-1]

    for size in (2, 1999, 3200, len(pcm)):
        chunked = _in_chunks(detectors, pcm, size)[:-1]
        assert chunked == pytest.approx(whole, abs=1e-6)


def test_two_streams_do_not_share_their_state(detectors: SileroDetectors) -> None:
    speech = _speech()
    alone = _in_chunks(detectors, speech, 3200)
    first, second = detectors.open(), detectors.open()

    first.probabilities(_noise(0.5, 3000))
    interleaved = []
    for start in range(0, len(speech), 3200):
        interleaved += second.probabilities(speech[start : start + 3200])
        first.probabilities(_noise(0.1, 3000))

    assert interleaved == alone


@pytest.mark.parametrize("level", [30, 3000, 10000], ids=["faint", "loud", "very-loud"])
def test_noise_of_any_loudness_is_no_speech(
    detectors: SileroDetectors, level: int
) -> None:
    probabilities = _in_chunks(detectors, _noise(10.0, level), 3200)

    assert max(probabilities) < START_THRESHOLD


def test_a_voice_is_one_turn_whatever_its_loudness(detectors: SileroDetectors) -> None:
    """A faint voice is a turn like a loud one: loudness is no part of speech."""
    speech = np.frombuffer(_speech(), dtype="<i2").astype(np.float32)
    for gain in (1.0, 0.03):
        session = HearingSession(lambda: 0.0, detectors.open())
        audio = _noise(1.0, 20) + _pcm(speech * gain) + _noise(1.0, 20)

        events = [
            event
            for start in range(0, len(audio), 3200)
            for event in session.add(audio[start : start + 3200])
        ]

        started = [event for event in events if isinstance(event, SpeechStart)]
        ended = [event for event in events if isinstance(event, TurnEnd)]
        assert len(started) == len(ended) == 1
        assert ended[0].voiced_seconds > MIN_VOICED_SECONDS
        assert started[0].audio_ms > LEAD_MS


def test_loud_noise_opens_no_turn(detectors: SileroDetectors) -> None:
    session = HearingSession(lambda: 0.0, detectors.open())

    events = session.add(_noise(20.0, 10000))

    assert events == []


class _OtherModel:
    def get_inputs(self) -> list[object]:
        return []

    def get_outputs(self) -> list[object]:
        return []


def test_a_model_of_another_shape_fails_the_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        onnxruntime, "InferenceSession", lambda *_args, **_kwargs: _OtherModel()
    )

    with pytest.raises(ModelShapeError):
        SileroDetectors().load()


def test_a_stream_of_an_unloaded_model_is_refused() -> None:
    with pytest.raises(RuntimeError, match="not loaded"):
        SileroDetectors().open()

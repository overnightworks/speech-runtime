"""POST /speak streams 16-bit mono WAV that is not silence."""

import pytest
from fastapi.testclient import TestClient
from starlette.status import (
    HTTP_200_OK,
    HTTP_422_UNPROCESSABLE_CONTENT,
    HTTP_503_SERVICE_UNAVAILABLE,
)

from speech.chatterbox import CHATTERBOX_SAMPLE_RATE
from speech.config import CHATTERBOX_SPEAKING_MODEL, Settings
from speech.pcm import is_silence, pcm_from_wav
from speech.speaking import PiperSpeaking, speaking_from_settings
from tests.conftest import SPEAK_SAMPLE_RATE, FakeSpeaking, an_app


def test_speak_streams_wav_that_plays_as_tone() -> None:
    response = TestClient(an_app()).post(
        "/speak",
        json={"text": "Guten Morgen", "language": "de"},
    )

    assert response.status_code == HTTP_200_OK
    assert response.headers["content-type"].startswith("audio/wav")
    rate, pcm = pcm_from_wav(response.content)
    assert rate == SPEAK_SAMPLE_RATE
    assert rate == 22_050
    assert len(pcm) > 0
    assert not is_silence(pcm)


def test_speak_refuses_an_empty_sentence() -> None:
    response = TestClient(an_app()).post(
        "/speak",
        json={"text": "", "language": "de"},
    )

    assert response.status_code == HTTP_422_UNPROCESSABLE_CONTENT


def test_speak_waits_when_the_voice_is_not_ready() -> None:
    response = TestClient(an_app(FakeSpeaking(ready=False))).post(
        "/speak",
        json={"text": "Hallo", "language": "de"},
    )

    assert response.status_code == HTTP_503_SERVICE_UNAVAILABLE


def test_speak_passes_language_to_the_voice() -> None:
    speaking = FakeSpeaking()
    TestClient(an_app(speaking)).post(
        "/speak",
        json={"text": "Hello", "language": "en"},
    )

    assert speaking.heard_language == "en"
    assert speaking.heard_text == "Hello"


def test_speak_wav_header_uses_the_voice_sample_rate() -> None:
    speaking = FakeSpeaking()
    speaking.sample_rate = CHATTERBOX_SAMPLE_RATE
    response = TestClient(an_app(speaking)).post(
        "/speak",
        json={"text": "Hallo", "language": "de"},
    )

    rate, _pcm = pcm_from_wav(response.content)
    assert rate == CHATTERBOX_SAMPLE_RATE


def test_default_speaking_model_is_piper(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SPEECH_SPEAKING_MODEL", raising=False)
    engine = speaking_from_settings(Settings())

    assert isinstance(engine, PiperSpeaking)
    assert engine.streams is False
    assert engine.model_name == "de_DE-thorsten-medium"


def test_chatterbox_model_name_selects_the_streaming_voice() -> None:
    engine = speaking_from_settings(
        Settings(speaking_model=CHATTERBOX_SPEAKING_MODEL, device="cuda"),
    )

    assert engine.model_name == CHATTERBOX_SPEAKING_MODEL
    assert engine.streams is True
    assert engine.sample_rate == CHATTERBOX_SAMPLE_RATE

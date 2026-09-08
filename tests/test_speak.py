"""POST /speak streams 16-bit mono WAV that is not silence."""

from fastapi.testclient import TestClient
from starlette.status import (
    HTTP_200_OK,
    HTTP_422_UNPROCESSABLE_CONTENT,
    HTTP_503_SERVICE_UNAVAILABLE,
)

from speech.pcm import is_silence, pcm_from_wav
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

"""GET /health answers the contract, including while the model is still loading."""

from fastapi.testclient import TestClient
from speech_runtime_provider_contract import CHATTERBOX_SAMPLE_RATE

from tests.support.fakes_support import (
    SPEAK_SAMPLE_RATE,
    FakeSpeaking,
    an_app,
)


def test_health_answers_while_models_are_not_ready() -> None:
    card_memory_mb = 17
    client = TestClient(
        an_app(
            FakeSpeaking(ready=False),
            memory_probe=lambda: card_memory_mb,
        ),
    )

    body = client.get("/health").json()

    assert set(body) == {"speaking", "hearing", "card_memory_mb"}
    assert body["speaking"] == {
        "model": "fake-voice",
        "ready": False,
        "streams": False,
        "sample_rate": SPEAK_SAMPLE_RATE,
    }
    assert body["card_memory_mb"] == card_memory_mb


def test_health_reports_a_streaming_voice_and_its_sample_rate() -> None:
    speaking = FakeSpeaking()
    speaking.streams = True
    speaking.sample_rate = CHATTERBOX_SAMPLE_RATE

    body = TestClient(an_app(speaking, memory_probe=lambda: 1)).get("/health").json()

    assert body["speaking"]["streams"] is True
    assert body["speaking"]["sample_rate"] == CHATTERBOX_SAMPLE_RATE


def test_health_reports_ready_models() -> None:
    card_memory_mb = 9
    body = TestClient(an_app(memory_probe=lambda: card_memory_mb)).get("/health").json()

    assert body["speaking"]["ready"] is True
    assert body["speaking"]["streams"] is False
    assert body["speaking"]["sample_rate"] == SPEAK_SAMPLE_RATE
    assert body["card_memory_mb"] == card_memory_mb

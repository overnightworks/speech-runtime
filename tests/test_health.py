"""GET /health answers the contract, including while models are still loading."""

from fastapi.testclient import TestClient

from tests.conftest import FakeHearing, FakeSpeaking, an_app


def test_health_answers_while_models_are_not_ready() -> None:
    client = TestClient(
        an_app(
            FakeSpeaking(ready=False),
            FakeHearing(ready=False),
            memory_probe=lambda: 17,
        ),
    )

    body = client.get("/health").json()

    assert body["speaking"] == {
        "model": "fake-voice",
        "ready": False,
        "streams": False,
        "sample_rate": 22_050,
    }
    assert body["hearing"] == {"model": "fake-ears", "ready": False}
    assert body["sample_rate"] == 16_000
    assert body["card_memory_mb"] == 17


def test_health_reports_a_streaming_voice_and_its_sample_rate() -> None:
    speaking = FakeSpeaking()
    speaking.streams = True
    speaking.sample_rate = 24_000

    body = TestClient(an_app(speaking, memory_probe=lambda: 1)).get("/health").json()

    assert body["speaking"]["streams"] is True
    assert body["speaking"]["sample_rate"] == 24_000
    assert body["sample_rate"] == 16_000


def test_health_reports_ready_models() -> None:
    body = TestClient(an_app(memory_probe=lambda: 9)).get("/health").json()

    assert body["speaking"]["ready"] is True
    assert body["speaking"]["streams"] is False
    assert body["speaking"]["sample_rate"] == 22_050
    assert body["hearing"]["ready"] is True
    assert body["sample_rate"] == 16_000
    assert body["card_memory_mb"] == 9

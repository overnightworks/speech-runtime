"""GET /health answers the contract, including while models are still loading."""

import threading

from fastapi.testclient import TestClient

from speech.service import Runtime, create_app
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
    }
    assert body["hearing"] == {"model": "fake-ears", "ready": False}
    assert body["sample_rate"] == 16_000
    assert body["card_memory_mb"] == 17


def test_health_reports_ready_models() -> None:
    body = TestClient(an_app(memory_probe=lambda: 9)).get("/health").json()

    assert body["speaking"]["ready"] is True
    assert body["speaking"]["streams"] is False
    assert body["hearing"]["ready"] is True
    assert body["sample_rate"] == 16_000
    assert body["card_memory_mb"] == 9


def test_health_answers_while_load_is_blocked() -> None:
    started = threading.Event()
    release = threading.Event()

    class BlockedSpeaking(FakeSpeaking):
        def __init__(self) -> None:
            super().__init__(ready=False)

        def load(self) -> None:
            started.set()
            if not release.wait(timeout=10):
                message = "test did not release the loader"
                raise RuntimeError(message)
            self.ready = True

    runtime = Runtime(
        BlockedSpeaking(),
        FakeHearing(ready=False),
        memory_probe=lambda: 42,
    )
    app = create_app(runtime=runtime, load_models=True)
    with TestClient(app) as client:
        try:
            assert started.wait(timeout=5)
            body = client.get("/health").json()
            assert body["speaking"]["ready"] is False
            assert body["hearing"]["ready"] is False
            assert body["sample_rate"] == 16_000
            assert body["card_memory_mb"] == 42
        finally:
            release.set()

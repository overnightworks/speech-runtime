"""WS /hear sends partials, then a final, and a second utterance without reload."""

import pytest
from fastapi.testclient import TestClient
from starlette.status import WS_1009_MESSAGE_TOO_BIG, WS_1011_INTERNAL_ERROR
from starlette.websockets import WebSocketDisconnect

from speech.config import HEAR_SAMPLE_RATE
from speech.hearing import MAX_FRAME_SECONDS, HearingSession
from speech.pcm import BYTES_PER_SAMPLE
from speech.service import FRAME_TOO_LARGE_REASON, HEARING_FAILED_REASON
from tests.conftest import FakeHearing, an_app, sine_pcm


def _silence(seconds: float) -> bytes:
    return bytes(int(seconds * HEAR_SAMPLE_RATE * BYTES_PER_SAMPLE))


def test_hear_sends_partials_then_a_final_and_a_second_utterance() -> None:
    hearing = FakeHearing()
    client = TestClient(an_app(hearing=hearing))
    speech = sine_pcm(0.5)
    pause = _silence(0.8)

    with client.websocket_connect("/hear?language=de") as socket:
        socket.send_bytes(speech)
        first = socket.receive_json()
        assert first["final"] is False
        assert "eins" in first["text"]
        socket.send_bytes(pause)
        final = socket.receive_json()
        assert final["final"] is True
        socket.send_bytes(speech)
        second = socket.receive_json()
        assert second["final"] is False
        assert "eins" in second["text"]

    assert hearing.sessions == 1


def test_hear_closes_while_the_model_is_not_ready() -> None:
    client = TestClient(an_app(hearing=FakeHearing(ready=False)))

    with (
        pytest.raises(WebSocketDisconnect),
        client.websocket_connect(
            "/hear?language=de",
        ) as socket,
    ):
        socket.receive_json()


class FailingSession(HearingSession):
    def __init__(self) -> None:
        super().__init__(lambda _pcm: "", HEAR_SAMPLE_RATE)

    def add_pcm(self, data: bytes) -> None:
        del data
        message = "transcribe exploded"
        raise RuntimeError(message)


class FailingHearing:
    model_name = "fake-ears"
    ready = True

    def load(self) -> None:
        self.ready = True

    def open_session(self, language: str) -> HearingSession:
        del language
        return FailingSession()


def test_hear_closes_with_a_reason_when_the_session_fails() -> None:
    client = TestClient(an_app(hearing=FailingHearing()))

    with client.websocket_connect("/hear?language=de") as socket:
        socket.send_bytes(sine_pcm(0.1))
        with pytest.raises(WebSocketDisconnect) as raised:
            socket.receive_json()

    assert raised.value.code == WS_1011_INTERNAL_ERROR
    assert raised.value.reason == HEARING_FAILED_REASON


def test_hear_closes_an_oversize_frame_with_a_reason() -> None:
    client = TestClient(an_app())
    too_big = sine_pcm(MAX_FRAME_SECONDS + 0.1)

    with client.websocket_connect("/hear?language=de") as socket:
        socket.send_bytes(too_big)
        with pytest.raises(WebSocketDisconnect) as raised:
            socket.receive_json()

    assert raised.value.code == WS_1009_MESSAGE_TOO_BIG
    assert raised.value.reason == FRAME_TOO_LARGE_REASON

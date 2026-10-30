"""WS /hear sends partials, then a final, and a second utterance without reload."""

import asyncio
import logging
import traceback

import pytest
from fastapi.testclient import TestClient
from starlette.status import WS_1009_MESSAGE_TOO_BIG, WS_1011_INTERNAL_ERROR
from starlette.websockets import WebSocketDisconnect

from speech_runtime_service.config import HEAR_SAMPLE_RATE
from speech_runtime_service.hearing import MAX_FRAME_SECONDS, HearingSession
from speech_runtime_service.pcm import BYTES_PER_SAMPLE
from speech_runtime_service.service import FRAME_TOO_LARGE_REASON, HEARING_FAILED_REASON
from tests.support.fakes_support import FakeHearing, an_app, sine_pcm


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


_HEARD = "Dieser Satz wurde gehoert und darf nicht ins Log"


class FailingSession(HearingSession):
    """A hearing session whose audio intake fails with the heard text in its message."""

    def __init__(self) -> None:
        """Build a session that transcribes nothing."""
        super().__init__(lambda _pcm: "", HEAR_SAMPLE_RATE)

    def add_pcm(self, data: bytes) -> None:
        """Raise instead of taking the audio."""
        del data
        raise RuntimeError(_HEARD)


class FailingHearing:
    """A hearing engine that opens only failing sessions."""

    model_name = "fake-ears"
    ready = True

    def load(self) -> None:
        """Mark the engine ready."""
        self.ready = True

    def open_session(self, language: str) -> HearingSession:
        """Open a session that fails on the first audio."""
        del language
        return FailingSession()


def test_hear_closes_with_a_reason_and_logs_the_failure_without_the_text(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG)
    client = TestClient(an_app(hearing=FailingHearing()))

    with client.websocket_connect("/hear?language=de") as socket:
        socket.send_bytes(sine_pcm(0.1))
        with pytest.raises(WebSocketDisconnect) as raised:
            socket.receive_json()

    assert raised.value.code == WS_1011_INTERNAL_ERROR
    assert raised.value.reason == HEARING_FAILED_REASON
    assert "hearing failed" in caplog.text
    assert "RuntimeError" in caplog.text
    assert 'File "' in caplog.text
    assert _HEARD not in caplog.text


def test_hear_writes_no_partial_or_final_to_the_log(
    caplog: pytest.LogCaptureFixture,
) -> None:
    class SayingHearing(FakeHearing):
        def open_session(self, language: str) -> HearingSession:
            del language
            return HearingSession(lambda _pcm: _HEARD, HEAR_SAMPLE_RATE)

    caplog.set_level(logging.DEBUG)
    client = TestClient(an_app(hearing=SayingHearing()))

    with client.websocket_connect("/hear?language=de") as socket:
        socket.send_bytes(sine_pcm(0.5))
        partial = socket.receive_json()
        socket.send_bytes(_silence(0.8))
        final = socket.receive_json()

    assert (partial["text"], final["text"], final["final"]) == (_HEARD, _HEARD, True)
    assert _HEARD not in caplog.text


class _CloseSendError(RuntimeError):
    """The peer is gone, so sending the close fails."""


def test_hear_failure_whose_close_also_fails_leaves_the_text_out_of_the_chain() -> None:
    app = an_app(hearing=FailingHearing())
    received = iter(
        [
            {"type": "websocket.connect"},
            {"type": "websocket.receive", "bytes": sine_pcm(0.1)},
        ]
    )

    async def receive() -> dict[str, object]:
        message = next(received, None)
        if message is not None:
            return message
        await asyncio.Event().wait()
        return {"type": "websocket.disconnect"}

    async def send(message: dict[str, object]) -> None:
        if message["type"] == "websocket.close":
            raise _CloseSendError

    scope = {
        "type": "websocket",
        "asgi": {"version": "3.0"},
        "path": "/hear",
        "raw_path": b"/hear",
        "query_string": b"language=de",
        "headers": [],
        "client": ("test", 123),
        "server": ("test", 80),
        "scheme": "ws",
        "subprotocols": [],
    }

    session = asyncio.wait_for(app(scope, receive, send), timeout=5)

    with pytest.raises(_CloseSendError) as raised:
        asyncio.run(session)

    assert _HEARD not in "".join(traceback.format_exception(raised.value))


def test_hear_closes_an_oversize_frame_with_a_reason() -> None:
    client = TestClient(an_app())
    too_big = sine_pcm(MAX_FRAME_SECONDS + 0.1)

    with client.websocket_connect("/hear?language=de") as socket:
        socket.send_bytes(too_big)
        with pytest.raises(WebSocketDisconnect) as raised:
            socket.receive_json()

    assert raised.value.code == WS_1009_MESSAGE_TOO_BIG
    assert raised.value.reason == FRAME_TOO_LARGE_REASON

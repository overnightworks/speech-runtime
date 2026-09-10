"""POST /speak streams 16-bit mono WAV that is not silence."""

import asyncio
import json
import threading
from collections.abc import Iterator

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.status import (
    HTTP_200_OK,
    HTTP_422_UNPROCESSABLE_CONTENT,
    HTTP_503_SERVICE_UNAVAILABLE,
)

from speech.chatterbox import CHATTERBOX_SAMPLE_RATE
from speech.config import CHATTERBOX_SPEAKING_MODEL, Settings
from speech.pcm import BYTES_PER_SAMPLE, is_silence, pcm_from_wav
from speech.speaking import PiperSpeaking, speaking_from_settings
from tests.conftest import SPEAK_SAMPLE_RATE, FakeSpeaking, an_app, sine_pcm


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


# TestClient's ASGITransport never delivers a mid-stream `http.disconnect`, so the
# regressions below drive the real ASGI 2.3 `/speak` boundary directly.

_ASGI_TIMEOUT_SECONDS = 5.0


class _LockSerializedVoice:
    """A synthesis fake serialised by a real lock, paced by test-controlled gates.

    Mirrors production Chatterbox's `with self._lock: yield ...` shape so a
    disconnect regression proves the voice lock is released, not a stand-in.
    The acquire itself is bounded: if a production regression ever leaves a
    prior generator holding this lock forever, a later request fails fast
    with a clear exception instead of hanging the test process.
    """

    model_name = "gated-voice"
    ready = True
    streams = True
    sample_rate = SPEAK_SAMPLE_RATE

    def __init__(self) -> None:
        self.entered_first_step = threading.Event()
        self.release_first_step = threading.Event()
        self._lock = threading.Lock()

    def load(self) -> None:
        self.ready = True

    def pcm_chunks(self, text: str, language: str) -> Iterator[bytes]:
        del text, language
        if not self._lock.acquire(timeout=_ASGI_TIMEOUT_SECONDS):
            message = "voice lock still held after the bound: synthesis never released"
            raise TimeoutError(message)
        try:
            self.entered_first_step.set()
            self.release_first_step.wait(timeout=_ASGI_TIMEOUT_SECONDS)
            pcm = sine_pcm(0.05, rate=self.sample_rate)
            mid = (len(pcm) // 2) // BYTES_PER_SAMPLE * BYTES_PER_SAMPLE
            yield pcm[:mid]
            yield pcm[mid:]
        finally:
            self._lock.release()


def _speak_scope() -> dict[str, object]:
    return {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "POST",
        "path": "/speak",
        "raw_path": b"/speak",
        "query_string": b"",
        "headers": [(b"content-type", b"application/json")],
        "client": ("test", 123),
        "server": ("test", 80),
    }


def _speak_body() -> bytes:
    return json.dumps({"text": "Hallo", "language": "de"}).encode()


class _DisconnectingReceive:
    """Delivers the request body once, then waits for the test to signal disconnect.

    `delivered` fires once this call has actually handed `http.disconnect` back to
    its caller (the response's own `listen_for_disconnect` loop), so a test can
    wait on that instead of guessing how long delivery takes.
    """

    def __init__(self) -> None:
        self._sent_body = False
        self._disconnect = asyncio.Event()
        self.delivered = asyncio.Event()

    def disconnect(self) -> None:
        self._disconnect.set()

    async def __call__(self) -> dict[str, object]:
        if not self._sent_body:
            self._sent_body = True
            return {"type": "http.request", "body": _speak_body(), "more_body": False}
        await self._disconnect.wait()
        self.delivered.set()
        return {"type": "http.disconnect"}


class _HoldingSend:
    """Records ASGI send messages, pausing after the first body chunk once armed."""

    def __init__(self, *, hold_after_first_chunk: bool = False) -> None:
        self.messages: list[dict[str, object]] = []
        self.entered_hold = asyncio.Event()
        self._holds = hold_after_first_chunk
        self._held_once = False
        self._release = asyncio.Event()

    async def __call__(self, message: dict[str, object]) -> None:
        self.messages.append(message)
        if (
            self._holds
            and not self._held_once
            and message["type"] == "http.response.body"
            and message.get("more_body")
        ):
            self._held_once = True
            self.entered_hold.set()
            await self._release.wait()


def _wav_body(messages: list[dict[str, object]]) -> bytes:
    return b"".join(
        message["body"]
        for message in messages
        if message["type"] == "http.response.body"
    )


def _status(messages: list[dict[str, object]]) -> int:
    return next(
        message["status"]
        for message in messages
        if message["type"] == "http.response.start"
    )


async def _speak_once(app: FastAPI) -> list[dict[str, object]]:
    """One full, undisturbed request through the raw ASGI boundary."""
    messages: list[dict[str, object]] = []
    sent = False
    no_disconnect = asyncio.Event()  # never set: this request is never disturbed

    async def receive() -> dict[str, object]:
        nonlocal sent
        if not sent:
            sent = True
            return {"type": "http.request", "body": _speak_body(), "more_body": False}
        await no_disconnect.wait()
        return {"type": "http.disconnect"}

    async def send(message: dict[str, object]) -> None:
        messages.append(message)

    await asyncio.wait_for(
        app(_speak_scope(), receive, send), timeout=_ASGI_TIMEOUT_SECONDS
    )
    return messages


def test_speak_streams_a_full_response_through_the_raw_asgi_boundary() -> None:
    messages = asyncio.run(_speak_once(an_app()))

    assert _status(messages) == HTTP_200_OK
    rate, pcm = pcm_from_wav(_wav_body(messages))
    assert rate == SPEAK_SAMPLE_RATE
    assert len(pcm) > 0
    assert not is_silence(pcm)


def test_speak_disconnect_between_chunks_frees_the_voice_for_the_next_request() -> None:
    """ASGI2.3 cancels the send-side task while it awaits `send()` holding a chunk."""

    async def scenario() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
        voice = _LockSerializedVoice()
        app = an_app(voice)
        receive = _DisconnectingReceive()
        send = _HoldingSend(hold_after_first_chunk=True)
        voice.release_first_step.set()  # synthesis proceeds straight to the first chunk

        call = asyncio.create_task(app(_speak_scope(), receive, send))
        await asyncio.wait_for(send.entered_hold.wait(), timeout=_ASGI_TIMEOUT_SECONDS)
        receive.disconnect()
        await asyncio.wait_for(call, timeout=_ASGI_TIMEOUT_SECONDS)

        later = await _speak_once(app)
        return send.messages, later

    disconnected, later = asyncio.run(scenario())

    assert [message["type"] for message in disconnected] == [
        "http.response.start",
        "http.response.body",
    ]
    assert _status(later) == HTTP_200_OK
    rate, pcm = pcm_from_wav(_wav_body(later))
    assert rate == SPEAK_SAMPLE_RATE
    assert len(pcm) > 0


def test_speak_disconnect_during_synthesis_frees_the_voice() -> None:
    """ASGI2.3 defers cancellation until the in-flight thread-pool `next()` returns."""

    async def scenario() -> list[dict[str, object]]:
        voice = _LockSerializedVoice()
        app = an_app(voice)
        receive = _DisconnectingReceive()
        send = _HoldingSend()

        call = asyncio.create_task(app(_speak_scope(), receive, send))
        await asyncio.to_thread(voice.entered_first_step.wait, _ASGI_TIMEOUT_SECONDS)
        receive.disconnect()
        await asyncio.wait_for(receive.delivered.wait(), timeout=_ASGI_TIMEOUT_SECONDS)
        voice.release_first_step.set()  # the in-flight step now returns normally
        await asyncio.wait_for(call, timeout=_ASGI_TIMEOUT_SECONDS)

        return await _speak_once(app)

    later = asyncio.run(scenario())

    assert _status(later) == HTTP_200_OK
    rate, pcm = pcm_from_wav(_wav_body(later))
    assert rate == SPEAK_SAMPLE_RATE
    assert len(pcm) > 0

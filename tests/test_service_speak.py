"""POST /speak streams 16-bit mono WAV that is not silence."""

import asyncio
import logging
import os
import sys
import threading
import traceback
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest
from fastapi.testclient import TestClient
from speech_runtime_provider_contract import CHATTERBOX_SAMPLE_RATE
from starlette.responses import StreamingResponse
from starlette.status import (
    HTTP_200_OK,
    HTTP_422_UNPROCESSABLE_CONTENT,
    HTTP_503_SERVICE_UNAVAILABLE,
)

from speech_runtime_service.config import CHATTERBOX_SPEAKING_MODEL, Settings
from speech_runtime_service.failures import SpeakingFailedError
from speech_runtime_service.pcm import AI_MARK, is_silence, pcm_from_wav
from speech_runtime_service.selection import VoiceSelectionStore
from speech_runtime_service.service import (
    Runtime,
    RuntimeDependencies,
)
from speech_runtime_service.speaking import PiperSpeaking, speaking_for_voice
from speech_runtime_service.voices import VoiceId, VoiceLoadOutcome
from tests.support.fakes_support import (
    SPEAK_SAMPLE_RATE,
    FakeHearing,
    FakeSpeaking,
    GatedSpeaking,
    an_app,
)
from tests.support.speak_support import (
    ASGI_TIMEOUT_SECONDS,
    info_comment_before_data,
    response_body,
    speak_body,
    speak_once,
    speak_response,
    speak_scope,
    wav_body,
)

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from fastapi import FastAPI


def test_speak_streams_wav_that_plays_as_tone() -> None:
    response = TestClient(an_app()).post(
        "/speak",
        json={"text": "Guten Morgen", "language": "de"},
    )

    assert response.status_code == HTTP_200_OK
    assert response.headers["content-type"].startswith("audio/wav")
    rate, pcm = pcm_from_wav(response.content)
    assert rate == SPEAK_SAMPLE_RATE
    assert len(pcm) > 0
    assert not is_silence(pcm)


def test_speak_wav_carries_the_ai_mark_before_its_data() -> None:
    """#293 E9 (a): a /speak WAV carries the IPTC mark in LIST/INFO/ICMT before data."""
    response = TestClient(an_app()).post(
        "/speak",
        json={"text": "Guten Morgen", "language": "de"},
    )

    assert info_comment_before_data(response.content) == AI_MARK


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


def test_piper_voice_id_constructs_the_local_piper_engine() -> None:
    engine = speaking_for_voice(
        Settings(PRESENTATOR_RUNTIME_UID=os.geteuid()), VoiceId.PIPER
    )

    assert isinstance(engine, PiperSpeaking)
    assert engine.streams is False
    assert engine.model_name == "de_DE-thorsten-medium"


def test_chatterbox_voice_id_constructs_the_streaming_voice() -> None:
    engine = speaking_for_voice(
        Settings(
            device="cuda",
            PRESENTATOR_RUNTIME_UID=os.geteuid(),
        ),
        VoiceId.CHATTERBOX,
    )

    assert engine.model_name == CHATTERBOX_SPEAKING_MODEL
    assert engine.streams is True
    assert engine.sample_rate == CHATTERBOX_SAMPLE_RATE


def test_piper_load_requires_local_onnx_and_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    loaded: list[object] = []

    class LocalVoice:
        config = SimpleNamespace(sample_rate=22_050)

    def load(path: object) -> LocalVoice:
        loaded.append(path)
        return LocalVoice()

    monkeypatch.setitem(
        sys.modules,
        "piper",
        SimpleNamespace(PiperVoice=SimpleNamespace(load=load)),
    )
    engine = PiperSpeaking("local-voice", tmp_path)

    with pytest.raises(FileNotFoundError):
        engine.load()
    (tmp_path / "local-voice.onnx").touch()
    (tmp_path / "local-voice.onnx.json").touch()

    engine.load()

    assert loaded == [tmp_path / "local-voice.onnx"]
    assert engine.ready is True


# TestClient's ASGITransport never delivers a mid-stream `http.disconnect`, so the
# regressions below drive the real ASGI 2.3 `/speak` boundary directly.


class _ResponseBoundaryError(RuntimeError):
    """A test-controlled response construction, synthesis, or send failure."""


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
            return {"type": "http.request", "body": speak_body(), "more_body": False}
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


def _status(messages: list[dict[str, object]]) -> int:
    return next(
        message["status"]
        for message in messages
        if message["type"] == "http.response.start"
    )


def test_speak_streams_a_full_response_through_the_raw_asgi_boundary() -> None:
    messages = asyncio.run(speak_once(an_app()))

    assert _status(messages) == HTTP_200_OK
    rate, pcm = pcm_from_wav(wav_body(messages))
    assert rate == SPEAK_SAMPLE_RATE
    assert len(pcm) > 0
    assert not is_silence(pcm)


def test_speak_disconnect_between_chunks_frees_the_voice_for_the_next_request() -> None:
    """ASGI2.3 cancels the send-side task while it awaits `send()` holding a chunk."""

    async def scenario() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
        voice = GatedSpeaking()
        app = an_app(voice)
        receive = _DisconnectingReceive()
        send = _HoldingSend(hold_after_first_chunk=True)
        voice.release_first_step.set()  # synthesis proceeds straight to the first chunk

        call = asyncio.create_task(app(speak_scope(), receive, send))
        await asyncio.wait_for(send.entered_hold.wait(), timeout=ASGI_TIMEOUT_SECONDS)
        receive.disconnect()
        await asyncio.wait_for(call, timeout=ASGI_TIMEOUT_SECONDS)

        later = await speak_once(app)
        return send.messages, later

    disconnected, later = asyncio.run(scenario())

    assert [message["type"] for message in disconnected] == [
        "http.response.start",
        "http.response.body",
    ]
    assert _status(later) == HTTP_200_OK
    rate, pcm = pcm_from_wav(wav_body(later))
    assert rate == SPEAK_SAMPLE_RATE
    assert len(pcm) > 0


def test_speak_disconnect_during_synthesis_frees_the_voice() -> None:
    """ASGI2.3 defers cancellation until the in-flight thread-pool `next()` returns."""

    async def scenario() -> list[dict[str, object]]:
        voice = GatedSpeaking()
        app = an_app(voice)
        receive = _DisconnectingReceive()
        send = _HoldingSend()

        call = asyncio.create_task(app(speak_scope(), receive, send))
        await asyncio.to_thread(voice.entered_first_step.wait, ASGI_TIMEOUT_SECONDS)
        receive.disconnect()
        await asyncio.wait_for(receive.delivered.wait(), timeout=ASGI_TIMEOUT_SECONDS)
        voice.release_first_step.set()  # the in-flight step now returns normally
        await asyncio.wait_for(call, timeout=ASGI_TIMEOUT_SECONDS)

        return await speak_once(app)

    later = asyncio.run(scenario())

    assert _status(later) == HTTP_200_OK
    rate, pcm = pcm_from_wav(wav_body(later))
    assert rate == SPEAK_SAMPLE_RATE
    assert len(pcm) > 0


def _runtime_for_deselection(
    tmp_path: Path, speaking: FakeSpeaking
) -> tuple[Runtime, list[int]]:
    store = VoiceSelectionStore(tmp_path / "state", owner_uid=os.geteuid())
    store.write(VoiceId.CHATTERBOX)
    closed: list[int] = []
    speaking.retain_when_inactive = False
    speaking.close = lambda: closed.append(1)
    runtime = Runtime(
        speaking,
        FakeHearing(),
        dependencies=RuntimeDependencies(
            selection=store,
            engine_factory=lambda _voice: FakeSpeaking(),
            artifact_checker=lambda _voice: True,
            default_voice=VoiceId.CHATTERBOX,
        ),
    )
    return runtime, closed


def test_speak_response_construction_failure_releases_admission_for_deselection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime, closed = _runtime_for_deselection(tmp_path, FakeSpeaking())

    def fail_response_construction(*_args: object, **_kwargs: object) -> None:
        raise _ResponseBoundaryError

    monkeypatch.setattr(StreamingResponse, "__init__", fail_response_construction)
    with pytest.raises(_ResponseBoundaryError):
        speak_response(runtime)

    outcomes: list[VoiceLoadOutcome | None] = []
    switching = threading.Thread(
        target=lambda: outcomes.append(runtime.load_voice(VoiceId.PIPER))
    )
    switching.start()
    switching.join(timeout=ASGI_TIMEOUT_SECONDS)
    assert not switching.is_alive()
    assert outcomes == [VoiceLoadOutcome.ACTIVATED]
    assert closed == [1]


@pytest.mark.parametrize(
    "close_path",
    ["normal", "pre-body", "send-failure", "between-chunks", "synthesis-error"],
)
def test_speak_close_paths_release_admission_for_deselection(
    tmp_path: Path, close_path: str
) -> None:
    class ClosingPathVoice(FakeSpeaking):
        def pcm_chunks(self, text: str, language: str) -> Iterator[bytes]:
            if close_path == "synthesis-error":
                raise _ResponseBoundaryError
            yield from super().pcm_chunks(text, language)

    runtime, closed = _runtime_for_deselection(tmp_path, ClosingPathVoice())
    response = speak_response(runtime)

    if close_path == "normal":
        response_body(response)
    elif close_path in {"send-failure", "synthesis-error"}:
        delivery = (
            _interrupt(response, close_path)
            if close_path == "send-failure"
            else _deliver(response)
        )
        expected = (
            SpeakingFailedError
            if close_path == "synthesis-error"
            else _ResponseBoundaryError
        )
        with pytest.raises(expected):
            asyncio.run(delivery)
    else:
        asyncio.run(_interrupt(response, close_path))

    outcomes: list[VoiceLoadOutcome | None] = []
    switching = threading.Thread(
        target=lambda: outcomes.append(runtime.load_voice(VoiceId.PIPER))
    )
    switching.start()
    switching.join(timeout=ASGI_TIMEOUT_SECONDS)
    assert not switching.is_alive()
    assert outcomes == [VoiceLoadOutcome.ACTIVATED]
    assert closed == [1]


async def _interrupt(response: StreamingResponse, close_path: str) -> None:
    disconnect = asyncio.Event()

    async def receive() -> dict[str, object]:
        if close_path == "pre-body":
            return {"type": "http.disconnect"}
        await disconnect.wait()
        return {"type": "http.disconnect"}

    async def send(message: dict[str, object]) -> None:
        if close_path == "send-failure":
            raise _ResponseBoundaryError
        if close_path == "between-chunks" and message["type"] == "http.response.body":
            disconnect.set()

    await response(
        {"type": "http", "asgi": {"version": "3.0", "spec_version": "2.3"}},
        receive,
        send,
    )


async def _deliver(response: StreamingResponse) -> None:
    never = asyncio.Event()

    async def receive() -> dict[str, object]:
        await never.wait()
        return {"type": "http.disconnect"}

    async def send(_message: dict[str, object]) -> None:
        return None

    await response(
        {"type": "http", "asgi": {"version": "3.0", "spec_version": "2.3"}},
        receive,
        send,
    )


_SPOKEN = "Nur dieser Satz darf nicht in den Logs stehen"


class _SentenceCarryingError(RuntimeError):
    """A third-party failure that holds the input in its message."""


def _error_holding_the_sentence() -> _SentenceCarryingError:
    cause = KeyError(_SPOKEN)
    error = _SentenceCarryingError(_SPOKEN)
    error.__cause__ = cause
    return error


class _FailingOnceSpeaking(FakeSpeaking):
    """Fails with the sentence in its error, optionally after one chunk."""

    def __init__(self, *, after_first_chunk: bool = False) -> None:
        super().__init__()
        self.fails = True
        self._after_first_chunk = after_first_chunk

    def pcm_chunks(self, text: str, language: str) -> Iterator[bytes]:
        if not self.fails:
            yield from super().pcm_chunks(text, language)
            return
        if self._after_first_chunk:
            yield next(iter(super().pcm_chunks(text, language)))
        raise _error_holding_the_sentence()


def _assert_failure_logged_without_text(
    caplog: pytest.LogCaptureFixture, raised: BaseException
) -> None:
    assert "speaking failed" in caplog.text
    assert "_SentenceCarryingError" in caplog.text
    assert "KeyError" in caplog.text
    assert _SPOKEN not in caplog.text
    assert _SPOKEN not in "".join(traceback.format_exception(raised))


def _speak_collecting(
    app: FastAPI,
) -> tuple[list[dict[str, object]], BaseException | None]:
    async def scenario() -> tuple[list[dict[str, object]], BaseException | None]:
        messages: list[dict[str, object]] = []
        sent = False
        never = asyncio.Event()

        async def receive() -> dict[str, object]:
            nonlocal sent
            if not sent:
                sent = True
                return {
                    "type": "http.request",
                    "body": speak_body(),
                    "more_body": False,
                }
            await never.wait()
            return {"type": "http.disconnect"}

        async def send(message: dict[str, object]) -> None:
            messages.append(message)

        try:
            await asyncio.wait_for(
                app(speak_scope(), receive, send), timeout=ASGI_TIMEOUT_SECONDS
            )
        except SpeakingFailedError as error:
            return messages, error
        return messages, None

    return asyncio.run(scenario())


def test_speak_writes_no_sentence_to_the_log_whatever_the_environment_says(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SPEECH_DEBUG", "true")
    caplog.set_level(logging.DEBUG)

    response = TestClient(an_app()).post(
        "/speak", json={"text": _SPOKEN, "language": "de"}
    )

    assert response.status_code == HTTP_200_OK
    assert _SPOKEN not in caplog.text


def test_speak_failure_before_the_first_chunk_logs_types_and_never_the_sentence(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG)

    _messages, raised = _speak_collecting(an_app(_FailingOnceSpeaking()))

    assert isinstance(raised, SpeakingFailedError)
    _assert_failure_logged_without_text(caplog, raised)


def test_speak_failure_after_a_chunk_never_ends_the_body_and_the_next_request_is_served(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG)
    speaking = _FailingOnceSpeaking(after_first_chunk=True)
    app = an_app(speaking)

    messages, raised = _speak_collecting(app)
    speaking.fails = False
    later = asyncio.run(speak_once(app))

    assert isinstance(raised, SpeakingFailedError)
    _assert_failure_logged_without_text(caplog, raised)
    bodies = [m for m in messages if m["type"] == "http.response.body"]
    assert bodies
    assert all(body.get("more_body") is True for body in bodies)
    assert _status(later) == HTTP_200_OK


def test_speak_cleanup_failure_after_a_disconnect_logs_no_sentence_and_frees_the_voice(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG)

    class CleanupFails(FakeSpeaking):
        failing_cleanup = True

        def pcm_chunks(self, text: str, language: str) -> Iterator[bytes]:
            try:
                yield from super().pcm_chunks(text, language)
            finally:
                if self.failing_cleanup:
                    raise _error_holding_the_sentence()

    speaking = CleanupFails()
    app = an_app(speaking)

    async def scenario() -> tuple[BaseException | None, list[dict[str, object]]]:
        receive = _DisconnectingReceive()
        send = _HoldingSend(hold_after_first_chunk=True)
        call = asyncio.create_task(app(speak_scope(), receive, send))
        await asyncio.wait_for(send.entered_hold.wait(), timeout=ASGI_TIMEOUT_SECONDS)
        receive.disconnect()
        failure: BaseException | None = None
        try:
            await asyncio.wait_for(call, timeout=ASGI_TIMEOUT_SECONDS)
        except SpeakingFailedError as error:
            failure = error
        speaking.failing_cleanup = False
        return failure, await speak_once(app)

    failure, later = asyncio.run(scenario())

    assert _SPOKEN not in caplog.text
    if failure is not None:
        assert _SPOKEN not in "".join(traceback.format_exception(failure))
    assert _status(later) == HTTP_200_OK

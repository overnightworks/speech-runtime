"""POST /v1/audio/speech: the refusals, the bounded wait, and the streamed WAV."""

import asyncio
import json
import logging
import traceback
from typing import TYPE_CHECKING

import pytest
from fastapi.testclient import TestClient
from starlette.responses import StreamingResponse
from starlette.status import (
    HTTP_200_OK,
    HTTP_400_BAD_REQUEST,
    HTTP_404_NOT_FOUND,
    HTTP_422_UNPROCESSABLE_CONTENT,
)

from speech_runtime_client.wire import (
    SPEECH_PATH,
    RefusalCode,
)
from speech_runtime_service.config import Settings
from speech_runtime_service.failures import SpeakingFailedError
from speech_runtime_service.pcm import AI_MARK, is_silence, pcm_from_wav
from speech_runtime_service.voices import VoiceId
from tests.support.fakes_support import (
    SPEAK_SAMPLE_RATE,
    FakeSpeaking,
    GatedSpeaking,
)
from tests.support.speak_support import (
    ASGI_TIMEOUT_SECONDS,
    app_over,
    assert_refusal,
    assert_unavailable,
    fake_runtime,
    gate_is_free,
    info_comment_before_data,
    post_speech,
    response_body,
    speak_app,
    speak_body,
    speak_once,
    speak_response,
    speak_scope,
    status_of,
    wav_body,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

    from fastapi import FastAPI

    from speech_runtime_service.service import Runtime

SHORT_WAIT_SECONDS = 0.2
SLACK_SECONDS = 0.5
CHATTERBOX = VoiceId.CHATTERBOX.value
ALIEN_SAMPLE_RATE = 24_000


def _valid() -> dict[str, object]:
    return json.loads(speak_body())


def _changed(**changes: object) -> bytes:
    return json.dumps(_valid() | changes).encode()


def _without(name: str) -> bytes:
    return json.dumps({k: v for k, v in _valid().items() if k != name}).encode()


def test_the_wire_streams_wav_that_plays_as_tone_and_speaks_german() -> None:
    speaking = FakeSpeaking()
    response = post_speech(
        TestClient(speak_app(speaking)), speak_body(text="Guten Morgen")
    )

    assert response.status_code == HTTP_200_OK
    assert response.headers["content-type"] == "audio/wav"
    rate, pcm = pcm_from_wav(response.content)
    assert rate == SPEAK_SAMPLE_RATE
    assert not is_silence(pcm)
    assert speaking.heard_text == "Guten Morgen"
    assert speaking.heard_language == "de"


def test_the_wav_carries_the_ai_mark_before_its_data() -> None:
    response = post_speech(TestClient(speak_app()), speak_body())

    assert info_comment_before_data(response.content) == AI_MARK


def test_the_wav_header_uses_the_voice_sample_rate() -> None:
    speaking = FakeSpeaking()
    speaking.sample_rate = ALIEN_SAMPLE_RATE

    response = post_speech(TestClient(speak_app(speaking)), speak_body())

    assert pcm_from_wav(response.content)[0] == ALIEN_SAMPLE_RATE


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("POST", "/v2/audio/speech"),
        ("GET", "/v2"),
        ("POST", "/v2/"),
        ("GET", "/v7/x/y"),
        ("POST", "/v10/audio/speech"),
    ],
)
def test_another_version_is_refused_as_unsupported(method: str, path: str) -> None:
    response = TestClient(speak_app()).request(method, path)

    assert_refusal(
        response, HTTP_400_BAD_REQUEST, RefusalCode.UNSUPPORTED_VERSION, None
    )


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("GET", "/v1"),
        ("GET", "/v1/"),
        ("POST", "/v1/audio/speech/"),
        ("POST", "/v01/audio/speech"),
        ("POST", "/v1/audio/transcriptions"),
        ("GET", "/v1/voices"),
        ("GET", SPEECH_PATH),
        ("PUT", SPEECH_PATH),
        ("PATCH", SPEECH_PATH),
        ("DELETE", SPEECH_PATH),
        ("OPTIONS", SPEECH_PATH),
        ("PROPFIND", SPEECH_PATH),
    ],
)
def test_any_other_path_or_method_under_version_one_is_not_found(
    method: str, path: str
) -> None:
    response = TestClient(speak_app()).request(method, path, follow_redirects=False)

    assert_refusal(response, HTTP_404_NOT_FOUND, RefusalCode.NOT_FOUND, None)


def test_head_on_the_speech_path_answers_the_status_and_type_without_a_body() -> None:
    response = TestClient(speak_app()).head(SPEECH_PATH)

    assert response.status_code == HTTP_404_NOT_FOUND
    assert response.headers["content-type"] == "application/json"
    assert response.content == b""


@pytest.mark.parametrize("path", ["/v-1/audio/speech", "/vx/audio/speech", "/speak"])
def test_a_path_that_is_no_version_path_gets_the_apps_default_not_found(
    path: str,
) -> None:
    response = TestClient(speak_app()).post(path)

    assert response.status_code == HTTP_404_NOT_FOUND
    assert response.json() == {"detail": "Not Found"}


def test_health_answers_beside_the_wire_and_voice_administration_is_not_mounted() -> (
    None
):
    client = TestClient(speak_app())

    assert client.get("/health").status_code == HTTP_200_OK
    assert client.get("/voices").status_code == HTTP_404_NOT_FOUND


_REFUSED_BODIES = [
    pytest.param(b"hallo", None, id="not-json"),
    pytest.param(b"\xff\xfe", None, id="not-utf8"),
    pytest.param(b"[]", None, id="array"),
    pytest.param(b'"x"', None, id="string"),
    pytest.param(b"{}", "model", id="empty-object"),
    pytest.param(_without("model"), "model", id="model-missing"),
    pytest.param(_changed(model="tts-1"), "model", id="model-other"),
    pytest.param(_changed(model=1), "model", id="model-number"),
    pytest.param(_without("voice"), "voice", id="voice-missing"),
    pytest.param(_changed(voice=""), "voice", id="voice-empty"),
    pytest.param(_changed(voice="alloy"), "voice", id="voice-outside-the-catalogue"),
    pytest.param(_changed(voice=5), "voice", id="voice-number"),
    pytest.param(_without("input"), "input", id="input-missing"),
    pytest.param(_changed(input=""), "input", id="input-empty"),
    pytest.param(_changed(input=" ... "), "input", id="input-without-content"),
    pytest.param(_changed(input=42), "input", id="input-number"),
    pytest.param(_without("response_format"), "response_format", id="format-missing"),
    pytest.param(_changed(response_format="pcm"), "response_format", id="format-pcm"),
    pytest.param(_changed(response_format="mp3"), "response_format", id="format-mp3"),
    pytest.param(_changed(response_format=1), "response_format", id="format-number"),
    pytest.param(_changed(language="de"), None, id="language-is-unknown"),
    pytest.param(_changed(speed=1.0), None, id="speed-is-unknown"),
]


@pytest.mark.parametrize(("body", "param"), _REFUSED_BODIES)
def test_an_invalid_body_is_refused_naming_only_a_field_of_the_request_model(
    body: bytes, param: str | None
) -> None:
    response = post_speech(TestClient(speak_app()), body)

    assert_refusal(
        response, HTTP_422_UNPROCESSABLE_CONTENT, RefusalCode.INVALID_REQUEST, param
    )


def _assert_input_limit(
    settings: Settings, lengths_served: list[int], length_refused: int
) -> None:
    client = TestClient(speak_app(settings=settings))

    served = [
        post_speech(client, speak_body(text="a" * length)).status_code
        for length in lengths_served
    ]
    refused = post_speech(client, speak_body(text="a" * length_refused))

    assert served == [HTTP_200_OK] * len(lengths_served)
    assert_refusal(
        refused, HTTP_422_UNPROCESSABLE_CONTENT, RefusalCode.INVALID_REQUEST, "input"
    )


@pytest.mark.parametrize(
    ("limit", "lengths_served", "length_refused"),
    [(10, [1, 10], 11), (5000, [4097, 5000], 5001)],
)
def test_the_input_limit_is_configurable_and_may_exceed_openais(
    limit: int, lengths_served: list[int], length_refused: int
) -> None:
    _assert_input_limit(
        Settings(max_input_characters=limit), lengths_served, length_refused
    )


def test_the_default_input_limit_is_the_settings_value() -> None:
    settings = Settings()
    limit = settings.max_input_characters

    _assert_input_limit(settings, [limit], limit + 1)


_SENTENCE = "Dieser Satz darf in keiner Antwort und keinem Log stehen"
_ECHOING_BODIES = [
    pytest.param(
        _changed(input=_SENTENCE * 100), HTTP_422_UNPROCESSABLE_CONTENT, id="too-long"
    ),
    pytest.param(
        _changed(voice=_SENTENCE), HTTP_422_UNPROCESSABLE_CONTENT, id="wrong-voice"
    ),
    pytest.param(
        _changed(**{_SENTENCE: _SENTENCE}),
        HTTP_422_UNPROCESSABLE_CONTENT,
        id="sentence-as-unknown-key",
    ),
    pytest.param(
        _SENTENCE.encode(), HTTP_422_UNPROCESSABLE_CONTENT, id="malformed-json"
    ),
]


@pytest.mark.parametrize(("body", "status"), _ECHOING_BODIES)
def test_a_refusal_and_the_log_never_hold_the_input_or_a_key_the_request_chose(
    caplog: pytest.LogCaptureFixture, body: bytes, status: int
) -> None:
    caplog.set_level(logging.DEBUG)
    client = TestClient(speak_app(settings=Settings(max_input_characters=1000)))

    response = post_speech(client, body)

    assert response.status_code == status
    assert _SENTENCE not in response.text
    assert _SENTENCE not in caplog.text


def test_a_voice_other_than_the_active_one_is_unavailable() -> None:
    response = post_speech(TestClient(speak_app()), speak_body(CHATTERBOX))

    assert_unavailable(response)


def test_a_runtime_that_is_still_loading_is_unavailable() -> None:
    runtime = fake_runtime()
    runtime.begin_loading()

    assert_unavailable(post_speech(TestClient(app_over(runtime)), speak_body()))


def test_a_runtime_that_is_shutting_down_is_unavailable() -> None:
    runtime = fake_runtime()
    runtime.begin_shutdown()

    assert_unavailable(post_speech(TestClient(app_over(runtime)), speak_body()))


class _CountingClose(FakeSpeaking):
    def __init__(self) -> None:
        super().__init__()
        self.closed = 0

    def close(self) -> None:
        self.closed += 1


def test_a_failed_voice_is_unavailable_and_its_engine_is_closed_once_afterwards() -> (
    None
):
    speaking = _CountingClose()
    runtime = fake_runtime(speaking)
    client = TestClient(app_over(runtime))
    speaking.ready = False

    first = post_speech(client, speak_body())
    second = post_speech(client, speak_body())

    assert_unavailable(first)
    assert_unavailable(second)
    assert speaking.closed == 1


# TestClient's ASGITransport never delivers a mid-stream `http.disconnect`, so the
# regressions below drive the real ASGI 2.3 wire boundary directly.


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


def test_speak_streams_a_full_response_through_the_raw_asgi_boundary() -> None:
    messages = asyncio.run(speak_once(speak_app()))

    assert status_of(messages) == HTTP_200_OK
    rate, pcm = pcm_from_wav(wav_body(messages))
    assert rate == SPEAK_SAMPLE_RATE
    assert len(pcm) > 0
    assert not is_silence(pcm)


def test_speak_disconnect_between_chunks_frees_the_voice_for_the_next_request() -> None:
    """ASGI2.3 cancels the send-side task while it awaits `send()` holding a chunk."""

    async def scenario() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
        voice = GatedSpeaking()
        app = speak_app(voice)
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
    assert status_of(later) == HTTP_200_OK
    rate, pcm = pcm_from_wav(wav_body(later))
    assert rate == SPEAK_SAMPLE_RATE
    assert len(pcm) > 0


def test_speak_disconnect_during_synthesis_frees_the_voice() -> None:
    """ASGI2.3 defers cancellation until the in-flight thread-pool `next()` returns."""

    async def scenario() -> list[dict[str, object]]:
        voice = GatedSpeaking()
        app = speak_app(voice)
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

    assert status_of(later) == HTTP_200_OK
    rate, pcm = pcm_from_wav(wav_body(later))
    assert rate == SPEAK_SAMPLE_RATE
    assert len(pcm) > 0


def _runtime_closing_on_shutdown(
    speaking: FakeSpeaking,
) -> tuple[Runtime, list[int]]:
    closed: list[int] = []
    speaking.close = lambda: closed.append(1)
    return fake_runtime(speaking, VoiceId.CHATTERBOX), closed


def _assert_gate_free_and_shutdown_closes_once(
    runtime: Runtime, closed: list[int]
) -> None:
    assert gate_is_free(runtime)
    runtime.close()
    assert closed == [1]


def test_speak_response_construction_failure_frees_the_gate_for_shutdown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, closed = _runtime_closing_on_shutdown(FakeSpeaking())

    def fail_response_construction(*_args: object, **_kwargs: object) -> None:
        raise _ResponseBoundaryError

    monkeypatch.setattr(StreamingResponse, "__init__", fail_response_construction)
    with pytest.raises(_ResponseBoundaryError):
        speak_response(runtime, CHATTERBOX)

    _assert_gate_free_and_shutdown_closes_once(runtime, closed)


@pytest.mark.parametrize(
    "close_path",
    ["normal", "pre-body", "send-failure", "between-chunks", "synthesis-error"],
)
def test_speak_close_paths_free_the_gate_for_shutdown(close_path: str) -> None:
    class ClosingPathVoice(FakeSpeaking):
        def pcm_chunks(self, text: str, language: str) -> Iterator[bytes]:
            if close_path == "synthesis-error":
                raise _ResponseBoundaryError
            yield from super().pcm_chunks(text, language)

    runtime, closed = _runtime_closing_on_shutdown(ClosingPathVoice())
    response = speak_response(runtime, CHATTERBOX)

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

    _assert_gate_free_and_shutdown_closes_once(runtime, closed)


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
    assert 'File "' in caplog.text
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

    response = post_speech(TestClient(speak_app()), speak_body(text=_SPOKEN))

    assert response.status_code == HTTP_200_OK
    assert _SPOKEN not in caplog.text


def test_speak_failure_before_the_first_chunk_logs_types_and_never_the_sentence(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG)

    _messages, raised = _speak_collecting(speak_app(_FailingOnceSpeaking()))

    assert isinstance(raised, SpeakingFailedError)
    _assert_failure_logged_without_text(caplog, raised)


def test_speak_failure_after_a_chunk_never_ends_the_body_and_the_next_request_is_served(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG)
    speaking = _FailingOnceSpeaking(after_first_chunk=True)
    app = speak_app(speaking)

    messages, raised = _speak_collecting(app)
    speaking.fails = False
    later = asyncio.run(speak_once(app))

    assert isinstance(raised, SpeakingFailedError)
    _assert_failure_logged_without_text(caplog, raised)
    bodies = [m for m in messages if m["type"] == "http.response.body"]
    assert bodies
    assert all(body.get("more_body") is True for body in bodies)
    assert status_of(later) == HTTP_200_OK


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
    app = speak_app(speaking)

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

    assert isinstance(failure, SpeakingFailedError)
    _assert_failure_logged_without_text(caplog, failure)
    assert status_of(later) == HTTP_200_OK

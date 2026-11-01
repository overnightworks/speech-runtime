"""POST /v1/audio/transcriptions: who may hear, what is refused, the bounded decode."""

import asyncio
import logging
import time
from typing import TYPE_CHECKING

import pytest
from fastapi.testclient import TestClient
from starlette.status import (
    HTTP_200_OK,
    HTTP_400_BAD_REQUEST,
    HTTP_401_UNAUTHORIZED,
    HTTP_403_FORBIDDEN,
    HTTP_413_CONTENT_TOO_LARGE,
    HTTP_422_UNPROCESSABLE_CONTENT,
    HTTP_429_TOO_MANY_REQUESTS,
    HTTP_503_SERVICE_UNAVAILABLE,
)

from speech_runtime_client.wire import (
    AUTHENTICATE_HEADER,
    AUTHORIZATION_HEADER,
    MODEL,
    RETRY_AFTER_HEADER,
    TRANSCRIPTIONS_PATH,
    WORKLOAD_TOKEN_HEADER,
    HearingLanguage,
    RefusalCode,
)
from speech_runtime_service.config import Settings
from speech_runtime_service.control import create_control_app
from speech_runtime_service.voices import VoiceId
from tests.support.hear_support import (
    HEARD_TEXT,
    MULTIPART,
    WAIT_SECONDS,
    FakeHearing,
    hear_app,
    hear_call,
    hear_runtime,
    hearing_gate_is_free,
    multipart_body,
    post_recording,
    recording,
)
from tests.support.identity_support import (
    APP,
    altered,
    valid_callers,
    valid_headers,
    valid_pair,
)
from tests.support.speak_support import (
    assert_call_refusal,
    assert_refusal,
    gate_is_free,
    post_speech,
    speak_body,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from speech_runtime_service.admission_gate import Lease
    from speech_runtime_service.service import Runtime

SHORT_WAIT_SECONDS = 0.2
SLACK_SECONDS = 0.5
ENGLISH = {"model": MODEL, "language": HearingLanguage.EN.value}


def _client(engine: FakeHearing | None = None, **settings: float) -> TestClient:
    return TestClient(hear_app(hear_runtime(engine or FakeHearing()), **settings))


def _valid() -> bytes:
    return multipart_body(file=recording())


def test_a_recording_is_answered_with_its_text() -> None:
    response = post_recording(_client(), _valid())

    assert response.status_code == HTTP_200_OK
    assert response.json() == {"text": HEARD_TEXT}


def test_a_recording_without_a_language_is_heard_as_german() -> None:
    engine = FakeHearing()

    post_recording(_client(engine), _valid())

    assert [language for _, language in engine.heard] == [HearingLanguage.DE]


def test_a_recording_is_heard_in_the_language_asked_for() -> None:
    engine = FakeHearing()

    post_recording(_client(engine), multipart_body(ENGLISH, recording()))

    assert [language for _, language in engine.heard] == [HearingLanguage.EN]


@pytest.mark.parametrize(
    ("fields", "param"),
    [
        ({"model": MODEL, "language": "fr"}, "language"),
        ({"model": MODEL, "response_format": "text"}, "response_format"),
        ({"model": MODEL, "temperature": "0"}, None),
        ({"model": "whisper-1"}, "model"),
        ({}, "model"),
    ],
    ids=["other-language", "other-format", "unknown-field", "other-model", "no-model"],
)
def test_a_form_that_is_not_the_one_shape_is_refused_without_a_decode(
    fields: dict[str, str], param: str | None
) -> None:
    engine = FakeHearing()

    response = post_recording(_client(engine), multipart_body(fields, recording()))

    assert_refusal(
        response, HTTP_422_UNPROCESSABLE_CONTENT, RefusalCode.INVALID_REQUEST, param
    )
    assert engine.heard == []


@pytest.mark.parametrize(
    ("body", "param"),
    [
        (multipart_body(), "file"),
        (b"not a multipart body", "file"),
        (b"", "file"),
    ],
    ids=["no-file", "malformed", "empty"],
)
def test_a_request_without_a_recording_is_refused_as_the_file(
    body: bytes, param: str
) -> None:
    response = post_recording(_client(), body)

    assert_refusal(
        response, HTTP_422_UNPROCESSABLE_CONTENT, RefusalCode.INVALID_REQUEST, param
    )


def test_a_recording_that_is_not_audio_is_refused_as_the_file() -> None:
    engine = FakeHearing()

    response = post_recording(_client(engine), multipart_body(file=b"not sound" * 99))

    assert_refusal(
        response, HTTP_422_UNPROCESSABLE_CONTENT, RefusalCode.INVALID_REQUEST, "file"
    )
    assert engine.heard == []


def test_a_recording_longer_than_the_length_cap_is_too_large_and_never_heard() -> None:
    engine = FakeHearing()
    client = _client(engine, max_recording_seconds=0.25)

    response = post_recording(client, multipart_body(file=recording(seconds=0.5)))

    assert_refusal(response, HTTP_413_CONTENT_TOO_LARGE, RefusalCode.TOO_LARGE, "file")
    assert engine.heard == []


@pytest.mark.parametrize(
    ("engine", "settings"),
    [
        pytest.param(FakeHearing(), {}, id="plain"),
        pytest.param(None, {}, id="and-hearing-unavailable"),
        pytest.param(FakeHearing(), {"max_recording_bytes": 1}, id="and-too-large"),
    ],
)
def test_an_app_without_the_hear_permission_is_refused_as_not_enabled_first(
    engine: FakeHearing | None, settings: dict[str, int]
) -> None:
    runtime = hear_runtime(engine)
    callers = valid_callers()
    client = TestClient(create_control_app(Settings(**settings), runtime, callers))

    response = post_recording(client, _valid())

    assert_refusal(response, HTTP_403_FORBIDDEN, RefusalCode.NOT_ENABLED, None)


def _hold_the_decode(runtime: Runtime) -> None:
    assert runtime.hearing.gate.try_acquire() is not None


@pytest.mark.parametrize(
    ("body", "engine", "settings", "arrange"),
    [
        pytest.param(b"not a multipart body", FakeHearing(), {}, None, id="malformed"),
        pytest.param(_valid(), None, {}, None, id="hearing-unavailable"),
        pytest.param(
            _valid(),
            FakeHearing(),
            {"hearing_max_waiting_requests": 0},
            _hold_the_decode,
            id="capacity-full",
        ),
        pytest.param(
            _valid(), FakeHearing(), {"max_recording_bytes": 1}, None, id="too-large"
        ),
    ],
)
def test_a_caller_without_tokens_is_told_so_before_anything_else(
    body: bytes,
    engine: FakeHearing | None,
    settings: dict[str, float],
    arrange: Callable[[Runtime], None] | None,
) -> None:
    runtime = hear_runtime(engine)
    if arrange is not None:
        arrange(runtime)
    client = TestClient(hear_app(runtime, **settings))

    response = post_recording(client, body, headers={})

    assert_refusal(response, HTTP_401_UNAUTHORIZED, RefusalCode.UNAUTHORIZED, None)
    assert response.headers[AUTHENTICATE_HEADER] == "Bearer"


def test_a_person_outside_the_service_group_is_forbidden() -> None:
    pair = valid_pair()
    headers = {
        WORKLOAD_TOKEN_HEADER: pair.workload,
        AUTHORIZATION_HEADER: "Bearer " + altered(pair.person, groups=[f"/apps/{APP}"]),
    }

    response = post_recording(_client(), _valid(), headers)

    assert_refusal(response, HTTP_403_FORBIDDEN, RefusalCode.FORBIDDEN, None)


@pytest.mark.parametrize("chunked", [False, True], ids=["declared", "chunked"])
def test_a_body_over_the_byte_cap_is_too_large_before_it_is_parsed(
    *, chunked: bool
) -> None:
    engine = FakeHearing()
    unparseable = b"x" * 4096
    content: bytes | Iterator[bytes] = (
        iter([unparseable[:2048], unparseable[2048:]]) if chunked else unparseable
    )

    response = _client(engine, max_recording_bytes=1024).post(
        TRANSCRIPTIONS_PATH,
        content=content,
        headers={"content-type": MULTIPART, **valid_headers()},
    )

    assert_refusal(response, HTTP_413_CONTENT_TOO_LARGE, RefusalCode.TOO_LARGE, None)
    assert engine.heard == []


def test_the_route_answers_the_wrong_version_before_anything_else() -> None:
    response = _client().post("/v2/audio/transcriptions", content=_valid())

    assert_refusal(
        response, HTTP_400_BAD_REQUEST, RefusalCode.UNSUPPORTED_VERSION, None
    )


def test_without_a_hearing_model_hearing_is_unavailable_and_speaking_works() -> None:
    runtime = hear_runtime(None)
    client = TestClient(hear_app(runtime))

    transcription = post_recording(client, _valid())
    speech = post_speech(client, speak_body(VoiceId.PIPER.value))

    assert_refusal(
        transcription, HTTP_503_SERVICE_UNAVAILABLE, RefusalCode.UNAVAILABLE, None
    )
    assert speech.status_code == HTTP_200_OK


def test_a_request_that_finds_the_decode_taken_waits_then_is_told_to_retry_after() -> (
    None
):
    runtime = hear_runtime(FakeHearing())
    client = TestClient(hear_app(runtime, hearing_busy_wait_seconds=SHORT_WAIT_SECONDS))
    _hold_the_decode(runtime)

    response = post_recording(client, _valid())

    assert_refusal(response, HTTP_429_TOO_MANY_REQUESTS, RefusalCode.BUSY, None)
    assert int(response.headers[RETRY_AFTER_HEADER]) >= 1


def test_a_request_beyond_the_waiting_places_is_told_busy_at_once() -> None:
    async def scenario() -> tuple[FakeHearing, object, float]:
        engine = FakeHearing(hold=True)
        app = hear_app(
            hear_runtime(engine),
            hearing_busy_wait_seconds=WAIT_SECONDS,
            hearing_max_waiting_requests=0,
        )
        running = hear_call(app)
        await asyncio.to_thread(engine.entered.wait, WAIT_SECONDS)
        beyond = hear_call(app)
        await beyond.status()
        engine.release.set()
        await asyncio.gather(running.finish(), beyond.finish())
        return engine, beyond, beyond.started_after

    engine, beyond, answered_after = asyncio.run(scenario())

    assert_call_refusal(beyond, HTTP_429_TOO_MANY_REQUESTS, RefusalCode.BUSY, None)
    assert answered_after < SLACK_SECONDS
    assert engine.most_concurrent == 1


def test_speaking_and_hearing_run_side_by_side_each_under_its_own_bound() -> None:
    async def scenario() -> tuple[int, int]:
        engine = FakeHearing(hold=True)
        runtime = hear_runtime(engine)
        app = hear_app(runtime, hearing_max_waiting_requests=0, max_waiting_requests=0)
        hearing = hear_call(app)
        await asyncio.to_thread(engine.entered.wait, WAIT_SECONDS)
        speaking = await asyncio.to_thread(
            post_speech, TestClient(app), speak_body(VoiceId.PIPER.value)
        )
        engine.release.set()
        await hearing.finish()
        return speaking.status_code, hearing.status_code

    speaking_status, hearing_status = asyncio.run(scenario())

    assert (speaking_status, hearing_status) == (HTTP_200_OK, HTTP_200_OK)


def test_a_held_synthesis_does_not_refuse_a_recording() -> None:
    runtime = hear_runtime(FakeHearing())
    held = runtime.synthesis_gate.try_acquire()
    assert held is not None
    client = TestClient(hear_app(runtime, max_waiting_requests=0))

    response = post_recording(client, _valid())

    assert response.status_code == HTTP_200_OK


def test_a_held_decode_does_not_refuse_speaking() -> None:
    runtime = hear_runtime(FakeHearing())
    _hold_the_decode(runtime)
    client = TestClient(hear_app(runtime, hearing_max_waiting_requests=0))

    response = post_speech(client, speak_body(VoiceId.PIPER.value))

    assert response.status_code == HTTP_200_OK


@pytest.mark.parametrize(
    "arrange",
    [
        pytest.param(
            lambda engine: setattr(engine, "fail_transcription", True),
            id="failed-decode",
        ),
        pytest.param(lambda _engine: None, id="served"),
    ],
)
def test_a_finished_request_leaves_the_decode_free(
    arrange: Callable[[FakeHearing], None],
) -> None:
    engine = FakeHearing()
    arrange(engine)
    runtime = hear_runtime(engine)

    post_recording(TestClient(hear_app(runtime)), _valid())

    assert hearing_gate_is_free(runtime)
    assert gate_is_free(runtime)


def test_a_failed_decode_is_unavailable() -> None:
    engine = FakeHearing()
    engine.fail_transcription = True

    response = post_recording(_client(engine), _valid())

    assert_refusal(
        response, HTTP_503_SERVICE_UNAVAILABLE, RefusalCode.UNAVAILABLE, None
    )


def test_the_decode_is_held_until_the_engine_returns_though_the_caller_left() -> None:
    async def scenario() -> tuple[bool, Lease | None]:
        engine = FakeHearing(hold=True)
        runtime = hear_runtime(engine)
        call = hear_call(hear_app(runtime))
        await asyncio.to_thread(engine.entered.wait, WAIT_SECONDS)
        call.task.cancel()
        await asyncio.gather(call.task, return_exceptions=True)
        held_after_leaving = runtime.hearing.gate.try_acquire() is None
        engine.release.set()
        deadline = time.monotonic() + WAIT_SECONDS
        return held_after_leaving, await runtime.hearing.gate.acquire(deadline, 1)

    held_after_leaving, served_after_return = asyncio.run(scenario())

    assert held_after_leaving
    assert served_after_return is not None


@pytest.mark.parametrize("fails", [False, True], ids=["heard", "failed"])
def test_no_log_record_holds_the_heard_text(
    caplog: pytest.LogCaptureFixture, *, fails: bool
) -> None:
    engine = FakeHearing()
    engine.fail_transcription = fails
    caplog.set_level(logging.DEBUG)

    post_recording(_client(engine), _valid())

    assert HEARD_TEXT not in caplog.text

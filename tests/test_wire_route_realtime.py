"""WS /v1/realtime?intent=transcription: who may upgrade, what is refused, a session."""

import pytest
from fastapi.testclient import TestClient
from starlette.status import (
    HTTP_400_BAD_REQUEST,
    HTTP_401_UNAUTHORIZED,
    HTTP_403_FORBIDDEN,
    HTTP_404_NOT_FOUND,
    HTTP_422_UNPROCESSABLE_CONTENT,
    HTTP_429_TOO_MANY_REQUESTS,
    HTTP_503_SERVICE_UNAVAILABLE,
)
from starlette.testclient import WebSocketDenialResponse
from starlette.websockets import WebSocketDisconnect

from speech_runtime_client.wire import (
    AUTHENTICATE_HEADER,
    AUTHORIZATION_HEADER,
    REALTIME_PATH,
    RETRY_AFTER_HEADER,
    WORKLOAD_TOKEN_HEADER,
    RefusalCode,
)
from speech_runtime_service.config import Settings
from speech_runtime_service.control import create_control_app
from tests.support.hear_support import (
    HEARD_TEXT,
    FakeHearing,
    hear_app,
    hear_runtime,
)
from tests.support.identity_support import (
    APP,
    altered,
    valid_callers,
    valid_headers,
    valid_pair,
)
from tests.support.live_support import append, audio, commit, update
from tests.support.speak_support import assert_refusal_body

URL = f"{REALTIME_PATH}?intent=transcription"
COMPLETED = "conversation.item.input_audio_transcription.completed"


def _client(engine: FakeHearing | None = None, **settings: float) -> TestClient:
    return TestClient(hear_app(hear_runtime(engine or FakeHearing()), **settings))


def _denied(
    client: TestClient, url: str = URL, headers: dict[str, str] | None = None
) -> WebSocketDenialResponse:
    sent = valid_headers() if headers is None else headers
    connection = client.websocket_connect(url, headers=sent)
    with pytest.raises(WebSocketDenialResponse) as denial:
        connection.__enter__()
    return denial.value


def _assert_denied(
    denial: WebSocketDenialResponse, status: int, code: RefusalCode, param: str | None
) -> None:
    assert_refusal_body(denial.status_code, denial.json(), status, code, param)


def test_a_session_hears_a_committed_turn_and_ends_when_the_client_leaves() -> None:
    with _client().websocket_connect(URL, headers=valid_headers()) as socket:
        assert socket.receive_json()["type"] == "session.created"
        socket.send_text(update("en"))
        assert socket.receive_json()["type"] == "session.updated"
        socket.send_text(append(audio(0.5)))
        socket.send_text(commit())
        events = [socket.receive_json() for _ in range(4)]

    assert [event["type"] for event in events] == [
        "input_audio_buffer.speech_started",
        "input_audio_buffer.committed",
        "conversation.item.created",
        COMPLETED,
    ]
    assert events[-1]["transcript"] == HEARD_TEXT


def test_a_setup_in_another_language_ends_the_session_with_an_error() -> None:
    with _client().websocket_connect(URL, headers=valid_headers()) as socket:
        socket.receive_json()
        socket.send_text(update("fr"))
        error = socket.receive_json()
        with pytest.raises(WebSocketDisconnect):
            socket.receive_json()

    assert error["type"] == "error"
    assert error["error"]["code"] == RefusalCode.INVALID_REQUEST
    assert error["error"]["param"] == "session.audio.input.transcription.language"


@pytest.mark.parametrize(
    ("hears", "engine", "settings", "hold_a_place"),
    [
        pytest.param(False, FakeHearing(), {}, False, id="not-permitted"),
        pytest.param(False, None, {}, False, id="and-hearing-unavailable"),
        pytest.param(
            False,
            FakeHearing(),
            {"max_hearing_sessions": 1},
            True,
            id="and-sessions-full",
        ),
    ],
)
def test_an_app_without_the_hear_permission_is_told_so_before_everything_else(
    *,
    hears: bool,
    engine: FakeHearing | None,
    settings: dict[str, int],
    hold_a_place: bool,
) -> None:
    runtime = hear_runtime(engine)
    if hold_a_place:
        assert runtime.hearing.open_session(settings["max_hearing_sessions"])
    client = TestClient(
        create_control_app(Settings(**settings), runtime, valid_callers(hear=hears))
    )

    denial = _denied(client)

    _assert_denied(denial, HTTP_403_FORBIDDEN, RefusalCode.NOT_ENABLED, None)


def test_a_caller_without_tokens_is_told_so_before_anything_else() -> None:
    runtime = hear_runtime(None)
    assert runtime.hearing.open_session(1)
    client = TestClient(hear_app(runtime, max_hearing_sessions=1))

    denial = _denied(client, headers={})

    _assert_denied(denial, HTTP_401_UNAUTHORIZED, RefusalCode.UNAUTHORIZED, None)
    assert denial.headers[AUTHENTICATE_HEADER] == "Bearer"


def test_a_person_outside_the_service_group_is_forbidden() -> None:
    pair = valid_pair()
    headers = {
        WORKLOAD_TOKEN_HEADER: pair.workload,
        AUTHORIZATION_HEADER: "Bearer " + altered(pair.person, groups=[f"/apps/{APP}"]),
    }

    denial = _denied(_client(), headers=headers)

    _assert_denied(denial, HTTP_403_FORBIDDEN, RefusalCode.FORBIDDEN, None)


@pytest.mark.parametrize(
    ("url", "status", "code", "param"),
    [
        (
            "/v2/realtime?intent=transcription",
            HTTP_400_BAD_REQUEST,
            RefusalCode.UNSUPPORTED_VERSION,
            None,
        ),
        ("/v1/elsewhere", HTTP_404_NOT_FOUND, RefusalCode.NOT_FOUND, None),
        (
            REALTIME_PATH,
            HTTP_422_UNPROCESSABLE_CONTENT,
            RefusalCode.INVALID_REQUEST,
            "intent",
        ),
        (
            f"{REALTIME_PATH}?intent=conversation",
            HTTP_422_UNPROCESSABLE_CONTENT,
            RefusalCode.INVALID_REQUEST,
            "intent",
        ),
    ],
    ids=["wrong-version", "no-such-path", "no-intent", "other-intent"],
)
def test_an_upgrade_that_is_not_the_one_route_is_denied(
    url: str, status: int, code: RefusalCode, param: str | None
) -> None:
    _assert_denied(_denied(_client(), url), status, code, param)


def test_hearing_that_is_unavailable_denies_the_upgrade() -> None:
    client = TestClient(hear_app(hear_runtime(None)))

    _assert_denied(
        _denied(client),
        HTTP_503_SERVICE_UNAVAILABLE,
        RefusalCode.UNAVAILABLE,
        None,
    )


def test_a_session_beyond_the_bound_is_denied_as_busy_until_one_leaves() -> None:
    client = _client(max_hearing_sessions=1)

    with client.websocket_connect(URL, headers=valid_headers()) as first:
        first.receive_json()
        denial = _denied(client)
    with client.websocket_connect(URL, headers=valid_headers()) as again:
        reopened = again.receive_json()

    _assert_denied(denial, HTTP_429_TOO_MANY_REQUESTS, RefusalCode.BUSY, None)
    assert int(denial.headers[RETRY_AFTER_HEADER]) >= 1
    assert reopened["type"] == "session.created"

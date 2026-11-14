"""A drain ends a live session at a turn boundary, with 1012 and no error."""

import asyncio
import logging

import pytest
from fastapi.testclient import TestClient
from starlette.status import HTTP_503_SERVICE_UNAVAILABLE, WS_1012_SERVICE_RESTART
from starlette.testclient import WebSocketDenialResponse

from speech_client.wire import RefusalCode
from speech_service.hearing import DRAIN_TURN_SECONDS
from speech_service.voice_activity import WINDOW_SECONDS
from tests.support.clock_support import FakeClock
from tests.support.hear_support import (
    FakeHearing,
    hear_app,
    hear_runtime,
    loaded,
    multipart_body,
    post_recording,
    recording,
)
from tests.support.identity_support import valid_headers
from tests.support.live_support import (
    COMPLETED,
    FAILED,
    FakeSocket,
    append,
    audio,
    commit,
    started,
)
from tests.support.speak_support import assert_refusal_body

_ANNOUNCED_AND_COMPLETED = [
    "session.created",
    "input_audio_buffer.speech_started",
    "input_audio_buffer.speech_stopped",
    "input_audio_buffer.committed",
    "conversation.item.created",
    COMPLETED,
]
_FINAL_WAIT_SECONDS = 30.0
_SESSION_URL = "/v1/realtime?intent=transcription"


def test_a_draining_session_opens_no_second_turn_though_audio_keeps_coming() -> None:
    """WIRE-146: the session takes audio only until its open turn ends."""

    async def scenario() -> tuple[FakeSocket, FakeHearing]:
        engine = FakeHearing(hold=True)
        hearing = loaded(engine, FakeClock())
        session, socket = await started(hearing)
        await session.on_message(append(audio(0.5)))
        hearing.begin_drain()
        await session.on_message(append(audio(0.8, 0) + audio(0.5)))
        await session.on_message(append(audio(0.5)))
        engine.release.set()
        await session.settled()
        await session.on_tick()
        return socket, engine

    socket, engine = asyncio.run(scenario())

    assert socket.types() == _ANNOUNCED_AND_COMPLETED
    assert len(engine.heard) == 1
    assert socket.close_code == WS_1012_SERVICE_RESTART


def test_a_commit_after_the_draining_turn_ended_opens_no_second_item() -> None:
    """WIRE-146: a commit of audio after the last turn adds no item and no error."""

    async def scenario() -> tuple[FakeSocket, FakeHearing]:
        engine = FakeHearing(hold=True)
        hearing = loaded(engine, FakeClock())
        session, socket = await started(hearing)
        await session.on_message(append(audio(0.5)))
        hearing.begin_drain()
        await session.on_message(append(audio(0.8, 0) + audio(0.5)))
        await session.on_message(commit())
        engine.release.set()
        await session.settled()
        await session.on_tick()
        return socket, engine

    socket, engine = asyncio.run(scenario())

    assert socket.types() == _ANNOUNCED_AND_COMPLETED
    assert len(engine.heard) == 1
    assert socket.close_code == WS_1012_SERVICE_RESTART


def test_a_final_unfinished_at_the_bound_fails_its_item_and_closes_1012() -> None:
    """WIRE-80, WIRE-147: an announced item is failed when the drain's bound passes."""

    async def scenario() -> tuple[FakeSocket, FakeHearing]:
        clock = FakeClock()
        engine = FakeHearing(hold=True)
        hearing = loaded(engine, clock)
        session, socket = await started(hearing, final_wait_seconds=_FINAL_WAIT_SECONDS)
        await session.on_message(append(audio(0.5)))
        await session.on_message(append(audio(0.8, 0)))
        hearing.begin_drain()
        clock.jump(DRAIN_TURN_SECONDS + _FINAL_WAIT_SECONDS)
        await session.on_tick()
        engine.release.set()
        await session.settled()
        return socket, engine

    socket, _engine = asyncio.run(scenario())

    assert socket.types()[-2:] == [FAILED, "error"]
    assert COMPLETED not in socket.types()
    assert socket.close_code == WS_1012_SERVICE_RESTART


def test_a_session_waits_for_a_final_in_the_queue_before_it_closes() -> None:
    """WIRE-146: a queued final completes before the session closes."""

    async def scenario() -> tuple[bool, FakeSocket]:
        engine = FakeHearing(hold=True)
        hearing = loaded(engine, FakeClock())
        session, socket = await started(hearing)
        await session.on_message(append(audio(0.5)))
        await session.on_message(append(audio(0.8, 0)))
        hearing.begin_drain()
        await session.on_tick()
        closed_while_deciding = socket.closed
        engine.release.set()
        await session.settled()
        await session.on_tick()
        return closed_while_deciding, socket

    closed_while_deciding, socket = asyncio.run(scenario())

    assert not closed_while_deciding
    assert socket.types() == _ANNOUNCED_AND_COMPLETED
    assert socket.close_code == WS_1012_SERVICE_RESTART


def test_a_turn_still_open_when_the_drain_is_30_seconds_old_ends_there_and_is_heard(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """WIRE-146, WIRE-147: a turn open 30 s into the drain ends there and is heard."""

    async def scenario() -> tuple[FakeSocket, bool]:
        clock = FakeClock()
        hearing = loaded(FakeHearing(), clock)
        session, socket = await started(hearing)
        await session.on_message(append(audio(0.5)))
        hearing.begin_drain()
        clock.jump(DRAIN_TURN_SECONDS - 0.1)
        await session.on_tick()
        open_before_the_bound = not socket.closed
        clock.jump(0.1)
        await session.on_tick()
        await session.settled()
        await session.on_tick()
        return socket, open_before_the_bound

    caplog.set_level(logging.INFO)

    socket, open_before_the_bound = asyncio.run(scenario())

    assert open_before_the_bound
    assert socket.types() == _ANNOUNCED_AND_COMPLETED
    assert FAILED not in socket.types()
    assert socket.close_code == WS_1012_SERVICE_RESTART
    assert "ended_by=drain" in caplog.text


def test_speech_still_being_confirmed_at_the_bound_closes_without_a_turn() -> None:
    """WIRE-146: speech still being confirmed at the bound closes without a turn."""

    async def scenario() -> FakeSocket:
        clock = FakeClock()
        hearing = loaded(FakeHearing(), clock)
        session, socket = await started(hearing)
        await session.on_message(append(audio(2 * WINDOW_SECONDS)))
        hearing.begin_drain()
        clock.jump(DRAIN_TURN_SECONDS)
        await session.on_tick()
        return socket

    socket = asyncio.run(scenario())

    assert socket.types() == ["session.created"]
    assert socket.close_code == WS_1012_SERVICE_RESTART


def test_a_session_ended_for_another_reason_is_not_closed_again_by_the_drain() -> None:
    """WIRE-81: a session ended for another reason is not closed again by the drain."""

    async def scenario() -> FakeSocket:
        hearing = loaded(FakeHearing(), FakeClock())
        session, socket = await started(hearing)
        await session.on_message("not json")
        hearing.begin_drain()
        await session.on_tick()
        return socket

    socket = asyncio.run(scenario())

    assert socket.close_code != WS_1012_SERVICE_RESTART


def test_a_draining_hearing_is_not_ready_but_keeps_its_engine_for_a_held_place() -> (
    None
):
    """WIRE-129, WIRE-144: a draining hearing is not ready; held places run on."""
    hearing = loaded(FakeHearing(), FakeClock())
    place = hearing.open_session(1)
    assert place is not None

    hearing.begin_drain()

    assert hearing.health()["ready"] is False
    assert hearing.capture() is None
    assert hearing.open_detector() is not None


def test_the_sessions_have_ended_once_every_place_is_given_back() -> None:
    """WIRE-147: the sessions have ended once every place is given back."""

    async def scenario() -> list[bool]:
        hearing = loaded(FakeHearing(), FakeClock())
        places = [hearing.open_session(2), hearing.open_session(2)]
        assert all(place is not None for place in places)
        waiting = asyncio.create_task(hearing.sessions_ended())
        await asyncio.sleep(0)
        places[0].close()
        await asyncio.sleep(0)
        after_one = waiting.done()
        places[1].close()
        await waiting
        return [after_one, waiting.done()]

    assert asyncio.run(scenario()) == [False, True]


def test_a_runtime_that_began_to_shut_down_reports_not_ready_and_refuses_new_work() -> (
    None
):
    """WIRE-103, WIRE-124, WIRE-129, WIRE-144: the drain refuses new work."""
    runtime = hear_runtime(FakeHearing())
    client = TestClient(hear_app(runtime))
    runtime.begin_shutdown()

    health = client.get("/health").json()
    upgrade = client.websocket_connect(_SESSION_URL, headers=valid_headers())
    with pytest.raises(WebSocketDenialResponse) as denial:
        upgrade.__enter__()
    transcription = post_recording(client, multipart_body(file=recording()))

    assert health["speaking"]["ready"] is False
    assert health["hearing"]["ready"] is False
    assert denial.value.status_code == HTTP_503_SERVICE_UNAVAILABLE
    assert_refusal_body(
        transcription.status_code,
        transcription.json(),
        HTTP_503_SERVICE_UNAVAILABLE,
        RefusalCode.UNAVAILABLE,
        None,
    )

"""A live session: its events in order, its items' terminals, and how it ends."""

import asyncio
import json
import logging
import time
from typing import TYPE_CHECKING

import pytest

from speech_runtime_service.live_session import POLL_WAIT_SECONDS, poll_forever
from tests.support.hear_support import HEARD_TEXT, FakeHearing, loaded
from tests.support.live_support import (
    WAIT_SECONDS,
    FakeSocket,
    append,
    audio,
    commit,
    started,
    update,
)

if TYPE_CHECKING:
    from speech_runtime_service.live_session import LiveSession

COMPLETED = "conversation.item.input_audio_transcription.completed"
FAILED = "conversation.item.input_audio_transcription.failed"
POLLS = 3
SETTLE_TURNS = 3
DELTA = "conversation.item.input_audio_transcription.delta"


async def _one_question(session: LiveSession) -> None:
    await session.on_message(append(audio(0.5)))
    await session.on_message(append(audio(0.8, 0)))
    await session.on_tick()


async def _commit_a_turn(session: LiveSession, seconds: float = 0.3) -> None:
    await session.on_message(append(audio(seconds)))
    await session.on_message(commit())


def _end_is(socket: FakeSocket, code: str, failed: int) -> bool:
    types = socket.types()
    error = socket.of_type("error")[0]["error"]
    assert isinstance(error, dict)
    return (
        types[-1] == "error"
        and types.count(FAILED) == failed
        and types[-1 - failed : -1] == [FAILED] * failed
        and error["code"] == code
        and socket.closed
    )


def test_the_session_opens_with_the_default_session() -> None:
    async def scenario() -> FakeSocket:
        _, socket, _ = await started(loaded(FakeHearing()))
        return socket

    socket = asyncio.run(scenario())

    assert socket.types() == ["session.created"]
    assert json.dumps(socket.events[0]["session"]).count('"de"') == 1


def test_a_question_without_a_commit_is_announced_ended_and_heard_once() -> None:
    async def scenario() -> FakeSocket:
        session, socket, _ = await started(loaded(FakeHearing()))
        await _one_question(session)
        await session.settled()
        return socket

    socket = asyncio.run(scenario())

    assert socket.types() == [
        "session.created",
        "input_audio_buffer.speech_started",
        "input_audio_buffer.speech_stopped",
        "input_audio_buffer.committed",
        "conversation.item.created",
        COMPLETED,
    ]
    assert len({event["item_id"] for event in socket.events[1:4]}) == 1
    assert socket.events[-1]["transcript"] == HEARD_TEXT
    assert socket.events[-1]["usage"] == {
        "type": "duration",
        "seconds": pytest.approx(1.3),
    }


@pytest.mark.parametrize(
    ("messages", "language"),
    [([], "de"), ([update("en")], "en")],
    ids=["without-an-update", "updated-to-english"],
)
def test_a_session_hears_in_its_language(messages: list[str], language: str) -> None:
    async def scenario() -> FakeHearing:
        engine = FakeHearing()
        session, _, _ = await started(loaded(engine))
        for message in messages:
            await session.on_message(message)
        await _one_question(session)
        await session.settled()
        return engine

    engine = asyncio.run(scenario())

    assert [heard.value for _, heard in engine.heard] == [language]


def test_a_valid_update_is_answered_with_the_applied_session() -> None:
    async def scenario() -> FakeSocket:
        session, socket, _ = await started(loaded(FakeHearing()))
        await session.on_message(update("en"))
        return socket

    socket = asyncio.run(scenario())

    assert socket.events[-1]["type"] == "session.updated"
    assert json.dumps(socket.events[-1]["session"]).count('"en"') == 1


def test_two_questions_make_two_items_chained_by_their_predecessor() -> None:
    async def scenario() -> FakeSocket:
        session, socket, _ = await started(loaded(FakeHearing()))
        for _ in range(2):
            await _one_question(session)
            await session.settled()
        return socket

    socket = asyncio.run(scenario())

    first, second = socket.of_type("input_audio_buffer.committed")
    created = socket.of_type("conversation.item.created")
    assert first["item_id"] != second["item_id"]
    assert (first["previous_item_id"], second["previous_item_id"]) == (
        None,
        first["item_id"],
    )
    assert created[1]["previous_item_id"] == first["item_id"]
    assert [done["item_id"] for done in socket.of_type(COMPLETED)] == [
        first["item_id"],
        second["item_id"],
    ]


def test_deltas_carry_the_words_two_decodes_agree_on() -> None:
    async def scenario() -> FakeSocket:
        session, socket, clock = await started(loaded(FakeHearing()))
        await session.on_message(append(audio(0.5)))
        await session.on_tick()
        await session.settled()
        clock.now += 0.4
        await session.on_message(append(audio(0.1)))
        await session.on_tick()
        await session.settled()
        return socket

    socket = asyncio.run(scenario())

    item = socket.of_type("input_audio_buffer.speech_started")[0]["item_id"]
    assert [(d["item_id"], d["delta"]) for d in socket.of_type(DELTA)] == [
        (item, HEARD_TEXT)
    ]


def test_a_contended_partial_tick_is_skipped_without_an_error() -> None:
    async def scenario() -> tuple[FakeSocket, FakeHearing]:
        engine = FakeHearing()
        hearing = loaded(engine)
        taken = hearing.gate.try_acquire()
        assert taken is not None
        session, socket, _ = await started(hearing)
        await session.on_message(append(audio(0.5)))
        await session.on_tick()
        await session.settled()
        return socket, engine

    socket, engine = asyncio.run(scenario())

    assert socket.types() == ["session.created", "input_audio_buffer.speech_started"]
    assert engine.heard == []


def test_a_commit_mid_speech_ends_the_turn_without_a_speech_stopped() -> None:
    async def scenario() -> FakeSocket:
        session, socket, _ = await started(loaded(FakeHearing()))
        await _commit_a_turn(session)
        await session.settled()
        return socket

    assert asyncio.run(scenario()).types() == [
        "session.created",
        "input_audio_buffer.speech_started",
        "input_audio_buffer.committed",
        "conversation.item.created",
        COMPLETED,
    ]


def test_a_commit_with_no_audio_ends_the_session_failing_the_announced_item() -> None:
    async def scenario() -> FakeSocket:
        engine = FakeHearing(hold=True)
        session, socket, _ = await started(loaded(engine))
        await _commit_a_turn(session)
        await session.on_message(commit("empty"))
        engine.release.set()
        await session.on_disconnect()
        return socket

    socket = asyncio.run(scenario())

    assert _end_is(socket, "invalid_request", failed=1)
    assert socket.of_type("error")[0]["error"]["event_id"] == "empty"
    committed = socket.of_type("input_audio_buffer.committed")[0]
    assert socket.of_type(FAILED)[0]["item_id"] == committed["item_id"]


@pytest.mark.parametrize(
    ("message", "param"),
    [
        (append("!!!"), "audio"),
        (append("AAAA"), "audio"),
        (json.dumps({"type": "input_audio_buffer.clear"}), "type"),
        (json.dumps({"type": "response.create"}), "type"),
        ("not json", None),
        (None, None),
    ],
    ids=["bad-base64", "odd-byte-count", "clear", "unknown", "malformed", "binary"],
)
def test_a_message_that_breaks_the_protocol_ends_the_session(
    message: str | None, param: str | None
) -> None:
    async def scenario() -> FakeSocket:
        session, socket, _ = await started(loaded(FakeHearing()))
        await session.on_message(message)
        return socket

    socket = asyncio.run(scenario())

    assert _end_is(socket, "invalid_request", failed=0)
    assert socket.of_type("error")[0]["error"]["param"] == param


@pytest.mark.parametrize(
    "messages",
    [[update("en"), update("en")], [append(audio(0.1, 0)), update("en")]],
    ids=["a-second-update", "an-update-after-audio"],
)
def test_the_setup_is_one_update_before_the_first_audio(messages: list[str]) -> None:
    async def scenario() -> FakeSocket:
        session, socket, _ = await started(loaded(FakeHearing()))
        for message in messages:
            await session.on_message(message)
        return socket

    assert _end_is(asyncio.run(scenario()), "invalid_request", failed=0)


def test_a_session_that_ended_takes_nothing_more() -> None:
    async def scenario() -> list[object]:
        session, socket, _ = await started(loaded(FakeHearing()))
        await session.on_message("not json")
        sent = len(socket.events)
        await session.on_message(update("en"))
        await session.on_tick()
        return socket.types()[sent:]

    assert asyncio.run(scenario()) == []


def test_a_decode_failure_fails_every_announced_item_then_errors_and_closes() -> None:
    async def scenario() -> FakeSocket:
        engine = FakeHearing(hold=True)
        session, socket, _ = await started(loaded(engine))
        await _commit_a_turn(session)
        await _commit_a_turn(session)
        engine.fail_transcription = True
        engine.release.set()
        await socket.until_closed()
        await session.settled()
        return socket

    socket = asyncio.run(scenario())

    assert _end_is(socket, "unavailable", failed=2)
    assert socket.of_type(COMPLETED) == []


def test_no_log_record_holds_the_heard_text_of_a_failed_decode(
    caplog: pytest.LogCaptureFixture,
) -> None:
    async def scenario() -> None:
        engine = FakeHearing()
        engine.fail_transcription = True
        session, socket, _ = await started(loaded(engine))
        await _commit_a_turn(session)
        await socket.until_closed()

    caplog.set_level(logging.DEBUG)

    asyncio.run(scenario())

    assert HEARD_TEXT not in caplog.text


def test_a_lost_engine_ends_the_session_as_unavailable() -> None:
    async def scenario() -> FakeSocket:
        hearing = loaded(FakeHearing())
        session, socket, _ = await started(hearing)
        hearing.close()
        await _commit_a_turn(session)
        return socket

    assert _end_is(asyncio.run(scenario()), "unavailable", failed=1)


def test_a_second_ended_turn_while_one_waits_ends_the_session_as_busy() -> None:
    async def scenario() -> FakeSocket:
        engine = FakeHearing(hold=True)
        session, socket, _ = await started(loaded(engine))
        await _commit_a_turn(session)
        await _commit_a_turn(session)
        await session.on_message(append(audio(0.3)))
        assert not socket.of_type("error")
        await session.on_message(commit())
        engine.release.set()
        await session.on_disconnect()
        return socket

    socket = asyncio.run(scenario())

    assert _end_is(socket, "busy", failed=3)


def test_a_final_whose_wait_ran_out_is_never_started_on_a_free_decode() -> None:
    async def scenario() -> tuple[FakeSocket, FakeHearing]:
        engine = FakeHearing()
        session, socket, _ = await started(loaded(engine), final_wait_seconds=0)
        await _commit_a_turn(session)
        await session.settled()
        return socket, engine

    socket, engine = asyncio.run(scenario())

    assert _end_is(socket, "busy", failed=1)
    assert engine.heard == []


def test_a_final_that_waits_out_its_time_leaves_the_running_decode_its_lease() -> None:
    async def scenario() -> tuple[FakeSocket, bool, bool]:
        engine = FakeHearing()
        hearing = loaded(engine)
        running = hearing.gate.try_acquire()
        assert running is not None
        session, socket, _ = await started(hearing, final_wait_seconds=0.05)
        await _commit_a_turn(session)
        await socket.until_closed()
        still_held = hearing.gate.try_acquire() is None
        running.close()
        next_job = await hearing.gate.acquire(time.monotonic() + WAIT_SECONDS, 1)
        return socket, still_held, next_job is not None

    socket, still_held, next_job_served = asyncio.run(scenario())

    assert _end_is(socket, "busy", failed=1)
    assert still_held
    assert next_job_served


def test_finals_of_several_sessions_are_served_in_arrival_order_before_file_jobs() -> (
    None
):
    async def scenario() -> tuple[list[int], int]:
        engine = FakeHearing()
        hearing = loaded(engine)
        running = hearing.gate.try_acquire()
        assert running is not None
        file_job = asyncio.create_task(
            hearing.gate.acquire(time.monotonic() + WAIT_SECONDS, 1)
        )
        sessions = [(await started(hearing)) for _ in range(2)]
        await asyncio.sleep(0)
        for (session, _, _), seconds in zip(sessions, (0.3, 0.4), strict=True):
            await _commit_a_turn(session, seconds)
        for _ in range(SETTLE_TURNS):
            await asyncio.sleep(0)
        running.close()
        for _, socket, _ in sessions:
            await socket.until(lambda socket=socket: bool(socket.of_type(COMPLETED)))
        file_lease = await file_job
        assert file_lease is not None
        heard_before_the_file_job = len(engine.heard)
        file_lease.close()
        return [size for size, _ in engine.heard], heard_before_the_file_job

    sizes, heard_before_the_file_job = asyncio.run(scenario())

    assert sizes[0] < sizes[1]
    assert heard_before_the_file_job == len(sizes)


def test_a_disconnect_mid_decode_sends_nothing_and_holds_the_decode() -> None:
    async def scenario() -> tuple[int, int, bool, bool, int]:
        engine = FakeHearing(hold=True)
        hearing = loaded(engine)
        session, socket, _ = await started(hearing)
        await _commit_a_turn(session)
        await asyncio.to_thread(engine.entered.wait, WAIT_SECONDS)
        sent = len(socket.events)
        await session.on_disconnect()
        held = hearing.gate.try_acquire() is None
        engine.release.set()
        next_job = await hearing.gate.acquire(time.monotonic() + WAIT_SECONDS, 1)
        return (
            sent,
            len(socket.events),
            held,
            next_job is not None,
            engine.most_concurrent,
        )

    sent, after, held, next_job_served, most_concurrent = asyncio.run(scenario())

    assert after == sent
    assert held
    assert next_job_served
    assert most_concurrent == 1


def test_a_result_that_arrives_after_the_session_ended_is_dropped() -> None:
    async def scenario() -> FakeSocket:
        engine = FakeHearing(hold=True)
        session, socket, _ = await started(loaded(engine))
        await _commit_a_turn(session)
        await session.on_message("not json")
        sent = len(socket.events)
        engine.release.set()
        await session.on_disconnect()
        assert len(socket.events) == sent
        return socket

    socket = asyncio.run(scenario())

    assert socket.of_type(COMPLETED) == []
    assert _end_is(socket, "invalid_request", failed=1)


def test_the_poll_runs_every_three_tenths_of_a_second() -> None:
    async def scenario() -> list[float]:
        session, _, _ = await started(loaded(FakeHearing()))
        waits: list[float] = []

        async def sleep(seconds: float) -> None:
            waits.append(seconds)
            if len(waits) == POLLS:
                raise asyncio.CancelledError
            await asyncio.sleep(0)

        with pytest.raises(asyncio.CancelledError):
            await poll_forever(session, sleep)
        return waits

    assert asyncio.run(scenario()) == [POLL_WAIT_SECONDS] * POLLS

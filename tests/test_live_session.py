"""A live session: its events in order, its items' terminals, and how it ends."""

import asyncio
import json
import logging
from typing import TYPE_CHECKING

import pytest

from speech_client.wire import Language
from speech_service.admission_gate import Lease
from speech_service.live_session import POLL_WAIT_SECONDS, poll_forever
from speech_service.voice_activity import WINDOW_SECONDS
from tests.support.clock_support import FakeClock
from tests.support.hear_support import HEARD_TEXT, FakeHearing, loaded
from tests.support.live_support import (
    COMPLETED,
    FAILED,
    WAIT_SECONDS,
    FakeSocket,
    append,
    audio,
    commit,
    commit_a_turn,
    end_is,
    started,
    update,
)

if TYPE_CHECKING:
    from speech_service.hearing import HearingEngine
    from speech_service.live_session import LiveSession

POLLS = 3
THREE_TURNS_SECONDS = 4 * 30 + 1
BLOCKER_AND_MARKER = 2
DELTA = "conversation.item.input_audio_transcription.delta"
# Half a second of speech touches 16 windows; 16 windows of silence end the turn.
QUESTION_WINDOWS = 32
SPEECH_WINDOWS = 16


def _transcribe_silence(engine: HearingEngine) -> str:
    return engine.transcribe(b"\0\0", (Language.DE,)).text


async def _one_question(session: LiveSession) -> None:
    await session.on_message(append(audio(0.5)))
    await session.on_message(append(audio(0.8, 0)))
    await session.on_tick()


def test_the_session_opens_with_the_default_session() -> None:
    """WIRE-55, WIRE-68: the session opens with the default session."""

    async def scenario() -> FakeSocket:
        _, socket = await started(loaded(FakeHearing(), FakeClock()))
        return socket

    socket = asyncio.run(scenario())

    assert socket.types() == ["session.created"]
    assert json.dumps(socket.events[0]["session"]).count('"de"') == 1


def test_a_question_without_a_commit_is_announced_ended_and_heard_once() -> None:
    """WIRE-65, WIRE-66, WIRE-70, WIRE-72: a question is heard without a commit."""

    async def scenario() -> FakeSocket:
        session, socket = await started(loaded(FakeHearing(), FakeClock()))
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
        "seconds": pytest.approx(QUESTION_WINDOWS * WINDOW_SECONDS),
    }


@pytest.mark.parametrize(
    ("messages", "language"),
    [([], "de"), ([update("en")], "en")],
    ids=["without-an-update", "updated-to-english"],
)
def test_a_session_hears_in_its_language(messages: list[str], language: str) -> None:
    """WIRE-56, WIRE-57: a session hears in its language."""

    async def scenario() -> FakeHearing:
        engine = FakeHearing()
        session, _ = await started(loaded(engine, FakeClock()))
        for message in messages:
            await session.on_message(message)
        await _one_question(session)
        await session.settled()
        return engine

    engine = asyncio.run(scenario())

    assert [heard for _, heard in engine.heard] == [(Language(language),)]


def test_a_valid_update_is_answered_with_the_applied_session() -> None:
    """WIRE-56, WIRE-69: a valid update is answered with the applied session."""

    async def scenario() -> FakeSocket:
        session, socket = await started(loaded(FakeHearing(), FakeClock()))
        await session.on_message(update("en"))
        return socket

    socket = asyncio.run(scenario())

    assert socket.events[-1]["type"] == "session.updated"
    assert json.dumps(socket.events[-1]["session"]).count('"en"') == 1


@pytest.mark.parametrize(
    ("messages", "heard_in"),
    [
        ([], None),
        ([update("en")], None),
        ([update(None, languages=["en"])], None),
        ([update(None, languages=["en", "de"])], Language.DE),
    ],
    ids=["default", "language", "one-element-set", "set-of-two"],
)
def test_every_completed_item_names_exactly_the_language_it_was_heard_in(
    messages: list[str], heard_in: Language | None
) -> None:
    """WIRE-75, WIRE-136: a completed item names one language, the engine's."""

    async def scenario() -> tuple[FakeSocket, FakeHearing]:
        engine = FakeHearing()
        engine.heard_in = heard_in
        session, socket = await started(loaded(engine, FakeClock()))
        for message in messages:
            await session.on_message(message)
        await _one_question(session)
        await session.settled()
        return socket, engine

    socket, engine = asyncio.run(scenario())

    (completed,) = socket.of_type(COMPLETED)
    (_, allowed) = engine.heard[0]
    assert completed["languages"] == [{"code": (heard_in or allowed[0]).value}]


def test_the_final_names_its_own_language_and_the_deltas_name_none() -> None:
    """WIRE-71, WIRE-75: a partial may be heard in another language than its final."""

    async def scenario() -> FakeSocket:
        engine = FakeHearing("Guten Morgen")
        clock = FakeClock()
        session, socket = await started(loaded(engine, clock))
        await session.on_message(update(None, languages=["de", "en"]))
        await session.on_message(append(audio(0.5)))
        engine.heard_in = Language.DE
        for _ in range(2):
            await session.on_tick()
            await session.settled()
            clock.advance(0.4)
            await session.on_message(append(audio(0.1)))
        engine.heard_in = Language.EN
        engine.text = "Good morning"
        await session.on_message(commit())
        await session.settled()
        return socket

    socket = asyncio.run(scenario())

    deltas = socket.of_type(DELTA)
    assert deltas
    assert all("languages" not in delta for delta in deltas)
    (completed,) = socket.of_type(COMPLETED)
    assert completed["languages"] == [{"code": "en"}]
    assert completed["transcript"] == "Good morning"


def test_two_questions_make_two_items_chained_by_their_predecessor() -> None:
    """WIRE-73, WIRE-74, WIRE-79: two questions make two chained items."""

    async def scenario() -> FakeSocket:
        session, socket = await started(loaded(FakeHearing(), FakeClock()))
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
    """WIRE-71: deltas carry the words two decodes agree on."""

    async def scenario() -> FakeSocket:
        clock = FakeClock()
        session, socket = await started(loaded(FakeHearing(), clock))
        await session.on_message(append(audio(0.5)))
        await session.on_tick()
        await session.settled()
        clock.advance(0.4)
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
    """WIRE-86: a contended partial tick is skipped without an error."""

    async def scenario() -> tuple[FakeSocket, FakeHearing]:
        engine = FakeHearing()
        hearing = loaded(engine, FakeClock())
        taken = hearing.gate.try_acquire()
        assert taken is not None
        session, socket = await started(hearing)
        await session.on_message(append(audio(0.5)))
        await session.on_tick()
        await session.settled()
        return socket, engine

    socket, engine = asyncio.run(scenario())

    assert socket.types() == ["session.created", "input_audio_buffer.speech_started"]
    assert engine.heard == []


def test_a_commit_mid_speech_ends_the_turn_without_a_speech_stopped() -> None:
    """WIRE-63, WIRE-72: a commit mid speech ends the turn without a speech stopped."""

    async def scenario() -> FakeSocket:
        session, socket = await started(loaded(FakeHearing(), FakeClock()))
        await commit_a_turn(session)
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
    """WIRE-62, WIRE-64, WIRE-76: a commit with no audio ends the session."""

    async def scenario() -> FakeSocket:
        engine = FakeHearing(hold=True)
        session, socket = await started(loaded(engine, FakeClock()))
        await commit_a_turn(session)
        await session.on_message(commit("empty"))
        engine.release.set()
        await session.on_disconnect()
        return socket

    socket = asyncio.run(scenario())

    assert end_is(socket, "invalid_request", failed=1)
    assert socket.of_type("error")[0]["error"]["event_id"] == "empty"
    committed = socket.of_type("input_audio_buffer.committed")[0]
    assert socket.of_type(FAILED)[0]["item_id"] == committed["item_id"]


@pytest.mark.parametrize(
    ("message", "param"),
    [
        (append("!!!"), "audio"),
        (append("ä"), "audio"),
        (append("AAAA"), "audio"),
        (json.dumps({"type": "input_audio_buffer.clear"}), "type"),
        (json.dumps({"type": "response.create"}), "type"),
        ("not json", None),
        (None, None),
    ],
    ids=[
        "bad-base64",
        "non-ascii-base64",
        "odd-byte-count",
        "clear",
        "unknown",
        "malformed",
        "binary",
    ],
)
def test_a_message_that_breaks_the_protocol_ends_the_session(
    message: str | None, param: str | None
) -> None:
    """WIRE-49, WIRE-59, WIRE-60, WIRE-61: a protocol break ends the session."""

    async def scenario() -> FakeSocket:
        session, socket = await started(loaded(FakeHearing(), FakeClock()))
        await session.on_message(message)
        return socket

    socket = asyncio.run(scenario())

    assert end_is(socket, "invalid_request", failed=0)
    assert socket.of_type("error")[0]["error"]["param"] == param


@pytest.mark.parametrize(
    "messages",
    [[update("en"), update("en")], [append(audio(0.1, 0)), update("en")]],
    ids=["a-second-update", "an-update-after-audio"],
)
def test_the_setup_is_one_update_before_the_first_audio(messages: list[str]) -> None:
    """WIRE-56, WIRE-58: the setup is one update before the first audio."""

    async def scenario() -> FakeSocket:
        session, socket = await started(loaded(FakeHearing(), FakeClock()))
        for message in messages:
            await session.on_message(message)
        return socket

    assert end_is(asyncio.run(scenario()), "invalid_request", failed=0)


def test_a_session_that_ended_takes_nothing_more() -> None:
    async def scenario() -> list[object]:
        session, socket = await started(loaded(FakeHearing(), FakeClock()))
        await session.on_message("not json")
        sent = len(socket.events)
        await session.on_message(update("en"))
        await session.on_tick()
        return socket.types()[sent:]

    assert asyncio.run(scenario()) == []


def test_an_item_whose_final_decode_is_empty_completes_once_with_empty_text() -> None:
    """WIRE-75, WIRE-80: an empty final decode completes once with empty text."""

    async def scenario() -> FakeSocket:
        session, socket = await started(loaded(FakeHearing(text=""), FakeClock()))
        await _one_question(session)
        await session.settled()
        return socket

    socket = asyncio.run(scenario())

    completed = socket.of_type(COMPLETED)
    assert [event["transcript"] for event in completed] == [""]
    assert [event["usage"] for event in completed] == [
        {
            "type": "duration",
            "seconds": pytest.approx(QUESTION_WINDOWS * WINDOW_SECONDS),
        }
    ]
    assert socket.of_type(FAILED) == []


def test_a_decode_failure_fails_every_announced_item_then_errors_and_closes() -> None:
    """WIRE-76, WIRE-80, WIRE-81, WIRE-82: a decode failure ends the session."""

    async def scenario() -> FakeSocket:
        engine = FakeHearing(hold=True)
        session, socket = await started(loaded(engine, FakeClock()))
        await commit_a_turn(session)
        await commit_a_turn(session)
        engine.fail_transcription = True
        engine.release.set()
        await socket.until_closed()
        await session.settled()
        return socket

    socket = asyncio.run(scenario())

    assert end_is(socket, "unavailable", failed=2)
    assert socket.of_type(COMPLETED) == []


def test_no_log_record_holds_the_heard_text_of_a_failed_decode(
    caplog: pytest.LogCaptureFixture,
) -> None:
    async def scenario() -> None:
        engine = FakeHearing()
        engine.fail_transcription = True
        session, socket = await started(loaded(engine, FakeClock()))
        await commit_a_turn(session)
        await socket.until_closed()

    caplog.set_level(logging.DEBUG)

    asyncio.run(scenario())

    assert HEARD_TEXT not in caplog.text


def test_a_lost_engine_ends_the_session_as_unavailable() -> None:
    """WIRE-81, WIRE-82: a lost engine ends the session as unavailable."""

    async def scenario() -> FakeSocket:
        hearing = loaded(FakeHearing(), FakeClock())
        session, socket = await started(hearing)
        hearing.close()
        await commit_a_turn(session)
        return socket

    assert end_is(asyncio.run(scenario()), "unavailable", failed=1)


def test_a_second_ended_turn_while_one_waits_ends_the_session_as_busy() -> None:
    """WIRE-67, WIRE-81: a turn past two outstanding ends the session busy."""

    async def scenario() -> FakeSocket:
        engine = FakeHearing(hold=True)
        session, socket = await started(loaded(engine, FakeClock()))
        await commit_a_turn(session)
        await commit_a_turn(session)
        await session.on_message(append(audio(0.3)))
        assert not socket.of_type("error")
        await session.on_message(commit())
        engine.release.set()
        await session.on_disconnect()
        return socket

    socket = asyncio.run(scenario())

    assert end_is(socket, "busy", failed=3)


def test_a_final_whose_wait_ran_out_is_never_started_on_a_free_decode() -> None:
    """WIRE-81, WIRE-85: a final past its wait is never started on a free decode."""

    async def scenario() -> tuple[FakeSocket, FakeHearing]:
        engine = FakeHearing()
        session, socket = await started(
            loaded(engine, FakeClock()), final_wait_seconds=0
        )
        await commit_a_turn(session)
        await session.settled()
        return socket, engine

    socket, engine = asyncio.run(scenario())

    assert end_is(socket, "busy", failed=1)
    assert engine.heard == []


def test_a_final_that_waits_out_its_time_leaves_the_running_decode_its_lease() -> None:
    """WIRE-85, WIRE-87: a final past its wait leaves the running decode its lease."""

    async def scenario() -> tuple[FakeSocket, bool, bool]:
        clock = FakeClock()
        hearing = loaded(FakeHearing(), clock)
        running = hearing.gate.try_acquire()
        assert running is not None
        session, socket = await started(hearing, final_wait_seconds=WAIT_SECONDS)
        await commit_a_turn(session)
        await clock.until_a_wait_begins()
        clock.advance(2 * WAIT_SECONDS)
        await socket.until_closed()
        still_held = hearing.gate.try_acquire() is None
        running.close()
        next_job = await hearing.gate.acquire(clock.now() + WAIT_SECONDS, 1)
        return socket, still_held, next_job is not None

    socket, still_held, next_job_served = asyncio.run(scenario())

    assert end_is(socket, "busy", failed=1)
    assert still_held
    assert next_job_served


def test_a_disconnect_mid_decode_sends_nothing_and_holds_the_decode() -> None:
    """WIRE-83, WIRE-87: a disconnect mid decode sends nothing and holds the decode."""

    async def scenario() -> tuple[int, int, bool, bool, int]:
        engine = FakeHearing(hold=True)
        hearing = loaded(engine, FakeClock())
        session, socket = await started(hearing)
        await commit_a_turn(session)
        await asyncio.to_thread(engine.entered.wait, WAIT_SECONDS)
        sent = len(socket.events)
        await session.on_disconnect()
        held = hearing.gate.try_acquire() is None
        engine.release.set()
        next_job = await hearing.gate.acquire(
            hearing.gate.clock.now() + WAIT_SECONDS, 1
        )
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


def test_a_partial_not_yet_started_when_the_client_leaves_never_runs() -> None:
    async def scenario() -> tuple[int, bool]:
        engine = FakeHearing(hold=True)
        hearing = loaded(engine, FakeClock())
        blocking = hearing.submit(Lease(lambda: None), _transcribe_silence)
        await asyncio.to_thread(engine.entered.wait, WAIT_SECONDS)
        session, _ = await started(hearing)
        await session.on_message(append(audio(0.5)))
        await session.on_tick()
        await session.on_disconnect()
        engine.release.set()
        await asyncio.wrap_future(blocking)
        await asyncio.wrap_future(
            hearing.submit(Lease(lambda: None), _transcribe_silence)
        )
        return len(engine.heard), hearing.gate.try_acquire() is not None

    heard, gate_is_free = asyncio.run(scenario())

    assert heard == BLOCKER_AND_MARKER
    assert gate_is_free


def test_a_result_that_arrives_after_the_session_ended_is_dropped() -> None:
    """WIRE-80, WIRE-81: a result that arrives after the session ended is dropped."""

    async def scenario() -> FakeSocket:
        engine = FakeHearing(hold=True)
        hearing = loaded(engine, FakeClock())
        session, socket = await started(hearing)
        await commit_a_turn(session)
        await asyncio.to_thread(engine.entered.wait, WAIT_SECONDS)
        await session.on_message("not json")
        sent = len(socket.events)
        engine.release.set()
        await asyncio.wrap_future(
            hearing.submit(Lease(lambda: None), _transcribe_silence)
        )
        await session.on_disconnect()
        assert len(socket.events) == sent
        return socket

    socket = asyncio.run(scenario())

    assert socket.of_type(COMPLETED) == []
    assert end_is(socket, "invalid_request", failed=1)


def test_the_poll_runs_every_three_tenths_of_a_second() -> None:
    async def scenario() -> list[float]:
        session, _ = await started(loaded(FakeHearing(), FakeClock()))
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


def test_a_waiting_turn_that_outlasts_its_wait_ends_the_session_as_busy() -> None:
    """WIRE-81, WIRE-82, WIRE-85: a waiting turn past its wait ends it as busy."""

    async def scenario() -> FakeSocket:
        engine = FakeHearing(hold=True)
        clock = FakeClock()
        session, socket = await started(
            loaded(engine, clock), final_wait_seconds=WAIT_SECONDS
        )
        await commit_a_turn(session)
        assert await asyncio.to_thread(engine.entered.wait, WAIT_SECONDS)
        await commit_a_turn(session)
        await clock.until_a_wait_begins()
        clock.advance(2 * WAIT_SECONDS)
        await socket.until_closed()
        engine.release.set()
        await session.on_disconnect()
        return socket

    assert end_is(asyncio.run(scenario()), "busy", failed=2)


def test_audio_past_the_retention_bound_ends_the_session_as_busy() -> None:
    """WIRE-67, WIRE-81: audio past the retention bound ends the session as busy."""

    async def scenario() -> FakeSocket:
        engine = FakeHearing(hold=True)
        session, socket = await started(loaded(engine, FakeClock()))
        await session.on_message(append(audio(THREE_TURNS_SECONDS)))
        engine.release.set()
        await session.on_disconnect()
        return socket

    assert end_is(asyncio.run(scenario()), "busy", failed=3)


def test_a_turn_that_fell_silent_while_one_waits_ends_the_session_as_busy() -> None:
    """WIRE-67, WIRE-81: a turn ending while two wait ends the session as busy."""

    async def scenario() -> FakeSocket:
        engine = FakeHearing(hold=True)
        session, socket = await started(loaded(engine, FakeClock()))
        await commit_a_turn(session)
        await commit_a_turn(session)
        await _one_question(session)
        engine.release.set()
        await session.on_disconnect()
        return socket

    assert end_is(asyncio.run(scenario()), "busy", failed=3)


def test_deltas_hold_back_words_the_decodes_disagree_on() -> None:
    """WIRE-71: deltas hold back words the decodes disagree on."""

    async def scenario() -> FakeSocket:
        engine = FakeHearing("Guten Tag")
        clock = FakeClock()
        session, socket = await started(loaded(engine, clock))
        await session.on_message(append(audio(0.5)))
        for text in ("Guten Tag", "Guten Morgen", "Guten Morgen Welt"):
            engine.text = text
            await session.on_tick()
            await session.settled()
            clock.advance(0.4)
            await session.on_message(append(audio(0.1)))
        return socket

    socket = asyncio.run(scenario())

    assert [event["delta"] for event in socket.of_type(DELTA)] == ["Guten", " Morgen"]


def test_a_partial_that_fails_ends_the_session_as_unavailable() -> None:
    """WIRE-81, WIRE-82: a partial that fails ends the session as unavailable."""

    async def scenario() -> FakeSocket:
        engine = FakeHearing()
        engine.fail_transcription = True
        session, socket = await started(loaded(engine, FakeClock()))
        await session.on_message(append(audio(0.5)))
        await session.on_tick()
        await socket.until_closed()
        return socket

    assert end_is(asyncio.run(scenario()), "unavailable", failed=1)


def test_a_partial_after_the_engine_was_lost_ends_the_session_as_unavailable() -> None:
    """WIRE-81, WIRE-82: a partial after the engine was lost ends it unavailable."""

    async def scenario() -> FakeSocket:
        hearing = loaded(FakeHearing(), FakeClock())
        session, socket = await started(hearing)
        await session.on_message(append(audio(0.5)))
        hearing.close()
        await session.on_tick()
        return socket

    assert end_is(asyncio.run(scenario()), "unavailable", failed=1)


def test_a_partial_that_returns_after_its_turn_ended_adds_no_delta() -> None:
    """WIRE-71: a partial that returns after its turn ended adds no delta."""

    async def scenario() -> FakeSocket:
        engine = FakeHearing(hold=True)
        session, socket = await started(loaded(engine, FakeClock()))
        await session.on_message(append(audio(0.5)))
        await session.on_tick()
        await asyncio.to_thread(engine.entered.wait, WAIT_SECONDS)
        await session.on_message(commit())
        engine.release.set()
        await socket.until(lambda: bool(socket.of_type(COMPLETED)))
        await session.settled()
        return socket

    socket = asyncio.run(scenario())

    assert socket.of_type(DELTA) == []


def _timing_lines(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [
        record.getMessage()
        for record in caplog.records
        if record.getMessage().startswith("live turn completed")
    ]


def test_a_completed_turn_logs_one_line_with_its_timings(
    caplog: pytest.LogCaptureFixture,
) -> None:
    async def scenario() -> str:
        engine = FakeHearing(hold=True)
        clock = FakeClock()
        hearing = loaded(engine, clock)
        session, socket = await started(hearing, final_wait_seconds=WAIT_SECONDS)
        await session.on_message(append(audio(0.5)))
        await session.on_tick()
        clock.jump(0.3)
        engine.release.set()
        await session.settled()
        engine.release.clear()
        engine.entered.clear()
        running = hearing.gate.try_acquire()
        assert running is not None
        await session.on_message(commit())
        await clock.until_a_wait_begins()
        clock.jump(0.2)
        running.close()
        assert await asyncio.to_thread(engine.entered.wait, WAIT_SECONDS)
        clock.jump(0.5)
        engine.release.set()
        await session.settled()
        return str(socket.of_type(COMPLETED)[0]["item_id"])

    caplog.set_level(logging.INFO)

    item_id = asyncio.run(scenario())

    (line,) = _timing_lines(caplog)
    assert line == (
        f"live turn completed item_id={item_id} ended_by=client audio_seconds=0.50"
        " voiced_seconds=0.48 no_speech_prob=none avg_logprob=none"
        " wait_ms=200 decode_ms=500 completed_ms=700"
        " partial_count=1 partial_decode_ms=300"
    )


def test_a_failed_turn_logs_no_timings(caplog: pytest.LogCaptureFixture) -> None:
    async def scenario() -> None:
        engine = FakeHearing()
        engine.fail_transcription = True
        session, socket = await started(loaded(engine, FakeClock()))
        await commit_a_turn(session)
        await socket.until_closed()

    caplog.set_level(logging.INFO)

    asyncio.run(scenario())

    assert _timing_lines(caplog) == []


def test_the_timing_line_holds_none_of_the_heard_text(
    caplog: pytest.LogCaptureFixture,
) -> None:
    async def scenario() -> None:
        session, _ = await started(loaded(FakeHearing(), FakeClock()))
        await commit_a_turn(session)
        await session.settled()

    caplog.set_level(logging.DEBUG)

    asyncio.run(scenario())

    (line,) = _timing_lines(caplog)
    assert HEARD_TEXT not in line
    assert HEARD_TEXT not in caplog.text


def test_the_decode_of_a_turn_is_given_its_voiced_seconds() -> None:
    async def scenario() -> FakeHearing:
        engine = FakeHearing()
        session, _ = await started(loaded(engine, FakeClock()))
        await _one_question(session)
        await session.settled()
        return engine

    engine = asyncio.run(scenario())

    assert engine.heard_voiced[-1] == pytest.approx(SPEECH_WINDOWS * WINDOW_SECONDS)


@pytest.mark.parametrize("text", [HEARD_TEXT, ""], ids=["heard", "judged-no-speech"])
def test_the_timing_line_carries_the_measures_of_the_decode_and_never_its_text(
    caplog: pytest.LogCaptureFixture, text: str
) -> None:
    async def scenario() -> None:
        engine = FakeHearing(text)
        engine.measures = (0.1, -0.3)
        session, _ = await started(loaded(engine, FakeClock()))
        await _one_question(session)
        await session.settled()

    caplog.set_level(logging.INFO)

    asyncio.run(scenario())

    (logged,) = _timing_lines(caplog)
    assert "no_speech_prob=0.100 avg_logprob=-0.300" in logged
    assert "voiced_seconds=0.51" in logged
    assert HEARD_TEXT not in caplog.text

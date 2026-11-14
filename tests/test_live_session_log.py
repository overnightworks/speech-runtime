"""A finished live turn logs its timings and measures, never its text."""

import asyncio
import logging
from typing import TYPE_CHECKING

import pytest

from speech_client.wire import Language
from speech_service.admission_gate import Lease
from speech_service.voice_activity import WINDOW_SECONDS
from tests.support.clock_support import FakeClock
from tests.support.hear_support import HEARD_TEXT, FakeHearing, loaded
from tests.support.live_support import (
    COMPLETED,
    WAIT_SECONDS,
    append,
    audio,
    commit,
    commit_a_turn,
    one_question,
    started,
)

if TYPE_CHECKING:
    from speech_service.hearing import HearingEngine

# Half a second of speech touches 16 windows.
SPEECH_WINDOWS = 16


def _marker(engine: HearingEngine) -> None:
    engine.transcribe(b"\0\0", (Language.DE,))


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
        await one_question(session)
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
        await one_question(session)
        await session.settled()

    caplog.set_level(logging.INFO)

    asyncio.run(scenario())

    (logged,) = _timing_lines(caplog)
    assert "no_speech_prob=0.100 avg_logprob=-0.300" in logged
    assert "voiced_seconds=0.51" in logged
    assert HEARD_TEXT not in caplog.text


def test_a_live_turn_is_decoded_without_the_recognisers_skip_of_no_speech() -> None:
    async def scenario() -> FakeHearing:
        engine = FakeHearing()
        session, _ = await started(loaded(engine, FakeClock()))
        await one_question(session)
        await session.settled()
        return engine

    engine = asyncio.run(scenario())

    assert engine.heard_skipping
    assert not any(engine.heard_skipping)


def test_a_partial_decode_logs_its_measures_and_never_its_text(
    caplog: pytest.LogCaptureFixture,
) -> None:
    async def scenario() -> None:
        engine = FakeHearing()
        engine.measures = (0.2, -0.4)
        session, _ = await started(loaded(engine, FakeClock()))
        await session.on_message(append(audio(0.5)))
        await session.on_tick()
        await session.settled()

    caplog.set_level(logging.INFO)

    asyncio.run(scenario())

    (logged,) = [
        record.getMessage()
        for record in caplog.records
        if record.getMessage().startswith("live partial decoded")
    ]
    assert "no_speech_prob=0.200 avg_logprob=-0.400" in logged
    assert HEARD_TEXT not in caplog.text


def test_a_partial_decoding_when_the_client_leaves_still_logs_its_measures(
    caplog: pytest.LogCaptureFixture,
) -> None:
    async def scenario() -> None:
        engine = FakeHearing(hold=True)
        engine.measures = (0.2, -0.4)
        hearing = loaded(engine, FakeClock())
        session, _ = await started(hearing)
        await session.on_message(append(audio(0.5)))
        await session.on_tick()
        assert await asyncio.to_thread(engine.entered.wait, WAIT_SECONDS)
        await session.on_disconnect()
        engine.release.set()
        # The worker runs one job at a time, so the marker ends after the partial.
        await asyncio.wrap_future(hearing.submit(Lease(lambda: None), _marker))

    caplog.set_level(logging.INFO)

    asyncio.run(scenario())

    (logged,) = [
        record.getMessage()
        for record in caplog.records
        if record.getMessage().startswith("live partial decoded")
    ]
    assert "no_speech_prob=0.200 avg_logprob=-0.400" in logged

"""One owner closes each decode lease as the native call returns or never starts."""

import asyncio
import threading
from typing import TYPE_CHECKING

import pytest

from speech_runtime_client.wire import HearingLanguage
from speech_runtime_service.admission_gate import Lease
from tests.support.hear_support import (
    WAIT_SECONDS,
    FakeHearing,
    hear_app,
    hear_call,
    hear_runtime,
    hearing_gate_is_free,
    hearing_gate_is_taken,
    loaded,
)

if TYPE_CHECKING:
    from speech_runtime_service.hearing import Hearing, HearingEngine

AUDIO = b"\0\0" * 160


def _transcribe(engine: HearingEngine) -> str:
    return engine.transcribe(AUDIO, HearingLanguage.DE)


class _CountedLease:
    """A lease that counts how often its resource was given back."""

    def __init__(self) -> None:
        self.released = 0
        self.lease = Lease(self._release)

    def _release(self) -> None:
        self.released += 1


def test_the_lease_is_closed_when_the_native_call_returns() -> None:
    engine = FakeHearing()
    hearing = loaded(engine)
    counted = _CountedLease()

    text = hearing.submit(counted.lease, _transcribe).result(timeout=WAIT_SECONDS)

    assert text == "Guten Tag"
    assert counted.released == 1


def test_the_lease_is_closed_when_the_native_call_fails() -> None:
    engine = FakeHearing()
    engine.fail_transcription = True
    hearing = loaded(engine)
    counted = _CountedLease()

    future = hearing.submit(counted.lease, _transcribe)
    with pytest.raises(RuntimeError):
        future.result(timeout=WAIT_SECONDS)

    assert counted.released == 1


def test_a_submission_cancelled_before_it_starts_closes_its_lease_and_never_runs() -> (
    None
):
    blocker = FakeHearing(hold=True)
    hearing = loaded(blocker)
    running = _CountedLease()
    waiting = _CountedLease()
    first = hearing.submit(running.lease, _transcribe)
    assert blocker.entered.wait(WAIT_SECONDS)
    second = hearing.submit(waiting.lease, _transcribe)

    cancelled = second.cancel()
    blocker.release.set()
    first.result(timeout=WAIT_SECONDS)

    assert cancelled
    assert (running.released, waiting.released) == (1, 1)
    assert len(blocker.heard) == 1


def test_a_submission_that_cannot_be_made_closes_its_lease() -> None:
    hearing = loaded(FakeHearing())
    hearing.close()
    counted = _CountedLease()

    with pytest.raises(RuntimeError):
        hearing.submit(counted.lease, _transcribe)

    assert counted.released == 1


def test_a_submission_without_an_engine_closes_its_lease() -> None:
    counted = _CountedLease()
    unavailable: Hearing = loaded(None)

    with pytest.raises(RuntimeError):
        unavailable.submit(counted.lease, _transcribe)

    assert counted.released == 1


def test_the_engine_is_freed_only_after_the_running_decode_returns() -> None:
    engine = FakeHearing(hold=True)
    hearing = loaded(engine)
    counted = _CountedLease()
    future = hearing.submit(counted.lease, _transcribe)
    assert engine.entered.wait(WAIT_SECONDS)
    closing = threading.Thread(target=hearing.close)

    late_lease = _CountedLease().lease

    closing.start()
    with pytest.raises(RuntimeError):
        hearing.submit(late_lease, _transcribe)
    freed_while_running = engine.closed
    engine.release.set()
    closing.join(WAIT_SECONDS)

    assert future.result(timeout=WAIT_SECONDS) == "Guten Tag"
    assert (freed_while_running, engine.closed) == (0, 1)


def test_a_file_request_cancelled_before_its_decode_starts_frees_the_gate() -> None:
    async def scenario() -> tuple[bool, bool]:
        engine = FakeHearing(hold=True)
        runtime = hear_runtime(engine)
        blocking = runtime.hearing.submit(_CountedLease().lease, _transcribe)
        await asyncio.to_thread(engine.entered.wait, WAIT_SECONDS)
        call = hear_call(hear_app(runtime))
        await hearing_gate_is_taken(runtime)
        call.task.cancel()
        await asyncio.gather(call.task, return_exceptions=True)
        engine.release.set()
        await asyncio.wrap_future(blocking)
        return hearing_gate_is_free(runtime), len(engine.heard) == 1

    gate_is_free, decoded_only_the_blocker = asyncio.run(scenario())

    assert gate_is_free
    assert decoded_only_the_blocker

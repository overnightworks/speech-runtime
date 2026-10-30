"""The one synthesis at a time: first come first served, no thread per waiter."""

import asyncio
import threading
import time

from speech_runtime_service.synthesis_gate import SynthesisGate

WAIT_SECONDS = 5.0
SHORT_SECONDS = 0.05


def _in(seconds: float) -> float:
    return time.monotonic() + seconds


async def _settle() -> None:
    for _ in range(3):
        await asyncio.sleep(0)


def test_a_free_gate_is_taken_even_past_the_deadline() -> None:
    gate = SynthesisGate()

    lease = asyncio.run(gate.acquire(_in(-1)))

    assert lease is not None


def test_try_acquire_refuses_while_the_gate_is_held_and_serves_after_release() -> None:
    gate = SynthesisGate()
    held = gate.try_acquire()
    assert held is not None

    assert gate.try_acquire() is None
    held.close()

    assert gate.try_acquire() is not None


def test_a_waiter_that_reaches_its_deadline_gets_none() -> None:
    async def scenario() -> tuple[object, bool]:
        gate = SynthesisGate()
        held = await gate.acquire(_in(0))
        assert held is not None
        timed_out = await gate.acquire(_in(SHORT_SECONDS))
        held.close()
        return timed_out, gate.try_acquire() is not None

    timed_out, free_again = asyncio.run(scenario())

    assert timed_out is None
    assert free_again


def test_waiters_are_served_in_the_order_they_came() -> None:
    async def scenario() -> list[str]:
        gate = SynthesisGate()
        first = await gate.acquire(_in(WAIT_SECONDS))
        assert first is not None
        served: list[str] = []

        async def wait(name: str) -> None:
            lease = await gate.acquire(_in(WAIT_SECONDS))
            assert lease is not None
            served.append(name)
            lease.close()

        second = asyncio.create_task(wait("second"))
        await _settle()
        third = asyncio.create_task(wait("third"))
        await _settle()
        first.close()
        await asyncio.wait_for(asyncio.gather(second, third), WAIT_SECONDS)
        return served

    assert asyncio.run(scenario()) == ["second", "third"]


def test_a_release_from_another_thread_serves_the_waiter() -> None:
    async def scenario() -> bool:
        gate = SynthesisGate()
        held = await gate.acquire(_in(0))
        assert held is not None
        releasing = threading.Timer(SHORT_SECONDS, held.close)
        releasing.start()
        lease = await gate.acquire(_in(WAIT_SECONDS))
        releasing.join()
        return lease is not None

    assert asyncio.run(scenario())


def test_a_cancelled_waiter_leaves_the_gate_to_the_next() -> None:
    async def scenario() -> bool:
        gate = SynthesisGate()
        held = await gate.acquire(_in(0))
        assert held is not None
        cancelled = asyncio.create_task(gate.acquire(_in(WAIT_SECONDS)))
        await _settle()
        next_in_line = asyncio.create_task(gate.acquire(_in(WAIT_SECONDS)))
        await _settle()
        cancelled.cancel()
        await _settle()
        held.close()
        lease = await asyncio.wait_for(next_in_line, WAIT_SECONDS)
        return lease is not None

    assert asyncio.run(scenario())


def test_a_waiter_cancelled_after_the_hand_off_is_scheduled_frees_the_gate() -> None:
    async def scenario() -> bool:
        gate = SynthesisGate()
        held = await gate.acquire(_in(0))
        assert held is not None
        waiting = asyncio.create_task(gate.acquire(_in(WAIT_SECONDS)))
        await _settle()
        held.close()  # the hand-off to `waiting` is now scheduled on the loop
        waiting.cancel()
        await asyncio.gather(waiting, return_exceptions=True)
        await _settle()
        return gate.try_acquire() is not None

    assert asyncio.run(scenario())


def test_a_lease_gives_the_gate_back_once() -> None:
    gate = SynthesisGate()
    lease = gate.try_acquire()
    assert lease is not None
    lease.close()
    other = gate.try_acquire()
    assert other is not None

    lease.close()

    assert gate.try_acquire() is None

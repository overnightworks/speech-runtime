"""The gate's live priority: finals go first, and an expired final never starts."""

import asyncio
import time
from types import SimpleNamespace
from typing import TYPE_CHECKING

from speech_runtime_service import admission_gate
from speech_runtime_service.admission_gate import AdmissionGate

if TYPE_CHECKING:
    import pytest

MANY_WAITERS = 100
WAIT_SECONDS = 5.0
SHORT_SECONDS = 0.05


def _in(seconds: float) -> float:
    return time.monotonic() + seconds


async def _settle() -> None:
    for _ in range(3):
        await asyncio.sleep(0)


def test_finals_are_served_in_arrival_order_before_every_waiting_file_job() -> None:
    async def scenario() -> list[str]:
        gate = AdmissionGate()
        running = gate.try_acquire()
        assert running is not None
        served: list[str] = []

        async def file_job() -> None:
            lease = await gate.acquire(_in(WAIT_SECONDS), MANY_WAITERS)
            assert lease is not None
            served.append("file")
            lease.close()

        async def final(name: str) -> None:
            lease = await gate.acquire_final(_in(WAIT_SECONDS))
            assert lease is not None
            served.append(name)
            lease.close()

        tasks = [asyncio.create_task(file_job())]
        await _settle()
        for name in ("first", "second"):
            tasks.append(asyncio.create_task(final(name)))
            await _settle()
        running.close()
        await asyncio.wait_for(asyncio.gather(*tasks), WAIT_SECONDS)
        return served

    assert asyncio.run(scenario()) == ["first", "second", "file"]


def test_a_final_whose_deadline_passed_is_never_started_on_a_free_gate() -> None:
    async def scenario() -> tuple[object, bool]:
        gate = AdmissionGate()
        lease = await gate.acquire_final(_in(-1))
        return lease, gate.try_acquire() is not None

    lease, gate_is_free = asyncio.run(scenario())

    assert lease is None
    assert gate_is_free


def test_a_final_that_expires_in_the_queue_leaves_the_running_lease_alone() -> None:
    async def scenario() -> tuple[object, bool, bool]:
        gate = AdmissionGate()
        running = gate.try_acquire()
        assert running is not None
        expired = await gate.acquire_final(_in(SHORT_SECONDS))
        still_held = gate.try_acquire() is None
        next_job = asyncio.create_task(gate.acquire(_in(WAIT_SECONDS), MANY_WAITERS))
        await _settle()
        running.close()
        served_after = await asyncio.wait_for(next_job, WAIT_SECONDS) is not None
        return expired, still_held, served_after

    expired, still_held, served_after = asyncio.run(scenario())

    assert expired is None
    assert still_held
    assert served_after


def test_a_final_handed_the_gate_after_its_deadline_is_not_started(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    offset = [0.0]
    monkeypatch.setattr(
        admission_gate,
        "time",
        SimpleNamespace(monotonic=lambda: time.monotonic() + offset[0]),
    )

    async def scenario() -> tuple[object, bool]:
        gate = AdmissionGate()
        running = gate.try_acquire()
        assert running is not None
        waiting = asyncio.create_task(gate.acquire_final(_in(WAIT_SECONDS)))
        await _settle()
        running.close()
        offset[0] = 2 * WAIT_SECONDS
        lease = await waiting
        return lease, gate.try_acquire() is not None

    lease, gate_is_free = asyncio.run(scenario())

    assert lease is None
    assert gate_is_free


def test_a_cancelled_final_leaves_the_gate_to_the_next_final() -> None:
    async def scenario() -> bool:
        gate = AdmissionGate()
        running = gate.try_acquire()
        assert running is not None
        cancelled = asyncio.create_task(gate.acquire_final(_in(WAIT_SECONDS)))
        await _settle()
        next_in_line = asyncio.create_task(gate.acquire_final(_in(WAIT_SECONDS)))
        await _settle()
        cancelled.cancel()
        await asyncio.gather(cancelled, return_exceptions=True)
        running.close()
        return await asyncio.wait_for(next_in_line, WAIT_SECONDS) is not None

    assert asyncio.run(scenario())

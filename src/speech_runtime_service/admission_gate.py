"""One job at a time: the gate a resource puts before its heavy work."""

from __future__ import annotations

import asyncio
import threading
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING

from speech_runtime_service.clock import MONOTONIC_CLOCK

if TYPE_CHECKING:
    from collections.abc import Callable

    from speech_runtime_service.clock import Clock


class Lease:
    """A held resource that is given back once, whoever closes it first."""

    def __init__(self, release: Callable[[], None]) -> None:
        """Hold `release` until the first `close`."""
        self._release = release
        self._closed = False
        self._guard = threading.Lock()

    def close(self) -> None:
        """Give the resource back; later calls do nothing."""
        with self._guard:
            if self._closed:
                return
            self._closed = True
            self._release()


@dataclass(frozen=True, slots=True)
class _Waiter:
    loop: asyncio.AbstractEventLoop
    future: asyncio.Future[Lease]


class AdmissionGate:
    """Serves waiters first come first served; a waiter holds no thread.

    A live final queues apart and is served before every waiting job.
    """

    def __init__(self, clock: Clock = MONOTONIC_CLOCK) -> None:
        """Start free; every deadline is on `clock`."""
        self.clock = clock
        self._guard = threading.Lock()
        self._held = False
        self._waiters: deque[_Waiter] = deque()
        self._finals: deque[_Waiter] = deque()

    def try_acquire(self) -> Lease | None:
        """Take the gate at once, or answer `None` while it is held."""
        with self._guard:
            if self._held:
                return None
            self._held = True
        return Lease(self._release)

    async def acquire(self, deadline: float, max_waiters: int) -> Lease | None:
        """Take the gate, waiting until `deadline` (on `clock`) for it.

        A free gate is taken even past the deadline. A held gate with
        `max_waiters` already waiting answers `None` at once. A cancellation or
        a timeout leaves the gate to the next waiter; a gate handed over after
        the deadline passed is given back, never kept.
        """
        return await self._wait(self._waiters, deadline, max_waiters)

    async def acquire_final(self, deadline: float) -> Lease | None:
        """Take the gate for a live final, waiting until `deadline` (on `clock`).

        Finals are served in arrival order, ahead of every `acquire` waiter, and
        are never refused for want of a place. A final whose deadline has passed
        is not started, even on a free gate or when the gate is handed over late.
        """
        if deadline <= self.clock.now():
            return None
        lease = await self._wait(self._finals, deadline, max_waiters=None)
        if lease is not None and deadline <= self.clock.now():
            lease.close()
            return None
        return lease

    async def _wait(
        self, queue: deque[_Waiter], deadline: float, max_waiters: int | None
    ) -> Lease | None:
        loop = asyncio.get_running_loop()
        waiter = _Waiter(loop, loop.create_future())
        with self._guard:
            if not self._held:
                self._held = True
                return Lease(self._release)
            if max_waiters is not None and len(queue) >= max_waiters:
                return None
            queue.append(waiter)
        timer = asyncio.create_task(self.clock.sleep_until(deadline))
        try:
            await asyncio.wait(
                {waiter.future, timer}, return_when=asyncio.FIRST_COMPLETED
            )
        except BaseException:
            self._abandon(waiter)
            raise
        finally:
            timer.cancel()
        if timer.done():
            self._abandon(waiter)
            return None
        return waiter.future.result()

    def _abandon(self, waiter: _Waiter) -> None:
        with self._guard:
            for queue in (self._finals, self._waiters):
                if waiter in queue:
                    queue.remove(waiter)
        if waiter.future.done() and not waiter.future.cancelled():
            waiter.future.result().close()
        else:
            waiter.future.cancel()

    def _release(self) -> None:
        while True:
            with self._guard:
                queue = self._finals or self._waiters
                if not queue:
                    self._held = False
                    return
                waiter = queue.popleft()
            try:
                waiter.loop.call_soon_threadsafe(self._grant, waiter)
            except RuntimeError:
                continue
            return

    def _grant(self, waiter: _Waiter) -> None:
        if waiter.future.done():
            self._release()
        else:
            waiter.future.set_result(Lease(self._release))

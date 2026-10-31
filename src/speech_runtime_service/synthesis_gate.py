"""One synthesis at a time: the gate every speaking request passes."""

from __future__ import annotations

import asyncio
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable


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


class SynthesisGate:
    """Serves waiters first come first served; a waiter holds no thread."""

    def __init__(self) -> None:
        """Start free."""
        self._guard = threading.Lock()
        self._held = False
        self._waiters: deque[_Waiter] = deque()

    def try_acquire(self) -> Lease | None:
        """Take the gate at once, or answer `None` while it is held."""
        with self._guard:
            if self._held:
                return None
            self._held = True
        return Lease(self._release)

    async def acquire(self, deadline: float, max_waiters: int) -> Lease | None:
        """Take the gate, waiting until `deadline` (monotonic) for it.

        A free gate is taken even past the deadline. A held gate with
        `max_waiters` already waiting answers `None` at once. A cancellation or
        a timeout leaves the gate to the next waiter.
        """
        loop = asyncio.get_running_loop()
        waiter = _Waiter(loop, loop.create_future())
        with self._guard:
            if not self._held:
                self._held = True
                return Lease(self._release)
            if len(self._waiters) >= max_waiters:
                return None
            self._waiters.append(waiter)
        try:
            async with asyncio.timeout(deadline - time.monotonic()):
                return await waiter.future
        except TimeoutError:
            self._abandon(waiter)
            return None
        except BaseException:
            self._abandon(waiter)
            raise

    def _abandon(self, waiter: _Waiter) -> None:
        with self._guard:
            if waiter in self._waiters:
                self._waiters.remove(waiter)
        if waiter.future.done() and not waiter.future.cancelled():
            waiter.future.result().close()
        else:
            waiter.future.cancel()

    def _release(self) -> None:
        while True:
            with self._guard:
                if not self._waiters:
                    self._held = False
                    return
                waiter = self._waiters.popleft()
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

"""A clock the test moves: waits end only when the test advances time."""

from __future__ import annotations

import asyncio


class FakeClock:
    """Time that stands still until `advance` moves it; it wakes the waits it passes."""

    def __init__(self) -> None:
        """Start at zero."""
        self._now = 0.0
        self._sleepers: list[tuple[float, asyncio.Future[None]]] = []
        self._someone_waits = asyncio.Event()

    def now(self) -> float:
        """The time now."""
        return self._now

    def advance(self, seconds: float) -> None:
        """Move time forward and wake every wait whose deadline has come."""
        self._now += seconds
        for deadline, sleeper in self._sleepers:
            if deadline <= self._now and not sleeper.done():
                sleeper.set_result(None)

    async def sleep_until(self, deadline: float) -> None:
        """Return once the test has moved time to `deadline`."""
        if deadline <= self._now:
            return
        sleeper = asyncio.get_running_loop().create_future()
        entry = (deadline, sleeper)
        self._sleepers.append(entry)
        self._someone_waits.set()
        try:
            await sleeper
        finally:
            self._sleepers.remove(entry)

    async def until_a_wait_begins(self) -> None:
        """Return once some wait has begun on this clock."""
        await self._someone_waits.wait()

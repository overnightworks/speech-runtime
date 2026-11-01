"""The monotonic clock behind every deadline: the time now and the wait until one."""

from __future__ import annotations

import asyncio
import time
from typing import Final, Protocol


class Clock(Protocol):
    """Seconds on one monotonic scale, and a wait that follows the same scale."""

    def now(self) -> float:
        """The time now, in seconds."""
        ...

    async def sleep_until(self, deadline: float) -> None:
        """Return once `now()` has reached `deadline`; at once if it already has."""
        ...


class LoopClock:
    """The process's monotonic time, waited on through the event loop."""

    def now(self) -> float:
        """The time now, in seconds."""
        return time.monotonic()

    async def sleep_until(self, deadline: float) -> None:
        """Return once `deadline` has passed."""
        await asyncio.sleep(max(0.0, deadline - time.monotonic()))


MONOTONIC_CLOCK: Final = LoopClock()

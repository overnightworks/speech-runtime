"""A live session over a fake socket: what it sent, and audio to feed it."""

from __future__ import annotations

import asyncio
import base64
import json
import struct
from typing import TYPE_CHECKING

from speech_runtime_service.live_session import LiveSession

if TYPE_CHECKING:
    from collections.abc import Callable

    from speech_runtime_service.hearing import Hearing

SPEECH_LEVEL = 5000
INPUT_RATE = 24_000
WAIT_SECONDS = 5.0


class FakeSocket:
    """The events a session sent, parsed, and whether it closed the socket."""

    def __init__(self) -> None:
        """Start empty and open."""
        self.events: list[dict[str, object]] = []
        self.closed = False
        self._changed = asyncio.Condition()

    async def send(self, text: str) -> None:
        """Record one event."""
        async with self._changed:
            self.events.append(json.loads(text))
            self._changed.notify_all()

    async def close(self) -> None:
        """Record the close."""
        async with self._changed:
            self.closed = True
            self._changed.notify_all()

    async def until(self, ready: Callable[[], bool]) -> None:
        """Wait, without polling, until `ready()` holds."""
        async with asyncio.timeout(WAIT_SECONDS):
            async with self._changed:
                await self._changed.wait_for(ready)

    async def until_closed(self) -> None:
        """Wait until the session closed the socket."""
        await self.until(lambda: self.closed)

    def types(self) -> list[object]:
        """The `type` of every event sent, in order."""
        return [event["type"] for event in self.events]

    def of_type(self, kind: str) -> list[dict[str, object]]:
        """The events of one `type`, in order."""
        return [event for event in self.events if event["type"] == kind]


class Clock:
    """A clock the test moves."""

    def __init__(self) -> None:
        """Start at zero."""
        self.now = 0.0

    def __call__(self) -> float:
        """The time now."""
        return self.now


def audio(seconds: float, level: int = SPEECH_LEVEL) -> str:
    """The base64 PCM16 at 24 kHz of a constant level `seconds` long."""
    count = int(seconds * INPUT_RATE)
    return base64.b64encode(struct.pack("<h", level) * count).decode()


def append(data: str, event_id: str | None = None) -> str:
    """An `input_audio_buffer.append` message."""
    event: dict[str, object] = {"type": "input_audio_buffer.append", "audio": data}
    if event_id:
        event["event_id"] = event_id
    return json.dumps(event)


def commit(event_id: str | None = None) -> str:
    """An `input_audio_buffer.commit` message."""
    event: dict[str, object] = {"type": "input_audio_buffer.commit"}
    if event_id:
        event["event_id"] = event_id
    return json.dumps(event)


def update(language: str = "de") -> str:
    """A `session.update` message for `language`."""
    audio_input = {"transcription": {"language": language}}
    session = {"type": "transcription", "audio": {"input": audio_input}}
    return json.dumps({"type": "session.update", "session": session})


async def started(
    hearing: Hearing, *, final_wait_seconds: float = 30.0
) -> tuple[LiveSession, FakeSocket, Clock]:
    """A session over `hearing` that has sent `session.created`."""
    socket = FakeSocket()
    clock = Clock()
    session = LiveSession(
        hearing,
        final_wait_seconds=final_wait_seconds,
        send=socket.send,
        close=socket.close,
        clock=clock,
    )
    await session.start()
    return session, socket, clock

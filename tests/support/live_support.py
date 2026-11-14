"""A live session over a fake socket: what it sent, and audio to feed it."""

from __future__ import annotations

import asyncio
import base64
import json
import struct
from typing import TYPE_CHECKING

from speech_service.live_session import LiveSession

if TYPE_CHECKING:
    from collections.abc import Callable

    from speech_service.hearing import Hearing

SPEECH_LEVEL = 5000
INPUT_RATE = 24_000
WAIT_SECONDS = 5.0
COMPLETED = "conversation.item.input_audio_transcription.completed"
FAILED = "conversation.item.input_audio_transcription.failed"


class FakeSocket:
    """The events a session sent, parsed, and whether it closed the socket."""

    def __init__(self) -> None:
        """Start empty and open."""
        self.events: list[dict[str, object]] = []
        self.closed = False
        self.close_code: int | None = None
        self._changed = asyncio.Condition()

    async def send(self, text: str) -> None:
        """Record one event."""
        async with self._changed:
            self.events.append(json.loads(text))
            self._changed.notify_all()

    async def close(self, code: int) -> None:
        """Record the close and its code."""
        async with self._changed:
            self.closed = True
            self.close_code = code
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


class StallingSocket(FakeSocket):
    """A socket whose send of one event of one `type` stalls until released."""

    def __init__(self, kind: str, occurrence: int = 1) -> None:
        """Stall the `occurrence`th event of `kind`."""
        super().__init__()
        self._kind = kind
        self._skipped = occurrence - 1
        self.stalled = asyncio.Event()
        self.proceed = asyncio.Event()

    async def send(self, text: str) -> None:
        """Record the event, after the stall if it is the stalled kind."""
        if not self.stalled.is_set() and json.loads(text)["type"] == self._kind:
            if self._skipped:
                self._skipped -= 1
            else:
                self.stalled.set()
                await self.proceed.wait()
        await super().send(text)


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


def update(
    language: str | None = "de",
    model: str | None = None,
    languages: list[str] | None = None,
) -> str:
    """A `session.update` message for `language` or `languages`, `model` if given."""
    transcription: dict[str, object] = {}
    if language is not None:
        transcription["language"] = language
    if languages is not None:
        transcription["languages"] = languages
    if model is not None:
        transcription["model"] = model
    audio_input = {"transcription": transcription}
    session = {"type": "transcription", "audio": {"input": audio_input}}
    return json.dumps({"type": "session.update", "session": session})


async def started(
    hearing: Hearing,
    *,
    final_wait_seconds: float = 30.0,
    socket: FakeSocket | None = None,
) -> tuple[LiveSession, FakeSocket]:
    """A session over `hearing` that has sent `session.created`."""
    socket = socket or FakeSocket()
    session = LiveSession(
        hearing,
        final_wait_seconds=final_wait_seconds,
        send=socket.send,
        close=socket.close,
    )
    await session.start()
    return session, socket


async def commit_a_turn(session: LiveSession, seconds: float = 0.3) -> None:
    """Feed `seconds` of speech, then commit it."""
    await session.on_message(append(audio(seconds)))
    await session.on_message(commit())


async def one_question(session: LiveSession) -> None:
    """Half a second of speech, then a silence that ends the turn, and a tick."""
    await session.on_message(append(audio(0.5)))
    await session.on_message(append(audio(0.8, 0)))
    await session.on_tick()


def end_is(socket: FakeSocket, code: str, failed: int) -> bool:
    """Whether the session failed `failed` items, sent `error` `code` and closed."""
    types = socket.types()
    error = socket.of_type("error")[0]["error"]
    assert isinstance(error, dict)
    return (
        types[-1] == "error"
        and types.count(FAILED) == failed
        and types[-1 - failed : -1] == [FAILED] * failed
        and error["code"] == code
        and socket.closed
    )

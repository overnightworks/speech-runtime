"""One live hearing session: its turns, its items' terminals and how it ends.

The session is the one owner of every item's terminal event. All its state moves
under one lock, so a message, a poll and a decode's result never interleave; a
decode runs in the hearing worker and its result comes back as a task.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

from speech_runtime_client.wire import (
    AudioAppend,
    AudioCommit,
    AudioCommitted,
    ClientEventError,
    DurationUsage,
    HearingLanguage,
    ItemCreated,
    RefusalCode,
    SessionCreated,
    SessionUpdate,
    SessionUpdated,
    SpeechStarted,
    SpeechStopped,
    TranscriptCompleted,
    TranscriptDelta,
    TranscriptFailed,
    TranscriptionSession,
    UserMessageItem,
    error_event,
    item_failure,
    new_item_id,
    parse_client_event,
)
from speech_runtime_service.failures import log_failure
from speech_runtime_service.hearing_session import (
    POLL_WAIT_SECONDS,
    HearingSession,
    SpeechStart,
    TurnEnd,
)
from speech_runtime_service.pcm import BYTES_PER_SAMPLE, Resampler

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Coroutine, Sequence
    from concurrent.futures import Future

    from pydantic import BaseModel

    from speech_runtime_service.hearing import Hearing, HearingEngine

logger = logging.getLogger(__name__)

__all__ = ["POLL_WAIT_SECONDS", "LiveSession", "poll_forever"]

_PCM_BYTES_PER_SECOND: Final = 16_000 * BYTES_PER_SAMPLE


@dataclass(frozen=True, slots=True)
class _Final:
    item_id: str
    pcm: bytes
    deadline: float


def _samples_of(encoded: str) -> bytes | None:
    """The PCM16 in `encoded`, or `None` for invalid base64 or an odd byte count."""
    try:
        raw = base64.b64decode(encoded, validate=True)
    except binascii.Error:
        return None
    return None if len(raw) % BYTES_PER_SAMPLE else raw


class LiveSession:
    """The state machine of one Realtime transcription session."""

    def __init__(
        self,
        hearing: Hearing,
        *,
        final_wait_seconds: float,
        send: Callable[[str], Awaitable[None]],
        close: Callable[[], Awaitable[None]],
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Hear through `hearing`; send events with `send` and end with `close`.

        A final may wait `final_wait_seconds` from its turn's end; `clock` paces
        the partials.
        """
        self._hearing = hearing
        self._final_wait_seconds = final_wait_seconds
        self._send = send
        self._close = close
        self._scheduler = HearingSession(clock)
        self._resampler = Resampler()
        self._lock = asyncio.Lock()
        self._tasks: set[asyncio.Task[None]] = set()
        self._language = HearingLanguage.DE
        self._configured = False
        self._audio_seen = False
        self._ended = False
        self._open_item: str | None = None
        self._last_item: str | None = None
        self._unfinished: dict[str, None] = {}
        self._turn_number = 0
        self._previous_words: list[str] = []
        self._sent_words = 0
        self._final_active = False
        self._waiting: _Final | None = None

    async def start(self) -> None:
        """Send `session.created`, the first event of every session."""
        await self._emit(SessionCreated(session=TranscriptionSession()))

    async def on_message(self, text: str | None) -> None:
        """Take one client message; `None` is a binary frame, which is refused."""
        async with self._lock:
            if self._ended:
                return
            try:
                event = parse_client_event(text or "")
            except ClientEventError as error:
                await self._end_locked(
                    RefusalCode.INVALID_REQUEST, error.param, error.event_id
                )
                return
            match event:
                case SessionUpdate():
                    await self._on_update(event)
                case AudioAppend():
                    await self._on_append(event)
                case AudioCommit():
                    await self._on_commit(event)

    async def on_tick(self) -> None:
        """End a turn that fell silent, and decode a partial when one is due."""
        async with self._lock:
            if self._ended:
                return
            await self._apply(self._scheduler.poll())
            if not self._ended:
                await self._decode_partial()

    async def on_disconnect(self) -> None:
        """The client left: cancel what is outstanding and send nothing."""
        self._ended = True
        self._cancel_decodes()
        await self.settled()

    async def settled(self) -> None:
        """Wait until every decode task has finished or been cancelled."""
        while pending := [task for task in self._tasks if not task.done()]:
            await asyncio.wait(pending)

    async def _decode_partial(self) -> None:
        pcm = self._scheduler.partial_due()
        if pcm is None:
            return
        lease = self._hearing.gate.try_acquire()
        if lease is None:
            return
        try:
            future = self._hearing.submit(lease, self._transcription(pcm))
        except RuntimeError:
            await self._end_locked(RefusalCode.UNAVAILABLE)
            return
        self._scheduler.partial_started()
        self._spawn(self._run_partial(future, self._turn_number))

    async def _on_update(self, event: SessionUpdate) -> None:
        if self._configured or self._audio_seen:
            await self._end_locked(
                RefusalCode.INVALID_REQUEST, "session", event.event_id
            )
            return
        self._configured = True
        self._language = event.session.audio.input.transcription.language
        await self._emit(SessionUpdated(session=event.session))

    async def _on_append(self, event: AudioAppend) -> None:
        raw = _samples_of(event.audio)
        if raw is None:
            await self._end_locked(RefusalCode.INVALID_REQUEST, "audio", event.event_id)
            return
        self._audio_seen = True
        await self._apply(self._scheduler.add(self._resampler.feed(raw)))

    async def _on_commit(self, event: AudioCommit) -> None:
        turn = self._scheduler.commit()
        if turn is None:
            await self._end_locked(RefusalCode.INVALID_REQUEST, None, event.event_id)
            return
        await self._finish_turn(turn)

    async def _apply(self, events: Sequence[SpeechStart | TurnEnd]) -> None:
        for event in events:
            if self._ended:
                return
            if isinstance(event, SpeechStart):
                await self._emit(
                    SpeechStarted(
                        item_id=self._open_item_id(), audio_start_ms=event.audio_ms
                    )
                )
            else:
                await self._finish_turn(event)

    def _open_item_id(self) -> str:
        if self._open_item is None:
            self._open_item = new_item_id()
            self._unfinished[self._open_item] = None
        return self._open_item

    async def _finish_turn(self, turn: TurnEnd) -> None:
        item_id = self._open_item_id()
        self._open_item = None
        self._turn_number += 1
        self._previous_words = []
        self._sent_words = 0
        if not turn.by_client:
            await self._emit(SpeechStopped(item_id=item_id, audio_end_ms=turn.end_ms))
        previous, self._last_item = self._last_item, item_id
        await self._emit(AudioCommitted(item_id=item_id, previous_item_id=previous))
        await self._emit(
            ItemCreated(previous_item_id=previous, item=UserMessageItem(id=item_id))
        )
        final = _Final(item_id, turn.pcm, time.monotonic() + self._final_wait_seconds)
        if not self._final_active:
            self._start_final(final)
        elif self._waiting is None:
            self._waiting = final
        else:
            await self._end_locked(RefusalCode.BUSY)

    def _start_final(self, final: _Final) -> None:
        self._final_active = True
        self._spawn(self._run_final(final))

    async def _run_final(self, final: _Final) -> None:
        lease = await self._hearing.gate.acquire_final(final.deadline)
        if lease is None:
            await self._end(RefusalCode.BUSY)
            return
        try:
            text = await asyncio.wrap_future(
                self._hearing.submit(lease, self._transcription(final.pcm))
            )
        except Exception as error:  # noqa: BLE001 - logged by type and never by message
            log_failure(logger, "hearing failed", error)
            await self._end(RefusalCode.UNAVAILABLE)
            return
        async with self._lock:
            if self._ended:
                return
            await self._complete(final, text)

    async def _complete(self, final: _Final, text: str) -> None:
        seconds = len(final.pcm) / _PCM_BYTES_PER_SECOND
        del self._unfinished[final.item_id]
        await self._emit(
            TranscriptCompleted(
                item_id=final.item_id,
                transcript=text,
                usage=DurationUsage(seconds=seconds),
            )
        )
        self._final_active = False
        waiting, self._waiting = self._waiting, None
        if waiting is not None:
            self._start_final(waiting)

    async def _run_partial(self, future: Future[str], turn_number: int) -> None:
        try:
            text = await asyncio.wrap_future(future)
        except Exception as error:  # noqa: BLE001 - logged by type and never by message
            log_failure(logger, "hearing failed", error)
            await self._end(RefusalCode.UNAVAILABLE)
            return
        async with self._lock:
            if self._ended or turn_number != self._turn_number:
                return
            await self._emit_agreed(text)

    async def _emit_agreed(self, text: str) -> None:
        words = text.split()
        agreed = 0
        for before, now in zip(self._previous_words, words, strict=False):
            if before != now:
                break
            agreed += 1
        self._previous_words = words
        if agreed <= self._sent_words:
            return
        piece = " ".join(words[self._sent_words : agreed])
        delta = f" {piece}" if self._sent_words else piece
        self._sent_words = agreed
        await self._emit(TranscriptDelta(item_id=self._open_item_id(), delta=delta))

    def _transcription(self, pcm: bytes) -> Callable[[HearingEngine], str]:
        language = self._language

        def job(engine: HearingEngine) -> str:
            return engine.transcribe(pcm, language)

        return job

    async def _end(self, code: RefusalCode) -> None:
        async with self._lock:
            await self._end_locked(code)

    async def _end_locked(
        self,
        code: RefusalCode,
        param: str | None = None,
        event_id: str | None = None,
    ) -> None:
        """Fail every announced unfinished item, then send `error`, then close."""
        if self._ended:
            return
        self._ended = True
        self._cancel_decodes()
        failure = item_failure(code)
        for item_id in self._unfinished:
            await self._emit(TranscriptFailed(item_id=item_id, error=failure))
        self._unfinished.clear()
        await self._emit(error_event(code, param, event_id))
        await self._close()

    def _cancel_decodes(self) -> None:
        for task in self._tasks:
            if task is not asyncio.current_task():
                task.cancel()

    def _spawn(self, work: Coroutine[object, object, None]) -> None:
        task = asyncio.create_task(work)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _emit(self, event: BaseModel) -> None:
        await self._send(event.model_dump_json())


async def poll_forever(
    session: LiveSession,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> None:
    """Poll `session` every C2 interval until cancelled."""
    while True:
        await sleep(POLL_WAIT_SECONDS)
        await session.on_tick()

"""C2's scheduling of a live session's audio: turns and partials.

Pure and synchronous: audio goes in, events and snapshots come out, and the clock
is injected. Decoding, transport and the item lifecycle belong to the caller.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

from speech_runtime_service.hearing import HEARING_SAMPLE_RATE
from speech_runtime_service.pcm import BYTES_PER_SAMPLE, is_silence

if TYPE_CHECKING:
    from collections.abc import Callable

PARTIAL_PERIOD_SECONDS: Final = 0.4
MIN_AUDIO_SECONDS: Final = 0.4
SILENCE_SECONDS: Final = 0.7
SPEECH_RMS: Final = 0.02
POLL_WAIT_SECONDS: Final = 0.3
MAX_TURN_SECONDS: Final = 30.0
_BYTES_PER_SECOND: Final = HEARING_SAMPLE_RATE * BYTES_PER_SAMPLE
_BYTES_PER_MILLISECOND: Final = _BYTES_PER_SECOND // 1000
_MAX_TURN_BYTES: Final = int(MAX_TURN_SECONDS * _BYTES_PER_SECOND)
_MIN_PARTIAL_BYTES: Final = int(MIN_AUDIO_SECONDS * _BYTES_PER_SECOND)
_SILENT_TAIL_BYTES: Final = int(SILENCE_SECONDS * _BYTES_PER_SECOND)


@dataclass(frozen=True, slots=True)
class SpeechStart:
    """C2 first heard speech in the open turn, `audio_ms` into the session."""

    audio_ms: int


@dataclass(frozen=True, slots=True)
class TurnEnd:
    """A turn's 16 kHz audio and where it lay in the session, in milliseconds."""

    pcm: bytes
    start_ms: int
    end_ms: int
    spoken: bool
    by_client: bool


class HearingSession:
    """One session's open turn, and the schedule of its partials."""

    def __init__(self, clock: Callable[[], float]) -> None:
        """Count time with `clock`, in seconds."""
        self._clock = clock
        self._turn_start_ms = 0
        self._pcm = bytearray()
        self._heard_speech = False
        self._last_partial_at: float | None = None
        self._last_partial_bytes = 0

    def add(self, pcm: bytes) -> list[SpeechStart | TurnEnd]:
        """Take 16 kHz mono 16-bit audio; a turn that reaches the cap ends there."""
        events: list[SpeechStart | TurnEnd] = []
        while pcm:
            room = _MAX_TURN_BYTES - len(self._pcm)
            chunk, pcm = pcm[:room], pcm[room:]
            events += self._append(chunk)
            if len(self._pcm) >= _MAX_TURN_BYTES:
                events += self._end(by_client=False)
        return events

    def poll(self) -> list[TurnEnd]:
        """End the turn once speech was heard and the last 0.7 s are silent."""
        if not self._heard_speech or len(self._pcm) < _SILENT_TAIL_BYTES:
            return []
        if not is_silence(bytes(self._pcm[-_SILENT_TAIL_BYTES:]), SPEECH_RMS):
            return []
        return self._end(by_client=False)

    def commit(self) -> TurnEnd | None:
        """End the open turn at the client's word; `None` when it holds no audio."""
        ended = self._end(by_client=True)
        return ended[0] if ended else None

    def partial_due(self) -> bytes | None:
        """The open turn's audio when a partial is due, else `None`."""
        if not self._heard_speech or len(self._pcm) < _MIN_PARTIAL_BYTES:
            return None
        if len(self._pcm) == self._last_partial_bytes:
            return None
        last = self._last_partial_at
        if last is not None and self._clock() - last < PARTIAL_PERIOD_SECONDS:
            return None
        return bytes(self._pcm)

    def partial_started(self) -> None:
        """Note that the caller decoded the audio `partial_due` gave."""
        self._last_partial_at = self._clock()
        self._last_partial_bytes = len(self._pcm)

    def _append(self, chunk: bytes) -> list[SpeechStart]:
        offset_ms = self._turn_start_ms + len(self._pcm) // _BYTES_PER_MILLISECOND
        self._pcm.extend(chunk)
        if self._heard_speech or is_silence(chunk, SPEECH_RMS):
            return []
        self._heard_speech = True
        return [SpeechStart(audio_ms=offset_ms)]

    def _end(self, *, by_client: bool) -> list[TurnEnd]:
        pcm = bytes(self._pcm)
        start_ms = self._turn_start_ms
        end_ms = start_ms + len(pcm) // _BYTES_PER_MILLISECOND
        spoken = self._heard_speech
        self._turn_start_ms = end_ms
        self._pcm.clear()
        self._heard_speech = False
        self._last_partial_at = None
        self._last_partial_bytes = 0
        if not pcm or not (spoken or by_client):
            return []
        return [TurnEnd(pcm, start_ms, end_ms, spoken, by_client)]

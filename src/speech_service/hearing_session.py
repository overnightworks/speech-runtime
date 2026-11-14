"""C2's scheduling of a live session's audio: turns and partials.

Pure and synchronous: audio goes in, events and snapshots come out, the detector
says which windows of the audio are speech, and the clock is injected. Decoding,
transport and the item lifecycle belong to the caller.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

from speech_service.voice_activity import (
    PRE_ROLL_SECONDS,
    SAMPLE_RATE,
    START_THRESHOLD,
    START_WINDOWS,
    WINDOW_SAMPLES,
    WINDOW_SECONDS,
    is_voiced,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from speech_service.voice_activity import SpeechDetector

PARTIAL_PERIOD_SECONDS: Final = 0.4
MIN_AUDIO_SECONDS: Final = 0.4
SILENCE_SECONDS: Final = 0.5
POLL_WAIT_SECONDS: Final = 0.3
MAX_TURN_SECONDS: Final = 30.0
_BYTES_PER_SAMPLE: Final = 2
_SAMPLES_PER_MILLISECOND: Final = SAMPLE_RATE // 1000
_MAX_TURN_SAMPLES: Final = int(MAX_TURN_SECONDS * SAMPLE_RATE)
_MIN_PARTIAL_SAMPLES: Final = int(MIN_AUDIO_SECONDS * SAMPLE_RATE)
_PRE_ROLL_SAMPLES: Final = int(PRE_ROLL_SECONDS * SAMPLE_RATE)
_SILENT_WINDOWS: Final = math.ceil(SILENCE_SECONDS * SAMPLE_RATE / WINDOW_SAMPLES)


@dataclass(frozen=True, slots=True)
class SpeechStart:
    """C2 first heard speech in the open turn, `audio_ms` into the session."""

    audio_ms: int


@dataclass(frozen=True, slots=True)
class TurnEnd:
    """A turn's 16 kHz audio, where it lay in the session and how much of it was speech.

    The audio starts a pre-roll before the speech; the milliseconds count the
    session's audio, the dropped audio between turns included.
    """

    pcm: bytes
    start_ms: int
    end_ms: int
    voiced_seconds: float
    by_client: bool
    by_drain: bool = False


@dataclass(frozen=True, slots=True)
class OpenTurn:
    """The open turn's audio so far and how much of it was speech."""

    pcm: bytes
    voiced_seconds: float


class HearingSession:
    """One session's open turn, and the schedule of its partials.

    A turn opens once `START_WINDOWS` windows in a row are speech and ends after
    `SILENCE_SECONDS` of windows that are not, or at the cap. Between turns only the
    last pre-roll of audio is held; its position stays counted.
    """

    def __init__(self, clock: Callable[[], float], detector: SpeechDetector) -> None:
        """Count time with `clock`, in seconds; read the speech from `detector`."""
        self._clock = clock
        self._detector = detector
        self._pcm = bytearray()
        self._held_from = 0
        self._windows = 0
        self._speaking = False
        self._speech_windows = 0
        self._run_start = 0
        self._silent_windows = 0
        self._voiced_windows = 0
        self._waiting_voiced: deque[int] = deque()
        self._last_partial_at: float | None = None
        self._last_partial_samples = 0
        self._last_turn = False

    def add(self, pcm: bytes) -> list[SpeechStart | TurnEnd]:
        """Take 16 kHz mono 16-bit audio; a turn that reaches the cap ends there."""
        self._pcm.extend(pcm)
        events: list[SpeechStart | TurnEnd] = []
        for probability in self._detector.probabilities(pcm):
            events += self._take_window(probability)
        return events

    def commit(self) -> TurnEnd | None:
        """End the open turn at the client's word; `None` when it holds no audio."""
        ended = self._end(self._held_end, by_client=True)
        return ended[0] if ended else None

    @property
    def turn_open(self) -> bool:
        """Whether a turn is open or its first speech windows are being confirmed."""
        return self._speaking or self._speech_windows > 0

    def stop_opening_turns(self) -> None:
        """Open no turn after this call; a turn already open or confirming runs on."""
        self._last_turn = True

    def cut(self) -> TurnEnd | None:
        """End the open turn for a drain; `None` when it was still being confirmed."""
        if not self._speaking:
            self._speech_windows = 0
            return None
        return self._end(self._held_end, by_client=False, by_drain=True)[0]

    def partial_due(self) -> OpenTurn | None:
        """The open turn's audio when a partial is due, else `None`."""
        samples = len(self._pcm) // _BYTES_PER_SAMPLE
        if not self._speaking or samples < _MIN_PARTIAL_SAMPLES:
            return None
        if samples == self._last_partial_samples:
            return None
        last = self._last_partial_at
        if last is not None and self._clock() - last < PARTIAL_PERIOD_SECONDS:
            return None
        return OpenTurn(bytes(self._pcm), self._voiced_windows * WINDOW_SECONDS)

    def partial_started(self) -> None:
        """Note that the caller decoded the audio `partial_due` gave."""
        self._last_partial_at = self._clock()
        self._last_partial_samples = len(self._pcm) // _BYTES_PER_SAMPLE

    @property
    def _held_end(self) -> int:
        return self._held_from + len(self._pcm) // _BYTES_PER_SAMPLE

    def _take_window(self, probability: float) -> list[SpeechStart | TurnEnd]:
        start = self._windows * WINDOW_SAMPLES
        self._windows += 1
        if self._speaking:
            return self._continue_turn(probability, start + WINDOW_SAMPLES)
        if is_voiced(probability):
            self._waiting_voiced.append(start)
        return self._wait_for_speech(probability, start)

    def _wait_for_speech(
        self, probability: float, start: int
    ) -> list[SpeechStart | TurnEnd]:
        if probability < START_THRESHOLD or (
            self._last_turn and not self._speech_windows
        ):
            self._speech_windows = 0
        else:
            if not self._speech_windows:
                self._run_start = start
            self._speech_windows += 1
            if self._speech_windows >= START_WINDOWS:
                return self._open_turn()
        newest = self._run_start if self._speech_windows else start + WINDOW_SAMPLES
        self._drop_before(newest - _PRE_ROLL_SAMPLES)
        return []

    def _open_turn(self) -> list[SpeechStart | TurnEnd]:
        self._drop_before(self._run_start - _PRE_ROLL_SAMPLES)
        self._speaking = True
        self._voiced_windows = len(self._waiting_voiced)
        self._waiting_voiced.clear()
        self._silent_windows = 0
        self._speech_windows = 0
        return [
            SpeechStart(
                audio_ms=max(self._run_start, self._held_from)
                // _SAMPLES_PER_MILLISECOND
            )
        ]

    def _continue_turn(self, probability: float, end: int) -> list[TurnEnd]:
        if is_voiced(probability):
            self._voiced_windows += 1
            self._silent_windows = 0
        else:
            self._silent_windows += 1
        if end - self._held_from >= _MAX_TURN_SAMPLES:
            return self._end(self._held_from + _MAX_TURN_SAMPLES, by_client=False)
        if self._silent_windows >= _SILENT_WINDOWS:
            return self._end(end, by_client=False)
        return []

    def _drop_before(self, sample: int) -> None:
        dropped = min(sample, self._held_end) - self._held_from
        if dropped > 0:
            del self._pcm[: dropped * _BYTES_PER_SAMPLE]
            self._held_from += dropped
        while self._waiting_voiced and self._waiting_voiced[0] < self._held_from:
            self._waiting_voiced.popleft()

    def _end(
        self, end: int, *, by_client: bool, by_drain: bool = False
    ) -> list[TurnEnd]:
        start = self._held_from
        size = (end - start) * _BYTES_PER_SAMPLE
        pcm = bytes(self._pcm[:size])
        voiced_seconds = (
            self._voiced_windows + len(self._waiting_voiced)
        ) * WINDOW_SECONDS
        del self._pcm[:size]
        self._held_from = end
        self._speaking = False
        self._speech_windows = 0
        self._silent_windows = 0
        self._voiced_windows = 0
        self._waiting_voiced.clear()
        self._last_partial_at = None
        self._last_partial_samples = 0
        if not pcm:
            return []
        return [
            TurnEnd(
                pcm,
                start // _SAMPLES_PER_MILLISECOND,
                end // _SAMPLES_PER_MILLISECOND,
                voiced_seconds,
                by_client,
                by_drain,
            )
        ]

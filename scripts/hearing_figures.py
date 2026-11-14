"""The pure figures of the hearing stage proof: timings, word errors, clips, record.

Nothing here starts a process or reads a clock; `prove_hearing` drives the
service and hands the observations to these functions. The clip functions hear
a clip through the service's own turn detection, with the shipped Silero model.
"""

from __future__ import annotations

import functools
import io
import re
import statistics
import wave
from dataclasses import asdict, dataclass
from enum import StrEnum
from http import HTTPStatus
from typing import TYPE_CHECKING, Final

import numpy as np

from speech_client.wire import Language, RefusalCode
from speech_service.hearing_session import (
    MAX_TURN_SECONDS,
    SILENCE_SECONDS,
    HearingSession,
    TurnEnd,
)
from speech_service.pcm import BYTES_PER_SAMPLE, LIVE_INPUT_RATE, Resampler
from speech_service.voice_activity import (
    PRE_ROLL_SECONDS,
    SAMPLE_RATE,
    WINDOW_SAMPLES,
    WINDOW_SECONDS,
    SileroDetectors,
    is_voiced,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Mapping, Sequence

MIN_RUNS: Final = 5
MIN_GROWING_DELTAS: Final = 2
FRAME_SECONDS: Final = 0.1
FRAME_BYTES: Final = int(LIVE_INPUT_RATE * FRAME_SECONDS) * BYTES_PER_SAMPLE
TRAILING_SILENCE_SECONDS: Final = 2.0
PARTIAL_FIGURES: Final = frozenset(
    {"first_partial_from_start", "first_partial_after_end"}
)


@dataclass(frozen=True)
class LiveTurn:
    """What a client saw of one turn; every time counts from its first audio."""

    question_seconds: float
    delta_seconds: tuple[float, ...]
    deltas: tuple[str, ...]
    completed_seconds: float
    transcript: str
    language: Language


@dataclass(frozen=True)
class Timing:
    """The first partial and the final of one turn, from its start and its end.

    A time after the question's end is negative when the partial came during
    the question. The first partial is `None` for a turn that had no delta.
    """

    first_partial_from_start: float | None
    first_partial_after_end: float | None
    final_from_start: float
    final_after_end: float


def timing_of(turn: LiveTurn) -> Timing:
    """The turn's first partial and final against its start and its end."""
    first = turn.delta_seconds[0] if turn.delta_seconds else None
    return Timing(
        first_partial_from_start=first,
        first_partial_after_end=None
        if first is None
        else first - turn.question_seconds,
        final_from_start=turn.completed_seconds,
        final_after_end=turn.completed_seconds - turn.question_seconds,
    )


def _median_final_after_end(turns: Sequence[LiveTurn]) -> float:
    return statistics.median(timing_of(turn).final_after_end for turn in turns)


def _added_seconds(single: Sequence[LiveTurn], two: Sequence[LiveTurn]) -> float:
    return _median_final_after_end(two) - _median_final_after_end(single)


def language_step_record(
    single: Sequence[LiveTurn], two: Sequence[LiveTurn]
) -> dict[str, object]:
    """The final's median time after the turn's end in a two-language session.

    Set against the same turns in a one-language session. Reported, not judged:
    each series also carries its lowest and highest final after the end, since
    one run's final moves by about a second and a median alone hides that.
    """
    single_finals = [timing_of(turn).final_after_end for turn in single]
    two_finals = [timing_of(turn).final_after_end for turn in two]
    return {
        "runs": len(two),
        "single_final_after_end": statistics.median(single_finals),
        "single_final_lowest_after_end": min(single_finals),
        "single_final_highest_after_end": max(single_finals),
        "two_final_after_end": statistics.median(two_finals),
        "two_final_lowest_after_end": min(two_finals),
        "two_final_highest_after_end": max(two_finals),
        "added_seconds": _added_seconds(single, two),
        "chosen": sorted({turn.language.value for turn in two}),
        "heard": sorted({turn.transcript for turn in two}),
    }


def count_growing_deltas(deltas: Sequence[str]) -> int:
    """The deltas that add words; each one grows the provisional text."""
    return sum(1 for delta in deltas if delta.strip())


def partial_runs(timings: Sequence[Timing]) -> int:
    """How many of the runs had a first partial."""
    return sum(1 for timing in timings if timing.first_partial_from_start is not None)


def _median_of_present(values: Sequence[float | None]) -> float | None:
    """The median over the runs that have it; `None` below `MIN_RUNS` of them."""
    present = [value for value in values if value is not None]
    return statistics.median(present) if len(present) >= MIN_RUNS else None


def median_timing(timings: Sequence[Timing]) -> Timing:
    """The median of each figure over `MIN_RUNS` runs or more.

    Raises `ValueError` for fewer runs: a median of fewer is no measurement. A
    first partial that fewer than `MIN_RUNS` runs had is `None` for that reason.
    """
    if len(timings) < MIN_RUNS:
        message = f"a median needs at least {MIN_RUNS} runs, got {len(timings)}"
        raise ValueError(message)
    return Timing(
        first_partial_from_start=_median_of_present(
            [timing.first_partial_from_start for timing in timings]
        ),
        first_partial_after_end=_median_of_present(
            [timing.first_partial_after_end for timing in timings]
        ),
        final_from_start=statistics.median(t.final_from_start for t in timings),
        final_after_end=statistics.median(t.final_after_end for t in timings),
    )


def words(text: str) -> list[str]:
    """The comparable words of a text: case and punctuation do not count."""
    return re.findall(r"\w+", text.casefold())


@dataclass(frozen=True)
class WordErrors:
    """Edits (substitutions, insertions, deletions) against a reference text.

    `reference_words_total` counts the reference's words once per comparison, so
    the errors of several runs added together have the words of all of them.
    """

    errors: int
    reference_words_total: int

    @property
    def rate(self) -> float:
        """Errors per reference word."""
        return self.errors / self.reference_words_total

    def plus(self, other: WordErrors) -> WordErrors:
        """The errors of both comparisons taken together."""
        return WordErrors(
            self.errors + other.errors,
            self.reference_words_total + other.reference_words_total,
        )


def word_errors(reference: str, heard: str) -> WordErrors:
    """The word edit distance of `heard` from `reference`."""
    known = words(reference)
    got = words(heard)
    previous = list(range(len(got) + 1))
    for row, known_word in enumerate(known, start=1):
        current = [row]
        for column, got_word in enumerate(got, start=1):
            current.append(
                min(
                    previous[column] + 1,
                    current[column - 1] + 1,
                    previous[column - 1] + (known_word != got_word),
                )
            )
        previous = current
    return WordErrors(previous[-1], len(known))


def seconds_of(pcm: bytes) -> float:
    """The length of 24 kHz PCM16 mono in seconds."""
    return len(pcm) / (LIVE_INPUT_RATE * BYTES_PER_SAMPLE)


class ClipError(ValueError):
    """A clip that cannot serve as one continuous turn."""


@functools.cache
def _silero() -> SileroDetectors:
    detectors = SileroDetectors()
    detectors.load()
    return detectors


def _voiced_windows(pcm: bytes) -> list[bool]:
    """Whether each window of 24 kHz `pcm`, as the service hears it, is voiced."""
    heard = Resampler().feed(pcm)
    return [is_voiced(p) for p in _silero().open().probabilities(heard)]


def voiced_seconds(pcm: bytes, rate: int) -> float:
    """The voiced seconds the service hears in `pcm` (PCM16 mono at `rate`)."""
    return sum(_voiced_windows(to_wire_rate(pcm, rate))) * WINDOW_SECONDS


def _turns(pcm: bytes) -> list[TurnEnd]:
    """The turns the service ends on 24 kHz `pcm` and the silence that follows it."""
    session = HearingSession(lambda: 0.0, _silero().open())
    resampler = Resampler()
    events = []
    for frame in frames_with_closing_silence(pcm):
        events += session.add(resampler.feed(frame))
    return [event for event in events if isinstance(event, TurnEnd)]


def check_continuous_turn(pcm: bytes) -> float:
    """Fail loud unless `pcm` (24 kHz) is one turn the service hears whole.

    The service ends a turn at `SILENCE_SECONDS` without speech; a clip with such
    a pause inside it would arrive as two turns, and the clip with the silence that
    ends it must stay below the turn cap. Returns the clip's seconds.
    """
    seconds = seconds_of(pcm)
    if seconds + SILENCE_SECONDS + PRE_ROLL_SECONDS >= MAX_TURN_SECONDS:
        message = (
            f"the clip is {seconds:.1f} s; with the closing silence it must stay"
            f" below the {MAX_TURN_SECONDS:.0f} s turn cap"
        )
        raise ClipError(message)
    turns = _turns(pcm)
    if not turns:
        message = "the clip holds no speech"
        raise ClipError(message)
    if len(turns) > 1:
        at = turns[0].end_ms / 1000
        message = (
            f"a pause of {SILENCE_SECONDS} s at {at:.1f} s would end the turn early"
        )
        raise ClipError(message)
    return seconds


def speech_only(pcm: bytes) -> bytes:
    """`pcm` (24 kHz) from a pre-roll before its first speech to its last speech."""
    voiced = [index for index, yes in enumerate(_voiced_windows(pcm)) if yes]
    if not voiced:
        message = "the clip holds no speech"
        raise ClipError(message)
    window_bytes = WINDOW_SAMPLES * LIVE_INPUT_RATE // SAMPLE_RATE * BYTES_PER_SAMPLE
    pre_roll = int(PRE_ROLL_SECONDS * LIVE_INPUT_RATE) * BYTES_PER_SAMPLE
    return pcm[
        max(voiced[0] * window_bytes - pre_roll, 0) : (voiced[-1] + 1) * window_bytes
    ]


def to_wire_rate(pcm: bytes, rate: int) -> bytes:
    """PCM16 mono at `rate` as PCM16 mono at the wire's 24 kHz, by interpolation."""
    samples = np.frombuffer(pcm, dtype="<i2")
    count = round(len(samples) * LIVE_INPUT_RATE / rate)
    positions = np.arange(count) * rate / LIVE_INPUT_RATE
    resampled = np.interp(positions, np.arange(len(samples)), samples)
    return resampled.round().astype("<i2").tobytes()


def wav_of(pcm: bytes) -> bytes:
    """A WAV file around 24 kHz PCM16 mono."""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(BYTES_PER_SAMPLE)
        writer.setframerate(LIVE_INPUT_RATE)
        writer.writeframes(pcm)
    return buffer.getvalue()


def frames_with_closing_silence(pcm: bytes) -> list[bytes]:
    """The clip in wire frames, then the silence that lets the service end the turn."""
    closed = pcm + bytes(
        int(LIVE_INPUT_RATE * TRAILING_SILENCE_SECONDS) * BYTES_PER_SAMPLE
    )
    return [
        closed[start : start + FRAME_BYTES]
        for start in range(0, len(closed), FRAME_BYTES)
    ]


@dataclass(frozen=True)
class MemoryReading:
    """The card's used and total memory in mebibytes."""

    used_mib: int
    total_mib: int


def parse_memory(text: str) -> MemoryReading:
    """Read `nvidia-smi --query-gpu=memory.used,memory.total` without units."""
    used, total = (int(part) for part in text.splitlines()[0].split(","))
    return MemoryReading(used, total)


def peak_over_baseline(baseline_mib: int, samples_mib: Sequence[int]) -> int:
    """How far the highest sample rose above the baseline, never below zero."""
    if not samples_mib:
        message = "no memory sample was taken"
        raise ValueError(message)
    return max(max(samples_mib) - baseline_mib, 0)


@dataclass(frozen=True)
class TakenCardObservation:
    """What a service started on a taken card answered."""

    speaking_ready: bool
    hearing_ready: bool
    file_status: int
    file_code: str | None
    live_status: int | None
    live_code: str | None
    spoke_bytes: int


def taken_card_failures(seen: TakenCardObservation) -> list[str]:
    """What differs from cell 30: speaking up, hearing refused as unavailable."""
    expectations = {
        "speaking is not ready": seen.speaking_ready,
        "hearing is ready on a taken card": not seen.hearing_ready,
        "the file route did not answer 503 unavailable": (
            seen.file_status == HTTPStatus.SERVICE_UNAVAILABLE
            and seen.file_code == RefusalCode.UNAVAILABLE
        ),
        "the live upgrade did not answer 503 unavailable": (
            seen.live_status == HTTPStatus.SERVICE_UNAVAILABLE
            and seen.live_code == RefusalCode.UNAVAILABLE
        ),
        "speaking returned no audio": seen.spoke_bytes > 0,
    }
    return [problem for problem, held in expectations.items() if not held]


class StageStatus(StrEnum):
    """Where a stage or model of the record stands; only the last two are finished."""

    UNFINISHED = "unfinished"
    MEASURED = "measured"
    NOT_APPLICABLE = "not applicable"


def open_stage(device: str, applies_on: str) -> dict[str, object]:
    """A stage's record before it runs: unfinished on `applies_on`, else why not."""
    if device != applies_on:
        return {
            "status": StageStatus.NOT_APPLICABLE,
            "reason": f"the device is {device}",
        }
    return {"status": StageStatus.UNFINISHED}


def _is_finished(entry: object) -> bool:
    return isinstance(entry, dict) and entry.get("status") in (
        StageStatus.MEASURED,
        StageStatus.NOT_APPLICABLE,
    )


def _heard_nothing(heard: object) -> bool:
    """True for a heard text, or a list of them, that holds a blank text."""
    texts = [heard] if isinstance(heard, str) else heard
    return isinstance(texts, list) and any(not text.strip() for text in texts)


def _null_figures(
    timing_figures: dict[str, object], path: tuple[str, ...], optional: frozenset[str]
) -> Iterator[str]:
    """The paths of a timing's null figures, except those the record may lack."""
    yield from (
        " ".join((*path, figure))
        for figure, number in timing_figures.items()
        if number is None and figure not in optional
    )


def _unmeasured_figures(entry: object, path: tuple[str, ...] = ()) -> Iterator[str]:
    """The paths of the figures of the record that measure nothing.

    A null figure of a question's `median`, except a first partial: a clip may
    draw no partial, which `runs_with_partial` records. A null figure of a long
    turn's `timing`, including its first partial, a long turn with fewer than
    `MIN_GROWING_DELTAS` growing deltas, and a `heard` text that is empty, which
    would score a word error rate of 1.0 yet prove no hearing.
    """
    if not isinstance(entry, dict):
        return
    for key, value in entry.items():
        if key in ("median", "timing") and isinstance(value, dict):
            optional = PARTIAL_FIGURES if key == "median" else frozenset()
            yield from _null_figures(value, path, optional)
        elif key == "growing_deltas":
            if value < MIN_GROWING_DELTAS:
                yield " ".join((*path, key))
        elif key == "heard":
            if _heard_nothing(value):
                yield " ".join((*path, key))
        else:
            yield from _unmeasured_figures(value, (*path, str(key)))


def not_measured(
    required_models: Iterable[str],
    models: Mapping[str, object],
    stages: Mapping[str, object],
) -> list[str]:
    """What the record lacks: unfinished models and stages, and unmeasured figures.

    The figures are those `_unmeasured_figures` names.
    """
    return [
        *(
            f"hearing model {model}"
            for model in required_models
            if not _is_finished(models.get(model))
        ),
        *(f"stage {name}" for name, stage in stages.items() if not _is_finished(stage)),
        *_unmeasured_figures(models),
    ]


def timing_record(timing: Timing) -> dict[str, float | None]:
    """A timing as record fields."""
    return asdict(timing)


def error_record(errors: WordErrors) -> dict[str, float | int]:
    """Word errors as record fields."""
    return {
        "errors": errors.errors,
        "reference_words_total": errors.reference_words_total,
        "rate": errors.rate,
    }


def errors_of(reference: str, heard: Sequence[str]) -> WordErrors:
    """The word errors of each heard text against the reference, added up."""
    return functools.reduce(
        WordErrors.plus, (word_errors(reference, text) for text in heard)
    )


def live_record(reference: str, turns: Sequence[LiveTurn]) -> dict[str, object]:
    """The figures of one question's live runs: medians, partials and words."""
    timings = [timing_of(turn) for turn in turns]
    heard = [turn.transcript for turn in turns]
    return {
        "runs": len(turns),
        "median": timing_record(median_timing(timings)),
        "runs_with_partial": partial_runs(timings),
        "timings": [timing_record(timing) for timing in timings],
        "word_errors": error_record(errors_of(reference, heard)),
        "heard": sorted(set(heard)),
        "languages": sorted({turn.language.value for turn in turns}),
    }


def file_record(
    reference: str, results: Sequence[tuple[float, str]]
) -> dict[str, object]:
    """The figures of the file route's runs: seconds and words."""
    seconds = [elapsed for elapsed, _ in results]
    heard = [text for _, text in results]
    return {
        "runs": len(results),
        "median_seconds": statistics.median(seconds),
        "seconds": seconds,
        "word_errors": error_record(errors_of(reference, heard)),
        "heard": sorted(set(heard)),
    }


def long_turn_record(
    reference: str,
    single: Sequence[LiveTurn],
    two: Sequence[LiveTurn],
    file_result: tuple[float, str],
) -> dict[str, object]:
    """A long turn over interleaved runs: its live figures are medians of the runs.

    `single` and `two` are the one-language and two-language runs, taken in turn;
    the language step is set against them. `file_result` is the one file run.
    """
    timings = [timing_of(turn) for turn in single]
    heard = [turn.transcript for turn in single]
    return {
        "runs": len(single),
        "clip_seconds": statistics.median(turn.question_seconds for turn in single),
        "two_languages": language_step_record(single, two),
        "live": {
            "growing_deltas": statistics.median(
                count_growing_deltas(turn.deltas) for turn in single
            ),
            "timing": timing_record(median_timing(timings)),
            "timings": [timing_record(timing) for timing in timings],
            "word_errors": error_record(errors_of(reference, heard)),
            "heard": sorted(set(heard)),
        },
        "file": file_record(reference, [file_result]),
    }

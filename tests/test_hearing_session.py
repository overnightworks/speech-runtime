"""C2's scheduling at its boundaries, with an injected clock and a level detector."""

import struct

import pytest

from speech_service.hearing_session import (
    MAX_TURN_SECONDS,
    SILENCE_SECONDS,
    HearingSession,
    SpeechStart,
    TurnEnd,
)
from speech_service.voice_activity import (
    PRE_ROLL_SECONDS,
    SAMPLE_RATE,
    START_WINDOWS,
    WINDOW_SAMPLES,
    WINDOW_SECONDS,
)
from tests.support.detector_support import LOUD_LEVEL, LevelDetector, ScriptedDetector

BYTES_PER_SECOND = SAMPLE_RATE * 2
SILENT_WINDOWS = round(SILENCE_SECONDS / WINDOW_SECONDS + 0.5)
PRE_ROLL_MS = int(PRE_ROLL_SECONDS * 1000)


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def _windows(count: int, level: int) -> bytes:
    return struct.pack("<h", level) * (count * WINDOW_SAMPLES)


def _seconds(seconds: float, level: int) -> bytes:
    return struct.pack("<h", level) * int(seconds * SAMPLE_RATE)


def _speech(count: int) -> bytes:
    return _windows(count, LOUD_LEVEL)


def _quiet(count: int) -> bytes:
    return _windows(count, 0)


def _session() -> tuple[HearingSession, _Clock]:
    clock = _Clock()
    return HearingSession(clock, LevelDetector()), clock


def _ends(events: list[SpeechStart | TurnEnd]) -> list[TurnEnd]:
    return [event for event in events if isinstance(event, TurnEnd)]


def test_speech_starts_a_turn_once_it_is_confirmed_at_the_first_window_of_it() -> None:
    """WIRE-70: a turn starts at the first window of the speech that Silero confirms."""
    session, _ = _session()
    session.add(_quiet(16))

    unconfirmed = session.add(_speech(START_WINDOWS - 1))
    confirmed = session.add(_speech(1))
    later = session.add(_speech(5))

    assert unconfirmed == []
    assert confirmed == [SpeechStart(audio_ms=round(16 * WINDOW_SECONDS * 1000))]
    assert later == []


def test_a_window_of_noise_between_speech_windows_restarts_the_confirmation() -> None:
    session, _ = _session()

    events = session.add(
        _speech(START_WINDOWS - 1) + _quiet(1) + _speech(START_WINDOWS - 1)
    )

    assert events == []


def test_a_turn_ends_once_the_silence_is_complete_and_not_a_window_before() -> None:
    """WIRE-66: a turn ends once its silence is complete and not a window before."""
    session, _ = _session()
    session.add(_speech(30))

    one_window_short = session.add(_quiet(SILENT_WINDOWS - 1))
    complete = session.add(_quiet(1))

    assert one_window_short == []
    (ended,) = _ends(complete)
    assert not ended.by_client


def test_a_dip_shorter_than_the_silence_does_not_end_the_turn() -> None:
    session, _ = _session()

    events = session.add(_speech(30) + _quiet(SILENT_WINDOWS - 1) + _speech(30))

    assert _ends(events) == []


def test_a_turn_holds_its_voiced_seconds_and_not_its_silence() -> None:
    session, _ = _session()
    session.add(_speech(30) + _quiet(5) + _speech(10))

    (ended,) = _ends(session.add(_quiet(SILENT_WINDOWS)))

    assert ended.voiced_seconds == pytest.approx(40 * WINDOW_SECONDS)


@pytest.mark.parametrize("leading", [10, 30])
def test_a_turn_counts_the_voiced_windows_of_its_pre_roll_and_no_older_ones(
    leading: int,
) -> None:
    """Windows from the end threshold up are voiced, confirmed or not."""
    faint, speech = 0.4, 0.9
    probabilities = [faint] * leading + [speech] * 50 + [0.0] * SILENT_WINDOWS
    session = HearingSession(_Clock(), ScriptedDetector(probabilities))

    (ended,) = _ends(session.add(_quiet(len(probabilities))))

    pre_roll_windows = int(PRE_ROLL_SECONDS / WINDOW_SECONDS)
    assert ended.voiced_seconds == pytest.approx(
        (min(leading, pre_roll_windows) + 50) * WINDOW_SECONDS
    )


def test_a_turn_starts_a_pre_roll_before_its_speech_and_drops_older_audio() -> None:
    """WIRE-66: the audio of a turn starts a pre-roll before its speech."""
    session, _ = _session()
    session.add(_quiet(100))
    speech_ms = round(100 * WINDOW_SECONDS * 1000)

    (ended,) = _ends(session.add(_speech(30) + _quiet(SILENT_WINDOWS)))

    assert ended.start_ms == pytest.approx(speech_ms - PRE_ROLL_MS, abs=1)
    held = len(ended.pcm) / BYTES_PER_SECOND
    assert held == pytest.approx(
        PRE_ROLL_SECONDS + (30 + SILENT_WINDOWS) * WINDOW_SECONDS, abs=WINDOW_SECONDS
    )


def test_a_turn_after_a_gap_counts_its_place_in_the_session_audio() -> None:
    """WIRE-70: positions count the session's audio, the dropped audio included."""
    session, _ = _session()
    session.add(_speech(30) + _quiet(SILENT_WINDOWS))
    session.add(_quiet(200))

    (ended,) = _ends(session.add(_speech(30) + _quiet(SILENT_WINDOWS)))

    first_end_ms = round((30 + SILENT_WINDOWS) * WINDOW_SECONDS * 1000)
    speech_ms = first_end_ms + round(200 * WINDOW_SECONDS * 1000)
    assert ended.start_ms == pytest.approx(speech_ms - PRE_ROLL_MS, abs=1)
    assert ended.end_ms == pytest.approx(
        speech_ms + (30 + SILENT_WINDOWS) * WINDOW_SECONDS * 1000, abs=1
    )


def test_a_long_quiet_holds_no_more_than_a_pre_roll() -> None:
    session, _ = _session()
    session.add(_seconds(60.0, 0))

    held = session.commit()

    assert isinstance(held, TurnEnd)
    assert len(held.pcm) / BYTES_PER_SECOND <= PRE_ROLL_SECONDS + WINDOW_SECONDS
    assert held.voiced_seconds == 0.0


def test_audio_that_crosses_the_cap_ends_the_turn_at_the_cap_and_opens_the_next() -> (
    None
):
    """WIRE-66: audio that crosses the cap ends the turn at the cap."""
    session, _ = _session()

    events = session.add(_seconds(31.0, LOUD_LEVEL))

    (ended,) = _ends(events)
    assert len(ended.pcm) == round(MAX_TURN_SECONDS * BYTES_PER_SECOND)
    assert ended.end_ms - ended.start_ms == round(MAX_TURN_SECONDS * 1000)
    rest = session.commit()
    assert isinstance(rest, TurnEnd)
    assert rest.start_ms == ended.end_ms
    assert len(rest.pcm) / BYTES_PER_SECOND == pytest.approx(1.0, abs=WINDOW_SECONDS)


def test_a_chunking_of_the_audio_does_not_change_its_turns() -> None:
    audio = _quiet(40) + _speech(60) + _quiet(40) + _speech(20) + _quiet(40)
    whole, _ = _session()
    expected = whole.add(audio)

    for size in (1, 333, WINDOW_SAMPLES * 2, 1999):
        session, _ = _session()
        events: list[SpeechStart | TurnEnd] = []
        for start in range(0, len(audio), size):
            events += session.add(audio[start : start + size])
        assert events == expected


def test_no_partial_is_due_before_four_tenths_of_a_second_of_audio() -> None:
    session, _ = _session()
    session.add(_speech(START_WINDOWS))

    before = session.partial_due()
    session.add(_speech(12))

    assert before is None
    assert session.partial_due() is not None


def test_a_partial_carries_the_voiced_seconds_of_the_turn_so_far() -> None:
    session, _ = _session()
    session.add(_quiet(4) + _speech(20))

    partial = session.partial_due()

    assert partial is not None
    assert partial.voiced_seconds == pytest.approx(20 * WINDOW_SECONDS)


def test_a_partial_is_due_only_every_four_tenths_of_a_second() -> None:
    session, clock = _session()
    session.add(_speech(20))
    session.partial_started()
    session.add(_speech(3))

    clock.now = 0.39
    too_soon = session.partial_due()
    clock.now = 0.4

    assert too_soon is None
    assert session.partial_due() is not None


def test_a_partial_waits_for_new_audio() -> None:
    session, clock = _session()
    session.add(_speech(20))
    session.partial_started()

    clock.now = 10.0

    assert session.partial_due() is None


def test_a_partial_is_not_due_before_speech() -> None:
    session, _ = _session()
    session.add(_quiet(60))

    assert session.partial_due() is None


def test_a_commit_ends_the_open_turn_at_once_for_the_client() -> None:
    """WIRE-63: a commit ends the open turn at once for the client."""
    session, _ = _session()
    session.add(_speech(20))

    ended = session.commit()

    assert ended == TurnEnd(
        pcm=_speech(20),
        start_ms=0,
        end_ms=round(20 * WINDOW_SECONDS * 1000),
        voiced_seconds=pytest.approx(20 * WINDOW_SECONDS),
        by_client=True,
    )


def test_a_commit_with_no_audio_since_the_last_turn_ended_is_nothing() -> None:
    session, _ = _session()
    session.add(_speech(20))
    session.commit()

    assert session.commit() is None


def test_a_turn_ended_by_the_client_leaves_the_next_to_start_at_its_end() -> None:
    session, _ = _session()
    session.add(_speech(20))
    first = session.commit()
    session.add(_speech(20))

    second = session.commit()

    assert isinstance(first, TurnEnd)
    assert isinstance(second, TurnEnd)
    assert second.start_ms == first.end_ms

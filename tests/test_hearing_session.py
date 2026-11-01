"""C2's scheduling at its boundaries, with an injected clock and synthetic audio."""

import struct

import pytest

from speech_runtime_service.hearing_session import (
    HearingSession,
    SpeechStart,
    TurnEnd,
)

RATE = 16_000
BYTES_PER_SECOND = RATE * 2
JUST_SPEECH = 656
JUST_SILENCE = 655


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def _level(seconds: float, level: int) -> bytes:
    count = int(seconds * RATE)
    return struct.pack(f"<{count}h", *[level] * count)


def _session() -> tuple[HearingSession, _Clock]:
    clock = _Clock()
    return HearingSession(clock), clock


def test_the_first_speech_of_a_turn_starts_it_once_at_its_audio_position() -> None:
    session, _ = _session()
    session.add(_level(0.5, 0))

    first = session.add(_level(0.1, 5000))
    second = session.add(_level(0.1, 5000))

    assert first == [SpeechStart(audio_ms=500)]
    assert second == []


def test_a_turn_ends_at_seven_tenths_of_a_second_of_silence_and_not_at_six() -> None:
    session, _ = _session()
    session.add(_level(1.0, 5000))
    session.add(_level(0.6, 0))

    at_six_tenths = session.poll()
    session.add(_level(0.1, 0))
    at_seven_tenths = session.poll()

    assert at_six_tenths == []
    assert [type(event) for event in at_seven_tenths] == [TurnEnd]
    ended = at_seven_tenths[0]
    assert isinstance(ended, TurnEnd)
    assert (ended.start_ms, ended.end_ms, ended.spoken) == (0, 1700, True)
    assert not ended.by_client


@pytest.mark.parametrize(
    ("level", "ends"),
    [(JUST_SILENCE, True), (JUST_SPEECH, False)],
    ids=["just-under-counts-as-silence", "just-over-does-not"],
)
def test_the_silence_threshold_is_an_rms_of_two_hundredths(
    level: int, *, ends: bool
) -> None:
    session, _ = _session()
    session.add(_level(0.5, 5000))
    session.add(_level(0.7, level))

    assert bool(session.poll()) is ends


def test_audio_that_crosses_the_cap_ends_the_turn_at_the_cap_and_opens_the_next() -> (
    None
):
    session, _ = _session()

    events = session.add(_level(31.0, 5000))

    ended = [event for event in events if isinstance(event, TurnEnd)]
    assert [(turn.start_ms, turn.end_ms) for turn in ended] == [(0, 30_000)]
    assert len(ended[0].pcm) == 30 * BYTES_PER_SECOND
    assert session.commit() == TurnEnd(
        pcm=_level(1.0, 5000),
        start_ms=30_000,
        end_ms=31_000,
        spoken=True,
        by_client=True,
    )


def test_a_cap_turn_without_speech_is_dropped_without_an_end() -> None:
    session, _ = _session()

    assert session.add(_level(30.0, 0)) == []
    assert session.commit() is None


def test_no_partial_is_due_before_four_tenths_of_a_second_of_audio() -> None:
    session, _ = _session()
    session.add(_level(0.39, 5000))

    before = session.partial_due()
    session.add(_level(0.01, 5000))

    assert before is None
    assert session.partial_due() is not None


def test_a_partial_is_due_only_every_four_tenths_of_a_second() -> None:
    session, clock = _session()
    session.add(_level(0.5, 5000))
    session.partial_started()
    session.add(_level(0.1, 5000))

    clock.now = 0.39
    too_soon = session.partial_due()
    clock.now = 0.4

    assert too_soon is None
    assert session.partial_due() is not None


def test_a_partial_waits_for_new_audio() -> None:
    session, clock = _session()
    session.add(_level(0.5, 5000))
    session.partial_started()

    clock.now = 10.0

    assert session.partial_due() is None


def test_a_partial_is_not_due_before_speech() -> None:
    session, _ = _session()
    session.add(_level(1.0, 0))

    assert session.partial_due() is None


def test_a_commit_ends_the_open_turn_at_once_for_the_client() -> None:
    session, _ = _session()
    session.add(_level(0.3, 5000))

    ended = session.commit()

    assert ended == TurnEnd(
        pcm=_level(0.3, 5000), start_ms=0, end_ms=300, spoken=True, by_client=True
    )


def test_a_commit_with_no_audio_since_the_last_turn_ended_is_nothing() -> None:
    session, _ = _session()
    session.add(_level(0.3, 5000))
    session.commit()

    assert session.commit() is None


def test_a_turn_starts_where_the_last_one_ended() -> None:
    session, _ = _session()
    session.add(_level(0.3, 5000))
    session.commit()
    session.add(_level(0.2, 5000))

    ended = session.commit()

    assert isinstance(ended, TurnEnd)
    assert (ended.start_ms, ended.end_ms) == (300, 500)

"""The hearing stage proof's figures hold for scenarios worked out by hand."""

from __future__ import annotations

import io
import wave
from pathlib import Path

import numpy as np
import pytest

import hearing_figures as figures
from speech_client.wire import Language, RefusalCode
from speech_service.hearing_session import SILENCE_SECONDS
from speech_service.pcm import pcm_from_wav
from speech_service.voice_activity import PRE_ROLL_SECONDS, START_WINDOWS

RATE = 24000
LOUD = 8000
# The silence the phrase carries before and after its words.
EDGES_SECONDS = 0.3
WINDOW_BYTES = 512 * RATE // 16_000 * 2
# Fewer Silero windows than confirm a turn, taken from inside the words, some of
# which Silero rates as voiced.
BURST_BYTES = (START_WINDOWS - 1) * WINDOW_BYTES
BURST_FROM_BYTES = 10 * WINDOW_BYTES
SPEECH = Path(__file__).parents[1] / "fixtures" / "speech" / "nein_danke.wav"


def speech() -> bytes:
    """A German phrase spoken by a Piper voice, at the wire rate."""
    rate, pcm = pcm_from_wav(SPEECH.read_bytes())
    return figures.to_wire_rate(pcm, rate)


def silence(seconds: float) -> bytes:
    return bytes(round(seconds * RATE) * 2)


def seconds_of(pcm: bytes) -> float:
    return figures.seconds_of(pcm)


def tone(seconds: float, amplitude: int = LOUD, rate: int = RATE) -> bytes:
    count = round(seconds * rate)
    wave_form = amplitude * np.sin(2 * np.pi * 440 * np.arange(count) / rate)
    return wave_form.astype("<i2").tobytes()


def turn(
    question: float,
    deltas: tuple[float, ...],
    final: float,
    language: Language = Language.DE,
) -> figures.LiveTurn:
    return figures.LiveTurn(
        question_seconds=question,
        delta_seconds=deltas,
        deltas=tuple("wort" for _ in deltas),
        completed_seconds=final,
        transcript="wort",
        language=language,
    )


def test_the_language_step_is_the_median_added_to_the_final_and_what_it_chose() -> None:
    single = [turn(2.0, (), 3.0), turn(2.0, (), 3.2), turn(2.0, (), 3.1)]
    two = [
        turn(2.0, (), 3.4, Language.EN),
        turn(2.0, (), 3.6, Language.EN),
        turn(2.0, (), 3.5, Language.DE),
    ]

    record = figures.language_step_record(single, two)

    assert record["added_seconds"] == pytest.approx(0.4)
    assert "above_limit" not in record
    assert record["chosen"] == ["de", "en"]


def test_a_turn_is_timed_from_its_start_and_from_its_end() -> None:
    timing = figures.timing_of(turn(question=3.0, deltas=(2.0, 4.5), final=9.0))

    assert timing == figures.Timing(
        first_partial_from_start=2.0,
        first_partial_after_end=-1.0,
        final_from_start=9.0,
        final_after_end=6.0,
    )


def test_a_turn_without_a_delta_has_no_first_partial() -> None:
    timing = figures.timing_of(turn(question=3.0, deltas=(), final=5.0))

    assert timing == figures.Timing(
        first_partial_from_start=None,
        first_partial_after_end=None,
        final_from_start=5.0,
        final_after_end=2.0,
    )


def test_the_median_takes_the_middle_run_of_each_figure() -> None:
    finals = [9.0, 5.0, 7.0, 6.0, 8.0]
    timings = [
        figures.timing_of(turn(question=3.0, deltas=(first,), final=final))
        for first, final in zip([1.0, 2.0, 4.0, 3.0, 5.0], finals, strict=True)
    ]

    median = figures.median_timing(timings)

    assert median == figures.Timing(
        first_partial_from_start=3.0,
        first_partial_after_end=0.0,
        final_from_start=7.0,
        final_after_end=4.0,
    )


@pytest.mark.parametrize(
    ("deltas", "partials", "first_partial"),
    [
        ([(1.0,), (), (3.0,), (), (2.0,)], 3, None),
        ([(1.0,), (), (3.0,), (2.0,), (), (5.0,), (4.0,)], 5, 3.0),
    ],
    ids=["three of five runs had a partial", "five of seven runs had a partial"],
)
def test_a_first_partial_median_needs_five_runs_that_had_a_partial(
    deltas: list[tuple[float, ...]], partials: int, first_partial: float | None
) -> None:
    timings = [
        figures.timing_of(turn(question=3.0, deltas=delta, final=5.0))
        for delta in deltas
    ]

    assert figures.partial_runs(timings) == partials
    assert figures.median_timing(timings).first_partial_from_start == first_partial


def test_fewer_than_five_runs_give_no_median() -> None:
    timings = [figures.timing_of(turn(3.0, (1.0,), 5.0))] * 4

    with pytest.raises(ValueError, match="at least 5 runs"):
        figures.median_timing(timings)


@pytest.mark.parametrize(
    ("deltas", "growing"),
    [
        ([], 0),
        (["Wie"], 1),
        (["Wie", " spät", " ist", " es"], 4),
        (["Wie", " ", ""], 1),
    ],
)
def test_only_deltas_that_add_words_are_counted(
    deltas: list[str], growing: int
) -> None:
    assert figures.count_growing_deltas(deltas) == growing


@pytest.mark.parametrize(
    ("reference", "heard", "errors", "reference_words_total"),
    [
        ("Wie spät ist es?", "wie spät ist es", 0, 4),
        ("Wie spät ist es", "Wie spät ist das", 1, 4),
        ("Wie spät ist es", "Wie ist es", 1, 4),
        ("Wie spät ist es", "Wie früh spät ist es", 1, 4),
        ("Wie spät ist es", "", 4, 4),
        ("Wie spät ist es", "Was kostet das Brot hier", 5, 4),
    ],
)
def test_word_errors_count_the_edits_against_the_known_sentence(
    reference: str, heard: str, errors: int, reference_words_total: int
) -> None:
    found = figures.word_errors(reference, heard)

    assert (found.errors, found.reference_words_total) == (
        errors,
        reference_words_total,
    )


def test_word_errors_of_several_runs_add_up_to_one_rate() -> None:
    first = figures.word_errors("a b c d", "a b c x")
    second = figures.word_errors("a b c d", "a b c d")

    assert first.plus(second).rate == 1 / 8


def test_a_clip_of_continuous_speech_is_one_turn() -> None:
    assert figures.check_continuous_turn(speech()) == pytest.approx(
        seconds_of(speech())
    )


def test_a_short_pause_does_not_end_the_turn() -> None:
    pause = SILENCE_SECONDS - EDGES_SECONDS - 0.1
    clip = speech() + silence(pause) + speech()

    assert figures.check_continuous_turn(clip) == pytest.approx(
        2 * seconds_of(speech()) + pause
    )


def test_a_pause_beyond_the_silence_length_ends_the_turn_early() -> None:
    clip = speech() + silence(SILENCE_SECONDS) + speech()

    with pytest.raises(figures.ClipError, match="would end the turn early"):
        figures.check_continuous_turn(clip)


def test_silence_before_and_after_the_speech_is_not_a_pause() -> None:
    clip = silence(1.0) + speech() + silence(1.0)

    assert figures.check_continuous_turn(clip) == pytest.approx(
        seconds_of(speech()) + 2.0
    )


def test_speech_only_cuts_the_silence_before_and_after_the_speech() -> None:
    clip = silence(2.0) + speech() + silence(2.0)

    cut = figures.speech_only(clip)

    assert seconds_of(cut) <= seconds_of(speech()) + PRE_ROLL_SECONDS
    assert seconds_of(cut) < seconds_of(clip) - 2.0


def test_speech_only_refuses_a_clip_without_speech() -> None:
    clip = silence(1.0)

    with pytest.raises(figures.ClipError, match="holds no speech"):
        figures.speech_only(clip)


def test_a_clip_that_reaches_the_turn_cap_is_refused() -> None:
    clip = silence(29.5)

    with pytest.raises(figures.ClipError, match="turn cap"):
        figures.check_continuous_turn(clip)


@pytest.mark.parametrize("clip", [silence(3.0), tone(3.0)], ids=["silence", "tone"])
def test_a_clip_without_speech_is_refused_however_loud(clip: bytes) -> None:
    with pytest.raises(figures.ClipError, match="no speech"):
        figures.check_continuous_turn(clip)


def test_the_voiced_seconds_of_speech_are_its_own_and_of_silence_are_none() -> None:
    clip = silence(1.0) + speech() + silence(1.0)

    assert 0.0 < figures.voiced_seconds(clip, RATE) <= seconds_of(speech())
    assert figures.voiced_seconds(silence(2.0), RATE) == 0.0


def test_a_sound_that_opens_no_turn_adds_no_voiced_seconds() -> None:
    burst = speech()[BURST_FROM_BYTES : BURST_FROM_BYTES + BURST_BYTES]
    without = silence(1.0) + speech() + silence(1.0)
    with_burst = silence(1.0) + burst + silence(1.5) + speech() + silence(1.0)

    assert figures.voiced_seconds(with_burst, RATE) <= figures.voiced_seconds(
        without, RATE
    )


def test_resampling_gives_the_samples_of_the_wire_rate() -> None:
    resampled = figures.to_wire_rate(tone(1.0, rate=22050), 22050)

    assert len(resampled) == RATE * 2


def test_a_wav_carries_the_clip_at_the_wire_rate() -> None:
    clip = tone(0.5)

    with wave.open(io.BytesIO(figures.wav_of(clip))) as reader:
        assert reader.getframerate() == RATE
        assert reader.getnchannels() == 1
        assert reader.readframes(reader.getnframes()) == clip


def test_the_frames_end_in_silence_for_two_seconds() -> None:
    clip = tone(1.0)

    frames = figures.frames_with_closing_silence(clip)

    assert b"".join(frames[:10]) == clip
    assert all(frame == bytes(figures.FRAME_BYTES) for frame in frames[10:])
    assert len(frames) * figures.FRAME_SECONDS == pytest.approx(3.0)


def test_the_card_reading_is_used_and_total_memory() -> None:
    assert figures.parse_memory("671, 24576\n") == figures.MemoryReading(671, 24576)


def test_the_peak_is_the_highest_sample_above_the_baseline() -> None:
    assert figures.peak_over_baseline(700, [710, 3900, 3100]) == 3900 - 700


def test_a_peak_below_the_baseline_is_zero() -> None:
    assert figures.peak_over_baseline(700, [600]) == 0


def test_no_sample_gives_no_peak() -> None:
    with pytest.raises(ValueError, match="no memory sample"):
        figures.peak_over_baseline(700, [])


def taken(**changes: object) -> list[str]:
    held = {
        "speaking_ready": True,
        "hearing_ready": False,
        "file_status": 503,
        "file_code": RefusalCode.UNAVAILABLE.value,
        "live_status": 503,
        "live_code": RefusalCode.UNAVAILABLE.value,
        "spoke_bytes": 4096,
    }
    return figures.taken_card_failures(
        figures.TakenCardObservation(**{**held, **changes})  # type: ignore[arg-type]
    )


@pytest.mark.parametrize(
    ("change", "problem"),
    [
        ({"speaking_ready": False}, "speaking is not ready"),
        ({"hearing_ready": True}, "hearing is ready on a taken card"),
        ({"file_status": 200}, "the file route did not answer 503 unavailable"),
        ({"file_code": "busy"}, "the file route did not answer 503 unavailable"),
        ({"live_status": 101}, "the live upgrade did not answer 503 unavailable"),
        ({"live_code": "busy"}, "the live upgrade did not answer 503 unavailable"),
        ({"live_code": None}, "the live upgrade did not answer 503 unavailable"),
        ({"spoke_bytes": 0}, "speaking returned no audio"),
    ],
)
def test_a_taken_card_that_differs_from_cell_30_names_what_differs(
    change: dict[str, object], problem: str
) -> None:
    assert taken(**change) == [problem]


def test_a_taken_card_as_cell_30_says_has_no_failures() -> None:
    assert taken() == []


def test_a_stage_that_needs_the_card_is_not_applicable_on_the_cpu() -> None:
    assert figures.open_stage("cpu", "cuda") == {
        "status": figures.StageStatus.NOT_APPLICABLE,
        "reason": "the device is cpu",
    }


def test_a_stage_that_needs_the_card_opens_unfinished_on_cuda() -> None:
    assert figures.open_stage("cuda", "cuda") == {
        "status": figures.StageStatus.UNFINISHED
    }


def test_only_unfinished_or_missing_models_and_stages_are_not_measured() -> None:
    models = {
        "small": {"status": figures.StageStatus.MEASURED},
        "large-v3": {"status": figures.StageStatus.UNFINISHED},
    }
    stages = {
        "gpu_memory": figures.open_stage("cuda", "cuda"),
        "taken_card": figures.open_stage("cpu", "cuda"),
    }

    assert figures.not_measured(["small", "large-v3", "base"], models, stages) == [
        "hearing model large-v3",
        "hearing model base",
        "stage gpu_memory",
    ]


@pytest.mark.parametrize(
    ("median", "unmeasured"),
    [
        (
            figures.timing_record(figures.Timing(None, None, 4.0, 3.5)),
            [],
        ),
        (
            {
                "first_partial_from_start": None,
                "first_partial_after_end": None,
                "final_from_start": 4.0,
                "final_after_end": None,
            },
            ["small languages de question live final_after_end"],
        ),
    ],
    ids=["no-partial-in-the-question", "no-final-in-the-question"],
)
def test_a_question_figure_that_is_null_is_named_only_when_it_is_not_a_partial(
    median: dict[str, float | None], unmeasured: list[str]
) -> None:
    models = {
        "small": {
            "status": figures.StageStatus.MEASURED,
            "languages": {"de": {"question": {"live": {"median": median}}}},
        }
    }

    assert figures.not_measured(["small"], models, {}) == unmeasured


def test_a_long_turn_without_growing_deltas_or_timing_is_not_measured() -> None:
    timed = figures.timing_record(figures.Timing(1.0, 0.5, 4.0, 3.5))
    untimed = figures.timing_record(figures.Timing(None, None, 4.0, 3.5))

    def model(growing: int, timing: dict[str, float | None]) -> dict[str, object]:
        live = {"growing_deltas": growing, "timing": timing}
        return {
            "status": figures.StageStatus.MEASURED,
            "languages": {"de": {"long_turn": {"live": live}}},
        }

    models = {
        "small": model(3, timed),
        "base": model(1, timed),
        "edge": model(figures.MIN_GROWING_DELTAS, timed),
        "tiny": model(0, untimed),
    }

    assert figures.not_measured(list(models), models, {}) == [
        "base languages de long_turn live growing_deltas",
        "tiny languages de long_turn live growing_deltas",
        "tiny languages de long_turn live first_partial_from_start",
        "tiny languages de long_turn live first_partial_after_end",
    ]


def test_the_long_turn_takes_medians_and_reports_its_step_with_spread() -> None:
    deltas = (1.0, 2.0)
    single = [
        turn(question=3.0, deltas=deltas, final=final)
        for final in (5.0, 6.0, 7.0, 8.0, 9.0)
    ]
    two = [
        turn(question=3.0, deltas=deltas, final=final + 0.5)
        for final in (5.0, 6.0, 7.0, 8.0, 9.0)
    ]

    record = figures.long_turn_record("wort", single, two, (0.5, "wort"))

    step = record["two_languages"]
    assert step["added_seconds"] == pytest.approx(0.5)
    assert "above_limit" not in step
    assert step["single_final_lowest_after_end"] == pytest.approx(2.0)
    assert step["single_final_highest_after_end"] == pytest.approx(6.0)
    assert step["two_final_lowest_after_end"] == pytest.approx(2.5)
    assert step["two_final_highest_after_end"] == pytest.approx(6.5)
    assert record["live"]["timing"]["final_after_end"] == pytest.approx(4.0)
    assert record["live"]["growing_deltas"] == len(deltas)
    assert record["live"]["word_errors"]["errors"] == 0


def test_an_empty_heard_text_is_not_measured() -> None:
    def model(question: list[str], long_turn: str) -> dict[str, object]:
        return {
            "status": figures.StageStatus.MEASURED,
            "languages": {
                "de": {
                    "question": {"file": {"heard": question}},
                    "long_turn": {"file": {"heard": long_turn}},
                }
            },
        }

    models = {
        "small": model(["Hallo Welt"], "Guten Tag"),
        "base": model(["Hallo Welt", "  "], "Guten Tag"),
        "tiny": model(["Hallo Welt"], ""),
    }

    assert figures.not_measured(list(models), models, {}) == [
        "base languages de question file heard",
        "tiny languages de long_turn file heard",
    ]


def test_timing_and_error_records_carry_plain_numbers() -> None:
    timing = figures.Timing(1.0, 0.5, 4.0, 3.5)

    assert figures.timing_record(timing) == {
        "first_partial_from_start": 1.0,
        "first_partial_after_end": 0.5,
        "final_from_start": 4.0,
        "final_after_end": 3.5,
    }
    assert figures.error_record(figures.WordErrors(1, 4)) == {
        "errors": 1,
        "reference_words_total": 4,
        "rate": 0.25,
    }

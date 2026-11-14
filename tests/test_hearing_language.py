"""The adapter names a language in one encoder pass and decodes as faster-whisper."""

import dataclasses
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
from faster_whisper import WhisperModel

from speech_client.wire import Language
from speech_service.hearing import (
    HEARING_SAMPLE_RATE,
    LanguageIdentifier,
    LanguageScores,
    ScoreIdentifier,
    Utterance,
    WhisperHearing,
)
from speech_service.hearing_models import HearingModel, voiced_floor
from speech_service.voice_activity import WINDOW_SECONDS
from tests.support.whisper_support import StandInWhisper

if TYPE_CHECKING:
    from faster_whisper.transcribe import TranscriptionOptions

DE = Language.DE
EN = Language.EN
FLOOR = voiced_floor(HearingModel.SMALL)
LOUD = 5000
ENOUGH = FLOOR + 1.0


class _ReadsOnlyThePcm:
    """A standalone identifier: it answers from the audio and ignores the scores."""

    def __init__(self, *answers: Language | None) -> None:
        self._answers = iter(answers)
        self.asked: list[tuple[Utterance, tuple[Language, ...]]] = []

    def identify(
        self, utterance: Utterance, allowed: tuple[Language, ...]
    ) -> Language | None:
        self.asked.append((utterance, allowed))
        return next(self._answers)


def _speech(seconds: float) -> bytes:
    """A noise-like signal, so the features differ per frame."""
    count = round(seconds * HEARING_SAMPLE_RATE)
    generator = np.random.default_rng(count)
    return generator.integers(-LOUD, LOUD, count, dtype=np.int16).tobytes()


def _engine(
    stand_in: StandInWhisper, identifier: LanguageIdentifier | None = None
) -> WhisperHearing:
    engine = WhisperHearing(
        HearingModel.SMALL,
        Path("model"),
        "cpu",
        identifier or ScoreIdentifier(FLOOR),
        stand_in.libraries,
    )
    engine.load()
    return engine


def _library_transcribe(stand_in: StandInWhisper, pcm: bytes, language: str) -> None:
    """Today's call: the library's own `transcribe` with the service's options."""
    audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    segments, _info = WhisperModel.transcribe(
        stand_in,
        audio,
        language=language,
        beam_size=1,
        temperature=0.0,
        vad_filter=False,
        without_timestamps=True,
        condition_on_previous_text=False,
    )
    assert "".join(segment.text for segment in segments).strip() == "Guten Tag"


def _without_the_librarys_skip(options: TranscriptionOptions) -> TranscriptionOptions:
    """The adapter judges no speech itself, so the library never skips a window."""
    return dataclasses.replace(options, no_speech_threshold=None)


@pytest.fixture
def stand_in() -> StandInWhisper:
    return StandInWhisper()


@pytest.mark.parametrize("seconds", [1.0, 40.0], ids=["one-window", "over-30-seconds"])
@pytest.mark.parametrize("language", list(Language))
def test_a_one_element_set_decodes_as_faster_whispers_own_transcribe_does(
    stand_in: StandInWhisper, language: Language, seconds: float
) -> None:
    """WIRE-38: greedy at temperature 0, with the options of faster-whisper's call."""
    pcm = _speech(seconds)
    _library_transcribe(stand_in, pcm, language.value)
    expected = stand_in.decodes.pop()

    heard = _engine(stand_in).transcribe(pcm, (language,))

    (decode,) = stand_in.decodes
    assert (heard.text, heard.language) == ("Guten Tag", language)
    assert decode.language_code == expected.language_code == language.value
    assert np.array_equal(decode.features, expected.features)
    assert decode.options == _without_the_librarys_skip(expected.options)
    assert decode.log_progress == expected.log_progress
    assert decode.encoder_output is None
    assert stand_in.encodes == []
    assert stand_in.detections == []


@pytest.mark.parametrize("seconds", [1.0, 40.0], ids=["one-window", "over-30-seconds"])
def test_a_set_of_two_decodes_with_the_same_options_from_its_one_encoder_pass(
    stand_in: StandInWhisper, seconds: float
) -> None:
    pcm = _speech(seconds)
    _library_transcribe(stand_in, pcm, "de")
    expected = stand_in.decodes.pop()

    heard = _engine(stand_in).transcribe(pcm, (DE, EN))

    (decode,) = stand_in.decodes
    assert heard.language is DE
    assert decode.language_code == "de"
    assert np.array_equal(decode.features, expected.features)
    assert decode.options == _without_the_librarys_skip(expected.options)
    assert len(stand_in.encodes) == 1
    assert decode.encoder_output is not None
    assert decode.encoder_output.number == 1
    assert stand_in.detections == [decode.encoder_output]


@pytest.mark.parametrize(
    "voiced", [ENOUGH, FLOOR - 0.1], ids=["above-the-floor", "below"]
)
def test_a_decode_of_a_set_of_two_encodes_once_also_when_the_identifier_gives_up(
    stand_in: StandInWhisper, voiced: float
) -> None:
    pcm = _speech(3.0)

    _engine(stand_in).transcribe(pcm, (DE, EN), voiced)

    (decode,) = stand_in.decodes
    assert len(stand_in.encodes) == 1
    assert len(stand_in.detections) == 1
    assert decode.encoder_output == stand_in.detections[0]


def test_the_encoder_is_fed_one_window_of_the_features_the_decode_gets(
    stand_in: StandInWhisper,
) -> None:
    pcm = _speech(40.0)

    _engine(stand_in).transcribe(pcm, (DE, EN))

    (decode,) = stand_in.decodes
    (encoded,) = stand_in.encodes
    window = stand_in.feature_extractor.nb_max_frames
    assert encoded.shape[-1] == window
    assert np.array_equal(encoded, decode.features[:, :window])


def test_the_options_are_built_anew_for_every_decode(stand_in: StandInWhisper) -> None:
    engine = _engine(stand_in)

    engine.transcribe(_speech(1.2), (DE,))
    engine.transcribe(_speech(1.2), (DE,))

    first, second = (decode.options for decode in stand_in.decodes)
    assert first is not second
    assert first.suppress_tokens is not second.suppress_tokens


def test_a_set_of_one_never_asks_the_identifier(stand_in: StandInWhisper) -> None:
    identifier = _ReadsOnlyThePcm()

    _engine(stand_in, identifier).transcribe(_speech(3.0), (EN,))

    assert identifier.asked == []
    assert stand_in.detections == []


def test_the_identifier_is_given_the_pcm_and_the_probabilities_of_the_set_only(
    stand_in: StandInWhisper,
) -> None:
    """WIRE-135: no language outside the set is chosen; no Whisper token crosses."""
    stand_in.probabilities = {"fr": 0.5, "de": 0.1, "en": 0.4}
    identifier = _ReadsOnlyThePcm(None)
    pcm = _speech(3.0)

    _engine(stand_in, identifier).transcribe(pcm, (EN, DE))

    ((utterance, allowed),) = identifier.asked
    assert allowed == (EN, DE)
    assert utterance == Utterance(pcm, 0.0, LanguageScores({DE: 0.1, EN: 0.4}))


@pytest.mark.parametrize(
    ("allowed", "language"),
    [((EN, DE), EN), ((DE, EN), DE)],
    ids=["english-main", "german-main"],
)
def test_when_the_identifier_does_not_know_the_first_language_of_the_set_holds(
    stand_in: StandInWhisper,
    allowed: tuple[Language, ...],
    language: Language,
) -> None:
    """WIRE-135: not knowing means the main language."""
    heard = _engine(stand_in, _ReadsOnlyThePcm(None)).transcribe(_speech(3.0), allowed)

    assert heard.language is language
    assert stand_in.decodes[0].language_code == language.value


def test_a_word_after_a_long_pause_below_the_floor_is_heard_in_the_main_language(
    stand_in: StandInWhisper,
) -> None:
    """A turn counts its voiced seconds, not its length."""
    stand_in.probabilities = {"de": 0.1, "en": 0.9}
    pcm = _speech(5.0 + FLOOR)

    heard = _engine(stand_in).transcribe(pcm, (DE, EN), FLOOR - 0.1)

    assert heard.language is DE


def test_above_the_floor_the_language_the_scores_favour_is_heard(
    stand_in: StandInWhisper,
) -> None:
    stand_in.probabilities = {"de": 0.1, "en": 0.9}

    heard = _engine(stand_in).transcribe(_speech(3.0), (DE, EN), ENOUGH)

    assert (heard.text, heard.language) == ("Guten Tag", EN)
    assert stand_in.decodes[0].language_code == "en"


def test_an_identifier_that_reads_only_the_pcm_can_replace_the_shipped_one(
    stand_in: StandInWhisper,
) -> None:
    """The identifier is exchangeable behind its port; the decode is unchanged."""
    stand_in.probabilities = {"de": 0.9, "en": 0.1}
    engine = _engine(stand_in, _ReadsOnlyThePcm(EN, None))

    first = engine.transcribe(_speech(3.0), (DE, EN), ENOUGH)
    second = engine.transcribe(_speech(3.0), (DE, EN), ENOUGH)

    assert (first.language, second.language) == (EN, DE)


def _utterance(voiced_seconds: float, *, de: float, en: float) -> Utterance:
    return Utterance(_speech(1.0), voiced_seconds, LanguageScores({DE: de, EN: en}))


def test_a_language_outside_the_set_is_never_chosen() -> None:
    """WIRE-135: no language outside the set is ever chosen."""
    scores = LanguageScores({DE: 0.04, EN: 0.03})

    named = ScoreIdentifier(FLOOR).identify(
        Utterance(_speech(1.0), ENOUGH, scores), (EN,)
    )

    assert named is EN


@pytest.mark.parametrize(
    ("de", "en", "named"),
    [(0.51, 0.49, DE), (0.49, 0.51, EN)],
    ids=["german-leads", "english-leads"],
)
def test_above_the_floor_the_higher_score_is_named_either_way(
    de: float, en: float, named: Language
) -> None:
    identifier = ScoreIdentifier(FLOOR)
    utterance = _utterance(ENOUGH, de=de, en=en)

    assert identifier.identify(utterance, (DE, EN)) is named
    assert identifier.identify(utterance, (EN, DE)) is named


@pytest.mark.parametrize(
    ("de", "en"), [(0.4, 0.4), (0.0, 0.0)], ids=["tie", "all-zero"]
)
def test_a_tie_or_scores_of_zero_are_not_known(de: float, en: float) -> None:
    utterance = _utterance(ENOUGH, de=de, en=en)

    assert ScoreIdentifier(FLOOR).identify(utterance, (DE, EN)) is None


@pytest.mark.parametrize(
    ("voiced", "named"),
    [(FLOOR, EN), (FLOOR - WINDOW_SECONDS, None)],
    ids=["at-the-floor", "one-window-below"],
)
def test_the_floor_is_met_at_exactly_its_voiced_seconds_and_not_one_window_below(
    voiced: float, named: Language | None
) -> None:
    utterance = _utterance(voiced, de=0.1, en=0.9)

    assert ScoreIdentifier(FLOOR).identify(utterance, (DE, EN)) is named


def test_silence_holds_no_voiced_speech_so_it_is_not_known() -> None:
    identifier = ScoreIdentifier(FLOOR)

    assert identifier.identify(_utterance(0.0, de=0.1, en=0.9), (EN, DE)) is None

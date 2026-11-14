"""Hearing: the resident faster-whisper engine and the slot requests capture it from."""

from __future__ import annotations

import gc
import logging
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final, Protocol

import numpy as np

from speech_service.admission_gate import AdmissionGate, Lease
from speech_service.cuda_libs import cuda_library_dirs, prepare_cuda_libraries
from speech_service.hearing_models import model_directory, voiced_floor
from speech_service.voice_activity import (
    SAMPLE_RATE,
    SileroDetectors,
    open_shipped_model,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping
    from pathlib import Path

    from faster_whisper.transcribe import Segment, TranscriptionOptions

    from speech_client.wire import Language
    from speech_service.clock import Clock
    from speech_service.config import Settings
    from speech_service.hearing_models import HearingModel
    from speech_service.voice_activity import DetectorSource, SpeechDetector

logger = logging.getLogger(__name__)

HEARING_SAMPLE_RATE: Final = SAMPLE_RATE
CPU_THREADS: Final = 4
_SETTING: Final = "SPEECH_HEARING_MODEL"
_DEVICES: Final = frozenset({"cpu", "cuda"})
# faster-whisper's own skip of a recording's window as no speech: its no-speech
# probability is above the first and its average log probability not above the second.
NO_SPEECH_THRESHOLD: Final = 0.6
LOG_PROB_THRESHOLD: Final = -1.0


class ResourceUnavailableError(Exception):
    """The load failed for want of a resource: a card, its libraries or room on it."""


@dataclass(frozen=True, slots=True)
class Heard:
    """The words of one decode, the language it was decoded in and its measures.

    The measures are those of the decode's first segment, and `None` when the decode
    yielded no segment.
    """

    text: str
    language: Language
    no_speech_prob: float | None = None
    avg_logprob: float | None = None


@dataclass(frozen=True, slots=True)
class LanguageScores:
    """The probability of each language of the allowed set, from one encoder pass."""

    allowed: Mapping[Language, float]


@dataclass(frozen=True, slots=True)
class Utterance:
    """What an identifier may read: the turn's PCM, its voiced seconds and the scores.

    The PCM is 16 kHz mono 16-bit; the voiced seconds are those of its speech windows.
    """

    pcm: bytes
    voiced_seconds: float
    scores: LanguageScores


class LanguageIdentifier(Protocol):
    """The step that names the language of a turn, before the recogniser decodes it."""

    def identify(
        self, utterance: Utterance, allowed: tuple[Language, ...]
    ) -> Language | None:
        """A language of `allowed`, or `None` when the identifier does not know.

        Called for a set of two or more.
        """


@dataclass(frozen=True, slots=True)
class ScoreIdentifier:
    """Names the set's best-scored language once the turn holds enough speech."""

    min_voiced_seconds: float

    def identify(
        self, utterance: Utterance, allowed: tuple[Language, ...]
    ) -> Language | None:
        """The language of `allowed` with the highest score, else `None`.

        `None` below `min_voiced_seconds` of voiced speech, and on a tie.
        """
        if utterance.voiced_seconds < self.min_voiced_seconds:
            return None
        scores = utterance.scores.allowed
        best = max(scores[language] for language in allowed)
        leaders = [language for language in allowed if scores[language] == best]
        return leaders[0] if len(leaders) == 1 else None


class HearingEngine(Protocol):
    """A resident hearing model."""

    model_name: str

    def load(self) -> None:
        """Load the weights, or raise `ResourceUnavailableError` or another error."""

    def open_detector(self) -> SpeechDetector:
        """A detector at the start of one live session's audio, once loaded."""

    def transcribe(
        self,
        pcm: bytes,
        languages: tuple[Language, ...],
        voiced_seconds: float = 0.0,
        *,
        skip_no_speech: bool = True,
    ) -> Heard:
        """The words in 16 kHz mono 16-bit PCM, and the language they were heard in.

        `languages` is ordered, non-empty and unique; its first element is the main
        language, heard when the set holds one language or the language is not known.
        `voiced_seconds` are those of `pcm`'s speech windows; a recording has no turns
        and names no language, so it leaves them out. A window the recogniser judges
        no speech adds no words unless `skip_no_speech` is off: a live turn drops
        nothing on its measures.
        """

    def close(self) -> None:
        """Free what the engine holds; safe after a failed load."""


def _is_resource_failure(error: RuntimeError) -> bool:
    # CTranslate2 reports a full card only as RuntimeError text, with no error type.
    return "out of memory" in str(error).lower()


@dataclass(frozen=True, slots=True)
class HearingLibraries:
    """The inference libraries' entry points that hearing needs."""

    model_factory: Callable[..., Any]
    cuda_device_count: Callable[[], int]
    cuda_library_dirs: Callable[[], list[Path]]
    detectors: DetectorSource


def installed_libraries() -> HearingLibraries:
    """The installed faster-whisper and CTranslate2, imported only now."""
    prepare_cuda_libraries()
    import ctranslate2  # noqa: PLC0415 - heavy and loaded only once hearing is configured
    from faster_whisper import WhisperModel  # noqa: PLC0415 - as above

    return HearingLibraries(
        WhisperModel,
        ctranslate2.get_cuda_device_count,
        cuda_library_dirs,
        SileroDetectors(open_shipped_model),
    )


def _scores(
    ranked: list[tuple[str, float]], languages: tuple[Language, ...]
) -> LanguageScores:
    """The probabilities of `languages` out of CTranslate2's `("<|xx|>", p)` ranking."""
    probabilities = dict(ranked)
    return LanguageScores(
        {language: probabilities[f"<|{language.value}|>"] for language in languages}
    )


def _decode_options(
    suppress_tokens: list[int], *, skip_no_speech: bool
) -> TranscriptionOptions:
    """The options of today's decode: greedy, one temperature, no timestamps.

    Every other field is the default of faster-whisper's own `transcribe`, and
    `skip_no_speech` off drops its skip of a window as no speech. A fresh
    object for every decode, because `generate_segments` rewrites its clip timestamps.
    """
    from faster_whisper.transcribe import TranscriptionOptions  # noqa: PLC0415 - as above

    return TranscriptionOptions(
        beam_size=1,
        best_of=5,
        patience=1,
        length_penalty=1,
        repetition_penalty=1,
        no_repeat_ngram_size=0,
        log_prob_threshold=LOG_PROB_THRESHOLD,
        no_speech_threshold=NO_SPEECH_THRESHOLD if skip_no_speech else None,
        compression_ratio_threshold=2.4,
        condition_on_previous_text=False,
        prompt_reset_on_temperature=0.5,
        temperatures=[0.0],
        initial_prompt=None,
        prefix=None,
        suppress_blank=True,
        suppress_tokens=suppress_tokens,
        without_timestamps=True,
        max_initial_timestamp=1.0,
        word_timestamps=False,
        prepend_punctuations="\"'\u201c\u00bf([{-",
        append_punctuations="\"'.\u3002,\uff0c!\uff01?\uff1f:\uff1a\u201d)]}\u3001",
        multilingual=False,
        max_new_tokens=None,
        clip_timestamps="0",
        hallucination_silence_threshold=None,
        hotwords=None,
    )


def _heard(segments: Iterable[Segment], language: Language) -> Heard:
    """The words of the segments, with the first segment's measures."""
    measured = list(segments)
    if not measured:
        return Heard("", language)
    words = "".join(segment.text for segment in measured)
    first = measured[0]
    return Heard(words.strip(), language, first.no_speech_prob, first.avg_logprob)


class WhisperHearing:
    """One faster-whisper model, loaded once from local files and held."""

    def __init__(
        self,
        model: HearingModel,
        directory: Path,
        device: str,
        identifier: LanguageIdentifier,
        libraries: Callable[[], HearingLibraries] = installed_libraries,
    ) -> None:
        """Remember the model, where its files are and the device it will occupy.

        `identifier` names the language of a turn when the set holds two or more.

        Raises `ValueError` for a device other than `cpu` or `cuda`.
        """
        if device not in _DEVICES:
            message = (
                f"SPEECH_DEVICE={device} is not a device hearing runs on: cpu or cuda"
            )
            raise ValueError(message)
        self.model_name = model.value
        self._directory = directory
        self._cuda = device == "cuda"
        self._identifier = identifier
        self._libraries = libraries
        self._model = None
        self._detectors: DetectorSource | None = None

    def load(self) -> None:
        """Load the CTranslate2 weights from the local directory, never the network."""
        libraries = self._libraries()
        if self._cuda:
            if libraries.cuda_device_count() == 0:
                message = "no CUDA device"
                raise ResourceUnavailableError(message)
            if not libraries.cuda_library_dirs():
                message = "the gpu extra is not installed: CUDA libraries are missing"
                raise ResourceUnavailableError(message)
        try:
            self._model = libraries.model_factory(
                str(self._directory),
                device="cuda" if self._cuda else "cpu",
                compute_type="float16" if self._cuda else "int8",
                cpu_threads=CPU_THREADS,
                local_files_only=True,
            )
        except RuntimeError as error:
            if _is_resource_failure(error):
                message = "the card is full"
                raise ResourceUnavailableError(message) from None
            raise
        libraries.detectors.load()
        self._detectors = libraries.detectors

    def open_detector(self) -> SpeechDetector:
        """A detector at the start of one live session's audio."""
        if self._detectors is None:
            message = "hearing model is not loaded"
            raise RuntimeError(message)
        return self._detectors.open()

    def transcribe(
        self,
        pcm: bytes,
        languages: tuple[Language, ...],
        voiced_seconds: float = 0.0,
        *,
        skip_no_speech: bool = True,
    ) -> Heard:
        """The recognised words and their language; the same audio gives the same text.

        A set of two or more encodes the first window once; the identifier chooses
        from that pass and the decode reuses its output. A window faster-whisper
        judges no speech adds no words unless `skip_no_speech` is off.
        """
        from faster_whisper.audio import pad_or_trim  # noqa: PLC0415 - as above
        from faster_whisper.tokenizer import Tokenizer  # noqa: PLC0415 - as above
        from faster_whisper.transcribe import (  # noqa: PLC0415 - as above
            get_suppressed_tokens,
        )

        model = self._model
        if model is None:
            message = "hearing model is not loaded"
            raise RuntimeError(message)
        audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        features = model.feature_extractor(audio)
        encoder_output = None
        language = languages[0]
        if len(languages) > 1:
            # The first window `generate_segments` would encode shares this pass.
            window = min(model.feature_extractor.nb_max_frames, features.shape[-1] - 1)
            encoder_output = model.encode(pad_or_trim(features[:, :window]))
            ranked = model.model.detect_language(encoder_output)[0]
            scores = _scores(ranked, languages)
            known = self._identifier.identify(
                Utterance(pcm, voiced_seconds, scores), languages
            )
            language = known or language
        tokenizer = Tokenizer(
            model.hf_tokenizer,
            model.model.is_multilingual,
            task="transcribe",
            language=language.value,
        )
        options = _decode_options(
            get_suppressed_tokens(tokenizer, [-1]), skip_no_speech=skip_no_speech
        )
        segments = model.generate_segments(
            features,
            tokenizer,
            options,
            log_progress=False,
            encoder_output=encoder_output,
        )
        return _heard(segments, language)

    def close(self) -> None:
        """Drop the model so its memory is freed."""
        self._model = None
        if self._detectors is not None:
            self._detectors.close()
            self._detectors = None
        gc.collect()


class Hearing:
    """The loaded hearing engine, the gate its decodes pass and their one worker."""

    def __init__(self, engine: HearingEngine | None, clock: Clock) -> None:
        """Hold `engine`, or none when no hearing model is configured.

        The gate's deadlines, and so the live sessions', are on `clock`.
        """
        self._engine = engine
        self._ready: HearingEngine | None = None
        self.gate = AdmissionGate(clock)
        self._worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="hearing")
        self._sessions = 0

    def load(self) -> None:
        """Load the engine.

        A load that fails for want of the resource leaves hearing unavailable and
        frees what it allocated; any other failure raises, naming the setting.
        """
        engine = self._engine
        if engine is None:
            return
        misconfigured = False
        try:
            engine.load()
        except ResourceUnavailableError:
            logger.exception("hearing model %s is unavailable", engine.model_name)
        except Exception:
            logger.exception("hearing model %s failed to load", engine.model_name)
            misconfigured = True
        else:
            self._ready = engine
            logger.info("hearing model %s ready", engine.model_name)
            return
        engine.close()
        if misconfigured:
            message = f"{_SETTING}={engine.model_name} failed to load"
            raise RuntimeError(message)

    def capture(self) -> HearingEngine | None:
        """The loaded engine, or `None` while hearing is unavailable."""
        return self._ready

    def open_detector(self) -> SpeechDetector:
        """A detector at the start of one live session's audio.

        Raises `RuntimeError` while hearing is unavailable.
        """
        engine = self._ready
        if engine is None:
            message = "hearing is unavailable"
            raise RuntimeError(message)
        return engine.open_detector()

    def open_session(self, limit: int) -> Lease | None:
        """A place for one live session, or `None` while `limit` sessions are open.

        Called on the event loop only; closing the lease frees the place.
        """
        if self._sessions >= limit:
            return None
        self._sessions += 1
        return Lease(self._end_session)

    def _end_session(self) -> None:
        self._sessions -= 1

    def submit[Result](
        self, lease: Lease, job: Callable[[HearingEngine], Result]
    ) -> Future[Result]:
        """Run `job` on the engine in the one worker, owning `lease` from here on.

        The worker closes the lease when the job returns; a future cancelled before
        it started closes it in its done-callback; a submission that cannot be made
        closes it here. Raises `RuntimeError` for the last two causes: no engine, or
        hearing already shut down.
        """
        engine = self._ready

        def run() -> Result:
            try:
                return job(engine)
            finally:
                lease.close()

        if engine is None:
            lease.close()
            message = "hearing is unavailable"
            raise RuntimeError(message)
        try:
            future = self._worker.submit(run)
        except RuntimeError:
            lease.close()
            raise
        future.add_done_callback(lambda _done: lease.close())
        return future

    def close(self) -> None:
        """Take no new decode, wait for the running one, then free the engine."""
        self._ready = None
        self._worker.shutdown(wait=True, cancel_futures=True)
        if self._engine is not None:
            self._engine.close()

    def health(self) -> dict[str, object]:
        """The contract body of hearing for GET /health."""
        return {
            "model": self._engine.model_name if self._engine else "unavailable",
            "ready": self._ready is not None,
        }


def hearing_from_settings(settings: Settings, clock: Clock) -> Hearing:
    """Hearing for the configured model, or none; a missing model raises at once."""
    if settings.hearing_model is None:
        return Hearing(None, clock)
    directory = model_directory(settings.hearing_model, settings.huggingface_cache)
    identifier = ScoreIdentifier(voiced_floor(settings.hearing_model))
    whisper = WhisperHearing(
        settings.hearing_model, directory, settings.device, identifier
    )
    return Hearing(whisper, clock)

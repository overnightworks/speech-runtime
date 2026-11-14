"""A stand-in for faster-whisper's model at the surface the hearing adapter touches."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from types import SimpleNamespace
from typing import TYPE_CHECKING, Self

from faster_whisper.feature_extractor import FeatureExtractor

from speech_service.hearing import HearingLibraries

if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np
    from faster_whisper.tokenizer import Tokenizer
    from faster_whisper.transcribe import TranscriptionOptions

DEFAULT_PROBABILITIES = {"de": 0.7, "en": 0.2, "fr": 0.1}


@dataclass(frozen=True, slots=True)
class Segment:
    """One decoded segment, by default a confident one."""

    text: str
    no_speech_prob: float = 0.01
    avg_logprob: float = -0.2


@dataclass(frozen=True, slots=True)
class EncoderOutput:
    """What `encode` returned for its `number`-th call."""

    number: int


@dataclass(frozen=True, slots=True)
class Decode:
    """The arguments of one `generate_segments` call."""

    features: np.ndarray
    tokenizer: Tokenizer
    options: TranscriptionOptions
    log_progress: object
    encoder_output: EncoderOutput | None

    @property
    def language_code(self) -> str:
        """The language the decode's prompt names."""
        return self.tokenizer.language_code


class _Tokens:
    def __init__(self) -> None:
        self._ids: dict[str, int] = {}

    def token_to_id(self, token: str) -> int:
        return self._ids.setdefault(token, len(self._ids) + 1)

    def encode(self, text: str, *, add_special_tokens: bool) -> SimpleNamespace:
        assert not add_special_tokens
        return SimpleNamespace(ids=[ord(text[-1])])


class StandInWhisper:
    """The factory and the model in one: it records how it was built and used.

    The model's attributes are the ones `WhisperModel.transcribe` and the adapter
    read; the feature extractor is the library's own.
    """

    is_multilingual = True

    def __init__(self) -> None:
        """Start unbuilt, with the default language probabilities."""
        self.built: list[dict[str, object]] = []
        self.failure: RuntimeError | None = None
        self.probabilities = dict(DEFAULT_PROBABILITIES)
        self.segments = [Segment(" Guten"), Segment(" Tag ")]
        self.encodes: list[np.ndarray] = []
        self.detections: list[EncoderOutput] = []
        self.decodes: list[Decode] = []
        self.logger = logging.getLogger("faster_whisper")
        self.feature_extractor = FeatureExtractor()
        self.hf_tokenizer = _Tokens()
        self.model = self

    def libraries(
        self, cuda_devices: int = 0, cuda_library_dirs: list[Path] | None = None
    ) -> HearingLibraries:
        """The library entry points, with this stand-in as the model factory."""
        return HearingLibraries(
            self, lambda: cuda_devices, lambda: cuda_library_dirs or []
        )

    def __call__(self, path: str, **options: object) -> Self:
        """Build the model, or raise the failure the test set."""
        if self.failure is not None:
            raise self.failure
        self.built.append({"path": path, **options})
        return self

    def encode(self, features: np.ndarray) -> EncoderOutput:
        """Count the encoder pass and hand back its numbered output."""
        self.encodes.append(features)
        return EncoderOutput(len(self.encodes))

    def detect_language(
        self, encoder_output: EncoderOutput
    ) -> list[list[tuple[str, float]]]:
        """The scripted probabilities, highest first, as CTranslate2 ranks them."""
        self.detections.append(encoder_output)
        ranked = sorted(self.probabilities.items(), key=lambda item: -item[1])
        return [[(f"<|{code}|>", probability) for code, probability in ranked]]

    def generate_segments(
        self,
        features: np.ndarray,
        tokenizer: Tokenizer,
        options: TranscriptionOptions,
        log_progress: object,
        encoder_output: EncoderOutput | None = None,
    ) -> list[Segment]:
        """Record the decode and answer the fixed text."""
        self.decodes.append(
            Decode(features, tokenizer, options, log_progress, encoder_output)
        )
        return self.segments

"""The resident speaking engines: Piper by default, Chatterbox when configured."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from speech.config import CHATTERBOX_SPEAKING_MODEL

if TYPE_CHECKING:
    from collections.abc import Generator
    from pathlib import Path

    from speech.config import Settings
    from speech.service import SpeakingEngine

_LOG = logging.getLogger(__name__)


class PiperSpeaking:
    """One Piper voice, loaded once and held for the process."""

    streams = False

    def __init__(self, model_name: str, cache: Path) -> None:
        """Remember the catalogue name and where ONNX files are kept."""
        self.model_name = model_name
        self.sample_rate = 0
        self.ready = False
        self._cache = cache
        self._voice = None

    def load(self) -> None:
        """Download the voice if needed and keep it resident."""
        from piper import PiperVoice
        from piper.download_voices import download_voice

        self._cache.mkdir(parents=True, exist_ok=True)
        onnx = self._cache / f"{self.model_name}.onnx"
        if not onnx.is_file():
            _LOG.info("downloading speaking model %s", self.model_name)
            download_voice(self.model_name, self._cache)
        voice = PiperVoice.load(onnx)
        self._voice = voice
        self.sample_rate = voice.config.sample_rate
        self.ready = True
        _LOG.info("speaking model %s ready", self.model_name)

    def pcm_chunks(self, text: str, language: str) -> Generator[bytes, None, None]:
        """Yield 16-bit mono PCM as soon as Piper produces a chunk."""
        del language
        if self._voice is None:
            message = "speaking model is not loaded"
            raise RuntimeError(message)
        for chunk in self._voice.synthesize(text):
            yield chunk.audio_int16_bytes


def speaking_from_settings(settings: Settings) -> SpeakingEngine:
    """The configured voice: Chatterbox when named, otherwise Piper."""
    if settings.speaking_model == CHATTERBOX_SPEAKING_MODEL:
        from speech.chatterbox import ChatterboxSpeaking

        return ChatterboxSpeaking(
            settings.speaking_model, settings.device, settings.huggingface_cache
        )
    return PiperSpeaking(settings.speaking_model, settings.voice_cache)

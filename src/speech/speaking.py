"""The resident Piper voice."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from speech.config import Settings

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

    def pcm_chunks(self, text: str) -> Iterator[bytes]:
        """Yield 16-bit mono PCM as soon as Piper produces a chunk."""
        if self._voice is None:
            message = "speaking model is not loaded"
            raise RuntimeError(message)
        for chunk in self._voice.synthesize(text):
            yield chunk.audio_int16_bytes


def speaking_from_settings(settings: Settings) -> PiperSpeaking:
    """The configured Piper voice."""
    return PiperSpeaking(settings.speaking_model, settings.voice_cache)

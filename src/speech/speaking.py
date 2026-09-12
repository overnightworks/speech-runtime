"""The resident speaking engines: Piper by default, Chatterbox when configured."""

from __future__ import annotations

from typing import TYPE_CHECKING

from speech.config import CHATTERBOX_SPEAKING_MODEL, DEFAULT_SPEAKING_MODEL

if TYPE_CHECKING:
    from collections.abc import Generator
    from pathlib import Path

    from speech.config import Settings
    from speech.service import SpeakingEngine
    from speech.voices import VoiceId


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
        """Load the exact local Piper artifacts and keep them resident."""
        from piper import PiperVoice

        onnx = self._cache / f"{self.model_name}.onnx"
        if not onnx.is_file() or not onnx.with_suffix(".onnx.json").is_file():
            raise FileNotFoundError
        voice = PiperVoice.load(onnx)
        self._voice = voice
        self.sample_rate = voice.config.sample_rate
        self.ready = True

    def pcm_chunks(self, text: str, language: str) -> Generator[bytes, None, None]:
        """Yield 16-bit mono PCM as soon as Piper produces a chunk."""
        del language
        if self._voice is None:
            message = "speaking model is not loaded"
            raise RuntimeError(message)
        for chunk in self._voice.synthesize(text):
            yield chunk.audio_int16_bytes


def speaking_for_voice(settings: Settings, voice: VoiceId) -> SpeakingEngine:
    """Construct one supported local engine from the closed catalogue id."""
    from speech.voices import VoiceId

    if voice is VoiceId.PIPER:
        return PiperSpeaking(DEFAULT_SPEAKING_MODEL, settings.voice_cache)
    if voice is VoiceId.CHATTERBOX:
        from speech.chatterbox import ChatterboxSpeaking

        return ChatterboxSpeaking(
            CHATTERBOX_SPEAKING_MODEL, settings.device, settings.huggingface_cache
        )
    raise ValueError

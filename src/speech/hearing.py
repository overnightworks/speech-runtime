"""The resident faster-whisper listener, with partials on a growing buffer."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Final

import numpy as np

from speech.config import HEAR_SAMPLE_RATE
from speech.pcm import BYTES_PER_SAMPLE, is_silence

if TYPE_CHECKING:
    from speech.config import Settings

_LOG = logging.getLogger(__name__)

PARTIAL_PERIOD_SECONDS: Final = 0.4
MIN_AUDIO_SECONDS: Final = 0.4
SILENCE_SECONDS: Final = 0.7
SPEECH_RMS: Final = 0.02
POLL_WAIT_SECONDS: Final = 0.3
MAX_FRAME_SECONDS: Final = 1.0
MAX_UTTERANCE_SECONDS: Final = 30.0

Transcribe = Callable[[bytes], str]


class FrameTooLargeError(Exception):
    """A single PCM frame exceeded the bound the session will accept."""


def _max_pcm_bytes(sample_rate: int, seconds: float) -> int:
    return int(seconds * sample_rate * BYTES_PER_SAMPLE)


def _device_kind(device: str) -> str:
    if device.startswith("cuda"):
        return "cuda"
    return "cpu"


def _compute_type(device: str) -> str:
    if _device_kind(device) == "cuda":
        return "float16"
    return "int8"


class HearingSession:
    """One websocket's audio, transcribed without reloading the model."""

    def __init__(self, transcribe: Transcribe, sample_rate: int) -> None:
        """Bind a transcriber; the model behind it is already loaded."""
        self._transcribe = transcribe
        self._sample_rate = sample_rate
        self._pcm = bytearray()
        self._last_partial = ""
        self._last_transcribe_at = 0.0
        self._last_pcm_len = 0
        self._heard_speech = False
        self._lock = threading.Lock()
        self._max_frame_bytes = _max_pcm_bytes(sample_rate, MAX_FRAME_SECONDS)
        self._max_utterance_bytes = _max_pcm_bytes(sample_rate, MAX_UTTERANCE_SECONDS)

    def add_pcm(self, data: bytes) -> None:
        """Append a raw 16-bit PCM frame."""
        if not data:
            return
        if len(data) > self._max_frame_bytes:
            raise FrameTooLargeError
        with self._lock:
            room = self._max_utterance_bytes - len(self._pcm)
            if room <= 0:
                return
            chunk = data if len(data) <= room else data[:room]
            self._pcm.extend(chunk)
            if not is_silence(chunk, SPEECH_RMS):
                self._heard_speech = True

    def poll(self) -> dict[str, object] | None:
        """A partial or a final when the model has something new, else None."""
        with self._lock:
            duration = self._duration()
            if duration < MIN_AUDIO_SECONDS:
                return None
            if duration >= MAX_UTTERANCE_SECONDS or (
                self._heard_speech and self._tail_is_silent()
            ):
                return self._take_final_locked()
            if not self._heard_speech:
                return None
            now = time.monotonic()
            if (
                now - self._last_transcribe_at < PARTIAL_PERIOD_SECONDS
                or len(self._pcm) == self._last_pcm_len
            ):
                return None
            self._last_transcribe_at = now
            self._last_pcm_len = len(self._pcm)
            pcm = bytes(self._pcm)
            text = self._transcribe(pcm)
            if not text or text == self._last_partial:
                return None
            self._last_partial = text
            return {"text": text, "final": False}

    def flush(self) -> dict[str, object] | None:
        """Force a final for whatever is still buffered."""
        with self._lock:
            return self._take_final_locked()

    def _take_final_locked(self) -> dict[str, object] | None:
        if not self._pcm or not self._heard_speech:
            self._reset()
            return None
        text = self._transcribe(bytes(self._pcm))
        self._reset()
        if not text:
            return None
        return {"text": text, "final": True}

    def _duration(self) -> float:
        return len(self._pcm) / (self._sample_rate * BYTES_PER_SAMPLE)

    def _tail_is_silent(self) -> bool:
        tail_bytes = int(SILENCE_SECONDS * self._sample_rate * BYTES_PER_SAMPLE)
        tail = bytes(self._pcm[-tail_bytes:])
        return len(tail) >= tail_bytes and is_silence(tail, SPEECH_RMS)

    def _reset(self) -> None:
        self._pcm.clear()
        self._last_partial = ""
        self._last_transcribe_at = 0.0
        self._last_pcm_len = 0
        self._heard_speech = False


class WhisperHearing:
    """One faster-whisper model, loaded once and held for the process."""

    def __init__(self, model_name: str, device: str) -> None:
        """Remember the Hub id and the device the weights will occupy."""
        self.model_name = model_name
        self.ready = False
        self._device = device
        self._model = None

    def load(self) -> None:
        """Load the CTranslate2 weights onto the configured device."""
        from speech.cuda_libs import prepare_cuda_libraries

        prepare_cuda_libraries()
        from faster_whisper import WhisperModel

        kind = _device_kind(self._device)
        _LOG.info("loading hearing model %s on %s", self.model_name, kind)
        self._model = WhisperModel(
            self.model_name,
            device=kind,
            compute_type=_compute_type(self._device),
        )
        self.ready = True
        _LOG.info("hearing model %s ready", self.model_name)

    def open_session(self, language: str) -> HearingSession:
        """A new utterance stream on the already-loaded model."""
        if self._model is None:
            message = "hearing model is not loaded"
            raise RuntimeError(message)
        model = self._model

        def transcribe(pcm: bytes) -> str:
            audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
            segments, _info = model.transcribe(
                audio,
                language=language,
                beam_size=1,
                vad_filter=False,
                without_timestamps=True,
                condition_on_previous_text=False,
            )
            return "".join(segment.text for segment in segments).strip()

        return HearingSession(transcribe, HEAR_SAMPLE_RATE)


def hearing_from_settings(settings: Settings) -> WhisperHearing:
    """The configured faster-whisper model."""
    return WhisperHearing(settings.hearing_model, settings.device)

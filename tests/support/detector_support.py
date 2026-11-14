"""A speech detector at the port: loud windows are speech, quiet ones are not."""

from __future__ import annotations

import numpy as np

from speech_service.voice_activity import WINDOW_SAMPLES

_BYTES_PER_WINDOW = WINDOW_SAMPLES * 2
LOUD_LEVEL = 5000
_SPEECH_LEVEL = 1000


class LevelDetector:
    """Answers 1.0 for a window whose mean level reaches 1000, else 0.0.

    It keeps the samples short of a window, as the real stream does.
    """

    def __init__(self) -> None:
        """Start before the first sample."""
        self._unfinished = b""

    def probabilities(self, pcm: bytes) -> list[float]:
        """One answer for each whole window `pcm` completes."""
        heard = self._unfinished + pcm
        complete = len(heard) // _BYTES_PER_WINDOW * _BYTES_PER_WINDOW
        self._unfinished = heard[complete:]
        samples = np.frombuffer(heard[:complete], dtype="<i2").astype(np.float64)
        levels = np.abs(samples).reshape(-1, WINDOW_SAMPLES).mean(axis=1)
        return [float(level >= _SPEECH_LEVEL) for level in levels]

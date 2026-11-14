"""Voice activity: streaming Silero VAD over 16 kHz audio, and what turns read of it.

One ONNX session is shared by every live session; each session opens a stream that
carries its own recurrent state, context and unfinished window.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final, Protocol

import numpy as np

if TYPE_CHECKING:
    from onnxruntime import InferenceSession

SAMPLE_RATE: Final = 16_000
WINDOW_SAMPLES: Final = 512
CONTEXT_SAMPLES: Final = 64
_STATE_SHAPE: Final = (1, 1, 128)
_OUTPUT_COUNT: Final = 3
_INT16_SCALE: Final = 32768.0
_BYTES_PER_SAMPLE: Final = 2

# Measured on Piper speech, recorded speech and noise (see the commit that set them).
START_THRESHOLD: Final = 0.5
END_THRESHOLD: Final = 0.35
START_WINDOWS: Final = 3
PRE_ROLL_SECONDS: Final = 0.35
WINDOW_SECONDS: Final = WINDOW_SAMPLES / SAMPLE_RATE


def is_voiced(probability: float) -> bool:
    """Whether a window of this speech probability counts as voiced speech."""
    return probability >= END_THRESHOLD


class SpeechDetector(Protocol):
    """One session's view of its audio, window by window."""

    def probabilities(self, pcm: bytes) -> list[float]:
        """The speech probability of each 512-sample window that `pcm` completes.

        `pcm` is 16 kHz mono 16-bit audio; samples short of a window wait for the
        next call.
        """


class ModelShapeError(RuntimeError):
    """The Silero model does not have the inputs and outputs the stream feeds."""


def _checked_shapes(session: InferenceSession) -> None:
    inputs = {entry.name: entry.shape for entry in session.get_inputs()}
    outputs = [entry.name for entry in session.get_outputs()]
    expected_inputs = {
        "input": ["seq_len", WINDOW_SAMPLES + CONTEXT_SAMPLES],
        "h": list(_STATE_SHAPE),
        "c": list(_STATE_SHAPE),
    }
    if inputs != expected_inputs or len(outputs) != _OUTPUT_COUNT:
        message = f"the Silero model has inputs {inputs} and outputs {outputs}"
        raise ModelShapeError(message)


class SileroDetectors:
    """The shared Silero session, and the source of one stream per live session."""

    def __init__(self) -> None:
        """Start unloaded."""
        self._session: InferenceSession | None = None

    def load(self) -> None:
        """Load the model that ships with faster-whisper and check its shapes.

        Raises `ModelShapeError` when the model is not the one the stream feeds.
        """
        import onnxruntime  # noqa: PLC0415 - heavy and loaded only once hearing is configured
        from faster_whisper.utils import get_assets_path  # noqa: PLC0415 - as above

        options = onnxruntime.SessionOptions()
        options.inter_op_num_threads = 1
        options.intra_op_num_threads = 1
        options.enable_cpu_mem_arena = False
        options.log_severity_level = 4
        session = onnxruntime.InferenceSession(
            f"{get_assets_path()}/silero_vad_v6.onnx",
            providers=["CPUExecutionProvider"],
            sess_options=options,
        )
        _checked_shapes(session)
        self._session = session
        silence = bytes(WINDOW_SAMPLES * _BYTES_PER_SAMPLE)
        if len(self.open().probabilities(silence)) != 1:
            message = "the Silero model does not answer one probability per window"
            raise ModelShapeError(message)

    def open(self) -> SileroStream:
        """A stream at the start of one session's audio."""
        if self._session is None:
            message = "the Silero model is not loaded"
            raise RuntimeError(message)
        return SileroStream(self._session)

    def close(self) -> None:
        """Drop the session."""
        self._session = None


class SileroStream:
    """Silero over one session's audio, window by window.

    Carries the recurrent state, the last 64 samples of the previous window and
    the samples short of the next window. A split of the audio into chunks gives
    the same probabilities as the whole.
    """

    def __init__(self, session: InferenceSession) -> None:
        """Start before the first sample."""
        self._session = session
        self._hidden = np.zeros(_STATE_SHAPE, dtype=np.float32)
        self._cell = np.zeros(_STATE_SHAPE, dtype=np.float32)
        self._context = np.zeros(CONTEXT_SAMPLES, dtype=np.float32)
        self._unfinished = b""

    def probabilities(self, pcm: bytes) -> list[float]:
        """The speech probability of each 512-sample window that `pcm` completes."""
        heard = self._unfinished + pcm
        window_bytes = WINDOW_SAMPLES * _BYTES_PER_SAMPLE
        complete = len(heard) // window_bytes * window_bytes
        self._unfinished = heard[complete:]
        if not complete:
            return []
        samples = np.frombuffer(heard[:complete], dtype="<i2").astype(np.float32)
        windows = (samples / _INT16_SCALE).reshape(-1, WINDOW_SAMPLES)
        contexts = np.concatenate(
            [self._context[None, :], windows[:-1, -CONTEXT_SAMPLES:]]
        )
        self._context = windows[-1, -CONTEXT_SAMPLES:].copy()
        probabilities, self._hidden, self._cell = self._session.run(
            None,
            {
                "input": np.concatenate([contexts, windows], axis=1),
                "h": self._hidden,
                "c": self._cell,
            },
        )
        return [float(probability) for probability in probabilities]

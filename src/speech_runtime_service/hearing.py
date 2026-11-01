"""Hearing: the resident faster-whisper engine and the slot requests capture it from."""

from __future__ import annotations

import gc
import logging
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final, Protocol

from speech_runtime_service.admission_gate import AdmissionGate, Lease
from speech_runtime_service.cuda_libs import prepare_cuda_libraries
from speech_runtime_service.hearing_models import model_directory

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from speech_runtime_client.wire import HearingLanguage
    from speech_runtime_service.config import Settings
    from speech_runtime_service.hearing_models import HearingModel

logger = logging.getLogger(__name__)

HEARING_SAMPLE_RATE: Final = 16_000
CPU_THREADS: Final = 4
_SETTING: Final = "SPEECH_HEARING_MODEL"
_DEVICES: Final = frozenset({"cpu", "cuda"})


class ResourceUnavailableError(Exception):
    """The load failed for want of the resource: no CUDA device, or the card is full."""


class HearingEngine(Protocol):
    """A resident hearing model."""

    model_name: str

    def load(self) -> None:
        """Load the weights, or raise `ResourceUnavailableError` or another error."""

    def transcribe(self, pcm: bytes, language: HearingLanguage) -> str:
        """The words in 16 kHz mono 16-bit PCM."""

    def close(self) -> None:
        """Free what the engine holds; safe after a failed load."""


def _is_resource_failure(error: RuntimeError) -> bool:
    # CTranslate2 reports a full card only as RuntimeError text, with no error type.
    return "out of memory" in str(error).lower()


@dataclass(frozen=True, slots=True)
class HearingLibraries:
    """The two entry points of the inference libraries that hearing needs."""

    model_factory: Callable[..., Any]
    cuda_device_count: Callable[[], int]


def installed_libraries() -> HearingLibraries:
    """The installed faster-whisper and CTranslate2, imported only now."""
    prepare_cuda_libraries()
    import ctranslate2  # noqa: PLC0415 - heavy and loaded only once hearing is configured
    from faster_whisper import WhisperModel  # noqa: PLC0415 - as above

    return HearingLibraries(WhisperModel, ctranslate2.get_cuda_device_count)


class WhisperHearing:
    """One faster-whisper model, loaded once from local files and held."""

    def __init__(
        self,
        model: HearingModel,
        directory: Path,
        device: str,
        libraries: Callable[[], HearingLibraries] = installed_libraries,
    ) -> None:
        """Remember the model, where its files are and the device it will occupy.

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
        self._libraries = libraries
        self._model = None

    def load(self) -> None:
        """Load the CTranslate2 weights from the local directory, never the network."""
        libraries = self._libraries()
        if self._cuda and libraries.cuda_device_count() == 0:
            message = "no CUDA device"
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

    def transcribe(self, pcm: bytes, language: HearingLanguage) -> str:
        """The recognised words; the same audio gives the same text."""
        import numpy as np  # noqa: PLC0415 - only the loaded engine needs it

        if self._model is None:
            message = "hearing model is not loaded"
            raise RuntimeError(message)
        audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        segments, _info = self._model.transcribe(
            audio,
            language=language.value,
            beam_size=1,
            temperature=0.0,
            vad_filter=False,
            without_timestamps=True,
            condition_on_previous_text=False,
        )
        return "".join(segment.text for segment in segments).strip()

    def close(self) -> None:
        """Drop the model so its memory is freed."""
        self._model = None
        gc.collect()


class Hearing:
    """The loaded hearing engine, the gate its decodes pass and their one worker."""

    def __init__(self, engine: HearingEngine | None) -> None:
        """Hold `engine`, or none when no hearing model is configured."""
        self._engine = engine
        self._ready: HearingEngine | None = None
        self.gate = AdmissionGate()
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


def hearing_from_settings(settings: Settings) -> Hearing:
    """Hearing for the configured model, or none; a missing model raises at once."""
    if settings.hearing_model is None:
        return Hearing(None)
    directory = model_directory(settings.hearing_model, settings.huggingface_cache)
    return Hearing(WhisperHearing(settings.hearing_model, directory, settings.device))

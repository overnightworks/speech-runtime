"""HTTP contract: health and the runtime that speaks."""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from speech_runtime_service.admission_gate import AdmissionGate
from speech_runtime_service.card import card_memory_mb
from speech_runtime_service.clock import MONOTONIC_CLOCK
from speech_runtime_service.failures import PayloadFreeFailure, SpeakingFailedError
from speech_runtime_service.hearing import Hearing, hearing_from_settings
from speech_runtime_service.pcm import wav_header
from speech_runtime_service.speaking import speaking_for_voice
from speech_runtime_service.voices import VoiceId, installed_voice_ids, voice_for_model

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Iterator

    from fastapi import FastAPI

    from speech_runtime_service.config import Settings

logger = logging.getLogger(__name__)


class SpeakingEngine(Protocol):
    """A resident voice."""

    model_name: str
    sample_rate: int
    ready: bool
    streams: bool

    def load(self) -> None:
        """Load weights and become ready, or raise."""

    def pcm_chunks(self, text: str, language: str) -> Generator[bytes]:
        """16-bit mono PCM as soon as the model produces it."""

    def close(self) -> None:
        """Release owned resources once."""


@dataclass(frozen=True, slots=True, kw_only=True)
class RuntimeDependencies:
    """The configured voice and the seams that check and build its engine."""

    voice: VoiceId = VoiceId.PIPER
    engine_factory: Callable[[VoiceId], SpeakingEngine] | None = None
    artifact_checker: Callable[[VoiceId], bool] | None = None


class Runtime:
    """The resident speaking model, hearing, and the flags a caller waits on."""

    def __init__(
        self,
        speaking: SpeakingEngine | None,
        *,
        memory_probe: Callable[[], int] | None = None,
        dependencies: RuntimeDependencies | None = None,
        hearing: Hearing | None = None,
    ) -> None:
        """Hold the speaking engine and hearing. Neither is loaded until `load`."""
        self.speaking = speaking
        self.hearing = hearing or Hearing(None, MONOTONIC_CLOCK)
        self.loading = False
        self._memory_probe = memory_probe or card_memory_mb
        self._dependencies = dependencies or RuntimeDependencies()
        self._state_lock = threading.Lock()
        self._state_changed = threading.Condition(self._state_lock)
        self._ready_speaking = speaking if speaking and speaking.ready else None
        self._pending_engine: SpeakingEngine | None = None
        self._closing_engine: SpeakingEngine | None = None
        self._stopping = False
        self.synthesis_gate = AdmissionGate(MONOTONIC_CLOCK)
        self._fatal_callback: Callable[[], None] = lambda: None

    @classmethod
    def from_settings(cls, settings: Settings) -> Runtime:
        """Build the production engines the environment configures."""
        return cls(
            None,
            hearing=hearing_from_settings(settings, MONOTONIC_CLOCK),
            dependencies=RuntimeDependencies(
                voice=voice_for_model(settings.speaking_model),
                engine_factory=lambda voice: speaking_for_voice(settings, voice),
                artifact_checker=lambda voice: voice in installed_voice_ids(settings),
            ),
        )

    def load(self) -> None:
        """Start the configured voice, then hearing; the orchestrator owns processes."""
        self.begin_loading()
        engine: SpeakingEngine | None = None
        registered = False
        try:
            engine = self._construct_engine()
            registered = self._register_pending_engine(engine)
            if registered:
                _load_or_raise(engine)
                self._publish(engine)
                self.hearing.load()
        finally:
            self._finish_loading(engine, registered=registered)

    def begin_loading(self) -> None:
        """Publish startup before its background task can be scheduled."""
        with self._state_lock:
            self.loading = True

    def health(self) -> dict[str, object]:
        """The contract body for GET /health."""
        self.reconcile_failed_active()
        return {
            "speaking": self._speaking_health(),
            "hearing": self.hearing.health(),
            "card_memory_mb": self._memory_probe(),
        }

    def capture_speaking(self) -> VoiceAdmission | None:
        """Capture one stable engine for an entire streaming response.

        It takes only the state lock, which every holder keeps briefly, so a
        caller on the event loop is never held up by a failed engine's close;
        `reconcile_failed_active` does that, after the refusal.
        """
        with self._state_lock:
            speaking = self.speaking
            if self.loading or self._stopping or speaking is None or not speaking.ready:
                return None
            return VoiceAdmission(speaking, self._dependencies.voice)

    def begin_shutdown(self) -> None:
        """Refuse new work before the server starts its own shutdown."""
        with self._state_changed:
            self._stopping = True
            self._state_changed.notify_all()

    def set_fatal_callback(self, callback: Callable[[], None]) -> None:
        """Route an owned speaking-engine failure to the service orchestrator."""
        with self._state_lock:
            self._fatal_callback = callback
            engines = [self._pending_engine, self.speaking]
        for engine in _unique_engines(engines):
            setter = getattr(engine, "set_fatal_callback", None)
            if callable(setter):
                setter(callback)

    def close(self) -> None:
        """Close the pending and the active engine outside the state lock."""
        with self._state_changed:
            self._stopping = True
            self._state_changed.notify_all()
            while self._closing_engine is not None:
                self._state_changed.wait()
            engines = [self._pending_engine, self.speaking]
            self._pending_engine = None
            self.speaking = None
            self._ready_speaking = None
        self.hearing.close()
        failed = False
        for engine in _unique_engines(engines):
            try:
                engine.close()
            except Exception:
                failed = True
                logger.exception("speaking engine cleanup failed")
        if failed:
            self._fatal_callback()
            message = "speaking engine cleanup failed"
            raise RuntimeError(message)

    def reconcile_failed_active(self) -> None:
        """Close an active engine that stopped being ready."""
        with self._state_lock:
            active = self.speaking
            if active is None:
                return
            if active.ready:
                self._ready_speaking = active
                return
            if active is not self._ready_speaking or self.loading or self._stopping:
                return
            self.speaking = None
            self._ready_speaking = None
            self._closing_engine = active
        try:
            active.close()
        except Exception:
            logger.exception("failed speaking engine cleanup failed")
            self._fatal_callback()
        finally:
            with self._state_changed:
                self._closing_engine = None
                self._state_changed.notify_all()

    def _construct_engine(self) -> SpeakingEngine:
        voice = self._dependencies.voice
        engine_factory = self._dependencies.engine_factory
        artifact_checker = self._dependencies.artifact_checker
        if engine_factory is None or artifact_checker is None:
            message = "the runtime has no engine factory"
            raise RuntimeError(message)
        if not artifact_checker(voice):
            message = f"voice {voice.value} is not installed"
            raise RuntimeError(message)
        engine = engine_factory(voice)
        setter = getattr(engine, "set_fatal_callback", None)
        if callable(setter):
            setter(self._fatal_callback)
        return engine

    def _register_pending_engine(self, engine: SpeakingEngine) -> bool:
        with self._state_lock:
            if self._stopping:
                return False
            self._pending_engine = engine
            return True

    def _publish(self, engine: SpeakingEngine) -> None:
        with self._state_lock:
            if self._stopping:
                return
            self.speaking = engine
            self._ready_speaking = engine if engine.ready else None
            self._pending_engine = None

    def _finish_loading(
        self, engine: SpeakingEngine | None, *, registered: bool
    ) -> None:
        with self._state_lock:
            self.loading = False
            still_pending = registered and self._pending_engine is engine
            if still_pending:
                self._pending_engine = None
        if engine is not None and (still_pending or not registered):
            try:
                engine.close()
            except Exception:
                logger.exception("uncommitted speaking engine cleanup failed")
                self._fatal_callback()

    def _speaking_health(self) -> dict[str, object]:
        with self._state_lock:
            speaking = self.speaking
        if speaking is None:
            return {
                "model": "unavailable",
                "ready": False,
                "streams": False,
                "sample_rate": 0,
            }
        return {
            "model": speaking.model_name,
            "ready": speaking.ready,
            "streams": speaking.streams,
            "sample_rate": speaking.sample_rate,
        }


def _unique_engines(
    engines: list[SpeakingEngine | None],
) -> list[SpeakingEngine]:
    unique: list[SpeakingEngine] = []
    identities: set[int] = set()
    for engine in engines:
        if engine is not None and id(engine) not in identities:
            identities.add(id(engine))
            unique.append(engine)
    return unique


@dataclass(frozen=True, slots=True)
class VoiceAdmission:
    """One captured engine and the voice it speaks for."""

    speaking: SpeakingEngine
    voice: VoiceId

    @property
    def sample_rate(self) -> int:
        """The captured engine's rate."""
        return self.speaking.sample_rate


def _load_or_raise(engine: SpeakingEngine) -> None:
    try:
        engine.load()
    except Exception:
        message = f"speaking model {engine.model_name} failed to load"
        logger.exception("%s", message)
        raise RuntimeError(message) from None


def wav_chunks(
    speaking: SpeakingEngine,
    text: str,
    language: str,
    sample_rate: int | None = None,
) -> Iterator[bytes]:
    """One WAV from `speaking`: its header leaves with the first PCM chunk."""
    first = True
    with PayloadFreeFailure(logger, "speaking failed", reraise_as=SpeakingFailedError):
        synthesis = speaking.pcm_chunks(text, language)
        try:
            for pcm in synthesis:
                if first:
                    yield wav_header(sample_rate or speaking.sample_rate) + pcm
                    first = False
                else:
                    yield pcm
        finally:
            synthesis.close()


def mount_health_route(app: FastAPI, runtime: Runtime) -> None:
    """Mount /health over the one runtime."""

    @app.get("/health")
    def health() -> dict[str, object]:
        return runtime.health()

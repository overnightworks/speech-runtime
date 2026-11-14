"""One voice's engine lifecycle and its admission gate."""

from __future__ import annotations

import logging
import threading
from concurrent.futures import Future
from typing import TYPE_CHECKING

from speech_client.wire import VOICES, VoiceHealth
from speech_service.admission_gate import AdmissionGate
from speech_service.failures import log_failure
from speech_service.voices import sample_rate_before_load

if TYPE_CHECKING:
    from collections.abc import Callable

    from speech_client.wire import VoiceId
    from speech_service.clock import LoopClock
    from speech_service.service import RuntimeDependencies, SpeakingEngine

logger = logging.getLogger(__name__)

_CLEANUP_FAILED = "speaking engine cleanup failed"


class VoiceSlot:
    """The engine one voice speaks with, from its build to its close."""

    def __init__(
        self,
        voice: VoiceId,
        dependencies: RuntimeDependencies,
        clock: LoopClock,
        speaking: SpeakingEngine | None = None,
    ) -> None:
        """Hold `speaking` if it is ready; otherwise nothing runs until a load."""
        self.voice = voice
        self.speaking = speaking
        self.gate = AdmissionGate(clock)
        self.loading = False
        self._dependencies = dependencies
        self._state_lock = threading.Lock()
        self._state_changed = threading.Condition(self._state_lock)
        self._ready_speaking = speaking if speaking and speaking.ready else None
        self._pending_engine: SpeakingEngine | None = None
        self._closing_engine: SpeakingEngine | None = None
        self._load_job: Future[SpeakingEngine | None] | None = None
        self._stopping = False
        self._cleanup_failed = False
        self._fatal_callback: Callable[[], None] = lambda: None

    def load(self) -> None:
        """Build and load the engine, then publish it."""
        engine: SpeakingEngine | None = None
        registered = False
        try:
            engine = self._construct_engine()
            registered = self._register_pending_engine(engine)
            if registered:
                _load_or_raise(self.voice, engine)
                self._publish(engine)
        finally:
            self._finish_loading(engine, registered=registered)

    def shared_load(self) -> Future[SpeakingEngine | None] | None:
        """The voice's one running load, started on its own thread when none runs.

        It resolves with the engine it left ready, or with `None` once a failed
        load has closed and reaped what it built; `None` once the slot stops.
        """
        with self._state_lock:
            if self._stopping:
                return None
            if self._load_job is not None:
                return self._load_job
            job: Future[SpeakingEngine | None] = Future()
            # A running future refuses `cancel`, so no waiter can end the shared load.
            job.set_running_or_notify_cancel()
            self._load_job = job
            self.loading = True
        threading.Thread(
            target=self._run_load_job,
            args=(job,),
            name=f"speech-voice-load-{self.voice.value}",
        ).start()
        return job

    def capture(self) -> SpeakingEngine | None:
        """Capture one stable engine for an entire streaming response.

        It takes only the state lock, which every holder keeps briefly, so a
        caller on the event loop is never held up by a failed engine's close;
        `reconcile_failed_active` or the next load does that, off the loop.
        """
        with self._state_lock:
            return self._capturable()

    def begin_shutdown(self) -> None:
        """Refuse new work before the server starts its own shutdown."""
        with self._state_changed:
            self._stopping = True
            self._state_changed.notify_all()

    def abandon_load(self) -> None:
        """Close the engine a load is building, so its waiters are refused.

        Blocks while the engine closes; call it off the event loop.
        """
        with self._state_lock:
            building, self._pending_engine = self._pending_engine, None
        if building is not None:
            self._closed(building, _CLEANUP_FAILED)

    def set_fatal_callback(self, callback: Callable[[], None]) -> None:
        """Route an owned engine failure to the service orchestrator."""
        with self._state_lock:
            self._fatal_callback = callback
            engines = [self._pending_engine, self.speaking]
        for engine in _unique_engines(engines):
            setter = getattr(engine, "set_fatal_callback", None)
            if callable(setter):
                setter(callback)

    def close(self) -> None:
        """Close the pending and the active engine, then wait for a running load.

        It raises once any cleanup of the slot failed, the running load's included.
        """
        with self._state_changed:
            self._stopping = True
            self._state_changed.notify_all()
            while self._closing_engine is not None:
                self._state_changed.wait()
            engines = [self._pending_engine, self.speaking]
            self._pending_engine = None
            self.speaking = None
            self._ready_speaking = None
        for engine in _unique_engines(engines):
            self._closed(engine, _CLEANUP_FAILED)
        with self._state_changed:
            while self._load_job is not None:
                self._state_changed.wait()
            cleanup_failed = self._cleanup_failed
        if cleanup_failed:
            raise RuntimeError(_CLEANUP_FAILED)

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
            self._closed(active, "failed speaking engine cleanup failed")
        finally:
            with self._state_changed:
                self._closing_engine = None
                self._state_changed.notify_all()

    def health(self, *, ready: bool) -> VoiceHealth:
        """The voice's part of GET /health: its catalogue facts, and `ready`."""
        with self._state_lock:
            speaking = self.speaking
        catalogued = VOICES[self.voice]
        loaded = speaking is not None
        return VoiceHealth(
            model=catalogued.model,
            name=catalogued.name,
            languages=catalogued.languages,
            takes_instructions=catalogued.takes_instructions,
            ready=loaded and ready,
            sample_rate=(
                speaking.sample_rate
                if speaking is not None
                else sample_rate_before_load(self.voice)
            ),
        )

    def _run_load_job(self, job: Future[SpeakingEngine | None]) -> None:
        try:
            if self._retire_dead_engine():
                self.load()
        except Exception as error:  # noqa: BLE001 - its waiters read the outcome as the slot's readiness
            log_failure(logger, f"voice {self.voice.value} failed to load", error)
        finally:
            with self._state_changed:
                self.loading = False
                self._load_job = None
                loaded = self._capturable()
                self._state_changed.notify_all()
            job.set_result(loaded)

    def _retire_dead_engine(self) -> bool:
        """Close the engine a crash left, before any new one; whether a load is due."""
        with self._state_changed:
            while self._closing_engine is not None:
                self._state_changed.wait()
            dead = self.speaking
            if dead is not None and dead.ready:
                return False
            self.speaking = None
            self._ready_speaking = None
        return dead is None or self._closed(dead, "dead speaking engine cleanup failed")

    def _closed(self, engine: SpeakingEngine, failure: str) -> bool:
        """Close `engine`; a failed close is fatal, because a worker may remain."""
        try:
            engine.close()
        except Exception:
            logger.exception("%s", failure)
            with self._state_lock:
                self._cleanup_failed = True
            self._fatal_callback()
            return False
        return True

    def _capturable(self) -> SpeakingEngine | None:
        """The engine a request may speak with now; the caller holds the state lock."""
        speaking = self.speaking
        if self.loading or self._stopping or speaking is None or not speaking.ready:
            return None
        return speaking

    def _construct_engine(self) -> SpeakingEngine:
        engine_factory = self._dependencies.engine_factory
        missing_artifacts = self._dependencies.missing_artifacts
        if engine_factory is None or missing_artifacts is None:
            message = "the runtime has no engine factory"
            raise RuntimeError(message)
        missing = missing_artifacts(self.voice)
        if missing:
            logger.error(
                "voice %s is missing %s; fetch it with speech-fetch-voice",
                self.voice.value,
                ", ".join(missing),
            )
            message = f"voice {self.voice.value} is not installed"
            raise RuntimeError(message)
        engine = engine_factory(self.voice)
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
            self._closed(engine, "uncommitted speaking engine cleanup failed")


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


def _load_or_raise(voice: VoiceId, engine: SpeakingEngine) -> None:
    try:
        engine.load()
    except Exception:
        message = f"voice {voice.value} failed to load"
        logger.exception("%s", message)
        raise RuntimeError(message) from None

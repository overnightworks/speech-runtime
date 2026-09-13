"""HTTP contract: health, a streaming speak, and a socket that hears."""

from __future__ import annotations

import asyncio
import logging
import os
import threading
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Annotated, Protocol

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field
from starlette.status import (
    HTTP_503_SERVICE_UNAVAILABLE,
    WS_1009_MESSAGE_TOO_BIG,
    WS_1011_INTERNAL_ERROR,
    WS_1013_TRY_AGAIN_LATER,
)
from starlette.websockets import WebSocketState

from speech.card import card_memory_mb
from speech.config import (
    CHATTERBOX_SPEAKING_MODEL,
    DEFAULT_SPEAKING_MODEL,
    HEAR_SAMPLE_RATE,
    load_settings,
)
from speech.hearing import (
    POLL_WAIT_SECONDS,
    FrameTooLargeError,
    HearingSession,
    hearing_from_settings,
)
from speech.pcm import wav_header
from speech.selection import (
    DurabilityUnconfirmedError,
    SelectionError,
    VoiceSelectionStore,
)
from speech.speaking import speaking_for_voice
from speech.voices import (
    VoiceId,
    VoiceLoadOutcome,
    VoiceRecovery,
    VoiceRecoveryKind,
    VoiceRuntimeState,
    VoiceSnapshot,
    installed_voice_ids,
    statuses,
)
from speech.wav_response import ClosingWavResponse

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Iterator

    from speech.config import Settings

_LOG = logging.getLogger(__name__)

SpeakText = Annotated[str, Field(min_length=1)]
HEARING_FAILED_REASON = "hearing failed"
FRAME_TOO_LARGE_REASON = "frame too large"


class SpeakRequest(BaseModel):
    """One sentence to speak."""

    text: SpeakText
    language: str


class SampleLanguage(StrEnum):
    """The two fixed phrases the private sample endpoint can synthesize."""

    GERMAN = "de"
    ENGLISH = "en"


_SAMPLE_TEXT: dict[SampleLanguage, str] = {
    SampleLanguage.GERMAN: "Hallo, ich bin die Stimme deiner Präsentation.",
    SampleLanguage.ENGLISH: "Hello, I am the voice of your presentation.",
}


class SpeakingEngine(Protocol):
    """A resident voice."""

    model_name: str
    sample_rate: int
    ready: bool
    streams: bool

    def load(self) -> None:
        """Load weights and become ready, or raise."""

    def pcm_chunks(self, text: str, language: str) -> Generator[bytes, None, None]:
        """16-bit mono PCM as soon as the model produces it."""

    def close(self) -> None:
        """Release owned resources once; retained pure engines may do nothing."""


class HearingEngine(Protocol):
    """A resident transcriber."""

    model_name: str
    ready: bool

    def load(self) -> None:
        """Load weights and become ready, or raise."""

    def open_session(self, language: str) -> HearingSession:
        """A new utterance stream on the already-loaded model."""


class Runtime:
    """The two resident models and the flags a caller waits on."""

    def __init__(
        self,
        speaking: SpeakingEngine | None,
        hearing: HearingEngine,
        *,
        debug: bool = False,
        memory_probe: Callable[[], int] | None = None,
        dependencies: RuntimeDependencies | None = None,
    ) -> None:
        """Hold the two engines. They are not loaded until `load`."""
        self.speaking = speaking
        self.hearing = hearing
        self.debug = debug
        self.loading = False
        self._memory_probe = memory_probe or card_memory_mb
        dependencies = dependencies or RuntimeDependencies()
        self._selection = dependencies.selection
        self._engine_factory = dependencies.engine_factory
        self._artifact_checker = dependencies.artifact_checker
        self._default_voice = dependencies.default_voice
        self._selected: VoiceId | None = (
            dependencies.default_voice if speaking is not None else None
        )
        self._pending: VoiceId | None = None
        self._pending_engine: SpeakingEngine | None = None
        self._closing_engine: SpeakingEngine | None = None
        self._admitted = 0
        self._stopping = False
        self._recovery: VoiceRecovery | None = None
        self._engines: dict[VoiceId, SpeakingEngine] = {}
        if speaking is not None and dependencies.default_voice is not None:
            self._engines[dependencies.default_voice] = speaking
        self._state_lock = threading.Lock()
        self._state_changed = threading.Condition(self._state_lock)
        self._transition_guard = threading.Lock()
        self._synthesis_gate = threading.Lock()
        self._fatal_callback: Callable[[], None] = lambda: None

    @classmethod
    def from_settings(cls, settings: Settings) -> Runtime:
        """Build the production engines from the environment."""
        default_voice = _voice_from_model(settings.speaking_model)
        return cls(
            None,
            hearing_from_settings(settings),
            debug=settings.debug,
            dependencies=RuntimeDependencies(
                selection=VoiceSelectionStore(
                    settings.state_directory, owner_uid=os.geteuid()
                ),
                engine_factory=lambda voice: speaking_for_voice(settings, voice),
                artifact_checker=lambda voice: voice in installed_voice_ids(settings),
                default_voice=default_voice,
            ),
        )

    def load(self, stopping: threading.Event | None = None) -> None:
        """Load both models, leaving process ownership to the orchestrator."""
        self.loading = True
        try:
            self._load_saved_selection()
            if stopping is not None and stopping.is_set():
                return
            _load_or_raise("hearing", self.hearing.model_name, self.hearing.load)
        finally:
            self.loading = False

    def health(self) -> dict[str, object]:
        """The contract body for GET /health."""
        self._reconcile_failed_active()
        return {
            "speaking": {
                **self._speaking_health(),
            },
            "hearing": {
                "model": self.hearing.model_name,
                "ready": self.hearing.ready,
            },
            "sample_rate": HEAR_SAMPLE_RATE,
            "card_memory_mb": self._memory_probe(),
        }

    def snapshot(self, settings: Settings) -> VoiceSnapshot:
        """Read all public row state while holding one short state lock."""
        self._reconcile_failed_active()
        with self._state_lock:
            speaking = self.speaking
            selected = self._selected
            recovery = self._recovery
            loading = self.loading
            pending = self._pending
        failed = (
            recovery.voice
            if recovery and recovery.kind is VoiceRecoveryKind.LOAD_FAILED
            else None
        )
        return VoiceSnapshot(
            voices=statuses(
                settings,
                VoiceRuntimeState(
                    selected=selected,
                    ready=speaking is not None and speaking.ready,
                    loading=loading,
                    pending=pending,
                    failed=failed,
                ),
            ),
            recovery=recovery,
        )

    def load_voice(self, voice: VoiceId) -> VoiceLoadOutcome | None:
        """Synchronously load, atomically select, and retain one downloaded baseline."""
        if not self._transition_guard.acquire(blocking=False):
            return None
        try:
            return self._activate_voice(voice)
        finally:
            self._transition_guard.release()

    def capture_speaking(self) -> _VoiceAdmission | None:
        """Capture one stable engine for an entire streaming response."""
        self._reconcile_failed_active()
        with self._state_lock:
            speaking = self.speaking
            if self.loading or self._stopping or speaking is None or not speaking.ready:
                return None
            return self._admit_locked(speaking)

    def preacquire_sample(self, voice: VoiceId) -> _SampleSynthesis | None:
        """Reserve one verified active voice without delaying an ongoing talk."""
        self._reconcile_failed_active()
        with self._state_lock:
            speaking = self.speaking
            if (
                self.loading
                or self._stopping
                or self._selected is not voice
                or speaking is None
                or not speaking.ready
            ):
                return None
            admission = self._admit_locked(speaking)
        if not self._synthesis_gate.acquire(blocking=False):
            admission.close()
            raise _SynthesisBusyError from None
        lease = _Lease(self._synthesis_gate.release)
        return _SampleSynthesis(admission, lease)

    def _admit_locked(self, speaking: SpeakingEngine) -> _VoiceAdmission:
        self._admitted += 1
        lifetime = _Lease(self._release_admission)
        return _VoiceAdmission(speaking, speaking.sample_rate, lifetime)

    def _release_admission(self) -> None:
        with self._state_changed:
            self._admitted -= 1
            if self._admitted == 0:
                self._state_changed.notify_all()

    def acquire_synthesis(self) -> _Lease:
        """Wait for the one request-synthesis owner before generating PCM."""
        self._synthesis_gate.acquire()
        return _Lease(self._synthesis_gate.release)

    def begin_shutdown(self) -> None:
        """Refuse new work before the server starts its own shutdown."""
        with self._state_changed:
            self._stopping = True
            self._state_changed.notify_all()

    def set_fatal_callback(self, callback: Callable[[], None]) -> None:
        """Route an owned speaking-engine failure to the service orchestrator."""
        with self._state_lock:
            self._fatal_callback = callback
            engines = [self._pending_engine, self.speaking, *self._engines.values()]
        for engine in _unique_engines(engines):
            setter = getattr(engine, "set_fatal_callback", None)
            if callable(setter):
                setter(callback)

    def close(self) -> None:
        """Close every retained or pending engine outside the state lock."""
        with self._state_changed:
            self._stopping = True
            self._state_changed.notify_all()
            while self._closing_engine is not None:
                self._state_changed.wait()
            engines = [self._pending_engine, self.speaking, *self._engines.values()]
            self._pending_engine = None
            self.speaking = None
            self._engines.clear()
        failed = False
        for engine in _unique_engines(engines):
            try:
                engine.close()
            except Exception:
                failed = True
                _LOG.exception("speaking engine cleanup failed")
        if failed:
            self._fatal_callback()
            message = "speaking engine cleanup failed"
            raise RuntimeError(message)

    def _load_saved_selection(self) -> None:
        selection = self._selection
        if selection is None:
            raise RuntimeError from None
        with self._transition_guard:
            try:
                voice = selection.read()
            except SelectionError:
                with self._state_lock:
                    self._recovery = VoiceRecovery(
                        kind=VoiceRecoveryKind.INVALID_SELECTION
                    )
                return
            selected = self._default_voice if voice is None else voice
            if selected is None:
                with self._state_lock:
                    self._recovery = VoiceRecovery(
                        kind=VoiceRecoveryKind.INVALID_SELECTION
                    )
                return
            self._activate_voice(selected)

    def _activate_voice(self, voice: VoiceId) -> VoiceLoadOutcome:
        with self._state_lock:
            if self._stopping:
                return VoiceLoadOutcome.NOT_ACTIVATED
            self.loading = True
            self._pending = voice
        engine: SpeakingEngine | None = None
        newly_constructed = False
        registered = False
        committed = False
        try:
            try:
                if not self._voice_dependencies_ready(voice):
                    return self._not_activated(voice)
                with self._state_lock:
                    engine = self._engines.get(voice)
                if engine is None:
                    engine = self._engine_factory(voice)
                    newly_constructed = True
                    setter = getattr(engine, "set_fatal_callback", None)
                    if callable(setter):
                        setter(self._fatal_callback)
                registered = self._register_pending_engine(engine)
                if not registered:
                    return VoiceLoadOutcome.NOT_ACTIVATED
                if newly_constructed or not engine.ready:
                    engine.load()
                with self._state_lock:
                    if self._stopping:
                        return VoiceLoadOutcome.NOT_ACTIVATED
                self._selection.write(voice)
                committed = True
                recovery = None
            except DurabilityUnconfirmedError:
                committed = True
                recovery = VoiceRecovery(
                    kind=VoiceRecoveryKind.DURABILITY_UNCONFIRMED, voice=voice
                )
            except Exception:
                _LOG.exception("voice %s failed to load", voice.value)
                return self._not_activated(voice)
            return self._replace_voice(voice, engine, recovery)
        finally:
            should_close = False
            with self._state_lock:
                self.loading = False
                self._pending = None
                if not committed and self._pending_engine is engine:
                    self._pending_engine = None
                    should_close = newly_constructed
                elif not registered:
                    should_close = newly_constructed
            if should_close and engine is not None:
                try:
                    engine.close()
                except Exception:
                    _LOG.exception("uncommitted speaking engine cleanup failed")
                    self._fatal_callback()

    def _voice_dependencies_ready(self, voice: VoiceId) -> bool:
        if (
            not self._selection
            or not self._engine_factory
            or not self._artifact_checker
        ):
            return False
        self._selection.preflight()
        return self._artifact_checker(voice)

    def _replace_voice(
        self, voice: VoiceId, engine: SpeakingEngine, recovery: VoiceRecovery | None
    ) -> VoiceLoadOutcome:
        closing: SpeakingEngine | None = None
        with self._state_changed:
            while self._admitted and not self._stopping:
                self._state_changed.wait()
            if self._stopping:
                return VoiceLoadOutcome.NOT_ACTIVATED
            if (
                self._selected is VoiceId.CHATTERBOX
                and self.speaking is not None
                and self.speaking is not engine
            ):
                closing = self.speaking
                self.speaking = None
                if self._engines.get(VoiceId.CHATTERBOX) is closing:
                    del self._engines[VoiceId.CHATTERBOX]
                self._closing_engine = closing
            else:
                self._publish_voice(voice, engine, recovery)
        if closing is not None:
            try:
                closing.close()
            except Exception:
                with self._state_changed:
                    self._closing_engine = None
                    self._engines[VoiceId.CHATTERBOX] = closing
                    self._state_changed.notify_all()
                self._fatal_callback()
                raise
            with self._state_changed:
                self._closing_engine = None
                self._state_changed.notify_all()
                if self._stopping:
                    return VoiceLoadOutcome.NOT_ACTIVATED
                self._publish_voice(voice, engine, recovery)
        if recovery is None:
            return VoiceLoadOutcome.ACTIVATED
        return VoiceLoadOutcome.ACTIVATED_DURABILITY_UNCONFIRMED

    def _publish_voice(
        self, voice: VoiceId, engine: SpeakingEngine, recovery: VoiceRecovery | None
    ) -> None:
        self.speaking = engine
        self._selected = voice
        self._engines[voice] = engine
        if self._pending_engine is engine:
            self._pending_engine = None
        self._recovery = recovery

    def _register_pending_engine(self, engine: SpeakingEngine) -> bool:
        with self._state_lock:
            if self._stopping:
                return False
            if self._pending_engine is not None:
                message = "voice activation already owns a pending engine"
                raise RuntimeError(message)
            self._pending_engine = engine
            return True

    def _not_activated(self, voice: VoiceId) -> VoiceLoadOutcome:
        with self._state_lock:
            self._recovery = VoiceRecovery(
                kind=VoiceRecoveryKind.LOAD_FAILED, voice=voice
            )
        return VoiceLoadOutcome.NOT_ACTIVATED

    def _reconcile_failed_active(self) -> None:
        if not self._transition_guard.acquire(blocking=False):
            return
        failed: SpeakingEngine | None = None
        try:
            with self._state_lock:
                active = self.speaking
                if active is None or active.ready or not active.streams:
                    return
                if self.loading or self._stopping:
                    return
                failed = active
                failed_voice = self._selected
                self.speaking = None
                if (
                    failed_voice is not None
                    and self._engines.get(failed_voice) is active
                ):
                    del self._engines[failed_voice]
                self._recovery = VoiceRecovery(
                    kind=VoiceRecoveryKind.LOAD_FAILED, voice=failed_voice
                )
                self._closing_engine = failed
            try:
                failed.close()
            except Exception:
                with self._state_lock:
                    if failed_voice is not None:
                        self._engines[failed_voice] = failed
                _LOG.exception("failed speaking engine cleanup failed")
                self._fatal_callback()
            finally:
                with self._state_changed:
                    self._closing_engine = None
                    self._state_changed.notify_all()
        finally:
            self._transition_guard.release()

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


def failed_to_load_message(which: str, model: str) -> str:
    """The line written when a named model cannot be loaded."""
    return f"{which} model {model} failed to load"


def _voice_from_model(model: str) -> VoiceId | None:
    if model == DEFAULT_SPEAKING_MODEL:
        return VoiceId.PIPER
    if model == CHATTERBOX_SPEAKING_MODEL:
        return VoiceId.CHATTERBOX
    return None


@dataclass(frozen=True, slots=True, kw_only=True)
class RuntimeDependencies:
    """Mutable-voice seams owned by the runtime's composition root."""

    selection: VoiceSelectionStore | None = None
    engine_factory: Callable[[VoiceId], SpeakingEngine] | None = None
    artifact_checker: Callable[[VoiceId], bool] | None = None
    default_voice: VoiceId | None = VoiceId.PIPER


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


class _SynthesisBusyError(RuntimeError):
    """The active voice is already generating a request."""


class _Lease:
    """One close-idempotent release callback."""

    def __init__(self, release: Callable[[], None]) -> None:
        self._release = release
        self._closed = False
        self._guard = threading.Lock()

    def close(self) -> None:
        with self._guard:
            if self._closed:
                return
            self._closed = True
            self._release()


@dataclass(frozen=True, slots=True)
class _VoiceAdmission:
    """One close-idempotent lifetime for a captured voice and native rate."""

    speaking: SpeakingEngine
    sample_rate: int
    lifetime: _Lease

    def close(self) -> None:
        self.lifetime.close()


@dataclass(frozen=True, slots=True)
class _SampleSynthesis:
    admission: _VoiceAdmission
    lease: _Lease


def _load_or_raise(which: str, model: str, load: Callable[[], None]) -> None:
    try:
        load()
    except Exception:
        message = failed_to_load_message(which, model)
        _LOG.exception("%s", message)
        raise RuntimeError(message) from None


def _log_text(*, debug: bool, kind: str, text: str) -> None:
    if debug:
        _LOG.debug("%s text: %s", kind, text)


def _wav_chunks(
    speaking: SpeakingEngine,
    text: str,
    language: str,
    *,
    sample_rate: int | None = None,
) -> Iterator[bytes]:
    first = True
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


def _public_wav_chunks(
    runtime: Runtime, admission: _VoiceAdmission, text: str, language: str
) -> Iterator[bytes]:
    """Acquire Runtime's gate only when a returned public stream first runs."""
    lease = runtime.acquire_synthesis()
    try:
        yield from _wav_chunks(
            admission.speaking,
            text,
            language,
            sample_rate=admission.sample_rate,
        )
    finally:
        lease.close()


def create_app(
    settings: Settings | None = None,
    runtime: Runtime | None = None,
) -> FastAPI:
    """Build the public application over the orchestrator-owned runtime."""
    if runtime is None:
        runtime = Runtime.from_settings(settings or load_settings())

    app = FastAPI(title="presentator-speech")
    app.state.runtime = runtime
    _mount_routes(app, runtime)
    return app


def create_control_app(settings: Settings, runtime: Runtime) -> FastAPI:
    """Build the private status and Load control surface over the shared runtime."""
    app = FastAPI(title="presentator-speech-control", docs_url=None, redoc_url=None)

    @app.get("/voices")
    def voices() -> VoiceSnapshot:
        return runtime.snapshot(settings)

    @app.post("/voices/{voice}/load")
    def load(voice: VoiceId) -> dict[str, VoiceLoadOutcome]:
        outcome = runtime.load_voice(voice)
        if outcome is None:
            raise HTTPException(
                status_code=409, detail="voice transition is in progress"
            )
        return {"outcome": outcome}

    @app.post("/voices/{voice}/sample/{language}")
    def sample(voice: VoiceId, language: SampleLanguage) -> ClosingWavResponse:
        try:
            admitted = runtime.preacquire_sample(voice)
        except _SynthesisBusyError:
            raise HTTPException(status_code=409, detail="speech is busy") from None
        if admitted is None:
            raise HTTPException(status_code=404, detail="active voice is unavailable")
        chunks = _wav_chunks(
            admitted.admission.speaking,
            _SAMPLE_TEXT[language],
            language.value,
            sample_rate=admitted.admission.sample_rate,
        )
        return ClosingWavResponse(chunks, admitted.admission, admitted.lease)

    return app


def _mount_routes(app: FastAPI, runtime: Runtime) -> None:
    @app.get("/health")
    def health() -> dict[str, object]:
        return runtime.health()

    @app.post("/speak")
    def speak(body: SpeakRequest) -> ClosingWavResponse:
        admitted = runtime.capture_speaking()
        if admitted is None:
            raise HTTPException(
                status_code=HTTP_503_SERVICE_UNAVAILABLE,
                detail="speaking model is not ready",
            )
        _log_text(debug=runtime.debug, kind="speak", text=body.text)
        return ClosingWavResponse(
            _public_wav_chunks(runtime, admitted, body.text, body.language), admitted
        )

    @app.websocket("/hear")
    async def hear(websocket: WebSocket, language: str = "de") -> None:
        await websocket.accept()
        if not runtime.hearing.ready:
            await websocket.close(code=WS_1013_TRY_AGAIN_LATER)
            return
        try:
            session = runtime.hearing.open_session(language)
            await _hear_loop(websocket, session, debug=runtime.debug)
            event = session.flush()
            if event is None or websocket.client_state != WebSocketState.CONNECTED:
                return
            _log_text(debug=runtime.debug, kind="hear", text=str(event["text"]))
            await websocket.send_json(event)
        except WebSocketDisconnect:
            return
        except FrameTooLargeError:
            await _close_open(
                websocket,
                code=WS_1009_MESSAGE_TOO_BIG,
                reason=FRAME_TOO_LARGE_REASON,
            )
        except Exception:
            _LOG.exception("hearing failed")
            await _close_open(
                websocket,
                code=WS_1011_INTERNAL_ERROR,
                reason=HEARING_FAILED_REASON,
            )


async def _close_open(websocket: WebSocket, *, code: int, reason: str) -> None:
    if websocket.client_state == WebSocketState.CONNECTED:
        await websocket.close(code=code, reason=reason)


async def _hear_loop(
    websocket: WebSocket,
    session: HearingSession,
    *,
    debug: bool,
) -> None:
    receive = asyncio.create_task(websocket.receive())
    tick = asyncio.create_task(asyncio.sleep(POLL_WAIT_SECONDS))
    try:
        while True:
            done, _pending = await asyncio.wait(
                {receive, tick},
                return_when=asyncio.FIRST_COMPLETED,
            )
            if receive in done:
                message = receive.result()
                if message.get("type") == "websocket.disconnect":
                    return
                data = message.get("bytes")
                if isinstance(data, (bytes, bytearray)):
                    session.add_pcm(bytes(data))
                receive = asyncio.create_task(websocket.receive())
            if tick in done:
                event = await asyncio.to_thread(session.poll)
                if event is not None:
                    _log_text(debug=debug, kind="hear", text=str(event["text"]))
                    await websocket.send_json(event)
                tick = asyncio.create_task(asyncio.sleep(POLL_WAIT_SECONDS))
    finally:
        receive.cancel()
        tick.cancel()

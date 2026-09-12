"""HTTP contract: health, a streaming speak, and a socket that hears."""

from __future__ import annotations

import asyncio
import logging
import os
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Protocol

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from starlette.concurrency import iterate_in_threadpool
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

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Iterator

    from starlette.types import Receive, Scope, Send

    from speech.config import Settings

_LOG = logging.getLogger(__name__)

SpeakText = Annotated[str, Field(min_length=1)]
HEARING_FAILED_REASON = "hearing failed"
FRAME_TOO_LARGE_REASON = "frame too large"


class SpeakRequest(BaseModel):
    """One sentence to speak."""

    text: SpeakText
    language: str


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
        self._recovery: VoiceRecovery | None = None
        self._engines: dict[VoiceId, SpeakingEngine] = {}
        if speaking is not None and dependencies.default_voice is not None:
            self._engines[dependencies.default_voice] = speaking
        self._state_lock = threading.Lock()
        self._transition_guard = threading.Lock()

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

    def capture_speaking(self) -> SpeakingEngine | None:
        """Capture one stable engine for an entire streaming response."""
        with self._state_lock:
            if self.loading:
                return None
            return self.speaking

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
            self.loading = True
            self._pending = voice
        try:
            if (
                self._selection is None
                or self._engine_factory is None
                or self._artifact_checker is None
            ):
                return self._not_activated(voice)
            self._selection.preflight()
            if not self._artifact_checker(voice):
                return self._not_activated(voice)
            engine = self._engines.get(voice)
            if engine is None:
                engine = self._engine_factory(voice)
                engine.load()
                self._engines[voice] = engine
            elif not engine.ready:
                engine.load()
            self._selection.write(voice)
        except DurabilityUnconfirmedError:
            with self._state_lock:
                self.speaking = engine
                self._selected = voice
                self._recovery = VoiceRecovery(
                    kind=VoiceRecoveryKind.DURABILITY_UNCONFIRMED, voice=voice
                )
            return VoiceLoadOutcome.ACTIVATED_DURABILITY_UNCONFIRMED
        except Exception:
            _LOG.exception("voice %s failed to load", voice.value)
            return self._not_activated(voice)
        else:
            with self._state_lock:
                self.speaking = engine
                self._selected = voice
                self._recovery = None
            return VoiceLoadOutcome.ACTIVATED
        finally:
            with self._state_lock:
                self.loading = False
                self._pending = None

    def _not_activated(self, voice: VoiceId) -> VoiceLoadOutcome:
        with self._state_lock:
            self._recovery = VoiceRecovery(
                kind=VoiceRecoveryKind.LOAD_FAILED, voice=voice
            )
        return VoiceLoadOutcome.NOT_ACTIVATED

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


def _wav_chunks(speaking: SpeakingEngine, text: str, language: str) -> Iterator[bytes]:
    first = True
    synthesis = speaking.pcm_chunks(text, language)
    try:
        for pcm in synthesis:
            if first:
                yield wav_header(speaking.sample_rate) + pcm
                first = False
            else:
                yield pcm
    finally:
        synthesis.close()


class _ClosingWavResponse(StreamingResponse):
    """A WAV stream that closes its synchronous generator once ASGI delivery ends.

    Starlette's ASGI 2.3 `__call__` waits for an in-flight thread-pool `next()`
    to return before delivering a client-disconnect cancellation, so by the
    time this `finally` runs, `chunks` is never mid-execution: closing it here
    releases synthesis (and the voice lock it holds) without relying on the
    cyclic garbage collector.
    """

    def __init__(self, chunks: Iterator[bytes]) -> None:
        """Wrap the sync generator for threaded iteration and remember it to close."""
        self._chunks = chunks
        super().__init__(iterate_in_threadpool(chunks), media_type="audio/wav")

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Stream as usual, then close the owned generator no matter the outcome."""
        try:
            await super().__call__(scope, receive, send)
        finally:
            self._chunks.close()


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

    return app


def _mount_routes(app: FastAPI, runtime: Runtime) -> None:
    @app.get("/health")
    def health() -> dict[str, object]:
        return runtime.health()

    @app.post("/speak")
    def speak(body: SpeakRequest) -> StreamingResponse:
        speaking = runtime.capture_speaking()
        if speaking is None or not speaking.ready:
            raise HTTPException(
                status_code=HTTP_503_SERVICE_UNAVAILABLE,
                detail="speaking model is not ready",
            )
        _log_text(debug=runtime.debug, kind="speak", text=body.text)
        return _ClosingWavResponse(_wav_chunks(speaking, body.text, body.language))

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

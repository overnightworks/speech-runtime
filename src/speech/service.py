"""HTTP contract: health, a streaming speak, and a socket that hears."""

from __future__ import annotations

import asyncio
import logging
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
from speech.config import HEAR_SAMPLE_RATE, load_settings
from speech.hearing import (
    POLL_WAIT_SECONDS,
    FrameTooLargeError,
    HearingSession,
    hearing_from_settings,
)
from speech.pcm import wav_header
from speech.speaking import speaking_from_settings
from speech.voices import VoiceStatus, statuses

if TYPE_CHECKING:
    import threading
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
        speaking: SpeakingEngine,
        hearing: HearingEngine,
        *,
        debug: bool = False,
        memory_probe: Callable[[], int] | None = None,
    ) -> None:
        """Hold the two engines. They are not loaded until `load`."""
        self.speaking = speaking
        self.hearing = hearing
        self.debug = debug
        self.loading = False
        self._memory_probe = memory_probe or card_memory_mb

    @classmethod
    def from_settings(cls, settings: Settings) -> Runtime:
        """Build the production engines from the environment."""
        return cls(
            speaking_from_settings(settings),
            hearing_from_settings(settings),
            debug=settings.debug,
        )

    def load(self, stopping: threading.Event | None = None) -> None:
        """Load both models, leaving process ownership to the orchestrator."""
        self.loading = True
        try:
            _load_or_raise("speaking", self.speaking.model_name, self.speaking.load)
            if stopping is not None and stopping.is_set():
                return
            _load_or_raise("hearing", self.hearing.model_name, self.hearing.load)
        finally:
            self.loading = False

    def health(self) -> dict[str, object]:
        """The contract body for GET /health."""
        return {
            "speaking": {
                "model": self.speaking.model_name,
                "ready": self.speaking.ready,
                "streams": self.speaking.streams,
                "sample_rate": self.speaking.sample_rate,
            },
            "hearing": {
                "model": self.hearing.model_name,
                "ready": self.hearing.ready,
            },
            "sample_rate": HEAR_SAMPLE_RATE,
            "card_memory_mb": self._memory_probe(),
        }


def failed_to_load_message(which: str, model: str) -> str:
    """The line written when a named model cannot be loaded."""
    return f"{which} model {model} failed to load"


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


def _wav_chunks(runtime: Runtime, text: str, language: str) -> Iterator[bytes]:
    first = True
    synthesis = runtime.speaking.pcm_chunks(text, language)
    try:
        for pcm in synthesis:
            if first:
                yield wav_header(runtime.speaking.sample_rate) + pcm
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
    """Build the private read-only control surface over the shared runtime."""
    app = FastAPI(title="presentator-speech-control", docs_url=None, redoc_url=None)

    @app.get("/voices")
    def voices() -> dict[str, tuple[VoiceStatus, ...]]:
        return {
            "voices": statuses(
                settings,
                ready=runtime.speaking.ready,
                loading=runtime.loading,
            )
        }

    return app


def _mount_routes(app: FastAPI, runtime: Runtime) -> None:
    @app.get("/health")
    def health() -> dict[str, object]:
        return runtime.health()

    @app.post("/speak")
    def speak(body: SpeakRequest) -> StreamingResponse:
        if not runtime.speaking.ready:
            raise HTTPException(
                status_code=HTTP_503_SERVICE_UNAVAILABLE,
                detail="speaking model is not ready",
            )
        _log_text(debug=runtime.debug, kind="speak", text=body.text)
        return _ClosingWavResponse(_wav_chunks(runtime, body.text, body.language))

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

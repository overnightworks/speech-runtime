"""HTTP contract: health, a streaming speak, and a socket that hears."""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from contextlib import asynccontextmanager
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

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable, Iterator

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

    def pcm_chunks(self, text: str, language: str) -> Iterator[bytes]:
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
        self._memory_probe = memory_probe or card_memory_mb

    @classmethod
    def from_settings(cls, settings: Settings) -> Runtime:
        """Build the production engines from the environment."""
        return cls(
            speaking_from_settings(settings),
            hearing_from_settings(settings),
            debug=settings.debug,
        )

    def load(self) -> None:
        """Load both models, exiting the process if either cannot."""
        _load_or_die("speaking", self.speaking.model_name, self.speaking.load)
        _load_or_die("hearing", self.hearing.model_name, self.hearing.load)

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


def _load_or_die(which: str, model: str, load: Callable[[], None]) -> None:
    try:
        load()
    except Exception:
        message = failed_to_load_message(which, model)
        _LOG.exception("%s", message)
        sys.stderr.write(f"{message}\n")
        os._exit(1)


def _log_text(*, debug: bool, kind: str, text: str) -> None:
    if debug:
        _LOG.debug("%s text: %s", kind, text)


def _wav_chunks(runtime: Runtime, text: str, language: str) -> Iterator[bytes]:
    first = True
    for pcm in runtime.speaking.pcm_chunks(text, language):
        if first:
            yield wav_header(runtime.speaking.sample_rate) + pcm
            first = False
        else:
            yield pcm


def create_app(
    settings: Settings | None = None,
    runtime: Runtime | None = None,
    *,
    load_models: bool = True,
) -> FastAPI:
    """The FastAPI app that holds both models."""
    settings = settings or load_settings()
    runtime = runtime or Runtime.from_settings(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        loader = None
        if load_models:
            loader = asyncio.create_task(asyncio.to_thread(runtime.load))
            app.state.loader = loader
        yield
        if loader is not None and not loader.done():
            loader.cancel()

    app = FastAPI(title="presentator-speech", lifespan=lifespan)
    app.state.runtime = runtime
    _mount_routes(app, runtime)
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
        return StreamingResponse(
            iterate_in_threadpool(_wav_chunks(runtime, body.text, body.language)),
            media_type="audio/wav",
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

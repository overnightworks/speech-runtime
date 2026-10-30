"""A real private-socket speech server for tests."""

import asyncio
import os
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING

import uvicorn

from speech_runtime_service.__main__ import PrivateSocket
from speech_runtime_service.config import Settings
from speech_runtime_service.control import create_control_app
from tests.support.fakes_support import GATE_TIMEOUT_SECONDS
from tests.support.identity_support import valid_callers

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path

    from starlette.types import Message, Receive, Scope, Send

    from speech_runtime_service.callers import SpeechCallers
    from speech_runtime_service.service import Runtime


@dataclass(frozen=True, slots=True)
class PrivateServer:
    """A running private-socket server and the event its handler sets."""

    path: Path
    speak_disconnect_received: asyncio.Event


@asynccontextmanager
async def serving_privately(
    directory: Path, runtime: Runtime, callers: SpeechCallers | None = None
) -> AsyncIterator[PrivateServer]:
    """Serve the private app on a real PrivateSocket and stop it on exit.

    The callers default to ones that accept `valid_headers()`.
    """
    settings = Settings(
        private_directory=directory, PRESENTATOR_RUNTIME_UID=os.geteuid()
    )
    app = create_control_app(settings, runtime, callers or valid_callers())
    speak_disconnect_received = asyncio.Event()

    async def observed(scope: Scope, receive: Receive, send: Send) -> None:
        async def watched_receive() -> Message:
            message = await receive()
            if message["type"] == "http.disconnect":
                speak_disconnect_received.set()
            return message

        await app(scope, watched_receive, send)

    server = uvicorn.Server(uvicorn.Config(observed, log_level="warning"))
    with PrivateSocket(directory, runtime_uid=os.geteuid()) as listener:
        serving = asyncio.create_task(server.serve(sockets=[listener]))
        try:
            yield PrivateServer(directory / "speech.sock", speak_disconnect_received)
        finally:
            server.should_exit = True
            await asyncio.wait_for(serving, timeout=GATE_TIMEOUT_SECONDS)

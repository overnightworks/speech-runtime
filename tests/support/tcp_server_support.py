"""A real TCP speech server on a free local port for tests."""

import asyncio
import socket
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING

import uvicorn

from speech_runtime_service.config import Settings
from speech_runtime_service.control import create_control_app
from tests.support.fakes_support import GATE_TIMEOUT_SECONDS
from tests.support.identity_support import valid_callers

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from starlette.types import Message, Receive, Scope, Send

    from speech_runtime_service.callers import SpeechCallers
    from speech_runtime_service.service import Runtime

_LOOPBACK = "127.0.0.1"


def free_loopback_port() -> int:
    """A port nothing listens on right now."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind((_LOOPBACK, 0))
        return probe.getsockname()[1]


def listens_on(port: int) -> bool:
    """Whether a TCP connection to the loopback port is accepted."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        return probe.connect_ex((_LOOPBACK, port)) == 0


@dataclass(frozen=True, slots=True)
class TcpServer:
    """A running TCP server and the event its handler sets."""

    base_url: str
    speak_disconnect_received: asyncio.Event


@asynccontextmanager
async def serving_over_tcp(
    runtime: Runtime, callers: SpeechCallers | None = None
) -> AsyncIterator[TcpServer]:
    """Serve the app on a free loopback port and stop it on exit.

    The callers default to ones that accept `valid_headers()`.
    """
    app = create_control_app(Settings(), runtime, callers or valid_callers())
    speak_disconnect_received = asyncio.Event()

    async def observed(scope: Scope, receive: Receive, send: Send) -> None:
        async def watched_receive() -> Message:
            message = await receive()
            if message["type"] == "http.disconnect":
                speak_disconnect_received.set()
            return message

        await app(scope, watched_receive, send)

    server = uvicorn.Server(uvicorn.Config(observed, log_level="warning"))
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind((_LOOPBACK, 0))
        listener.listen()
        serving = asyncio.create_task(server.serve(sockets=[listener]))
        try:
            yield TcpServer(
                f"http://{_LOOPBACK}:{listener.getsockname()[1]}",
                speak_disconnect_received,
            )
        finally:
            server.should_exit = True
            await asyncio.wait_for(serving, timeout=GATE_TIMEOUT_SECONDS)

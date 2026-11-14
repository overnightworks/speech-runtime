"""A test balancer that routes each connection by `/health`, as readiness does."""

from __future__ import annotations

import asyncio
import contextlib
import http.client
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from speech_client.wire import HEALTH_PATH, Health

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence

_LOOPBACK = "127.0.0.1"
_CHUNK_BYTES = 65_536
_HEALTH_TIMEOUT_SECONDS = 5


def health_of(port: int) -> Health | None:
    """The health the server on `port` answers, or `None` when it does not answer."""
    connection = http.client.HTTPConnection(
        _LOOPBACK, port, timeout=_HEALTH_TIMEOUT_SECONDS
    )
    try:
        connection.request("GET", HEALTH_PATH)
        return Health.model_validate_json(connection.getresponse().read())
    except OSError, http.client.HTTPException:
        return None
    finally:
        connection.close()


def is_ready(port: int) -> bool:
    """Whether the server on `port` can both speak and hear."""
    health = health_of(port)
    return health is not None and health.speaking.ready and health.hearing.ready


async def _pipe(source: asyncio.StreamReader, target: asyncio.StreamWriter) -> None:
    while data := await source.read(_CHUNK_BYTES):
        target.write(data)
        await target.drain()


async def _relay(
    client: tuple[asyncio.StreamReader, asyncio.StreamWriter],
    backend: tuple[asyncio.StreamReader, asyncio.StreamWriter],
) -> None:
    """Copy both directions until either side ends, then close both."""
    pipes = [
        asyncio.create_task(_pipe(client[0], backend[1])),
        asyncio.create_task(_pipe(backend[0], client[1])),
    ]
    try:
        await asyncio.wait(pipes, return_when=asyncio.FIRST_COMPLETED)
    finally:
        for pipe in pipes:
            pipe.cancel()
        await asyncio.gather(*pipes, return_exceptions=True)
        for writer in (client[1], backend[1]):
            writer.close()


@asynccontextmanager
async def balancing(backend_ports: Sequence[int]) -> AsyncIterator[int]:
    """Yield the port of a balancer sending each connection to the first ready backend.

    A connection finds no ready backend closed at once.
    """

    async def route(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        for port in backend_ports:
            if await asyncio.to_thread(is_ready, port):
                with contextlib.suppress(ConnectionError):
                    backend = await asyncio.open_connection(_LOOPBACK, port)
                    await _relay((reader, writer), backend)
                return
        writer.close()

    server = await asyncio.start_server(route, _LOOPBACK, 0)
    try:
        yield server.sockets[0].getsockname()[1]
    finally:
        server.close()
        await server.wait_closed()

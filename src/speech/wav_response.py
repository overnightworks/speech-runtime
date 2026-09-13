"""WAV response lifetime cleanup at the ASGI boundary."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from fastapi.responses import StreamingResponse
from starlette.concurrency import iterate_in_threadpool

if TYPE_CHECKING:
    from collections.abc import Generator

    from starlette.types import Receive, Scope, Send


class Closeable(Protocol):
    """A resource whose close operation is idempotent."""

    def close(self) -> None:
        """Release the owned resource once."""


class ClosingWavResponse(StreamingResponse):
    """Close a WAV generator and its leases when ASGI delivery ends."""

    def __init__(
        self,
        chunks: Generator[bytes, None, None],
        admission: Closeable,
        lease: Closeable | None = None,
    ) -> None:
        """Attach every already-owned response-lifetime resource."""
        self._chunks = chunks
        self._admission = admission
        self._lease = lease
        try:
            super().__init__(iterate_in_threadpool(chunks), media_type="audio/wav")
        except Exception:
            self._close_owned()
            raise

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Stream the response and release ownership on every exit."""
        try:
            await super().__call__(scope, receive, send)
        finally:
            self._close_owned()

    def _close_owned(self) -> None:
        try:
            self._chunks.close()
        finally:
            try:
                if self._lease is not None:
                    self._lease.close()
            finally:
                self._admission.close()

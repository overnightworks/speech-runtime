"""Start the public speech API and its private status listener together."""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import socket
import stat
import threading
from contextlib import AbstractContextManager
from typing import TYPE_CHECKING

import uvicorn

from speech.config import Settings, load_settings
from speech.cuda_libs import prepare_cuda_libraries
from speech.service import Runtime, create_app, create_control_app

if TYPE_CHECKING:
    from pathlib import Path

_SOCKET_NAME = "speech.sock"
_DIRECTORY_MODE = 0o700
_SOCKET_MODE = 0o600


class PrivateSocket(AbstractContextManager[socket.socket]):
    """Own exactly the UDS inode this speech process bound."""

    def __init__(self, directory: Path, *, runtime_uid: int) -> None:
        """Remember the directory and identity that may own this listener."""
        self._directory = directory
        self._runtime_uid = runtime_uid
        self._path = directory / _SOCKET_NAME
        self._socket: socket.socket | None = None
        self._inode: tuple[int, int] | None = None

    def __enter__(self) -> socket.socket:
        """Validate the directory and bind a new private socket inode."""
        self._prepare_directory()
        if self._path.exists() or self._path.is_symlink():
            message = f"private speech socket already exists: {self._path}"
            raise FileExistsError(message)
        bound = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            bound.bind(str(self._path))
            inode = self._path.stat()
            self._socket = bound
            self._inode = (inode.st_dev, inode.st_ino)
            self._path.chmod(_SOCKET_MODE)
            bound.listen()
        except BaseException:
            bound.close()
            self._remove_if_owned()
            raise
        return bound

    def __exit__(self, _type: object, _value: object, _traceback: object) -> None:
        """Close and remove only the inode this context created."""
        if self._socket is not None:
            self._socket.close()
        self._remove_if_owned()

    def _prepare_directory(self) -> None:
        self._directory.mkdir(mode=_DIRECTORY_MODE, parents=True, exist_ok=True)
        directory = self._directory.stat()
        if (
            not stat.S_ISDIR(directory.st_mode)
            or directory.st_uid != self._runtime_uid
            or stat.S_IMODE(directory.st_mode) != _DIRECTORY_MODE
        ):
            message = f"private speech directory is unsafe: {self._directory}"
            raise PermissionError(message)

    def _remove_if_owned(self) -> None:
        if self._inode is None:
            return
        try:
            current = self._path.stat()
        except FileNotFoundError:
            return
        if (current.st_dev, current.st_ino) == self._inode:
            self._path.unlink()


def main() -> None:
    """Read one configuration and exit nonzero after a fatal lifecycle error."""
    settings = load_settings()
    logging.basicConfig(
        level=logging.DEBUG if settings.debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    raise SystemExit(asyncio.run(serve(settings)))


async def serve(settings: Settings) -> int:
    """Run both servers until a signal or fatal lifecycle outcome."""
    runtime_uid = _require_runtime_uid(settings)
    prepare_cuda_libraries()
    runtime = Runtime.from_settings(settings)
    public = _server(create_app(settings, runtime), settings)
    private = _server(create_control_app(settings, runtime), settings)
    stopping = asyncio.Event()
    loading_stop = threading.Event()
    _install_signal_handlers(stopping, loading_stop)
    with PrivateSocket(settings.private_directory, runtime_uid=runtime_uid) as uds:
        public_task = asyncio.create_task(public.serve())
        private_task = asyncio.create_task(private.serve(sockets=[uds]))
        runtime.loading = True
        loader = asyncio.create_task(asyncio.to_thread(runtime.load, loading_stop))
        stopped = asyncio.create_task(stopping.wait())
        try:
            done, _pending = await asyncio.wait(
                {loader, public_task, private_task, stopped},
                return_when=asyncio.FIRST_COMPLETED,
            )
            if stopped in done:
                return 0
            if loader in done:
                if loader.exception() is not None:
                    logging.getLogger(__name__).error(
                        "speech model loading failed", exc_info=loader.exception()
                    )
                    return 1
                done, _pending = await asyncio.wait(
                    {public_task, private_task, stopped},
                    return_when=asyncio.FIRST_COMPLETED,
                )
                return 0 if stopped in done else 1
            return 1
        finally:
            loading_stop.set()
            public.should_exit = True
            private.should_exit = True
            stopped.cancel()
            await asyncio.gather(
                public_task, private_task, loader, return_exceptions=True
            )


def _server(app: object, settings: Settings) -> uvicorn.Server:
    server = uvicorn.Server(
        uvicorn.Config(app, host=settings.host, port=settings.port, log_level="info")
    )
    server.install_signal_handlers = lambda: None
    return server


def _require_runtime_uid(settings: Settings) -> int:
    """Require the configured shared identity before any runtime work starts."""
    if settings.runtime_uid is None:
        message = "PRESENTATOR_RUNTIME_UID is required for private speech transport"
        raise PermissionError(message)
    if settings.runtime_uid != os.geteuid():
        message = "PRESENTATOR_RUNTIME_UID must match the effective UID"
        raise PermissionError(message)
    return settings.runtime_uid


def _install_signal_handlers(
    stopping: asyncio.Event, loading_stop: threading.Event
) -> None:
    def stop() -> None:
        loading_stop.set()
        stopping.set()

    loop = asyncio.get_running_loop()
    for received in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(received, stop)

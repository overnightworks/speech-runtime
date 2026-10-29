"""Start the speech service on its private socket."""

from __future__ import annotations

import asyncio
import fcntl
import logging
import os
import signal
import socket
import stat
import threading
from contextlib import AbstractContextManager
from typing import TYPE_CHECKING

import uvicorn

from speech_runtime_service.config import Settings, load_settings
from speech_runtime_service.control import create_control_app
from speech_runtime_service.cuda_libs import prepare_cuda_libraries
from speech_runtime_service.service import Runtime

if TYPE_CHECKING:
    from pathlib import Path

_SOCKET_NAME = "speech.sock"
_DIRECTORY_MODE = 0o700
_SOCKET_MODE = 0o600
_LIVENESS_TIMEOUT_SECONDS = 0.2
_LOCK_SUFFIX = ".lock"


class PrivateSocket(AbstractContextManager[socket.socket]):
    """Own exactly the UDS inode this speech process bound."""

    def __init__(self, directory: Path, *, runtime_uid: int) -> None:
        """Remember the directory and identity that may own this listener."""
        self._directory = directory
        self._runtime_uid = runtime_uid
        self._path = directory / _SOCKET_NAME
        self._lock_path = directory / (_SOCKET_NAME + _LOCK_SUFFIX)
        self._socket: socket.socket | None = None
        self._inode: tuple[int, int] | None = None
        self._lock_fd: int | None = None

    def __enter__(self) -> socket.socket:
        """Validate the directory and bind a new private socket inode."""
        self._prepare_directory()
        self._lock_fd = self._acquire_start_lock()
        try:
            self._reclaim_stale_socket()
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
        except BaseException:
            self._release_start_lock()
            raise
        return bound

    def __exit__(self, _type: object, _value: object, _traceback: object) -> None:
        """Close and remove only the inode this context created."""
        if self._socket is not None:
            self._socket.close()
        self._remove_if_owned()
        self._release_start_lock()

    def _acquire_start_lock(self) -> int:
        """Hold the one exclusive start lock for this socket's whole lifetime."""
        fd = os.open(self._lock_path, os.O_CREAT | os.O_RDWR, _SOCKET_MODE)
        os.fchmod(fd, _SOCKET_MODE)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(fd)
            message = (
                f"another speech service is starting or running: {self._lock_path}"
            )
            raise FileExistsError(message) from None
        return fd

    def _release_start_lock(self) -> None:
        """Unlock and close only; never unlink.

        Unlinking here would let a third starter reuse this path's inode for
        a brand-new lock while a second starter (who acquired it between this
        release and the unlink) still believes it holds the one true lock.
        The file's existence never means a lock is held; the kernel lock
        already ended when this descriptor closes.
        """
        if self._lock_fd is None:
            return
        try:
            fcntl.flock(self._lock_fd, fcntl.LOCK_UN)
        finally:
            os.close(self._lock_fd)
        self._lock_fd = None

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

    def _reclaim_stale_socket(self) -> None:
        """Remove a leftover socket inode; refuse anything live, foreign, or swapped."""
        try:
            before = self._path.lstat()
        except FileNotFoundError:
            return
        if not stat.S_ISSOCK(before.st_mode) or before.st_uid != self._runtime_uid:
            message = f"private speech socket already exists: {self._path}"
            raise FileExistsError(message)
        if not _probe_is_dead(self._path):
            message = f"private speech socket already exists: {self._path}"
            raise FileExistsError(message)
        try:
            after = self._path.lstat()
        except FileNotFoundError:
            return
        if not _same_socket_inode(before, after, runtime_uid=self._runtime_uid):
            message = f"private speech socket changed during reclamation: {self._path}"
            raise FileExistsError(message)
        self._path.unlink()

    def _remove_if_owned(self) -> None:
        if self._inode is None:
            return
        try:
            current = self._path.stat()
        except FileNotFoundError:
            return
        if (current.st_dev, current.st_ino) == self._inode:
            self._path.unlink()


def _probe_is_dead(path: Path) -> bool:
    """Return True when nothing accepts connections on the socket path."""
    probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        probe.settimeout(_LIVENESS_TIMEOUT_SECONDS)
        probe.connect(str(path))
    except (ConnectionRefusedError, FileNotFoundError):
        return True
    else:
        return False
    finally:
        probe.close()


def _same_socket_inode(
    before: os.stat_result, after: os.stat_result, *, runtime_uid: int
) -> bool:
    return (
        stat.S_ISSOCK(after.st_mode)
        and after.st_dev == before.st_dev
        and after.st_ino == before.st_ino
        and after.st_uid == runtime_uid
    )


def main() -> None:
    """Read one configuration and exit nonzero after a fatal lifecycle error."""
    settings = load_settings()
    logging.basicConfig(
        level=logging.DEBUG if settings.debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    raise SystemExit(asyncio.run(serve(settings)))


async def serve(settings: Settings) -> int:
    """Run the server until a signal or fatal lifecycle outcome."""
    runtime_uid = _require_runtime_uid(settings)
    prepare_cuda_libraries()
    loop = asyncio.get_running_loop()
    fatal = asyncio.Event()
    runtime = Runtime.from_settings(settings)
    runtime.set_fatal_callback(lambda: loop.call_soon_threadsafe(fatal.set))
    server = _server(create_control_app(settings, runtime))
    stopping = asyncio.Event()
    loading_stop = threading.Event()
    _install_signal_handlers(stopping, loading_stop)
    with PrivateSocket(settings.private_directory, runtime_uid=runtime_uid) as uds:
        server_task = asyncio.create_task(server.serve(sockets=[uds]))
        runtime.begin_loading()
        loader = asyncio.create_task(asyncio.to_thread(runtime.load, loading_stop))
        stopped = asyncio.create_task(stopping.wait())
        fatal_task = asyncio.create_task(fatal.wait())
        result = 1
        try:
            done, _pending = await asyncio.wait(
                {loader, server_task, stopped, fatal_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            if fatal_task in done:
                result = 1
            elif stopped in done:
                result = 0
            elif loader in done:
                if loader.exception() is not None:
                    logging.getLogger(__name__).error(
                        "speech model loading failed", exc_info=loader.exception()
                    )
                    result = 1
                else:
                    done, _pending = await asyncio.wait(
                        {server_task, stopped, fatal_task},
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    result = 0 if stopped in done else 1
        finally:
            runtime.begin_shutdown()
            loading_stop.set()
            server.should_exit = True
            stopped.cancel()
            fatal_task.cancel()
            try:
                await asyncio.to_thread(runtime.close)
            except RuntimeError:
                logging.getLogger(__name__).exception("speech model cleanup failed")
                result = 1
            await asyncio.gather(server_task, loader, return_exceptions=True)
        return result


def _server(app: object) -> uvicorn.Server:
    return uvicorn.Server(uvicorn.Config(app, log_level="info"))


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


if __name__ == "__main__":
    main()

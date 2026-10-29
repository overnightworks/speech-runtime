"""Parent adapter for one isolated local speech provider."""

from __future__ import annotations

import logging
import os
import queue
import selectors
import signal
import stat
import subprocess
import threading
import time
from contextlib import suppress
from dataclasses import dataclass
from enum import Enum, StrEnum
from typing import IO, TYPE_CHECKING

from speech_runtime_provider_contract import (
    Frame,
    FrameDecoder,
    FrameKind,
    ProtocolError,
    ready_sample_rate,
    write_frame,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Generator
    from pathlib import Path

logger = logging.getLogger(__name__)

TERMINATE_GRACE_SECONDS = 5
CANCEL_DRAIN_SECONDS = 5
_COMMAND_WAIT_SECONDS = 0.25
_READ_SIZE = 64 * 1024
_NVIDIA_LIBRARY_DIRECTORIES = (
    "cublas:cuda_cupti:cuda_nvrtc:cuda_runtime:cudnn:cufft:curand:"
    "cusolver:cusparse:cusparselt:nccl:nvjitlink:nvtx"
)


class ProviderProtocolError(RuntimeError):
    """The provider broke its private, sanitized wire contract."""


class ProviderFailedError(RuntimeError):
    """The provider failed and cannot serve requests."""

    def __init__(self) -> None:
        """Carry the one message every failed provider reports."""
        super().__init__("speech provider failed")


class ProviderSynthesisFailedError(RuntimeError):
    """The provider reported that one synthesis request failed."""

    def __init__(self) -> None:
        """Carry the one message every failed synthesis reports."""
        super().__init__("speech provider synthesis failed")


class ProviderCleanupFailedError(RuntimeError):
    """The provider process could not be terminated and reaped."""

    def __init__(self) -> None:
        """Carry the one message every failed cleanup reports."""
        super().__init__("speech provider cleanup failed")


class ProviderUnavailableError(RuntimeError):
    """The configured provider executable cannot be launched."""

    def __init__(self) -> None:
        """Carry the one message every unavailable provider reports."""
        super().__init__("speech provider unavailable")


class ProviderNotReadyError(RuntimeError):
    """The provider is not in a state that accepts requests."""

    def __init__(self) -> None:
        """Carry the one message every not-ready provider reports."""
        super().__init__("speech provider is not ready")


class ProviderRequestIdsExhaustedError(RuntimeError):
    """The provider used every request id its wire format can carry."""

    def __init__(self) -> None:
        """Carry the one message every id exhaustion reports."""
        super().__init__("speech provider request ids exhausted")


class UnsupportedLanguageError(ValueError):
    """The requested language is not one the provider speaks."""

    def __init__(self) -> None:
        """Carry the one message every unsupported language reports."""
        super().__init__("unsupported speech provider language")


class ProviderId(StrEnum):
    """The provider directories and executable suffixes the service may use."""

    CHATTERBOX = "chatterbox"
    QWEN = "qwen"
    VOXCPM = "voxcpm"
    MAGPIE = "magpie"


class _OwnerExit(Enum):
    ORDINARY_CLOSE = "ordinary_close"
    PROVIDER_FAILURE = "provider_failure"


@dataclass(frozen=True, slots=True)
class ProviderLaunch:
    """Immutable executable and voice facts for one isolated provider."""

    model_name: str
    sample_rate: int
    streams: bool
    executable: Path | None
    arguments: tuple[str, ...]


class ProviderProcess:
    """One retained provider process, owned by a dedicated creator thread."""

    def __init__(
        self,
        launch: ProviderLaunch,
        *,
        fatal_callback: Callable[[], None] | None = None,
    ) -> None:
        """Store immutable launch data; construction never creates a process."""
        self.model_name = launch.model_name
        self.sample_rate = launch.sample_rate
        self.streams = launch.streams
        self.retain_when_inactive = False
        self._executable = launch.executable
        self._arguments = launch.arguments
        self._fatal_callback = fatal_callback or (lambda: None)
        self.ready = False
        self._state_lock = threading.Lock()
        self._commands: queue.Queue[_OwnerExit] = queue.Queue(maxsize=1)
        self._ready = threading.Event()
        self._stopped = threading.Event()
        self._request_finished = threading.Event()
        self._request_finished.set()
        self._owner: threading.Thread | None = None
        self._process: subprocess.Popen[bytes] | None = None
        self._response_decoder: FrameDecoder | None = None
        self._failed = False
        self._closing = False
        self._cleanup_failed = False
        self._next_request_id = 1

    def load(self) -> None:
        """Start the stable creator and wait for its single READY handshake."""
        with self._state_lock:
            if self._failed or self._closing:
                raise ProviderFailedError
            if self.ready:
                return
            if self._owner is None:
                self._owner = threading.Thread(
                    target=self._run_owner,
                    name="speech-provider-owner",
                )
                self._owner.start()
        self._ready.wait()
        with self._state_lock:
            ready = self.ready
        if not ready:
            raise ProviderFailedError

    def pcm_chunks(self, text: str, language: str) -> Generator[bytes]:
        """Stream one request and cancel/drain it if the response closes early."""
        normalized = language.strip().lower().replace("_", "-")
        language_code = {"de": 1, "en": 2}.get((normalized or "de").split("-", 1)[0])
        if language_code is None:
            raise UnsupportedLanguageError
        payload = bytes([language_code]) + text.encode("utf-8")
        process = self._live_process()
        completed = False
        request_id = 0
        try:
            request_id = self._allocate_request_id()
            self._write(process, Frame(FrameKind.SYNTHESIZE, request_id, payload))
            while True:
                frame = self._read(process)
                self._validate_response(frame, request_id, cancelling=False)
                if frame.kind is FrameKind.PCM:
                    yield frame.payload
                    continue
                if frame.kind is FrameKind.FAILED:
                    self._mark_failed()
                    raise ProviderSynthesisFailedError
                completed = True
                return
        finally:
            try:
                if request_id and not completed:
                    self._cancel_and_drain(process, request_id)
            finally:
                self._request_finished.set()

    def close(self) -> None:
        """Ask the owner to terminate and reap; this operation is idempotent."""
        with self._state_lock:
            self._closing = True
            owner = self._owner
        if owner is None:
            return
        self._request_finished.wait(CANCEL_DRAIN_SECONDS + _COMMAND_WAIT_SECONDS)
        self._request_owner_stop(_OwnerExit.ORDINARY_CLOSE)
        if not self._stopped.wait((2 * TERMINATE_GRACE_SECONDS) + 2):
            self._fatal_callback()
            raise ProviderCleanupFailedError
        owner.join(timeout=1)
        if owner.is_alive() or self._cleanup_failed:
            self._fatal_callback()
            raise ProviderCleanupFailedError

    def set_fatal_callback(self, callback: Callable[[], None]) -> None:
        """Give the runtime its process-lifecycle failure notification seam."""
        with self._state_lock:
            self._fatal_callback = callback

    def _run_owner(self) -> None:
        child: subprocess.Popen[bytes] | None = None
        lifecycle_failed = False
        try:
            child = self._spawn()
            decoder = FrameDecoder()
            os.set_blocking(_require_stdout(child).fileno(), False)
            with self._state_lock:
                self._process = child
                self._response_decoder = decoder
                closing = self._closing
            outcome = (
                _OwnerExit.ORDINARY_CLOSE
                if closing
                else self._serve_owner(child, decoder)
            )
            if outcome is _OwnerExit.PROVIDER_FAILURE:
                self._set_failed()
        except Exception:
            logger.exception("speech provider owner thread failed")
            lifecycle_failed = True
            self._set_failed()
        finally:
            self._ready.set()
            try:
                self._terminate_and_reap(child)
            except Exception:
                logger.exception("speech provider cleanup failed")
                lifecycle_failed = True
                self._cleanup_failed = True
            with self._state_lock:
                self._process = None
                self._response_decoder = None
                self.ready = False
            self._stopped.set()
            if child is not None and lifecycle_failed:
                self._fatal_callback()

    def _serve_owner(
        self, child: subprocess.Popen[bytes], decoder: FrameDecoder
    ) -> _OwnerExit:
        try:
            frame = self._read_available_frame(child, decoder)
        except EOFError, OSError, ProtocolError:
            return _OwnerExit.PROVIDER_FAILURE
        if (
            frame.kind is not FrameKind.READY
            or frame.request_id != 0
            or ready_sample_rate(frame.payload) != self.sample_rate
        ):
            return _OwnerExit.PROVIDER_FAILURE
        with self._state_lock:
            self.ready = True
        self._ready.set()
        while True:
            try:
                command = self._commands.get(timeout=_COMMAND_WAIT_SECONDS)
            except queue.Empty:
                if child.poll() is not None:
                    return _OwnerExit.PROVIDER_FAILURE
                continue
            return command

    def _spawn(self) -> subprocess.Popen[bytes]:
        entrypoint = self._executable
        if not _is_executable_file(entrypoint):
            raise ProviderUnavailableError
        provider_directory = entrypoint.parents[2]
        return subprocess.Popen(  # noqa: S603 - argv is the fixed provider entrypoint and flags with no shell
            [
                str(entrypoint),
                "--expected-parent-pid",
                str(os.getpid()),
                *self._arguments,
            ],
            shell=False,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            cwd=provider_directory,
            env=provider_environment(provider_directory),
            bufsize=0,
            close_fds=True,
            start_new_session=True,
        )

    def _live_process(self) -> subprocess.Popen[bytes]:
        with self._state_lock:
            process = self._process
            if process is None or not self.ready or self._closing or self._failed:
                raise ProviderNotReadyError
            self._request_finished.clear()
            return process

    def _allocate_request_id(self) -> int:
        with self._state_lock:
            request_id = self._next_request_id
            if request_id >= 1 << 64:
                self._failed = True
                self.ready = False
                self._request_owner_stop(_OwnerExit.PROVIDER_FAILURE)
                raise ProviderRequestIdsExhaustedError
            self._next_request_id += 1
            return request_id

    def _write(self, process: subprocess.Popen[bytes], frame: Frame) -> None:
        try:
            write_frame(process.stdin, frame)
        except (OSError, ValueError) as error:
            self._mark_failed()
            raise ProviderProtocolError from error

    def _read(self, process: subprocess.Popen[bytes]) -> Frame:
        with self._state_lock:
            decoder = self._response_decoder
        if decoder is None:
            self._mark_failed()
            raise ProviderProtocolError
        try:
            return self._read_available_frame(process, decoder)
        except (EOFError, OSError, ValueError) as error:
            self._mark_failed()
            raise ProviderProtocolError from error

    def _read_available_frame(
        self,
        process: subprocess.Popen[bytes],
        decoder: FrameDecoder,
        *,
        deadline: float | None = None,
    ) -> Frame:
        if process.stdout is None:
            raise EOFError
        with selectors.DefaultSelector() as selector:
            selector.register(process.stdout, selectors.EVENT_READ)
            while True:
                frame = decoder.next_frame()
                if frame is not None:
                    return frame
                wait = self._next_wait_seconds(deadline)
                if selector.select(wait):
                    self._feed_available_chunk(process.stdout, decoder)
                elif process.poll() is not None:
                    raise EOFError

    def _next_wait_seconds(self, deadline: float | None) -> float:
        if deadline is None:
            if self._closing:
                raise EOFError
            return _COMMAND_WAIT_SECONDS
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError
        return min(_COMMAND_WAIT_SECONDS, remaining)

    @staticmethod
    def _feed_available_chunk(stdout: IO[bytes], decoder: FrameDecoder) -> None:
        chunk = os.read(stdout.fileno(), _READ_SIZE)
        if not chunk:
            decoder.finish()
            raise EOFError
        decoder.feed(chunk)

    def _validate_response(
        self, frame: Frame, request_id: int, *, cancelling: bool
    ) -> None:
        allowed = {FrameKind.PCM, FrameKind.COMPLETE, FrameKind.FAILED}
        if cancelling:
            allowed.add(FrameKind.CANCELLED)
        if frame.request_id != request_id or frame.kind not in allowed:
            self._mark_failed()
            raise ProviderProtocolError

    def _cancel_and_drain(
        self, process: subprocess.Popen[bytes], request_id: int
    ) -> None:
        try:
            self._drain_cancelled(process, request_id)
        except ProviderProtocolError, OSError, ValueError, EOFError:
            self._mark_failed()

    def _drain_cancelled(
        self, process: subprocess.Popen[bytes], request_id: int
    ) -> None:
        self._write(process, Frame(FrameKind.CANCEL, request_id, b""))
        deadline = time.monotonic() + CANCEL_DRAIN_SECONDS
        with self._state_lock:
            decoder = self._response_decoder
        if decoder is None:
            raise ProviderProtocolError
        while True:
            frame = self._read_available_frame(process, decoder, deadline=deadline)
            self._validate_response(frame, request_id, cancelling=True)
            if frame.kind in {FrameKind.COMPLETE, FrameKind.CANCELLED}:
                return
            if frame.kind is FrameKind.FAILED:
                raise ProviderProtocolError

    def _mark_failed(self) -> None:
        self._set_failed()
        self._request_owner_stop(_OwnerExit.PROVIDER_FAILURE)

    def _set_failed(self) -> None:
        with self._state_lock:
            self._failed = True
            self.ready = False

    def _request_owner_stop(self, command: _OwnerExit) -> None:
        with suppress(queue.Full):
            self._commands.put_nowait(command)

    def _terminate_and_reap(self, child: subprocess.Popen[bytes] | None) -> None:
        if child is None:
            return
        if child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=TERMINATE_GRACE_SECONDS)
            except subprocess.TimeoutExpired:
                if child.poll() is None:
                    os.killpg(child.pid, signal.SIGKILL)
                child.wait(timeout=TERMINATE_GRACE_SECONDS)
        else:
            child.wait()
        for stream in (child.stdin, child.stdout):
            if stream is not None:
                stream.close()


def _require_stdout(child: subprocess.Popen[bytes]) -> IO[bytes]:
    if child.stdout is None:
        raise ProviderProtocolError
    return child.stdout


def provider_entrypoint(
    provider_root: Path | None, provider: ProviderId
) -> Path | None:
    """Return one closed worker path without searching another location."""
    if provider_root is None:
        return None
    return (
        provider_root
        / provider.value
        / ".venv"
        / "bin"
        / (f"speech-runtime-{provider.value}-worker")
    )


def provider_entrypoint_is_usable(
    provider_root: Path | None, provider: ProviderId
) -> bool:
    """Report whether the configured closed worker is a regular executable."""
    return _is_executable_file(provider_entrypoint(provider_root, provider))


def _is_executable_file(entrypoint: Path | None) -> bool:
    if entrypoint is None:
        return False
    try:
        mode = entrypoint.stat().st_mode
    except OSError:
        return False
    return stat.S_ISREG(mode) and os.access(entrypoint, os.X_OK)


def provider_environment(provider_directory: Path) -> dict[str, str]:
    """Return the minimal offline environment, with the venv's CUDA libraries."""
    environment = {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "PYTHONUNBUFFERED": "1",
        "PATH": os.defpath,
    }
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible_devices is not None:
        environment["CUDA_VISIBLE_DEVICES"] = visible_devices
    site_packages_directories = sorted(
        (provider_directory / ".venv" / "lib").glob("python3.*/site-packages")
    )
    libraries = (
        site_packages / "nvidia" / package / "lib"
        for site_packages in site_packages_directories
        for package in _NVIDIA_LIBRARY_DIRECTORIES.split(":")
    )
    existing = [str(directory) for directory in libraries if directory.is_dir()]
    if existing:
        environment["LD_LIBRARY_PATH"] = os.pathsep.join(existing)
    return environment

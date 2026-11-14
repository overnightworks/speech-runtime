"""Start the speech service on its TCP listener."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import logging
import os
import signal
import socket
import threading
import time
from concurrent.futures import Future
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import uvicorn
from webauth.https_client import fetch_key_set_over_https

from speech_service.callers import SpeechCallers
from speech_service.config import Settings, load_caller_config, load_settings
from speech_service.control import create_control_app
from speech_service.service import Runtime

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from pathlib import Path

_CLOSE_GRACE_SECONDS: Final = 5.0
_USAGE_DESCRIPTION = "Serve the speech wire over TCP."
_USAGE_EPILOG = (
    "Configuration comes from the SPEECH_* environment and the file named by "
    "SPEECH_CONFIG_FILE; docs/wire.md lists the wire's settings and the "
    "Settings section of README.md the process's own."
)


@dataclass(frozen=True, slots=True, kw_only=True)
class StopPolicy:
    """What the stop allows past the drain, and how it ends a process that is stuck.

    The sessions get `grace_seconds` past the drain's bound to close, the server
    and the engines each `grace_seconds` to stop; `abort` ends the process when
    the engines will not close, as a running decode cannot be interrupted.
    """

    grace_seconds: float = _CLOSE_GRACE_SECONDS
    abort: Callable[[int], object] = os._exit


def configure_logging() -> None:
    """Log at INFO and hold Piper back: it logs its input and phonemes below ERROR."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    logging.getLogger("piper").setLevel(logging.ERROR)


def main() -> None:
    """Read one configuration and exit nonzero after a fatal lifecycle error."""
    _argument_parser().parse_args()
    settings = load_settings()
    configure_logging()
    raise SystemExit(asyncio.run(serve(settings)))


def _argument_parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(
        prog="speech-service",
        allow_abbrev=False,
        description=_USAGE_DESCRIPTION,
        epilog=_USAGE_EPILOG,
    )


def bind_listener(settings: Settings) -> socket.socket:
    """Bind and listen on the configured address; the caller closes the socket."""
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind((settings.host, settings.port))
        listener.listen()
    except BaseException:
        listener.close()
        raise
    return listener


async def serve(settings: Settings, stop: StopPolicy = StopPolicy()) -> int:  # noqa: B008 - a frozen value
    """Run the server until a signal or fatal lifecycle outcome."""
    callers = SpeechCallers.from_config(
        load_caller_config(_require_config_file(settings)),
        fetch_key_set=fetch_key_set_over_https,
        wall_clock=time.time,
        monotonic_clock=time.monotonic,
    )
    try:
        listener = bind_listener(settings)
    except OSError:
        logging.getLogger(__name__).exception(
            "speech service cannot listen on %s:%s", settings.host, settings.port
        )
        return 1
    with listener:
        return await _serve_on(settings, callers, listener, stop)


async def _serve_on(
    settings: Settings,
    callers: SpeechCallers,
    listener: socket.socket,
    stop: StopPolicy = StopPolicy(),  # noqa: B008 - a frozen value
) -> int:
    loop = asyncio.get_running_loop()
    fatal = asyncio.Event()
    runtime = Runtime.from_settings(settings)
    runtime.set_fatal_callback(lambda: loop.call_soon_threadsafe(fatal.set))
    server = build_server(create_control_app(settings, runtime, callers))
    stopping = asyncio.Event()
    _install_signal_handlers(stopping)
    server_task = asyncio.create_task(server.serve(sockets=[listener]))
    runtime.begin_loading()
    loader = asyncio.create_task(asyncio.to_thread(runtime.load))
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
        stopped.cancel()
        fatal_task.cancel()
        result = await _stop(
            _Stopping(runtime, server, server_task, loader, settings, stop), result
        )
    return result


@dataclass(frozen=True, slots=True)
class _Stopping:
    """What the stop winds down."""

    runtime: Runtime
    server: uvicorn.Server
    server_task: asyncio.Task[None]
    loader: asyncio.Task[None]
    settings: Settings
    policy: StopPolicy


async def _stop(stopping: _Stopping, result: int) -> int:
    """Refuse new work, let the admitted run to the drain's bound, then close.

    Sessions, then the server, then the engines; each step has its own bound, and
    engines that will not close end the process through `policy.abort`.
    """
    runtime, policy = stopping.runtime, stopping.policy
    log = logging.getLogger(__name__)
    loop = asyncio.get_running_loop()
    runtime.begin_shutdown()
    drain_left = runtime.hearing.drain_left(
        stopping.settings.hearing_final_wait_seconds
    )
    drain_ends = loop.time() + drain_left + policy.grace_seconds
    if not await _finished_within(runtime.abandon_loads, policy.grace_seconds):
        log.warning("a voice load outlasted the stop")
    if result == 0:
        await _until_sessions_end(runtime, drain_ends - loop.time())
    stopping.server.should_exit = True
    server_stop = max(drain_ends - loop.time(), policy.grace_seconds)
    await asyncio.wait({stopping.server_task}, timeout=server_stop)
    stopping.server_task.cancel()
    try:
        closed = await _finished_within(runtime.close, policy.grace_seconds)
    except RuntimeError:
        log.exception("speech model cleanup failed")
        result, closed = 1, True
    with contextlib.suppress(TimeoutError):
        async with asyncio.timeout(policy.grace_seconds):
            await asyncio.gather(
                stopping.server_task, stopping.loader, return_exceptions=True
            )
    if not closed:
        log.error("speech engines did not close within the stop; the process ends")
        policy.abort(result)
    return result


async def _until_sessions_end(runtime: Runtime, seconds: float) -> None:
    """Wait up to `seconds` for the live hearing sessions to drain and close."""
    try:
        async with asyncio.timeout(seconds):
            await runtime.hearing.sessions_ended()
    except TimeoutError:
        logging.getLogger(__name__).warning("live hearing sessions outlasted the drain")


async def _finished_within(work: Callable[[], None], seconds: float) -> bool:
    """Run `work` on a daemon thread; whether it returned within `seconds`.

    A thread that outlasts the wait does not hold the process back at its exit,
    and its `work` still runs on. An error `work` raises is raised here.
    """
    outcome: Future[None] = Future()

    def run() -> None:
        try:
            work()
        except Exception as error:  # noqa: BLE001 - handed to the waiting coroutine
            outcome.set_exception(error)
        else:
            outcome.set_result(None)

    threading.Thread(target=run, daemon=True, name="stop").start()
    try:
        async with asyncio.timeout(seconds):
            await asyncio.shield(asyncio.wrap_future(outcome))
    except TimeoutError:
        return False
    return True


class _ServiceServer(uvicorn.Server):
    """uvicorn's server without its own signal handling: the service owns the stop."""

    @contextlib.contextmanager
    def capture_signals(self) -> Iterator[None]:
        yield


def build_server(app: object) -> uvicorn.Server:
    """Build the uvicorn server for the control application."""
    return _ServiceServer(uvicorn.Config(app, log_level="info"))


def _require_config_file(settings: Settings) -> Path:
    """Require the caller configuration before any runtime work starts."""
    if settings.config_file is None:
        message = "SPEECH_CONFIG_FILE is required for the speak wire's callers"
        raise ValueError(message)
    return settings.config_file


def _install_signal_handlers(stopping: asyncio.Event) -> None:
    loop = asyncio.get_running_loop()
    for received in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(received, stopping.set)


if __name__ == "__main__":
    main()

"""Start the speech service on its TCP listener."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import logging
import signal
import socket
import time
from typing import TYPE_CHECKING

import uvicorn
from webauth.https_client import fetch_key_set_over_https

from speech_service.callers import SpeechCallers
from speech_service.config import Settings, load_caller_config, load_settings
from speech_service.control import create_control_app
from speech_service.hearing_session import DRAIN_TURN_SECONDS
from speech_service.service import Runtime

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

_USAGE_DESCRIPTION = "Serve the speech wire over TCP."
_USAGE_EPILOG = (
    "Configuration comes from the SPEECH_* environment and the file named by "
    "SPEECH_CONFIG_FILE; docs/wire.md lists the wire's settings and the "
    "Settings section of README.md the process's own."
)


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


async def serve(settings: Settings) -> int:
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
        return await _serve_on(settings, callers, listener)


async def _serve_on(
    settings: Settings, callers: SpeechCallers, listener: socket.socket
) -> int:
    loop = asyncio.get_running_loop()
    fatal = asyncio.Event()
    runtime = Runtime.from_settings(settings)
    runtime.set_fatal_callback(lambda: loop.call_soon_threadsafe(fatal.set))
    server = build_server(create_control_app(settings, runtime, callers))
    drain_seconds = DRAIN_TURN_SECONDS + settings.hearing_final_wait_seconds
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
        deadline = loop.time() + drain_seconds
        runtime.begin_shutdown()
        await asyncio.to_thread(runtime.abandon_loads)
        if result == 0:
            await _until_sessions_end(runtime, deadline - loop.time())
        server.should_exit = True
        stopped.cancel()
        fatal_task.cancel()
        await asyncio.wait({server_task}, timeout=max(0.0, deadline - loop.time()))
        server_task.cancel()
        try:
            await asyncio.to_thread(runtime.close)
        except RuntimeError:
            logging.getLogger(__name__).exception("speech model cleanup failed")
            result = 1
        await asyncio.gather(server_task, loader, return_exceptions=True)
    return result


async def _until_sessions_end(runtime: Runtime, seconds: float) -> None:
    """Wait up to `seconds` for the live hearing sessions to drain and close."""
    try:
        async with asyncio.timeout(seconds):
            await runtime.hearing.sessions_ended()
    except TimeoutError:
        logging.getLogger(__name__).warning("live hearing sessions outlasted the drain")


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

"""Start the speech service on its TCP listener."""

from __future__ import annotations

import argparse
import asyncio
import logging
import signal
import socket
import time
from typing import TYPE_CHECKING

import uvicorn
from webauth.https_client import fetch_key_set_over_https

from speech_runtime_service.callers import SpeechCallers
from speech_runtime_service.config import Settings, load_caller_config, load_settings
from speech_runtime_service.control import create_control_app
from speech_runtime_service.cuda_libs import prepare_cuda_libraries
from speech_runtime_service.service import Runtime

if TYPE_CHECKING:
    from pathlib import Path

_USAGE_DESCRIPTION = "Serve the speech wire over TCP."
_USAGE_EPILOG = (
    "Configuration comes from the SPEECH_* environment and the file named by "
    "SPEECH_CONFIG_FILE; the settings table in README.md lists every one."
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
        prog="speech-runtime-service",
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
    prepare_cuda_libraries()
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
        runtime.begin_shutdown()
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


def build_server(app: object) -> uvicorn.Server:
    """Build the uvicorn server for the control application."""
    return uvicorn.Server(uvicorn.Config(app, log_level="info"))


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

"""One-command-at-a-time Chatterbox provider worker."""

import os
import queue
import sys
import threading
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import BinaryIO

from presentator_chatterbox_contract import (
    Frame,
    FrameKind,
    ProtocolError,
    ProviderFailure,
    read_frame,
    write_frame,
)


def run(
    *,
    device: str,
    cache: str,
    protocol_stdout: int,
    model_functions: tuple[
        Callable[[str, Path], object],
        Callable[[object, str, str], Iterable[bytes]],
    ]
    | None = None,
) -> int:
    """Load once, then process serialized commands."""
    stream = os.fdopen(protocol_stdout, "wb", buffering=0)
    try:
        if model_functions is None:
            from presentator_chatterbox.model import load_model, pcm_chunks

            model_functions = (load_model, pcm_chunks)
        model_loader, synthesizer = model_functions
        model = model_loader(device, Path(cache))
        for _ in synthesizer(model, "Hallo.", "de"):
            pass
        write_frame(stream, Frame(FrameKind.READY, 0, b"\x00\x01\x00\x00]\xc0"))
    except Exception:
        _failed(stream, 0)
        stream.close()
        return 1
    commands: queue.Queue[Frame | BaseException] = queue.Queue()
    reader_stopped = threading.Event()
    reader = threading.Thread(
        target=_read_commands,
        args=(commands, reader_stopped),
        name="chatterbox-provider-input",
        daemon=True,
    )
    reader.start()
    try:
        return serve_commands(commands, stream, synthesizer, model)
    finally:
        if reader_stopped.is_set():
            reader.join()
        stream.close()


def _read_commands(
    commands: queue.Queue[Frame | BaseException], stopped: threading.Event
) -> None:
    try:
        while True:
            commands.put(read_frame(sys.stdin.buffer))
    except (EOFError, ProtocolError, OSError) as error:
        stopped.set()
        commands.put(error)


def serve_commands(
    commands: queue.Queue[Frame | BaseException],
    stream: BinaryIO,
    synthesize: Callable[[object, str, str], Iterable[bytes]],
    model: object,
) -> int:
    """Serve validated commands until input or synthesis fails."""
    previous_terminal = 0
    while True:
        command = commands.get()
        if isinstance(command, BaseException):
            _failed(stream, 0)
            return 1
        if command.kind is FrameKind.CANCEL and command.request_id == previous_terminal:
            continue
        if (
            command.kind is not FrameKind.SYNTHESIZE
            or command.request_id <= previous_terminal
        ):
            _failed(stream, 0)
            return 1
        request_id = command.request_id
        language = {1: "de", 2: "en"}[command.payload[0]]
        text = command.payload[1:].decode("utf-8")
        try:
            cancelled = False
            for pcm in synthesize(model, text, language):
                cancelled = _consume_cancellation(
                    commands, request_id, previous_terminal
                )
                if cancelled:
                    break
                write_frame(stream, Frame(FrameKind.PCM, request_id, pcm))
            if not cancelled:
                cancelled = _consume_cancellation(
                    commands, request_id, previous_terminal
                )
            terminal = FrameKind.CANCELLED if cancelled else FrameKind.COMPLETE
            write_frame(stream, Frame(terminal, request_id, b""))
            previous_terminal = request_id
        except Exception:
            _failed(stream, request_id)
            return 1


def _consume_cancellation(
    commands: queue.Queue[Frame | BaseException],
    request_id: int,
    previous_terminal: int,
) -> bool:
    cancelled = False
    while True:
        try:
            command = commands.get_nowait()
        except queue.Empty:
            return cancelled
        if isinstance(command, BaseException):
            raise command
        if command.kind is not FrameKind.CANCEL:
            raise ProtocolError
        if command.request_id == previous_terminal:
            continue
        if command.request_id != request_id:
            raise ProtocolError
        cancelled = True


def _failed(stream: BinaryIO, request_id: int) -> None:
    try:
        write_frame(
            stream,
            Frame(
                FrameKind.FAILED, request_id, bytes([ProviderFailure.PROVIDER_FAILURE])
            ),
        )
    except (OSError, ProtocolError):
        pass

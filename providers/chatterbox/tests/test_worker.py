"""The provider emits only ordered frames and one sanitized failure value."""

from __future__ import annotations

import builtins
import multiprocessing
import os
import queue
import struct
import sys
import threading
from collections.abc import Iterator
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import pytest
from presentator_chatterbox_contract import (
    Frame,
    FrameKind,
    ProviderFailure,
    read_frame,
    write_frame,
)

from presentator_chatterbox import worker


def _encoded(*frames: Frame) -> bytes:
    stream = BytesIO()
    for frame in frames:
        write_frame(stream, frame)
    return stream.getvalue()


def _frames(encoded: bytes) -> list[Frame]:
    stream = BytesIO(encoded)
    frames = []
    while stream.tell() < len(encoded):
        frames.append(read_frame(stream))
    return frames


def _run_worker(
    monkeypatch: pytest.MonkeyPatch,
    input_bytes: bytes,
    loader,
    synthesizer,
) -> tuple[int, list[Frame], list[tuple[str, Path]]]:
    observed: list[tuple[str, Path]] = []

    def recording_loader(device: str, cache: Path) -> object:
        observed.append((device, cache))
        return loader(device, cache)

    monkeypatch.setattr(
        worker.sys, "stdin", SimpleNamespace(buffer=BytesIO(input_bytes))
    )
    read_fd, write_fd = os.pipe()
    result = worker.run(
        device="cpu",
        cache="/closed/cache",
        protocol_stdout=write_fd,
        model_functions=(recording_loader, synthesizer),
    )
    with os.fdopen(read_fd, "rb") as output:
        frames = _frames(output.read())
    return result, frames, observed


def test_run_loads_warms_then_maps_active_input_eof_to_sanitized_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str]] = []

    def synthesize(_model: object, text: str, language: str) -> Iterator[bytes]:
        calls.append((text, language))
        if text != "Hallo.":
            yield b"\x00\x01"

    result, frames, observed = _run_worker(
        monkeypatch,
        _encoded(Frame(FrameKind.SYNTHESIZE, 1, b"\x02Hello")),
        lambda _device, _cache: object(),
        synthesize,
    )

    assert result == 1
    assert observed == [("cpu", Path("/closed/cache"))]
    assert calls == [("Hallo.", "de"), ("Hello", "en")]
    assert frames == [
        Frame(FrameKind.READY, 0, b"\x00\x01\x00\x00]\xc0"),
        Frame(FrameKind.FAILED, 1, bytes([ProviderFailure.PROVIDER_FAILURE])),
    ]
    assert not any(
        thread.name == "chatterbox-provider-input" and thread.is_alive()
        for thread in threading.enumerate()
    )


def test_run_joins_a_cooperative_reader_that_is_still_returning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    queued = threading.Event()
    release = threading.Event()
    finished = threading.Event()

    class HeldQueue(queue.Queue):
        def put(self, item: object, *args: object, **kwargs: object) -> None:
            super().put(item, *args, **kwargs)
            if isinstance(item, BaseException):
                queued.set()
                assert release.wait(timeout=5)

    monkeypatch.setattr(worker.queue, "Queue", HeldQueue)
    monkeypatch.setattr(worker.sys, "stdin", SimpleNamespace(buffer=BytesIO()))
    read_fd, write_fd = os.pipe()
    outcomes: list[int] = []

    def run_worker() -> None:
        outcomes.append(
            worker.run(
                device="cpu",
                cache="/closed/cache",
                protocol_stdout=write_fd,
                model_functions=(lambda *_args: object(), lambda *_args: ()),
            )
        )
        finished.set()

    running = threading.Thread(target=run_worker)
    running.start()
    assert queued.wait(timeout=5)
    returned_before_reader = finished.wait(timeout=0.1)
    release.set()
    running.join(timeout=5)
    with os.fdopen(read_fd, "rb") as output:
        _frames(output.read())

    assert returned_before_reader is False
    assert not running.is_alive()
    assert outcomes == [1]
    assert not any(
        thread.name == "chatterbox-provider-input" and thread.is_alive()
        for thread in threading.enumerate()
    )


def test_startup_failure_emits_only_the_sanitized_startup_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider_detail = "provider secret detail"

    def fail(_device: str, _cache: Path) -> object:
        raise RuntimeError(provider_detail)

    result, frames, _ = _run_worker(monkeypatch, b"", fail, lambda *_args: ())
    assert result == 1
    assert frames == [Frame(FrameKind.FAILED, 0, b"\x01")]
    assert provider_detail.encode() not in _encoded(*frames)


def test_malformed_input_emits_one_sanitized_startup_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    malformed = struct.pack("!I", 8)
    result, frames, _ = _run_worker(
        monkeypatch,
        malformed,
        lambda _device, _cache: object(),
        lambda *_args: (),
    )
    assert result == 1
    assert frames == [
        Frame(FrameKind.READY, 0, b"\x00\x01\x00\x00]\xc0"),
        Frame(FrameKind.FAILED, 0, b"\x01"),
    ]


def test_cancellation_after_an_inflight_step_and_late_cancel_do_not_cross_ids() -> None:
    commands: queue.Queue[Frame | BaseException] = queue.Queue()
    output = BytesIO()
    entered = threading.Event()
    release = threading.Event()

    def synthesize(_model: object, text: str, _language: str) -> Iterator[bytes]:
        if text == "first":
            entered.set()
            assert release.wait(timeout=5)
        yield b"\x00\x01"

    commands.put(Frame(FrameKind.SYNTHESIZE, 1, b"\x01first"))
    serving = threading.Thread(
        target=worker.serve_commands, args=(commands, output, synthesize, object())
    )
    serving.start()
    assert entered.wait(timeout=5)
    commands.put(Frame(FrameKind.CANCEL, 1, b""))
    release.set()
    _wait_for_frames(output, 1)
    commands.put(Frame(FrameKind.CANCEL, 1, b""))
    commands.put(Frame(FrameKind.SYNTHESIZE, 2, b"\x02second"))
    _wait_for_frames(output, 3)
    commands.put(EOFError())
    serving.join(timeout=5)

    assert not serving.is_alive()
    assert _frames(output.getvalue()) == [
        Frame(FrameKind.CANCELLED, 1, b""),
        Frame(FrameKind.PCM, 2, b"\x00\x01"),
        Frame(FrameKind.COMPLETE, 2, b""),
        Frame(FrameKind.FAILED, 0, b"\x01"),
    ]


@pytest.mark.parametrize(
    ("commands", "failure_id"),
    [
        ([Frame(FrameKind.PCM, 1, b"\x00\x01")], 0),
        ([Frame(FrameKind.READY, 0, b"\x00\x01\x00\x00]\xc0")], 0),
        ([Frame(FrameKind.CANCEL, 7, b"")], 0),
        (
            [
                Frame(FrameKind.SYNTHESIZE, 2, b"\x01one"),
                Frame(FrameKind.SYNTHESIZE, 2, b"\x01stale"),
            ],
            2,
        ),
    ],
)
def test_wrong_direction_out_of_order_and_stale_commands_fail_once(
    commands: list[Frame], failure_id: int
) -> None:
    command_queue: queue.Queue[Frame | BaseException] = queue.Queue()
    output = BytesIO()
    for command in commands:
        command_queue.put(command)
    command_queue.put(EOFError())
    assert (
        worker.serve_commands(command_queue, output, lambda *_args: (), object()) == 1
    )
    frames = _frames(output.getvalue())
    assert frames[-1] == Frame(FrameKind.FAILED, failure_id, b"\x01")
    assert sum(frame.kind is FrameKind.FAILED for frame in frames) == 1


def _wait_for_frames(output: BytesIO, expected: int) -> None:
    deadline = threading.Event()
    for _ in range(500):
        if len(_frames(output.getvalue())) >= expected:
            return
        deadline.wait(0.01)
    pytest.fail("provider did not produce the expected frames")


def test_model_and_pcm_validation_failure_emit_only_active_failure() -> None:
    for synthesizer in (
        lambda *_args: (_ for _ in ()).throw(RuntimeError("raw model detail")),
        lambda *_args: iter((b"\x00",)),
    ):
        commands: queue.Queue[Frame | BaseException] = queue.Queue()
        output = BytesIO()
        commands.put(Frame(FrameKind.SYNTHESIZE, 9, b"\x01Hallo"))
        assert worker.serve_commands(commands, output, synthesizer, object()) == 1
        assert _frames(output.getvalue()) == [Frame(FrameKind.FAILED, 9, b"\x01")]
        assert b"raw model detail" not in output.getvalue()


def _run_bootstrap(write_fd: int) -> None:
    os.dup2(write_fd, 1)
    os.close(write_fd)
    sys.argv = [
        "presentator-chatterbox-worker",
        "--expected-parent-pid",
        str(os.getppid()),
        "--device",
        "cpu",
        "--cache",
        "/closed/cache",
    ]
    from presentator_chatterbox import __main__ as bootstrap

    installed = False
    install = bootstrap.install_parent_death_signal

    def recording_install(expected_parent_pid: int) -> None:
        nonlocal installed
        install(expected_parent_pid)
        installed = True

    bootstrap.install_parent_death_signal = recording_install
    real_import = builtins.__import__

    def importing(name: str, *args: object, **kwargs: object) -> object:
        if name == "presentator_chatterbox.worker":
            assert installed
            vars(builtins)["print"]("import print noise", flush=True)
            os.write(1, b"import native noise\n")

            def fake_run(**arguments: object) -> int:
                protocol = int(arguments["protocol_stdout"])
                os.write(protocol, b"protocol-only")
                os.close(protocol)
                return 0

            return SimpleNamespace(run=fake_run)
        return real_import(name, *args, **kwargs)

    builtins.__import__ = importing
    raise SystemExit(bootstrap.main())


def test_bootstrap_redirects_python_and_fd_one_before_provider_import() -> None:
    read_fd, write_fd = os.pipe()
    process = multiprocessing.get_context("fork").Process(
        target=_run_bootstrap, args=(write_fd,)
    )
    process.start()
    os.close(write_fd)
    with os.fdopen(read_fd, "rb") as output:
        assert output.read() == b"protocol-only"
    process.join(timeout=5)
    assert process.exitcode == 0

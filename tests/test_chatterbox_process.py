"""The parent keeps Chatterbox protocol failures private and terminal."""

from __future__ import annotations

import ast
import os
import struct
import subprocess
import sys
import threading
import time
from io import BytesIO
from pathlib import Path

import pytest
from anyio import run, run_process
from presentator_chatterbox_contract import (
    MAX_FRAME_LENGTH,
    Frame,
    FrameDecoder,
    FrameKind,
    ProtocolError,
    read_frame,
    write_frame,
)

from speech.chatterbox import ChatterboxProtocolError, ChatterboxSpeaking
from speech.config import CHATTERBOX_SPEAKING_MODEL, Settings
from speech.selection import VoiceSelectionStore
from speech.service import Runtime, RuntimeDependencies
from speech.voices import VoiceId, VoiceLoadOutcome, VoiceRecoveryKind
from tests.conftest import FakeHearing, FakeSpeaking

_WORKER = r"""#!{python}
import argparse
import os
import signal
import struct
import time
from pathlib import Path

protocol_fd = os.dup(1)
null = os.open(os.devnull, os.O_WRONLY)
os.dup2(null, 1)
os.close(null)
print("provider print noise", flush=True)
os.write(1, b"native provider noise\n")

from presentator_chatterbox_contract import (
    MAX_FRAME_LENGTH, Frame, FrameKind, ProviderFailure, read_frame, write_frame
)

parser = argparse.ArgumentParser()
parser.add_argument("--expected-parent-pid", required=True)
parser.add_argument("--device", required=True)
parser.add_argument("--cache", required=True)
arguments = parser.parse_args()
cache = Path(arguments.cache)
cache.mkdir(parents=True, exist_ok=True)
(cache / "worker-observed").write_text(
    repr(os.getcwd()) + "\n" + repr(os.environ.copy()) + "\n" +
    repr(os.getpid()) + "\n" + repr(arguments.__dict__), encoding="utf-8"
)
stream = os.fdopen(protocol_fd, "wb", buffering=0)
behavior = {behavior!r}
if behavior == "startup-exit":
    raise SystemExit(1)
if behavior == "startup-oversized":
    stream.write(struct.pack("!I", MAX_FRAME_LENGTH + 1))
    raise SystemExit(1)
if behavior == "startup-failed":
    write_frame(stream, Frame(FrameKind.FAILED, 0, b"\x01"))
    raise SystemExit(1)
write_frame(stream, Frame(FrameKind.READY, 0, b"\x00\x01\x00\x00]\xc0"))
if behavior == "idle-exit":
    while not (cache / "exit-now").exists():
        time.sleep(0.01)
    raise SystemExit(1)
if behavior == "term-resistant":
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
last_terminal = 0
while True:
    frame = read_frame(__import__("sys").stdin.buffer)
    if frame.kind is FrameKind.CANCEL and frame.request_id == last_terminal:
        continue
    if frame.kind is not FrameKind.SYNTHESIZE:
        write_frame(stream, Frame(FrameKind.FAILED, 0, b"\x01"))
        raise SystemExit(1)
    request_id = frame.request_id
    text = frame.payload[1:].decode()
    print("synthesis print noise", flush=True)
    os.write(1, b"synthesis native noise\n")
    with (cache / "request-ids").open("a", encoding="utf-8") as ids:
        ids.write(f"{{request_id}}\n")
    if text == "stale":
        write_frame(stream, Frame(FrameKind.PCM, request_id + 1, b"\x00\x01"))
        continue
    if text == "wrong-direction":
        write_frame(stream, Frame(FrameKind.READY, 0, b"\x00\x01\x00\x00]\xc0"))
        continue
    if text == "unexpected-cancelled":
        write_frame(stream, Frame(FrameKind.CANCELLED, request_id, b""))
        continue
    if text == "failed":
        write_frame(stream, Frame(FrameKind.FAILED, request_id, b"\x01"))
        raise SystemExit(1)
    write_frame(stream, Frame(FrameKind.PCM, request_id, b"\x00\x01"))
    if text == "midstream-exit":
        raise SystemExit(1)
    if text == "wait-cancel":
        cancel = read_frame(__import__("sys").stdin.buffer)
        (cache / "cancel-observed").touch()
        write_frame(stream, Frame(FrameKind.CANCELLED, cancel.request_id, b""))
        last_terminal = request_id
        continue
    if behavior == "partial-cancel":
        read_frame(__import__("sys").stdin.buffer)
        (cache / "cancel-observed").touch()
        stream.write(struct.pack("!IBQ", 11, FrameKind.COMPLETE, request_id) + b"x")
        time.sleep(30)
    write_frame(stream, Frame(FrameKind.COMPLETE, request_id, b""))
    if behavior == "duplicate-terminal":
        write_frame(stream, Frame(FrameKind.COMPLETE, request_id, b""))
    last_terminal = request_id
"""


def _closed_provider(tmp_path: Path, behavior: str = "normal") -> tuple[Path, Path]:
    root = tmp_path / "providers"
    provider = root / "chatterbox"
    entrypoint = provider / ".venv" / "bin" / "presentator-chatterbox-worker"
    entrypoint.parent.mkdir(parents=True)
    entrypoint.write_text(
        _WORKER.format(python=sys.executable, behavior=behavior), encoding="utf-8"
    )
    entrypoint.chmod(0o700)
    return root, tmp_path / "cache"


def _engine(tmp_path: Path, behavior: str = "normal") -> ChatterboxSpeaking:
    root, cache = _closed_provider(tmp_path, behavior)
    return ChatterboxSpeaking(CHATTERBOX_SPEAKING_MODEL, "cpu", cache, root)


def _observed(cache: Path) -> tuple[str, dict[str, str], int, dict[str, str]]:
    lines = (cache / "worker-observed").read_text(encoding="utf-8").splitlines()
    return (
        ast.literal_eval(lines[0]),
        ast.literal_eval(lines[1]),
        ast.literal_eval(lines[2]),
        ast.literal_eval(lines[3]),
    )


def _assert_reaped(pid: int) -> None:
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


def test_protocol_round_trips_partial_reads_and_consecutive_frames() -> None:
    encoded = BytesIO()
    write_frame(encoded, Frame(FrameKind.PCM, 1, b"\x00\x01"))
    write_frame(encoded, Frame(FrameKind.COMPLETE, 1, b""))
    decoder = FrameDecoder()
    frames = []
    for byte in encoded.getvalue():
        decoder.feed(bytes([byte]))
        frame = decoder.next_frame()
        if frame is not None:
            frames.append(frame)
    decoder.finish()
    assert frames == [
        Frame(FrameKind.PCM, 1, b"\x00\x01"),
        Frame(FrameKind.COMPLETE, 1, b""),
    ]


@pytest.mark.parametrize(
    "encoded",
    [
        struct.pack("!I", 8),
        struct.pack("!I", MAX_FRAME_LENGTH + 1),
        struct.pack("!IBQ", 9, 255, 1),
        struct.pack("!IBQ", 9, FrameKind.PCM, 1),
        struct.pack("!IBQ", 10, FrameKind.FAILED, 1) + b"\x02",
        struct.pack("!IBQ", 15, FrameKind.READY, 0) + b"\x00\x02\x00\x00]\xc0",
        struct.pack("!IBQ", 10, FrameKind.SYNTHESIZE, 1) + b"\x03",
        struct.pack("!IBQ", 12, FrameKind.PCM, 1) + b"\x00\x01\x02",
    ],
)
def test_protocol_rejects_illegal_lengths_kinds_and_payloads(encoded: bytes) -> None:
    with pytest.raises((ProtocolError, EOFError)):
        read_frame(BytesIO(encoded))


@pytest.mark.parametrize("payload", [b"", b"\x03text", b"\x01\xff"])
def test_protocol_refuses_invalid_synthesis_payloads(payload: bytes) -> None:
    with pytest.raises(ProtocolError):
        write_frame(BytesIO(), Frame(FrameKind.SYNTHESIZE, 1, payload))


def test_direct_closed_entrypoint_streams_after_transient_loader_exits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, cache = _closed_provider(tmp_path)
    cublas = root / "chatterbox/.venv/lib/python3.12/site-packages/nvidia/cublas/lib"
    cublas.mkdir(parents=True)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    real_popen = subprocess.Popen
    launches: list[tuple[list[str], dict[str, object]]] = []

    def recording_popen(
        arguments: list[str], **keywords: object
    ) -> subprocess.Popen[bytes]:
        launches.append((arguments, keywords))
        return real_popen(arguments, **keywords)

    monkeypatch.setattr("speech.chatterbox.subprocess.Popen", recording_popen)
    engine = ChatterboxSpeaking(CHATTERBOX_SPEAKING_MODEL, "cpu", cache, root)
    loaded = threading.Thread(target=engine.load)
    loaded.start()
    loaded.join(timeout=5)
    assert not loaded.is_alive()

    response: list[bytes] = []
    request = threading.Thread(
        target=lambda: response.extend(engine.pcm_chunks("Hallo", "de-DE"))
    )
    request.start()
    request.join(timeout=5)
    assert not request.is_alive()
    assert response == [b"\x00\x01"]
    assert any(
        thread.name == "chatterbox-provider-owner" for thread in threading.enumerate()
    )
    cwd, environment, pid, arguments = _observed(cache)
    assert list(engine.pcm_chunks("later", "en")) == [b"\x00\x01"]
    engine.close()

    assert cwd == str(root / "chatterbox")
    assert arguments == {
        "expected_parent_pid": str(os.getpid()),
        "device": "cpu",
        "cache": str(cache),
    }
    expected_environment = {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "PYTHONUNBUFFERED": "1",
        "LD_LIBRARY_PATH": str(cublas),
    }
    assert set(environment) <= {*expected_environment, "LC_CTYPE"}
    assert all(environment[key] == value for key, value in expected_environment.items())
    assert len(launches) == 1
    launch_arguments, launch = launches[0]
    assert launch_arguments == [
        str(root / "chatterbox/.venv/bin/presentator-chatterbox-worker"),
        "--expected-parent-pid",
        str(os.getpid()),
        "--device",
        "cpu",
        "--cache",
        str(cache),
    ]
    assert launch == {
        "shell": False,
        "stdin": subprocess.PIPE,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.DEVNULL,
        "cwd": root / "chatterbox",
        "env": expected_environment,
        "bufsize": 0,
        "close_fds": True,
        "start_new_session": True,
    }
    _assert_reaped(pid)


@pytest.mark.parametrize("first", ["wait-cancel", "normal"])
def test_cancel_drain_keeps_the_worker_reusable_and_late_cancel_isolated(
    tmp_path: Path, first: str
) -> None:
    engine = _engine(tmp_path)
    engine.load()
    chunks = engine.pcm_chunks(first, "de")
    assert next(chunks) == b"\x00\x01"
    chunks.close()

    assert list(engine.pcm_chunks("later", "en")) == [b"\x00\x01"]
    assert (tmp_path / "cache/request-ids").read_text(encoding="utf-8").split() == [
        "1",
        "2",
    ]
    engine.close()


@pytest.mark.parametrize(
    "text",
    ["stale", "wrong-direction", "unexpected-cancelled", "midstream-exit"],
)
def test_stale_wrong_direction_out_of_order_and_midstream_exit_fail_closed(
    tmp_path: Path, text: str
) -> None:
    engine = _engine(tmp_path)
    engine.load()
    with pytest.raises(ChatterboxProtocolError):
        list(engine.pcm_chunks(text, "de"))
    engine.close()


def test_request_failure_is_sanitized_and_marks_the_adapter_unready(
    tmp_path: Path,
) -> None:
    engine = _engine(tmp_path)
    engine.load()
    with pytest.raises(RuntimeError, match="chatterbox synthesis failed"):
        list(engine.pcm_chunks("failed", "de"))
    assert engine.ready is False
    engine.close()


@pytest.mark.parametrize(
    "behavior", ["startup-exit", "startup-oversized", "startup-failed"]
)
def test_early_exit_and_oversized_handshake_fail_without_an_owned_process(
    tmp_path: Path, behavior: str
) -> None:
    engine = _engine(tmp_path, behavior)
    with pytest.raises(RuntimeError, match="chatterbox provider failed"):
        engine.load()
    _, _, pid, _ = _observed(tmp_path / "cache")
    engine.close()
    _assert_reaped(pid)


def test_duplicate_terminal_is_rejected_before_a_later_request(tmp_path: Path) -> None:
    engine = _engine(tmp_path, "duplicate-terminal")
    engine.load()
    assert list(engine.pcm_chunks("first", "de")) == [b"\x00\x01"]
    with pytest.raises(ChatterboxProtocolError):
        list(engine.pcm_chunks("second", "de"))
    engine.close()


def test_partial_cancel_frame_cannot_extend_the_single_drain_deadline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("speech.chatterbox.CANCEL_DRAIN_SECONDS", 0.05)
    monkeypatch.setattr("speech.chatterbox.TERMINATE_GRACE_SECONDS", 0.05)
    engine = _engine(tmp_path, "partial-cancel")
    engine.load()
    chunks = engine.pcm_chunks("cancel", "de")
    assert next(chunks) == b"\x00\x01"
    started = time.monotonic()
    chunks.close()
    assert time.monotonic() - started < 1
    _, _, pid, _ = _observed(tmp_path / "cache")
    engine.close()
    assert engine.ready is False
    _assert_reaped(pid)


def test_constructor_has_no_thread_process_or_filesystem_side_effect(
    tmp_path: Path,
) -> None:
    before = {thread.ident for thread in threading.enumerate()}
    engine = ChatterboxSpeaking(
        CHATTERBOX_SPEAKING_MODEL, "cpu", tmp_path / "cache", tmp_path / "missing"
    )
    engine.close()
    assert {thread.ident for thread in threading.enumerate()} == before
    assert list(tmp_path.iterdir()) == []


def test_startup_provider_failure_preserves_piper_without_fatal_exit(
    tmp_path: Path,
) -> None:
    fatal = threading.Event()
    piper = FakeSpeaking()
    engine = _engine(tmp_path, "startup-failed")
    runtime = Runtime(
        piper,
        FakeHearing(),
        dependencies=RuntimeDependencies(
            selection=VoiceSelectionStore(tmp_path / "state", owner_uid=os.geteuid()),
            engine_factory=lambda _voice: engine,
            artifact_checker=lambda _voice: True,
            default_voice=VoiceId.PIPER,
        ),
    )
    runtime.set_fatal_callback(fatal.set)

    assert runtime.load_voice(VoiceId.CHATTERBOX) is VoiceLoadOutcome.NOT_ACTIVATED
    admission = runtime.capture_speaking()
    assert admission is not None
    assert admission.speaking is piper
    admission.close()
    assert fatal.is_set() is False
    _, _, pid, _ = _observed(tmp_path / "cache")
    _assert_reaped(pid)


def test_deselecting_chatterbox_reaps_provider_before_load_returns(
    tmp_path: Path,
) -> None:
    piper = FakeSpeaking()
    chatterboxes: list[ChatterboxSpeaking] = []
    root, cache = _closed_provider(tmp_path)

    def factory(voice: VoiceId) -> FakeSpeaking | ChatterboxSpeaking:
        if voice is VoiceId.PIPER:
            return piper
        engine = ChatterboxSpeaking(CHATTERBOX_SPEAKING_MODEL, "cpu", cache, root)
        chatterboxes.append(engine)
        return engine

    runtime = Runtime(
        None,
        FakeHearing(),
        dependencies=RuntimeDependencies(
            selection=VoiceSelectionStore(tmp_path / "state", owner_uid=os.geteuid()),
            engine_factory=factory,
            artifact_checker=lambda _voice: True,
        ),
    )
    try:
        assert runtime.load_voice(VoiceId.CHATTERBOX) is VoiceLoadOutcome.ACTIVATED
        _, _, first_pid, _ = _observed(tmp_path / "cache")

        assert runtime.load_voice(VoiceId.PIPER) is VoiceLoadOutcome.ACTIVATED
        _assert_reaped(first_pid)
        assert runtime.load_voice(VoiceId.CHATTERBOX) is VoiceLoadOutcome.ACTIVATED
        _, _, second_pid, _ = _observed(tmp_path / "cache")

        assert second_pid != first_pid
        assert len(chatterboxes) == 2
    finally:
        runtime.close()


@pytest.mark.parametrize("failure", ["request", "idle-exit"])
def test_active_provider_failure_reconciles_without_fatal_exit(
    tmp_path: Path, failure: str
) -> None:
    fatal = threading.Event()
    engine = _engine(tmp_path, "idle-exit" if failure == "idle-exit" else "normal")
    store = VoiceSelectionStore(tmp_path / "state", owner_uid=os.geteuid())
    runtime = Runtime(
        None,
        FakeHearing(),
        dependencies=RuntimeDependencies(
            selection=store,
            engine_factory=lambda _voice: engine,
            artifact_checker=lambda _voice: True,
        ),
    )
    runtime.set_fatal_callback(fatal.set)
    assert runtime.load_voice(VoiceId.CHATTERBOX) is VoiceLoadOutcome.ACTIVATED
    if failure == "request":
        with pytest.raises(RuntimeError, match="chatterbox synthesis failed"):
            list(engine.pcm_chunks("failed", "de"))
    else:
        (tmp_path / "cache/exit-now").touch()
    deadline = time.monotonic() + 5
    while engine.ready and time.monotonic() < deadline:
        threading.Event().wait(0.01)

    snapshot = runtime.snapshot(Settings(huggingface_cache=tmp_path))

    assert snapshot.recovery is not None
    assert snapshot.recovery.kind is VoiceRecoveryKind.LOAD_FAILED
    assert store.read() is VoiceId.CHATTERBOX
    assert runtime.capture_speaking() is None
    assert fatal.is_set() is False


@pytest.mark.parametrize(
    ("behavior", "text"), [("normal", "wait-cancel"), ("partial-cancel", "cancel")]
)
def test_close_allows_bounded_active_cancel_before_reaping(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    behavior: str,
    text: str,
) -> None:
    monkeypatch.setattr("speech.chatterbox.CANCEL_DRAIN_SECONDS", 0.2)
    monkeypatch.setattr("speech.chatterbox.TERMINATE_GRACE_SECONDS", 0.05)
    engine = _engine(tmp_path, behavior)
    engine.load()
    chunks = engine.pcm_chunks(text, "de")
    assert next(chunks) == b"\x00\x01"
    closed: list[bool] = []

    def close_engine() -> None:
        engine.close()
        closed.append(True)

    closing = threading.Thread(target=close_engine)
    closing.start()
    threading.Event().wait(0.05)

    started = time.monotonic()
    chunks.close()
    closing.join(timeout=5)

    assert not closing.is_alive()
    assert closed == [True]
    assert time.monotonic() - started < 1
    assert (tmp_path / "cache/cancel-observed").exists()
    _, _, pid, _ = _observed(tmp_path / "cache")
    _assert_reaped(pid)


def test_shutdown_during_handshake_closes_pending_without_publication(
    tmp_path: Path,
) -> None:
    root, cache = _closed_provider(tmp_path, "term-resistant")
    entrypoint = root / "chatterbox/.venv/bin/presentator-chatterbox-worker"
    source = entrypoint.read_text(encoding="utf-8")
    entrypoint.write_text(
        source.replace(
            "write_frame(stream, Frame(FrameKind.READY",
            "time.sleep(30)\nwrite_frame(stream, Frame(FrameKind.READY",
            1,
        ),
        encoding="utf-8",
    )
    store = VoiceSelectionStore(tmp_path / "state", owner_uid=os.geteuid())
    runtime = Runtime(
        None,
        FakeHearing(),
        dependencies=RuntimeDependencies(
            selection=store,
            engine_factory=lambda _voice: ChatterboxSpeaking(
                CHATTERBOX_SPEAKING_MODEL, "cpu", cache, root
            ),
            artifact_checker=lambda _voice: True,
        ),
    )
    activation = threading.Thread(target=runtime.load_voice, args=(VoiceId.CHATTERBOX,))
    activation.start()
    deadline = time.monotonic() + 5
    while not (cache / "worker-observed").exists() and time.monotonic() < deadline:
        time.sleep(0.01)

    runtime.begin_shutdown()
    runtime.close()
    activation.join(timeout=5)

    assert not activation.is_alive()
    assert runtime.capture_speaking() is None
    assert not store.path.exists()
    _, _, pid, _ = _observed(cache)
    _assert_reaped(pid)


def test_missing_entrypoint_failed_exec_leaves_no_retained_engine(
    tmp_path: Path,
) -> None:
    fatal = threading.Event()
    runtime = Runtime(
        None,
        FakeHearing(),
        dependencies=RuntimeDependencies(
            selection=VoiceSelectionStore(tmp_path / "state", owner_uid=os.geteuid()),
            engine_factory=lambda _voice: ChatterboxSpeaking(
                CHATTERBOX_SPEAKING_MODEL, "cpu", tmp_path, tmp_path / "missing"
            ),
            artifact_checker=lambda _voice: True,
        ),
    )
    runtime.set_fatal_callback(fatal.set)
    assert runtime.load_voice(VoiceId.CHATTERBOX) is VoiceLoadOutcome.NOT_ACTIVATED
    assert runtime.capture_speaking() is None
    assert not any(
        thread.name == "chatterbox-provider-owner" and thread.is_alive()
        for thread in threading.enumerate()
    )
    assert fatal.is_set() is False


def test_real_child_is_reaped_when_publication_setup_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fatal = threading.Event()
    engine = _engine(tmp_path, "term-resistant")
    engine.set_fatal_callback(fatal.set)
    real_popen = subprocess.Popen
    started: list[int] = []

    def recording_popen(*arguments: object, **keywords: object) -> subprocess.Popen:
        child = real_popen(*arguments, **keywords)
        started.append(child.pid)
        return child

    class BrokenDecoder:
        def __init__(self) -> None:
            raise RuntimeError

    monkeypatch.setattr("speech.chatterbox.FrameDecoder", BrokenDecoder)
    monkeypatch.setattr("speech.chatterbox.subprocess.Popen", recording_popen)
    with pytest.raises(RuntimeError, match="chatterbox provider failed"):
        engine.load()
    assert fatal.wait(timeout=5)
    engine.close()
    assert len(started) == 1
    _assert_reaped(started[0])


def test_cleanup_failure_reaps_and_reports_fatal_lifecycle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fatal = threading.Event()
    real_popen = subprocess.Popen
    started: list[int] = []

    class FailingCleanupProcess:
        def __init__(self, process: subprocess.Popen) -> None:
            self._process = process

        def __getattr__(self, name: str) -> object:
            return getattr(self._process, name)

        def wait(self, timeout: float | None = None) -> int:
            self._process.wait(timeout=timeout)
            message = "cleanup failed after reap"
            raise RuntimeError(message)

    def failing_cleanup_popen(
        *arguments: object, **keywords: object
    ) -> FailingCleanupProcess:
        process = real_popen(*arguments, **keywords)
        started.append(process.pid)
        return FailingCleanupProcess(process)

    monkeypatch.setattr("speech.chatterbox.subprocess.Popen", failing_cleanup_popen)
    engine = _engine(tmp_path)
    engine.set_fatal_callback(fatal.set)
    engine.load()

    with pytest.raises(RuntimeError, match="chatterbox provider cleanup failed"):
        engine.close()

    assert fatal.wait(timeout=5)
    assert len(started) == 1
    _assert_reaped(started[0])


def test_term_resistant_leader_reaches_kill_and_reap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("speech.chatterbox.TERMINATE_GRACE_SECONDS", 0.05)
    engine = _engine(tmp_path, "term-resistant")
    engine.load()
    _, _, pid, _ = _observed(tmp_path / "cache")
    engine.close()
    _assert_reaped(pid)


def test_parent_death_signal_follows_the_real_creator_thread(tmp_path: Path) -> None:
    provider_source = Path(__file__).parents[1] / "providers/chatterbox/src"
    contract_source = Path(__file__).parents[1] / "providers/chatterbox-contract/src"
    helper = r'''
import os
import subprocess
import sys
import threading
from pathlib import Path

ready = Path(sys.argv[1])
child_code = """
import os, sys, time
from pathlib import Path
from presentator_chatterbox.__main__ import install_parent_death_signal
install_parent_death_signal(int(sys.argv[1]))
Path(sys.argv[2]).write_text(str(os.getpid()))
while True: time.sleep(1)
"""
owned = []
def create():
    child = subprocess.Popen([
        sys.executable, "-c", child_code, str(os.getpid()), str(ready)
    ])
    owned.append(child)
    for _ in range(500):
        if ready.exists():
            return
        threading.Event().wait(0.01)
    raise RuntimeError("child never installed PDEATHSIG")
creator = threading.Thread(target=create)
creator.start()
creator.join()
child = owned[0]
print("responsive", flush=True)
raise SystemExit(0 if child.wait(timeout=5) == -9 else 1)
'''
    environment = {
        "PYTHONPATH": os.pathsep.join((str(provider_source), str(contract_source)))
    }

    async def execute_helper() -> tuple[int, bytes]:
        completed = await run_process(
            [sys.executable, "-c", helper, str(tmp_path / "ready")],
            env=environment,
            check=False,
        )
        return completed.returncode, completed.stdout

    returncode, output = run(execute_helper)
    assert returncode == 0
    assert output == b"responsive\n"

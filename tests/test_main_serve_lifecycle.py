"""Serving ends with the signal that stops it and binds nothing but the socket."""

from __future__ import annotations

import asyncio
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from speech_runtime_service import __main__
from speech_runtime_service.config import Settings
from speech_runtime_service.selection import VoiceSelectionStore
from speech_runtime_service.service import Runtime, RuntimeDependencies
from speech_runtime_service.voices import VoiceId
from tests.support.fakes_support import FakeHearing, FakeSpeaking
from tests.support.identity_support import (
    caller_configuration_text,
    write_caller_configuration,
)

if TYPE_CHECKING:
    import threading

    import uvicorn

_SIGTERM_SERVER = """
import asyncio
import os
from pathlib import Path

from speech_runtime_service import __main__
from speech_runtime_service.config import Settings
from speech_runtime_service.selection import VoiceSelectionStore
from speech_runtime_service.service import Runtime, RuntimeDependencies
from speech_runtime_service.voices import VoiceId
from tests.support.fakes_support import FakeHearing, FakeSpeaking

directory = Path(os.environ["SPEECH_TEST_DIRECTORY"])
__main__.prepare_cuda_libraries = lambda: None
Runtime.from_settings = lambda _settings: Runtime(
    None,
    FakeHearing(),
    dependencies=RuntimeDependencies(
        selection=VoiceSelectionStore(directory / "state", owner_uid=os.geteuid()),
        engine_factory=lambda _voice: FakeSpeaking(),
        artifact_checker=lambda _voice: True,
        default_voice=VoiceId.PIPER,
    ),
)
settings = Settings(
    private_directory=directory,
    PRESENTATOR_RUNTIME_UID=os.geteuid(),
)
raise SystemExit(asyncio.run(__main__.serve(settings)))
"""


def test_sigterm_stops_the_real_server_and_exits_zero_with_the_socket_removed(
    tmp_path: Path,
) -> None:
    socket_path = tmp_path / "speech.sock"
    speech_root = Path(__file__).resolve().parents[1]
    environment = {
        **os.environ,
        "SPEECH_TEST_DIRECTORY": str(tmp_path),
        "SPEECH_CONFIG_FILE": str(write_caller_configuration(tmp_path)),
        "PYTHONPATH": os.pathsep.join([str(speech_root), str(speech_root / "scripts")]),
    }
    with subprocess.Popen(
        [sys.executable, "-c", _SIGTERM_SERVER], env=environment
    ) as server:
        try:
            deadline = time.monotonic() + 30
            while not _accepts_connections(socket_path):
                assert server.poll() is None
                assert time.monotonic() < deadline
                time.sleep(0.05)
            server.send_signal(signal.SIGTERM)
            assert server.wait(timeout=30) == 0
        finally:
            server.kill()

    assert not socket_path.exists()


def test_python_dash_m_starts_the_service_on_its_private_socket(tmp_path: Path) -> None:
    socket_path = tmp_path / "speech.sock"
    speech_root = Path(__file__).resolve().parents[1]
    environment = {
        **os.environ,
        "SPEECH_PRIVATE_DIRECTORY": str(tmp_path),
        "SPEECH_STATE_DIRECTORY": str(tmp_path / "state"),
        "HF_HUB_OFFLINE": "1",
        "HF_HOME": str(tmp_path / "hf"),
        "PRESENTATOR_RUNTIME_UID": str(os.geteuid()),
        "SPEECH_CONFIG_FILE": str(write_caller_configuration(tmp_path)),
        "PYTHONPATH": str(speech_root),
    }
    with subprocess.Popen(
        [sys.executable, "-m", "speech_runtime_service"], env=environment
    ) as server:
        try:
            deadline = time.monotonic() + 30
            while not _accepts_connections(socket_path):
                assert time.monotonic() < deadline
                time.sleep(0.05)
        finally:
            server.kill()


def _accepts_connections(path: Path) -> bool:
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as probe:
        try:
            probe.connect(str(path))
        except OSError:
            return False
    return True


def test_serve_listens_only_on_the_private_socket(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    built: list[uvicorn.Server] = []
    stopping: list[asyncio.Event] = []
    build_server = __main__.build_server

    started = asyncio.Event()

    def observed_server(app: object) -> uvicorn.Server:
        server = build_server(app)
        start = server.startup

        async def startup(sockets: list[socket.socket] | None = None) -> None:
            await start(sockets=sockets)
            started.set()

        server.startup = startup
        built.append(server)
        return server

    def install(stop: asyncio.Event, _loading_stop: threading.Event) -> None:
        stopping.append(stop)

    async def scenario() -> tuple[int, list[socket.AddressFamily]]:
        serving = asyncio.create_task(
            __main__.serve(
                Settings(
                    private_directory=tmp_path,
                    PRESENTATOR_RUNTIME_UID=os.geteuid(),
                    config_file=write_caller_configuration(tmp_path),
                )
            )
        )
        await started.wait()
        families = [
            bound.family
            for listening in built[0].servers
            for bound in listening.sockets
        ]
        stopping[0].set()
        return await serving, families

    runtime = _loadable_runtime(tmp_path)
    monkeypatch.setattr(Runtime, "from_settings", lambda _settings: runtime)
    monkeypatch.setattr(__main__, "prepare_cuda_libraries", lambda: None)
    monkeypatch.setattr(__main__, "build_server", observed_server)
    monkeypatch.setattr(__main__, "_install_signal_handlers", install)

    outcome, families = asyncio.run(scenario())

    assert outcome == 0
    assert families == [socket.AF_UNIX]


def _loadable_runtime(directory: Path) -> Runtime:
    return Runtime(
        None,
        FakeHearing(),
        dependencies=RuntimeDependencies(
            selection=VoiceSelectionStore(directory / "state", owner_uid=os.geteuid()),
            engine_factory=lambda _voice: FakeSpeaking(),
            artifact_checker=lambda _voice: True,
            default_voice=VoiceId.PIPER,
        ),
    )


@pytest.mark.parametrize(
    ("apps", "named"),
    [
        ('workload_client = "quiz-client"', "voices"),
        ('workload_client = "quiz-client"\nvoices = []', "voices"),
        ('workload_client = "quiz-client"\nvoices = ["alloy"]', "voices"),
        ('workload_client = "quiz-client"\nvoices = "piper"', "voices"),
        ('workload_client = "quiz-client"\nvoices = ["piper"]\nspeak = true', "speak"),
    ],
    ids=["missing", "empty", "unknown voice", "not a list", "speak flag"],
)
def test_the_service_does_not_start_when_an_apps_voices_are_wrong(
    tmp_path: Path, apps: str, named: str
) -> None:
    speech_root = Path(__file__).resolve().parents[1]
    environment = {
        **os.environ,
        "SPEECH_PRIVATE_DIRECTORY": str(tmp_path),
        "SPEECH_STATE_DIRECTORY": str(tmp_path / "state"),
        "PRESENTATOR_RUNTIME_UID": str(os.geteuid()),
        "SPEECH_CONFIG_FILE": str(
            write_caller_configuration(
                tmp_path, caller_configuration_text(apps=f"[apps.quiz]\n{apps}\n")
            )
        ),
        "PYTHONPATH": str(speech_root),
    }

    finished = subprocess.run(
        [sys.executable, "-m", "speech_runtime_service"],
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    assert finished.returncode != 0
    assert f"apps.quiz.{named}" in finished.stderr
    assert "alloy" not in finished.stderr
    assert "chatterbox" not in finished.stderr
    assert not (tmp_path / "speech.sock").exists()

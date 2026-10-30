"""Serving listens on the configured TCP address and ends with its stop signal."""

from __future__ import annotations

import asyncio
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from speech_runtime_service import __main__
from speech_runtime_service.config import Settings
from speech_runtime_service.service import Runtime, RuntimeDependencies
from speech_runtime_service.voices import VoiceId
from tests.support.fakes_support import FakeSpeaking
from tests.support.identity_support import (
    caller_configuration_text,
    write_caller_configuration,
)
from tests.support.tcp_server_support import free_loopback_port, listens_on

if TYPE_CHECKING:
    import socket

    import uvicorn

_SIGTERM_SERVER = """
import asyncio
import os
from speech_runtime_service import __main__
from speech_runtime_service.config import Settings
from speech_runtime_service.service import Runtime, RuntimeDependencies
from speech_runtime_service.voices import VoiceId
from tests.support.fakes_support import FakeSpeaking

__main__.prepare_cuda_libraries = lambda: None
Runtime.from_settings = lambda _settings: Runtime(
    None,
    dependencies=RuntimeDependencies(
        engine_factory=lambda _voice: FakeSpeaking(),
        artifact_checker=lambda _voice: True,
        voice=VoiceId.PIPER,
    ),
)
settings = Settings(port=int(os.environ["SPEECH_PORT"]))
raise SystemExit(asyncio.run(__main__.serve(settings)))
"""


def test_sigterm_stops_the_real_server_and_exits_zero_with_nothing_listening(
    tmp_path: Path,
) -> None:
    port = free_loopback_port()
    speech_root = Path(__file__).resolve().parents[1]
    environment = {
        **os.environ,
        "SPEECH_PORT": str(port),
        "SPEECH_CONFIG_FILE": str(write_caller_configuration(tmp_path)),
        "PYTHONPATH": os.pathsep.join([str(speech_root), str(speech_root / "scripts")]),
    }
    with subprocess.Popen(
        [sys.executable, "-c", _SIGTERM_SERVER], env=environment
    ) as server:
        try:
            deadline = time.monotonic() + 30
            while not listens_on(port):
                assert server.poll() is None
                assert time.monotonic() < deadline
                time.sleep(0.05)
            server.send_signal(signal.SIGTERM)
            assert server.wait(timeout=30) == 0
        finally:
            server.kill()

    assert not listens_on(port)


def test_python_dash_m_starts_the_service_listening_on_the_configured_port(
    tmp_path: Path,
) -> None:
    port = free_loopback_port()
    speech_root = Path(__file__).resolve().parents[1]
    environment = {
        **os.environ,
        "SPEECH_PORT": str(port),
        "SPEECH_STATE_DIRECTORY": str(tmp_path / "state"),
        "HF_HUB_OFFLINE": "1",
        "HF_HOME": str(tmp_path / "hf"),
        "SPEECH_CONFIG_FILE": str(write_caller_configuration(tmp_path)),
        "PYTHONPATH": str(speech_root),
    }
    with subprocess.Popen(
        [sys.executable, "-m", "speech_runtime_service"],
        cwd=tmp_path,
        env=environment,
    ) as server:
        try:
            deadline = time.monotonic() + 30
            while not listens_on(port):
                assert time.monotonic() < deadline
                time.sleep(0.05)
        finally:
            server.kill()

    assert list(tmp_path.glob("*.sock")) == []


_ENTRY_POINTS = [
    [sys.executable, "-m", "speech_runtime_service"],
    [str(Path(sys.executable).parent / "speech-runtime-service")],
]


@pytest.mark.parametrize("command", _ENTRY_POINTS, ids=["module", "console script"])
def test_help_prints_usage_and_exits_zero_without_loading_settings(
    command: list[str],
) -> None:
    environment = {**os.environ, "SPEECH_PORT": "not-a-port"}
    finished = subprocess.run(
        [*command, "--help"],
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    assert finished.returncode == 0
    assert "usage:" in finished.stdout
    assert "SPEECH_CONFIG_FILE" in finished.stdout
    assert "README" in finished.stdout
    assert finished.stderr == ""


@pytest.mark.parametrize("command", _ENTRY_POINTS, ids=["module", "console script"])
def test_an_unknown_argument_is_a_usage_error_without_starting(
    command: list[str],
) -> None:
    finished = subprocess.run(
        [*command, "--serve-faster"],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    assert finished.returncode != 0
    assert "usage:" in finished.stderr


def test_serve_listens_only_on_one_tcp_socket_of_the_configured_address(
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

    def install(stop: asyncio.Event) -> None:
        stopping.append(stop)

    async def scenario() -> tuple[int, list[object]]:
        serving = asyncio.create_task(
            __main__.serve(
                Settings(
                    port=port,
                    config_file=write_caller_configuration(tmp_path),
                )
            )
        )
        await started.wait()
        addresses = [
            bound.getsockname()
            for listening in built[0].servers
            for bound in listening.sockets
        ]
        stopping[0].set()
        return await serving, addresses

    port = free_loopback_port()
    runtime = _loadable_runtime()
    monkeypatch.setattr(Runtime, "from_settings", lambda _settings: runtime)
    monkeypatch.setattr(__main__, "prepare_cuda_libraries", lambda: None)
    monkeypatch.setattr(__main__, "build_server", observed_server)
    monkeypatch.setattr(__main__, "_install_signal_handlers", install)

    outcome, addresses = asyncio.run(scenario())

    assert outcome == 0
    assert addresses == [("127.0.0.1", port)]


def _loadable_runtime() -> Runtime:
    return Runtime(
        None,
        dependencies=RuntimeDependencies(
            engine_factory=lambda _voice: FakeSpeaking(),
            artifact_checker=lambda _voice: True,
            voice=VoiceId.PIPER,
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
        "SPEECH_PORT": str(free_loopback_port()),
        "SPEECH_STATE_DIRECTORY": str(tmp_path / "state"),
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

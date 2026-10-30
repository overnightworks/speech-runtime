"""Serving binds the configured address first and ends with the listener closed."""

import asyncio
import signal
import socket
import subprocess
import sys
import threading
import traceback
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pytest
from fastapi.testclient import TestClient
from speech_runtime_provider_contract import CHATTERBOX_SAMPLE_RATE

from speech_runtime_service import __main__
from speech_runtime_service.config import Settings
from speech_runtime_service.provider_process import ProviderLaunch, ProviderProcess
from speech_runtime_service.service import Runtime, RuntimeDependencies
from speech_runtime_service.voices import CHATTERBOX_SPEAKING_MODEL, VoiceId
from tests.support.fakes_support import FakeSpeaking
from tests.support.identity_support import write_caller_configuration
from tests.support.tcp_server_support import free_loopback_port, listens_on

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


def _serving_settings(directory: Path, **changes: object) -> Settings:
    """Settings that start `serve` in `directory` with a valid caller configuration."""
    return Settings(
        **{
            "port": free_loopback_port(),
            "config_file": write_caller_configuration(directory),
        }
        | changes
    )


class _BlockedSpeaking(FakeSpeaking):
    def __init__(self, started: threading.Event, release: threading.Event) -> None:
        super().__init__(ready=False)
        self._started = started
        self._release = release

    def load(self) -> None:
        self._started.set()
        assert self._release.wait(timeout=5)
        self.ready = True


@dataclass(slots=True)
class _CloseOrderingScenario:
    spawned: threading.Event
    stopping: list[asyncio.Event]
    closed: threading.Event
    cache: Path
    observations: list[bool]


class _CloseOrderingServer:
    def __init__(self, *, scenario: _CloseOrderingScenario) -> None:
        self.scenario = scenario
        self.stopped = asyncio.Event()

    @property
    def should_exit(self) -> bool:
        return self.stopped.is_set()

    @should_exit.setter
    def should_exit(self, value: bool) -> None:
        if value:
            self.stopped.set()

    async def serve(self, *, sockets: list[socket.socket] | None = None) -> None:
        del sockets
        assert await asyncio.to_thread(self.scenario.spawned.wait, 5)
        self.scenario.stopping[0].set()
        await self.stopped.wait()
        closed_first = await asyncio.to_thread(self.scenario.closed.wait, 1)
        self.scenario.observations.append(closed_first)
        if not closed_first:
            self.scenario.cache.mkdir(parents=True, exist_ok=True)
            (self.scenario.cache / "allow-ready").touch()


class _HealthServer:
    def __init__(
        self,
        *,
        started: threading.Event,
        release: threading.Event,
        observed: dict[str, object],
    ) -> None:
        self._app: object | None = None
        self._started = started
        self._release = release
        self._observed = observed
        self.should_exit = False

    def use(self, app: object) -> None:
        self._app = app

    async def serve(self, *, sockets: list[socket.socket] | None = None) -> None:
        del sockets
        assert await asyncio.to_thread(self._started.wait, 5)
        assert self._app is not None
        self._observed.update(TestClient(self._app).get("/health").json())
        self._release.set()


def test_failed_loader_stops_the_server_and_closes_the_listener(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = Runtime(
        None,
        dependencies=RuntimeDependencies(
            engine_factory=lambda _voice: FakeSpeaking(fail=True),
            artifact_checker=lambda _voice: True,
            voice=VoiceId.PIPER,
        ),
    )

    class Server:
        """A server that ends only when its orchestrator requests shutdown."""

        def __init__(self) -> None:
            self._stopped = asyncio.Event()

        @property
        def should_exit(self) -> bool:
            return self._stopped.is_set()

        @should_exit.setter
        def should_exit(self, value: bool) -> None:
            if value:
                self._stopped.set()

        async def serve(self, *, sockets: list[socket.socket] | None = None) -> None:
            del sockets
            await self._stopped.wait()

    monkeypatch.setattr(Runtime, "from_settings", lambda _settings: runtime)
    monkeypatch.setattr(__main__, "prepare_cuda_libraries", lambda: None)
    monkeypatch.setattr(__main__, "build_server", lambda _app: Server())
    monkeypatch.setattr(__main__, "_install_signal_handlers", lambda _stopping: None)

    settings = _serving_settings(tmp_path)

    outcome = asyncio.run(__main__.serve(settings))

    assert outcome == 1
    assert not listens_on(settings.port)


def test_shutdown_closes_a_pending_real_adapter_before_awaiting_tasks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "providers"
    entrypoint = root / "chatterbox/.venv/bin/speech-runtime-chatterbox-worker"
    entrypoint.parent.mkdir(parents=True)
    entrypoint.write_text(
        f"""#!{sys.executable}
import os
import time
from pathlib import Path
protocol = os.fdopen(os.dup(1), "wb", buffering=0)
null = os.open(os.devnull, os.O_WRONLY)
os.dup2(null, 1)
os.close(null)
cache = Path(__import__("sys").argv[-1])
while not (cache / "allow-ready").exists():
    time.sleep(0.01)
from speech_runtime_provider_contract import Frame, FrameKind, write_frame
write_frame(protocol, Frame(FrameKind.READY, 0, b"\\x00\\x01\\x00\\x00]\\xc0"))
while True:
    time.sleep(1)
""",
        encoding="utf-8",
    )
    entrypoint.chmod(0o700)
    spawned = threading.Event()
    closed = threading.Event()
    cache = tmp_path / "cache"

    class ObservedProvider(ProviderProcess):
        def close(self) -> None:
            try:
                super().close()
            finally:
                closed.set()

    engine = ObservedProvider(
        ProviderLaunch(
            model_name=CHATTERBOX_SPEAKING_MODEL,
            sample_rate=CHATTERBOX_SAMPLE_RATE,
            streams=True,
            executable=entrypoint,
            arguments=("--device", "cpu", "--cache", str(cache)),
        )
    )
    runtime = Runtime(
        None,
        dependencies=RuntimeDependencies(
            engine_factory=lambda _voice: engine,
            artifact_checker=lambda _voice: True,
            voice=VoiceId.CHATTERBOX,
        ),
    )
    real_popen = subprocess.Popen

    def observed_popen(*args: object, **kwargs: object) -> subprocess.Popen:
        process = real_popen(*args, **kwargs)
        spawned.set()
        return process

    stopping: list[asyncio.Event] = []
    close_before_await: list[bool] = []
    scenario = _CloseOrderingScenario(
        spawned=spawned,
        stopping=stopping,
        closed=closed,
        cache=cache,
        observations=close_before_await,
    )
    server = _CloseOrderingServer(scenario=scenario)

    def install(stop: asyncio.Event) -> None:
        stopping.append(stop)

    monkeypatch.setattr(Runtime, "from_settings", lambda _settings: runtime)
    monkeypatch.setattr(__main__, "prepare_cuda_libraries", lambda: None)
    monkeypatch.setattr(__main__, "build_server", lambda _app: server)
    monkeypatch.setattr(__main__, "_install_signal_handlers", install)
    monkeypatch.setattr(
        "speech_runtime_service.provider_process.subprocess.Popen", observed_popen
    )

    settings = _serving_settings(tmp_path)

    outcome = asyncio.run(__main__.serve(settings))

    assert outcome == 0
    assert close_before_await == [True]
    assert not listens_on(settings.port)


def test_owned_speaking_failure_stops_the_server_and_returns_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FatalSpeaking(FakeSpeaking):
        def __init__(self) -> None:
            super().__init__()
            self.callback: Callable[[], None] = lambda: None

        def set_fatal_callback(self, callback: Callable[[], None]) -> None:
            self.callback = callback

    speaking = FatalSpeaking()
    runtime = Runtime(
        speaking,
        dependencies=RuntimeDependencies(
            engine_factory=lambda _voice: speaking,
            artifact_checker=lambda _voice: True,
            voice=VoiceId.PIPER,
        ),
    )

    class Server:
        def __init__(self) -> None:
            self.stopped = asyncio.Event()

        @property
        def should_exit(self) -> bool:
            return self.stopped.is_set()

        @should_exit.setter
        def should_exit(self, value: bool) -> None:
            if value:
                self.stopped.set()

        async def serve(self, *, sockets: list[socket.socket] | None = None) -> None:
            del sockets
            speaking.callback()
            await self.stopped.wait()

    server = Server()
    monkeypatch.setattr(Runtime, "from_settings", lambda _settings: runtime)
    monkeypatch.setattr(__main__, "prepare_cuda_libraries", lambda: None)
    monkeypatch.setattr(__main__, "build_server", lambda _app: server)
    monkeypatch.setattr(__main__, "_install_signal_handlers", lambda _stopping: None)

    settings = _serving_settings(tmp_path)

    outcome = asyncio.run(__main__.serve(settings))

    assert outcome == 1
    assert server.should_exit
    assert not listens_on(settings.port)


def test_orchestrator_serves_health_from_its_loading_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    started = threading.Event()
    release = threading.Event()
    observed: dict[str, object] = {}
    card_memory_mb = 42

    runtime = Runtime(
        None,
        memory_probe=lambda: card_memory_mb,
        dependencies=RuntimeDependencies(
            engine_factory=lambda _voice: _BlockedSpeaking(started, release),
            artifact_checker=lambda _voice: True,
            voice=VoiceId.PIPER,
        ),
    )

    def healthy_server(app: object) -> _HealthServer:
        server = _HealthServer(started=started, release=release, observed=observed)
        server.use(app)
        return server

    monkeypatch.setattr(Runtime, "from_settings", lambda _settings: runtime)
    monkeypatch.setattr(__main__, "prepare_cuda_libraries", lambda: None)
    monkeypatch.setattr(__main__, "build_server", healthy_server)
    monkeypatch.setattr(__main__, "_install_signal_handlers", lambda _stopping: None)

    settings = _serving_settings(tmp_path)

    outcome = asyncio.run(__main__.serve(settings))

    assert outcome == 1
    assert observed["speaking"] == {
        "model": "unavailable",
        "ready": False,
        "streams": False,
        "sample_rate": 0,
    }
    assert observed["card_memory_mb"] == card_memory_mb


def test_signal_handler_stops_the_async_orchestrator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handlers: dict[signal.Signals, Callable[[], None]] = {}

    class Loop:
        def add_signal_handler(
            self, received: signal.Signals, callback: Callable[[], None]
        ) -> None:
            handlers[received] = callback

    stopping = asyncio.Event()

    def running_loop() -> Loop:
        return Loop()

    monkeypatch.setattr(__main__.asyncio, "get_running_loop", running_loop)

    handler_name = "_install_signal_handlers"
    install_handlers = getattr(__main__, handler_name)
    install_handlers(stopping)
    handlers[signal.SIGTERM]()

    assert stopping.is_set()


def test_signal_exit_is_zero_and_closes_the_listener(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = Runtime(FakeSpeaking())

    class Server:
        def __init__(self) -> None:
            self._stopped = asyncio.Event()

        @property
        def should_exit(self) -> bool:
            return self._stopped.is_set()

        @should_exit.setter
        def should_exit(self, value: bool) -> None:
            if value:
                self._stopped.set()

        async def serve(self, *, sockets: list[socket.socket] | None = None) -> None:
            del sockets
            await self._stopped.wait()

    monkeypatch.setattr(Runtime, "from_settings", lambda _settings: runtime)
    monkeypatch.setattr(__main__, "prepare_cuda_libraries", lambda: None)
    monkeypatch.setattr(__main__, "build_server", lambda _app: Server())
    monkeypatch.setattr(
        __main__,
        "_install_signal_handlers",
        lambda stopping: stopping.set(),
    )

    settings = _serving_settings(tmp_path)

    outcome = asyncio.run(__main__.serve(settings))

    assert outcome == 0
    assert not listens_on(settings.port)


def test_server_ending_on_its_own_returns_failure_and_closes_the_listener(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = Runtime(FakeSpeaking())

    class Server:
        should_exit = False

        async def serve(self, *, sockets: list[socket.socket] | None = None) -> None:
            del sockets

    monkeypatch.setattr(Runtime, "from_settings", lambda _settings: runtime)
    monkeypatch.setattr(__main__, "prepare_cuda_libraries", lambda: None)
    monkeypatch.setattr(__main__, "build_server", lambda _app: Server())
    monkeypatch.setattr(__main__, "_install_signal_handlers", lambda _stopping: None)

    settings = _serving_settings(tmp_path)

    outcome = asyncio.run(__main__.serve(settings))

    assert outcome == 1
    assert not listens_on(settings.port)


@pytest.mark.parametrize(
    "file_name",
    [None, "absent.toml"],
    ids=["unset", "missing"],
)
def test_serve_refuses_to_start_without_a_readable_caller_configuration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    file_name: str | None,
) -> None:
    monkeypatch.setattr(
        __main__, "prepare_cuda_libraries", lambda: pytest.fail("prepared CUDA")
    )
    monkeypatch.setattr(
        Runtime,
        "from_settings",
        lambda _settings: pytest.fail("constructed runtime"),
    )
    settings = _serving_settings(
        tmp_path, config_file=None if file_name is None else tmp_path / file_name
    )

    serving = __main__.serve(settings)

    with pytest.raises(ValueError, match="SPEECH_CONFIG_FILE") as refusal:
        asyncio.run(serving)
    rendered = "".join(traceback.format_exception(refusal.value))
    assert str(tmp_path) not in rendered


@pytest.mark.parametrize(
    "taken",
    [True, False],
    ids=["address taken", "host not on this machine"],
)
def test_serve_exits_nonzero_naming_the_address_before_any_runtime_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    taken: bool,  # noqa: FBT001 - a parametrized case selector
) -> None:
    monkeypatch.setattr(
        __main__, "prepare_cuda_libraries", lambda: pytest.fail("prepared CUDA")
    )
    monkeypatch.setattr(
        Runtime,
        "from_settings",
        lambda _settings: pytest.fail("constructed runtime"),
    )
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as holder:
        holder.bind(("127.0.0.1", 0))
        holder.listen()
        port = holder.getsockname()[1]
        host = "127.0.0.1" if taken else "192.0.2.1"
        settings = _serving_settings(tmp_path, host=host, port=port)

        outcome = asyncio.run(__main__.serve(settings))

    assert outcome == 1
    assert f"{host}:{port}" in caplog.text

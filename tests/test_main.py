"""The private listener has a narrow, ownership-checked lifetime."""

import asyncio
import fcntl
import os
import signal
import socket
import stat
import subprocess
import sys
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pytest
import uvicorn
from fastapi.testclient import TestClient
from speech_runtime_provider_contract import CHATTERBOX_SAMPLE_RATE

from speech_runtime_service import __main__
from speech_runtime_service.__main__ import PrivateSocket
from speech_runtime_service.config import CHATTERBOX_SPEAKING_MODEL, Settings
from speech_runtime_service.control import create_control_app
from speech_runtime_service.provider_process import ProviderLaunch, ProviderProcess
from speech_runtime_service.selection import VoiceSelectionStore
from speech_runtime_service.service import Runtime, RuntimeDependencies
from speech_runtime_service.voices import VoiceId
from tests.support.fakes_support import FakeHearing, FakeSpeaking
from tests.support.identity_support import valid_callers, write_caller_configuration

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


def _serving_settings(directory: Path, **changes: object) -> Settings:
    """Settings that start `serve` in `directory` with a valid caller configuration."""
    return Settings(
        **{
            "private_directory": directory,
            "PRESENTATOR_RUNTIME_UID": os.geteuid(),
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


def test_closed_uds_client_does_not_abandon_an_accepted_voice_load(
    tmp_path: Path,
) -> None:
    asyncio.run(_closed_uds_client_finishes_load(tmp_path))


async def _closed_uds_client_finishes_load(tmp_path: Path) -> None:
    entered = threading.Event()
    release = threading.Event()
    completed = threading.Event()

    class HeldSpeaking(FakeSpeaking):
        def load(self) -> None:
            entered.set()
            assert release.wait(timeout=5)
            self.ready = True
            completed.set()

    selection = VoiceSelectionStore(tmp_path / "state", owner_uid=os.geteuid())
    engine = HeldSpeaking(ready=False)
    runtime = Runtime(
        None,
        FakeHearing(),
        dependencies=RuntimeDependencies(
            selection=selection,
            engine_factory=lambda _voice: engine,
            artifact_checker=lambda _voice: True,
        ),
    )
    settings = Settings(
        private_directory=tmp_path, PRESENTATOR_RUNTIME_UID=os.geteuid()
    )
    shutdown_started = asyncio.Event()

    class ObservingServer(uvicorn.Server):
        async def shutdown(self, sockets: list[socket.socket] | None = None) -> None:
            shutdown_started.set()
            await super().shutdown(sockets=sockets)

    server = ObservingServer(
        uvicorn.Config(create_control_app(settings, runtime, valid_callers()))
    )
    with PrivateSocket(tmp_path, runtime_uid=os.geteuid()) as listener:
        serving = asyncio.create_task(server.serve(sockets=[listener]))
        await asyncio.to_thread(_close_after_posting_load, tmp_path / "speech.sock")
        assert await asyncio.to_thread(entered.wait, 5)
        assert not completed.is_set()
        server.should_exit = True
        await shutdown_started.wait()
        release.set()
        assert await asyncio.to_thread(completed.wait, 5)
        await serving

    assert selection.read() is VoiceId.CHATTERBOX
    admission = runtime.capture_speaking()
    assert admission is not None
    assert admission.speaking is engine
    admission.close()


def _close_after_posting_load(path: Path) -> None:
    request = (
        b"POST /voices/chatterbox/load HTTP/1.1\r\n"
        b"Host: speech\r\nContent-Length: 0\r\n\r\n"
    )
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        client.connect(str(path))
        client.sendall(request)


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


def test_private_socket_removes_only_the_inode_it_bound(tmp_path: Path) -> None:
    owner_only_file = 0o600
    with PrivateSocket(tmp_path, runtime_uid=os.geteuid()) as listener:
        socket_path = tmp_path / "speech.sock"
        inode = socket_path.stat().st_ino
        assert stat.S_IMODE(socket_path.stat().st_mode) == owner_only_file
        assert listener.fileno() >= 0

    assert not (tmp_path / "speech.sock").exists()
    assert inode > 0


@pytest.mark.parametrize(
    ("directory_mode", "uid_offset"),
    [(0o755, 0), (0o700, 1)],
    ids=["other-mode", "other-owner"],
)
def test_private_socket_refuses_an_unsafe_directory(
    tmp_path: Path, directory_mode: int, uid_offset: int
) -> None:
    directory = tmp_path / "private"
    directory.mkdir()
    directory.chmod(directory_mode)

    with (
        pytest.raises(PermissionError, match="private speech directory is unsafe"),
        PrivateSocket(directory, runtime_uid=os.geteuid() + uid_offset),
    ):
        pytest.fail("the socket must not be bound")

    assert not (directory / "speech.sock").exists()


def test_private_socket_preserves_an_existing_socket_path(tmp_path: Path) -> None:
    existing = tmp_path / "speech.sock"
    existing.touch()

    with (
        pytest.raises(FileExistsError),
        PrivateSocket(tmp_path, runtime_uid=os.geteuid()),
    ):
        pass

    assert existing.exists()


def test_private_socket_replaces_a_dead_leftover_socket(tmp_path: Path) -> None:
    path = tmp_path / "speech.sock"
    dead = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    dead.bind(str(path))
    dead.close()

    with PrivateSocket(tmp_path, runtime_uid=os.geteuid()) as listener:
        assert listener.family == socket.AF_UNIX
        assert stat.S_ISSOCK(path.lstat().st_mode)

    assert not path.exists()


def test_private_socket_refuses_a_concurrent_start_while_the_listener_is_running(
    tmp_path: Path,
) -> None:
    with (
        PrivateSocket(tmp_path, runtime_uid=os.geteuid()),
        pytest.raises(FileExistsError, match="starting or running"),
        PrivateSocket(tmp_path, runtime_uid=os.geteuid()),
    ):
        pass


def test_private_socket_refuses_when_the_socket_is_replaced_mid_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "speech.sock"
    dead = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    dead.bind(str(path))
    dead.close()

    def swap_then_report_dead(swapped_path: Path) -> bool:
        swapped_path.unlink()
        swapped_path.write_text("replacement", encoding="utf-8")
        return True

    monkeypatch.setattr(__main__, "_probe_is_dead", swap_then_report_dead)

    with (
        pytest.raises(FileExistsError, match="changed during reclamation"),
        PrivateSocket(tmp_path, runtime_uid=os.geteuid()),
    ):
        pass

    assert path.read_text(encoding="utf-8") == "replacement"


def test_private_socket_refuses_a_socket_with_a_live_listener(tmp_path: Path) -> None:
    path = tmp_path / "speech.sock"
    listening = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listening.bind(str(path))
    listening.listen()

    try:
        with (
            pytest.raises(FileExistsError),
            PrivateSocket(tmp_path, runtime_uid=os.geteuid()),
        ):
            pass

        assert stat.S_ISSOCK(path.lstat().st_mode)
    finally:
        listening.close()
        path.unlink()


def test_private_socket_lock_survives_release_and_the_next_start_reacquires_its_inode(
    tmp_path: Path,
) -> None:
    lock_path = tmp_path / "speech.sock.lock"

    with PrivateSocket(tmp_path, runtime_uid=os.geteuid()):
        pass

    first_inode = lock_path.stat().st_ino

    with PrivateSocket(tmp_path, runtime_uid=os.geteuid()):
        assert lock_path.stat().st_ino == first_inode


def test_private_socket_refuses_when_another_process_holds_the_released_lock_inode(
    tmp_path: Path,
) -> None:
    lock_path = tmp_path / "speech.sock.lock"

    with PrivateSocket(tmp_path, runtime_uid=os.geteuid()):
        pass  # starter A acquired, bound, and released; the lock path's inode remains

    # starter B takes that same, still-existing lock inode directly, as it would
    # between A's release and C's next attempt.
    holder_b = os.open(lock_path, os.O_RDWR)
    fcntl.flock(holder_b, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        with (
            pytest.raises(FileExistsError, match="starting or running"),
            PrivateSocket(tmp_path, runtime_uid=os.geteuid()),
        ):
            pass
    finally:
        fcntl.flock(holder_b, fcntl.LOCK_UN)
        os.close(holder_b)


def test_failed_loader_stops_the_server_and_cleans_the_owned_socket(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = Runtime(
        None,
        FakeHearing(fail=True),
        dependencies=RuntimeDependencies(
            selection=VoiceSelectionStore(tmp_path / "state", owner_uid=os.geteuid()),
            engine_factory=lambda _voice: FakeSpeaking(),
            artifact_checker=lambda _voice: True,
            default_voice=VoiceId.PIPER,
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
    monkeypatch.setattr(
        __main__, "_install_signal_handlers", lambda _stopping, _loading_stop: None
    )
    monkeypatch.setenv("PRESENTATOR_RUNTIME_UID", str(os.geteuid()))

    outcome = asyncio.run(__main__.serve(_serving_settings(tmp_path)))

    assert outcome == 1
    assert not (tmp_path / "speech.sock").exists()


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
        FakeHearing(),
        dependencies=RuntimeDependencies(
            selection=VoiceSelectionStore(tmp_path / "state", owner_uid=os.geteuid()),
            engine_factory=lambda _voice: engine,
            artifact_checker=lambda _voice: True,
            default_voice=VoiceId.CHATTERBOX,
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

    def install(stop: asyncio.Event, _loading_stop: threading.Event) -> None:
        stopping.append(stop)

    monkeypatch.setattr(Runtime, "from_settings", lambda _settings: runtime)
    monkeypatch.setattr(__main__, "prepare_cuda_libraries", lambda: None)
    monkeypatch.setattr(__main__, "build_server", lambda _app: server)
    monkeypatch.setattr(__main__, "_install_signal_handlers", install)
    monkeypatch.setattr(
        "speech_runtime_service.provider_process.subprocess.Popen", observed_popen
    )

    outcome = asyncio.run(__main__.serve(_serving_settings(tmp_path)))

    assert outcome == 0
    assert close_before_await == [True]
    assert not (tmp_path / "speech.sock").exists()


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
        FakeHearing(),
        dependencies=RuntimeDependencies(
            selection=VoiceSelectionStore(tmp_path / "state", owner_uid=os.geteuid()),
            engine_factory=lambda _voice: speaking,
            artifact_checker=lambda _voice: True,
            default_voice=VoiceId.PIPER,
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
    monkeypatch.setattr(
        __main__, "_install_signal_handlers", lambda _stopping, _loading_stop: None
    )

    outcome = asyncio.run(__main__.serve(_serving_settings(tmp_path)))

    assert outcome == 1
    assert server.should_exit
    assert not (tmp_path / "speech.sock").exists()


def test_runtime_skips_hearing_after_a_stop_during_speaking_load(
    tmp_path: Path,
) -> None:
    started = threading.Event()
    release = threading.Event()
    stop = threading.Event()

    hearing = FakeHearing(ready=False)
    runtime = Runtime(
        None,
        hearing,
        dependencies=RuntimeDependencies(
            selection=VoiceSelectionStore(tmp_path / "state", owner_uid=os.geteuid()),
            engine_factory=lambda _voice: _BlockedSpeaking(started, release),
            artifact_checker=lambda _voice: True,
            default_voice=VoiceId.PIPER,
        ),
    )
    loader = threading.Thread(target=runtime.load, args=(stop,))
    loader.start()
    assert started.wait(timeout=5)
    stop.set()
    release.set()
    loader.join(timeout=5)

    assert not loader.is_alive()
    assert hearing.ready is False


def test_orchestrator_serves_health_from_its_loading_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    started = threading.Event()
    release = threading.Event()
    observed: dict[str, object] = {}
    card_memory_mb = 42

    runtime = Runtime(
        None,
        FakeHearing(ready=False),
        memory_probe=lambda: card_memory_mb,
        dependencies=RuntimeDependencies(
            selection=VoiceSelectionStore(tmp_path / "state", owner_uid=os.geteuid()),
            engine_factory=lambda _voice: _BlockedSpeaking(started, release),
            artifact_checker=lambda _voice: True,
            default_voice=VoiceId.PIPER,
        ),
    )

    def healthy_server(app: object) -> _HealthServer:
        server = _HealthServer(started=started, release=release, observed=observed)
        server.use(app)
        return server

    monkeypatch.setattr(Runtime, "from_settings", lambda _settings: runtime)
    monkeypatch.setattr(__main__, "prepare_cuda_libraries", lambda: None)
    monkeypatch.setattr(__main__, "build_server", healthy_server)
    monkeypatch.setattr(
        __main__, "_install_signal_handlers", lambda _stopping, _loading_stop: None
    )

    outcome = asyncio.run(__main__.serve(_serving_settings(tmp_path)))

    assert outcome == 1
    assert observed["speaking"] == {
        "model": "unavailable",
        "ready": False,
        "streams": False,
        "sample_rate": 0,
    }
    assert observed["hearing"] == {"model": "fake-ears", "ready": False}
    assert observed["card_memory_mb"] == card_memory_mb


def test_signal_handler_stops_the_runtime_loader_and_async_orchestrator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handlers: dict[signal.Signals, Callable[[], None]] = {}

    class Loop:
        def add_signal_handler(
            self, received: signal.Signals, callback: Callable[[], None]
        ) -> None:
            handlers[received] = callback

    stopping = asyncio.Event()
    loading_stop = threading.Event()

    def running_loop() -> Loop:
        return Loop()

    monkeypatch.setattr(__main__.asyncio, "get_running_loop", running_loop)

    handler_name = "_install_signal_handlers"
    install_handlers = getattr(__main__, handler_name)
    install_handlers(stopping, loading_stop)
    handlers[signal.SIGTERM]()

    assert stopping.is_set()
    assert loading_stop.is_set()


def test_signal_exit_is_zero_and_cleans_the_owned_socket(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = Runtime(FakeSpeaking(), FakeHearing())

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
        lambda stopping, loading_stop: (loading_stop.set(), stopping.set()),
    )

    outcome = asyncio.run(__main__.serve(_serving_settings(tmp_path)))

    assert outcome == 0
    assert not (tmp_path / "speech.sock").exists()


def test_server_ending_on_its_own_returns_failure_and_cleans_the_owned_socket(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = Runtime(FakeSpeaking(), FakeHearing())

    class Server:
        should_exit = False

        async def serve(self, *, sockets: list[socket.socket] | None = None) -> None:
            del sockets

    monkeypatch.setattr(Runtime, "from_settings", lambda _settings: runtime)
    monkeypatch.setattr(__main__, "prepare_cuda_libraries", lambda: None)
    monkeypatch.setattr(__main__, "build_server", lambda _app: Server())
    monkeypatch.setattr(
        __main__, "_install_signal_handlers", lambda _stopping, _loading_stop: None
    )

    outcome = asyncio.run(__main__.serve(_serving_settings(tmp_path)))

    assert outcome == 1
    assert not (tmp_path / "speech.sock").exists()


@pytest.mark.parametrize(
    ("runtime_uid", "reason"),
    [(None, "required"), (os.geteuid() + 1, "effective UID")],
)
def test_invalid_runtime_uid_refuses_before_runtime_or_listener_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    runtime_uid: int | None,
    reason: str,
) -> None:
    monkeypatch.setattr(
        __main__, "prepare_cuda_libraries", lambda: pytest.fail("prepared CUDA")
    )
    monkeypatch.setattr(
        Runtime,
        "from_settings",
        lambda _settings: pytest.fail("constructed runtime"),
    )

    settings = _serving_settings(tmp_path, PRESENTATOR_RUNTIME_UID=runtime_uid)
    serving = __main__.serve(settings)
    with pytest.raises(PermissionError, match=reason):
        asyncio.run(serving)


@pytest.mark.parametrize(
    ("file_name", "failure"),
    [(None, ValueError), ("absent.toml", FileNotFoundError)],
    ids=["unset", "missing"],
)
def test_serve_refuses_to_start_without_a_readable_caller_configuration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    file_name: str | None,
    failure: type[Exception],
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

    with pytest.raises(failure):
        asyncio.run(serving)

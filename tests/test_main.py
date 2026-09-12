"""The private listener has a narrow, ownership-checked lifetime."""

import asyncio
import os
import signal
import socket
import stat
import threading
from collections.abc import Callable
from pathlib import Path

import pytest
import uvicorn
from fastapi.testclient import TestClient

from speech import __main__
from speech.__main__ import PrivateSocket
from speech.config import Settings
from speech.selection import VoiceSelectionStore
from speech.service import Runtime, RuntimeDependencies, create_app, create_control_app
from speech.voices import VoiceId
from tests.conftest import FakeHearing, FakeSpeaking


class _BlockedSpeaking(FakeSpeaking):
    def __init__(self, started: threading.Event, release: threading.Event) -> None:
        super().__init__(ready=False)
        self._started = started
        self._release = release

    def load(self) -> None:
        self._started.set()
        assert self._release.wait(timeout=5)
        self.ready = True


def test_closed_uds_client_does_not_abandon_an_accepted_voice_load(tmp_path) -> None:
    asyncio.run(_closed_uds_client_finishes_load(tmp_path))


async def _closed_uds_client_finishes_load(tmp_path) -> None:
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

    server = ObservingServer(uvicorn.Config(create_control_app(settings, runtime)))
    server.install_signal_handlers = lambda: None
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
    assert runtime.capture_speaking() is engine


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
        observes_health: bool,
        started: threading.Event,
        release: threading.Event,
        observed: dict[str, object],
    ) -> None:
        self._app: object | None = None
        self._observes_health = observes_health
        self._started = started
        self._release = release
        self._observed = observed
        self._stopped = asyncio.Event()

    @property
    def should_exit(self) -> bool:
        return self._stopped.is_set()

    @should_exit.setter
    def should_exit(self, value: bool) -> None:
        if value:
            self._stopped.set()

    def use(self, app: object) -> None:
        self._app = app

    async def serve(self, *, sockets=None) -> None:
        del sockets
        if self._observes_health:
            assert await asyncio.to_thread(self._started.wait, 5)
            assert self._app is not None
            self._observed.update(TestClient(self._app).get("/health").json())
            self._release.set()
            return
        await self._stopped.wait()


def test_private_socket_removes_only_the_inode_it_bound(tmp_path) -> None:
    with PrivateSocket(tmp_path, runtime_uid=os.geteuid()) as listener:
        socket_path = tmp_path / "speech.sock"
        inode = socket_path.stat().st_ino
        assert stat.S_IMODE(socket_path.stat().st_mode) == 0o600
        assert listener.fileno() >= 0

    assert not (tmp_path / "speech.sock").exists()
    assert inode > 0


def test_private_socket_preserves_an_existing_socket_path(tmp_path) -> None:
    existing = tmp_path / "speech.sock"
    existing.touch()

    with (
        pytest.raises(FileExistsError),
        PrivateSocket(tmp_path, runtime_uid=os.geteuid()),
    ):
        pass

    assert existing.exists()


def test_failed_loader_stops_both_servers_and_cleans_the_owned_socket(
    tmp_path, monkeypatch: pytest.MonkeyPatch
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

        async def serve(self, *, sockets=None) -> None:
            del sockets
            await self._stopped.wait()

    servers = iter((Server(), Server()))
    monkeypatch.setattr(Runtime, "from_settings", lambda _settings: runtime)
    monkeypatch.setattr(__main__, "prepare_cuda_libraries", lambda: None)
    monkeypatch.setattr(__main__, "_server", lambda _app, _settings: next(servers))
    monkeypatch.setattr(
        __main__, "_install_signal_handlers", lambda _stopping, _loading_stop: None
    )
    monkeypatch.setenv("PRESENTATOR_RUNTIME_UID", str(os.geteuid()))

    outcome = asyncio.run(__main__.serve(Settings(private_directory=tmp_path)))

    assert outcome == 1
    assert not (tmp_path / "speech.sock").exists()


def test_runtime_skips_hearing_after_a_stop_during_speaking_load(tmp_path) -> None:
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
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    started = threading.Event()
    release = threading.Event()
    observed: dict[str, object] = {}

    runtime = Runtime(
        None,
        FakeHearing(ready=False),
        memory_probe=lambda: 42,
        dependencies=RuntimeDependencies(
            selection=VoiceSelectionStore(tmp_path / "state", owner_uid=os.geteuid()),
            engine_factory=lambda _voice: _BlockedSpeaking(started, release),
            artifact_checker=lambda _voice: True,
            default_voice=VoiceId.PIPER,
        ),
    )
    servers = iter(
        (
            _HealthServer(
                observes_health=True,
                started=started,
                release=release,
                observed=observed,
            ),
            _HealthServer(
                observes_health=False,
                started=started,
                release=release,
                observed=observed,
            ),
        )
    )

    def build_server(app: object, _settings: Settings) -> _HealthServer:
        server = next(servers)
        server.use(app)
        return server

    monkeypatch.setattr(Runtime, "from_settings", lambda _settings: runtime)
    monkeypatch.setattr(__main__, "prepare_cuda_libraries", lambda: None)
    monkeypatch.setattr(__main__, "_server", build_server)
    monkeypatch.setattr(
        __main__, "_install_signal_handlers", lambda _stopping, _loading_stop: None
    )

    outcome = asyncio.run(
        __main__.serve(
            Settings(private_directory=tmp_path, PRESENTATOR_RUNTIME_UID=os.geteuid())
        )
    )

    assert outcome == 1
    assert observed["speaking"] == {
        "model": "unavailable",
        "ready": False,
        "streams": False,
        "sample_rate": 0,
    }
    assert observed["hearing"] == {"model": "fake-ears", "ready": False}
    assert observed["card_memory_mb"] == 42


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
    tmp_path, monkeypatch: pytest.MonkeyPatch
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

        async def serve(self, *, sockets=None) -> None:
            del sockets
            await self._stopped.wait()

    servers = iter((Server(), Server()))
    monkeypatch.setattr(Runtime, "from_settings", lambda _settings: runtime)
    monkeypatch.setattr(__main__, "prepare_cuda_libraries", lambda: None)
    monkeypatch.setattr(__main__, "_server", lambda _app, _settings: next(servers))
    monkeypatch.setattr(
        __main__,
        "_install_signal_handlers",
        lambda stopping, loading_stop: (loading_stop.set(), stopping.set()),
    )

    outcome = asyncio.run(
        __main__.serve(
            Settings(private_directory=tmp_path, PRESENTATOR_RUNTIME_UID=os.geteuid())
        )
    )

    assert outcome == 0
    assert not (tmp_path / "speech.sock").exists()


@pytest.mark.parametrize("ending_server", [0, 1])
def test_either_server_ending_stops_its_peer_and_cleans_the_owned_socket(
    tmp_path, monkeypatch: pytest.MonkeyPatch, ending_server: int
) -> None:
    runtime = Runtime(FakeSpeaking(), FakeHearing())

    class Server:
        def __init__(self, *, ends: bool) -> None:
            self._ends = ends
            self._stopped = asyncio.Event()

        @property
        def should_exit(self) -> bool:
            return self._stopped.is_set()

        @should_exit.setter
        def should_exit(self, value: bool) -> None:
            if value:
                self._stopped.set()

        async def serve(self, *, sockets=None) -> None:
            del sockets
            if not self._ends:
                await self._stopped.wait()

    servers = iter((Server(ends=ending_server == 0), Server(ends=ending_server == 1)))
    monkeypatch.setattr(Runtime, "from_settings", lambda _settings: runtime)
    monkeypatch.setattr(__main__, "prepare_cuda_libraries", lambda: None)
    monkeypatch.setattr(__main__, "_server", lambda _app, _settings: next(servers))
    monkeypatch.setattr(
        __main__, "_install_signal_handlers", lambda _stopping, _loading_stop: None
    )

    outcome = asyncio.run(
        __main__.serve(
            Settings(private_directory=tmp_path, PRESENTATOR_RUNTIME_UID=os.geteuid())
        )
    )

    assert outcome == 1
    assert not (tmp_path / "speech.sock").exists()


def test_server_wrapper_disables_uvicorn_signal_capture(tmp_path) -> None:
    server_name = "_server"
    build_server = getattr(__main__, server_name)
    server = build_server(
        create_app(runtime=Runtime(FakeSpeaking(), FakeHearing())),
        Settings(private_directory=tmp_path, PRESENTATOR_RUNTIME_UID=os.geteuid()),
    )

    assert server.install_signal_handlers() is None


@pytest.mark.parametrize(
    ("runtime_uid", "reason"),
    [(None, "required"), (os.geteuid() + 1, "effective UID")],
)
def test_invalid_runtime_uid_refuses_before_runtime_or_listener_work(
    tmp_path,
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

    settings = Settings(private_directory=tmp_path, PRESENTATOR_RUNTIME_UID=runtime_uid)
    with pytest.raises(PermissionError, match=reason):
        asyncio.run(__main__.serve(settings))

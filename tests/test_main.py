"""Serving binds the configured address first and ends with the listener closed."""

import asyncio
import os
import signal
import socket
import subprocess
import sys
import threading
import traceback
from typing import TYPE_CHECKING

import pytest
from fastapi.testclient import TestClient
from speech_provider_contract import QWEN_MODEL_ID, QWEN_SAMPLE_RATE

from speech_client.wire import VOICES, Language, VoiceId
from speech_service import __main__
from speech_service.__main__ import StopPolicy
from speech_service.clock import LoopClock
from speech_service.config import Settings
from speech_service.hearing import Hearing
from speech_service.provider_process import ProviderLaunch, ProviderProcess
from speech_service.service import Runtime, RuntimeDependencies
from tests.support.fakes_support import FakeSpeaking
from tests.support.hear_support import WAIT_SECONDS, FakeHearing, loaded
from tests.support.identity_support import write_caller_configuration
from tests.support.tcp_server_support import free_loopback_port, listens_on
from tests.support.worker_support import provider_worker

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


class _ShutdownOnlyServer:
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


class _SignalledServer(_ShutdownOnlyServer):
    """A server that SIGTERMs the process once it runs, or once `after` is set."""

    def __init__(self, after: threading.Event | None = None) -> None:
        super().__init__()
        self._after = after

    async def serve(self, *, sockets: list[socket.socket] | None = None) -> None:
        if self._after is not None:
            assert await asyncio.to_thread(self._after.wait, 5)
        os.kill(os.getpid(), signal.SIGTERM)
        await super().serve(sockets=sockets)


def test_failed_loader_stops_the_server_and_closes_the_listener(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = Runtime(
        {VoiceId.THORSTEN: None},
        dependencies=RuntimeDependencies(
            engine_factory=lambda _voice: FakeSpeaking(fail=True),
            missing_artifacts=lambda _voice: (),
        ),
    )
    monkeypatch.setattr(Runtime, "from_settings", lambda _settings: runtime)
    monkeypatch.setattr(__main__, "build_server", lambda _app: _ShutdownOnlyServer())
    monkeypatch.setattr(__main__, "_install_signal_handlers", lambda _stopping: None)

    settings = _serving_settings(tmp_path)

    outcome = asyncio.run(__main__.serve(settings))

    assert outcome == 1
    assert not listens_on(settings.port)


def test_an_offered_piper_voice_without_its_files_ends_the_start_naming_the_voice(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setattr(__main__, "build_server", lambda _app: _ShutdownOnlyServer())
    monkeypatch.setattr(__main__, "_install_signal_handlers", lambda _stopping: None)
    settings = _serving_settings(tmp_path).model_copy(
        update={"voices": frozenset({VoiceId.NORMAN}), "voice_cache": tmp_path}
    )

    outcome = asyncio.run(__main__.serve(settings))

    assert outcome == 1
    assert VoiceId.NORMAN.value in caplog.text
    assert "speech-fetch-voice" in caplog.text
    assert not listens_on(settings.port)


def _provider_ready_on_release(root: Path) -> Path:
    """A provider entrypoint that reports ready once its cache holds `allow-ready`."""
    entrypoint = provider_worker(root)
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
from speech_provider_contract import Frame, FrameKind, write_frame
write_frame(protocol, Frame(FrameKind.READY, 0, b"\\x00\\x01\\x00\\x00]\\xc0"))
while True:
    time.sleep(1)
""",
        encoding="utf-8",
    )
    entrypoint.chmod(0o700)
    return entrypoint


class _ObservedProvider(ProviderProcess):
    def __init__(self, launch: ProviderLaunch, closed: threading.Event) -> None:
        super().__init__(launch)
        self._observed_close = closed

    def close(self) -> None:
        try:
            super().close()
        finally:
            self._observed_close.set()


def test_shutdown_closes_a_pending_real_adapter_before_awaiting_its_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    entrypoint = _provider_ready_on_release(tmp_path / "providers")
    spawned = threading.Event()
    closed = threading.Event()
    cache = tmp_path / "cache"

    engine = _ObservedProvider(
        ProviderLaunch(
            model_name=QWEN_MODEL_ID,
            sample_rate=QWEN_SAMPLE_RATE,
            executable=entrypoint,
            arguments=("--device", "cpu", "--cache", str(cache)),
        ),
        closed,
    )
    runtime = Runtime(
        {VoiceId.THORSTEN: None},
        dependencies=RuntimeDependencies(
            engine_factory=lambda _voice: engine,
            missing_artifacts=lambda _voice: (),
        ),
    )
    real_popen = subprocess.Popen

    def observed_popen(*args: object, **kwargs: object) -> subprocess.Popen:
        process = real_popen(*args, **kwargs)
        spawned.set()
        return process

    server = _SignalledServer(spawned)

    def release_the_load() -> None:
        cache.mkdir(parents=True, exist_ok=True)
        (cache / "allow-ready").touch()

    # Only a missing close would need it: the load then waits for its release.
    watchdog = threading.Timer(5, release_the_load)

    monkeypatch.setattr(Runtime, "from_settings", lambda _settings: runtime)
    monkeypatch.setattr(__main__, "build_server", lambda _app: server)
    monkeypatch.setattr(
        "speech_service.provider_process.subprocess.Popen", observed_popen
    )

    watchdog.start()
    try:
        outcome = asyncio.run(__main__.serve(_serving_settings(tmp_path)))
    finally:
        watchdog.cancel()

    assert outcome == 0
    assert closed.is_set()
    assert not (cache / "allow-ready").exists()


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
        {VoiceId.THORSTEN: speaking},
        dependencies=RuntimeDependencies(
            engine_factory=lambda _voice: speaking,
            missing_artifacts=lambda _voice: (),
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
    """WIRE-123: orchestrator serves health from its loading runtime."""
    started = threading.Event()
    release = threading.Event()
    observed: dict[str, object] = {}
    card_memory_mb = 42

    runtime = Runtime(
        {VoiceId.THORSTEN: None},
        memory_probe=lambda: card_memory_mb,
        dependencies=RuntimeDependencies(
            engine_factory=lambda _voice: _BlockedSpeaking(started, release),
            missing_artifacts=lambda _voice: (),
        ),
    )

    def healthy_server(app: object) -> _HealthServer:
        server = _HealthServer(started=started, release=release, observed=observed)
        server.use(app)
        return server

    monkeypatch.setattr(Runtime, "from_settings", lambda _settings: runtime)
    monkeypatch.setattr(__main__, "build_server", healthy_server)
    monkeypatch.setattr(__main__, "_install_signal_handlers", lambda _stopping: None)

    settings = _serving_settings(tmp_path)

    outcome = asyncio.run(__main__.serve(settings))

    assert outcome == 1
    assert observed["speaking"] == {
        "ready": False,
        "voices": {
            VoiceId.THORSTEN.value: {
                "model": VOICES[VoiceId.THORSTEN].model,
                "name": VOICES[VoiceId.THORSTEN].name,
                "languages": [*VOICES[VoiceId.THORSTEN].languages],
                "takes_instructions": VOICES[VoiceId.THORSTEN].takes_instructions,
                "ready": False,
                "sample_rate": 0,
            }
        },
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
    runtime = Runtime({VoiceId.THORSTEN: FakeSpeaking()})

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


def _signalled_stop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    hearing: Hearing,
    after: threading.Event | None = None,
) -> Settings:
    """Serve over `hearing` with a server that SIGTERMs the process once it runs.

    With `after` the SIGTERM waits until that event is set.
    """
    runtime = Runtime(
        {VoiceId.THORSTEN: None},
        dependencies=RuntimeDependencies(
            engine_factory=lambda _voice: FakeSpeaking(),
            missing_artifacts=lambda _voice: (),
        ),
        hearing=hearing,
    )
    monkeypatch.setattr(Runtime, "from_settings", lambda _settings: runtime)
    monkeypatch.setattr(__main__, "build_server", lambda _app: _SignalledServer(after))
    return _serving_settings(tmp_path, hearing_final_wait_seconds=0.05)


def test_a_session_that_outlasts_the_drain_bound_leaves_the_exit_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """WIRE-147: the drain ends at its bound, whatever still runs."""
    hearing = loaded(FakeHearing(), drain_turn_seconds=0.0)
    place = hearing.open_session(1)
    assert place is not None
    settings = _signalled_stop(tmp_path, monkeypatch, hearing)

    outcome = asyncio.run(__main__.serve(settings, StopPolicy(grace_seconds=0.05)))

    assert outcome == 0


def test_engines_that_will_not_close_end_the_process_through_the_abort_with_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """WIRE-147: a decode that cannot be interrupted does not hold the exit."""
    engine = FakeHearing(hold=True)
    hearing = loaded(engine)
    lease = hearing.gate.try_acquire()
    assert lease is not None
    decode = hearing.submit(lease, lambda ear: ear.transcribe(b"", (Language.DE,)))
    assert engine.entered.wait(WAIT_SECONDS)
    settings = _signalled_stop(tmp_path, monkeypatch, hearing)
    aborted: list[int] = []
    policy = StopPolicy(grace_seconds=0.05, abort=aborted.append)

    outcome = asyncio.run(__main__.serve(settings, policy))

    engine.release.set()
    decode.result(WAIT_SECONDS)
    assert outcome == 0
    assert aborted == [0]


def test_a_hearing_load_that_will_not_end_ends_the_process_through_the_abort_with_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """WIRE-147: a load that cannot be interrupted does not hold the exit."""
    engine = FakeHearing(hold_load=True)
    settings = _signalled_stop(
        tmp_path, monkeypatch, Hearing(engine, LoopClock()), engine.load_entered
    )
    aborted: list[int] = []
    policy = StopPolicy(grace_seconds=0.05, abort=aborted.append)

    async def stopped_during_the_load() -> int:
        try:
            return await __main__.serve(settings, policy)
        finally:
            engine.load_release.set()

    outcome = asyncio.run(stopped_during_the_load())

    assert outcome == 0
    assert aborted == [0]


def test_server_ending_on_its_own_returns_failure_and_closes_the_listener(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = Runtime({VoiceId.THORSTEN: FakeSpeaking()})

    class Server:
        should_exit = False

        async def serve(self, *, sockets: list[socket.socket] | None = None) -> None:
            del sockets

    monkeypatch.setattr(Runtime, "from_settings", lambda _settings: runtime)
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
    """WIRE-112: serve refuses to start without a readable configuration."""
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

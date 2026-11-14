"""A service started from its settings, over a fake Piper, ear and worker."""

from __future__ import annotations

import asyncio
import contextlib
import socket
import sys
import threading
import time
from dataclasses import dataclass
from types import SimpleNamespace
from typing import TYPE_CHECKING

import httpx2
from speech_provider_contract import (
    QWEN_ARTIFACTS,
    QWEN_MODEL_ID,
    QWEN_REVISION,
)

from speech_client.wire import HEALTH_PATH, Health, VoiceId
from speech_service import __main__
from speech_service.__main__ import StopPolicy
from speech_service.config import Settings
from speech_service.control import create_control_app
from speech_service.hearing import DRAIN_TURN_SECONDS, Hearing
from speech_service.service import Runtime
from speech_service.voices import PIPER_VOICES, piper_voice_paths
from tests.support.hear_support import FakeHearing
from tests.support.identity_support import valid_callers, valid_headers
from tests.support.speak_support import ASGI_TIMEOUT_SECONDS, wav_body
from tests.support.tcp_server_support import answers_http
from tests.support.worker_support import (
    WORKER_WAIT_SECONDS,
    closed_provider,
    fail_every_reap,
    release_exit,
    worker_starts,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator
    from concurrent.futures import Future
    from pathlib import Path

    import pytest
    from fastapi import FastAPI

    from speech_service.service import SpeakingEngine

PIPER_SAMPLE_RATE = 22_050
PIPER_PCM = b"\x00\x01"
LOOPBACK = "127.0.0.1"
STOP_GRACE_SECONDS = 1.0


def touch_snapshot(
    cache: Path, repository: str, revision: str, files: Iterable[str]
) -> None:
    """The files of `repository` at `revision`, as the Hugging Face cache keeps them."""
    snapshot = cache / f"models--{repository.replace('/', '--')}" / "snapshots"
    for name in files:
        artifact = snapshot / revision / name
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.touch()


def install_qwen(cache: Path) -> None:
    """Qwen's pinned weights in `cache`."""
    touch_snapshot(cache, QWEN_MODEL_ID, QWEN_REVISION, QWEN_ARTIFACTS)


def install_piper(monkeypatch: pytest.MonkeyPatch, voice_cache: Path) -> None:
    """Piper's files in `voice_cache`, loaded by a fake `piper` module."""

    class LocalVoice:
        config = SimpleNamespace(sample_rate=PIPER_SAMPLE_RATE)

        def synthesize(self, _text: str) -> Iterable[SimpleNamespace]:
            return [SimpleNamespace(audio_int16_bytes=PIPER_PCM)]

    monkeypatch.setitem(
        sys.modules,
        "piper",
        SimpleNamespace(PiperVoice=SimpleNamespace(load=lambda _path: LocalVoice())),
    )
    voice_cache.mkdir(exist_ok=True)
    for voice in PIPER_VOICES:
        for path in piper_voice_paths(voice_cache, voice.value):
            path.touch()


@dataclass(frozen=True, slots=True)
class GpuVoiceStage:
    """The runtime `Runtime.from_settings` built, its app, and the worker's cache."""

    runtime: Runtime
    settings: Settings
    cache: Path
    hearing: FakeHearing

    def app(self) -> FastAPI:
        """The private app over the runtime, with one registered app."""
        return create_control_app(self.settings, self.runtime, valid_callers())

    def starts(self) -> list[int]:
        """The pids of every worker started so far."""
        return worker_starts(self.cache)


@dataclass(frozen=True, slots=True)
class StageSetup:
    """The fake worker's behaviour and what the stage installs and offers.

    The start load has run unless `started` is false. With `reaps_fail`, every
    worker's cleanup fails once it is reaped, so the stage's close may raise.
    """

    behavior: str = "held-ready"
    voices: frozenset[VoiceId] = frozenset({VoiceId.THORSTEN, VoiceId.RYAN})
    qwen_installed: bool = True
    hearing: FakeHearing | None = None
    started: bool = True
    reaps_fail: bool = False
    drain_turn_seconds: float = DRAIN_TURN_SECONDS


@contextlib.contextmanager
def gpu_voice_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    setup: StageSetup | None = None,
    **settings: float,
) -> Iterator[GpuVoiceStage]:
    """A service built by `Runtime.from_settings` as `setup` says.

    On exit every held worker exit is released and the runtime is closed.
    """
    setup = setup or StageSetup()
    behavior = setup.behavior
    root, cache = closed_provider(tmp_path, behavior)
    install_piper(monkeypatch, tmp_path / "piper")
    if setup.qwen_installed:
        install_qwen(cache)
    ear = setup.hearing or FakeHearing()
    if setup.reaps_fail:
        fail_every_reap(monkeypatch)
    configured = Settings(
        voices=setup.voices,
        provider_root=root,
        huggingface_cache=cache,
        voice_cache=tmp_path / "piper",
        device="cpu",
        **settings,
    )
    runtime = Runtime.from_settings(configured)
    runtime.hearing = Hearing(
        ear, runtime.clock, drain_turn_seconds=setup.drain_turn_seconds
    )
    try:
        if setup.started:
            runtime.load()
        yield GpuVoiceStage(runtime, configured, cache, ear)
    finally:
        for pid in worker_starts(cache):
            release_exit(cache, pid)
        try:
            runtime.close()
        except RuntimeError:
            if not setup.reaps_fail:
                raise


async def health_of(app: FastAPI) -> Health:
    """GET /health through the raw ASGI boundary, on the running loop."""
    messages: list[dict[str, object]] = []

    async def receive() -> dict[str, object]:
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message: dict[str, object]) -> None:
        messages.append(message)

    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "GET",
        "path": HEALTH_PATH,
        "raw_path": HEALTH_PATH.encode(),
        "query_string": b"",
        "headers": [],
        "client": ("test", 123),
        "server": ("test", 80),
    }
    await asyncio.wait_for(app(scope, receive, send), timeout=ASGI_TIMEOUT_SECONDS)
    return Health.model_validate_json(wav_body(messages))


class LoadWaits:
    """Counts the requests that took hold of a voice's load through the runtime."""

    def __init__(self, runtime: Runtime) -> None:
        """Observe `runtime.load_on_demand`, which hands each request its load."""
        self._count = 0
        self._changed = threading.Condition()
        hand_out = runtime.load_on_demand

        def counted(voice: VoiceId) -> Future[SpeakingEngine | None] | None:
            load = hand_out(voice)
            with self._changed:
                self._count += 1
                self._changed.notify_all()
            return load

        runtime.load_on_demand = counted

    def wait_for(self, count: int) -> None:
        """Return once `count` requests hold the load; fails after a bounded wait."""
        with self._changed:
            assert self._changed.wait_for(
                lambda: self._count >= count, timeout=WORKER_WAIT_SECONDS
            ), f"{count} requests never waited for the load"


def speak_headers() -> dict[str, str]:
    """The headers of a speak request with a valid pair of tokens."""
    return {"content-type": "application/json", **valid_headers()}


class Serving:
    """`__main__`'s serve loop over a stage's runtime, on a loopback listener.

    It stops as its SIGTERM handler stops it; `fatal` notes the fatal callback.
    """

    def __init__(self, stage: GpuVoiceStage, monkeypatch: pytest.MonkeyPatch) -> None:
        """Route `__main__` to the stage's runtime; nothing serves until `start`."""
        self._stage = stage
        self._stopping: list[asyncio.Event] = []
        self.fatal = threading.Event()
        runtime = stage.runtime
        set_fatal_callback = runtime.set_fatal_callback

        def noting_fatal(callback: Callable[[], None]) -> None:
            def both() -> None:
                self.fatal.set()
                callback()

            set_fatal_callback(both)

        runtime.set_fatal_callback = noting_fatal
        monkeypatch.setattr(Runtime, "from_settings", lambda _settings: runtime)
        monkeypatch.setattr(__main__, "_install_signal_handlers", self._stopping.append)
        self.listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.listener.bind((LOOPBACK, 0))
        self.listener.listen()
        self.port: int = self.listener.getsockname()[1]
        self._task: asyncio.Task[int] | None = None

    async def start(self) -> httpx2.AsyncClient:
        """Serve; once the start load ended, a client whose tokens are valid."""
        self._task = asyncio.create_task(
            __main__._serve_on(  # noqa: SLF001 - the serve loop behind the TCP listener
                self._stage.settings,
                valid_callers(),
                self.listener,
                StopPolicy(grace_seconds=STOP_GRACE_SECONDS),
            )
        )
        deadline = time.monotonic() + WORKER_WAIT_SECONDS
        while not await asyncio.to_thread(answers_http, self.port):
            assert time.monotonic() < deadline
        while self._stage.runtime.loading:
            assert time.monotonic() < deadline
            await asyncio.sleep(0.01)
        return httpx2.AsyncClient(
            base_url=f"http://{LOOPBACK}:{self.port}",
            timeout=WORKER_WAIT_SECONDS,
            headers=speak_headers(),
        )

    def sigterm(self) -> None:
        """Stop the serve loop as its SIGTERM handler does."""
        self._stopping[0].set()

    @property
    def task(self) -> asyncio.Task[int]:
        """The running serve loop; its result is the exit status."""
        assert self._task is not None
        return self._task

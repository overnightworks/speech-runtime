"""The fixed Qwen download never broadens into selection or ambient access."""

import logging
import os
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from presentator_speech_provider_contract import (
    QWEN_ARTIFACTS,
    QWEN_MODEL_ID,
    QWEN_REVISION,
)

from speech import qwen_download, voices
from speech.config import Settings
from speech.control import create_control_app
from speech.selection import VoiceSelectionStore
from speech.service import Runtime, RuntimeDependencies, create_app
from speech.voices import (
    VoiceDownloadOutcome,
    VoiceId,
    VoiceLoadOutcome,
    VoiceState,
)
from tests.conftest import FakeHearing, FakeSpeaking


class _TrackedSpeaking(FakeSpeaking):
    def __init__(self, *, ready: bool = True) -> None:
        super().__init__(ready=ready)
        self.closed = 0

    def close(self) -> None:
        self.closed += 1


@dataclass(frozen=True, slots=True)
class _RuntimeOverrides:
    worker_usable: Callable[[], bool] = lambda: True
    speaking: FakeSpeaking | None = None
    factory: Callable[[VoiceId], FakeSpeaking] | None = None
    selection: VoiceSelectionStore | None = None


def _runtime(
    tmp_path: Path,
    *,
    complete: Callable[[], bool],
    download: Callable[[], None] | None,
    overrides: _RuntimeOverrides | None = None,
) -> Runtime:
    options = overrides or _RuntimeOverrides()
    return Runtime(
        options.speaking,
        FakeHearing(),
        dependencies=RuntimeDependencies(
            selection=options.selection
            or VoiceSelectionStore(tmp_path / "state", owner_uid=os.geteuid()),
            engine_factory=options.factory or (lambda _voice: FakeSpeaking()),
            artifact_checker=lambda voice: voice is not VoiceId.QWEN or complete(),
            qwen_worker_is_usable=options.worker_usable,
            qwen_download=download,
            default_voice=VoiceId.PIPER,
        ),
    )


def _qwen_state(
    runtime: Runtime,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    complete: bool,
) -> VoiceState:
    _patch_catalogue(monkeypatch, complete=lambda: complete)
    settings = Settings(
        voice_cache=tmp_path,
        huggingface_cache=tmp_path,
        provider_root=tmp_path,
        PRESENTATOR_RUNTIME_UID=os.geteuid(),
    )
    return runtime.snapshot(settings).voices[2].state


def _patch_catalogue(
    monkeypatch: pytest.MonkeyPatch, complete: Callable[[], bool]
) -> None:
    monkeypatch.setattr(
        voices,
        "installed_voice_ids",
        lambda _settings: {VoiceId.QWEN} if complete() else set(),
    )
    monkeypatch.setattr(
        voices, "provider_entrypoint_is_usable", lambda _root, _provider: True
    )


def _capture_thread_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> list[BaseException]:
    failures: list[BaseException] = []
    monkeypatch.setattr(
        qwen_download.threading,
        "excepthook",
        lambda arguments: failures.append(arguments.exc_value),
    )
    return failures


def _record_daemon_threads(
    monkeypatch: pytest.MonkeyPatch,
) -> list[threading.Thread]:
    real_thread = threading.Thread
    workers: list[threading.Thread] = []

    def thread(*arguments: object, **keywords: object) -> threading.Thread:
        worker = real_thread(*arguments, **keywords)
        if keywords.get("daemon") is True:
            workers.append(worker)
        return worker

    monkeypatch.setattr(qwen_download.threading, "Thread", thread)
    return workers


def _join(worker: threading.Thread) -> None:
    worker.join(timeout=5)
    assert not worker.is_alive()


def test_qwen_download_uses_only_the_pinned_artifacts_without_a_token(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    requested: dict[str, object] = {}

    def snapshot_download(**arguments: object) -> None:
        requested.update(arguments)

    monkeypatch.setattr(qwen_download, "snapshot_download", snapshot_download)
    monkeypatch.setattr(
        qwen_download.voices, "qwen_worker_is_usable", lambda _settings: True
    )
    monkeypatch.setattr(
        qwen_download.voices,
        "installed_voice_ids",
        lambda _settings: {VoiceId.QWEN},
    )

    qwen_download.download_qwen(Settings(huggingface_cache=tmp_path))

    assert requested == {
        "repo_id": QWEN_MODEL_ID,
        "revision": QWEN_REVISION,
        "cache_dir": tmp_path,
        "allow_patterns": list(QWEN_ARTIFACTS),
        "token": False,
        "force_download": False,
    }


def test_failed_hub_transfer_retains_its_resumable_partial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    partial = tmp_path / "models--Qwen--Qwen3-TTS/incomplete/model.part"

    def snapshot_download(**_arguments: object) -> None:
        partial.parent.mkdir(parents=True)
        partial.write_bytes(b"resume me")
        message = "transfer failed"
        raise RuntimeError(message)

    monkeypatch.setattr(qwen_download, "snapshot_download", snapshot_download)
    monkeypatch.setattr(
        qwen_download.voices, "qwen_worker_is_usable", lambda _settings: True
    )

    with pytest.raises(RuntimeError, match="transfer failed"):
        qwen_download.download_qwen(Settings(huggingface_cache=tmp_path))

    assert partial.read_bytes() == b"resume me"


@pytest.mark.parametrize(
    ("fails", "terminal"),
    [(False, VoiceState.DOWNLOADED), (True, VoiceState.DOWNLOAD_FAILED)],
)
def test_private_download_can_be_discarded_and_revisited_through_truthful_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    fails,
    terminal: VoiceState,
) -> None:
    complete = False
    entered = threading.Event()
    release = threading.Event()
    sentinel = "secret-upstream-url-and-token"
    thread_failures = _capture_thread_failures(monkeypatch)

    def download() -> None:
        nonlocal complete
        entered.set()
        assert release.wait(5)
        complete = not fails
        if fails:
            raise RuntimeError(sentinel)

    runtime = _runtime(tmp_path, complete=lambda: complete, download=download)
    settings = Settings(
        voice_cache=tmp_path,
        huggingface_cache=tmp_path,
        provider_root=tmp_path,
        PRESENTATOR_RUNTIME_UID=os.geteuid(),
    )
    _patch_catalogue(monkeypatch, complete=lambda: complete)
    caplog.set_level(logging.ERROR)

    with TestClient(create_control_app(settings, runtime)) as client:
        workers = _record_daemon_threads(monkeypatch)
        response = client.post("/voices/qwen3-tts-0.6b/download")

        assert response.json() == {"outcome": VoiceDownloadOutcome.STARTED}
        assert entered.wait(5)
        assert client.get("/voices").json()["voices"][2]["state"] == "downloading"
        del response
        release.set()
        _join(workers[-1])
        body = client.get("/voices").json()

    assert body["voices"][2]["state"] == terminal
    assert sentinel not in str(body)
    assert sentinel not in caplog.text
    assert thread_failures == []
    if fails:
        assert "Qwen download failed" in caplog.text


def test_failed_download_releases_ownership_for_a_complete_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    complete = False
    attempts = 0

    def download() -> None:
        nonlocal attempts, complete
        attempts += 1
        if attempts == 1:
            message = "private failure"
            raise RuntimeError(message)
        complete = True

    runtime = _runtime(tmp_path, complete=lambda: complete, download=download)
    thread_failures = _capture_thread_failures(monkeypatch)
    _patch_catalogue(monkeypatch, complete=lambda: complete)
    settings = Settings(voice_cache=tmp_path, huggingface_cache=tmp_path)

    with TestClient(create_control_app(settings, runtime)) as client:
        workers = _record_daemon_threads(monkeypatch)
        assert client.post("/voices/qwen3-tts-0.6b/download").status_code == 200
        _join(workers[-1])
        assert client.get("/voices").json()["voices"][2]["state"] == "download_failed"
        assert client.post("/voices/qwen3-tts-0.6b/download").status_code == 200
        _join(workers[-1])
        terminal = client.get("/voices").json()["voices"][2]["state"]

    assert attempts == 2
    assert terminal == "downloaded"
    assert thread_failures == []


@pytest.mark.parametrize("worker_usable", [False, True])
def test_unavailable_or_already_complete_download_has_no_other_side_effect(
    tmp_path: Path, worker_usable
) -> None:
    complete = worker_usable
    downloads = 0
    constructed: list[VoiceId] = []
    speaking = _TrackedSpeaking()

    def download() -> None:
        nonlocal downloads
        downloads += 1

    runtime = _runtime(
        tmp_path,
        complete=lambda: complete,
        download=download,
        overrides=_RuntimeOverrides(
            worker_usable=lambda: worker_usable,
            speaking=speaking,
            factory=lambda voice: (constructed.append(voice), FakeSpeaking())[1],
        ),
    )

    outcome = runtime.start_qwen_download()

    assert outcome is (VoiceDownloadOutcome.ALREADY_COMPLETE if complete else None)
    assert downloads == 0
    assert constructed == []
    assert not (tmp_path / "state/selected-voice").exists()
    assert speaking.closed == 0
    admission = runtime.capture_speaking()
    assert admission is not None
    assert admission.speaking is speaking
    admission.close()


def test_worker_disappearance_before_hub_access_makes_no_network_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hub_calls = 0

    def snapshot_download(**_arguments: object) -> None:
        nonlocal hub_calls
        hub_calls += 1

    monkeypatch.setattr(
        qwen_download.voices, "qwen_worker_is_usable", lambda _settings: False
    )
    monkeypatch.setattr(qwen_download, "snapshot_download", snapshot_download)

    with pytest.raises(RuntimeError):
        qwen_download.download_qwen(Settings(huggingface_cache=tmp_path))

    assert hub_calls == 0


def test_exact_completion_wins_when_the_callback_raises_after_writing_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    complete = False
    thread_failures = _capture_thread_failures(monkeypatch)

    def download() -> None:
        nonlocal complete
        complete = True
        message = "must stay private"
        raise RuntimeError(message)

    runtime = _runtime(tmp_path, complete=lambda: complete, download=download)
    workers = _record_daemon_threads(monkeypatch)

    assert runtime.start_qwen_download() is VoiceDownloadOutcome.STARTED
    _join(workers[-1])

    assert (
        _qwen_state(runtime, tmp_path, monkeypatch, complete=True)
        is VoiceState.DOWNLOADED
    )
    assert thread_failures == []


@pytest.mark.parametrize(
    ("complete_after_refusal", "terminal"),
    [(False, VoiceState.DOWNLOAD_FAILED), (True, VoiceState.DOWNLOADED)],
)
def test_thread_start_failure_rechecks_artifacts_and_releases_ownership(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    complete_after_refusal,
    terminal: VoiceState,
) -> None:
    complete = False
    downloads = 0

    def download() -> None:
        nonlocal downloads
        downloads += 1

    runtime = _runtime(tmp_path, complete=lambda: complete, download=download)

    class RefusedThread:
        def __init__(self, **_arguments: object) -> None:
            pass

        def start(self) -> None:
            nonlocal complete
            complete = complete_after_refusal
            message = "cannot start"
            raise RuntimeError(message)

    with monkeypatch.context() as patch:
        patch.setattr(qwen_download.threading, "Thread", RefusedThread)
        assert runtime.start_qwen_download() is None

    assert (
        _qwen_state(
            runtime,
            tmp_path,
            monkeypatch,
            complete=complete_after_refusal,
        )
        is terminal
    )
    assert downloads == 0
    complete = True
    assert runtime.start_qwen_download() is VoiceDownloadOutcome.ALREADY_COMPLETE


def test_download_conflicts_only_with_duplicate_and_qwen_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    complete = False
    entered = threading.Event()
    release = threading.Event()

    def download() -> None:
        nonlocal complete
        entered.set()
        assert release.wait(5)
        complete = True

    runtime = _runtime(tmp_path, complete=lambda: complete, download=download)
    workers = _record_daemon_threads(monkeypatch)

    outcome = runtime.start_qwen_download()

    assert outcome is VoiceDownloadOutcome.STARTED
    assert entered.wait(5)
    assert runtime.start_qwen_download() is None
    assert runtime.load_voice(VoiceId.QWEN) is None
    release.set()
    _join(workers[-1])
    assert runtime.load_voice(VoiceId.QWEN) is VoiceLoadOutcome.ACTIVATED


def test_current_synthesis_and_other_voice_load_remain_usable_during_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    entered = threading.Event()
    release = threading.Event()
    speaking = _TrackedSpeaking()
    other = _TrackedSpeaking(ready=False)
    runtime = _runtime(
        tmp_path,
        complete=lambda: False,
        download=lambda: (entered.set(), release.wait(5)),
        overrides=_RuntimeOverrides(speaking=speaking, factory=lambda _voice: other),
    )
    workers = _record_daemon_threads(monkeypatch)

    assert runtime.start_qwen_download() is VoiceDownloadOutcome.STARTED
    assert entered.wait(5)
    admission = runtime.capture_speaking()
    assert admission is not None
    assert admission.speaking is speaking
    admission.close()
    assert runtime.load_voice(VoiceId.CHATTERBOX) is VoiceLoadOutcome.ACTIVATED
    assert speaking.closed == 0
    assert other.ready is True
    release.set()
    _join(workers[-1])


def test_saved_qwen_startup_does_not_take_an_active_download_guard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    entered = threading.Event()
    release = threading.Event()
    complete = False
    constructed: list[VoiceId] = []
    store = VoiceSelectionStore(tmp_path / "state", owner_uid=os.geteuid())
    store.write(VoiceId.QWEN)

    def download() -> None:
        nonlocal complete
        entered.set()
        assert release.wait(5)
        complete = True

    runtime = _runtime(
        tmp_path,
        complete=lambda: complete,
        download=download,
        overrides=_RuntimeOverrides(
            selection=store,
            factory=lambda voice: (constructed.append(voice), FakeSpeaking())[1],
        ),
    )
    workers = _record_daemon_threads(monkeypatch)
    assert runtime.start_qwen_download() is VoiceDownloadOutcome.STARTED
    assert entered.wait(5)

    try:
        runtime.load()
        constructed_after_startup = list(constructed)
        recovery = runtime.snapshot(Settings(voice_cache=tmp_path)).recovery
        duplicate = runtime.start_qwen_download()
    finally:
        release.set()
        for worker in workers:
            _join(worker)

    assert constructed_after_startup == []
    assert recovery is None
    assert duplicate is None


def test_saved_qwen_startup_activates_when_no_download_owns_it(tmp_path: Path) -> None:
    store = VoiceSelectionStore(tmp_path / "state", owner_uid=os.geteuid())
    store.write(VoiceId.QWEN)
    qwen = FakeSpeaking(ready=False)
    runtime = _runtime(
        tmp_path,
        complete=lambda: True,
        download=lambda: None,
        overrides=_RuntimeOverrides(selection=store, factory=lambda _voice: qwen),
    )

    runtime.load()

    assert qwen.ready is True
    assert store.read() is VoiceId.QWEN
    assert runtime.snapshot(Settings(voice_cache=tmp_path)).recovery is None
    admission = runtime.capture_speaking()
    assert admission is not None
    assert admission.speaking is qwen
    admission.close()


def test_shutdown_refuses_new_downloads_without_orphaning_the_running_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    complete = False
    entered = threading.Event()
    release = threading.Event()

    def download() -> None:
        nonlocal complete
        entered.set()
        assert release.wait(5)
        complete = True

    runtime = _runtime(tmp_path, complete=lambda: complete, download=download)
    workers = _record_daemon_threads(monkeypatch)
    assert runtime.start_qwen_download() is VoiceDownloadOutcome.STARTED
    assert entered.wait(5)

    runtime.begin_shutdown()
    runtime.close()

    assert runtime.start_qwen_download() is None
    release.set()
    _join(workers[-1])
    assert (
        _qwen_state(runtime, tmp_path, monkeypatch, complete=True)
        is VoiceState.DOWNLOADED
    )


def test_get_and_public_speech_cannot_start_the_download(tmp_path: Path) -> None:
    downloads = 0

    def download() -> None:
        nonlocal downloads
        downloads += 1

    runtime = _runtime(tmp_path, complete=lambda: False, download=download)
    settings = Settings(voice_cache=tmp_path, huggingface_cache=tmp_path)

    assert (
        TestClient(create_control_app(settings, runtime)).get("/voices").status_code
        == 200
    )
    public = TestClient(create_app(runtime=runtime))
    assert public.post("/voices/qwen3-tts-0.6b/download").status_code == 404
    assert downloads == 0


def test_download_refuses_a_missing_worker_before_starting_the_callback() -> None:
    called = False

    def download() -> None:
        nonlocal called
        called = True

    runtime = Runtime(
        None,
        FakeHearing(),
        dependencies=RuntimeDependencies(
            artifact_checker=lambda _voice: False,
            qwen_worker_is_usable=lambda: False,
            qwen_download=download,
        ),
    )

    assert runtime.start_qwen_download() is None
    assert not called

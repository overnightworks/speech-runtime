"""Switching the speaking voice keeps one engine live and releases the old one."""

import logging
import os
import threading
from typing import TYPE_CHECKING

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from speech_runtime_provider_contract import (
    QWEN_SAMPLE_RATE,
)
from starlette.status import (
    HTTP_409_CONFLICT,
    HTTP_503_SERVICE_UNAVAILABLE,
)

from speech_runtime_service.config import Settings
from speech_runtime_service.control import create_control_app
from speech_runtime_service.pcm import pcm_from_wav
from speech_runtime_service.selection import VoiceSelectionStore
from speech_runtime_service.service import (
    Runtime,
    RuntimeDependencies,
    SpeakingEngine,
)
from speech_runtime_service.voices import (
    VoiceId,
    VoiceLoadOutcome,
    VoiceRecoveryKind,
    VoiceState,
)
from tests.support.fakes_support import FakeHearing, FakeSpeaking
from tests.support.speak_support import (
    response_body,
    speak_response,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


def _captured_engine(runtime: Runtime) -> SpeakingEngine:
    admission = runtime.capture_speaking()
    assert admission is not None
    try:
        return admission.speaking
    finally:
        admission.close()


class _TrackedSpeaking(FakeSpeaking):
    def __init__(self, *, sample_rate: int = 22_050) -> None:
        super().__init__()
        self.sample_rate = sample_rate
        self.closed = 0

    def close(self) -> None:
        self.closed += 1


def _runtime_loading(
    selection: VoiceSelectionStore,
    engine_factory: Callable[[VoiceId], SpeakingEngine],
    artifact_checker: Callable[[VoiceId], bool] = lambda _voice: True,
) -> Runtime:
    return Runtime(
        None,
        FakeHearing(),
        dependencies=RuntimeDependencies(
            selection=selection,
            engine_factory=engine_factory,
            artifact_checker=artifact_checker,
        ),
    )


def _switching_runtime(
    tmp_path: Path,
    old: FakeSpeaking,
    target: FakeSpeaking,
    selection: VoiceSelectionStore | None = None,
) -> tuple[Runtime, VoiceSelectionStore]:
    old.retain_when_inactive = False
    store = selection or VoiceSelectionStore(tmp_path / "state", owner_uid=os.geteuid())
    store.write(VoiceId.CHATTERBOX)
    runtime = Runtime(
        old,
        FakeHearing(),
        dependencies=RuntimeDependencies(
            selection=store,
            engine_factory=lambda _voice: target,
            artifact_checker=lambda _voice: True,
            default_voice=VoiceId.CHATTERBOX,
        ),
    )
    return runtime, store


def test_deselecting_chatterbox_waits_for_an_admitted_talk_and_refuses_new_speech(
    tmp_path: Path,
) -> None:
    committed = threading.Event()

    class ObservedSelection(VoiceSelectionStore):
        def write(self, voice: VoiceId) -> None:
            super().write(voice)
            if voice is VoiceId.PIPER:
                committed.set()

    old_voice_rate = QWEN_SAMPLE_RATE
    old = _TrackedSpeaking(sample_rate=old_voice_rate)
    target = _TrackedSpeaking()
    runtime, _store = _switching_runtime(
        tmp_path,
        old,
        target,
        ObservedSelection(tmp_path / "state", owner_uid=os.geteuid()),
    )
    response = speak_response(runtime)
    outcomes = []
    switching = threading.Thread(
        target=lambda: outcomes.append(runtime.load_voice(VoiceId.PIPER))
    )
    switching.start()
    assert committed.wait(timeout=5)

    with pytest.raises(HTTPException) as refused:
        speak_response(runtime)
    assert refused.value.status_code == HTTP_503_SERVICE_UNAVAILABLE
    assert switching.is_alive()

    rate, pcm = pcm_from_wav(response_body(response))
    switching.join(timeout=5)
    assert not switching.is_alive()
    assert rate == old_voice_rate
    assert pcm
    assert outcomes == [VoiceLoadOutcome.ACTIVATED]
    assert old.closed == 1
    assert _captured_engine(runtime) is target


def test_shutdown_before_chatterbox_detach_transfers_old_and_pending_cleanup_once(
    tmp_path: Path,
) -> None:
    committed = threading.Event()

    class ObservedSelection(VoiceSelectionStore):
        def write(self, voice: VoiceId) -> None:
            super().write(voice)
            if voice is VoiceId.PIPER:
                committed.set()

    old = _TrackedSpeaking(sample_rate=24_000)
    target = _TrackedSpeaking()
    runtime, store = _switching_runtime(
        tmp_path,
        old,
        target,
        ObservedSelection(tmp_path / "state", owner_uid=os.geteuid()),
    )
    response = speak_response(runtime)
    outcomes = []
    switching = threading.Thread(
        target=lambda: outcomes.append(runtime.load_voice(VoiceId.PIPER))
    )
    switching.start()
    assert committed.wait(timeout=5)

    runtime.begin_shutdown()
    closing = threading.Thread(target=runtime.close)
    closing.start()
    response_body(response)
    switching.join(timeout=5)
    closing.join(timeout=5)

    assert not switching.is_alive()
    assert not closing.is_alive()
    assert store.read() is VoiceId.PIPER
    assert outcomes == [VoiceLoadOutcome.NOT_ACTIVATED]
    assert runtime.capture_speaking() is None
    assert old.closed == 1
    assert target.closed == 1


def test_shutdown_during_chatterbox_close_never_publishes_or_loses_an_engine(
    tmp_path: Path,
) -> None:
    close_started = threading.Event()
    release_close = threading.Event()

    class HeldCloseSpeaking(_TrackedSpeaking):
        def close(self) -> None:
            self.closed += 1
            close_started.set()
            assert release_close.wait(timeout=5)

    old = HeldCloseSpeaking(sample_rate=24_000)
    target = _TrackedSpeaking()
    runtime, store = _switching_runtime(tmp_path, old, target)
    outcomes = []
    switching = threading.Thread(
        target=lambda: outcomes.append(runtime.load_voice(VoiceId.PIPER))
    )
    switching.start()
    assert close_started.wait(timeout=5)

    runtime.begin_shutdown()
    closing = threading.Thread(target=runtime.close)
    closing.start()
    release_close.set()
    switching.join(timeout=5)
    closing.join(timeout=5)

    assert not switching.is_alive()
    assert not closing.is_alive()
    assert store.read() is VoiceId.PIPER
    assert outcomes == [VoiceLoadOutcome.NOT_ACTIVATED]
    assert runtime.capture_speaking() is None
    assert old.closed == 1
    assert target.closed == 1


def _close_failure_after_selection_commit_is_fatal_and_not_a_load_outcome(
    tmp_path: Path,
) -> None:
    class FailingCloseSpeaking(_TrackedSpeaking):
        def close(self) -> None:
            self.closed += 1
            if self.closed == 1:
                message = "provider close failed"
                raise RuntimeError(message)

    fatal = threading.Event()
    old = FailingCloseSpeaking(sample_rate=24_000)
    target = _TrackedSpeaking()
    close_attempts = 2
    runtime, store = _switching_runtime(tmp_path, old, target)
    runtime.set_fatal_callback(fatal.set)

    with pytest.raises(RuntimeError, match="provider close failed"):
        runtime.load_voice(VoiceId.PIPER)

    assert fatal.is_set()
    assert store.read() is VoiceId.PIPER
    assert runtime.capture_speaking() is None
    runtime.close()
    assert old.closed == close_attempts
    assert target.closed == 1


test_chatterbox_close_failure_after_selection_commit_is_fatal_and_not_a_load_outcome = (
    _close_failure_after_selection_commit_is_fatal_and_not_a_load_outcome
)


def test_pre_replace_failure_preserves_active_and_stored_chatterbox(
    tmp_path: Path,
) -> None:
    settings = Settings(
        voice_cache=tmp_path,
        huggingface_cache=tmp_path,
        PRESENTATOR_RUNTIME_UID=os.geteuid(),
    )
    old = _TrackedSpeaking(sample_rate=24_000)
    target = FakeSpeaking(ready=False, fail=True)
    runtime, store = _switching_runtime(tmp_path, old, target)

    assert runtime.load_voice(VoiceId.PIPER) is VoiceLoadOutcome.NOT_ACTIVATED

    assert _captured_engine(runtime) is old
    assert store.read() is VoiceId.CHATTERBOX
    assert runtime.snapshot(settings).voices[1].state is VoiceState.ACTIVE


def test_startup_selects_the_default_then_restores_a_saved_voice(
    tmp_path: Path,
) -> None:
    store = VoiceSelectionStore(tmp_path / "state", owner_uid=os.geteuid())
    engines = {VoiceId.PIPER: FakeSpeaking(), VoiceId.CHATTERBOX: FakeSpeaking()}

    def runtime() -> Runtime:
        return Runtime(
            None,
            FakeHearing(ready=False),
            dependencies=RuntimeDependencies(
                selection=store,
                engine_factory=engines.__getitem__,
                artifact_checker=lambda voice: voice in engines,
                default_voice=VoiceId.PIPER,
            ),
        )

    first = runtime()
    first.load()
    store.write(VoiceId.CHATTERBOX)
    restored = runtime()
    restored.load()

    assert _captured_engine(first) is engines[VoiceId.PIPER]
    assert _captured_engine(restored) is engines[VoiceId.CHATTERBOX]


def test_a_failed_load_preserves_the_active_engine_and_records_recovery(
    tmp_path: Path,
) -> None:
    active = FakeSpeaking()
    runtime = _runtime_loading(
        VoiceSelectionStore(tmp_path / "state", owner_uid=os.geteuid()),
        lambda voice: active if voice is VoiceId.PIPER else FakeSpeaking(fail=True),
    )
    assert runtime.load_voice(VoiceId.PIPER) is VoiceLoadOutcome.ACTIVATED

    assert runtime.load_voice(VoiceId.CHATTERBOX) is VoiceLoadOutcome.NOT_ACTIVATED

    assert _captured_engine(runtime) is active
    recovery = runtime.snapshot(Settings(PRESENTATOR_RUNTIME_UID=os.geteuid())).recovery
    assert recovery is not None
    assert recovery.kind is VoiceRecoveryKind.LOAD_FAILED
    assert recovery.voice is VoiceId.CHATTERBOX


def test_a_failed_voice_can_be_loaded_again(tmp_path: Path) -> None:
    ready = False

    def factory(_voice: VoiceId) -> FakeSpeaking:
        return FakeSpeaking(fail=not ready)

    runtime = _runtime_loading(
        VoiceSelectionStore(tmp_path / "state", owner_uid=os.geteuid()), factory
    )

    assert runtime.load_voice(VoiceId.CHATTERBOX) is VoiceLoadOutcome.NOT_ACTIVATED
    ready = True

    assert runtime.load_voice(VoiceId.CHATTERBOX) is VoiceLoadOutcome.ACTIVATED
    assert runtime.snapshot(Settings(voice_cache=tmp_path)).recovery is None


def test_directory_sync_failure_activates_the_visible_choice_with_a_notice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    selection = VoiceSelectionStore(tmp_path / "state", owner_uid=os.geteuid())
    engine = FakeSpeaking()
    monkeypatch.setattr(
        selection, "_sync_directory", lambda: (_ for _ in ()).throw(OSError())
    )
    runtime = _runtime_loading(selection, lambda _voice: engine)

    assert (
        runtime.load_voice(VoiceId.PIPER)
        is VoiceLoadOutcome.ACTIVATED_DURABILITY_UNCONFIRMED
    )
    snapshot = runtime.snapshot(Settings(voice_cache=tmp_path))

    assert _captured_engine(runtime) is engine
    assert snapshot.recovery is not None
    assert snapshot.recovery.kind is VoiceRecoveryKind.DURABILITY_UNCONFIRMED


def test_failed_voice_load_logs_the_voice_without_request_content(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    runtime = _runtime_loading(
        VoiceSelectionStore(tmp_path / "state", owner_uid=os.geteuid()),
        lambda _voice: FakeSpeaking(fail=True),
    )
    caplog.set_level(logging.ERROR)

    assert runtime.load_voice(VoiceId.CHATTERBOX) is VoiceLoadOutcome.NOT_ACTIVATED

    assert "voice chatterbox failed to load" in caplog.text
    assert "weights missing" in caplog.text


def test_artifact_check_failure_returns_a_logged_typed_outcome(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    runtime = _runtime_loading(
        VoiceSelectionStore(tmp_path / "state", owner_uid=os.geteuid()),
        lambda _voice: pytest.fail("constructed an engine"),
        lambda _voice: (_ for _ in ()).throw(OSError("cache")),
    )
    caplog.set_level(logging.ERROR)

    assert runtime.load_voice(VoiceId.CHATTERBOX) is VoiceLoadOutcome.NOT_ACTIVATED
    recovery = runtime.snapshot(Settings(voice_cache=tmp_path)).recovery

    assert recovery is not None
    assert recovery.kind is VoiceRecoveryKind.LOAD_FAILED
    assert "voice chatterbox failed to load" in caplog.text


def test_unsafe_state_store_refuses_before_constructing_an_engine(
    tmp_path: Path,
) -> None:
    directory = tmp_path / "state"
    directory.mkdir(mode=0o700)
    directory.chmod(0o755)
    constructed: list[VoiceId] = []

    def factory(voice: VoiceId) -> FakeSpeaking:
        constructed.append(voice)
        return FakeSpeaking()

    runtime = _runtime_loading(
        VoiceSelectionStore(directory, owner_uid=os.geteuid()), factory
    )

    assert runtime.load_voice(VoiceId.CHATTERBOX) is VoiceLoadOutcome.NOT_ACTIVATED
    assert constructed == []


def test_startup_selection_holds_load_guard_until_its_activation_finishes(
    tmp_path: Path,
) -> None:
    entered = threading.Event()
    release = threading.Event()

    class HeldSelection(VoiceSelectionStore):
        def read(self) -> VoiceId | None:
            entered.set()
            assert release.wait(timeout=5)
            return VoiceId.PIPER

    engines = {VoiceId.PIPER: FakeSpeaking(), VoiceId.CHATTERBOX: FakeSpeaking()}
    runtime = _runtime_loading(
        HeldSelection(tmp_path / "state", owner_uid=os.geteuid()),
        engines.__getitem__,
        lambda voice: voice in engines,
    )
    loader = threading.Thread(target=runtime.load)
    loader.start()
    assert entered.wait(timeout=5)

    response = TestClient(
        create_control_app(Settings(voice_cache=tmp_path), runtime)
    ).post("/voices/chatterbox/load")
    release.set()
    loader.join(timeout=5)

    assert response.status_code == HTTP_409_CONFLICT
    assert not loader.is_alive()
    assert _captured_engine(runtime) is engines[VoiceId.PIPER]
    assert runtime.load_voice(VoiceId.CHATTERBOX) is VoiceLoadOutcome.ACTIVATED
    assert _captured_engine(runtime) is engines[VoiceId.CHATTERBOX]
    assert runtime.snapshot(Settings(voice_cache=tmp_path)).recovery is None


def test_duplicate_load_is_refused_and_retained_chatterbox_is_not_rebuilt(
    tmp_path: Path,
) -> None:
    store = VoiceSelectionStore(tmp_path / "state", owner_uid=os.geteuid())
    engines: list[FakeSpeaking] = []
    entered = threading.Event()
    release = threading.Event()

    class HeldSpeaking(FakeSpeaking):
        def load(self) -> None:
            entered.set()
            assert release.wait(timeout=5)
            self.ready = True

    def factory(_voice: VoiceId) -> FakeSpeaking:
        engine = HeldSpeaking(ready=False)
        engines.append(engine)
        return engine

    runtime = _runtime_loading(store, factory)
    first = threading.Thread(target=runtime.load_voice, args=(VoiceId.CHATTERBOX,))
    first.start()
    assert entered.wait(timeout=5)
    response = TestClient(
        create_control_app(Settings(voice_cache=tmp_path), runtime)
    ).post("/voices/chatterbox/load")
    release.set()
    first.join(timeout=5)

    assert response.status_code == HTTP_409_CONFLICT
    assert runtime.load_voice(VoiceId.CHATTERBOX) is VoiceLoadOutcome.ACTIVATED
    assert runtime.load_voice(VoiceId.CHATTERBOX) is VoiceLoadOutcome.ACTIVATED
    assert len(engines) == 1


@pytest.mark.parametrize("stored", ["unknown", "chatterbox"])
def test_recoverable_startup_keeps_hearing_live_and_refuses_speaking(
    tmp_path: Path, stored: str
) -> None:
    store = VoiceSelectionStore(tmp_path / "state", owner_uid=os.geteuid())
    if stored == "unknown":
        store.path.parent.mkdir(mode=0o700)
        store.path.write_text(stored, encoding="utf-8")
        store.path.chmod(0o600)
    else:
        store.write(VoiceId.CHATTERBOX)
    hearing = FakeHearing(ready=False)
    runtime = Runtime(
        None,
        hearing,
        dependencies=RuntimeDependencies(
            selection=store,
            engine_factory=lambda _voice: FakeSpeaking(fail=True),
            artifact_checker=lambda _voice: True,
        ),
    )

    runtime.load()
    private = TestClient(create_control_app(Settings(voice_cache=tmp_path), runtime))

    assert private.get("/health").json()["speaking"] == {
        "model": "unavailable",
        "ready": False,
        "streams": False,
        "sample_rate": 0,
    }
    assert (
        private.post("/speak", json={"text": "Hallo", "language": "de"}).status_code
        == HTTP_503_SERVICE_UNAVAILABLE
    )
    assert hearing.ready is True
    assert private.get("/voices").json()["recovery"] is not None


def test_shutdown_winning_before_pending_registration_never_loads_or_publishes(
    tmp_path: Path,
) -> None:
    """A side-effect-free factory result is closed if shutdown wins its handoff."""
    constructed = threading.Event()
    release = threading.Event()

    class DeferredSpeaking(FakeSpeaking):
        def __init__(self) -> None:
            super().__init__(ready=False)
            self.loaded = False
            self.closed = 0

        def load(self) -> None:
            self.loaded = True

        def close(self) -> None:
            self.closed += 1

    engine = DeferredSpeaking()

    def factory(_voice: VoiceId) -> DeferredSpeaking:
        constructed.set()
        assert release.wait(timeout=5)
        return engine

    runtime = _runtime_loading(
        VoiceSelectionStore(tmp_path / "state", owner_uid=os.geteuid()), factory
    )
    activation = threading.Thread(target=runtime.load_voice, args=(VoiceId.CHATTERBOX,))
    activation.start()
    assert constructed.wait(timeout=5)

    runtime.begin_shutdown()
    runtime.close()
    release.set()
    activation.join(timeout=5)

    assert not activation.is_alive()
    assert engine.loaded is False
    assert engine.closed == 1
    assert runtime.capture_speaking() is None


@pytest.mark.parametrize("surface", ["health", "snapshot", "capture", "sample"])
def test_every_speaking_surface_reconciles_a_dead_active_without_fallback(
    tmp_path: Path, surface: str
) -> None:
    class ClosableSpeaking(FakeSpeaking):
        def __init__(self) -> None:
            super().__init__()
            self.closed = 0

        def close(self) -> None:
            self.closed += 1

    engine = ClosableSpeaking()
    engine.retain_when_inactive = False
    store = VoiceSelectionStore(tmp_path / "state", owner_uid=os.geteuid())
    runtime = _runtime_loading(store, lambda _voice: engine)
    assert runtime.load_voice(VoiceId.CHATTERBOX) is VoiceLoadOutcome.ACTIVATED
    engine.ready = False
    settings = Settings(voice_cache=tmp_path, huggingface_cache=tmp_path)

    if surface == "health":
        runtime.health()
    elif surface == "snapshot":
        runtime.snapshot(settings)
    elif surface == "capture":
        runtime.capture_speaking()
    else:
        runtime.preacquire_sample(VoiceId.CHATTERBOX)

    snapshot = runtime.snapshot(settings)
    assert runtime.capture_speaking() is None
    assert store.read() is VoiceId.CHATTERBOX
    assert snapshot.recovery is not None
    assert snapshot.recovery.kind is VoiceRecoveryKind.LOAD_FAILED
    assert snapshot.recovery.voice is VoiceId.CHATTERBOX
    assert engine.closed == 1


def test_nonstreaming_provider_failure_reconciles_and_allows_a_fresh_load(
    tmp_path: Path,
) -> None:
    class ClosableSpeaking(FakeSpeaking):
        retain_when_inactive = False

        def __init__(self) -> None:
            super().__init__()
            self.closed = 0

        def close(self) -> None:
            self.closed += 1

    failed = ClosableSpeaking()
    replacement = ClosableSpeaking()
    engines = iter((failed, replacement))
    store = VoiceSelectionStore(tmp_path / "state", owner_uid=os.geteuid())
    runtime = _runtime_loading(store, lambda _voice: next(engines))
    assert runtime.load_voice(VoiceId.QWEN) is VoiceLoadOutcome.ACTIVATED
    failed.ready = False

    snapshot = runtime.snapshot(Settings(voice_cache=tmp_path))

    assert failed.closed == 1
    assert snapshot.recovery is not None
    assert snapshot.recovery.kind is VoiceRecoveryKind.LOAD_FAILED
    assert snapshot.recovery.voice is VoiceId.QWEN
    assert store.read() is VoiceId.QWEN
    assert runtime.capture_speaking() is None
    assert runtime.load_voice(VoiceId.QWEN) is VoiceLoadOutcome.ACTIVATED
    assert _captured_engine(runtime) is replacement


def test_switching_releases_any_provider_process_before_load_returns(
    tmp_path: Path,
) -> None:
    old = _TrackedSpeaking(sample_rate=24_000)
    old.retain_when_inactive = False
    target = _TrackedSpeaking()
    store = VoiceSelectionStore(tmp_path / "state", owner_uid=os.geteuid())
    store.write(VoiceId.QWEN)
    runtime = Runtime(
        old,
        FakeHearing(),
        dependencies=RuntimeDependencies(
            selection=store,
            engine_factory=lambda _voice: target,
            artifact_checker=lambda _voice: True,
            default_voice=VoiceId.QWEN,
        ),
    )

    assert runtime.load_voice(VoiceId.PIPER) is VoiceLoadOutcome.ACTIVATED

    assert old.closed == 1
    assert _captured_engine(runtime) is target


def test_failed_active_reconciliation_excludes_retry_until_close_finishes(
    tmp_path: Path,
) -> None:
    close_started = threading.Event()
    release_close = threading.Event()

    class HeldCloseSpeaking(FakeSpeaking):
        def close(self) -> None:
            close_started.set()
            assert release_close.wait(timeout=5)

    failed = HeldCloseSpeaking()
    failed.retain_when_inactive = False
    replacement = FakeSpeaking()
    created = iter((failed, replacement))
    runtime = _runtime_loading(
        VoiceSelectionStore(tmp_path / "state", owner_uid=os.geteuid()),
        lambda _voice: next(created),
    )
    assert runtime.load_voice(VoiceId.CHATTERBOX) is VoiceLoadOutcome.ACTIVATED
    failed.ready = False
    reconciliation = threading.Thread(
        target=runtime.snapshot, args=(Settings(voice_cache=tmp_path),)
    )
    reconciliation.start()
    assert close_started.wait(timeout=5)

    assert runtime.load_voice(VoiceId.CHATTERBOX) is None
    release_close.set()
    reconciliation.join(timeout=5)
    assert not reconciliation.is_alive()
    assert runtime.load_voice(VoiceId.CHATTERBOX) is VoiceLoadOutcome.ACTIVATED
    assert _captured_engine(runtime) is replacement

"""The private voice endpoint reports evidence, never package support."""

import logging
import os
import threading

import pytest
from fastapi.testclient import TestClient

from speech import voices
from speech.config import Settings
from speech.selection import VoiceSelectionStore
from speech.service import Runtime, RuntimeDependencies, create_app, create_control_app
from speech.voices import (
    VoiceId,
    VoiceLoadOutcome,
    VoiceRecoveryKind,
    VoiceRuntimeState,
    VoiceState,
    statuses,
)
from tests.conftest import FakeHearing, FakeSpeaking


def test_status_requires_both_piper_artifacts_and_keeps_unimplemented_rows_unavailable(
    tmp_path,
) -> None:
    settings = Settings(
        voice_cache=tmp_path,
        huggingface_cache=tmp_path,
        PRESENTATOR_RUNTIME_UID=os.geteuid(),
    )
    (tmp_path / "de_DE-thorsten-medium.onnx").touch()

    missing_metadata = statuses(
        settings,
        VoiceRuntimeState(
            selected=None, ready=False, loading=False, pending=None, failed=None
        ),
    )

    (tmp_path / "de_DE-thorsten-medium.onnx.json").touch()
    complete = statuses(
        settings,
        VoiceRuntimeState(
            selected=None, ready=False, loading=False, pending=None, failed=None
        ),
    )

    assert missing_metadata[0].state is VoiceState.NOT_DOWNLOADED
    assert complete[0].state is VoiceState.DOWNLOADED
    assert [voice.id for voice in complete] == list(VoiceId)
    assert all(voice.state is VoiceState.UNAVAILABLE for voice in complete[-3:])


def test_chatterbox_weights_need_the_exact_executable_but_active_survives_its_removal(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(voices, "try_to_load_from_cache", lambda *_args, **_kwargs: "x")
    provider_root = tmp_path / "providers"
    settings = Settings(
        provider_root=provider_root,
        huggingface_cache=tmp_path,
        PRESENTATOR_RUNTIME_UID=os.geteuid(),
    )
    entrypoint = provider_root / "chatterbox/.venv/bin/presentator-chatterbox-worker"
    entrypoint.parent.mkdir(parents=True)

    absent = statuses(
        settings,
        VoiceRuntimeState(
            selected=None, ready=False, loading=False, pending=None, failed=None
        ),
    )
    entrypoint.write_text("#!/bin/false\n", encoding="utf-8")
    entrypoint.chmod(0o700)
    downloaded = statuses(
        settings,
        VoiceRuntimeState(
            selected=None, ready=False, loading=False, pending=None, failed=None
        ),
    )
    entrypoint.unlink()
    active = statuses(
        settings,
        VoiceRuntimeState(
            selected=VoiceId.CHATTERBOX,
            ready=True,
            loading=False,
            pending=None,
            failed=None,
        ),
    )

    assert absent[1].state is VoiceState.UNAVAILABLE
    assert downloaded[1].state is VoiceState.DOWNLOADED
    assert active[1].state is VoiceState.ACTIVE


def test_private_endpoint_reports_loading_and_active_from_the_shared_runtime(
    tmp_path,
) -> None:
    settings = Settings(
        voice_cache=tmp_path,
        huggingface_cache=tmp_path,
        PRESENTATOR_RUNTIME_UID=os.geteuid(),
    )
    runtime = Runtime(FakeSpeaking(ready=False), FakeHearing())
    runtime.loading = True
    client = TestClient(create_control_app(settings, runtime))

    loading = client.get("/voices").json()["voices"]
    runtime.loading = False
    runtime.speaking.ready = True
    active = client.get("/voices").json()["voices"]

    assert loading[0]["state"] == VoiceState.LOADING
    assert active[0]["state"] == VoiceState.ACTIVE


def test_public_tcp_application_has_no_voice_control_route() -> None:
    app = create_app(runtime=Runtime(FakeSpeaking(), FakeHearing()))

    assert TestClient(app).get("/voices").status_code == 404


def test_status_refuses_the_whole_snapshot_when_cache_lookup_fails(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = Settings(
        speaking_model="ResembleAI/chatterbox",
        voice_cache=tmp_path,
        huggingface_cache=tmp_path,
        PRESENTATOR_RUNTIME_UID=os.geteuid(),
    )

    def cache_unavailable(*_arguments: object, **_keywords: object) -> object:
        message = "cache unavailable"
        raise OSError(message)

    monkeypatch.setattr(voices, "try_to_load_from_cache", cache_unavailable)

    with pytest.raises(OSError, match="cache unavailable"):
        statuses(
            settings,
            VoiceRuntimeState(
                selected=None, ready=False, loading=False, pending=None, failed=None
            ),
        )


def test_loading_switches_the_next_capture_but_keeps_the_prior_capture_alive(
    tmp_path,
) -> None:
    settings = Settings(
        voice_cache=tmp_path,
        huggingface_cache=tmp_path,
        PRESENTATOR_RUNTIME_UID=os.geteuid(),
    )
    engines = {VoiceId.PIPER: FakeSpeaking(), VoiceId.CHATTERBOX: FakeSpeaking()}
    runtime = Runtime(
        None,
        FakeHearing(),
        dependencies=RuntimeDependencies(
            selection=VoiceSelectionStore(tmp_path / "state", owner_uid=os.geteuid()),
            engine_factory=engines.__getitem__,
            artifact_checker=lambda voice: voice in engines,
        ),
    )

    assert runtime.load_voice(VoiceId.PIPER) is VoiceLoadOutcome.ACTIVATED
    captured = runtime.capture_speaking()
    assert runtime.load_voice(VoiceId.CHATTERBOX) is VoiceLoadOutcome.ACTIVATED

    assert captured is engines[VoiceId.PIPER]
    assert runtime.capture_speaking() is engines[VoiceId.CHATTERBOX]
    assert (tmp_path / "state" / "selected-voice").read_text() == "chatterbox"
    assert runtime.snapshot(settings).voices[1].state is VoiceState.ACTIVE


def test_startup_selects_the_default_then_restores_a_saved_voice(tmp_path) -> None:
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

    assert first.capture_speaking() is engines[VoiceId.PIPER]
    assert restored.capture_speaking() is engines[VoiceId.CHATTERBOX]


def test_a_failed_load_preserves_the_active_engine_and_records_recovery(
    tmp_path,
) -> None:
    active = FakeSpeaking()
    runtime = Runtime(
        None,
        FakeHearing(),
        dependencies=RuntimeDependencies(
            selection=VoiceSelectionStore(tmp_path / "state", owner_uid=os.geteuid()),
            engine_factory=lambda voice: (
                active if voice is VoiceId.PIPER else FakeSpeaking(fail=True)
            ),
            artifact_checker=lambda _voice: True,
        ),
    )
    assert runtime.load_voice(VoiceId.PIPER) is VoiceLoadOutcome.ACTIVATED

    assert runtime.load_voice(VoiceId.CHATTERBOX) is VoiceLoadOutcome.NOT_ACTIVATED

    assert runtime.capture_speaking() is active
    recovery = runtime.snapshot(Settings(PRESENTATOR_RUNTIME_UID=os.geteuid())).recovery
    assert recovery is not None
    assert recovery.kind is VoiceRecoveryKind.LOAD_FAILED
    assert recovery.voice is VoiceId.CHATTERBOX


def test_a_failed_voice_can_be_loaded_again(tmp_path) -> None:
    ready = False

    def factory(_voice: VoiceId) -> FakeSpeaking:
        return FakeSpeaking(fail=not ready)

    runtime = Runtime(
        None,
        FakeHearing(),
        dependencies=RuntimeDependencies(
            selection=VoiceSelectionStore(tmp_path / "state", owner_uid=os.geteuid()),
            engine_factory=factory,
            artifact_checker=lambda _voice: True,
        ),
    )

    assert runtime.load_voice(VoiceId.CHATTERBOX) is VoiceLoadOutcome.NOT_ACTIVATED
    ready = True

    assert runtime.load_voice(VoiceId.CHATTERBOX) is VoiceLoadOutcome.ACTIVATED
    assert runtime.snapshot(Settings(voice_cache=tmp_path)).recovery is None


def test_directory_sync_failure_activates_the_visible_choice_with_a_notice(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    selection = VoiceSelectionStore(tmp_path / "state", owner_uid=os.geteuid())
    engine = FakeSpeaking()
    monkeypatch.setattr(
        selection, "_sync_directory", lambda: (_ for _ in ()).throw(OSError())
    )
    runtime = Runtime(
        None,
        FakeHearing(),
        dependencies=RuntimeDependencies(
            selection=selection,
            engine_factory=lambda _voice: engine,
            artifact_checker=lambda _voice: True,
        ),
    )

    assert (
        runtime.load_voice(VoiceId.PIPER)
        is VoiceLoadOutcome.ACTIVATED_DURABILITY_UNCONFIRMED
    )
    snapshot = runtime.snapshot(Settings(voice_cache=tmp_path))

    assert runtime.capture_speaking() is engine
    assert snapshot.recovery is not None
    assert snapshot.recovery.kind is VoiceRecoveryKind.DURABILITY_UNCONFIRMED


def test_failed_voice_load_logs_the_voice_without_request_content(
    tmp_path, caplog: pytest.LogCaptureFixture
) -> None:
    runtime = Runtime(
        None,
        FakeHearing(),
        dependencies=RuntimeDependencies(
            selection=VoiceSelectionStore(tmp_path / "state", owner_uid=os.geteuid()),
            engine_factory=lambda _voice: FakeSpeaking(fail=True),
            artifact_checker=lambda _voice: True,
        ),
    )
    caplog.set_level(logging.ERROR)

    assert runtime.load_voice(VoiceId.CHATTERBOX) is VoiceLoadOutcome.NOT_ACTIVATED

    assert "voice chatterbox failed to load" in caplog.text
    assert "weights missing" in caplog.text


def test_artifact_check_failure_returns_a_logged_typed_outcome(
    tmp_path, caplog: pytest.LogCaptureFixture
) -> None:
    runtime = Runtime(
        None,
        FakeHearing(),
        dependencies=RuntimeDependencies(
            selection=VoiceSelectionStore(tmp_path / "state", owner_uid=os.geteuid()),
            engine_factory=lambda _voice: pytest.fail("constructed an engine"),
            artifact_checker=lambda _voice: (_ for _ in ()).throw(OSError("cache")),
        ),
    )
    caplog.set_level(logging.ERROR)

    assert runtime.load_voice(VoiceId.CHATTERBOX) is VoiceLoadOutcome.NOT_ACTIVATED
    recovery = runtime.snapshot(Settings(voice_cache=tmp_path)).recovery

    assert recovery is not None
    assert recovery.kind is VoiceRecoveryKind.LOAD_FAILED
    assert "voice chatterbox failed to load" in caplog.text


def test_unsafe_state_store_refuses_before_constructing_an_engine(tmp_path) -> None:
    directory = tmp_path / "state"
    directory.mkdir(mode=0o700)
    directory.chmod(0o755)
    constructed: list[VoiceId] = []

    def factory(voice: VoiceId) -> FakeSpeaking:
        constructed.append(voice)
        return FakeSpeaking()

    runtime = Runtime(
        None,
        FakeHearing(),
        dependencies=RuntimeDependencies(
            selection=VoiceSelectionStore(directory, owner_uid=os.geteuid()),
            engine_factory=factory,
            artifact_checker=lambda _voice: True,
        ),
    )

    assert runtime.load_voice(VoiceId.CHATTERBOX) is VoiceLoadOutcome.NOT_ACTIVATED
    assert constructed == []


def test_startup_selection_holds_load_guard_until_its_activation_finishes(
    tmp_path,
) -> None:
    entered = threading.Event()
    release = threading.Event()

    class HeldSelection(VoiceSelectionStore):
        def read(self) -> VoiceId | None:
            entered.set()
            assert release.wait(timeout=5)
            return VoiceId.PIPER

    engines = {VoiceId.PIPER: FakeSpeaking(), VoiceId.CHATTERBOX: FakeSpeaking()}
    runtime = Runtime(
        None,
        FakeHearing(),
        dependencies=RuntimeDependencies(
            selection=HeldSelection(tmp_path / "state", owner_uid=os.geteuid()),
            engine_factory=engines.__getitem__,
            artifact_checker=lambda voice: voice in engines,
        ),
    )
    loader = threading.Thread(target=runtime.load)
    loader.start()
    assert entered.wait(timeout=5)

    response = TestClient(
        create_control_app(Settings(voice_cache=tmp_path), runtime)
    ).post("/voices/chatterbox/load")
    release.set()
    loader.join(timeout=5)

    assert response.status_code == 409
    assert not loader.is_alive()
    assert runtime.capture_speaking() is engines[VoiceId.PIPER]
    assert runtime.load_voice(VoiceId.CHATTERBOX) is VoiceLoadOutcome.ACTIVATED
    assert runtime.capture_speaking() is engines[VoiceId.CHATTERBOX]
    assert runtime.snapshot(Settings(voice_cache=tmp_path)).recovery is None


def test_duplicate_load_is_refused_and_retained_chatterbox_is_not_rebuilt(
    tmp_path,
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

    runtime = Runtime(
        None,
        FakeHearing(),
        dependencies=RuntimeDependencies(
            selection=store,
            engine_factory=factory,
            artifact_checker=lambda _voice: True,
        ),
    )
    first = threading.Thread(target=runtime.load_voice, args=(VoiceId.CHATTERBOX,))
    first.start()
    assert entered.wait(timeout=5)
    response = TestClient(
        create_control_app(Settings(voice_cache=tmp_path), runtime)
    ).post("/voices/chatterbox/load")
    release.set()
    first.join(timeout=5)

    assert response.status_code == 409
    assert runtime.load_voice(VoiceId.CHATTERBOX) is VoiceLoadOutcome.ACTIVATED
    assert runtime.load_voice(VoiceId.CHATTERBOX) is VoiceLoadOutcome.ACTIVATED
    assert len(engines) == 1


@pytest.mark.parametrize("stored", ["unknown", "chatterbox"])
def test_recoverable_startup_keeps_hearing_live_and_refuses_speaking(
    tmp_path, stored: str
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
    public = TestClient(create_app(runtime=runtime))
    private = TestClient(create_control_app(Settings(voice_cache=tmp_path), runtime))

    assert public.get("/health").json()["speaking"] == {
        "model": "unavailable",
        "ready": False,
        "streams": False,
        "sample_rate": 0,
    }
    assert (
        public.post("/speak", json={"text": "Hallo", "language": "de"}).status_code
        == 503
    )
    assert hearing.ready is True
    assert private.get("/voices").json()["recovery"] is not None


def test_shutdown_winning_before_pending_registration_never_loads_or_publishes(
    tmp_path,
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

    runtime = Runtime(
        None,
        FakeHearing(),
        dependencies=RuntimeDependencies(
            selection=VoiceSelectionStore(tmp_path / "state", owner_uid=os.geteuid()),
            engine_factory=factory,
            artifact_checker=lambda _voice: True,
        ),
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
    tmp_path, surface: str
) -> None:
    class ClosableSpeaking(FakeSpeaking):
        def __init__(self) -> None:
            super().__init__()
            self.closed = 0

        def close(self) -> None:
            self.closed += 1

    engine = ClosableSpeaking()
    engine.streams = True
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


def test_failed_active_reconciliation_excludes_retry_until_close_finishes(
    tmp_path,
) -> None:
    close_started = threading.Event()
    release_close = threading.Event()

    class HeldCloseSpeaking(FakeSpeaking):
        def close(self) -> None:
            close_started.set()
            assert release_close.wait(timeout=5)

    failed = HeldCloseSpeaking()
    failed.streams = True
    replacement = FakeSpeaking()
    created = iter((failed, replacement))
    runtime = Runtime(
        None,
        FakeHearing(),
        dependencies=RuntimeDependencies(
            selection=VoiceSelectionStore(tmp_path / "state", owner_uid=os.geteuid()),
            engine_factory=lambda _voice: next(created),
            artifact_checker=lambda _voice: True,
        ),
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
    assert runtime.capture_speaking() is replacement

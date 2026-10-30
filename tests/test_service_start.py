"""The runtime starts the configured voice or fails loudly, and shutdown wins races."""

import threading

import pytest
from fastapi.testclient import TestClient
from starlette.status import HTTP_503_SERVICE_UNAVAILABLE

from speech_runtime_service.service import Runtime, RuntimeDependencies, SpeakingEngine
from speech_runtime_service.voices import VoiceId
from tests.support.fakes_support import FakeSpeaking
from tests.support.speak_support import app_over, post_speech, speak_body

WAIT_SECONDS = 5


class _ClosableSpeaking(FakeSpeaking):
    def __init__(self, *, ready: bool = True, fail: bool = False) -> None:
        super().__init__(ready=ready, fail=fail)
        self.closed = 0

    def close(self) -> None:
        self.closed += 1


def _loading(
    engine: SpeakingEngine, *, installed: bool = True, voice: VoiceId = VoiceId.PIPER
) -> Runtime:
    return Runtime(
        None,
        dependencies=RuntimeDependencies(
            voice=voice,
            engine_factory=lambda _voice: engine,
            artifact_checker=lambda _voice: installed,
        ),
    )


def test_a_voice_whose_artifacts_are_missing_stops_the_start_before_building_it() -> (
    None
):
    built: list[VoiceId] = []

    def build(voice: VoiceId) -> SpeakingEngine:
        built.append(voice)
        return FakeSpeaking()

    runtime = Runtime(
        None,
        dependencies=RuntimeDependencies(
            voice=VoiceId.QWEN,
            engine_factory=build,
            artifact_checker=lambda _voice: False,
        ),
    )

    with pytest.raises(RuntimeError, match=VoiceId.QWEN.value):
        runtime.load()

    assert built == []
    assert runtime.loading is False
    assert runtime.capture_speaking() is None


def test_a_runtime_without_an_engine_factory_refuses_to_start() -> None:
    runtime = Runtime(None)

    with pytest.raises(RuntimeError, match="no engine factory"):
        runtime.load()


def test_a_voice_that_fails_to_load_names_its_model_and_is_closed() -> None:
    engine = _ClosableSpeaking(fail=True)
    runtime = _loading(engine)

    with pytest.raises(RuntimeError, match=f"{engine.model_name} failed to load"):
        runtime.load()

    assert engine.closed == 1
    assert runtime.loading is False
    assert runtime.capture_speaking() is None


def test_an_engine_that_cannot_be_closed_after_a_failed_load_ends_the_service() -> None:
    fatal = threading.Event()

    class StuckSpeaking(_ClosableSpeaking):
        def close(self) -> None:
            raise OSError

    runtime = _loading(StuckSpeaking(fail=True))
    runtime.set_fatal_callback(fatal.set)

    with pytest.raises(RuntimeError, match="failed to load"):
        runtime.load()

    assert fatal.is_set()


def test_a_started_voice_is_captured_for_its_configured_voice() -> None:
    engine = _ClosableSpeaking(ready=False)
    runtime = _loading(engine, voice=VoiceId.CHATTERBOX)

    runtime.load()
    admission = runtime.capture_speaking()

    assert admission is not None
    assert admission.speaking is engine
    assert admission.voice is VoiceId.CHATTERBOX
    assert admission.sample_rate == engine.sample_rate


def test_shutdown_winning_before_pending_registration_never_loads_or_publishes() -> (
    None
):
    constructed = threading.Event()
    release = threading.Event()

    class DeferredSpeaking(_ClosableSpeaking):
        loaded = False

        def load(self) -> None:
            self.loaded = True

    engine = DeferredSpeaking(ready=False)

    def build(_voice: VoiceId) -> DeferredSpeaking:
        constructed.set()
        assert release.wait(timeout=WAIT_SECONDS)
        return engine

    runtime = Runtime(
        None,
        dependencies=RuntimeDependencies(
            engine_factory=build, artifact_checker=lambda _voice: True
        ),
    )
    starting = threading.Thread(target=runtime.load)
    starting.start()
    assert constructed.wait(timeout=WAIT_SECONDS)

    runtime.begin_shutdown()
    runtime.close()
    release.set()
    starting.join(timeout=WAIT_SECONDS)

    assert not starting.is_alive()
    assert engine.loaded is False
    assert engine.closed == 1
    assert runtime.capture_speaking() is None


def test_shutdown_during_the_load_closes_the_engine_once_and_publishes_nothing() -> (
    None
):
    loading = threading.Event()
    release = threading.Event()

    class SlowSpeaking(_ClosableSpeaking):
        def load(self) -> None:
            loading.set()
            assert release.wait(timeout=WAIT_SECONDS)
            self.ready = True

    engine = SlowSpeaking(ready=False)
    runtime = _loading(engine)
    starting = threading.Thread(target=runtime.load)
    starting.start()
    assert loading.wait(timeout=WAIT_SECONDS)

    runtime.begin_shutdown()
    release.set()
    starting.join(timeout=WAIT_SECONDS)
    runtime.close()

    assert not starting.is_alive()
    assert engine.closed == 1
    assert runtime.capture_speaking() is None


@pytest.mark.parametrize("surface", ["health", "wire"])
def test_a_voice_that_stops_being_ready_is_closed_once_and_not_replaced(
    surface: str,
) -> None:
    engine = _ClosableSpeaking()
    runtime = _loading(engine)
    runtime.load()
    engine.ready = False

    if surface == "health":
        runtime.health()
    else:
        refused = post_speech(TestClient(app_over(runtime)), speak_body())
        assert refused.status_code == HTTP_503_SERVICE_UNAVAILABLE

    assert runtime.capture_speaking() is None
    assert engine.closed == 1

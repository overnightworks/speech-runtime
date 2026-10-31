"""Hearing starts beside speaking; a missing resource isolates it."""

from typing import TYPE_CHECKING

import pytest
from fastapi.testclient import TestClient
from starlette.status import HTTP_200_OK, HTTP_503_SERVICE_UNAVAILABLE

from speech_runtime_client.wire import RefusalCode
from speech_runtime_service import hearing_models
from speech_runtime_service.config import Settings
from speech_runtime_service.hearing import Hearing, ResourceUnavailableError
from speech_runtime_service.hearing_models import (
    HearingModel,
    HearingModelMissingError,
)
from speech_runtime_service.service import Runtime, RuntimeDependencies
from speech_runtime_service.voices import VoiceId
from tests.support.fakes_support import FakeSpeaking
from tests.support.hear_support import (
    FakeHearing,
    hear_app,
    multipart_body,
    post_recording,
    recording,
    write_snapshot,
)
from tests.support.speak_support import assert_refusal, post_speech, speak_body

if TYPE_CHECKING:
    from pathlib import Path


def _starting(hearing: FakeHearing | None) -> Runtime:
    return Runtime(
        None,
        memory_probe=lambda: 0,
        dependencies=RuntimeDependencies(
            voice=VoiceId.PIPER,
            engine_factory=lambda _voice: FakeSpeaking(),
            artifact_checker=lambda _voice: True,
        ),
        hearing=Hearing(hearing),
    )


def test_a_load_that_lacks_the_resource_leaves_speaking_up_hearing_unavailable() -> (
    None
):
    engine = FakeHearing(load_error=ResourceUnavailableError("the card is full"))
    runtime = _starting(engine)

    runtime.load()
    client = TestClient(hear_app(runtime))

    health = client.get("/health").json()
    transcription = post_recording(client, multipart_body(file=recording()))
    speech = post_speech(client, speak_body(VoiceId.PIPER.value))
    assert health["hearing"]["ready"] is False
    assert health["speaking"]["ready"] is True
    assert_refusal(
        transcription, HTTP_503_SERVICE_UNAVAILABLE, RefusalCode.UNAVAILABLE, None
    )
    assert speech.status_code == HTTP_200_OK
    assert engine.closed == 1


def test_a_load_that_fails_otherwise_stops_the_start_naming_the_setting() -> None:
    engine = FakeHearing(load_error=RuntimeError("weights are corrupt"))
    runtime = _starting(engine)

    with pytest.raises(RuntimeError, match="SPEECH_HEARING_MODEL"):
        runtime.load()

    assert engine.closed == 1


def test_a_loaded_engine_is_ready_in_health() -> None:
    engine = FakeHearing()
    runtime = _starting(engine)

    runtime.load()

    assert runtime.health()["hearing"] == {"model": engine.model_name, "ready": True}


def test_closing_the_runtime_frees_the_hearing_engine() -> None:
    engine = FakeHearing()
    runtime = _starting(engine)
    runtime.load()

    runtime.close()

    assert engine.closed == 1


def test_without_a_hearing_model_health_names_hearing_unavailable() -> None:
    runtime = _starting(None)

    runtime.load()

    assert runtime.health()["hearing"] == {"model": "unavailable", "ready": False}


def test_a_configured_model_that_was_not_fetched_stops_the_start_without_a_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden(*_arguments: object, **_keywords: object) -> str:
        message = "the start must not download"
        raise AssertionError(message)

    monkeypatch.setattr(hearing_models, "snapshot_download", forbidden)
    settings = Settings(hearing_model=HearingModel.SMALL, huggingface_cache=tmp_path)

    with pytest.raises(HearingModelMissingError, match="SPEECH_HEARING_MODEL"):
        Runtime.from_settings(settings)


def test_a_configured_model_that_was_fetched_is_named_in_health_before_it_loads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(hearing_models, "snapshot_download", write_snapshot)
    hearing_models.fetch_model(HearingModel.SMALL, tmp_path)
    settings = Settings(hearing_model=HearingModel.SMALL, huggingface_cache=tmp_path)

    runtime = Runtime.from_settings(settings)

    assert runtime.hearing.health() == {
        "model": HearingModel.SMALL.value,
        "ready": False,
    }

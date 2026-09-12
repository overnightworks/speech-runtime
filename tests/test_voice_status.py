"""The private voice endpoint reports evidence, never package support."""

import os

import pytest
from fastapi.testclient import TestClient

from speech import voices
from speech.config import Settings
from speech.service import Runtime, create_app, create_control_app
from speech.voices import VoiceId, VoiceState, statuses
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

    missing_metadata = statuses(settings, ready=False, loading=False)

    (tmp_path / "de_DE-thorsten-medium.onnx.json").touch()
    complete = statuses(settings, ready=False, loading=False)

    assert missing_metadata[0].state is VoiceState.NOT_DOWNLOADED
    assert complete[0].state is VoiceState.DOWNLOADED
    assert [voice.id for voice in complete] == list(VoiceId)
    assert all(voice.state is VoiceState.UNAVAILABLE for voice in complete[-3:])


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


def test_status_refuses_an_unknown_configured_model(tmp_path) -> None:
    settings = Settings(
        speaking_model="unknown-model",
        voice_cache=tmp_path,
        huggingface_cache=tmp_path,
        PRESENTATOR_RUNTIME_UID=os.geteuid(),
    )

    with pytest.raises(ValueError, match="not in the catalogue"):
        statuses(settings, ready=False, loading=False)


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
        statuses(settings, ready=False, loading=False)

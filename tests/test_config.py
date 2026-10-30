"""Configuration comes from the environment; a failed load names the model."""

import os
from typing import TYPE_CHECKING

import pytest

from speech_runtime_service.config import (
    HEAR_SAMPLE_RATE,
    MagpieSpeaker,
    QwenSpeaker,
    Settings,
)
from speech_runtime_service.pcm import wav_header
from speech_runtime_service.selection import VoiceSelectionStore
from speech_runtime_service.service import (
    Runtime,
    RuntimeDependencies,
    failed_to_load_message,
)
from speech_runtime_service.voices import VoiceId
from tests.support.fakes_support import FakeHearing, FakeSpeaking

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("SPEECH_HOST", "example.invalid"),
        ("SPEECH_PORT", "8090"),
        ("SPEECH_DEBUG", "true"),
    ],
)
def test_a_stale_setting_in_the_environment_is_ignored(
    monkeypatch: pytest.MonkeyPatch, name: str, value: str
) -> None:
    monkeypatch.setenv(name, value)

    Settings()


def test_settings_read_speech_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SPEECH_DEVICE", "cpu")
    monkeypatch.setenv("SPEECH_SPEAKING_MODEL", "ResembleAI/chatterbox")
    monkeypatch.setenv("SPEECH_HEARING_MODEL", "Systran/faster-whisper-large-v3")
    monkeypatch.setenv("PRESENTATOR_RUNTIME_UID", str(os.geteuid()))

    settings = Settings()

    assert settings.device == "cpu"
    assert settings.speaking_model == "ResembleAI/chatterbox"
    assert settings.hearing_model == "Systran/faster-whisper-large-v3"


def test_settings_default_uses_the_provider_hub_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider_cache = tmp_path / "provider-hub-cache"
    monkeypatch.delenv("SPEECH_HUGGINGFACE_CACHE", raising=False)
    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_CACHE", provider_cache)

    assert Settings().huggingface_cache == provider_cache


def test_settings_preserves_an_explicit_huggingface_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    explicit_cache = tmp_path / "operator-hub-cache"
    monkeypatch.setenv("SPEECH_HUGGINGFACE_CACHE", str(explicit_cache))

    assert Settings().huggingface_cache == explicit_cache


def test_settings_accepts_only_a_positive_shared_runtime_uid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("PRESENTATOR_RUNTIME_UID", raising=False)

    assert Settings().runtime_uid is None

    monkeypatch.setenv("PRESENTATOR_RUNTIME_UID", "0")
    with pytest.raises(ValueError, match="greater than 0"):
        Settings()


@pytest.mark.parametrize(
    ("environment", "value", "reason"),
    [
        ("SPEECH_STATE_DIRECTORY", "relative/state", "state directory"),
        ("SPEECH_PROVIDER_ROOT", "relative/providers", "provider root"),
        ("SPEECH_SPEAKING_MODEL", "unsupported-baseline", "speaking model"),
    ],
)
def test_settings_refuses_unsafe_voice_selection_configuration(
    monkeypatch: pytest.MonkeyPatch, environment: str, value: str, reason: str
) -> None:
    monkeypatch.setenv(environment, value)

    with pytest.raises(ValueError, match=reason) as refused:
        Settings()

    assert value not in str(refused.value)


def test_provider_root_is_optional_for_the_default_piper_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SPEECH_PROVIDER_ROOT", raising=False)
    assert Settings().provider_root is None


def test_qwen_speaker_is_a_closed_setting_with_ryan_as_its_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert Settings().qwen_speaker is QwenSpeaker.RYAN

    monkeypatch.setenv("SPEECH_QWEN_SPEAKER", "Sohee")
    assert Settings().qwen_speaker is QwenSpeaker.SOHEE

    monkeypatch.setenv("SPEECH_QWEN_SPEAKER", "invented")
    with pytest.raises(ValueError, match="qwen_speaker"):
        Settings()


def test_magpie_speaker_is_a_closed_setting_with_sofia_as_its_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert Settings().magpie_speaker is MagpieSpeaker.SOFIA

    monkeypatch.setenv("SPEECH_MAGPIE_SPEAKER", "Aria")
    assert Settings().magpie_speaker is MagpieSpeaker.ARIA

    monkeypatch.setenv("SPEECH_MAGPIE_SPEAKER", "invented")
    with pytest.raises(ValueError, match="magpie_speaker"):
        Settings()


def test_a_failed_hearing_load_names_the_model(tmp_path: Path) -> None:
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

    with pytest.raises(RuntimeError, match="hearing model fake-ears failed to load"):
        runtime.load()

    assert (
        failed_to_load_message("hearing", "fake-ears")
        == "hearing model fake-ears failed to load"
    )


def test_wav_header_is_sixteen_bit_mono() -> None:
    header = wav_header(16_000, 100)
    assert header[20:22] == (1).to_bytes(2, "little")
    assert header[22:24] == (1).to_bytes(2, "little")
    assert int.from_bytes(header[24:28], "little") == HEAR_SAMPLE_RATE
    assert header[34:36] == (16).to_bytes(2, "little")

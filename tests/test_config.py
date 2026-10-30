"""Configuration comes from the environment."""

import ipaddress
from typing import TYPE_CHECKING

import pytest
from pydantic import ValidationError
from speech_runtime_provider_contract import (
    MAGPIE_MODEL_ID,
    QWEN_MODEL_ID,
    VOXCPM_MODEL_ID,
)

from speech_runtime_service.config import (
    MagpieSpeaker,
    QwenSpeaker,
    Settings,
)
from speech_runtime_service.pcm import wav_header
from speech_runtime_service.voices import (
    CHATTERBOX_SPEAKING_MODEL,
    DEFAULT_SPEAKING_MODEL,
)
from tests.support.fakes_support import SPEAK_SAMPLE_RATE

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("SPEECH_DEBUG", "true"),
        ("SPEECH_HEARING_MODEL", "Systran/faster-whisper-large-v3"),
        ("SPEECH_STATE_DIRECTORY", "relative/state"),
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

    settings = Settings()

    assert settings.device == "cpu"
    assert settings.speaking_model == "ResembleAI/chatterbox"


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("SPEECH_BUSY_WAIT_SECONDS", "-1"),
        ("SPEECH_BUSY_WAIT_SECONDS", "nan"),
        ("SPEECH_BUSY_WAIT_SECONDS", "inf"),
        ("SPEECH_MAX_INPUT_CHARACTERS", "0"),
        ("SPEECH_MAX_INPUT_CHARACTERS", "-1"),
    ],
)
def test_the_wire_settings_refuse_values_that_cannot_hold(
    monkeypatch: pytest.MonkeyPatch, name: str, value: str
) -> None:
    monkeypatch.setenv(name, value)

    with pytest.raises(ValidationError, match=name.removeprefix("SPEECH_").lower()):
        Settings()


def test_the_wire_settings_accept_the_smallest_whole_limit_and_no_wait(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SPEECH_BUSY_WAIT_SECONDS", "0")
    monkeypatch.setenv("SPEECH_MAX_INPUT_CHARACTERS", "1")

    settings = Settings()

    assert settings.busy_wait_seconds == 0
    assert settings.max_input_characters == 1


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


def test_settings_bind_to_a_loopback_tcp_port_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SPEECH_HOST", raising=False)
    monkeypatch.delenv("SPEECH_PORT", raising=False)

    settings = Settings()

    assert ipaddress.ip_address(settings.host).is_loopback
    assert settings.port in range(1, 2**16)


def test_settings_read_the_bind_address_from_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SPEECH_HOST", "0.0.0.0")  # noqa: S104 - the container form
    monkeypatch.setenv("SPEECH_PORT", "65535")

    settings = Settings()

    assert (settings.host, settings.port) == ("0.0.0.0", 65535)  # noqa: S104


@pytest.mark.parametrize("port", ["0", "65536", "-1", "http", ""])
def test_settings_refuse_a_port_that_is_not_a_tcp_port(
    monkeypatch: pytest.MonkeyPatch, port: str
) -> None:
    monkeypatch.setenv("SPEECH_PORT", port)

    with pytest.raises(ValidationError, match="port"):
        Settings()


@pytest.mark.parametrize(
    "model",
    [
        DEFAULT_SPEAKING_MODEL,
        CHATTERBOX_SPEAKING_MODEL,
        QWEN_MODEL_ID,
        VOXCPM_MODEL_ID,
        MAGPIE_MODEL_ID,
    ],
)
def test_every_catalogue_voice_model_is_a_speaking_model(
    monkeypatch: pytest.MonkeyPatch, model: str
) -> None:
    monkeypatch.setenv("SPEECH_SPEAKING_MODEL", model)

    assert Settings().speaking_model == model


@pytest.mark.parametrize(
    ("environment", "value", "reason"),
    [
        ("SPEECH_PROVIDER_ROOT", "relative/providers", "provider root"),
        ("SPEECH_SPEAKING_MODEL", "unsupported-baseline", "speaking model"),
    ],
)
def test_settings_refuses_unsafe_voice_configuration(
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


def test_wav_header_is_sixteen_bit_mono() -> None:
    header = wav_header(SPEAK_SAMPLE_RATE, 100)
    assert header[20:22] == (1).to_bytes(2, "little")
    assert header[22:24] == (1).to_bytes(2, "little")
    assert int.from_bytes(header[24:28], "little") == SPEAK_SAMPLE_RATE
    assert header[34:36] == (16).to_bytes(2, "little")

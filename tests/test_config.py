"""Configuration comes from the environment; a failed load names the model."""

import logging

import pytest
from fastapi.testclient import TestClient

from speech.config import HEAR_SAMPLE_RATE, Settings
from speech.pcm import wav_header
from speech.service import Runtime, failed_to_load_message
from tests.conftest import FakeHearing, FakeSpeaking, an_app


def test_settings_read_speech_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SPEECH_PORT", "9001")
    monkeypatch.setenv("SPEECH_DEVICE", "cpu")
    monkeypatch.setenv("SPEECH_SPEAKING_MODEL", "de_DE-thorsten-medium")
    monkeypatch.setenv("SPEECH_HEARING_MODEL", "Systran/faster-whisper-large-v3")
    monkeypatch.setenv("SPEECH_DEBUG", "true")

    settings = Settings()

    assert settings.port == 9001
    assert settings.device == "cpu"
    assert settings.speaking_model == "de_DE-thorsten-medium"
    assert settings.hearing_model == "Systran/faster-whisper-large-v3"
    assert settings.debug is True


def test_a_failed_load_names_the_model(monkeypatch: pytest.MonkeyPatch) -> None:
    exits: list[int] = []
    monkeypatch.setattr(
        "speech.service.os._exit",
        lambda code: exits.append(code) or (_ for _ in ()).throw(SystemExit(code)),
    )
    runtime = Runtime(FakeSpeaking(fail=True), FakeHearing())

    with pytest.raises(SystemExit):
        runtime.load()

    assert exits == [1]
    assert (
        failed_to_load_message("speaking", "fake-voice")
        == "speaking model fake-voice failed to load"
    )


def test_speak_does_not_log_the_sentence_without_debug(
    caplog: pytest.LogCaptureFixture,
) -> None:
    sentence = "Nur dieser Satz darf nicht in den Logs stehen"
    caplog.set_level(logging.DEBUG)
    TestClient(an_app(debug=False)).post(
        "/speak",
        json={"text": sentence, "language": "de"},
    )

    assert sentence not in caplog.text


def test_speak_logs_the_sentence_when_debug_is_on(
    caplog: pytest.LogCaptureFixture,
) -> None:
    sentence = "Dieser Satz steht im Debug-Log"
    caplog.set_level(logging.DEBUG)
    TestClient(an_app(debug=True)).post(
        "/speak",
        json={"text": sentence, "language": "de"},
    )

    assert sentence in caplog.text


def test_wav_header_is_sixteen_bit_mono() -> None:
    header = wav_header(16_000, 100)
    assert header[20:22] == (1).to_bytes(2, "little")
    assert header[22:24] == (1).to_bytes(2, "little")
    assert int.from_bytes(header[24:28], "little") == HEAR_SAMPLE_RATE
    assert header[34:36] == (16).to_bytes(2, "little")

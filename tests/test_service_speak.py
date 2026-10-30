"""The speaking engines the runtime builds: the local Piper voice and the providers."""

import os
import sys
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest
from speech_runtime_provider_contract import CHATTERBOX_SAMPLE_RATE

from speech_runtime_service.config import Settings
from speech_runtime_service.speaking import PiperSpeaking, speaking_for_voice
from speech_runtime_service.voices import CHATTERBOX_SPEAKING_MODEL, VoiceId

if TYPE_CHECKING:
    from pathlib import Path


def test_piper_voice_id_constructs_the_local_piper_engine() -> None:
    engine = speaking_for_voice(
        Settings(PRESENTATOR_RUNTIME_UID=os.geteuid()), VoiceId.PIPER
    )

    assert isinstance(engine, PiperSpeaking)
    assert engine.streams is False
    assert engine.model_name == "de_DE-thorsten-medium"


def test_chatterbox_voice_id_constructs_the_streaming_voice() -> None:
    engine = speaking_for_voice(
        Settings(
            device="cuda",
            PRESENTATOR_RUNTIME_UID=os.geteuid(),
        ),
        VoiceId.CHATTERBOX,
    )

    assert engine.model_name == CHATTERBOX_SPEAKING_MODEL
    assert engine.streams is True
    assert engine.sample_rate == CHATTERBOX_SAMPLE_RATE


def test_piper_load_requires_local_onnx_and_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    loaded: list[object] = []

    class LocalVoice:
        config = SimpleNamespace(sample_rate=22_050)

    def load(path: object) -> LocalVoice:
        loaded.append(path)
        return LocalVoice()

    monkeypatch.setitem(
        sys.modules,
        "piper",
        SimpleNamespace(PiperVoice=SimpleNamespace(load=load)),
    )
    engine = PiperSpeaking("local-voice", tmp_path)

    with pytest.raises(FileNotFoundError):
        engine.load()
    (tmp_path / "local-voice.onnx").touch()
    (tmp_path / "local-voice.onnx.json").touch()

    engine.load()

    assert loaded == [tmp_path / "local-voice.onnx"]
    assert engine.ready is True

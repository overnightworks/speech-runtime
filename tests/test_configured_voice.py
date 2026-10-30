"""The configured speaking model is the one source of the active voice."""

import asyncio
import sys
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest
from fastapi.testclient import TestClient
from speech_runtime_provider_contract import (
    CHATTERBOX_ARTIFACTS,
    CHATTERBOX_REVISION,
    CHATTERBOX_SAMPLE_RATE,
    MAGPIE_CODEC_FILENAME,
    MAGPIE_CODEC_ID,
    MAGPIE_CODEC_REVISION,
    MAGPIE_MODEL_FILENAME,
    MAGPIE_MODEL_ID,
    MAGPIE_MODEL_REVISION,
    MAGPIE_SAMPLE_RATE,
    QWEN_ARTIFACTS,
    QWEN_MODEL_ID,
    QWEN_REVISION,
    QWEN_SAMPLE_RATE,
    VOXCPM_ARTIFACTS,
    VOXCPM_MODEL_ID,
    VOXCPM_REVISION,
    VOXCPM_SAMPLE_RATE,
)
from starlette.status import HTTP_200_OK, HTTP_404_NOT_FOUND
from starlette.websockets import WebSocketDisconnect

from speech_runtime_service.config import Settings
from speech_runtime_service.control import create_control_app
from speech_runtime_service.pcm import pcm_from_wav
from speech_runtime_service.service import Runtime
from speech_runtime_service.voices import (
    CHATTERBOX_SPEAKING_MODEL,
    DEFAULT_SPEAKING_MODEL,
    VoiceId,
)
from tests.support.fakes_support import FakeSpeaking
from tests.support.identity_support import valid_callers
from tests.support.speak_support import (
    app_over,
    fake_runtime,
    post_speech,
    speak_body,
    speak_once,
    wav_body,
)
from tests.support.worker_support import closed_provider

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

PIPER_SAMPLE_RATE = 22_050
WORKER_PCM = b"\x00\x01"


def _touch_snapshot(
    cache: Path, repository: str, revision: str, files: Iterable[str]
) -> None:
    snapshot = cache / f"models--{repository.replace('/', '--')}" / "snapshots"
    for name in files:
        artifact = snapshot / revision / name
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.touch()


def _install_gpu_voice(voice: VoiceId, cache: Path) -> None:
    if voice is VoiceId.CHATTERBOX:
        _touch_snapshot(
            cache, CHATTERBOX_SPEAKING_MODEL, CHATTERBOX_REVISION, CHATTERBOX_ARTIFACTS
        )
    elif voice is VoiceId.QWEN:
        _touch_snapshot(cache, QWEN_MODEL_ID, QWEN_REVISION, QWEN_ARTIFACTS)
    elif voice is VoiceId.VOXCPM:
        _touch_snapshot(cache, VOXCPM_MODEL_ID, VOXCPM_REVISION, VOXCPM_ARTIFACTS)
    else:
        _touch_snapshot(
            cache, MAGPIE_MODEL_ID, MAGPIE_MODEL_REVISION, [MAGPIE_MODEL_FILENAME]
        )
        _touch_snapshot(
            cache, MAGPIE_CODEC_ID, MAGPIE_CODEC_REVISION, [MAGPIE_CODEC_FILENAME]
        )


def _install_piper(monkeypatch: pytest.MonkeyPatch, voice_cache: Path) -> None:
    class LocalVoice:
        config = SimpleNamespace(sample_rate=PIPER_SAMPLE_RATE)

        def synthesize(self, _text: str) -> Iterable[SimpleNamespace]:
            return [SimpleNamespace(audio_int16_bytes=WORKER_PCM)]

    monkeypatch.setitem(
        sys.modules,
        "piper",
        SimpleNamespace(PiperVoice=SimpleNamespace(load=lambda _path: LocalVoice())),
    )
    voice_cache.mkdir()
    (voice_cache / f"{DEFAULT_SPEAKING_MODEL}.onnx").touch()
    (voice_cache / f"{DEFAULT_SPEAKING_MODEL}.onnx.json").touch()


def _speak(runtime: Runtime, settings: Settings, voice: VoiceId) -> tuple[int, bytes]:
    app = create_control_app(settings, runtime, valid_callers())
    messages = asyncio.run(speak_once(app, voice.value))
    return pcm_from_wav(wav_body(messages))


def _settings(tmp_path: Path, model: str, provider: str | None) -> Settings:
    root = closed_provider(tmp_path, provider_name=provider)[0] if provider else None
    return Settings(
        speaking_model=model,
        provider_root=root,
        huggingface_cache=tmp_path / "cache",
        voice_cache=tmp_path / "piper",
        device="cpu",
    )


@pytest.mark.parametrize(
    ("voice", "model", "provider_and_rate"),
    [
        (VoiceId.PIPER, DEFAULT_SPEAKING_MODEL, (None, PIPER_SAMPLE_RATE)),
        (
            VoiceId.CHATTERBOX,
            CHATTERBOX_SPEAKING_MODEL,
            ("chatterbox", CHATTERBOX_SAMPLE_RATE),
        ),
        (VoiceId.QWEN, QWEN_MODEL_ID, ("qwen", QWEN_SAMPLE_RATE)),
        (VoiceId.VOXCPM, VOXCPM_MODEL_ID, ("voxcpm", VOXCPM_SAMPLE_RATE)),
        (VoiceId.MAGPIE, MAGPIE_MODEL_ID, ("magpie", MAGPIE_SAMPLE_RATE)),
    ],
)
def test_each_catalogue_voice_starts_from_its_model_and_the_wire_speaks_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    voice: VoiceId,
    model: str,
    provider_and_rate: tuple[str | None, int],
) -> None:
    provider, sample_rate = provider_and_rate
    _install_piper(monkeypatch, tmp_path / "piper")
    _install_gpu_voice(voice, tmp_path / "cache")
    settings = _settings(tmp_path, model, provider)
    runtime = Runtime.from_settings(settings)
    try:
        runtime.load()

        rate, pcm = _speak(runtime, settings, voice)
        health = TestClient(create_control_app(settings, runtime, valid_callers()))
        speaking = health.get("/health").json()["speaking"]
    finally:
        runtime.close()

    assert (rate, pcm) == (sample_rate, WORKER_PCM)
    assert speaking["model"] == model
    assert speaking["ready"] is True


def test_a_selection_file_from_an_older_build_is_ignored_and_nothing_is_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_piper(monkeypatch, tmp_path / "piper")
    _install_gpu_voice(VoiceId.CHATTERBOX, tmp_path / "cache")
    state = tmp_path / "state"
    state.mkdir()
    older_selection = state / "selected-voice"
    older_selection.write_text(VoiceId.CHATTERBOX.value, encoding="utf-8")
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("SPEECH_STATE_DIRECTORY", str(state))
    monkeypatch.setenv("HOME", str(home))
    settings = _settings(tmp_path, DEFAULT_SPEAKING_MODEL, None)
    before = sorted(tmp_path.rglob("*"))
    runtime = Runtime.from_settings(settings)
    try:
        runtime.load()

        rate, pcm = _speak(runtime, settings, VoiceId.PIPER)
        refused = post_speech(
            TestClient(app_over(runtime)), speak_body(VoiceId.CHATTERBOX.value)
        )
    finally:
        runtime.close()

    assert (rate, pcm) == (PIPER_SAMPLE_RATE, WORKER_PCM)
    assert refused.status_code != HTTP_200_OK
    assert older_selection.read_text(encoding="utf-8") == VoiceId.CHATTERBOX.value
    assert sorted(tmp_path.rglob("*")) == before


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("GET", "/voices"),
        ("POST", "/voices/piper/load"),
        ("POST", "/voices/qwen3-tts-0.6b/download"),
        ("POST", "/voices/piper/sample/de"),
    ],
)
def test_voice_administration_is_not_mounted(method: str, path: str) -> None:
    speaking = FakeSpeaking()
    client = TestClient(
        create_control_app(Settings(), fake_runtime(speaking), valid_callers())
    )

    response = client.request(method, path)

    assert response.status_code == HTTP_404_NOT_FOUND
    assert speaking.heard_text == ""


def test_hearing_is_not_mounted() -> None:
    client = TestClient(create_control_app(Settings(), fake_runtime(), valid_callers()))

    with pytest.raises(WebSocketDisconnect), client.websocket_connect("/hear"):
        pass

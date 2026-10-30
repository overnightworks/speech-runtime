"""The private voice endpoint reports evidence, never package support."""

import asyncio
import builtins
import importlib
import os
import sys
import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pytest
from fastapi.testclient import TestClient
from speech_runtime_provider_contract import (
    MAGPIE_SAMPLE_RATE,
    QWEN_SAMPLE_RATE,
    VOXCPM_ARTIFACTS,
    VOXCPM_MODEL_ID,
    VOXCPM_REVISION,
    VOXCPM_SAMPLE_RATE,
)
from starlette.status import (
    HTTP_200_OK,
)

from speech_runtime_service import voices
from speech_runtime_service.config import Settings
from speech_runtime_service.control import create_control_app
from speech_runtime_service.pcm import pcm_from_wav
from speech_runtime_service.selection import VoiceSelectionStore
from speech_runtime_service.service import (
    Runtime,
    RuntimeDependencies,
)
from speech_runtime_service.speaking import speaking_for_voice
from speech_runtime_service.voices import (
    VoiceId,
    VoiceLoadOutcome,
    VoiceRuntimeState,
    VoiceState,
    statuses,
)
from tests.support.fakes_support import FakeHearing, FakeSpeaking
from tests.support.speak_support import (
    ASGI_TIMEOUT_SECONDS,
    speak_body,
    speak_once,
    speak_scope,
    wav_body,
)
from tests.support.worker_support import assert_reaped, closed_provider, observed

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

    from fastapi import FastAPI

_SERVICE_MODULES = ("speech_runtime_service.voices", "speech_runtime_service.speaking")


def test_status_requires_both_piper_artifacts_and_keeps_unimplemented_rows_unavailable(
    tmp_path: Path,
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
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(voices, "try_to_load_from_cache", lambda *_args, **_kwargs: "x")
    provider_root = tmp_path / "providers"
    settings = Settings(
        provider_root=provider_root,
        huggingface_cache=tmp_path,
        PRESENTATOR_RUNTIME_UID=os.geteuid(),
    )
    entrypoint = provider_root / "chatterbox/.venv/bin/speech-runtime-chatterbox-worker"
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


def _touch_files(paths: Iterable[Path]) -> None:
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()


def _complete_qwen_snapshot(cache: Path) -> Path:
    snapshot = (
        cache
        / "models--Qwen--Qwen3-TTS-12Hz-0.6B-CustomVoice"
        / "snapshots"
        / "85e237c12c027371202489a0ec509ded67b5e4b5"
    )
    artifacts = (
        ".gitattributes",
        "README.md",
        "config.json",
        "generation_config.json",
        "merges.txt",
        "model.safetensors",
        "preprocessor_config.json",
        "speech_tokenizer/config.json",
        "speech_tokenizer/configuration.json",
        "speech_tokenizer/model.safetensors",
        "speech_tokenizer/preprocessor_config.json",
        "tokenizer_config.json",
        "vocab.json",
    )
    _touch_files(snapshot / artifact for artifact in artifacts)
    return snapshot


def _complete_voxcpm_snapshot(cache: Path) -> Path:
    repository = f"models--{VOXCPM_MODEL_ID.replace('/', '--')}"
    snapshot = cache / repository / "snapshots" / VOXCPM_REVISION
    _touch_files(snapshot / artifact for artifact in VOXCPM_ARTIFACTS)
    return snapshot


def _complete_magpie_snapshots(cache: Path) -> tuple[Path, Path]:
    model = (
        cache
        / "models--nvidia--magpie_tts_multilingual_357m"
        / "snapshots"
        / "5023df68bd3f5b5ce6d666a50979bc501af145cc"
        / "magpie_tts_multilingual_357m.nemo"
    )
    codec = (
        cache
        / "models--nvidia--nemo-nano-codec-22khz-1.89kbps-21.5fps"
        / "snapshots"
        / "fc00890b604aa2de298d2641ffc6c5f6caf8c4d7"
        / "nemo-nano-codec-22khz-1.89kbps-21.5fps.nemo"
    )
    _touch_files((model, codec))
    return model, codec


def test_qwen_requires_the_exact_snapshot_and_executable_and_has_a_truthful_preset(
    tmp_path: Path,
) -> None:
    provider_root = tmp_path / "providers"
    cache = tmp_path / "hub"
    snapshot = _complete_qwen_snapshot(cache)
    settings = Settings(
        provider_root=provider_root,
        huggingface_cache=cache,
        PRESENTATOR_RUNTIME_UID=os.geteuid(),
    )

    missing_worker = statuses(settings, _RuntimeFacts().state())
    worker = provider_root / "qwen/.venv/bin/speech-runtime-qwen-worker"
    worker.parent.mkdir(parents=True)
    worker.write_text("#!/bin/false\n", encoding="utf-8")
    worker.chmod(0o700)
    (snapshot / "vocab.json").unlink()
    missing_artifact = statuses(settings, _RuntimeFacts().state())
    (snapshot / "vocab.json").touch()
    downloaded = statuses(settings, _RuntimeFacts().state())
    sohee = statuses(
        Settings(
            provider_root=provider_root,
            huggingface_cache=cache,
            qwen_speaker="Sohee",
            PRESENTATOR_RUNTIME_UID=os.geteuid(),
        ),
        _RuntimeFacts().state(),
    )

    assert missing_worker[2].state is VoiceState.UNAVAILABLE
    assert missing_artifact[2].state is VoiceState.NOT_DOWNLOADED
    assert downloaded[2].state is VoiceState.DOWNLOADED
    assert downloaded[2].language == "German and English — Ryan preset"
    assert sohee[2].language == "German and English — Sohee preset"

    engine = speaking_for_voice(settings, VoiceId.QWEN)
    assert engine.model_name == "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
    assert engine.sample_rate == QWEN_SAMPLE_RATE
    assert engine.streams is False
    assert engine.retain_when_inactive is False


def test_voxcpm_requires_the_exact_snapshot_and_worker_and_uses_48_khz(
    tmp_path: Path,
) -> None:
    provider_root = tmp_path / "providers"
    cache = tmp_path / "hub"
    snapshot = _complete_voxcpm_snapshot(cache)
    settings = Settings(
        provider_root=provider_root,
        huggingface_cache=cache,
        PRESENTATOR_RUNTIME_UID=os.geteuid(),
    )

    missing_worker = statuses(settings, _RuntimeFacts().state())
    worker = provider_root / "voxcpm/.venv/bin/speech-runtime-voxcpm-worker"
    worker.parent.mkdir(parents=True)
    worker.write_text("#!/bin/false\n", encoding="utf-8")
    worker.chmod(0o700)
    (snapshot / "model.safetensors").unlink()
    missing_artifact = statuses(settings, _RuntimeFacts().state())
    (snapshot / "model.safetensors").touch()
    downloaded = statuses(settings, _RuntimeFacts().state())

    assert missing_worker[3].state is VoiceState.UNAVAILABLE
    assert missing_artifact[3].state is VoiceState.UNAVAILABLE
    assert downloaded[3].state is VoiceState.DOWNLOADED
    assert downloaded[3].language == "German and English — text-only"

    engine = speaking_for_voice(settings, VoiceId.VOXCPM)
    assert engine.model_name == "openbmb/VoxCPM2"
    assert engine.sample_rate == VOXCPM_SAMPLE_RATE
    assert engine.streams is True
    assert engine.retain_when_inactive is False


def test_magpie_requires_both_exact_archives_and_its_worker(tmp_path: Path) -> None:
    provider_root = tmp_path / "providers"
    cache = tmp_path / "hub"
    model, codec = _complete_magpie_snapshots(cache)
    settings = Settings(
        provider_root=provider_root,
        huggingface_cache=cache,
        PRESENTATOR_RUNTIME_UID=os.geteuid(),
    )

    missing_worker = statuses(settings, _RuntimeFacts().state())
    worker = provider_root / "magpie/.venv/bin/speech-runtime-magpie-worker"
    worker.parent.mkdir(parents=True)
    worker.write_text("#!/bin/false\n", encoding="utf-8")
    worker.chmod(0o700)
    model.unlink()
    missing_model = statuses(settings, _RuntimeFacts().state())
    model.touch()
    codec.unlink()
    missing_codec = statuses(settings, _RuntimeFacts().state())
    codec.touch()
    downloaded = statuses(settings, _RuntimeFacts().state())

    assert missing_worker[4].state is VoiceState.UNAVAILABLE
    assert missing_model[4].state is VoiceState.UNAVAILABLE
    assert missing_codec[4].state is VoiceState.UNAVAILABLE
    assert downloaded[4].state is VoiceState.DOWNLOADED
    assert downloaded[4].language == "German and English — Sofia fixed voice"

    engine = speaking_for_voice(settings, VoiceId.MAGPIE)
    assert engine.model_name == "nvidia/magpie_tts_multilingual_357m"
    assert engine.sample_rate == MAGPIE_SAMPLE_RATE
    assert engine.streams is False


def test_voxcpm_catalogue_and_factory_import_no_provider_packages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider_root = tmp_path / "providers"
    cache = tmp_path / "hub"
    _complete_voxcpm_snapshot(cache)
    worker = provider_root / "voxcpm/.venv/bin/speech-runtime-voxcpm-worker"
    worker.parent.mkdir(parents=True)
    worker.write_text("#!/bin/false\n", encoding="utf-8")
    worker.chmod(0o700)
    settings = Settings(
        provider_root=provider_root,
        huggingface_cache=cache,
        PRESENTATOR_RUNTIME_UID=os.geteuid(),
    )
    host = importlib.import_module("speech_runtime_service")
    marker = object()
    previous_attributes = {
        name: getattr(host, name, marker) for name in ("voices", "speaking")
    }
    previous_modules = {
        name: sys.modules.pop(name) for name in _SERVICE_MODULES if name in sys.modules
    }
    original_import = builtins.__import__

    def guard_provider_import(
        name: str,
        global_namespace: dict[str, object] | None = None,
        local_namespace: dict[str, object] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> object:
        if name == "speech_runtime_voxcpm" or name.startswith("speech_runtime_voxcpm."):
            raise AssertionError
        if name == "voxcpm" or name.startswith("voxcpm."):
            raise AssertionError
        return original_import(name, global_namespace, local_namespace, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guard_provider_import)
    try:
        fresh_voices = importlib.import_module("speech_runtime_service.voices")
        fresh_speaking = importlib.import_module("speech_runtime_service.speaking")
        rows = fresh_voices.statuses(
            settings,
            fresh_voices.VoiceRuntimeState(
                selected=None, ready=False, loading=False, pending=None, failed=None
            ),
        )
        voxcpm = next(row for row in rows if row.id is fresh_voices.VoiceId.VOXCPM)
        engine = fresh_speaking.speaking_for_voice(
            settings, fresh_voices.VoiceId.VOXCPM
        )
        assert voxcpm.state is fresh_voices.VoiceState.DOWNLOADED
        assert engine.model_name == VOXCPM_MODEL_ID
        assert engine.sample_rate == VOXCPM_SAMPLE_RATE
    finally:
        for name in _SERVICE_MODULES:
            sys.modules.pop(name, None)
        sys.modules.update(previous_modules)
        for name, value in previous_attributes.items():
            if value is marker:
                delattr(host, name)
            else:
                setattr(host, name, value)


def test_magpie_catalogue_and_factory_import_no_provider_packages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider_root, cache = closed_provider(tmp_path, provider_name="magpie")
    _complete_magpie_snapshots(cache)
    settings = Settings(
        provider_root=provider_root,
        huggingface_cache=cache,
        PRESENTATOR_RUNTIME_UID=os.geteuid(),
    )
    host = importlib.import_module("speech_runtime_service")
    marker = object()
    previous_attributes = {
        name: getattr(host, name, marker) for name in ("voices", "speaking")
    }
    previous_modules = {
        name: sys.modules.pop(name) for name in _SERVICE_MODULES if name in sys.modules
    }
    original_import = builtins.__import__

    def guard_provider_import(
        name: str,
        global_namespace: dict[str, object] | None = None,
        local_namespace: dict[str, object] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> object:
        if name.split(".", 1)[0] in {"nemo", "torch", "speech_runtime_magpie"}:
            raise AssertionError
        return original_import(name, global_namespace, local_namespace, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guard_provider_import)
    try:
        fresh_voices = importlib.import_module("speech_runtime_service.voices")
        fresh_speaking = importlib.import_module("speech_runtime_service.speaking")
        rows = fresh_voices.statuses(settings, _RuntimeFacts().state())
        magpie = next(row for row in rows if row.id is fresh_voices.VoiceId.MAGPIE)
        engine = fresh_speaking.speaking_for_voice(
            settings, fresh_voices.VoiceId.MAGPIE
        )
        assert magpie.state is fresh_voices.VoiceState.DOWNLOADED
        assert engine.sample_rate == MAGPIE_SAMPLE_RATE
    finally:
        for name in _SERVICE_MODULES:
            sys.modules.pop(name, None)
        sys.modules.update(previous_modules)
        for name, value in previous_attributes.items():
            if value is marker:
                delattr(host, name)
            else:
                setattr(host, name, value)


async def _disconnect_after_request_reaches_child(
    app: FastAPI, voice: VoiceId, request_ids: Path
) -> list[dict[str, object]]:
    messages: list[dict[str, object]] = []
    sent_body = False
    body_chunks = 0
    disconnect = asyncio.Event()
    delivered = asyncio.Event()
    pcm_sent = asyncio.Event()
    never = asyncio.Event()

    async def receive() -> dict[str, object]:
        nonlocal sent_body
        if not sent_body:
            sent_body = True
            return {
                "type": "http.request",
                "body": speak_body(voice.value, "wait-cancel"),
                "more_body": False,
            }
        await disconnect.wait()
        delivered.set()
        return {"type": "http.disconnect"}

    async def send(message: dict[str, object]) -> None:
        nonlocal body_chunks
        messages.append(message)
        if message["type"] == "http.response.body" and message.get("more_body"):
            body_chunks += 1
            if body_chunks == 1:
                pcm_sent.set()
                await never.wait()

    call = asyncio.create_task(app(speak_scope(), receive, send))
    await asyncio.to_thread(_wait_for_request_count, request_ids, 3)
    await asyncio.wait_for(pcm_sent.wait(), timeout=ASGI_TIMEOUT_SECONDS)
    disconnect.set()
    await asyncio.wait_for(delivered.wait(), timeout=ASGI_TIMEOUT_SECONDS)
    await asyncio.wait_for(call, timeout=ASGI_TIMEOUT_SECONDS)
    return messages


def _wait_for_request_count(request_ids: Path, expected_request_count: int) -> None:
    deadline = time.monotonic() + ASGI_TIMEOUT_SECONDS
    while (
        not request_ids.exists()
        or len(request_ids.read_text().splitlines()) < expected_request_count
    ) and time.monotonic() < deadline:
        threading.Event().wait(0.01)
    assert request_ids.exists()
    assert len(request_ids.read_text().splitlines()) == expected_request_count


def test_voxcpm_load_restore_samples_and_public_disconnect_cancel_the_child(
    tmp_path: Path,
) -> None:
    provider_root, cache = closed_provider(tmp_path, provider_name="voxcpm")
    _complete_voxcpm_snapshot(cache)
    settings = Settings(
        provider_root=provider_root,
        huggingface_cache=cache,
        device="cpu",
        PRESENTATOR_RUNTIME_UID=os.geteuid(),
    )
    selection = VoiceSelectionStore(tmp_path / "state", owner_uid=os.geteuid())

    def runtime() -> Runtime:
        return Runtime(
            None,
            FakeHearing(),
            dependencies=RuntimeDependencies(
                selection=selection,
                engine_factory=lambda voice: speaking_for_voice(settings, voice),
                artifact_checker=lambda voice: voice is VoiceId.VOXCPM,
            ),
        )

    initial = runtime()
    restored: Runtime | None = None
    try:
        control = TestClient(create_control_app(settings, initial))
        rows = control.get("/voices").json()["voices"]
        voxcpm = next(row for row in rows if row["id"] == VoiceId.VOXCPM)
        assert voxcpm == {
            "id": VoiceId.VOXCPM,
            "name": "VoxCPM2",
            "language": "German and English — text-only",
            "state": VoiceState.DOWNLOADED,
        }
        assert control.post("/voices/voxcpm2/load").json() == {
            "outcome": VoiceLoadOutcome.ACTIVATED
        }
        for language in ("de", "en"):
            response = control.post(f"/voices/voxcpm2/sample/{language}")
            rate, pcm = pcm_from_wav(response.content)
            assert response.status_code == HTTP_200_OK
            assert rate == VOXCPM_SAMPLE_RATE
            assert pcm == b"\x00\x01"
        assert selection.read() is VoiceId.VOXCPM
        initial.close()

        restored = runtime()
        restored.load()
        public = create_control_app(settings, restored)
        request_ids = cache / "request-ids"
        disconnected = asyncio.run(
            _disconnect_after_request_reaches_child(public, VoiceId.VOXCPM, request_ids)
        )
        assert any(message["type"] == "http.response.start" for message in disconnected)
        assert (cache / "cancel-observed").read_text().splitlines() == ["1"]

        later = asyncio.run(speak_once(public, VoiceId.VOXCPM.value))
        rate, pcm = pcm_from_wav(wav_body(later))
        assert rate == VOXCPM_SAMPLE_RATE
        assert pcm == b"\x00\x01"
        assert request_ids.read_text().splitlines() == ["1", "2", "1", "2"]
    finally:
        initial.close()
        if restored is not None:
            restored.close()


def test_magpie_load_restore_samples_public_speech_and_deselect_reap(
    tmp_path: Path,
) -> None:
    provider_root, cache = closed_provider(tmp_path, provider_name="magpie")
    model, codec = _complete_magpie_snapshots(cache)
    settings = Settings(
        provider_root=provider_root,
        huggingface_cache=cache,
        device="cpu",
        PRESENTATOR_RUNTIME_UID=os.geteuid(),
    )
    selection = VoiceSelectionStore(tmp_path / "state", owner_uid=os.geteuid())

    def runtime() -> Runtime:
        return Runtime(
            None,
            FakeHearing(),
            dependencies=RuntimeDependencies(
                selection=selection,
                engine_factory=lambda voice: (
                    speaking_for_voice(settings, voice)
                    if voice is VoiceId.MAGPIE
                    else FakeSpeaking()
                ),
                artifact_checker=lambda _voice: True,
            ),
        )

    initial = runtime()
    restored: Runtime | None = None
    try:
        control = TestClient(create_control_app(settings, initial))
        assert control.post("/voices/nvidia-magpie/load").json() == {
            "outcome": VoiceLoadOutcome.ACTIVATED
        }
        for language in ("de", "en"):
            response = control.post(f"/voices/nvidia-magpie/sample/{language}")
            rate, pcm = pcm_from_wav(response.content)
            assert response.status_code == HTTP_200_OK
            assert rate == MAGPIE_SAMPLE_RATE
            assert pcm == b"\x00\x01"
        _, _, process_id, _ = observed(cache)
        assert selection.read() is VoiceId.MAGPIE
        initial.close()

        restored = runtime()
        restored.load()
        public = create_control_app(settings, restored)
        request_ids = cache / "request-ids"
        disconnected = asyncio.run(
            _disconnect_after_request_reaches_child(public, VoiceId.MAGPIE, request_ids)
        )
        assert any(message["type"] == "http.response.start" for message in disconnected)
        assert (cache / "cancel-observed").read_text().splitlines() == ["1"]

        later = asyncio.run(speak_once(public, VoiceId.MAGPIE.value))
        rate, pcm = pcm_from_wav(wav_body(later))
        assert rate == MAGPIE_SAMPLE_RATE
        assert pcm == b"\x00\x01"
        assert model.is_file()
        assert codec.is_file()

        assert restored.load_voice(VoiceId.PIPER) is VoiceLoadOutcome.ACTIVATED
        assert_reaped(process_id)
        assert model.is_file()
        assert codec.is_file()
    finally:
        initial.close()
        if restored is not None:
            restored.close()


@dataclass(frozen=True, slots=True)
class _RuntimeFacts:
    selected: VoiceId | None = None
    ready: bool = False
    loading: bool = False
    pending: VoiceId | None = None
    failed: VoiceId | None = None
    downloading: bool = False
    download_failed: bool = False

    def state(self) -> VoiceRuntimeState:
        return VoiceRuntimeState(
            selected=self.selected,
            ready=self.ready,
            loading=self.loading,
            pending=self.pending,
            failed=self.failed,
            qwen_downloading=self.downloading,
            qwen_download_failed=self.download_failed,
        )


@dataclass(frozen=True, slots=True)
class _StatusScenario:
    voice: VoiceId
    runtime: VoiceRuntimeState
    installed: tuple[VoiceId, ...]
    usable: tuple[VoiceId, ...]
    expected: VoiceState


@pytest.mark.parametrize(
    "scenario",
    [
        _StatusScenario(
            VoiceId.QWEN,
            _RuntimeFacts(failed=VoiceId.QWEN).state(),
            (VoiceId.QWEN,),
            (VoiceId.QWEN,),
            VoiceState.FAILED,
        ),
        _StatusScenario(
            VoiceId.QWEN,
            _RuntimeFacts(selected=VoiceId.QWEN, ready=True).state(),
            (VoiceId.QWEN,),
            (VoiceId.QWEN,),
            VoiceState.ACTIVE,
        ),
        _StatusScenario(
            VoiceId.QWEN,
            _RuntimeFacts(
                selected=VoiceId.PIPER,
                ready=True,
                loading=True,
                pending=VoiceId.QWEN,
            ).state(),
            (VoiceId.QWEN,),
            (VoiceId.QWEN,),
            VoiceState.LOADING,
        ),
        _StatusScenario(
            VoiceId.QWEN,
            _RuntimeFacts().state(),
            (),
            (),
            VoiceState.UNAVAILABLE,
        ),
        _StatusScenario(
            VoiceId.QWEN,
            _RuntimeFacts(downloading=True).state(),
            (),
            (VoiceId.QWEN,),
            VoiceState.DOWNLOADING,
        ),
        _StatusScenario(
            VoiceId.QWEN,
            _RuntimeFacts().state(),
            (VoiceId.QWEN,),
            (VoiceId.QWEN,),
            VoiceState.DOWNLOADED,
        ),
        _StatusScenario(
            VoiceId.QWEN,
            _RuntimeFacts(download_failed=True).state(),
            (),
            (VoiceId.QWEN,),
            VoiceState.DOWNLOAD_FAILED,
        ),
        _StatusScenario(
            VoiceId.QWEN,
            _RuntimeFacts().state(),
            (),
            (VoiceId.QWEN,),
            VoiceState.NOT_DOWNLOADED,
        ),
        _StatusScenario(
            VoiceId.CHATTERBOX,
            _RuntimeFacts(
                selected=VoiceId.PIPER,
                ready=True,
                loading=True,
                pending=VoiceId.CHATTERBOX,
            ).state(),
            (VoiceId.CHATTERBOX,),
            (VoiceId.CHATTERBOX,),
            VoiceState.LOADING,
        ),
        pytest.param(
            _StatusScenario(
                VoiceId.CHATTERBOX,
                _RuntimeFacts().state(),
                (),
                (),
                VoiceState.UNAVAILABLE,
            ),
            id="512-line4-chatterbox-without-worker",
        ),
        pytest.param(
            _StatusScenario(
                VoiceId.CHATTERBOX,
                _RuntimeFacts().state(),
                (),
                (VoiceId.CHATTERBOX,),
                VoiceState.NOT_DOWNLOADED,
            ),
            id="512-line4-chatterbox-worker-without-weights",
        ),
    ],
)
def test_catalogue_preserves_runtime_precedence_before_download_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    scenario: _StatusScenario,
) -> None:
    installed_voices = {VoiceId.PIPER, *scenario.installed}
    monkeypatch.setattr(
        voices, "installed_voice_ids", lambda _settings: installed_voices
    )
    monkeypatch.setattr(
        voices,
        "provider_entrypoint_is_usable",
        lambda _root, _provider: scenario.voice in scenario.usable,
    )

    catalogue = statuses(
        Settings(
            voice_cache=tmp_path,
            huggingface_cache=tmp_path,
            provider_root=tmp_path,
            PRESENTATOR_RUNTIME_UID=os.geteuid(),
        ),
        scenario.runtime,
    )

    row = next(row for row in catalogue if row.id is scenario.voice)
    assert row.state is scenario.expected


def test_private_endpoint_reports_loading_and_active_from_the_shared_runtime(
    tmp_path: Path,
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


def test_status_refuses_the_whole_snapshot_when_cache_lookup_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
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

    state = VoiceRuntimeState(
        selected=None, ready=False, loading=False, pending=None, failed=None
    )
    with pytest.raises(OSError, match="cache unavailable"):
        statuses(settings, state)

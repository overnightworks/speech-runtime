"""The private voice endpoint reports evidence, never package support."""

import asyncio
import builtins
import importlib
import json
import logging
import os
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from presentator_speech_provider_contract import (
    MAGPIE_SAMPLE_RATE,
    VOXCPM_ARTIFACTS,
    VOXCPM_MODEL_ID,
    VOXCPM_REVISION,
    VOXCPM_SAMPLE_RATE,
)

from speech import voices
from speech.config import Settings
from speech.control import create_control_app
from speech.pcm import pcm_from_wav
from speech.selection import VoiceSelectionStore
from speech.service import (
    Runtime,
    RuntimeDependencies,
    SpeakingEngine,
    create_app,
)
from speech.speaking import speaking_for_voice
from speech.voices import (
    VoiceId,
    VoiceLoadOutcome,
    VoiceRecoveryKind,
    VoiceRuntimeState,
    VoiceState,
    statuses,
)
from tests.conftest import FakeHearing, FakeSpeaking
from tests.test_provider_process import _assert_reaped, _closed_provider, _observed
from tests.test_speak import (
    _ASGI_TIMEOUT_SECONDS,
    _response_body,
    _speak_once,
    _speak_response,
    _speak_scope,
    _wav_body,
)


def _captured_engine(runtime: Runtime) -> SpeakingEngine:
    admission = runtime.capture_speaking()
    assert admission is not None
    try:
        return admission.speaking
    finally:
        admission.close()


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
    for artifact in artifacts:
        path = snapshot / artifact
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    return snapshot


def _complete_voxcpm_snapshot(cache: Path) -> Path:
    snapshot = (
        cache
        / f"models--{VOXCPM_MODEL_ID.replace('/', '--')}"
        / "snapshots"
        / VOXCPM_REVISION
    )
    for artifact in VOXCPM_ARTIFACTS:
        path = snapshot / artifact
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
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
    for artifact in (model, codec):
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.touch()
    return model, codec


def test_qwen_requires_the_exact_snapshot_and_executable_and_has_a_truthful_preset(
    tmp_path,
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
    worker = provider_root / "qwen/.venv/bin/presentator-qwen-worker"
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
    assert engine.sample_rate == 24_000
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
    worker = provider_root / "voxcpm/.venv/bin/presentator-voxcpm-worker"
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
    assert engine.sample_rate == 48_000
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
    worker = provider_root / "magpie/.venv/bin/presentator-magpie-worker"
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
    assert engine.sample_rate == 22_050
    assert engine.streams is False


def test_voxcpm_catalogue_and_factory_import_no_provider_packages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider_root = tmp_path / "providers"
    cache = tmp_path / "hub"
    _complete_voxcpm_snapshot(cache)
    worker = provider_root / "voxcpm/.venv/bin/presentator-voxcpm-worker"
    worker.parent.mkdir(parents=True)
    worker.write_text("#!/bin/false\n", encoding="utf-8")
    worker.chmod(0o700)
    settings = Settings(
        provider_root=provider_root,
        huggingface_cache=cache,
        PRESENTATOR_RUNTIME_UID=os.geteuid(),
    )
    host = importlib.import_module("speech")
    marker = object()
    previous_attributes = {
        name: getattr(host, name, marker) for name in ("voices", "speaking")
    }
    previous_modules = {
        name: sys.modules.pop(name)
        for name in ("speech.voices", "speech.speaking")
        if name in sys.modules
    }
    original_import = builtins.__import__

    def guard_provider_import(
        name: str,
        global_namespace: dict[str, object] | None = None,
        local_namespace: dict[str, object] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> object:
        if name == "presentator_voxcpm" or name.startswith("presentator_voxcpm."):
            raise AssertionError
        if name == "voxcpm" or name.startswith("voxcpm."):
            raise AssertionError
        return original_import(name, global_namespace, local_namespace, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guard_provider_import)
    try:
        fresh_voices = importlib.import_module("speech.voices")
        fresh_speaking = importlib.import_module("speech.speaking")
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
        for name in ("speech.voices", "speech.speaking"):
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
    provider_root, cache = _closed_provider(tmp_path, provider_name="magpie")
    _complete_magpie_snapshots(cache)
    settings = Settings(
        provider_root=provider_root,
        huggingface_cache=cache,
        PRESENTATOR_RUNTIME_UID=os.geteuid(),
    )
    host = importlib.import_module("speech")
    marker = object()
    previous_attributes = {
        name: getattr(host, name, marker) for name in ("voices", "speaking")
    }
    previous_modules = {
        name: sys.modules.pop(name)
        for name in ("speech.voices", "speech.speaking")
        if name in sys.modules
    }
    original_import = builtins.__import__

    def guard_provider_import(
        name: str,
        global_namespace: dict[str, object] | None = None,
        local_namespace: dict[str, object] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> object:
        if name.split(".", 1)[0] in {"nemo", "torch", "presentator_magpie"}:
            raise AssertionError
        return original_import(name, global_namespace, local_namespace, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guard_provider_import)
    try:
        fresh_voices = importlib.import_module("speech.voices")
        fresh_speaking = importlib.import_module("speech.speaking")
        rows = fresh_voices.statuses(settings, _RuntimeFacts().state())
        magpie = next(row for row in rows if row.id is fresh_voices.VoiceId.MAGPIE)
        engine = fresh_speaking.speaking_for_voice(
            settings, fresh_voices.VoiceId.MAGPIE
        )
        assert magpie.state is fresh_voices.VoiceState.DOWNLOADED
        assert engine.sample_rate == MAGPIE_SAMPLE_RATE
    finally:
        for name in ("speech.voices", "speech.speaking"):
            sys.modules.pop(name, None)
        sys.modules.update(previous_modules)
        for name, value in previous_attributes.items():
            if value is marker:
                delattr(host, name)
            else:
                setattr(host, name, value)


async def _disconnect_after_child_receives_request(
    app: FastAPI, request_ids: Path, expected_request_count: int
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
                "body": json.dumps({"text": "wait-cancel", "language": "de"}).encode(),
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

    call = asyncio.create_task(app(_speak_scope(), receive, send))
    await asyncio.to_thread(
        _wait_for_request_count, request_ids, expected_request_count
    )
    await asyncio.wait_for(pcm_sent.wait(), timeout=_ASGI_TIMEOUT_SECONDS)
    disconnect.set()
    await asyncio.wait_for(delivered.wait(), timeout=_ASGI_TIMEOUT_SECONDS)
    await asyncio.wait_for(call, timeout=_ASGI_TIMEOUT_SECONDS)
    return messages


def _wait_for_request_count(request_ids: Path, expected_request_count: int) -> None:
    deadline = time.monotonic() + _ASGI_TIMEOUT_SECONDS
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
    provider_root, cache = _closed_provider(tmp_path, provider_name="voxcpm")
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
            assert response.status_code == 200
            assert rate == VOXCPM_SAMPLE_RATE
            assert pcm == b"\x00\x01"
        assert selection.read() is VoiceId.VOXCPM
        initial.close()

        restored = runtime()
        restored.load()
        public = create_app(runtime=restored)
        request_ids = cache / "request-ids"
        disconnected = asyncio.run(
            _disconnect_after_child_receives_request(public, request_ids, 3)
        )
        assert any(message["type"] == "http.response.start" for message in disconnected)
        assert (cache / "cancel-observed").read_text().splitlines() == ["1"]

        later = asyncio.run(_speak_once(public))
        rate, pcm = pcm_from_wav(_wav_body(later))
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
    provider_root, cache = _closed_provider(tmp_path, provider_name="magpie")
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
            assert response.status_code == 200
            assert rate == MAGPIE_SAMPLE_RATE
            assert pcm == b"\x00\x01"
        _, _, process_id, _ = _observed(cache)
        assert selection.read() is VoiceId.MAGPIE
        initial.close()

        restored = runtime()
        restored.load()
        public = create_app(runtime=restored)
        request_ids = cache / "request-ids"
        disconnected = asyncio.run(
            _disconnect_after_child_receives_request(public, request_ids, 3)
        )
        assert any(message["type"] == "http.response.start" for message in disconnected)
        assert (cache / "cancel-observed").read_text().splitlines() == ["1"]

        later = asyncio.run(_speak_once(public))
        rate, pcm = pcm_from_wav(_wav_body(later))
        assert rate == MAGPIE_SAMPLE_RATE
        assert pcm == b"\x00\x01"
        assert model.is_file()
        assert codec.is_file()

        assert restored.load_voice(VoiceId.PIPER) is VoiceLoadOutcome.ACTIVATED
        _assert_reaped(process_id)
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
    ],
)
def test_catalogue_preserves_runtime_precedence_before_download_evidence(
    tmp_path,
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


class _TrackedSpeaking(FakeSpeaking):
    def __init__(self, *, sample_rate: int = 22_050) -> None:
        super().__init__()
        self.sample_rate = sample_rate
        self.closed = 0

    def close(self) -> None:
        self.closed += 1


def _switching_runtime(
    tmp_path, old, target, selection: VoiceSelectionStore | None = None
) -> tuple[Runtime, VoiceSelectionStore]:
    old.retain_when_inactive = False
    store = selection or VoiceSelectionStore(tmp_path / "state", owner_uid=os.geteuid())
    store.write(VoiceId.CHATTERBOX)
    runtime = Runtime(
        old,
        FakeHearing(),
        dependencies=RuntimeDependencies(
            selection=store,
            engine_factory=lambda _voice: target,
            artifact_checker=lambda _voice: True,
            default_voice=VoiceId.CHATTERBOX,
        ),
    )
    return runtime, store


def test_deselecting_chatterbox_waits_for_an_admitted_talk_and_refuses_new_speech(
    tmp_path,
) -> None:
    committed = threading.Event()

    class ObservedSelection(VoiceSelectionStore):
        def write(self, voice: VoiceId) -> None:
            super().write(voice)
            if voice is VoiceId.PIPER:
                committed.set()

    old = _TrackedSpeaking(sample_rate=24_000)
    target = _TrackedSpeaking()
    runtime, _store = _switching_runtime(
        tmp_path,
        old,
        target,
        ObservedSelection(tmp_path / "state", owner_uid=os.geteuid()),
    )
    response = _speak_response(runtime)
    outcomes = []
    switching = threading.Thread(
        target=lambda: outcomes.append(runtime.load_voice(VoiceId.PIPER))
    )
    switching.start()
    assert committed.wait(timeout=5)

    with pytest.raises(HTTPException) as refused:
        _speak_response(runtime)
    assert refused.value.status_code == 503
    assert switching.is_alive()

    rate, pcm = pcm_from_wav(_response_body(response))
    switching.join(timeout=5)
    assert not switching.is_alive()
    assert rate == 24_000
    assert pcm
    assert outcomes == [VoiceLoadOutcome.ACTIVATED]
    assert old.closed == 1
    assert _captured_engine(runtime) is target


def test_shutdown_before_chatterbox_detach_transfers_old_and_pending_cleanup_once(
    tmp_path,
) -> None:
    committed = threading.Event()

    class ObservedSelection(VoiceSelectionStore):
        def write(self, voice: VoiceId) -> None:
            super().write(voice)
            if voice is VoiceId.PIPER:
                committed.set()

    old = _TrackedSpeaking(sample_rate=24_000)
    target = _TrackedSpeaking()
    runtime, store = _switching_runtime(
        tmp_path,
        old,
        target,
        ObservedSelection(tmp_path / "state", owner_uid=os.geteuid()),
    )
    response = _speak_response(runtime)
    outcomes = []
    switching = threading.Thread(
        target=lambda: outcomes.append(runtime.load_voice(VoiceId.PIPER))
    )
    switching.start()
    assert committed.wait(timeout=5)

    runtime.begin_shutdown()
    closing = threading.Thread(target=runtime.close)
    closing.start()
    _response_body(response)
    switching.join(timeout=5)
    closing.join(timeout=5)

    assert not switching.is_alive()
    assert not closing.is_alive()
    assert store.read() is VoiceId.PIPER
    assert outcomes == [VoiceLoadOutcome.NOT_ACTIVATED]
    assert runtime.capture_speaking() is None
    assert old.closed == 1
    assert target.closed == 1


def test_shutdown_during_chatterbox_close_never_publishes_or_loses_an_engine(
    tmp_path,
) -> None:
    close_started = threading.Event()
    release_close = threading.Event()

    class HeldCloseSpeaking(_TrackedSpeaking):
        def close(self) -> None:
            self.closed += 1
            close_started.set()
            assert release_close.wait(timeout=5)

    old = HeldCloseSpeaking(sample_rate=24_000)
    target = _TrackedSpeaking()
    runtime, store = _switching_runtime(tmp_path, old, target)
    outcomes = []
    switching = threading.Thread(
        target=lambda: outcomes.append(runtime.load_voice(VoiceId.PIPER))
    )
    switching.start()
    assert close_started.wait(timeout=5)

    runtime.begin_shutdown()
    closing = threading.Thread(target=runtime.close)
    closing.start()
    release_close.set()
    switching.join(timeout=5)
    closing.join(timeout=5)

    assert not switching.is_alive()
    assert not closing.is_alive()
    assert store.read() is VoiceId.PIPER
    assert outcomes == [VoiceLoadOutcome.NOT_ACTIVATED]
    assert runtime.capture_speaking() is None
    assert old.closed == 1
    assert target.closed == 1


def _close_failure_after_selection_commit_is_fatal_and_not_a_load_outcome(
    tmp_path,
) -> None:
    class FailingCloseSpeaking(_TrackedSpeaking):
        def close(self) -> None:
            self.closed += 1
            if self.closed == 1:
                message = "provider close failed"
                raise RuntimeError(message)

    fatal = threading.Event()
    old = FailingCloseSpeaking(sample_rate=24_000)
    target = _TrackedSpeaking()
    runtime, store = _switching_runtime(tmp_path, old, target)
    runtime.set_fatal_callback(fatal.set)

    with pytest.raises(RuntimeError, match="provider close failed"):
        runtime.load_voice(VoiceId.PIPER)

    assert fatal.is_set()
    assert store.read() is VoiceId.PIPER
    assert runtime.capture_speaking() is None
    runtime.close()
    assert old.closed == 2
    assert target.closed == 1


test_chatterbox_close_failure_after_selection_commit_is_fatal_and_not_a_load_outcome = (
    _close_failure_after_selection_commit_is_fatal_and_not_a_load_outcome
)


def test_pre_replace_failure_preserves_active_and_stored_chatterbox(tmp_path) -> None:
    settings = Settings(
        voice_cache=tmp_path,
        huggingface_cache=tmp_path,
        PRESENTATOR_RUNTIME_UID=os.geteuid(),
    )
    old = _TrackedSpeaking(sample_rate=24_000)
    target = FakeSpeaking(ready=False, fail=True)
    runtime, store = _switching_runtime(tmp_path, old, target)

    assert runtime.load_voice(VoiceId.PIPER) is VoiceLoadOutcome.NOT_ACTIVATED

    assert _captured_engine(runtime) is old
    assert store.read() is VoiceId.CHATTERBOX
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

    assert _captured_engine(first) is engines[VoiceId.PIPER]
    assert _captured_engine(restored) is engines[VoiceId.CHATTERBOX]


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

    assert _captured_engine(runtime) is active
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

    assert _captured_engine(runtime) is engine
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
    assert _captured_engine(runtime) is engines[VoiceId.PIPER]
    assert runtime.load_voice(VoiceId.CHATTERBOX) is VoiceLoadOutcome.ACTIVATED
    assert _captured_engine(runtime) is engines[VoiceId.CHATTERBOX]
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
    engine.retain_when_inactive = False
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


def test_nonstreaming_provider_failure_reconciles_and_allows_a_fresh_load(
    tmp_path,
) -> None:
    class ClosableSpeaking(FakeSpeaking):
        retain_when_inactive = False

        def __init__(self) -> None:
            super().__init__()
            self.closed = 0

        def close(self) -> None:
            self.closed += 1

    failed = ClosableSpeaking()
    replacement = ClosableSpeaking()
    engines = iter((failed, replacement))
    store = VoiceSelectionStore(tmp_path / "state", owner_uid=os.geteuid())
    runtime = Runtime(
        None,
        FakeHearing(),
        dependencies=RuntimeDependencies(
            selection=store,
            engine_factory=lambda _voice: next(engines),
            artifact_checker=lambda _voice: True,
        ),
    )
    assert runtime.load_voice(VoiceId.QWEN) is VoiceLoadOutcome.ACTIVATED
    failed.ready = False

    snapshot = runtime.snapshot(Settings(voice_cache=tmp_path))

    assert failed.closed == 1
    assert snapshot.recovery is not None
    assert snapshot.recovery.kind is VoiceRecoveryKind.LOAD_FAILED
    assert snapshot.recovery.voice is VoiceId.QWEN
    assert store.read() is VoiceId.QWEN
    assert runtime.capture_speaking() is None
    assert runtime.load_voice(VoiceId.QWEN) is VoiceLoadOutcome.ACTIVATED
    assert _captured_engine(runtime) is replacement


def test_switching_releases_any_provider_process_before_load_returns(tmp_path) -> None:
    old = _TrackedSpeaking(sample_rate=24_000)
    old.retain_when_inactive = False
    target = _TrackedSpeaking()
    store = VoiceSelectionStore(tmp_path / "state", owner_uid=os.geteuid())
    store.write(VoiceId.QWEN)
    runtime = Runtime(
        old,
        FakeHearing(),
        dependencies=RuntimeDependencies(
            selection=store,
            engine_factory=lambda _voice: target,
            artifact_checker=lambda _voice: True,
            default_voice=VoiceId.QWEN,
        ),
    )

    assert runtime.load_voice(VoiceId.PIPER) is VoiceLoadOutcome.ACTIVATED

    assert old.closed == 1
    assert _captured_engine(runtime) is target


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
    failed.retain_when_inactive = False
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
    assert _captured_engine(runtime) is replacement

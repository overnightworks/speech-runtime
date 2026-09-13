"""The Qwen adapter stays local, closed, and bounds completed PCM."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from presentator_speech_provider_contract import MAX_PCM_BYTES

from presentator_qwen import model


class _GeneratedModel:
    def __init__(self, waveform: np.ndarray, rate: int = 24_000) -> None:
        self._waveform = waveform
        self._rate = rate
        self.calls: list[dict[str, object]] = []

    def generate_custom_voice(
        self, **arguments: object
    ) -> tuple[list[np.ndarray], int]:
        self.calls.append(arguments)
        return [self._waveform], self._rate


def test_load_uses_the_pinned_local_snapshot_and_closed_cuda_mapping(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    observed: dict[str, object] = {}
    checkpoint = tmp_path / "snapshot"
    loaded = object()

    def snapshot_download(**arguments: object) -> str:
        observed["snapshot"] = arguments
        return str(checkpoint)

    class FakeQwen:
        @classmethod
        def from_pretrained(cls, path: str, **arguments: object) -> object:
            observed["model"] = (path, arguments)
            return loaded

    monkeypatch.setattr("huggingface_hub.snapshot_download", snapshot_download)
    monkeypatch.setitem(
        sys.modules, "qwen_tts", SimpleNamespace(Qwen3TTSModel=FakeQwen)
    )
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(bfloat16="bfloat16"))

    assert model.load_model("cuda", tmp_path) is loaded
    assert observed == {
        "snapshot": {
            "repo_id": "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
            "repo_type": "model",
            "revision": "85e237c12c027371202489a0ec509ded67b5e4b5",
            "allow_patterns": model.QWEN_ARTIFACTS,
            "local_files_only": True,
            "cache_dir": tmp_path,
        },
        "model": (
            str(checkpoint),
            {
                "device_map": "cuda:0",
                "dtype": "bfloat16",
                "local_files_only": True,
            },
        ),
    }


@pytest.mark.parametrize("device", ["cpu", "cuda:1", "mps"])
def test_load_rejects_unsupported_devices_before_sdk_import(
    tmp_path: Path, device: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delitem(sys.modules, "qwen_tts", raising=False)

    with pytest.raises(ValueError, match="unsupported Qwen device"):
        model.load_model(device, tmp_path)

    assert "qwen_tts" not in sys.modules


def test_complete_waveform_is_clipped_and_split_into_even_bounded_pcm() -> None:
    waveform = np.linspace(-2.0, 2.0, (MAX_PCM_BYTES // 2) + 17)
    generated = _GeneratedModel(waveform)

    chunks = list(model.pcm_chunks(generated, "Hallo", "de", speaker="Ryan"))

    assert generated.calls == [
        {"text": "Hallo", "language": "German", "speaker": "Ryan"}
    ]
    assert chunks
    assert all(len(chunk) <= MAX_PCM_BYTES and len(chunk) % 2 == 0 for chunk in chunks)
    expected = np.clip(waveform * 32767.0, -32768, 32767).astype("<i2").tobytes()
    assert b"".join(chunks) == expected


def test_english_is_explicit_and_instruction_is_absent() -> None:
    generated = _GeneratedModel(np.array([0.25]))

    assert list(model.pcm_chunks(generated, "Hello", "en", speaker="Sohee"))

    assert generated.calls == [
        {"text": "Hello", "language": "English", "speaker": "Sohee"}
    ]


def test_generation_rejects_an_unexpected_sample_rate() -> None:
    generated = _GeneratedModel(np.array([0.25]), rate=16_000)

    with pytest.raises(ValueError, match="Qwen sample rate"):
        list(model.pcm_chunks(generated, "Hallo", "de", speaker="Ryan"))

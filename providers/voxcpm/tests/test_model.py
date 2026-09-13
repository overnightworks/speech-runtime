"""The VoxCPM adapter uses only the pinned local SDK seam."""

from __future__ import annotations

import sys
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from presentator_speech_provider_contract import MAX_PCM_BYTES

from presentator_voxcpm import model


@pytest.mark.parametrize("sample_rate", [48_000, 24_000])
def test_load_uses_the_pinned_local_snapshot_and_validates_native_rate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sample_rate: int
) -> None:
    observed: dict[str, object] = {}
    loaded = SimpleNamespace(tts_model=SimpleNamespace(sample_rate=sample_rate))

    def snapshot_download(**arguments: object) -> str:
        observed["snapshot"] = arguments
        return str(tmp_path / "snapshot")

    class FakeVoxCPM:
        @classmethod
        def from_pretrained(cls, path: str, **arguments: object) -> object:
            observed["model"] = (path, arguments)
            return loaded

    monkeypatch.setattr("huggingface_hub.snapshot_download", snapshot_download)
    monkeypatch.setitem(sys.modules, "voxcpm", SimpleNamespace(VoxCPM=FakeVoxCPM))

    if sample_rate == 48_000:
        assert model.load_model("cuda", tmp_path) is loaded
    else:
        with pytest.raises(ValueError, match="unexpected VoxCPM sample rate"):
            model.load_model("cuda", tmp_path)
    assert observed["snapshot"] == {
        "repo_id": "openbmb/VoxCPM2",
        "repo_type": "model",
        "revision": "32279effe8c19989596f05d353d1447f51d9e915",
        "allow_patterns": model.VOXCPM_ARTIFACTS,
        "local_files_only": True,
        "cache_dir": tmp_path,
    }
    assert observed["model"] == (
        str(tmp_path / "snapshot"),
        {"load_denoiser": False, "optimize": True, "device": "cuda"},
    )


@pytest.mark.parametrize(("language", "text"), [("de", "Hallo"), ("en", "Hello")])
def test_streaming_preserves_text_clips_pcm_and_closes_the_upstream_generator(
    language: str, text: str
) -> None:
    closed = False
    calls: list[tuple[str, dict[str, object]]] = []
    samples = np.concatenate(
        (
            np.array((-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0)),
            np.zeros((MAX_PCM_BYTES // 2) - 6),
        )
    )

    class Generated:
        def __iter__(self) -> Iterator[np.ndarray]:
            nonlocal closed
            try:
                yield samples
            finally:
                closed = True

    class FakeModel:
        def generate_streaming(self, text: str, **arguments: object) -> Generated:
            calls.append((text, arguments))
            return Generated()

    chunks = list(model.pcm_chunks(FakeModel(), text, language))

    assert closed is True
    assert chunks == [
        b"\x00\x80\x01\x80\x01\xc0\x00\x00\xff\x3f\xff\x7f\xff\x7f"
        + (b"\x00\x00" * ((MAX_PCM_BYTES // 2) - 7)),
        b"\x00\x00",
    ]
    assert all(len(chunk) <= MAX_PCM_BYTES and len(chunk) % 2 == 0 for chunk in chunks)
    assert calls == [
        (
            text,
            {
                "cfg_value": 2.0,
                "inference_timesteps": 10,
                "normalize": False,
                "denoise": False,
                "retry_badcase": False,
            },
        )
    ]


def test_closing_pcm_iterator_closes_the_upstream_generator_early() -> None:
    closed = False
    second_chunk_started = False

    class Generated:
        def __iter__(self) -> Iterator[np.ndarray]:
            nonlocal closed, second_chunk_started
            try:
                yield np.array((0.0,))
                second_chunk_started = True
                yield np.array((0.0,))
            finally:
                closed = True

    class FakeModel:
        def generate_streaming(self, *_args: object, **_kwargs: object) -> Generated:
            return Generated()

    chunks = model.pcm_chunks(FakeModel(), "Hallo", "de")
    assert next(chunks) == b"\x00\x00"
    chunks.close()

    assert closed is True
    assert second_chunk_started is False


def test_streaming_refuses_non_protocol_languages() -> None:
    with pytest.raises(ValueError, match="unsupported VoxCPM language"):
        list(model.pcm_chunks(object(), "Hallo", "fr"))

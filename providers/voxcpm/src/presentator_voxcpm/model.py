"""The pinned VoxCPM2 adapter, local-cache only."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
from presentator_speech_provider_contract import (
    MAX_PCM_BYTES,
    VOXCPM_ARTIFACTS,
    VOXCPM_MODEL_ID,
    VOXCPM_REVISION,
    VOXCPM_SAMPLE_RATE,
)


def load_model(device: str, cache: Path) -> object:
    """Load the exact local VoxCPM2 snapshot on the configured device."""
    from huggingface_hub import snapshot_download
    from voxcpm import VoxCPM

    checkpoint = snapshot_download(
        repo_id=VOXCPM_MODEL_ID,
        repo_type="model",
        revision=VOXCPM_REVISION,
        allow_patterns=VOXCPM_ARTIFACTS,
        local_files_only=True,
        cache_dir=cache,
    )
    model = VoxCPM.from_pretrained(
        checkpoint, load_denoiser=False, optimize=True, device=device
    )
    if model.tts_model.sample_rate != VOXCPM_SAMPLE_RATE:
        message = "unexpected VoxCPM sample rate"
        raise ValueError(message)
    return model


def pcm_chunks(model: object, text: str, language: str) -> Iterator[bytes]:
    """Stream capped signed PCM while preserving the upstream cleanup boundary."""
    if language not in {"de", "en"}:
        message = "unsupported VoxCPM language"
        raise ValueError(message)
    generated = model.generate_streaming(
        text,
        cfg_value=2.0,
        inference_timesteps=10,
        normalize=False,
        denoise=False,
        retry_badcase=False,
    )
    try:
        for waveform in generated:
            pcm = _float_to_pcm(waveform)
            for start in range(0, len(pcm), MAX_PCM_BYTES):
                yield pcm[start : start + MAX_PCM_BYTES]
    finally:
        close = getattr(generated, "close", None)
        if callable(close):
            close()


def _float_to_pcm(waveform: object) -> bytes:
    samples = waveform
    detach = getattr(samples, "detach", None)
    if callable(detach):
        samples = detach()
    to_cpu = getattr(samples, "cpu", None)
    if callable(to_cpu):
        samples = to_cpu()
    as_numpy = getattr(samples, "numpy", None)
    if callable(as_numpy):
        samples = as_numpy()
    clipped = np.clip(np.asarray(samples, dtype=np.float64) * 32767.0, -32768, 32767)
    return clipped.astype("<i2").tobytes()

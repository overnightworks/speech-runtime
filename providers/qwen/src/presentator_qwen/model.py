"""The pinned Qwen3-TTS 0.6B adapter, local-cache only."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
from presentator_speech_provider_contract import (
    MAX_PCM_BYTES,
    QWEN_ARTIFACTS,
    QWEN_MODEL_ID,
    QWEN_REVISION,
    QWEN_SAMPLE_RATE,
    QwenSpeaker,
)


def load_model(device: str, cache: Path) -> object:
    """Load the pinned custom-voice model from the local Hub cache."""
    if device != "cuda":
        message = "unsupported Qwen device"
        raise ValueError(message)

    import torch
    from huggingface_hub import snapshot_download
    from qwen_tts import Qwen3TTSModel

    checkpoint = snapshot_download(
        repo_id=QWEN_MODEL_ID,
        repo_type="model",
        revision=QWEN_REVISION,
        allow_patterns=QWEN_ARTIFACTS,
        local_files_only=True,
        cache_dir=cache,
    )
    return Qwen3TTSModel.from_pretrained(
        checkpoint,
        device_map="cuda:0",
        dtype=torch.bfloat16,
        local_files_only=True,
    )


def pcm_chunks(
    model: object,
    text: str,
    language: str,
    *,
    speaker: QwenSpeaker | str,
) -> Iterator[bytes]:
    """Generate one complete waveform and yield bounded signed PCM frames."""
    language_name = {"de": "German", "en": "English"}.get(language)
    if language_name is None:
        message = "unsupported Qwen language"
        raise ValueError(message)
    selected_speaker = (
        speaker.value
        if isinstance(speaker, QwenSpeaker)
        else QwenSpeaker(speaker).value
    )
    waveforms, sample_rate = model.generate_custom_voice(
        text=text,
        language=language_name,
        speaker=selected_speaker,
    )
    if sample_rate != QWEN_SAMPLE_RATE:
        message = "unexpected Qwen sample rate"
        raise ValueError(message)
    pcm = _float_to_pcm(waveforms[0])
    for start in range(0, len(pcm), MAX_PCM_BYTES):
        yield pcm[start : start + MAX_PCM_BYTES]


def _float_to_pcm(waveform: np.ndarray) -> bytes:
    clipped = np.clip(np.asarray(waveform, dtype=np.float64) * 32767.0, -32768, 32767)
    return clipped.astype("<i2").tobytes()

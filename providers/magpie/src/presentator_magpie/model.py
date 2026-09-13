"""The exact cached NVIDIA Magpie v2607 adapter."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from nemo.collections.tts.models import MagpieTTSModel
from omegaconf import DictConfig
from presentator_speech_provider_contract import (
    MAGPIE_CODEC_FILENAME,
    MAGPIE_CODEC_ID,
    MAGPIE_CODEC_REVISION,
    MAGPIE_MODEL_FILENAME,
    MAGPIE_MODEL_ID,
    MAGPIE_MODEL_REVISION,
    MAGPIE_SAMPLE_RATE,
    MAX_PCM_BYTES,
    MagpieSpeaker,
)

_SPEAKER_INDICES = {
    MagpieSpeaker.ARIA: 0,
    MagpieSpeaker.JASON: 1,
    MagpieSpeaker.JOHN: 2,
    MagpieSpeaker.LEO: 3,
    MagpieSpeaker.SOFIA: 4,
}
_MONO_WAVEFORM_DIMENSIONS = 2
_ONE_CHANNEL = 1
_CONFIGURATION_ERROR = "Magpie restore did not return a configuration"
_SAMPLE_RATE_ERROR = "unexpected Magpie sample rate"
_LANGUAGE_ERROR = "unsupported Magpie language"
_WAVEFORM_ERROR = "malformed Magpie waveform"


def load_model(device: str, cache: Path) -> MagpieTTSModel:
    """Restore the two exact local archives and verify their native rate."""
    model_path = Path(
        hf_hub_download(
            repo_id=MAGPIE_MODEL_ID,
            filename=MAGPIE_MODEL_FILENAME,
            revision=MAGPIE_MODEL_REVISION,
            cache_dir=cache,
            local_files_only=True,
        )
    )
    codec_path = Path(
        hf_hub_download(
            repo_id=MAGPIE_CODEC_ID,
            filename=MAGPIE_CODEC_FILENAME,
            revision=MAGPIE_CODEC_REVISION,
            cache_dir=cache,
            local_files_only=True,
        )
    )
    config = MagpieTTSModel.restore_from(model_path, return_config=True)
    if not isinstance(config, DictConfig):
        raise TypeError(_CONFIGURATION_ERROR)
    config.codecmodel_path = str(codec_path)
    model = MagpieTTSModel.restore_from(
        restore_path=model_path,
        override_config_path=config,
        map_location=torch.device("cpu"),
    )
    model.to(device)
    model.eval()
    if model.output_sample_rate != MAGPIE_SAMPLE_RATE:
        raise ValueError(_SAMPLE_RATE_ERROR)
    return model


def pcm_chunks(
    model: MagpieTTSModel, text: str, language: str, *, speaker: MagpieSpeaker
) -> Iterator[bytes]:
    """Generate one bounded signed PCM response in a fixed supported voice."""
    if language not in {"de", "en"}:
        raise ValueError(_LANGUAGE_ERROR)
    audio, audio_length = model.do_tts(
        transcript=text,
        language=language,
        apply_TN=False,
        use_cfg=True,
        speaker_index=_SPEAKER_INDICES[speaker],
    )
    sample_count = int(audio_length.squeeze().item())
    if (
        audio.ndim != _MONO_WAVEFORM_DIMENSIONS
        or audio.shape[0] != _ONE_CHANNEL
        or not 0 < sample_count <= audio.shape[1]
    ):
        raise ValueError(_WAVEFORM_ERROR)
    waveform = audio[0, :sample_count]
    if not torch.isfinite(waveform).all():
        raise ValueError(_WAVEFORM_ERROR)
    pcm = _float_to_pcm(waveform)
    for start in range(0, len(pcm), MAX_PCM_BYTES):
        yield pcm[start : start + MAX_PCM_BYTES]


def _float_to_pcm(waveform: torch.Tensor) -> bytes:
    samples = waveform.detach().cpu().numpy()
    clipped = np.clip(np.asarray(samples, dtype=np.float64) * 32767.0, -32768, 32767)
    return clipped.astype("<i2").tobytes()

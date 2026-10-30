"""The closed voice catalogue and its side-effect-free artifact check."""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING

from huggingface_hub import try_to_load_from_cache
from speech_runtime_provider_contract import (
    CHATTERBOX_ARTIFACTS,
    CHATTERBOX_REVISION,
    MAGPIE_CODEC_FILENAME,
    MAGPIE_CODEC_ID,
    MAGPIE_CODEC_REVISION,
    MAGPIE_MODEL_FILENAME,
    MAGPIE_MODEL_ID,
    MAGPIE_MODEL_REVISION,
    QWEN_ARTIFACTS,
    QWEN_MODEL_ID,
    QWEN_REVISION,
    VOXCPM_ARTIFACTS,
    VOXCPM_MODEL_ID,
    VOXCPM_REVISION,
)

if TYPE_CHECKING:
    from pathlib import Path

    from speech_runtime_service.config import Settings

DEFAULT_SPEAKING_MODEL = "de_DE-thorsten-medium"
CHATTERBOX_SPEAKING_MODEL = "ResembleAI/chatterbox"


class VoiceId(StrEnum):
    """The closed catalogue of voices the service can speak with."""

    PIPER = "piper"
    CHATTERBOX = "chatterbox"
    QWEN = "qwen3-tts-0.6b"
    VOXCPM = "voxcpm2"
    MAGPIE = "nvidia-magpie"


_VOICES_BY_MODEL: dict[str, VoiceId] = {
    DEFAULT_SPEAKING_MODEL: VoiceId.PIPER,
    CHATTERBOX_SPEAKING_MODEL: VoiceId.CHATTERBOX,
    QWEN_MODEL_ID: VoiceId.QWEN,
    VOXCPM_MODEL_ID: VoiceId.VOXCPM,
    MAGPIE_MODEL_ID: VoiceId.MAGPIE,
}


def voice_for_model(model: str) -> VoiceId:
    """The voice whose speaking model is `model`; a model outside it is a KeyError."""
    return _VOICES_BY_MODEL[model]


def installed_voice_ids(settings: Settings) -> frozenset[VoiceId]:
    """Inspect only exact artifacts; this does not read credentials or the network."""
    installed: set[VoiceId] = set()
    if _piper_is_present(settings.voice_cache, DEFAULT_SPEAKING_MODEL):
        installed.add(VoiceId.PIPER)
    if _chatterbox_is_present(settings.huggingface_cache):
        installed.add(VoiceId.CHATTERBOX)
    if _qwen_is_present(settings.huggingface_cache):
        installed.add(VoiceId.QWEN)
    if _voxcpm_is_present(settings.huggingface_cache):
        installed.add(VoiceId.VOXCPM)
    if _magpie_is_present(settings.huggingface_cache):
        installed.add(VoiceId.MAGPIE)
    return frozenset(installed)


def _piper_is_present(cache: Path, model: str) -> bool:
    return all(
        (cache / f"{model}{suffix}").is_file() for suffix in (".onnx", ".onnx.json")
    )


def _chatterbox_is_present(cache: Path) -> bool:
    return all(
        isinstance(
            try_to_load_from_cache(
                CHATTERBOX_SPEAKING_MODEL,
                filename,
                cache_dir=cache,
                revision=CHATTERBOX_REVISION,
            ),
            str,
        )
        for filename in CHATTERBOX_ARTIFACTS
    )


def _qwen_is_present(cache: Path) -> bool:
    return all(
        isinstance(
            try_to_load_from_cache(
                QWEN_MODEL_ID,
                filename,
                cache_dir=cache,
                revision=QWEN_REVISION,
            ),
            str,
        )
        for filename in QWEN_ARTIFACTS
    )


def _voxcpm_is_present(cache: Path) -> bool:
    return all(
        isinstance(
            try_to_load_from_cache(
                VOXCPM_MODEL_ID,
                filename,
                cache_dir=cache,
                revision=VOXCPM_REVISION,
            ),
            str,
        )
        for filename in VOXCPM_ARTIFACTS
    )


def _magpie_is_present(cache: Path) -> bool:
    return all(
        isinstance(
            try_to_load_from_cache(
                repository,
                filename,
                cache_dir=cache,
                revision=revision,
            ),
            str,
        )
        for repository, filename, revision in (
            (MAGPIE_MODEL_ID, MAGPIE_MODEL_FILENAME, MAGPIE_MODEL_REVISION),
            (MAGPIE_CODEC_ID, MAGPIE_CODEC_FILENAME, MAGPIE_CODEC_REVISION),
        )
    )

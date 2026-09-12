"""The closed voice catalogue and its side-effect-free local status check."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

from huggingface_hub import try_to_load_from_cache

from speech.chatterbox import CHATTERBOX_ARTIFACTS, CHATTERBOX_REVISION
from speech.config import CHATTERBOX_SPEAKING_MODEL, DEFAULT_SPEAKING_MODEL

if TYPE_CHECKING:
    from pathlib import Path

    from speech.config import Settings


class VoiceId(StrEnum):
    """The only models the administration page may name."""

    PIPER = "piper"
    CHATTERBOX = "chatterbox"
    QWEN = "qwen3-tts-0.6b"
    VOXCPM = "voxcpm2"
    MAGPIE = "nvidia-magpie"


class VoiceState(StrEnum):
    """A status whose meaning is established by local evidence."""

    ACTIVE = "active"
    LOADING = "loading"
    DOWNLOADED = "downloaded"
    NOT_DOWNLOADED = "not_downloaded"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True, slots=True, kw_only=True)
class VoiceStatus:
    """One catalogue row the private endpoint may reveal."""

    id: VoiceId
    name: str
    language: str
    state: VoiceState


def statuses(
    settings: Settings, *, ready: bool, loading: bool
) -> tuple[VoiceStatus, ...]:
    """Report the complete catalogue, refusing to guess after a failed lookup."""
    installed = _installed(settings)
    configured = _configured_id(settings.speaking_model)
    if configured is None:
        message = "the configured speaking model is not in the catalogue"
        raise ValueError(message)
    return tuple(
        VoiceStatus(
            id=voice_id,
            name=name,
            language=language,
            state=_state(
                voice_id,
                configured=configured,
                installed=installed,
                ready=ready,
                loading=loading,
            ),
        )
        for voice_id, name, language in _CATALOGUE
    )


_CATALOGUE = (
    (VoiceId.PIPER, "Piper", "German — Thorsten voice"),
    (VoiceId.CHATTERBOX, "Chatterbox", "Multilingual"),
    (VoiceId.QWEN, "Qwen3-TTS 0.6B", "Not yet available"),
    (VoiceId.VOXCPM, "VoxCPM2", "Not yet available"),
    (VoiceId.MAGPIE, "NVIDIA Magpie", "Not yet available"),
)


def _configured_id(model: str) -> VoiceId | None:
    if model == DEFAULT_SPEAKING_MODEL:
        return VoiceId.PIPER
    if model == CHATTERBOX_SPEAKING_MODEL:
        return VoiceId.CHATTERBOX
    return None


def _installed(settings: Settings) -> frozenset[VoiceId]:
    """Inspect only exact artifacts; this does not read credentials or the network."""
    installed: set[VoiceId] = set()
    if _piper_is_present(settings.voice_cache, DEFAULT_SPEAKING_MODEL):
        installed.add(VoiceId.PIPER)
    if _chatterbox_is_present(settings.huggingface_cache):
        installed.add(VoiceId.CHATTERBOX)
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


def _state(
    voice_id: VoiceId,
    *,
    configured: VoiceId,
    installed: frozenset[VoiceId],
    ready: bool,
    loading: bool,
) -> VoiceState:
    if voice_id in {VoiceId.QWEN, VoiceId.VOXCPM, VoiceId.MAGPIE}:
        return VoiceState.UNAVAILABLE
    if voice_id is configured and ready:
        return VoiceState.ACTIVE
    if voice_id is configured and loading:
        return VoiceState.LOADING
    if voice_id in installed:
        return VoiceState.DOWNLOADED
    return VoiceState.NOT_DOWNLOADED

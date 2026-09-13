"""The closed voice catalogue and its side-effect-free local status check."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

from huggingface_hub import try_to_load_from_cache
from presentator_chatterbox_contract import CHATTERBOX_ARTIFACTS, CHATTERBOX_REVISION

from speech.chatterbox import chatterbox_entrypoint_is_usable
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
    FAILED = "failed"
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


class VoiceRecoveryKind(StrEnum):
    """Why speaking currently has no active engine."""

    INVALID_SELECTION = "invalid_selection"
    LOAD_FAILED = "load_failed"
    DURABILITY_UNCONFIRMED = "durability_unconfirmed"


@dataclass(frozen=True, slots=True, kw_only=True)
class VoiceRecovery:
    """Typed, non-sensitive recovery detail for the private snapshot."""

    kind: VoiceRecoveryKind
    voice: VoiceId | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class VoiceSnapshot:
    """The complete private status and any recoverable selection fact."""

    voices: tuple[VoiceStatus, ...]
    recovery: VoiceRecovery | None


class VoiceLoadOutcome(StrEnum):
    """The only outcomes a Load caller receives."""

    ACTIVATED = "activated"
    ACTIVATED_DURABILITY_UNCONFIRMED = "activated_durability_unconfirmed"
    NOT_ACTIVATED = "not_activated"


@dataclass(frozen=True, slots=True, kw_only=True)
class VoiceRuntimeState:
    """The runtime facts that determine the closed catalogue's row states."""

    selected: VoiceId | None
    ready: bool
    loading: bool
    pending: VoiceId | None
    failed: VoiceId | None


def statuses(settings: Settings, runtime: VoiceRuntimeState) -> tuple[VoiceStatus, ...]:
    """Report the complete catalogue, refusing to guess after a failed lookup."""
    installed = installed_voice_ids(settings)
    chatterbox_usable = chatterbox_entrypoint_is_usable(settings.provider_root)
    return tuple(
        VoiceStatus(
            id=voice_id,
            name=name,
            language=language,
            state=_state(
                voice_id,
                runtime=runtime,
                installed=installed,
                chatterbox_usable=chatterbox_usable,
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


def installed_voice_ids(settings: Settings) -> frozenset[VoiceId]:
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
    runtime: VoiceRuntimeState,
    installed: frozenset[VoiceId],
    chatterbox_usable: bool,
) -> VoiceState:
    if voice_id in {VoiceId.QWEN, VoiceId.VOXCPM, VoiceId.MAGPIE}:
        return VoiceState.UNAVAILABLE
    if voice_id is runtime.failed:
        return VoiceState.FAILED
    if voice_id is runtime.selected and runtime.ready:
        return VoiceState.ACTIVE
    if runtime.loading and voice_id is (runtime.pending or runtime.selected):
        return VoiceState.LOADING
    provider_usable = voice_id is not VoiceId.CHATTERBOX or chatterbox_usable
    if voice_id not in installed:
        return VoiceState.NOT_DOWNLOADED
    return VoiceState.DOWNLOADED if provider_usable else VoiceState.UNAVAILABLE

"""The hearing models the service knows: where they live and which revision is meant."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from huggingface_hub import snapshot_download, try_to_load_from_cache

_CONFIGURATION_FILE = "config.json"


class HearingModel(StrEnum):
    """The closed catalogue of hearing models."""

    LARGE_V3 = "large-v3"
    SMALL = "small"


@dataclass(frozen=True, slots=True)
class _Artifacts:
    repository: str
    revision: str
    filenames: tuple[str, ...]


_ARTIFACTS: dict[HearingModel, _Artifacts] = {
    HearingModel.LARGE_V3: _Artifacts(
        "Systran/faster-whisper-large-v3",
        "edaa852ec7e145841d8ffdb056a99866b5f0a478",
        (
            _CONFIGURATION_FILE,
            "model.bin",
            "preprocessor_config.json",
            "tokenizer.json",
            "vocabulary.json",
        ),
    ),
    HearingModel.SMALL: _Artifacts(
        "Systran/faster-whisper-small",
        "536b0662742c02347bc0e980a01041f333bce120",
        (_CONFIGURATION_FILE, "model.bin", "tokenizer.json", "vocabulary.txt"),
    ),
}


class HearingModelMissingError(RuntimeError):
    """The model's pinned files are not in the cache."""


def model_directory(model: HearingModel, cache: Path) -> Path:
    """The cached directory of the pinned `model`, found without the network.

    Raises `HearingModelMissingError` when a file of the pinned revision is absent.
    """
    artifacts = _ARTIFACTS[model]
    found = [
        try_to_load_from_cache(
            artifacts.repository,
            filename,
            cache_dir=cache,
            revision=artifacts.revision,
        )
        for filename in artifacts.filenames
    ]
    if not all(isinstance(path, str) for path in found):
        message = (
            f"SPEECH_HEARING_MODEL={model.value} is not in {cache}; "
            f"fetch it with: speech-runtime-fetch-hearing-model {model.value}"
        )
        raise HearingModelMissingError(message)
    return Path(found[0]).parent


def fetch_model(model: HearingModel, cache: Path) -> Path:
    """Download the pinned files of `model` into `cache`, and return their directory."""
    artifacts = _ARTIFACTS[model]
    snapshot_download(
        artifacts.repository,
        revision=artifacts.revision,
        cache_dir=cache,
        allow_patterns=list(artifacts.filenames),
    )
    return model_directory(model, cache)

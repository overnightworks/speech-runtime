"""Fetch a hearing model at its pinned revision into the Hugging Face cache."""

# ruff: noqa: T201 - this command prints where the model went

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING

from huggingface_hub import snapshot_download

from speech_runtime_service.config import load_settings
from speech_runtime_service.hearing_models import HearingModel, fetch_model

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


def main(
    argv: Sequence[str] | None = None,
    download: Callable[..., str] = snapshot_download,
) -> None:
    """Download the model named in `argv` into `SPEECH_HUGGINGFACE_CACHE`, once.

    `argv` defaults to the command line.
    """
    parser = argparse.ArgumentParser(
        prog="speech-runtime-fetch-hearing-model",
        allow_abbrev=False,
        description="Fetch a hearing model at its pinned revision so the "
        "service can load it from local files.",
    )
    parser.add_argument("model", type=HearingModel, choices=list(HearingModel))
    model = parser.parse_args(argv).model
    print(fetch_model(model, load_settings().huggingface_cache, download))

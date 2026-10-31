"""Fetch a hearing model at its pinned revision into the Hugging Face cache."""

# ruff: noqa: T201 - this command prints where the model went

from __future__ import annotations

import argparse

from speech_runtime_service.config import load_settings
from speech_runtime_service.hearing_models import HearingModel, fetch_model


def main() -> None:
    """Download the named model into `SPEECH_HUGGINGFACE_CACHE`, once."""
    parser = argparse.ArgumentParser(
        prog="speech-runtime-fetch-hearing-model",
        allow_abbrev=False,
        description="Fetch a hearing model at its pinned revision so the "
        "service can load it from local files.",
    )
    parser.add_argument("model", type=HearingModel, choices=list(HearingModel))
    model = parser.parse_args().model
    print(fetch_model(model, load_settings().huggingface_cache))


if __name__ == "__main__":
    main()

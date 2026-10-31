"""The fetch command puts the named model into the configured cache."""

from typing import TYPE_CHECKING

import pytest

from speech_runtime_service import fetch_hearing_model, hearing_models
from speech_runtime_service.hearing_models import HearingModel, model_directory
from tests.support.hear_support import write_snapshot

if TYPE_CHECKING:
    from pathlib import Path


def test_the_command_fetches_the_named_model_into_the_configured_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(hearing_models, "snapshot_download", write_snapshot)
    monkeypatch.setenv("SPEECH_HUGGINGFACE_CACHE", str(tmp_path))
    monkeypatch.setattr("sys.argv", ["fetch", HearingModel.SMALL.value])

    fetch_hearing_model.main()

    directory = model_directory(HearingModel.SMALL, tmp_path)
    assert capsys.readouterr().out.strip() == str(directory)


def test_the_command_refuses_a_model_it_does_not_know(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("sys.argv", ["fetch", "gigantic"])

    with pytest.raises(SystemExit) as stopped:
        fetch_hearing_model.main()

    assert stopped.value.code != 0

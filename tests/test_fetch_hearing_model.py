"""The fetch command puts the named model into the configured cache."""

from typing import TYPE_CHECKING

import pytest

from speech_runtime_service import fetch_hearing_model
from speech_runtime_service.hearing_models import HearingModel, model_directory
from tests.support.hear_support import write_snapshot

if TYPE_CHECKING:
    from pathlib import Path


def test_the_command_fetches_the_named_model_into_the_configured_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("SPEECH_HUGGINGFACE_CACHE", str(tmp_path))

    fetch_hearing_model.main([HearingModel.SMALL.value], write_snapshot)

    directory = model_directory(HearingModel.SMALL, tmp_path)
    assert capsys.readouterr().out.strip() == str(directory)


def test_the_command_refuses_a_model_it_does_not_know() -> None:
    with pytest.raises(SystemExit) as stopped:
        fetch_hearing_model.main(["gigantic"])

    assert stopped.value.code != 0

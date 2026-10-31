"""The pinned hearing models: found in the cache without the network, or fetched."""

import re
from typing import TYPE_CHECKING

import pytest

from speech_runtime_service import hearing_models
from speech_runtime_service.hearing_models import (
    HearingModel,
    HearingModelMissingError,
    fetch_model,
    model_directory,
)
from tests.support.hear_support import write_snapshot

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def requested_revisions(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """The revisions the downloader is asked for; it writes the files of each."""
    revisions: list[str] = []

    def download(repository: str, *, revision: str, **arguments: object) -> str:
        revisions.append(revision)
        return write_snapshot(repository, revision=revision, **arguments)

    monkeypatch.setattr(hearing_models, "snapshot_download", download)
    return revisions


@pytest.mark.parametrize("model", list(HearingModel))
def test_a_model_that_was_not_fetched_stops_the_start_naming_the_setting(
    tmp_path: Path, model: HearingModel
) -> None:
    with pytest.raises(HearingModelMissingError, match="SPEECH_HEARING_MODEL"):
        model_directory(model, tmp_path)


@pytest.mark.usefixtures("requested_revisions")
@pytest.mark.parametrize("model", list(HearingModel))
def test_a_fetched_model_is_found_where_it_was_fetched(
    tmp_path: Path, model: HearingModel
) -> None:
    fetched = fetch_model(model, tmp_path)

    assert model_directory(model, tmp_path) == fetched
    assert (fetched / "model.bin").is_file()


@pytest.mark.parametrize("model", list(HearingModel))
def test_a_model_is_fetched_at_a_commit_not_a_branch(
    tmp_path: Path, requested_revisions: list[str], model: HearingModel
) -> None:
    fetch_model(model, tmp_path)

    assert re.fullmatch("[0-9a-f]{40}", requested_revisions[0])

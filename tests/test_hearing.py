"""The faster-whisper engine loads from local files only and hears deterministically."""

import sys
import types
from typing import TYPE_CHECKING

import pytest

from speech_runtime_client.wire import HearingLanguage
from speech_runtime_service import hearing
from speech_runtime_service.hearing import ResourceUnavailableError, WhisperHearing
from speech_runtime_service.hearing_models import HearingModel

if TYPE_CHECKING:
    from pathlib import Path


class _Segment:
    def __init__(self, text: str) -> None:
        self.text = text


class _FakeWhisperModel:
    """Stands in for faster-whisper's model: it records how it was built and used."""

    built: list[dict[str, object]] = []  # noqa: RUF012 - reset by the fixture
    transcribed: list[dict[str, object]] = []  # noqa: RUF012 - reset by the fixture
    failure: RuntimeError | None = None

    def __init__(self, path: str, **options: object) -> None:
        if self.failure is not None:
            raise self.failure
        self.built.append({"path": path, **options})

    def transcribe(
        self, audio: object, **options: object
    ) -> tuple[list[_Segment], None]:
        self.transcribed.append({"audio": audio, **options})
        return [_Segment(" Guten"), _Segment(" Tag ")], None


@pytest.fixture
def library(monkeypatch: pytest.MonkeyPatch) -> type[_FakeWhisperModel]:
    _FakeWhisperModel.built = []
    _FakeWhisperModel.transcribed = []
    _FakeWhisperModel.failure = None
    monkeypatch.setitem(
        sys.modules,
        "faster_whisper",
        types.SimpleNamespace(WhisperModel=_FakeWhisperModel),
    )
    monkeypatch.setitem(
        sys.modules,
        "ctranslate2",
        types.SimpleNamespace(get_cuda_device_count=lambda: 0),
    )
    monkeypatch.setattr(hearing, "prepare_cuda_libraries", lambda: None)
    return _FakeWhisperModel


def _engine(directory: Path, device: str = "cpu") -> WhisperHearing:
    return WhisperHearing(HearingModel.SMALL, directory, device)


def test_the_model_loads_from_the_local_directory_and_never_downloads(
    library: type[_FakeWhisperModel], tmp_path: Path
) -> None:
    _engine(tmp_path).load()

    assert library.built[0]["path"] == str(tmp_path)
    assert library.built[0]["local_files_only"] is True


def test_on_the_cpu_the_model_runs_int8_on_four_threads(
    library: type[_FakeWhisperModel], tmp_path: Path
) -> None:
    _engine(tmp_path, "cpu").load()

    assert library.built[0]["device"] == "cpu"
    assert library.built[0]["compute_type"] == "int8"
    assert library.built[0]["cpu_threads"] == hearing.CPU_THREADS


def test_a_cuda_device_that_is_not_there_is_a_missing_resource(
    library: type[_FakeWhisperModel], tmp_path: Path
) -> None:
    with pytest.raises(ResourceUnavailableError):
        _engine(tmp_path, "cuda").load()

    assert library.built == []


def test_a_full_card_is_a_missing_resource(
    library: type[_FakeWhisperModel], tmp_path: Path
) -> None:
    library.failure = RuntimeError("CUDA failed with error out of memory")

    with pytest.raises(ResourceUnavailableError):
        _engine(tmp_path).load()


def test_any_other_runtime_error_of_the_load_is_not_hidden(
    library: type[_FakeWhisperModel], tmp_path: Path
) -> None:
    library.failure = RuntimeError("unsupported model format")

    with pytest.raises(RuntimeError, match="unsupported model format"):
        _engine(tmp_path).load()


@pytest.mark.parametrize("language", list(HearingLanguage))
def test_the_audio_is_heard_greedily_at_temperature_zero_in_the_asked_language(
    library: type[_FakeWhisperModel], tmp_path: Path, language: HearingLanguage
) -> None:
    engine = _engine(tmp_path)
    engine.load()

    text = engine.transcribe(b"\x00\x00" * 160, language)

    assert text == "Guten Tag"
    heard = library.transcribed[0]
    assert heard["language"] == language.value
    assert heard["temperature"] == 0.0
    assert heard["beam_size"] == 1


def test_hearing_before_the_load_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="not loaded"):
        _engine(tmp_path).transcribe(b"", HearingLanguage.DE)


@pytest.mark.usefixtures("library")
def test_a_closed_engine_no_longer_hears(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    engine.load()

    engine.close()

    with pytest.raises(RuntimeError, match="not loaded"):
        engine.transcribe(b"", HearingLanguage.DE)

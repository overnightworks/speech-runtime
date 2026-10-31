"""The faster-whisper engine loads from local files only and hears deterministically."""

from typing import TYPE_CHECKING

import pytest

from speech_runtime_client.wire import HearingLanguage
from speech_runtime_service import hearing
from speech_runtime_service.hearing import (
    HearingLibraries,
    ResourceUnavailableError,
    WhisperHearing,
    installed_libraries,
)
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
def library() -> type[_FakeWhisperModel]:
    _FakeWhisperModel.built = []
    _FakeWhisperModel.transcribed = []
    _FakeWhisperModel.failure = None
    return _FakeWhisperModel


def _engine(
    directory: Path,
    library: type[_FakeWhisperModel] | None = None,
    device: str = "cpu",
) -> WhisperHearing:
    libraries = HearingLibraries(library or _FakeWhisperModel, lambda: 0)
    return WhisperHearing(HearingModel.SMALL, directory, device, lambda: libraries)


def test_the_model_loads_from_the_local_directory_and_never_downloads(
    library: type[_FakeWhisperModel], tmp_path: Path
) -> None:
    _engine(tmp_path, library).load()

    assert library.built[0]["path"] == str(tmp_path)
    assert library.built[0]["local_files_only"] is True


def test_on_the_cpu_the_model_runs_int8_on_four_threads(
    library: type[_FakeWhisperModel], tmp_path: Path
) -> None:
    _engine(tmp_path, library, "cpu").load()

    assert library.built[0]["device"] == "cpu"
    assert library.built[0]["compute_type"] == "int8"
    assert library.built[0]["cpu_threads"] == hearing.CPU_THREADS


def test_a_cuda_device_that_is_not_there_is_a_missing_resource(
    library: type[_FakeWhisperModel], tmp_path: Path
) -> None:
    with pytest.raises(ResourceUnavailableError):
        _engine(tmp_path, library, "cuda").load()

    assert library.built == []


def test_a_full_card_is_a_missing_resource(
    library: type[_FakeWhisperModel], tmp_path: Path
) -> None:
    library.failure = RuntimeError("CUDA failed with error out of memory")

    with pytest.raises(ResourceUnavailableError):
        _engine(tmp_path, library).load()


def test_any_other_runtime_error_of_the_load_is_not_hidden(
    library: type[_FakeWhisperModel], tmp_path: Path
) -> None:
    library.failure = RuntimeError("unsupported model format")

    with pytest.raises(RuntimeError, match="unsupported model format"):
        _engine(tmp_path, library).load()


@pytest.mark.parametrize("language", list(HearingLanguage))
def test_the_audio_is_heard_greedily_at_temperature_zero_in_the_asked_language(
    library: type[_FakeWhisperModel], tmp_path: Path, language: HearingLanguage
) -> None:
    engine = _engine(tmp_path, library)
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


def test_a_closed_engine_no_longer_hears(
    library: type[_FakeWhisperModel], tmp_path: Path
) -> None:
    engine = _engine(tmp_path, library)
    engine.load()

    engine.close()

    with pytest.raises(RuntimeError, match="not loaded"):
        engine.transcribe(b"", HearingLanguage.DE)


@pytest.mark.parametrize("device", ["gpu", "cuda-invalid", "cuda:0", "CPU", ""])
def test_a_device_hearing_does_not_know_stops_the_start(
    tmp_path: Path, device: str
) -> None:
    with pytest.raises(ValueError, match="SPEECH_DEVICE"):
        _engine(tmp_path, device=device)


def test_the_installed_libraries_count_the_cuda_devices() -> None:
    assert installed_libraries().cuda_device_count() >= 0

"""The faster-whisper engine loads from local files only and hears deterministically."""

from typing import TYPE_CHECKING

import pytest

from speech_client.wire import Language
from speech_service import hearing
from speech_service.clock import LoopClock
from speech_service.hearing import (
    Hearing,
    ResourceUnavailableError,
    ScoreIdentifier,
    WhisperHearing,
)
from speech_service.hearing_models import HearingModel
from tests.support.whisper_support import Segment, StandInWhisper

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def library() -> StandInWhisper:
    return StandInWhisper()


def _engine(
    directory: Path,
    library: StandInWhisper | None = None,
    device: str = "cpu",
    cuda_devices: int = 0,
    cuda_library_dirs: list[Path] | None = None,
) -> WhisperHearing:
    libraries = (library or StandInWhisper()).libraries(cuda_devices, cuda_library_dirs)
    return WhisperHearing(
        HearingModel.SMALL, directory, device, ScoreIdentifier(0.0), lambda: libraries
    )


def test_the_model_loads_from_the_local_directory_and_never_downloads(
    library: StandInWhisper, tmp_path: Path
) -> None:
    _engine(tmp_path, library).load()

    assert library.built[0]["path"] == str(tmp_path)
    assert library.built[0]["local_files_only"] is True


def test_on_the_cpu_the_model_runs_int8_on_four_threads(
    library: StandInWhisper, tmp_path: Path
) -> None:
    _engine(tmp_path, library, "cpu").load()

    assert library.built[0]["device"] == "cpu"
    assert library.built[0]["compute_type"] == "int8"
    assert library.built[0]["cpu_threads"] == hearing.CPU_THREADS


def test_a_cuda_device_that_is_not_there_is_a_missing_resource(
    library: StandInWhisper, tmp_path: Path
) -> None:
    engine = _engine(tmp_path, library, "cuda")

    with pytest.raises(ResourceUnavailableError):
        engine.load()

    assert library.built == []


def test_a_cuda_device_without_the_gpu_extra_is_refused_naming_the_extra(
    library: StandInWhisper, tmp_path: Path
) -> None:
    engine = _engine(tmp_path, library, "cuda", cuda_devices=1)

    with pytest.raises(ResourceUnavailableError, match="gpu extra"):
        engine.load()

    assert library.built == []


def test_without_the_gpu_extra_hearing_is_not_ready(
    library: StandInWhisper, tmp_path: Path
) -> None:
    engine = _engine(tmp_path, library, "cuda", cuda_devices=1)
    loading = Hearing(engine, LoopClock())

    loading.load()

    assert loading.health()["ready"] is False
    assert loading.capture() is None


def test_a_cuda_device_with_the_libraries_loads_on_the_card(
    library: StandInWhisper, tmp_path: Path
) -> None:
    engine = _engine(
        tmp_path, library, "cuda", cuda_devices=1, cuda_library_dirs=[tmp_path]
    )

    engine.load()

    assert library.built[0]["device"] == "cuda"


def test_a_full_card_is_a_missing_resource(
    library: StandInWhisper, tmp_path: Path
) -> None:
    library.failure = RuntimeError("CUDA failed with error out of memory")

    engine = _engine(tmp_path, library)

    with pytest.raises(ResourceUnavailableError):
        engine.load()


def test_any_other_runtime_error_of_the_load_is_not_hidden(
    library: StandInWhisper, tmp_path: Path
) -> None:
    library.failure = RuntimeError("unsupported model format")

    engine = _engine(tmp_path, library)

    with pytest.raises(RuntimeError, match="unsupported model format"):
        engine.load()


def test_hearing_before_the_load_is_an_error(tmp_path: Path) -> None:
    engine = _engine(tmp_path)

    with pytest.raises(RuntimeError, match="not loaded"):
        engine.transcribe(b"", (Language.DE,))


def test_a_closed_engine_no_longer_hears(
    library: StandInWhisper, tmp_path: Path
) -> None:
    engine = _engine(tmp_path, library)
    engine.load()

    engine.close()

    with pytest.raises(RuntimeError, match="not loaded"):
        engine.transcribe(b"", (Language.DE,))


@pytest.mark.parametrize("device", ["gpu", "cuda-invalid", "cuda:0", "CPU", ""])
def test_a_device_hearing_does_not_know_stops_the_start(
    tmp_path: Path, device: str
) -> None:
    with pytest.raises(ValueError, match="SPEECH_DEVICE"):
        _engine(tmp_path, device=device)


@pytest.mark.parametrize(
    ("no_speech_prob", "avg_logprob", "text"),
    [
        (0.7, -1.2, ""),
        (0.7, -1.0, ""),
        (0.7, -0.9, "Guten"),
        (0.6, -1.2, "Guten"),
        (0.1, -1.2, "Guten"),
    ],
    ids=[
        "improbable-words-of-no-speech",
        "at-the-log-probability-line",
        "likely-words-despite-no-speech",
        "at-the-no-speech-line",
        "improbable-words-of-speech",
    ],
)
def test_a_decode_is_no_speech_when_both_measures_say_so_and_still_reports_them(
    library: StandInWhisper,
    tmp_path: Path,
    no_speech_prob: float,
    avg_logprob: float,
    text: str,
) -> None:
    library.segments = [Segment(" Guten", no_speech_prob, avg_logprob)]
    engine = _engine(tmp_path, library)
    engine.load()

    heard = engine.transcribe(b"\0\0" * 16_000, (Language.DE,))

    assert (heard.text, heard.no_speech_prob, heard.avg_logprob) == (
        text,
        no_speech_prob,
        avg_logprob,
    )


def test_a_decode_without_a_segment_has_no_measures(
    library: StandInWhisper, tmp_path: Path
) -> None:
    library.segments = []
    engine = _engine(tmp_path, library)
    engine.load()

    heard = engine.transcribe(b"\0\0" * 16_000, (Language.DE,))

    assert (heard.text, heard.no_speech_prob, heard.avg_logprob) == ("", None, None)


def test_the_load_fails_loud_when_the_voice_detector_does_not_load(
    library: StandInWhisper, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse() -> None:
        message = "no model"
        raise RuntimeError(message)

    engine = _engine(tmp_path, library)
    monkeypatch.setattr(engine._detectors, "load", refuse)  # noqa: SLF001 - the shipped model cannot be broken from outside

    with pytest.raises(RuntimeError, match="no model"):
        engine.load()


def test_a_loaded_engine_opens_a_detector_for_a_session(
    library: StandInWhisper, tmp_path: Path
) -> None:
    engine = _engine(tmp_path, library)
    engine.load()

    detector = engine.open_detector()

    assert detector.probabilities(bytes(2 * 1024)) != []


def test_a_closed_engine_opens_no_detector(
    library: StandInWhisper, tmp_path: Path
) -> None:
    engine = _engine(tmp_path, library)
    engine.load()

    engine.close()

    with pytest.raises(RuntimeError, match="not loaded"):
        engine.open_detector()

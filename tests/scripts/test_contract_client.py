"""The contract client counts a partial only while PCM frames remain unsent."""

from __future__ import annotations

import asyncio
import importlib.util
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from presentator_speech.pcm import pcm_from_wav
from presentator_speech.service import Runtime
from tests.support.fakes_support import SPEAK_SAMPLE_RATE, FakeHearing, FakeSpeaking
from tests.support.private_server_support import serving_privately

if TYPE_CHECKING:
    from types import ModuleType


def _script(name: str) -> ModuleType:
    path = Path(__file__).resolve().parents[2] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(name)
    return module


CLIENT = _script("contract_client")
PROVE = _script("prove_speech")
SAMPLE_RATE = 16_000
SENTENCE_START = "Die Kamera zeigt den Vortragenden"


def _two_pcm_frames() -> tuple[bytes, int]:
    frame_bytes = int(CLIENT.FRAME_SECONDS * SAMPLE_RATE * 2)
    return bytes(frame_bytes * 2), frame_bytes


def test_a_partial_arriving_with_one_frame_still_unsent_is_accepted() -> None:
    pcm, frame_bytes = _two_pcm_frames()
    next_offset = frame_bytes
    assert CLIENT.frames_remain_unsent(pcm, next_offset) is True


def test_a_partial_arriving_after_the_last_frame_is_rejected_as_late() -> None:
    pcm, frame_bytes = _two_pcm_frames()
    next_offset = frame_bytes * 2
    assert CLIENT.frames_remain_unsent(pcm, next_offset) is False


def test_both_proof_scripts_reach_a_running_speech_over_its_socket(tmp_path) -> None:
    async def scenario() -> tuple[object, object, bytes, bytes]:
        runtime = Runtime(FakeSpeaking(), FakeHearing(), memory_probe=lambda: 0)
        async with serving_privately(tmp_path, runtime) as server:
            return (
                await asyncio.to_thread(CLIENT._get_json, server.path, "/health"),  # noqa: SLF001
                await asyncio.to_thread(PROVE._get_json, server.path, "/health"),  # noqa: SLF001
                await asyncio.to_thread(CLIENT._speak, server.path, "Hallo"),  # noqa: SLF001
                await asyncio.to_thread(
                    PROVE._speak_measured,  # noqa: SLF001
                    server.path,
                    "Hallo",
                ),
            )

    contract_health, prove_health, contract_wav, measured = asyncio.run(scenario())

    assert contract_health == prove_health
    assert pcm_from_wav(contract_wav)[0] == SPEAK_SAMPLE_RATE
    assert pcm_from_wav(measured[0])[0] == SPEAK_SAMPLE_RATE


def test_prove_points_the_service_at_the_repository_provider_environments(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.delenv("SPEECH_PROVIDER_ROOT", raising=False)

    environment = PROVE._speech_environment(tmp_path, "model")  # noqa: SLF001

    provider_root = Path(environment["SPEECH_PROVIDER_ROOT"])
    assert provider_root.is_absolute()
    assert (provider_root / "chatterbox" / "pyproject.toml").is_file()


def test_prove_keeps_a_provider_root_the_caller_set(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("SPEECH_PROVIDER_ROOT", "/elsewhere")

    environment = PROVE._speech_environment(tmp_path, "model")  # noqa: SLF001

    assert environment["SPEECH_PROVIDER_ROOT"] == "/elsewhere"


def _repetition(**changes: object) -> object:
    arrivals = {
        "audio_seconds": 6.0,
        "speech_ended_s": 6.0,
        "sending_ended_s": 6.9,
        "partial_at_s": 2.0,
        "final_at_s": 6.5,
        "final_text": "Hallo Welt",
    }
    return CLIENT.Repetition(**{**arrivals, **changes})


@pytest.mark.parametrize(
    ("final_at_s", "sending_to_final_s", "speech_end_to_final_s"),
    [(6.5, 6.9, 0.5), (14.0, 14.0, 8.0)],
    ids=["final before the silence is sent", "final after sending ended"],
)
def test_the_two_final_timings_count_from_sending_start_and_from_speech_end(
    final_at_s: float, sending_to_final_s: float, speech_end_to_final_s: float
) -> None:
    """#515 acceptance 4: the two final timings, pure."""
    repetition = _repetition(final_at_s=final_at_s)

    assert repetition.sending_to_final_s == pytest.approx(sending_to_final_s)
    assert repetition.speech_end_to_final_s == pytest.approx(speech_end_to_final_s)


@pytest.mark.parametrize(
    ("arrived_s", "expected"),
    [(39.9, True), (40.0, True), (40.1, False)],
    ids=["inside the wait", "at the wait", "past the wait"],
)
def test_a_final_counts_only_within_the_wait_after_sending_ended(
    arrived_s: float, *, expected: bool
) -> None:
    """#515: a final received after the 30 s wait does not count."""
    assert CLIENT.final_arrived_in_time(arrived_s, sending_ended_s=10.0) is expected


class _RecordingWebsocket:
    """Marks, in order, every send and every receive attempt of the frame loop."""

    def __init__(self, order: list[str]) -> None:
        self._order = order

    async def send(self, _frame: bytes) -> None:
        self._order.append("send")

    async def recv(self) -> str:
        self._order.append("receive")
        await asyncio.sleep(1)
        return "{}"


class _OrderObserver:
    def __init__(self, order: list[str]) -> None:
        self._order = order

    def note_sent(self) -> None:
        self._order.append("sent")

    def note(self) -> None:
        self._order.append("noted")


def test_the_send_moment_is_taken_before_any_wait_of_the_frame() -> None:
    """#515: speech ended is the last frame sent, not the end of its waits."""
    pcm, _ = _two_pcm_frames()
    order: list[str] = []

    asyncio.run(
        CLIENT.send_pcm(
            _RecordingWebsocket(order),
            pcm,
            SAMPLE_RATE,
            [],
            _OrderObserver(order),
        )
    )

    assert order[:2] == ["send", "sent"]


@pytest.mark.parametrize(
    ("changes", "expected"),
    [
        ({"partial_at_s": 2.0}, "2.000 s while sending"),
        ({"partial_at_s": 11.5}, "11.500 s after speech ended"),
        ({"partial_at_s": None}, "no partial"),
    ],
)
def test_the_first_partial_says_whether_it_arrived_while_sending(
    changes: dict[str, object], expected: str
) -> None:
    """#515 Change: the first partial says whether it came while sending."""
    assert _repetition(**changes).first_partial == expected


def test_a_missed_final_is_recorded_as_a_value_not_an_abort() -> None:
    """#515 Change: a missed gate is a value."""
    repetition = _repetition(final_at_s=None, final_text=None)

    assert repetition.sending_to_final_s is None
    assert repetition.speech_end_to_final_s is None
    assert repetition.final_description == "no final within 30 s"


def test_the_summary_gives_median_worst_and_count_of_completed_repetitions() -> None:
    """#515 Change: median and worst of the completed repetitions and their count."""
    repetitions = [
        _repetition(final_at_s=value, partial_at_s=partial)
        for value, partial in ((7.0, 1.0), (8.0, 2.0), (12.0, 4.0), (None, 20.0))
    ]

    summary = CLIENT.summarize(repetitions)

    assert summary["completed"] == 3
    assert summary["sending_to_final_s"] == (8.0, 12.0)
    assert summary["speech_end_to_final_s"] == (2.0, 6.0)
    assert summary["partial_s"] == (2.0, 4.0)


def test_a_final_matches_the_spoken_sentence_without_case_and_punctuation() -> None:
    """#515 Change: a final is judged by its words."""
    assert CLIENT.same_words("die kamera zeigt, den Vortragenden!", SENTENCE_START)
    assert not CLIENT.same_words("Die Kamera zeigt Vortragenden", SENTENCE_START)


def test_prove_defaults_to_the_card_proof_and_never_the_operators_voice_choice(
    tmp_path, monkeypatch
) -> None:
    """#515 Change: own temporary state directory, card proof by default."""
    for name in ("SPEECH_DEVICE", "SPEECH_HEARING_MODEL", "CUDA_VISIBLE_DEVICES"):
        monkeypatch.delenv(name, raising=False)

    environment = PROVE._speech_environment(tmp_path, "model")  # noqa: SLF001

    assert environment["SPEECH_DEVICE"] == "cuda"
    assert environment["SPEECH_HEARING_MODEL"] == "Systran/faster-whisper-large-v3"
    assert "CUDA_VISIBLE_DEVICES" not in environment
    state_directory = Path(environment["SPEECH_STATE_DIRECTORY"])
    assert state_directory.is_dir()
    assert tmp_path in state_directory.parents


def test_prove_takes_device_and_hearing_model_and_hides_the_card_on_the_cpu(
    tmp_path, monkeypatch
) -> None:
    """#515 Change: CUDA_VISIBLE_DEVICES empty on cpu."""
    monkeypatch.setenv("SPEECH_DEVICE", "cpu")
    monkeypatch.setenv("SPEECH_HEARING_MODEL", "elsewhere/whisper")

    environment = PROVE._speech_environment(tmp_path, "model")  # noqa: SLF001

    assert environment["SPEECH_DEVICE"] == "cpu"
    assert environment["SPEECH_HEARING_MODEL"] == "elsewhere/whisper"
    assert environment["CUDA_VISIBLE_DEVICES"] == ""


def test_peak_resident_memory_is_read_from_the_process_status_in_mebibytes() -> None:
    """#515 Change: peak resident memory from the process status."""
    status = "Name:\tpython\nVmPeak:\t 9999 kB\nVmHWM:\t 2048000 kB\nVmRSS:\t 1 kB\n"

    assert PROVE.peak_rss_mib(status) == pytest.approx(2000.0)


@pytest.mark.parametrize("arguments", [["--help"], ["--measur"], ["extra"]])
def test_prove_exits_before_any_start_on_help_or_an_unknown_argument(arguments) -> None:
    """#515 Never: --help and an unknown argument never start a proof."""
    with pytest.raises(SystemExit) as exit_info:
        PROVE.parse_arguments(arguments)

    assert (exit_info.value.code == 0) == (arguments == ["--help"])


def test_prove_reads_measure_as_a_flag() -> None:
    """#515 Change: the CPU measurement is chosen by the --measure flag."""
    assert PROVE.parse_arguments(["--measure"]).measure is True
    assert PROVE.parse_arguments([]).measure is False

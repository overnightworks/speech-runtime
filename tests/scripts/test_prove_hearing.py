"""The hearing proof: its command line, record, wait, memory peak and taken card."""

from __future__ import annotations

import contextlib
import json
import re
import tomllib
from http import HTTPStatus
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING

import httpx2
import pytest

from speech_client.hearing import Refused
from speech_client.outcomes import Unreachable
from speech_client.wire import Health, RefusalCode, refusal
from speech_service.hearing_models import HearingModel
from speech_service.pcm import pcm_from_wav
from tests.support.clock_support import FakeClock
from tests.support.nvidia_support import FakeNvidiaSmi
from tests.support.script_support import script_module

if TYPE_CHECKING:
    import argparse
    import wave
    from collections.abc import Callable, Iterator

    from piper.config import SynthesisConfig

    from speech_client.wire import VoiceId

HEARING = script_module("prove_hearing")
VOICES = Path.home() / "voices"
RUNS = "7"
WIRE_RATE = 24_000
SPEECH = Path(__file__).parents[1] / "fixtures" / "speech" / "nein_danke.wav"


def command(*changes: str) -> list[str]:
    return [
        "--device",
        "cpu",
        "--hearing-model",
        "small",
        "--record",
        "record.json",
        "--english-voice-directory",
        str(VOICES),
        *changes,
    ]


def test_the_device_the_models_and_the_paths_come_from_the_command_line() -> None:
    arguments = HEARING.parse_arguments(
        command("--hearing-model", "large-v3", "--runs", RUNS)
    )

    assert arguments.device == "cpu"
    assert arguments.hearing_models == [HearingModel.SMALL, HearingModel.LARGE_V3]
    assert arguments.record == Path("record.json")
    assert arguments.english_voice_directory == VOICES
    assert arguments.runs == int(RUNS)


def test_a_record_in_a_missing_directory_is_refused_before_anything_starts(
    tmp_path: Path,
) -> None:
    arguments = command("--record", str(tmp_path / "missing" / "record.json"))

    with pytest.raises(SystemExit):
        HEARING.parse_arguments(arguments)


def test_the_runs_default_to_the_fewest_a_median_needs() -> None:
    assert HEARING.parse_arguments(command()).runs == HEARING.figures.MIN_RUNS


def test_fewer_runs_than_a_median_needs_are_refused() -> None:
    arguments = command("--runs", "4")

    with pytest.raises(SystemExit):
        HEARING.parse_arguments(arguments)


@pytest.mark.parametrize("missing", ["--device", "--hearing-model", "--record"])
def test_a_command_without_a_required_option_is_refused(missing: str) -> None:
    arguments = command()
    position = arguments.index(missing)
    del arguments[position : position + 2]

    with pytest.raises(SystemExit):
        HEARING.parse_arguments(arguments)


def test_a_model_missing_from_the_cache_names_the_command_that_fetches_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SPEECH_HUGGINGFACE_CACHE", str(tmp_path))

    with pytest.raises(SystemExit, match=r"fetch it with: (\S+) small") as refused:
        HEARING.require_cached_models([HearingModel.SMALL])

    hinted = re.search(r"fetch it with: (\S+) small", str(refused.value))
    scripts = tomllib.loads(
        (Path(__file__).parents[2] / "pyproject.toml").read_text(encoding="utf-8")
    )["project"]["scripts"]
    assert hinted is not None
    assert scripts[hinted[1]] == "speech_service.fetch_hearing_model:main"


def test_the_resident_memory_with_a_voice_includes_what_its_synthesis_took(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    card = FakeNvidiaSmi(tmp_path, monkeypatch)
    card.use(1000)
    service = HEARING.Service("http://127.0.0.1:1", None, None)  # type: ignore[arg-type]
    synthesis_mib = 700

    @contextlib.contextmanager
    def serve(*_: str) -> Iterator[HEARING.Service]:
        yield service

    def speak(*_: object) -> tuple[bytes, float, float]:
        card.use(1000 + synthesis_mib)
        return b"", 0.0, 0.0

    ports = HEARING.Ports(serve=serve, speak=speak, hear=lambda *_: (0.0, ""))

    peak = HEARING.measure_peak("cuda", "small", HEARING.PIPER_PROOF_VOICE, b"", ports)

    assert peak["warmed_resident_over_baseline_mib"] == synthesis_mib


@contextlib.contextmanager
def sampled_before_and_after() -> Iterator[object]:
    """A sampler that saw the card only before the allocation and after teardown."""
    yield SimpleNamespace(samples=[1010, 1005])


def test_the_peak_is_never_below_the_warmed_resident_reading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    card = FakeNvidiaSmi(tmp_path, monkeypatch)
    card.use(1000)
    synthesis_mib = 700

    @contextlib.contextmanager
    def serve(*_: str) -> Iterator[object]:
        yield HEARING.Service("http://127.0.0.1:1", None, None)  # type: ignore[arg-type]

    def speak(*_: object) -> tuple[bytes, float, float]:
        card.use(1000 + synthesis_mib)
        return b"", 0.0, 0.0

    ports = HEARING.Ports(
        serve=serve,
        speak=speak,
        hear=lambda *_: (0.0, ""),
        sample=sampled_before_and_after,
    )

    peak = HEARING.measure_peak("cuda", "small", HEARING.PIPER_PROOF_VOICE, b"", ports)

    assert peak["peak_over_baseline_mib"] == synthesis_mib
    assert peak["peak_sampled_every_seconds"] == HEARING.MEMORY_SAMPLE_SECONDS


def test_the_record_names_the_clips_a_language_was_measured_with() -> None:
    german = HEARING.LANGUAGES[0]

    def clips_recorded(question: bytes) -> object:
        record = HEARING.measure_language(
            SimpleNamespace(),  # type: ignore[arg-type]
            german,
            HEARING.Clip(question, b"long turn"),
            5,
            HEARING.Ports(live=_one_delta_turn, hear=lambda *_: (0.5, "Wie")),
        )
        return record["clip_sha256"]

    first = clips_recorded(b"one")
    again = clips_recorded(b"one")
    other = clips_recorded(b"another")

    assert again == first
    assert other != first


def test_both_languages_run_in_one_fixed_order_for_every_clip() -> None:
    handed: list[tuple[object, ...]] = []

    async def live(
        _: object, allowed: tuple[object, ...], __: bytes
    ) -> HEARING.LiveTurn:
        handed.append(allowed)
        return await _one_delta_turn(_, allowed, b"")

    english = HEARING.LANGUAGES[1]

    HEARING.measure_language(
        SimpleNamespace(),  # type: ignore[arg-type]
        english,
        HEARING.Clip(b"question", b"long turn"),
        5,
        HEARING.Ports(live=live, hear=lambda *_: (0.5, "Wie")),
    )

    two_languages = [allowed for allowed in handed if len(allowed) > 1]
    assert two_languages
    assert set(two_languages) == {(HEARING.Language.DE, HEARING.Language.EN)}


class VoiceWithDurationNoise:
    """A voice whose sentences pause past the cut-off unless its durations are fixed."""

    def synthesize_wav(
        self,
        text: str,
        wav_file: wave.Wave_write,
        syn_config: SynthesisConfig | None = None,
    ) -> None:
        """Each sentence as two halves of speech with a pause between them."""
        pause = 0.1 if syn_config and syn_config.noise_w_scale == 0.0 else 0.6
        wav_file.setparams((1, 2, WIRE_RATE, 0, "NONE", "not compressed"))
        rate, phrase = pcm_from_wav(SPEECH.read_bytes())
        half = HEARING.figures.to_wire_rate(phrase, rate)
        for _ in re.split(r"(?<=\.) ", text):
            wav_file.writeframes(half + bytes(int(2 * WIRE_RATE * pause)) + half)


def test_an_english_long_turn_stays_one_turn_whatever_the_durations() -> None:
    clip = HEARING.english_clip(VoiceWithDurationNoise(), HEARING.ENGLISH_LONG_TURN)

    HEARING.figures.check_continuous_turn(clip)


async def _one_delta_turn(_: object, __: object, ___: bytes) -> HEARING.LiveTurn:
    return HEARING.LiveTurn(2.0, (2.5,), ("Wie",), 6.0, "Wie", HEARING.Language.DE)


def test_a_language_measured_with_one_delta_in_its_long_turn_is_not_measured() -> None:
    runs = 5
    question, long_turn = b"question", b"long turn"
    german = HEARING.LANGUAGES[0]

    async def hear_turn(_: object, __: object, pcm: bytes) -> HEARING.LiveTurn:
        deltas = ("Wie",) if pcm == long_turn else ("Wie", "spät")
        return HEARING.LiveTurn(
            2.0,
            tuple(2.5 + index for index in range(len(deltas))),
            deltas,
            6.0,
            "Wie spät",
            HEARING.Language.DE,
        )

    def hear_recording(_: object, __: object, ___: bytes) -> tuple[float, str]:
        return 0.5, "Wie spät"

    record = HEARING.measure_language(
        SimpleNamespace(),  # type: ignore[arg-type]
        german,
        HEARING.Clip(question, long_turn),
        runs,
        HEARING.Ports(live=hear_turn, hear=hear_recording),
    )
    models = {
        "small": {
            "status": HEARING.figures.StageStatus.MEASURED,
            "languages": {german.code.value: record},
        }
    }

    assert record["question"]["live"]["runs_with_partial"] == runs
    assert HEARING.figures.not_measured(["small"], models, {}) == [
        f"small languages {german.code.value} long_turn live growing_deltas"
    ]


def test_the_long_turn_runs_one_and_two_languages_interleaved_once_per_run() -> None:
    german = HEARING.LANGUAGES[0]
    runs = 5
    long_turn = b"long turn"
    handed: list[tuple[object, ...]] = []

    async def live(
        _: object, allowed: tuple[object, ...], pcm: bytes
    ) -> HEARING.LiveTurn:
        if pcm == long_turn:
            handed.append(allowed)
        return await _one_delta_turn(_, allowed, pcm)

    record = HEARING.measure_language(
        SimpleNamespace(),  # type: ignore[arg-type]
        german,
        HEARING.Clip(b"question", long_turn),
        runs,
        HEARING.Ports(live=live, hear=lambda *_: (0.5, "Wie")),
    )

    assert handed == [(german.code,), HEARING.BOTH_LANGUAGES] * runs
    assert record["long_turn"]["two_languages"]["runs"] == runs


def test_the_question_reports_its_language_step_without_a_limit() -> None:
    german = HEARING.LANGUAGES[0]
    runs = 5

    record = HEARING.measure_language(
        SimpleNamespace(),  # type: ignore[arg-type]
        german,
        HEARING.Clip(b"question", b"long turn"),
        runs,
        HEARING.Ports(live=_one_delta_turn, hear=lambda *_: (0.5, "Wie")),
    )

    step = record["question"]["two_languages"]
    assert step["runs"] == runs
    assert "added_seconds" in step
    assert "above_limit" not in step


class FakeCard:
    """Replaces everything of `prove` that needs a service or a card."""

    def __init__(
        self, monkeypatch: pytest.MonkeyPatch, cache: Path, *, fails_in: str = ""
    ) -> None:
        """Patch the proof to start no service; `fails_in` names a failing model."""
        self.fails_in = fails_in
        clips = {language: HEARING.Clip(b"", b"") for language in HEARING.Language}
        monkeypatch.setenv("SPEECH_HUGGINGFACE_CACHE", str(cache))
        monkeypatch.setattr(HEARING, "require_cached_models", lambda *_: None)
        monkeypatch.setattr(HEARING, "card_line", lambda _: "card")
        monkeypatch.setattr(HEARING, "hearing_service", self.service)
        monkeypatch.setattr(HEARING, "make_clips", lambda *_: clips)
        monkeypatch.setattr(HEARING, "measure_language", lambda *_: {"figures": 1})
        monkeypatch.setattr(HEARING, "measure_peak", lambda *_: {"peak_mib": 7})
        monkeypatch.setattr(HEARING, "prove_taken_card", lambda *_: {"failures": []})

    @contextlib.contextmanager
    def service(self, _device: str, model: str, _speaking: str) -> Iterator[object]:
        """A service that starts, or fails for the model named in `fails_in`."""
        if self.fails_in == model:
            message = f"{model} failed"
            raise RuntimeError(message)
        running = SimpleNamespace(ready_seconds=1.5)
        yield SimpleNamespace(running=running)


def cuda_arguments(tmp_path: Path, *changes: str) -> argparse.Namespace:
    return HEARING.parse_arguments(
        command("--record", str(tmp_path / "record.json"), "--device", "cuda", *changes)
    )


def prove(tmp_path: Path, *changes: str) -> dict[str, object]:
    arguments = cuda_arguments(tmp_path, *changes)
    with contextlib.suppress(SystemExit, RuntimeError):
        HEARING.prove(arguments)
    return json.loads(arguments.record.read_text())


def test_a_run_of_one_model_on_cuda_names_the_other_as_not_measured(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    FakeCard(monkeypatch, tmp_path)

    record = prove(tmp_path)

    assert record["not_measured"] == ["hearing model large-v3"]
    assert record["complete"] is False


def test_every_model_and_stage_measured_makes_the_record_complete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    FakeCard(monkeypatch, tmp_path)

    record = prove(tmp_path, "--hearing-model", "large-v3")

    assert record["not_measured"] == []
    assert record["complete"] is True


def test_a_taken_card_that_differs_is_never_written_as_complete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    FakeCard(monkeypatch, tmp_path)
    monkeypatch.setattr(HEARING, "prove_taken_card", lambda *_: {"failures": ["x"]})
    written: list[object] = []
    write = HEARING.write_record

    def write_noting_completeness(path: Path, record: dict[str, object]) -> None:
        written.append(record["complete"])
        write(path, record)

    monkeypatch.setattr(HEARING, "write_record", write_noting_completeness)
    arguments = cuda_arguments(tmp_path, "--hearing-model", "large-v3")

    with pytest.raises(SystemExit, match="differs from cell 30: x"):
        HEARING.prove(arguments)

    assert True not in written


def test_a_failing_second_model_leaves_the_figures_of_the_first_and_names_the_rest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    FakeCard(monkeypatch, tmp_path, fails_in="large-v3")

    record = prove(tmp_path, "--hearing-model", "large-v3")

    assert set(record["models"]) == {"small"}
    assert record["not_measured"] == [
        "hearing model large-v3",
        "stage gpu_memory",
        "stage taken_card",
    ]
    assert record["complete"] is False


def test_a_stage_that_fails_keeps_what_it_measured_before(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    FakeCard(monkeypatch, tmp_path)
    calls = []

    def peak(*_: object) -> dict[str, object]:
        calls.append(1)
        if len(calls) > 1:
            message = "second peak failed"
            raise RuntimeError(message)
        return {"peak_mib": 7}

    monkeypatch.setattr(HEARING, "measure_peak", peak)

    record = prove(tmp_path, "--hearing-model", "large-v3")

    assert record["gpu_memory"]["models"]["small"]["hearing_alone"] == {"peak_mib": 7}
    assert record["not_measured"] == ["stage gpu_memory", "stage taken_card"]


def test_a_stage_that_does_not_apply_on_cpu_is_not_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    FakeCard(monkeypatch, tmp_path)
    record_path = tmp_path / "record.json"
    arguments = HEARING.parse_arguments(command("--record", str(record_path)))

    HEARING.prove(arguments)

    assert json.loads(record_path.read_text())["not_measured"] == []


HELD_MIB = 512


def health_of(*, speaking: bool, hearing: bool) -> Health:
    return Health.model_validate(
        {
            "speaking": {"ready": speaking, "voices": {}},
            "hearing": {"model": "small", "ready": hearing},
            "card_memory_mb": 0,
        }
    )


HEALTH_READY = health_of(speaking=True, hearing=True)
HEALTH_NO_HEARING = health_of(speaking=True, hearing=False)
HEALTH_LOADING = health_of(speaking=False, hearing=False)


def wait_loaded_with(health: Health, *, speaks: bool) -> Health:
    clock = FakeClock()
    return HEARING.wait_loaded(
        "http://127.0.0.1:1",
        lambda: None,
        HEARING.LoadWait(lambda: speaks, lambda *_: health, clock.jump, clock.now),
    )


@pytest.mark.parametrize(
    ("health", "speaks"),
    [(HEALTH_READY, False), (HEALTH_NO_HEARING, True)],
    ids=["hearing ready", "speaking served after a load without hearing"],
)
def test_the_service_has_loaded_when_hearing_is_ready_or_speaking_is_served(
    health: Health, *, speaks: bool
) -> None:
    assert wait_loaded_with(health, speaks=speaks) == health


@pytest.mark.parametrize(
    ("health", "speaks"),
    [(HEALTH_LOADING, True), (HEALTH_NO_HEARING, False)],
    ids=["still loading", "speaking ready but not served"],
)
def test_a_service_that_never_finishes_loading_fails_the_wait(
    health: Health, *, speaks: bool
) -> None:
    with pytest.raises(SystemExit, match="did not finish loading"):
        wait_loaded_with(health, speaks=speaks)


def test_a_wait_with_slow_health_answers_still_ends_at_its_time_limit() -> None:
    clock = FakeClock()
    slow_seconds = 35.0

    def slow_health(*_: object) -> Health:
        clock.jump(slow_seconds)
        return HEALTH_LOADING

    wait = HEARING.LoadWait(lambda: True, slow_health, clock.jump, clock.now)
    with pytest.raises(SystemExit, match="did not finish loading"):
        HEARING.wait_loaded("http://127.0.0.1:1", lambda: None, wait)

    assert clock.now() < HEARING.READY_TIMEOUT_SECONDS + slow_seconds + 1


def test_a_wait_that_never_heard_the_service_names_no_health() -> None:
    clock = FakeClock()

    def unreachable(*_: object) -> Unreachable:
        return Unreachable()

    wait = HEARING.LoadWait(lambda: True, unreachable, clock.jump, clock.now)
    with pytest.raises(SystemExit) as failure:
        HEARING.wait_loaded("http://127.0.0.1:1", lambda: None, wait)

    assert "None" not in str(failure.value)


def test_a_wait_ends_at_once_with_the_last_health_when_the_server_exited() -> None:
    clock = FakeClock()
    polls = iter([None, None, 1])

    def exit_status() -> int | None:
        return next(polls)

    wait = HEARING.LoadWait(
        lambda: True, lambda *_: HEALTH_LOADING, clock.jump, clock.now
    )
    with pytest.raises(SystemExit, match="exited with status 1") as failure:
        HEARING.wait_loaded("http://127.0.0.1:1", exit_status, wait)

    assert str(HEALTH_LOADING) in str(failure.value)
    assert clock.now() < HEARING.READY_TIMEOUT_SECONDS


@pytest.fixture
def smi(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> FakeNvidiaSmi:
    return FakeNvidiaSmi(tmp_path, monkeypatch)


def taken_card(
    smi: FakeNvidiaSmi,
    *,
    health: Health,
    file_response: httpx2.Response,
    freed: bool,
    live_outcome: object = Refused(
        HTTPStatus.SERVICE_UNAVAILABLE, RefusalCode.UNAVAILABLE
    ),
) -> dict[str, object]:
    smi.use(100)

    @contextlib.contextmanager
    def hold() -> Iterator[int]:
        yield HELD_MIB

    @contextlib.contextmanager
    def serve(*_: str) -> Iterator[object]:
        yield SimpleNamespace(
            base_url="http://127.0.0.1:1",
            minting=None,
            running=SimpleNamespace(health=health),
        )

    async def upgrade(_: object) -> object:
        return live_outcome

    ports = HEARING.Ports(
        hold=hold,
        serve=serve,
        post=lambda *_: file_response,
        upgrade=upgrade,
        speak=lambda *_: (b"wav", 0.0, 0.0),
        freed=lambda _: freed,
    )
    clips = {HEARING.Language.DE: HEARING.Clip(b"", b"")}
    return HEARING.prove_taken_card("cuda", "small", clips, ports)


def unavailable() -> httpx2.Response:
    return httpx2.Response(
        HTTPStatus.SERVICE_UNAVAILABLE,
        content=refusal(RefusalCode.UNAVAILABLE).model_dump_json().encode(),
    )


def test_a_taken_card_that_behaves_as_cell_30_says_gives_no_failure(
    smi: FakeNvidiaSmi,
) -> None:
    result = taken_card(
        smi,
        health=HEALTH_NO_HEARING,
        file_response=unavailable(),
        freed=True,
    )

    assert result["failures"] == []
    assert result["held_mib"] == HELD_MIB
    refused = {
        "status": HTTPStatus.SERVICE_UNAVAILABLE,
        "code": RefusalCode.UNAVAILABLE,
    }
    assert result["refusals"] == {"file": refused, "live": refused}


def test_a_taken_card_that_is_not_freed_afterwards_is_a_failure(
    smi: FakeNvidiaSmi,
) -> None:
    result = taken_card(
        smi,
        health=HEALTH_NO_HEARING,
        file_response=unavailable(),
        freed=False,
    )

    assert result["failures"] == ["the card was not freed"]


def test_a_taken_card_that_accepts_the_live_upgrade_is_a_failure(
    smi: FakeNvidiaSmi,
) -> None:
    result = taken_card(
        smi,
        health=HEALTH_NO_HEARING,
        file_response=unavailable(),
        freed=True,
        live_outcome=None,
    )

    assert result["failures"] == ["the live upgrade did not answer 503 unavailable"]


def test_hearing_ready_on_a_taken_card_is_a_failure(
    smi: FakeNvidiaSmi,
) -> None:
    result = taken_card(
        smi,
        health=HEALTH_READY,
        file_response=unavailable(),
        freed=True,
    )

    assert result["failures"] == ["hearing is ready on a taken card"]


def test_a_file_route_that_answers_ok_on_a_taken_card_is_a_failure(
    smi: FakeNvidiaSmi,
) -> None:
    result = taken_card(
        smi,
        health=HEALTH_NO_HEARING,
        file_response=httpx2.Response(HTTPStatus.OK),
        freed=True,
    )

    assert result["failures"] == ["the file route did not answer 503 unavailable"]


@pytest.mark.parametrize(
    "offered", [frozenset(), frozenset(HEARING.ON_DEMAND_VOICES)], ids=["piper", "all"]
)
def test_only_offered_on_demand_voices_are_measured_and_the_rest_are_named(
    monkeypatch: pytest.MonkeyPatch, offered: frozenset[VoiceId]
) -> None:
    monkeypatch.setattr(HEARING, "measure_peak", lambda *_: {})
    memory: dict[str, object] = {}
    clips = {HEARING.Language.DE: HEARING.Clip(b"", b"")}

    HEARING.measure_memory("cuda", ["small"], clips, memory, offered)

    measured = set(memory["models"]["small"]["with_voice"])
    assert measured == {voice.value for voice in offered}
    assert (
        set(memory["voices_not_offered"])
        == {voice.value for voice in HEARING.ON_DEMAND_VOICES} - measured
    )


def test_a_cpu_run_that_fails_names_no_stage_that_never_applied(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    FakeCard(monkeypatch, tmp_path, fails_in="small")
    record_path = tmp_path / "record.json"
    arguments = HEARING.parse_arguments(command("--record", str(record_path)))

    with pytest.raises(RuntimeError, match="small failed"):
        HEARING.prove(arguments)

    assert json.loads(record_path.read_text())["not_measured"] == [
        "hearing model small"
    ]


def test_a_proof_killed_after_its_first_model_has_left_that_model_on_disk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeCard(monkeypatch, tmp_path)
    record_path = tmp_path / "record.json"
    on_disk: list[dict[str, object]] = []
    started = fake.service

    @contextlib.contextmanager
    def second_model_starts(device: str, model: str, speaking: str) -> Iterator[object]:
        if model == "large-v3":
            on_disk.append(json.loads(record_path.read_text()))
        with started(device, model, speaking) as service:
            yield service

    monkeypatch.setattr(HEARING, "hearing_service", second_model_starts)

    prove(tmp_path, "--hearing-model", "large-v3")

    assert on_disk[0]["models"]["small"]["status"] == "measured"
    assert on_disk[0]["not_measured"] == [
        "hearing model large-v3",
        "stage gpu_memory",
        "stage taken_card",
    ]


def test_a_language_that_fails_keeps_the_other_and_leaves_its_model_unfinished(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    FakeCard(monkeypatch, tmp_path)

    def hear(_service: object, language: object, *_: object) -> dict[str, object]:
        if language.code == HEARING.Language.EN:
            message = "english failed"
            raise RuntimeError(message)
        return {"figures": 1}

    monkeypatch.setattr(HEARING, "measure_language", hear)

    record = prove(tmp_path)

    small = record["models"]["small"]
    assert small["languages"] == {"de": {"figures": 1}}
    assert small["status"] == "unfinished"
    assert "hearing model small" in record["not_measured"]


def test_the_record_is_on_disk_after_each_language_and_each_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    FakeCard(monkeypatch, tmp_path)
    record_path = tmp_path / "record.json"
    seen: list[dict[str, object]] = []

    def read_then(value: dict[str, object]) -> Callable[..., dict[str, object]]:
        def measure(*_: object) -> dict[str, object]:
            seen.append(json.loads(record_path.read_text()))
            return value

        return measure

    monkeypatch.setattr(HEARING, "measure_language", read_then({"figures": 1}))
    monkeypatch.setattr(HEARING, "measure_peak", read_then({"peak_mib": 7}))
    monkeypatch.setattr(HEARING, "prove_taken_card", read_then({"failures": []}))

    prove(tmp_path)

    first_language, second_language, first_peak = seen[:3]
    assert "stage gpu_memory" not in seen[-1]["not_measured"]
    assert first_language["models"] == {}
    assert list(second_language["models"]["small"]["languages"]) == ["de"]
    assert first_peak["models"]["small"]["status"] == "measured"


def test_a_taken_card_names_a_live_outcome_that_is_not_a_refusal(
    smi: FakeNvidiaSmi,
) -> None:
    result = taken_card(
        smi,
        health=HEALTH_NO_HEARING,
        file_response=unavailable(),
        freed=True,
        live_outcome=Unreachable(),
    )

    live = result["refusals"]["live"]  # type: ignore[index]
    assert live["outcome"] == repr(Unreachable())

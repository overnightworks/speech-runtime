"""The contract client counts a partial only while PCM frames remain unsent."""

from __future__ import annotations

import asyncio
import importlib.util
from pathlib import Path
from typing import TYPE_CHECKING

from presentator_speech.pcm import pcm_from_wav
from presentator_speech.service import Runtime
from tests.conftest import SPEAK_SAMPLE_RATE, FakeHearing, FakeSpeaking
from tests.test_private_socket import _serving

if TYPE_CHECKING:
    from types import ModuleType


def _script(name: str) -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CLIENT = _script("contract_client")
PROVE = _script("prove")
SAMPLE_RATE = 16_000


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
        async with _serving(tmp_path, runtime) as server:
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

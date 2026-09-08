"""The contract client counts a partial only while PCM frames remain unsent."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types import ModuleType


def _contract_client() -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "scripts" / "contract_client.py"
    spec = importlib.util.spec_from_file_location("contract_client", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CLIENT = _contract_client()
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

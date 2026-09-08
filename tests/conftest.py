"""Fakes that stand in for the resident models."""

from __future__ import annotations

import math
import struct
from typing import TYPE_CHECKING

from speech.config import HEAR_SAMPLE_RATE
from speech.hearing import HearingSession
from speech.pcm import BYTES_PER_SAMPLE
from speech.service import Runtime, create_app

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from fastapi import FastAPI

    from speech.service import HearingEngine, SpeakingEngine

SAMPLE_RATE = HEAR_SAMPLE_RATE
SPEAK_SAMPLE_RATE = 22_050


def sine_pcm(seconds: float, rate: int = SAMPLE_RATE, freq: float = 440.0) -> bytes:
    n = int(seconds * rate)
    samples = [
        int(16_000 * math.sin(2 * math.pi * freq * index / rate)) for index in range(n)
    ]
    return struct.pack(f"<{n}h", *samples)


class FakeSpeaking:
    model_name = "fake-voice"
    sample_rate = SPEAK_SAMPLE_RATE
    ready = True
    streams = False

    def __init__(self, *, ready: bool = True, fail: bool = False) -> None:
        self.ready = ready
        self._fail = fail
        self.heard_text = ""
        self.heard_language = ""

    def load(self) -> None:
        if self._fail:
            message = "weights missing"
            raise RuntimeError(message)
        self.ready = True

    def pcm_chunks(self, text: str, language: str) -> Iterator[bytes]:
        self.heard_text = text
        self.heard_language = language
        pcm = sine_pcm(0.3, rate=self.sample_rate)
        mid = (len(pcm) // 2) // BYTES_PER_SAMPLE * BYTES_PER_SAMPLE
        yield pcm[:mid]
        yield pcm[mid:]


class FakeHearing:
    model_name = "fake-ears"
    ready = True

    def __init__(self, *, ready: bool = True, fail: bool = False) -> None:
        self.ready = ready
        self._fail = fail
        self.sessions = 0

    def load(self) -> None:
        if self._fail:
            message = "weights missing"
            raise RuntimeError(message)
        self.ready = True

    def open_session(self, language: str) -> HearingSession:
        del language
        self.sessions += 1
        spoken = "eins"

        def transcribe(pcm: bytes) -> str:
            duration = len(pcm) / (SAMPLE_RATE * BYTES_PER_SAMPLE)
            if duration < 0.2:
                return ""
            if duration < 0.8:
                return spoken
            return f"{spoken} zwei"

        return HearingSession(transcribe, SAMPLE_RATE)


def an_app(
    speaking: SpeakingEngine | None = None,
    hearing: HearingEngine | None = None,
    *,
    debug: bool = False,
    memory_probe: Callable[[], int] | None = None,
) -> FastAPI:
    runtime = Runtime(
        speaking or FakeSpeaking(),
        hearing or FakeHearing(),
        debug=debug,
        memory_probe=memory_probe if memory_probe is not None else (lambda: 0),
    )
    return create_app(runtime=runtime, load_models=False)

"""Fakes that stand in for the resident models."""

from __future__ import annotations

import math
import struct
import threading
from typing import TYPE_CHECKING

from speech_runtime_service.config import Settings
from speech_runtime_service.control import create_control_app
from speech_runtime_service.pcm import BYTES_PER_SAMPLE
from speech_runtime_service.service import Runtime
from tests.support.identity_support import valid_callers

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from fastapi import FastAPI

    from speech_runtime_service.service import SpeakingEngine

SPEAK_SAMPLE_RATE = 22_050
GATE_TIMEOUT_SECONDS = 5.0


def sine_pcm(seconds: float, rate: int, freq: float = 440.0) -> bytes:
    n = int(seconds * rate)
    samples = [
        int(16_000 * math.sin(2 * math.pi * freq * index / rate)) for index in range(n)
    ]
    return struct.pack(f"<{n}h", *samples)


class FakeSpeaking:
    """A speaking engine that returns a short tone without loading a model."""

    model_name = "fake-voice"
    sample_rate = SPEAK_SAMPLE_RATE
    ready = True
    streams = False
    retain_when_inactive = True

    def __init__(self, *, ready: bool = True, fail: bool = False) -> None:
        """Start ready or not, and optionally fail the load."""
        self.ready = ready
        self._fail = fail
        self.heard_text = ""
        self.heard_language = ""

    def load(self) -> None:
        """Load the fake weights, or raise when built to fail."""
        if self._fail:
            message = "weights missing"
            raise RuntimeError(message)
        self.ready = True

    def pcm_chunks(self, text: str, language: str) -> Iterator[bytes]:
        """Record the request and yield the tone in two chunks."""
        self.heard_text = text
        self.heard_language = language
        pcm = sine_pcm(0.3, rate=self.sample_rate)
        mid = (len(pcm) // 2) // BYTES_PER_SAMPLE * BYTES_PER_SAMPLE
        yield pcm[:mid]
        yield pcm[mid:]

    def close(self) -> None:
        """The in-memory fake owns no external resources."""


class GateTimeoutError(RuntimeError):
    """A test gate stayed closed, so the synthesis fails instead of continuing."""


def _pass_gate(gate: threading.Event) -> None:
    if not gate.wait(timeout=GATE_TIMEOUT_SECONDS):
        message = "gate stayed closed"
        raise GateTimeoutError(message)


class GatedSpeaking:
    """A synthesis fake whose first and last PCM chunks each wait for a gate."""

    model_name = "gated-voice"
    ready = True
    streams = True
    retain_when_inactive = False
    sample_rate = SPEAK_SAMPLE_RATE

    def __init__(self, *, hold_last_chunk: bool = False) -> None:
        """Start with every gate open or the last chunk held."""
        self.entered_first_step = threading.Event()
        self.release_first_step = threading.Event()
        self.entered_last_step = threading.Event()
        self.release_last_chunk = threading.Event()
        if not hold_last_chunk:
            self.release_last_chunk.set()
        self._lock = threading.Lock()
        self.active_syntheses = 0
        self.most_concurrent_syntheses = 0
        self.audio = sine_pcm(0.05, rate=self.sample_rate)

    def release_all(self) -> None:
        """Open every gate so a synthesis blocked by a failed test can finish."""
        self.release_first_step.set()
        self.release_last_chunk.set()

    def load(self) -> None:
        """Mark the engine ready."""
        self.ready = True

    def pcm_chunks(self, text: str, language: str) -> Iterator[bytes]:
        """Yield the tone in two chunks, gated by the test's events."""
        del text, language
        with self._lock:
            self.active_syntheses += 1
            self.most_concurrent_syntheses = max(
                self.most_concurrent_syntheses, self.active_syntheses
            )
        try:
            self.entered_first_step.set()
            _pass_gate(self.release_first_step)
            mid = (len(self.audio) // 2) // BYTES_PER_SAMPLE * BYTES_PER_SAMPLE
            yield self.audio[:mid]
            self.entered_last_step.set()
            _pass_gate(self.release_last_chunk)
            yield self.audio[mid:]
        finally:
            with self._lock:
                self.active_syntheses -= 1


def an_app(
    speaking: SpeakingEngine | None = None,
    *,
    memory_probe: Callable[[], int] | None = None,
) -> FastAPI:
    runtime = Runtime(
        speaking or FakeSpeaking(),
        memory_probe=memory_probe if memory_probe is not None else (lambda: 0),
    )
    return create_control_app(Settings(), runtime, valid_callers())

"""One speech server process over fake engines, started by the two-process drain tests.

Run as `python -m tests.support.drain_server_support`; it serves `SPEECH_PORT` through
the real entry point, which fetches the key set over HTTPS as configured.
"""

from __future__ import annotations

import asyncio
import os
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING

from speech_client.wire import Language, VoiceId
from speech_service import __main__
from speech_service.clock import LoopClock
from speech_service.config import Settings
from speech_service.hearing import Heard, Hearing
from speech_service.service import Runtime, RuntimeDependencies
from tests.support.fakes_support import FakeSpeaking
from tests.support.hear_support import FakeHearing

if TYPE_CHECKING:
    from collections.abc import Iterator

RELEASE_FILE_VARIABLE = "DRAIN_RELEASE_FILE"
HEARING_HOLD_FILE_VARIABLE = "DRAIN_HEARING_HOLD_FILE"
HEARING_ENTERED_SUFFIX = ".entered"
_RELEASE_WAIT_SECONDS = 30.0
_RELEASE_POLL_SECONDS = 0.01


class HeldSpeaking(FakeSpeaking):
    """Speaks its first chunk, holds the rest until a file appears, and fails if closed.

    A close under a running stream is what a cut speak stream looks like.
    """

    def __init__(self, release_file: Path) -> None:
        """Hold every stream until `release_file` exists."""
        super().__init__()
        self._release_file = release_file
        self._closed = threading.Event()

    def pcm_chunks(
        self, text: str, language: str, instructions: str | None = None
    ) -> Iterator[bytes]:
        """The first chunk at once, the rest once released."""
        chunks = super().pcm_chunks(text, language, instructions)
        yield next(chunks)
        deadline = time.monotonic() + _RELEASE_WAIT_SECONDS
        while not self._release_file.exists():
            if self._closed.is_set() or time.monotonic() > deadline:
                message = "the engine was closed under a running stream"
                raise RuntimeError(message)
            time.sleep(_RELEASE_POLL_SECONDS)
        yield from chunks

    def close(self) -> None:
        """Fail every stream that is still held."""
        self._closed.set()


class HeldHearing(FakeHearing):
    """Hears at once, or once its hold file is gone, and notes it began to hear."""

    def __init__(self, hold_file: Path) -> None:
        """Hold every decode while `hold_file` exists."""
        super().__init__()
        self._hold_file = hold_file

    def transcribe(
        self,
        pcm: bytes,
        languages: tuple[Language, ...],
        voiced_seconds: float = 0.0,
    ) -> Heard:
        """Note the decode, wait out the hold, then hear as the fake does."""
        self._hold_file.with_suffix(HEARING_ENTERED_SUFFIX).touch()
        deadline = time.monotonic() + _RELEASE_WAIT_SECONDS
        while self._hold_file.exists() and time.monotonic() < deadline:
            time.sleep(_RELEASE_POLL_SECONDS)
        return super().transcribe(pcm, languages, voiced_seconds)


def _runtime() -> Runtime:
    release_file = Path(os.environ[RELEASE_FILE_VARIABLE])
    hold_file = Path(os.environ[HEARING_HOLD_FILE_VARIABLE])
    return Runtime(
        {VoiceId.THORSTEN: None},
        dependencies=RuntimeDependencies(
            engine_factory=lambda _voice: HeldSpeaking(release_file),
            missing_artifacts=lambda _voice: (),
        ),
        hearing=Hearing(HeldHearing(hold_file), LoopClock()),
        memory_probe=lambda: 0,
    )


def main() -> None:
    """Serve over the fake engines until a stop signal."""
    Runtime.from_settings = lambda _settings: _runtime()
    raise SystemExit(asyncio.run(__main__.serve(Settings())))


if __name__ == "__main__":
    main()

"""One speech server process over fake engines, started by the two-process drain tests.

Run as `python -m tests.support.drain_server_support`; it serves `SPEECH_PORT` through
the real entry point, trusting the key set in `DRAIN_KEY_SET`.
"""

from __future__ import annotations

import asyncio
import os
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING

from speech_client.wire import VoiceId
from speech_service import __main__
from speech_service.clock import LoopClock
from speech_service.config import Settings
from speech_service.hearing import Hearing
from speech_service.service import Runtime, RuntimeDependencies
from tests.support.fakes_support import FakeSpeaking
from tests.support.hear_support import FakeHearing

if TYPE_CHECKING:
    from collections.abc import Iterator

KEY_SET_VARIABLE = "DRAIN_KEY_SET"
RELEASE_FILE_VARIABLE = "DRAIN_RELEASE_FILE"
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


def _runtime() -> Runtime:
    release_file = Path(os.environ[RELEASE_FILE_VARIABLE])
    return Runtime(
        {VoiceId.THORSTEN: None},
        dependencies=RuntimeDependencies(
            engine_factory=lambda _voice: HeldSpeaking(release_file),
            missing_artifacts=lambda _voice: (),
        ),
        hearing=Hearing(FakeHearing(), LoopClock()),
        memory_probe=lambda: 0,
    )


def main() -> None:
    """Serve until a stop signal, trusting the key set the test hands over."""
    key_set = os.environ[KEY_SET_VARIABLE].encode()
    Runtime.from_settings = lambda _settings: _runtime()
    __main__.fetch_key_set_over_https = lambda _url: key_set
    raise SystemExit(asyncio.run(__main__.serve(Settings())))


if __name__ == "__main__":
    main()

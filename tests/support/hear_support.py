"""A fake hearing engine, recordings and multipart bodies for the transcription wire."""

from __future__ import annotations

import asyncio
import threading
from typing import TYPE_CHECKING

from speech_runtime_client.wire import (
    MODEL,
    TRANSCRIPTIONS_PATH,
    HearingLanguage,
)
from speech_runtime_service.config import Settings
from speech_runtime_service.control import create_control_app
from speech_runtime_service.hearing import Hearing
from speech_runtime_service.pcm import wav_header
from speech_runtime_service.service import Runtime
from tests.support.fakes_support import FakeSpeaking, sine_pcm
from tests.support.identity_support import valid_callers, valid_headers
from tests.support.speak_support import SpeakCall, speak_scope

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from httpx2 import Response as ClientResponse

BOUNDARY = "recording-boundary"
MULTIPART = f"multipart/form-data; boundary={BOUNDARY}"
RECORDING_RATE = 22_050
HEARD_TEXT = "Guten Tag"
WAIT_SECONDS = 5.0


class FakeHearing:
    """A hearing engine at the port: it answers a fixed text, or holds its decode."""

    model_name = "fake-ear"

    def __init__(
        self,
        text: str = HEARD_TEXT,
        *,
        load_error: Exception | None = None,
        hold: bool = False,
    ) -> None:
        """Answer `text`; fail the load with `load_error`; hold decodes if `hold`."""
        self.text = text
        self._load_error = load_error
        self.closed = 0
        self.heard: list[tuple[int, HearingLanguage]] = []
        self.entered = threading.Event()
        self.release = threading.Event()
        if not hold:
            self.release.set()
        self._guard = threading.Lock()
        self._active = 0
        self.most_concurrent = 0
        self.fail_transcription = False

    def load(self) -> None:
        """Raise the load error, if one was set."""
        if self._load_error is not None:
            raise self._load_error

    def transcribe(self, pcm: bytes, language: HearingLanguage) -> str:
        """Record the audio and language, wait for the test's release, answer."""
        with self._guard:
            self.heard.append((len(pcm), language))
            self._active += 1
            self.most_concurrent = max(self.most_concurrent, self._active)
        try:
            self.entered.set()
            if not self.release.wait(timeout=WAIT_SECONDS):
                message = "the decode was never released"
                raise TimeoutError(message)
            if self.fail_transcription:
                raise RuntimeError(self.text)
            return self.text
        finally:
            with self._guard:
                self._active -= 1

    def close(self) -> None:
        """Count a release of the engine's resources."""
        self.closed += 1


def loaded(engine: FakeHearing | None) -> Hearing:
    """Hearing over `engine`, loaded."""
    hearing = Hearing(engine)
    hearing.load()
    return hearing


def hear_runtime(engine: FakeHearing | None) -> Runtime:
    """A runtime whose voice speaks and whose hearing is `engine`, loaded."""
    return Runtime(FakeSpeaking(), memory_probe=lambda: 0, hearing=loaded(engine))


def hear_app(runtime: Runtime, **settings: float) -> FastAPI:
    """The private app over `runtime`, with an app that may hear."""
    return create_control_app(Settings(**settings), runtime, valid_callers(hear=True))


def recording(seconds: float = 0.5) -> bytes:
    """A WAV file of a tone `seconds` long."""
    pcm = sine_pcm(seconds, RECORDING_RATE)
    return wav_header(RECORDING_RATE, len(pcm)) + pcm


def multipart_body(
    fields: Mapping[str, str] | None = None, file: bytes | None = None
) -> bytes:
    """A multipart form of `fields` and, unless `file` is None, the file `file`.

    `fields` default to the one model name.
    """
    sent = {"model": MODEL} if fields is None else fields
    parts = [
        (
            f'--{BOUNDARY}\r\nContent-Disposition: form-data; name="{name}"'
            f"\r\n\r\n{value}\r\n"
        ).encode()
        for name, value in sent.items()
    ]
    if file is not None:
        parts.append(
            (
                f'--{BOUNDARY}\r\nContent-Disposition: form-data; name="file"; '
                'filename="recording.wav"\r\nContent-Type: audio/wav\r\n\r\n'
            ).encode()
            + file
            + b"\r\n"
        )
    return b"".join(parts) + f"--{BOUNDARY}--\r\n".encode()


def post_recording(
    client: TestClient,
    body: bytes,
    headers: Mapping[str, str] | None = None,
) -> ClientResponse:
    """POST `body` to the transcriptions path; `headers` default to a valid pair."""
    sent = valid_headers() if headers is None else headers
    return client.post(
        TRANSCRIPTIONS_PATH, content=body, headers={"content-type": MULTIPART, **sent}
    )


def hear_call(app: FastAPI, body: bytes | None = None) -> SpeakCall:
    """Start a transcription request now; `body` defaults to a valid recording."""
    scope = speak_scope(path=TRANSCRIPTIONS_PATH, content_type=MULTIPART)
    return SpeakCall(app, body or multipart_body(file=recording()), scope)


def hearing_gate_is_free(runtime: Runtime) -> bool:
    """Whether the hearing gate is free: no request left its lease behind."""
    lease = runtime.hearing.gate.try_acquire()
    if lease is None:
        return False
    lease.close()
    return True


async def hearing_gate_is_taken(runtime: Runtime) -> None:
    """Return once a request holds the hearing gate; its decode is then submitted."""
    async with asyncio.timeout(WAIT_SECONDS):
        while hearing_gate_is_free(runtime):  # noqa: ASYNC110 - the gate has no event to await
            await asyncio.sleep(0)


def write_snapshot(
    repository: str, *, revision: str, cache_dir: Path, allow_patterns: list[str]
) -> str:
    """The Hugging Face downloader's work: the files of `revision` in the cache."""
    snapshot = (
        cache_dir / f"models--{repository.replace('/', '--')}" / "snapshots" / revision
    )
    snapshot.mkdir(parents=True)
    for filename in allow_patterns:
        (snapshot / filename).write_text("weights")
    return str(snapshot)

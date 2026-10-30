"""Raw-ASGI drivers and WAV readers for the speak wire."""

import asyncio
import itertools
import json
import os
import threading
import time
from http import HTTPStatus
from typing import TYPE_CHECKING

from starlette.requests import Request

from speech_runtime_client.wire import SPEECH_PATH, RefusalCode, speech_request_body
from speech_runtime_service.config import Settings
from speech_runtime_service.control import create_control_app
from speech_runtime_service.selection import VoiceSelectionStore
from speech_runtime_service.service import Runtime, RuntimeDependencies
from speech_runtime_service.voices import VoiceId, VoiceLoadOutcome
from tests.support.fakes_support import FakeHearing, FakeSpeaking, GatedSpeaking

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from httpx2 import Response as ClientResponse
    from starlette.responses import Response, StreamingResponse

    from speech_runtime_service.service import SpeakingEngine
    from speech_runtime_service.synthesis_gate import Lease


def info_comment_before_data(wav: bytes) -> str | None:
    """The RIFF INFO ICMT text, found by walking the chunks up to `data`."""
    position = 12
    while position + 8 <= len(wav):
        chunk_id = wav[position : position + 4]
        size = int.from_bytes(wav[position + 4 : position + 8], "little")
        body = wav[position + 8 : position + 8 + size]
        if chunk_id == b"data":
            return None
        if chunk_id == b"LIST" and body[:4] == b"INFO":
            assert size % 2 == 0
            assert body[4:8] == b"ICMT"
            comment_size = int.from_bytes(body[8:12], "little")
            assert body[12 : 12 + comment_size].endswith(b"\0")
            return body[12 : 12 + comment_size - 1].decode("ascii")
        position += 8 + size + size % 2
    return None


ASGI_TIMEOUT_SECONDS = 5.0


def speak_app(
    speaking: SpeakingEngine | None = None, settings: Settings | None = None
) -> FastAPI:
    """The private app over a fake runtime, with the wire settings a test needs."""
    runtime = Runtime(speaking or FakeSpeaking(), FakeHearing(), memory_probe=lambda: 0)
    return create_control_app(settings or Settings(), runtime)


def speak_response(runtime: Runtime, voice: str = VoiceId.PIPER.value) -> Response:
    """Drive the wire route's endpoint with the default body for `voice`."""
    app = create_control_app(Settings(), runtime)
    route = next(
        route
        for route in app.routes
        if getattr(route, "path", None) == SPEECH_PATH
        and "POST" in getattr(route, "methods", ())
    )
    body = speak_body(voice)

    async def receive() -> dict[str, object]:
        return {"type": "http.request", "body": body, "more_body": False}

    return asyncio.run(route.endpoint(Request(speak_scope(), receive)))


def response_body(response: StreamingResponse) -> bytes:
    async def deliver() -> bytes:
        messages: list[dict[str, object]] = []
        never = asyncio.Event()

        async def receive() -> dict[str, object]:
            await never.wait()
            return {"type": "http.disconnect"}

        async def send(message: dict[str, object]) -> None:
            messages.append(message)

        await response(
            {"type": "http", "asgi": {"version": "3.0", "spec_version": "2.3"}},
            receive,
            send,
        )
        return wav_body(messages)

    return asyncio.run(deliver())


def speak_scope() -> dict[str, object]:
    return {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "POST",
        "path": SPEECH_PATH,
        "raw_path": SPEECH_PATH.encode(),
        "query_string": b"",
        "headers": [(b"content-type", b"application/json")],
        "client": ("test", 123),
        "server": ("test", 80),
    }


def speak_body(voice: str = VoiceId.PIPER.value, text: str = "Hallo") -> bytes:
    return speech_request_body(text, voice)


def wav_body(messages: list[dict[str, object]]) -> bytes:
    return b"".join(
        message["body"]
        for message in messages
        if message["type"] == "http.response.body"
    )


async def speak_once(
    app: FastAPI, voice: str = VoiceId.PIPER.value
) -> list[dict[str, object]]:
    """One full, undisturbed request through the raw ASGI boundary."""
    messages: list[dict[str, object]] = []
    sent = False
    no_disconnect = asyncio.Event()  # never set: this request is never disturbed

    async def receive() -> dict[str, object]:
        nonlocal sent
        if not sent:
            sent = True
            return {
                "type": "http.request",
                "body": speak_body(voice),
                "more_body": False,
            }
        await no_disconnect.wait()
        return {"type": "http.disconnect"}

    async def send(message: dict[str, object]) -> None:
        messages.append(message)

    await asyncio.wait_for(
        app(speak_scope(), receive, send), timeout=ASGI_TIMEOUT_SECONDS
    )
    return messages


class SpeakCall:
    """One request running as a task, so a test sees its status before its end."""

    def __init__(self, app: FastAPI, body: bytes | None = None) -> None:
        """Start the request now."""
        self.messages: list[dict[str, object]] = []
        self.began = time.monotonic()
        self.started_after = 0.0
        self.status_code = 0
        self.headers: dict[str, str] = {}
        self._started = asyncio.Event()
        self._disconnect = asyncio.Event()
        self.task = asyncio.create_task(self._run(app, body or speak_body()))

    async def _run(self, app: FastAPI, body: bytes) -> None:
        sent = False

        async def receive() -> dict[str, object]:
            nonlocal sent
            if not sent:
                sent = True
                return {"type": "http.request", "body": body, "more_body": False}
            await self._disconnect.wait()
            return {"type": "http.disconnect"}

        async def send(message: dict[str, object]) -> None:
            self.messages.append(message)
            if message["type"] == "http.response.start":
                self.started_after = time.monotonic() - self.began
                self.status_code = int(message["status"])
                self.headers = {
                    key.decode(): value.decode() for key, value in message["headers"]
                }
                self._started.set()

        await app(speak_scope(), receive, send)

    async def status(self) -> int:
        """Wait for the status line and return it."""
        await asyncio.wait_for(self._started.wait(), timeout=ASGI_TIMEOUT_SECONDS)
        return self.status_code

    async def finish(self) -> bytes:
        """Wait for the end of the response and return its whole body."""
        await asyncio.wait_for(self.task, timeout=ASGI_TIMEOUT_SECONDS)
        return wav_body(self.messages)


def fake_runtime(speaking: SpeakingEngine | None = None) -> Runtime:
    """A runtime over a fake voice whose active voice is Piper."""
    return Runtime(speaking or FakeSpeaking(), FakeHearing(), memory_probe=lambda: 0)


def app_over(runtime: Runtime, **settings: float) -> FastAPI:
    """The private app over `runtime` with the wire settings a test sets."""
    return create_control_app(Settings(**settings), runtime)


class CommittingSelectionStore(VoiceSelectionStore):
    """A selection store that tells when a switch to Piper has been written."""

    def __init__(self, directory: Path, *, owner_uid: int) -> None:
        """Start unsignalled."""
        super().__init__(directory, owner_uid=owner_uid)
        self.piper_committed = threading.Event()

    def write(self, voice: VoiceId) -> None:
        """Write the selection, then signal a switch to Piper."""
        super().write(voice)
        if voice is VoiceId.PIPER:
            self.piper_committed.set()


def switchable_runtime(
    tmp_path: Path,
    speaking: FakeSpeaking | GatedSpeaking,
    store: CommittingSelectionStore | None = None,
) -> Runtime:
    """A runtime whose active voice is Chatterbox and that can switch to Piper."""
    store = store or CommittingSelectionStore(
        tmp_path / "state", owner_uid=os.geteuid()
    )
    store.write(VoiceId.CHATTERBOX)
    speaking.retain_when_inactive = True
    return Runtime(
        speaking,
        FakeHearing(),
        memory_probe=lambda: 0,
        dependencies=RuntimeDependencies(
            selection=store,
            engine_factory=lambda _voice: FakeSpeaking(),
            artifact_checker=lambda _voice: True,
            default_voice=VoiceId.CHATTERBOX,
        ),
    )


def gate_acquisition_began(runtime: Runtime) -> asyncio.Event:
    """An event set when a request calls the gate's `acquire`.

    `acquire` queues its waiter before its first suspension, so a test that
    resumes on this event finds the waiter queued.
    """
    began = asyncio.Event()
    gate = runtime.synthesis_gate
    acquire = gate.acquire

    async def noting(deadline: float) -> Lease | None:
        began.set()
        return await acquire(deadline)

    gate.acquire = noting
    return began


def switch_completes(runtime: Runtime) -> bool:
    """Whether a switch to Piper finishes: no admission is left behind."""
    outcomes: list[VoiceLoadOutcome | None] = []
    switching = threading.Thread(
        target=lambda: outcomes.append(runtime.load_voice(VoiceId.PIPER))
    )
    switching.start()
    switching.join(timeout=ASGI_TIMEOUT_SECONDS)
    return not switching.is_alive() and outcomes == [VoiceLoadOutcome.ACTIVATED]


def post_speech(client: TestClient, body: bytes) -> ClientResponse:
    """POST `body` to the wire's speech path."""
    return client.post(
        SPEECH_PATH, content=body, headers={"content-type": "application/json"}
    )


def assert_refusal(
    response: ClientResponse, status: int, code: RefusalCode, param: str | None
) -> None:
    """The response is OpenAI's error form with exactly the closed keys."""
    assert response.headers["content-type"] == "application/json"
    assert_refusal_body(response.status_code, response.json(), status, code, param)


def assert_call_refusal(
    call: SpeakCall, status: int, code: RefusalCode, param: str | None
) -> None:
    """The finished call was refused in OpenAI's error form."""
    assert call.headers["content-type"] == "application/json"
    assert_refusal_body(
        call.status_code, json.loads(wav_body(call.messages)), status, code, param
    )


def assert_refusal_body(
    actual_status: int,
    body: dict[str, dict[str, object]],
    status: int,
    code: RefusalCode,
    param: str | None,
) -> None:
    """`body` is the closed error form of the refusal `status`, `code`, `param`."""
    assert actual_status == status
    assert set(body) == {"error"}
    assert set(body["error"]) == {"message", "type", "param", "code"}
    assert body["error"]["code"] == code
    assert body["error"]["param"] == param
    assert body["error"]["type"] == (
        "server_error"
        if status >= HTTPStatus.INTERNAL_SERVER_ERROR
        else "invalid_request_error"
    )


def assert_unavailable(response: ClientResponse) -> None:
    """The voice is refused as unavailable."""
    assert_refusal(
        response, HTTPStatus.SERVICE_UNAVAILABLE, RefusalCode.UNAVAILABLE, None
    )


def status_of(messages: list[dict[str, object]]) -> int:
    """The status of an ASGI response sent as `messages`."""
    return next(
        int(message["status"])
        for message in messages
        if message["type"] == "http.response.start"
    )


class FailingSpeaking(FakeSpeaking):
    """A voice that fails once, after the given number of chunks, then speaks."""

    def __init__(self, *, chunks_before_failing: int) -> None:
        """Fail the first synthesis after `chunks_before_failing` chunks."""
        super().__init__()
        self._chunks_before_failing = chunks_before_failing
        self.failed = False

    def pcm_chunks(self, text: str, language: str) -> Iterator[bytes]:
        """Yield the tone, or fail once."""
        if self.failed:
            yield from super().pcm_chunks(text, language)
            return
        self.failed = True
        yield from itertools.islice(
            super().pcm_chunks(text, language), self._chunks_before_failing
        )
        message = "engine failed"
        raise RuntimeError(message)

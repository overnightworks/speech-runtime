"""Raw-ASGI drivers and WAV readers for the speak route."""

import asyncio
import json

from fastapi import FastAPI
from starlette.responses import StreamingResponse

from presentator_speech.config import Settings
from presentator_speech.control import create_control_app
from presentator_speech.service import Runtime, SpeakRequest


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


def speak_response(runtime: Runtime) -> StreamingResponse:
    app = create_control_app(Settings(), runtime)
    route = next(
        route for route in app.routes if getattr(route, "path", None) == "/speak"
    )
    return route.endpoint(SpeakRequest(text="Hallo", language="de"))


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
        "path": "/speak",
        "raw_path": b"/speak",
        "query_string": b"",
        "headers": [(b"content-type", b"application/json")],
        "client": ("test", 123),
        "server": ("test", 80),
    }


def speak_body() -> bytes:
    return json.dumps({"text": "Hallo", "language": "de"}).encode()


def wav_body(messages: list[dict[str, object]]) -> bytes:
    return b"".join(
        message["body"]
        for message in messages
        if message["type"] == "http.response.body"
    )


async def speak_once(app: FastAPI) -> list[dict[str, object]]:
    """One full, undisturbed request through the raw ASGI boundary."""
    messages: list[dict[str, object]] = []
    sent = False
    no_disconnect = asyncio.Event()  # never set: this request is never disturbed

    async def receive() -> dict[str, object]:
        nonlocal sent
        if not sent:
            sent = True
            return {"type": "http.request", "body": speak_body(), "more_body": False}
        await no_disconnect.wait()
        return {"type": "http.disconnect"}

    async def send(message: dict[str, object]) -> None:
        messages.append(message)

    await asyncio.wait_for(
        app(speak_scope(), receive, send), timeout=ASGI_TIMEOUT_SECONDS
    )
    return messages

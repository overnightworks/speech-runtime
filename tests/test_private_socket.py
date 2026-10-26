"""speech.sock serves the speech routes and the voice controls over one runtime."""

import asyncio
import json
import os
import stat
import threading
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx2
import pytest
import uvicorn
from starlette.types import Message, Receive, Scope, Send
from websockets.asyncio.client import ClientConnection, unix_connect

from presentator_speech.__main__ import PrivateSocket
from presentator_speech.config import HEAR_SAMPLE_RATE, Settings
from presentator_speech.control import create_control_app
from presentator_speech.pcm import BYTES_PER_SAMPLE, WAV_HEADER_BYTES, pcm_from_wav
from presentator_speech.service import Runtime
from tests.conftest import (
    GATE_TIMEOUT_SECONDS,
    SPEAK_SAMPLE_RATE,
    FakeHearing,
    GatedSpeaking,
    sine_pcm,
)

_SPEAK_BODY = {"text": "Hallo", "language": "de"}


@dataclass(frozen=True, slots=True)
class _PrivateServer:
    path: Path
    hear_handler_returned: asyncio.Event
    speak_disconnect_received: asyncio.Event


@asynccontextmanager
async def _serving(directory: Path, runtime: Runtime) -> AsyncIterator[_PrivateServer]:
    """Serve the private app on a real PrivateSocket and stop it on exit."""
    settings = Settings(
        private_directory=directory, PRESENTATOR_RUNTIME_UID=os.geteuid()
    )
    app = create_control_app(settings, runtime)
    hear_handler_returned = asyncio.Event()
    speak_disconnect_received = asyncio.Event()

    async def observed(scope: Scope, receive: Receive, send: Send) -> None:
        async def watched_receive() -> Message:
            message = await receive()
            if message["type"] == "http.disconnect":
                speak_disconnect_received.set()
            return message

        try:
            await app(scope, watched_receive, send)
        finally:
            if scope["type"] == "websocket":
                hear_handler_returned.set()

    server = uvicorn.Server(uvicorn.Config(observed, log_level="warning"))
    with PrivateSocket(directory, runtime_uid=os.geteuid()) as listener:
        serving = asyncio.create_task(server.serve(sockets=[listener]))
        try:
            yield _PrivateServer(
                directory / "speech.sock",
                hear_handler_returned,
                speak_disconnect_received,
            )
        finally:
            server.should_exit = True
            await asyncio.wait_for(serving, timeout=GATE_TIMEOUT_SECONDS)


@asynccontextmanager
async def _gates_released(voice: GatedSpeaking) -> AsyncIterator[None]:
    """Open the gates on exit, so a failed assertion still lets the server stop."""
    try:
        yield
    finally:
        voice.release_all()


def _socket_client(server: _PrivateServer) -> httpx2.AsyncClient:
    return httpx2.AsyncClient(
        transport=httpx2.AsyncHTTPTransport(uds=str(server.path)),
        base_url="http://speech",
        timeout=GATE_TIMEOUT_SECONDS,
    )


def _gated_runtime(voice: GatedSpeaking, hearing: FakeHearing | None = None) -> Runtime:
    return Runtime(voice, hearing or FakeHearing(), memory_probe=lambda: 0)


async def _first_chunk(chunks: AsyncIterator[bytes]) -> bytes:
    first = b""
    while len(first) <= WAV_HEADER_BYTES:
        first += await asyncio.wait_for(anext(chunks), timeout=GATE_TIMEOUT_SECONDS)
    return first


async def _wait_for(event: threading.Event) -> None:
    assert await asyncio.to_thread(event.wait, GATE_TIMEOUT_SECONDS)


def _assert_whole_wav(status_code: int, body: bytes, voice: GatedSpeaking) -> None:
    assert status_code == 200
    rate, pcm = pcm_from_wav(body)
    assert rate == SPEAK_SAMPLE_RATE
    assert pcm == voice.audio


async def _receive_event(connection: ClientConnection) -> dict[str, Any]:
    message = await asyncio.wait_for(connection.recv(), timeout=GATE_TIMEOUT_SECONDS)
    return json.loads(message)


async def _speak_then_pause(connection: ClientConnection) -> None:
    await connection.send(sine_pcm(0.5))
    partial = await _receive_event(connection)
    assert partial["final"] is False
    assert "eins" in partial["text"]
    await connection.send(bytes(int(0.8 * HEAR_SAMPLE_RATE * BYTES_PER_SAMPLE)))
    assert (await _receive_event(connection))["final"] is True


def _hear_uri() -> str:
    return "ws://speech/hear?language=de"


def test_socket_streams_speak_while_the_voice_is_still_synthesizing(tmp_path) -> None:
    async def scenario() -> tuple[GatedSpeaking, int, bytes, bytes, list[int]]:
        voice = GatedSpeaking(hold_last_chunk=True)
        voice.release_first_step.set()
        async with (
            _serving(tmp_path, _gated_runtime(voice)) as server,
            _socket_client(server) as client,
            _gates_released(voice),
            client.stream("POST", "/speak", json=_SPEAK_BODY) as response,
        ):
            assert response.headers["content-type"] == "audio/wav"
            chunks = response.aiter_bytes()
            first = await _first_chunk(chunks)
            assert not voice.release_last_chunk.is_set()
            modes = [
                stat.S_IMODE(tmp_path.stat().st_mode),
                stat.S_IMODE(server.path.stat().st_mode),
                tmp_path.stat().st_uid,
                server.path.stat().st_uid,
            ]
            voice.release_last_chunk.set()
            rest = b"".join([chunk async for chunk in chunks])
        return voice, response.status_code, first, first + rest, modes

    voice, status_code, first, whole, modes = asyncio.run(scenario())

    first_rate, first_pcm = pcm_from_wav(first)
    assert first_rate == SPEAK_SAMPLE_RATE
    assert 0 < len(first_pcm) < len(voice.audio)
    _assert_whole_wav(status_code, whole, voice)
    assert modes == [0o700, 0o600, os.geteuid(), os.geteuid()]


def test_socket_speak_disconnect_frees_the_voice_for_the_next_caller(tmp_path) -> None:
    async def scenario() -> tuple[GatedSpeaking, httpx2.Response]:
        voice = GatedSpeaking(hold_last_chunk=True)
        voice.release_first_step.set()
        async with (
            _serving(tmp_path, _gated_runtime(voice)) as server,
            _socket_client(server) as client,
            _gates_released(voice),
        ):
            async with client.stream("POST", "/speak", json=_SPEAK_BODY) as leaving:
                await _first_chunk(leaving.aiter_bytes())
                await _wait_for(voice.entered_last_step)
            await asyncio.wait_for(
                server.speak_disconnect_received.wait(), timeout=GATE_TIMEOUT_SECONDS
            )
            voice.release_last_chunk.set()
            return voice, await client.post("/speak", json=_SPEAK_BODY)

    voice, later = asyncio.run(scenario())

    _assert_whole_wav(later.status_code, later.content, voice)


def test_a_speak_holds_the_one_synthesis_gate(tmp_path) -> None:
    async def scenario() -> tuple[
        GatedSpeaking, httpx2.Response, httpx2.Response, httpx2.Response
    ]:
        voice = GatedSpeaking()
        runtime = _gated_runtime(voice)
        async with (
            _serving(tmp_path, runtime) as server,
            _socket_client(server) as socket_client,
            _gates_released(voice),
        ):
            talk = asyncio.create_task(socket_client.post("/speak", json=_SPEAK_BODY))
            await _wait_for(voice.entered_first_step)
            busy = await socket_client.post("/voices/piper/sample/de")
            voice.release_first_step.set()
            talked = await asyncio.wait_for(talk, timeout=GATE_TIMEOUT_SECONDS)
            free = await socket_client.post("/voices/piper/sample/de")
        return voice, busy, talked, free

    voice, busy, talked, free = asyncio.run(scenario())

    assert (busy.status_code, busy.json()) == (409, {"detail": "speech is busy"})
    assert voice.most_concurrent_syntheses == 1
    _assert_whole_wav(talked.status_code, talked.content, voice)
    _assert_whole_wav(free.status_code, free.content, voice)


def test_socket_hear_sends_partials_then_a_final_and_a_second_utterance(
    tmp_path,
) -> None:
    async def scenario() -> dict[str, Any]:
        runtime = _gated_runtime(GatedSpeaking())
        async with (
            _serving(tmp_path, runtime) as server,
            unix_connect(str(server.path), uri=_hear_uri()) as connection,
        ):
            await _speak_then_pause(connection)
            await connection.send(sine_pcm(0.5))
            return await _receive_event(connection)

    second = asyncio.run(scenario())

    assert second["final"] is False
    assert "eins" in second["text"]


def test_socket_hear_disconnect_runs_no_decode_and_frees_the_next_session(
    tmp_path,
) -> None:
    async def scenario() -> tuple[int, int]:
        hearing = FakeHearing()
        runtime = _gated_runtime(GatedSpeaking(), hearing)
        async with _serving(tmp_path, runtime) as server:
            async with unix_connect(str(server.path), uri=_hear_uri()) as leaving:
                await leaving.send(sine_pcm(0.5))
                assert (await _receive_event(leaving))["final"] is False
                at_close = hearing.transcriptions
                leaving.transport.abort()
                await asyncio.wait_for(
                    server.hear_handler_returned.wait(), timeout=GATE_TIMEOUT_SECONDS
                )
            after_handler = hearing.transcriptions
            async with unix_connect(str(server.path), uri=_hear_uri()) as staying:
                await _speak_then_pause(staying)
        return at_close, after_handler

    at_close, after_handler = asyncio.run(scenario())

    # No audio follows the partial, so no poll decodes; a final decode would count.
    assert after_handler == at_close


@pytest.mark.parametrize("path", ["/docs", "/redoc", "/openapi.json"])
def test_socket_serves_no_api_documentation(tmp_path, path: str) -> None:
    async def scenario() -> httpx2.Response:
        async with (
            _serving(tmp_path, _gated_runtime(GatedSpeaking())) as server,
            _socket_client(server) as client,
        ):
            return await client.get(path)

    assert asyncio.run(scenario()).status_code == 404

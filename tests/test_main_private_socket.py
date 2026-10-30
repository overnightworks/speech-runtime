"""speech.sock serves the speech routes over one runtime."""

import asyncio
import os
import stat
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

import httpx2
import pytest
from starlette.status import HTTP_200_OK, HTTP_401_UNAUTHORIZED, HTTP_404_NOT_FOUND

from speech_runtime_client.wire import AUTHORIZATION_HEADER, SPEECH_PATH
from speech_runtime_service.pcm import WAV_HEADER_BYTES, pcm_from_wav
from speech_runtime_service.service import Runtime
from tests.support.fakes_support import (
    GATE_TIMEOUT_SECONDS,
    SPEAK_SAMPLE_RATE,
    FakeSpeaking,
    GatedSpeaking,
)
from tests.support.identity_support import valid_headers
from tests.support.private_server_support import PrivateServer, serving_privately
from tests.support.speak_support import FailingSpeaking, gate_is_free, speak_body

if TYPE_CHECKING:
    import threading
    from collections.abc import AsyncIterator, Mapping
    from pathlib import Path

_SPEAK_BODY = speak_body()


@asynccontextmanager
async def _gates_released(voice: GatedSpeaking) -> AsyncIterator[None]:
    """Open the gates on exit, so a failed assertion still lets the server stop."""
    try:
        yield
    finally:
        voice.release_all()


def _socket_client(
    server: PrivateServer, headers: Mapping[str, str] | None = None
) -> httpx2.AsyncClient:
    """A client on the socket; `headers` default to a valid pair of tokens."""
    return httpx2.AsyncClient(
        transport=httpx2.AsyncHTTPTransport(uds=str(server.path)),
        base_url="http://speech",
        timeout=GATE_TIMEOUT_SECONDS,
        headers=valid_headers() if headers is None else headers,
    )


def _gated_runtime(voice: FakeSpeaking | GatedSpeaking) -> Runtime:
    return Runtime(voice, memory_probe=lambda: 0)


async def _first_chunk(chunks: AsyncIterator[bytes]) -> bytes:
    first = b""
    while len(first) <= WAV_HEADER_BYTES:
        first += await asyncio.wait_for(anext(chunks), timeout=GATE_TIMEOUT_SECONDS)
    return first


async def _wait_for(event: threading.Event) -> None:
    assert await asyncio.to_thread(event.wait, GATE_TIMEOUT_SECONDS)


def _assert_whole_wav(status_code: int, body: bytes, voice: GatedSpeaking) -> None:
    assert status_code == HTTP_200_OK
    rate, pcm = pcm_from_wav(body)
    assert rate == SPEAK_SAMPLE_RATE
    assert pcm == voice.audio


def _modes_and_owners(directory: Path, socket_path: Path) -> list[int]:
    return [
        stat.S_IMODE(directory.stat().st_mode),
        stat.S_IMODE(socket_path.stat().st_mode),
        directory.stat().st_uid,
        socket_path.stat().st_uid,
    ]


def test_socket_streams_speak_while_the_voice_is_still_synthesizing(
    tmp_path: Path,
) -> None:
    async def scenario() -> tuple[GatedSpeaking, int, bytes, bytes, list[int]]:
        voice = GatedSpeaking(hold_last_chunk=True)
        voice.release_first_step.set()
        async with (
            serving_privately(tmp_path, _gated_runtime(voice)) as server,
            _socket_client(server) as client,
            _gates_released(voice),
            client.stream("POST", SPEECH_PATH, content=_SPEAK_BODY) as response,
        ):
            assert response.headers["content-type"] == "audio/wav"
            chunks = response.aiter_bytes()
            first = await _first_chunk(chunks)
            assert not voice.release_last_chunk.is_set()
            modes = await asyncio.to_thread(_modes_and_owners, tmp_path, server.path)
            voice.release_last_chunk.set()
            rest = b"".join([chunk async for chunk in chunks])
        return voice, response.status_code, first, first + rest, modes

    voice, status_code, first, whole, modes = asyncio.run(scenario())

    first_rate, first_pcm = pcm_from_wav(first)
    assert first_rate == SPEAK_SAMPLE_RATE
    assert 0 < len(first_pcm) < len(voice.audio)
    _assert_whole_wav(status_code, whole, voice)
    assert modes == [0o700, 0o600, os.geteuid(), os.geteuid()]


def test_socket_speak_disconnect_frees_the_voice_for_the_next_caller(
    tmp_path: Path,
) -> None:
    async def scenario() -> tuple[GatedSpeaking, httpx2.Response]:
        voice = GatedSpeaking(hold_last_chunk=True)
        voice.release_first_step.set()
        async with (
            serving_privately(tmp_path, _gated_runtime(voice)) as server,
            _socket_client(server) as client,
            _gates_released(voice),
        ):
            async with client.stream(
                "POST", SPEECH_PATH, content=_SPEAK_BODY
            ) as leaving:
                await _first_chunk(leaving.aiter_bytes())
                await _wait_for(voice.entered_last_step)
            await asyncio.wait_for(
                server.speak_disconnect_received.wait(), timeout=GATE_TIMEOUT_SECONDS
            )
            voice.release_last_chunk.set()
            return voice, await client.post(SPEECH_PATH, content=_SPEAK_BODY)

    voice, later = asyncio.run(scenario())

    _assert_whole_wav(later.status_code, later.content, voice)


def test_a_speak_holds_the_one_synthesis_gate(tmp_path: Path) -> None:
    async def scenario() -> tuple[GatedSpeaking, bool, httpx2.Response, bool]:
        voice = GatedSpeaking()
        runtime = _gated_runtime(voice)
        async with (
            serving_privately(tmp_path, runtime) as server,
            _socket_client(server) as socket_client,
            _gates_released(voice),
        ):
            talk = asyncio.create_task(
                socket_client.post(SPEECH_PATH, content=_SPEAK_BODY)
            )
            await _wait_for(voice.entered_first_step)
            held = runtime.synthesis_gate.try_acquire() is None
            voice.release_first_step.set()
            talked = await asyncio.wait_for(talk, timeout=GATE_TIMEOUT_SECONDS)
            return voice, held, talked, gate_is_free(runtime)

    voice, held, talked, free = asyncio.run(scenario())

    assert held
    assert voice.most_concurrent_syntheses == 1
    _assert_whole_wav(talked.status_code, talked.content, voice)
    assert free


@pytest.mark.parametrize("chunks_before_failing", [0, 1])
def test_socket_speak_failure_aborts_the_body_and_the_next_request_is_whole(
    tmp_path: Path, chunks_before_failing: int
) -> None:
    async def scenario() -> tuple[bool, httpx2.Response]:
        voice = FailingSpeaking(chunks_before_failing=chunks_before_failing)
        async with (
            serving_privately(tmp_path, _gated_runtime(voice)) as server,
            _socket_client(server) as client,
        ):
            try:
                await client.post(SPEECH_PATH, content=_SPEAK_BODY)
            except httpx2.RemoteProtocolError:
                aborted = True
            else:
                aborted = False
            return aborted, await client.post(SPEECH_PATH, content=_SPEAK_BODY)

    aborted, later = asyncio.run(scenario())

    assert aborted
    assert later.status_code == HTTP_200_OK
    assert pcm_from_wav(later.content)[1]


@pytest.mark.parametrize("path", ["/docs", "/redoc", "/openapi.json"])
def test_socket_serves_no_api_documentation(tmp_path: Path, path: str) -> None:
    async def scenario() -> httpx2.Response:
        async with (
            serving_privately(tmp_path, _gated_runtime(GatedSpeaking())) as server,
            _socket_client(server) as client,
        ):
            return await client.get(path)

    assert asyncio.run(scenario()).status_code == HTTP_404_NOT_FOUND


def test_the_socket_refuses_a_speak_without_the_persons_token(
    tmp_path: Path,
) -> None:
    async def scenario() -> tuple[httpx2.Response, httpx2.Response]:
        headers = valid_headers()
        del headers[AUTHORIZATION_HEADER]
        async with (
            serving_privately(tmp_path, _gated_runtime(FakeSpeaking())) as server,
            _socket_client(server, headers) as client,
            _socket_client(server) as admitted,
        ):
            return (
                await client.post(SPEECH_PATH, content=_SPEAK_BODY),
                await admitted.post(SPEECH_PATH, content=_SPEAK_BODY),
            )

    refused, served = asyncio.run(scenario())

    assert refused.status_code == HTTP_401_UNAUTHORIZED
    assert served.status_code == HTTP_200_OK

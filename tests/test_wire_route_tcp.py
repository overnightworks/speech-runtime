"""The speech routes over one runtime answer on a real TCP listener."""

import asyncio
import base64
import os
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING
from urllib.parse import urlsplit

import httpx2
import pytest
from starlette.status import (
    HTTP_200_OK,
    HTTP_400_BAD_REQUEST,
    HTTP_401_UNAUTHORIZED,
    HTTP_404_NOT_FOUND,
)

from speech_runtime_client.wire import AUTHORIZATION_HEADER, SPEECH_PATH, RefusalCode
from speech_runtime_service.pcm import WAV_HEADER_BYTES, pcm_from_wav
from speech_runtime_service.service import Runtime
from tests.support.fakes_support import (
    GATE_TIMEOUT_SECONDS,
    SPEAK_SAMPLE_RATE,
    FakeSpeaking,
    GatedSpeaking,
)
from tests.support.identity_support import valid_headers
from tests.support.speak_support import (
    FailingSpeaking,
    assert_refusal,
    gate_is_free,
    speak_body,
)
from tests.support.tcp_server_support import TcpServer, serving_over_tcp

if TYPE_CHECKING:
    import threading
    from collections.abc import AsyncIterator, Mapping

_SPEAK_BODY = speak_body()


@asynccontextmanager
async def _gates_released(voice: GatedSpeaking) -> AsyncIterator[None]:
    """Open the gates on exit, so a failed assertion still lets the server stop."""
    try:
        yield
    finally:
        voice.release_all()


def _client(
    server: TcpServer, headers: Mapping[str, str] | None = None
) -> httpx2.AsyncClient:
    """A client on the listener; `headers` default to a valid pair of tokens."""
    return httpx2.AsyncClient(
        base_url=server.base_url,
        timeout=GATE_TIMEOUT_SECONDS,
        headers=valid_headers() if headers is None else headers,
    )


def _address(server: TcpServer) -> tuple[str, int]:
    address = urlsplit(server.base_url)
    return address.hostname or "", address.port or 0


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


def test_tcp_streams_speak_while_the_voice_is_still_synthesizing() -> None:
    async def scenario() -> tuple[GatedSpeaking, int, bytes, bytes]:
        voice = GatedSpeaking(hold_last_chunk=True)
        voice.release_first_step.set()
        async with (
            serving_over_tcp(_gated_runtime(voice)) as server,
            _client(server) as client,
            _gates_released(voice),
            client.stream("POST", SPEECH_PATH, content=_SPEAK_BODY) as response,
        ):
            assert response.headers["content-type"] == "audio/wav"
            chunks = response.aiter_bytes()
            first = await _first_chunk(chunks)
            assert not voice.release_last_chunk.is_set()
            voice.release_last_chunk.set()
            rest = b"".join([chunk async for chunk in chunks])
        return voice, response.status_code, first, first + rest

    voice, status_code, first, whole = asyncio.run(scenario())

    first_rate, first_pcm = pcm_from_wav(first)
    assert first_rate == SPEAK_SAMPLE_RATE
    assert 0 < len(first_pcm) < len(voice.audio)
    _assert_whole_wav(status_code, whole, voice)


def test_tcp_speak_disconnect_frees_the_voice_for_the_next_caller() -> None:
    async def scenario() -> tuple[GatedSpeaking, httpx2.Response]:
        voice = GatedSpeaking(hold_last_chunk=True)
        voice.release_first_step.set()
        async with (
            serving_over_tcp(_gated_runtime(voice)) as server,
            _client(server) as client,
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


def test_a_speak_holds_the_one_synthesis_gate() -> None:
    async def scenario() -> tuple[GatedSpeaking, bool, httpx2.Response, bool]:
        voice = GatedSpeaking()
        runtime = _gated_runtime(voice)
        async with (
            serving_over_tcp(runtime) as server,
            _client(server) as socket_client,
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
def test_tcp_speak_failure_aborts_the_body_and_the_next_request_is_whole(
    chunks_before_failing: int,
) -> None:
    async def scenario() -> tuple[bool, httpx2.Response]:
        voice = FailingSpeaking(chunks_before_failing=chunks_before_failing)
        async with (
            serving_over_tcp(_gated_runtime(voice)) as server,
            _client(server) as client,
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
def test_tcp_serves_no_api_documentation(path: str) -> None:
    async def scenario() -> httpx2.Response:
        async with (
            serving_over_tcp(_gated_runtime(GatedSpeaking())) as server,
            _client(server) as client,
        ):
            return await client.get(path)

    assert asyncio.run(scenario()).status_code == HTTP_404_NOT_FOUND


def test_tcp_refuses_a_speak_without_the_persons_token() -> None:
    async def scenario() -> tuple[httpx2.Response, httpx2.Response]:
        headers = valid_headers()
        del headers[AUTHORIZATION_HEADER]
        async with (
            serving_over_tcp(_gated_runtime(FakeSpeaking())) as server,
            _client(server, headers) as client,
            _client(server) as admitted,
        ):
            return (
                await client.post(SPEECH_PATH, content=_SPEAK_BODY),
                await admitted.post(SPEECH_PATH, content=_SPEAK_BODY),
            )

    refused, served = asyncio.run(scenario())

    assert refused.status_code == HTTP_401_UNAUTHORIZED
    assert served.status_code == HTTP_200_OK


@pytest.mark.parametrize(
    ("method", "path", "status", "code"),
    [
        ("GET", "/health", HTTP_200_OK, None),
        ("POST", SPEECH_PATH, HTTP_401_UNAUTHORIZED, RefusalCode.UNAUTHORIZED),
        ("POST", "/v1/other", HTTP_404_NOT_FOUND, RefusalCode.NOT_FOUND),
        (
            "POST",
            "/v2/audio/speech",
            HTTP_400_BAD_REQUEST,
            RefusalCode.UNSUPPORTED_VERSION,
        ),
        ("GET", "/voices", HTTP_404_NOT_FOUND, None),
        ("GET", "/hear", HTTP_404_NOT_FOUND, None),
    ],
)
def test_tcp_answers_without_tokens_only_health_and_the_wires_plain_refusals(
    method: str, path: str, status: int, code: RefusalCode | None
) -> None:
    async def scenario() -> httpx2.Response:
        async with (
            serving_over_tcp(_gated_runtime(FakeSpeaking())) as server,
            _client(server, {}) as client,
        ):
            return await client.request(method, path, content=_SPEAK_BODY)

    response = asyncio.run(scenario())

    if code is None:
        assert response.status_code == status
    else:
        assert_refusal(response, status, code, None)


def test_tcp_refuses_a_websocket_upgrade_on_hear() -> None:
    upgrade = (
        b"GET /hear HTTP/1.1\r\nHost: speech\r\nUpgrade: websocket\r\n"
        b"Connection: Upgrade\r\nSec-WebSocket-Version: 13\r\n"
        b"Sec-WebSocket-Key: " + base64.b64encode(os.urandom(16)) + b"\r\n\r\n"
    )

    async def scenario() -> bytes:
        async with serving_over_tcp(_gated_runtime(FakeSpeaking())) as server:
            host, port = _address(server)
            reader, writer = await asyncio.open_connection(host, port)
            writer.write(upgrade)
            status_line = await asyncio.wait_for(
                reader.readline(), timeout=GATE_TIMEOUT_SECONDS
            )
            writer.close()
            return status_line

    status_line = asyncio.run(scenario())

    assert status_line.split()[1] == b"404"

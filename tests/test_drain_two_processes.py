"""A stop signal drains one of two server processes while the other takes new work."""

from __future__ import annotations

import asyncio
import http.client
import json
import os
import signal
import subprocess
import sys
import time
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosed, InvalidStatus

from speech_client.wire import REALTIME_PATH, SPEECH_PATH, VoiceId
from tests.support.balancer_support import balancing, is_ready
from tests.support.drain_server_support import (
    KEY_SET_VARIABLE,
    RELEASE_FILE_VARIABLE,
)
from tests.support.identity_support import (
    DEFAULT_KEY,
    valid_headers,
    write_caller_configuration,
)
from tests.support.live_support import COMPLETED, FAILED, append, audio
from tests.support.speak_support import speak_body
from tests.support.tcp_server_support import free_loopback_port, wait_until_listening

if TYPE_CHECKING:
    from collections.abc import Iterator

    from websockets.asyncio.client import ClientConnection

_LOOPBACK = "127.0.0.1"
_WAIT_SECONDS = 30
_SERVICE_RESTART = 1012
_SESSION_URL = "ws://{host}:{port}" + f"{REALTIME_PATH}?intent=transcription"


@dataclass(frozen=True, slots=True)
class _Pair:
    """Two server processes of one fake service, and the file that releases speech."""

    first: subprocess.Popen[bytes]
    second: subprocess.Popen[bytes]
    first_port: int
    second_port: int
    release_file: Path


@pytest.fixture
def pair(tmp_path: Path) -> Iterator[_Pair]:
    speech_root = Path(__file__).resolve().parents[1]
    release_file = tmp_path / "release-speech"
    environment = {
        **os.environ,
        "SPEECH_CONFIG_FILE": str(write_caller_configuration(tmp_path)),
        KEY_SET_VARIABLE: json.dumps({"keys": [DEFAULT_KEY.public_jwk]}),
        RELEASE_FILE_VARIABLE: str(release_file),
        "PYTHONPATH": os.pathsep.join([str(speech_root), str(speech_root / "scripts")]),
    }
    ports = [free_loopback_port(), free_loopback_port()]
    with ExitStack() as processes:
        started = [
            processes.enter_context(
                subprocess.Popen(
                    [sys.executable, "-m", "tests.support.drain_server_support"],
                    env={**environment, "SPEECH_PORT": str(port)},
                    cwd=tmp_path,
                )
            )
            for port in ports
        ]
        processes.callback(lambda: [process.kill() for process in started])
        for process, port in zip(started, ports, strict=True):
            wait_until_listening(process, port)
        deadline = time.monotonic() + _WAIT_SECONDS
        while not all(is_ready(port) for port in ports):
            assert time.monotonic() < deadline, "the servers never became ready"
            time.sleep(0.05)
        yield _Pair(started[0], started[1], ports[0], ports[1], release_file)


async def _received(connection: ClientConnection) -> dict[str, object]:
    return json.loads(await asyncio.wait_for(connection.recv(), _WAIT_SECONDS))


async def _until_closed(
    connection: ClientConnection,
) -> tuple[list[dict[str, object]], int]:
    """The events that arrive before the server closes, and the code it closes with."""
    events: list[dict[str, object]] = []
    while True:
        try:
            events.append(await _received(connection))
        except ConnectionClosed as closed:
            return events, closed.rcvd.code if closed.rcvd else -1


async def _open_session(port: int) -> ClientConnection:
    connection = await connect(
        _SESSION_URL.format(host=_LOOPBACK, port=port),
        additional_headers=valid_headers(),
    )
    assert (await _received(connection))["type"] == "session.created"
    return connection


def _stream_speech(port: int) -> tuple[http.client.HTTPResponse, bytes]:
    """Start a speak request and read the first byte of its body."""
    connection = http.client.HTTPConnection(_LOOPBACK, port, timeout=_WAIT_SECONDS)
    connection.request(
        "POST", SPEECH_PATH, body=speak_body(VoiceId.THORSTEN), headers=valid_headers()
    )
    response = connection.getresponse()
    assert response.status == http.HTTPStatus.OK
    return response, response.read(1)


async def _until_not_ready(port: int) -> None:
    deadline = time.monotonic() + _WAIT_SECONDS
    while await asyncio.to_thread(is_ready, port):
        assert time.monotonic() < deadline, "the stopped server still reports ready"
        await asyncio.sleep(0.05)


def test_a_stopped_process_finishes_its_turn_and_stream_while_another_takes_work(
    pair: _Pair,
) -> None:
    """WIRE-144, WIRE-145, WIRE-146: the stopped process drains, the other serves."""

    async def scenario() -> tuple[list[dict[str, object]], int, bytes, int]:
        async with balancing([pair.first_port, pair.second_port]) as entry:
            hearing = await _open_session(entry)
            await hearing.send(append(audio(0.5)))
            assert (await _received(hearing))["type"] == (
                "input_audio_buffer.speech_started"
            )
            response, first_byte = await asyncio.to_thread(_stream_speech, entry)

            pair.first.send_signal(signal.SIGTERM)
            await _until_not_ready(pair.first_port)

            with pytest.raises(InvalidStatus) as refused:
                await connect(
                    _SESSION_URL.format(host=_LOOPBACK, port=pair.first_port),
                    additional_headers=valid_headers(),
                )
            assert (
                refused.value.response.status_code
                == http.HTTPStatus.SERVICE_UNAVAILABLE
            )
            other = await _open_session(entry)
            await other.close()

            await hearing.send(append(audio(1.0, level=0)))
            events, code = await _until_closed(hearing)
            pair.release_file.touch()
            body = first_byte + await asyncio.to_thread(response.read)
            return (
                events,
                code,
                body,
                await asyncio.to_thread(pair.first.wait, _WAIT_SECONDS),
            )

    events, code, speech, exit_code = asyncio.run(scenario())

    types = [event["type"] for event in events]
    assert types[-1] == COMPLETED
    assert "error" not in types
    assert FAILED not in types
    assert code == _SERVICE_RESTART
    assert speech.startswith(b"RIFF")
    assert exit_code == 0
    assert pair.second.poll() is None


def test_a_stopped_process_closes_an_idle_session_at_once(pair: _Pair) -> None:
    """WIRE-146: a session with no open turn closes 1012 when the process is stopped."""

    async def scenario() -> tuple[list[dict[str, object]], int, int]:
        hearing = await _open_session(pair.first_port)
        pair.first.send_signal(signal.SIGTERM)
        events, code = await _until_closed(hearing)
        return events, code, await asyncio.to_thread(pair.first.wait, _WAIT_SECONDS)

    events, code, exit_code = asyncio.run(scenario())

    assert events == []
    assert code == _SERVICE_RESTART
    assert exit_code == 0

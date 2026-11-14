"""A stop refuses what waits for a GPU voice, reaps its worker, exits by its cause."""

from __future__ import annotations

import asyncio
import http.client
import json
import time
from http import HTTPStatus
from typing import TYPE_CHECKING

import httpx2
import pytest

from speech_client.wire import SPEECH_PATH, RefusalCode, VoiceId
from tests.support.fakes_support import GatedSpeaking
from tests.support.gpu_voice_support import (
    LOOPBACK,
    LoadWaits,
    Serving,
    StageSetup,
    gpu_voice_stage,
    speak_headers,
)
from tests.support.speak_support import (
    SpeakCall,
    app_over,
    assert_call_refusal,
    assert_refusal_body,
    assert_unavailable,
    fake_runtime,
    gate_acquisition_began,
    speak_body,
)
from tests.support.tcp_server_support import listens_on
from tests.support.worker_support import (
    WORKER_WAIT_SECONDS,
    LoadOutcome,
    assert_reaped,
    hold_exits,
    is_running,
    release_exit,
    release_load,
    wait_for_held_exit,
    wait_for_starts,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable
    from pathlib import Path

    from tests.support.gpu_voice_support import GpuVoiceStage

_LOADING = StageSetup(started=False)
_SERVING = StageSetup("normal", started=False)
_DRAIN_BOUND_SECONDS = 1.0
# The fake worker holds a stream until the engine closes, so a stream the test
# leaves running lasts until the bound.
_SHORT_DRAIN = StageSetup(
    "normal", started=False, drain_turn_seconds=_DRAIN_BOUND_SECONDS / 2
)
_SHORT_DRAIN_WAIT = {"hearing_final_wait_seconds": _DRAIN_BOUND_SECONDS / 2}


def test_sigterm_during_a_load_refuses_its_waiters_reaps_the_worker_and_exits_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def scenario(serving: Serving, stage: GpuVoiceStage) -> tuple[int, list]:
        async with await serving.start() as client:
            waits = LoadWaits(stage.runtime)
            waiters = [
                asyncio.create_task(
                    client.post(SPEECH_PATH, content=speak_body(VoiceId.RYAN))
                )
                for _ in range(2)
            ]
            await asyncio.to_thread(waits.wait_for, 2)
            await asyncio.to_thread(wait_for_starts, stage.cache, 1)
            serving.sigterm()
            outcome = await serving.task
            return outcome, await asyncio.gather(*waiters)

    with gpu_voice_stage(tmp_path, monkeypatch, _LOADING) as stage:
        serving = Serving(stage, monkeypatch)
        with serving.listener:
            outcome, responses = asyncio.run(scenario(serving, stage))

    assert outcome == 0
    for response in responses:
        assert_unavailable(response)
    (pid,) = stage.starts()
    assert_reaped(pid)
    assert serving.fatal.is_set() is False


@pytest.mark.parametrize(
    ("reap_fails", "exit_status"),
    [(False, 0), (True, 1)],
    ids=["reaped", "unreapable"],
)
def test_sigterm_during_a_failed_loads_cleanup_waits_for_it_and_exits_by_its_reap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    reap_fails: bool,
    exit_status: int,
) -> None:
    async def scenario(serving: Serving, stage: GpuVoiceStage) -> tuple[bool, int]:
        async with await serving.start() as client:
            refused = asyncio.create_task(
                client.post(SPEECH_PATH, content=speak_body(VoiceId.RYAN))
            )
            (pid,) = await asyncio.to_thread(wait_for_starts, stage.cache, 1)
            release_load(stage.cache, pid, LoadOutcome.FAILED)
            await asyncio.to_thread(wait_for_held_exit, stage.cache, pid)
            serving.sigterm()
            deadline = time.monotonic() + WORKER_WAIT_SECONDS
            while await asyncio.to_thread(listens_on, serving.port):
                assert time.monotonic() < deadline
            waited = not serving.task.done() and is_running(pid)
            release_exit(stage.cache, pid)
            outcome = await serving.task
            assert_unavailable(await refused)
            return waited, outcome

    setup = StageSetup(started=False, reaps_fail=reap_fails)
    with gpu_voice_stage(tmp_path, monkeypatch, setup) as stage:
        hold_exits(stage.cache)
        serving = Serving(stage, monkeypatch)
        with serving.listener:
            waited, outcome = asyncio.run(scenario(serving, stage))

    assert waited is True
    assert outcome == exit_status
    (pid,) = stage.starts()
    assert_reaped(pid)
    assert serving.fatal.is_set() is reap_fails


def test_a_failed_load_whose_worker_cannot_be_reaped_ends_the_service_nonzero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def scenario(serving: Serving, stage: GpuVoiceStage) -> int:
        async with await serving.start() as client:
            asked = asyncio.create_task(
                client.post(SPEECH_PATH, content=speak_body(VoiceId.RYAN))
            )
            (pid,) = await asyncio.to_thread(wait_for_starts, stage.cache, 1)
            release_load(stage.cache, pid, LoadOutcome.FAILED)
            outcome = await serving.task
            await asyncio.gather(asked, return_exceptions=True)
            return outcome

    setup = StageSetup(started=False, reaps_fail=True)
    with gpu_voice_stage(tmp_path, monkeypatch, setup) as stage:
        serving = Serving(stage, monkeypatch)
        with serving.listener:
            outcome = asyncio.run(scenario(serving, stage))

    assert outcome == 1
    assert serving.fatal.is_set() is True


async def _load_and_stream(client: httpx2.AsyncClient) -> httpx2.Response:
    """Load Qwen, then open a stream that holds its gate until the stop ends it."""
    loaded = await client.post(SPEECH_PATH, content=speak_body(VoiceId.RYAN))
    assert loaded.status_code == HTTPStatus.OK
    request = client.build_request(
        "POST", SPEECH_PATH, content=speak_body(VoiceId.RYAN, "wait-cancel")
    )
    streaming = await client.send(request, stream=True)
    assert streaming.status_code == HTTPStatus.OK
    return streaming


def test_sigterm_mid_stream_lets_the_stream_run_to_the_drain_bound_then_aborts_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """WIRE-145, WIRE-147: a stream still running at the drain bound is cut."""

    async def scenario(serving: Serving) -> tuple[bool, int, float]:
        async with await serving.start() as client:
            streaming = await _load_and_stream(client)
            stopped_at = time.monotonic()
            serving.sigterm()
            try:
                await streaming.aread()
            except httpx2.RemoteProtocolError:
                aborted = True
            else:
                aborted = False
            outcome = await serving.task
            return aborted, outcome, time.monotonic() - stopped_at

    with gpu_voice_stage(
        tmp_path, monkeypatch, _SHORT_DRAIN, **_SHORT_DRAIN_WAIT
    ) as stage:
        serving = Serving(stage, monkeypatch)
        with serving.listener:
            aborted, outcome, waited_seconds = asyncio.run(scenario(serving))

    assert aborted is True
    assert waited_seconds >= _DRAIN_BOUND_SECONDS
    assert outcome == 0
    (pid,) = stage.starts()
    assert_reaped(pid)
    assert serving.fatal.is_set() is False


def _kept_alive(port: int) -> http.client.HTTPConnection:
    """A connection that answered GET /health whole and stays open for more."""
    connection = http.client.HTTPConnection(LOOPBACK, port, timeout=WORKER_WAIT_SECONDS)
    connection.request("GET", "/health")
    answer = connection.getresponse()
    answer.read()
    assert answer.getheader("connection") != "close"
    return connection


def _send_qwen_request(connection: http.client.HTTPConnection) -> None:
    connection.request("POST", SPEECH_PATH, speak_body(VoiceId.RYAN), speak_headers())


def _answer(connection: http.client.HTTPConnection) -> tuple[int, dict[str, object]]:
    answer = connection.getresponse()
    return answer.status, json.loads(answer.read())


async def _queued_before_the_stop(
    serving: Serving, stage: GpuVoiceStage, connection: http.client.HTTPConnection
) -> None:
    """The server reads the request before it handles SIGTERM, as a race lets it."""
    queued = gate_acquisition_began(stage.runtime, VoiceId.RYAN)
    _send_qwen_request(connection)
    await asyncio.wait_for(queued.wait(), WORKER_WAIT_SECONDS)
    serving.sigterm()


async def _sent_after_sigterm(
    serving: Serving, _stage: GpuVoiceStage, connection: http.client.HTTPConnection
) -> None:
    serving.sigterm()
    # Sent before the loop runs again, so the server reads it before its stop
    # could close the idle connection.
    _send_qwen_request(connection)


@pytest.mark.parametrize("arrive", [_queued_before_the_stop, _sent_after_sigterm])
def test_a_request_the_stop_meets_on_a_kept_alive_connection_is_unavailable_not_busy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    arrive: Callable[
        [Serving, GpuVoiceStage, http.client.HTTPConnection], Awaitable[None]
    ],
) -> None:
    """WIRE-103: a request the stop meets is unavailable, not busy."""

    async def scenario(
        serving: Serving, stage: GpuVoiceStage
    ) -> tuple[int, dict[str, object], int]:
        async with await serving.start() as client:
            kept_alive = await asyncio.to_thread(_kept_alive, serving.port)
            streaming = await _load_and_stream(client)
            await arrive(serving, stage, kept_alive)
            status, body = await asyncio.to_thread(_answer, kept_alive)
            kept_alive.close()
            await streaming.aclose()
            return status, body, await serving.task

    with gpu_voice_stage(
        tmp_path, monkeypatch, _SHORT_DRAIN, **_SHORT_DRAIN_WAIT
    ) as stage:
        serving = Serving(stage, monkeypatch)
        with serving.listener:
            status, body, outcome = asyncio.run(scenario(serving, stage))

    assert_refusal_body(
        status, body, HTTPStatus.SERVICE_UNAVAILABLE, RefusalCode.UNAVAILABLE, "voice"
    )
    assert outcome == 0


def test_a_request_waiting_at_the_gate_when_the_service_stops_is_refused_not_busy() -> (
    None
):
    """WIRE-103: a request waiting at the gate when the service stops is not busy."""

    async def scenario() -> SpeakCall:
        voice = GatedSpeaking()
        runtime = fake_runtime(voice)
        app = app_over(runtime, busy_wait_seconds=0.2)
        holder = SpeakCall(app)
        await holder.status()
        await asyncio.to_thread(voice.entered_first_step.wait, WORKER_WAIT_SECONDS)
        queued = gate_acquisition_began(runtime)
        waiting = SpeakCall(app)
        await queued.wait()
        runtime.begin_shutdown()
        await waiting.finish()
        voice.release_all()
        await asyncio.gather(holder.task, return_exceptions=True)
        return waiting

    waiting = asyncio.run(scenario())

    assert_call_refusal(
        waiting, HTTPStatus.SERVICE_UNAVAILABLE, RefusalCode.UNAVAILABLE, "voice"
    )

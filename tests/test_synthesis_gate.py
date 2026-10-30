"""The one synthesis at a time: first come first served, no thread per waiter.

The gate's own behaviour comes first; the rest drives it through the wire, where
a second request waits a configured few seconds and then answers 429.
"""

import asyncio
import os
import threading
import time
from typing import TYPE_CHECKING

import anyio.to_thread
import pytest
from fastapi.testclient import TestClient
from starlette.status import (
    HTTP_200_OK,
    HTTP_422_UNPROCESSABLE_CONTENT,
    HTTP_429_TOO_MANY_REQUESTS,
    HTTP_503_SERVICE_UNAVAILABLE,
)

from speech_runtime_client.wire import RETRY_AFTER_HEADER, RefusalCode
from speech_runtime_service.pcm import pcm_from_wav
from speech_runtime_service.service import Runtime, RuntimeDependencies
from speech_runtime_service.synthesis_gate import Lease, SynthesisGate
from speech_runtime_service.voices import VoiceId, VoiceLoadOutcome
from tests.support.fakes_support import (
    SPEAK_SAMPLE_RATE,
    FakeHearing,
    FakeSpeaking,
    GatedSpeaking,
)
from tests.support.speak_support import (
    ASGI_TIMEOUT_SECONDS,
    CommittingSelectionStore,
    SpeakCall,
    app_over,
    assert_call_refusal,
    assert_refusal,
    assert_unavailable,
    fake_runtime,
    gate_acquisition_began,
    post_speech,
    speak_body,
    speak_once,
    status_of,
    switch_completes,
    switchable_runtime,
    wav_body,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from fastapi import FastAPI
    from httpx2 import Response

SHORT_WAIT_SECONDS = 0.2
SLACK_SECONDS = 0.5
CHATTERBOX = VoiceId.CHATTERBOX.value
ALIEN_SAMPLE_RATE = 24_000

WAIT_SECONDS = 5.0
SHORT_SECONDS = 0.05


def _in(seconds: float) -> float:
    return time.monotonic() + seconds


async def _settle() -> None:
    for _ in range(3):
        await asyncio.sleep(0)


def test_a_free_gate_is_taken_even_past_the_deadline() -> None:
    gate = SynthesisGate()

    lease = asyncio.run(gate.acquire(_in(-1)))

    assert lease is not None


def test_try_acquire_refuses_while_the_gate_is_held_and_serves_after_release() -> None:
    gate = SynthesisGate()
    held = gate.try_acquire()
    assert held is not None

    assert gate.try_acquire() is None
    held.close()

    assert gate.try_acquire() is not None


def test_a_waiter_that_reaches_its_deadline_gets_none() -> None:
    async def scenario() -> tuple[object, bool]:
        gate = SynthesisGate()
        held = await gate.acquire(_in(0))
        assert held is not None
        timed_out = await gate.acquire(_in(SHORT_SECONDS))
        held.close()
        return timed_out, gate.try_acquire() is not None

    timed_out, free_again = asyncio.run(scenario())

    assert timed_out is None
    assert free_again


def test_waiters_are_served_in_the_order_they_came() -> None:
    async def scenario() -> list[str]:
        gate = SynthesisGate()
        first = await gate.acquire(_in(WAIT_SECONDS))
        assert first is not None
        served: list[str] = []

        async def wait(name: str) -> None:
            lease = await gate.acquire(_in(WAIT_SECONDS))
            assert lease is not None
            served.append(name)
            lease.close()

        second = asyncio.create_task(wait("second"))
        await _settle()
        third = asyncio.create_task(wait("third"))
        await _settle()
        first.close()
        await asyncio.wait_for(asyncio.gather(second, third), WAIT_SECONDS)
        return served

    served = asyncio.run(scenario())

    assert served == ["second", "third"]


def test_a_release_from_another_thread_serves_the_waiter() -> None:
    async def scenario() -> bool:
        gate = SynthesisGate()
        held = await gate.acquire(_in(0))
        assert held is not None
        waiting = asyncio.create_task(gate.acquire(_in(WAIT_SECONDS)))
        await _settle()
        releasing = threading.Thread(target=held.close)
        releasing.start()
        lease = await waiting
        releasing.join()
        return lease is not None

    assert asyncio.run(scenario())


class _InterleavingGuard:
    """A guard that runs one action, once, right after it is next left."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.after_exit: Callable[[], None] | None = None

    def __enter__(self) -> None:
        self._lock.acquire()

    def __exit__(self, *_exc: object) -> None:
        self._lock.release()
        action, self.after_exit = self.after_exit, None
        if action is not None:
            action()


class _InterleavedGate(SynthesisGate):
    def __init__(self) -> None:
        super().__init__()
        self.interleaving = _InterleavingGuard()
        self._guard = self.interleaving


def test_a_release_right_after_the_gate_is_found_held_still_serves_the_waiter() -> None:
    def release_from_another_thread(held: Lease) -> None:
        releasing = threading.Thread(target=held.close)
        releasing.start()
        releasing.join()

    async def scenario() -> Lease | None:
        gate = _InterleavedGate()
        held = gate.try_acquire()
        assert held is not None
        gate.interleaving.after_exit = lambda: release_from_another_thread(held)
        return await gate.acquire(_in(WAIT_SECONDS))

    assert asyncio.run(scenario()) is not None


def test_a_cancelled_waiter_leaves_the_gate_to_the_next() -> None:
    async def scenario() -> bool:
        gate = SynthesisGate()
        held = await gate.acquire(_in(0))
        assert held is not None
        cancelled = asyncio.create_task(gate.acquire(_in(WAIT_SECONDS)))
        await _settle()
        next_in_line = asyncio.create_task(gate.acquire(_in(WAIT_SECONDS)))
        await _settle()
        cancelled.cancel()
        await _settle()
        held.close()
        lease = await asyncio.wait_for(next_in_line, WAIT_SECONDS)
        return lease is not None

    assert asyncio.run(scenario())


def test_a_waiter_cancelled_after_the_hand_off_is_scheduled_frees_the_gate() -> None:
    async def scenario() -> bool:
        gate = SynthesisGate()
        held = await gate.acquire(_in(0))
        assert held is not None
        waiting = asyncio.create_task(gate.acquire(_in(WAIT_SECONDS)))
        await _settle()
        held.close()  # the hand-off to `waiting` is now scheduled on the loop
        waiting.cancel()
        await asyncio.gather(waiting, return_exceptions=True)
        await _settle()
        return gate.try_acquire() is not None

    assert asyncio.run(scenario())


def test_a_lease_gives_the_gate_back_once() -> None:
    gate = SynthesisGate()
    lease = gate.try_acquire()
    assert lease is not None
    lease.close()
    other = gate.try_acquire()
    assert other is not None

    lease.close()

    assert gate.try_acquire() is None


def _busy(response: Response, wait_header: str) -> None:
    assert_refusal(response, HTTP_429_TOO_MANY_REQUESTS, RefusalCode.BUSY, None)
    assert response.headers[RETRY_AFTER_HEADER] == wait_header


@pytest.mark.parametrize(
    ("wait", "retry_after"), [(0, "1"), (SHORT_WAIT_SECONDS, "1"), (2.5, "3")]
)
def test_a_request_that_finds_the_gate_held_waits_then_is_told_to_retry_after(
    wait: float, retry_after: str
) -> None:
    runtime = fake_runtime()
    client = TestClient(app_over(runtime, busy_wait_seconds=wait))
    held = runtime.synthesis_gate.try_acquire()
    assert held is not None
    began = time.monotonic()

    response = post_speech(client, speak_body())

    assert time.monotonic() - began < wait + SLACK_SECONDS
    _busy(response, retry_after)


def _async_runtime(
    tmp_path: Path,
    voice: GatedSpeaking | FakeSpeaking,
    wait: float = SHORT_WAIT_SECONDS,
) -> tuple[Runtime, FastAPI]:
    runtime = switchable_runtime(tmp_path, voice)
    return runtime, app_over(runtime, busy_wait_seconds=wait)


def test_a_second_request_is_told_busy_within_the_wait_while_the_first_streams(
    tmp_path: Path,
) -> None:
    async def scenario() -> tuple[SpeakCall, SpeakCall, bytes]:
        voice = GatedSpeaking()
        _, app = _async_runtime(tmp_path, voice)
        first = SpeakCall(app, speak_body(CHATTERBOX))
        await asyncio.to_thread(voice.entered_first_step.wait, ASGI_TIMEOUT_SECONDS)
        second = SpeakCall(app, speak_body(CHATTERBOX))
        await second.status()
        voice.release_all()
        return first, second, await first.finish()

    first, second, first_body = asyncio.run(scenario())

    assert second.status_code == HTTP_429_TOO_MANY_REQUESTS
    assert second.headers[RETRY_AFTER_HEADER.lower()] == "1"
    assert second.started_after < SHORT_WAIT_SECONDS + SLACK_SECONDS
    assert first.status_code == HTTP_200_OK
    assert pcm_from_wav(first_body)[0] == SPEAK_SAMPLE_RATE


def test_after_a_busy_refusal_the_next_request_is_served_and_a_switch_completes(
    tmp_path: Path,
) -> None:
    async def scenario() -> tuple[Runtime, int, int]:
        voice = GatedSpeaking()
        runtime, app = _async_runtime(tmp_path, voice)
        first = SpeakCall(app, speak_body(CHATTERBOX))
        await asyncio.to_thread(voice.entered_first_step.wait, ASGI_TIMEOUT_SECONDS)
        refused = await _status_of(SpeakCall(app, speak_body(CHATTERBOX)))
        voice.release_all()
        await first.finish()
        return (
            runtime,
            refused,
            status_of(await speak_once(app, CHATTERBOX)),
        )

    runtime, refused, served = asyncio.run(scenario())

    assert refused == HTTP_429_TOO_MANY_REQUESTS
    assert served == HTTP_200_OK
    assert switch_completes(runtime)


async def _status_of(call: SpeakCall) -> int:
    status = await call.status()
    await call.finish()
    return status


def test_two_requests_within_the_wait_are_served_one_synthesis_at_a_time(
    tmp_path: Path,
) -> None:
    async def scenario() -> tuple[GatedSpeaking, list[int]]:
        voice = GatedSpeaking()
        _, app = _async_runtime(tmp_path, voice, ASGI_TIMEOUT_SECONDS)
        first = SpeakCall(app, speak_body(CHATTERBOX))
        await asyncio.to_thread(voice.entered_first_step.wait, ASGI_TIMEOUT_SECONDS)
        second = SpeakCall(app, speak_body(CHATTERBOX))
        voice.release_all()
        await asyncio.gather(first.finish(), second.finish())
        return voice, [first.status_code, second.status_code]

    voice, statuses = asyncio.run(scenario())

    assert statuses == [HTTP_200_OK, HTTP_200_OK]
    assert voice.most_concurrent_syntheses == 1


def test_a_voice_switch_waits_for_a_waiting_request_which_speaks_with_its_own_voice(
    tmp_path: Path,
) -> None:
    async def scenario() -> tuple[SpeakCall, list[bool], FastAPI]:
        speaking = FakeSpeaking()
        speaking.sample_rate = ALIEN_SAMPLE_RATE
        store = CommittingSelectionStore(tmp_path / "state", owner_uid=os.geteuid())
        runtime = switchable_runtime(tmp_path, speaking, store)
        app = app_over(runtime, busy_wait_seconds=ASGI_TIMEOUT_SECONDS)
        held = runtime.synthesis_gate.try_acquire()
        assert held is not None
        waiting_began = gate_acquisition_began(runtime)
        waiting = SpeakCall(app, speak_body(CHATTERBOX))
        await waiting_began.wait()
        outcome: list[bool] = []
        switching = threading.Thread(
            target=lambda: outcome.append(
                runtime.load_voice(VoiceId.PIPER) is VoiceLoadOutcome.ACTIVATED
            )
        )
        switching.start()
        assert await asyncio.to_thread(store.piper_committed.wait, ASGI_TIMEOUT_SECONDS)
        assert switching.is_alive()
        assert outcome == []
        held.close()
        await waiting.finish()
        await asyncio.to_thread(switching.join, ASGI_TIMEOUT_SECONDS)
        return waiting, outcome, app

    waiting, outcome, app = asyncio.run(scenario())

    assert waiting.status_code == HTTP_200_OK
    assert pcm_from_wav(wav_body(waiting.messages))[0] == ALIEN_SAMPLE_RATE
    assert outcome == [True]
    after = TestClient(app)
    assert_unavailable(post_speech(after, speak_body(CHATTERBOX)))
    assert (
        post_speech(after, speak_body(VoiceId.PIPER.value)).status_code == HTTP_200_OK
    )


def _leave_gate_free(_runtime: Runtime) -> Lease | None:
    return None


def _hold_gate(runtime: Runtime) -> Lease | None:
    return runtime.synthesis_gate.try_acquire()


@pytest.mark.parametrize(
    ("body", "hold_gate", "status"),
    [
        pytest.param(
            b"{}", _leave_gate_free, HTTP_422_UNPROCESSABLE_CONTENT, id="invalid-body"
        ),
        pytest.param(
            speak_body(VoiceId.PIPER.value),
            _leave_gate_free,
            HTTP_503_SERVICE_UNAVAILABLE,
            id="voice-is-not-the-active-one",
        ),
        pytest.param(
            speak_body(CHATTERBOX),
            _hold_gate,
            HTTP_429_TOO_MANY_REQUESTS,
            id="gate-is-held",
        ),
    ],
)
def test_a_refusal_leaves_no_admission_behind_so_a_voice_switch_completes(
    tmp_path: Path,
    body: bytes,
    hold_gate: Callable[[Runtime], Lease | None],
    status: int,
) -> None:
    runtime = switchable_runtime(tmp_path, FakeSpeaking())
    _held = hold_gate(runtime)
    client = TestClient(app_over(runtime, busy_wait_seconds=0))

    response = post_speech(client, body)

    assert response.status_code == status
    assert switch_completes(runtime)


def test_a_request_cancelled_while_it_waits_for_the_gate_leaves_nothing_behind(
    tmp_path: Path,
) -> None:
    async def scenario() -> tuple[Runtime, int]:
        runtime = switchable_runtime(tmp_path, FakeSpeaking())
        app = app_over(runtime, busy_wait_seconds=ASGI_TIMEOUT_SECONDS)
        held = runtime.synthesis_gate.try_acquire()
        assert held is not None
        waiting_began = gate_acquisition_began(runtime)
        waiting = SpeakCall(app, speak_body(CHATTERBOX))
        await waiting_began.wait()
        waiting.task.cancel()
        await asyncio.gather(waiting.task, return_exceptions=True)
        held.close()
        return runtime, status_of(await speak_once(app, CHATTERBOX))

    runtime, served = asyncio.run(scenario())

    assert served == HTTP_200_OK
    assert switch_completes(runtime)


def test_a_stream_holding_every_worker_cannot_push_the_busy_answer_past_the_wait(
    tmp_path: Path,
) -> None:
    async def scenario() -> tuple[Runtime, SpeakCall, SpeakCall, int]:
        anyio.to_thread.current_default_thread_limiter().total_tokens = 1
        voice = GatedSpeaking()
        runtime, app = _async_runtime(tmp_path, voice)
        first = SpeakCall(app, speak_body(CHATTERBOX))
        await first.status()
        await asyncio.to_thread(voice.entered_first_step.wait, ASGI_TIMEOUT_SECONDS)
        second = SpeakCall(app, speak_body(CHATTERBOX))
        await second.status()
        voice.release_all()
        await asyncio.gather(first.finish(), second.finish())
        return (
            runtime,
            first,
            second,
            status_of(await speak_once(app, CHATTERBOX)),
        )

    runtime, first, second, later = asyncio.run(scenario())

    assert_call_refusal(second, HTTP_429_TOO_MANY_REQUESTS, RefusalCode.BUSY, None)
    assert second.started_after < SHORT_WAIT_SECONDS + SLACK_SECONDS
    assert second.headers[RETRY_AFTER_HEADER.lower()] == "1"
    assert first.status_code == HTTP_200_OK
    assert later == HTTP_200_OK
    assert switch_completes(runtime)


def test_a_free_gate_answers_within_the_wait_though_every_worker_is_occupied(
    tmp_path: Path,
) -> None:
    async def scenario() -> tuple[Runtime, SpeakCall]:
        anyio.to_thread.current_default_thread_limiter().total_tokens = 1
        runtime, app = _async_runtime(tmp_path, FakeSpeaking())
        occupied = threading.Event()
        occupying = threading.Event()

        def occupy() -> None:
            occupying.set()
            occupied.wait(ASGI_TIMEOUT_SECONDS)

        job = asyncio.create_task(anyio.to_thread.run_sync(occupy))
        await asyncio.to_thread(occupying.wait, ASGI_TIMEOUT_SECONDS)
        call = SpeakCall(app, speak_body(CHATTERBOX))
        await call.status()
        occupied.set()
        await job
        await call.finish()
        return runtime, call

    runtime, call = asyncio.run(scenario())

    assert call.status_code == HTTP_200_OK
    assert call.started_after < SHORT_WAIT_SECONDS + SLACK_SECONDS
    assert switch_completes(runtime)


def test_a_failed_voice_whose_close_blocks_is_still_answered_within_the_wait() -> None:
    release_close = threading.Event()
    closing = threading.Event()

    class BlockingClose(FakeSpeaking):
        def close(self) -> None:
            closing.set()
            release_close.wait(ASGI_TIMEOUT_SECONDS)

    speaking = BlockingClose()
    app = app_over(fake_runtime(speaking), busy_wait_seconds=SHORT_WAIT_SECONDS)
    speaking.ready = False

    async def scenario() -> tuple[SpeakCall, SpeakCall, bool]:
        first = SpeakCall(app)
        await first.status()
        await asyncio.to_thread(closing.wait, ASGI_TIMEOUT_SECONDS)
        second = SpeakCall(app)
        await second.status()
        sent_last = _last_message_is_final(first)
        release_close.set()
        await asyncio.gather(first.finish(), second.finish())
        return first, second, sent_last

    first, second, sent_last = asyncio.run(scenario())

    assert sent_last
    for call in (first, second):
        assert_call_refusal(
            call, HTTP_503_SERVICE_UNAVAILABLE, RefusalCode.UNAVAILABLE, None
        )
    assert first.started_after < SHORT_WAIT_SECONDS + SLACK_SECONDS
    assert second.started_after < SHORT_WAIT_SECONDS + SLACK_SECONDS


def _last_message_is_final(call: SpeakCall) -> bool:
    last = call.messages[-1]
    return last["type"] == "http.response.body" and not last.get("more_body")


def test_a_download_start_held_in_its_artifact_check_does_not_delay_the_answer() -> (
    None
):
    entered = threading.Event()
    release = threading.Event()

    def artifacts_are_complete() -> bool:
        entered.set()
        release.wait(ASGI_TIMEOUT_SECONDS)
        return True

    runtime = Runtime(
        FakeSpeaking(),
        FakeHearing(),
        memory_probe=lambda: 0,
        dependencies=RuntimeDependencies(
            artifact_checker=lambda _voice: artifacts_are_complete(),
            qwen_download=lambda: None,
            qwen_worker_is_usable=lambda: True,
        ),
    )
    starting = threading.Thread(target=runtime.start_qwen_download)
    starting.start()
    assert entered.wait(ASGI_TIMEOUT_SECONDS)
    app = app_over(runtime, busy_wait_seconds=SHORT_WAIT_SECONDS)

    async def scenario() -> SpeakCall:
        call = SpeakCall(app)
        await call.status()
        await call.finish()
        return call

    call = asyncio.run(scenario())
    release.set()
    starting.join(timeout=ASGI_TIMEOUT_SECONDS)

    assert call.status_code == HTTP_200_OK
    assert call.started_after < SHORT_WAIT_SECONDS + SLACK_SECONDS
    assert not starting.is_alive()

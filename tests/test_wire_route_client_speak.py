"""The client's speaking call against the real service over TCP."""

from __future__ import annotations

import asyncio
import math
import time
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

import httpx2
import pytest

from speech_client.speak import (
    Busy,
    Cut,
    Finished,
    OverCap,
    PastDeadline,
    Refused,
    Spoken,
    speak,
    speak_stream,
)
from speech_client.voice_preference import (
    SpokenWithPreference,
    speak_with_preference,
)
from speech_client.wire import (
    REFUSAL_STATUS,
    SPEECH_PATH,
    VOICES,
    Language,
    RefusalCode,
    ResponseFormat,
    VoiceId,
    refusal,
    speech_request_body,
)
from speech_service.config import Settings
from speech_service.pcm import WAV_HEADER_BYTES
from speech_service.service import Runtime
from tests.support.fakes_support import (
    GATE_TIMEOUT_SECONDS,
    SPEAK_SAMPLE_RATE,
    FakeSpeaking,
    GatedSpeaking,
)
from tests.support.identity_support import valid_headers, valid_pair
from tests.support.speak_support import FailingSpeaking
from tests.support.tcp_server_support import TcpServer, serving_over_tcp

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator

    from proof_identity import TokenPair
    from speech_client.speak import SpeakOutcome, SpeakStreamItem
    from speech_client.voice_preference import PreferenceOutcome

DEADLINE_SECONDS = 0.5
MARGIN_SECONDS = 3.0
CAP_BYTES = 1024 * 1024


def _speak(  # noqa: PLR0913 - one keyword per call-site option
    base_url: str,
    *,
    voice: VoiceId = VoiceId.THORSTEN,
    language: Language | None = None,
    response_format: ResponseFormat = ResponseFormat.WAV,
    deadline_seconds: float = GATE_TIMEOUT_SECONDS,
    cap_bytes: int = CAP_BYTES,
    pair: TokenPair | None = None,
) -> SpeakOutcome:
    pair = pair or valid_pair()
    return speak(
        base_url,
        "Hallo",
        voice,
        language=language,
        response_format=response_format,
        workload_token=pair.workload,
        person_token=pair.person,
        deadline_seconds=deadline_seconds,
        cap_bytes=cap_bytes,
    )


def _speak_stream(
    base_url: str,
    *,
    voice: VoiceId = VoiceId.THORSTEN,
    language: Language | None = None,
    deadline_seconds: float = GATE_TIMEOUT_SECONDS,
    cap_bytes: int = CAP_BYTES,
) -> Iterator[SpeakStreamItem]:
    pair = valid_pair()
    return speak_stream(
        base_url,
        "Hallo",
        voice,
        language=language,
        workload_token=pair.workload,
        person_token=pair.person,
        deadline_seconds=deadline_seconds,
        cap_bytes=cap_bytes,
    )


def _runtime(voice: FakeSpeaking | GatedSpeaking) -> Runtime:
    return Runtime({VoiceId.THORSTEN: voice}, memory_probe=lambda: 0)


@asynccontextmanager
async def _serving(
    voice: FakeSpeaking | GatedSpeaking,
) -> AsyncIterator[TcpServer]:
    """Serve `voice` over TCP; open every gate on exit so the server can stop."""
    async with serving_over_tcp(_runtime(voice)) as server:
        try:
            yield server
        finally:
            if isinstance(voice, GatedSpeaking):
                voice.release_all()


def _over_the_service(
    voice: FakeSpeaking | GatedSpeaking, **options: float | ResponseFormat
) -> SpeakOutcome:
    async def scenario() -> SpeakOutcome:
        async with _serving(voice) as server:
            return await asyncio.to_thread(_speak, server.base_url, **options)

    return asyncio.run(scenario())


def test_the_call_returns_the_wav_the_service_sent() -> None:
    """WIRE-88, WIRE-90: the whole chunked body comes back as the WAV, byte for byte."""

    async def scenario() -> tuple[SpeakOutcome, httpx2.Response]:
        async with _serving(FakeSpeaking()) as server:
            outcome = await asyncio.to_thread(_speak, server.base_url)
            async with httpx2.AsyncClient(
                base_url=server.base_url, headers=valid_headers()
            ) as client:
                return outcome, await client.post(
                    SPEECH_PATH,
                    content=speech_request_body(
                        "Hallo", VoiceId.THORSTEN, ResponseFormat.WAV
                    ),
                )

    outcome, sent = asyncio.run(scenario())

    assert outcome == Spoken(sent.content)


def test_the_call_returns_the_ogg_opus_the_service_sent() -> None:
    """WIRE-133: an opus order comes back as the whole chunked Ogg body."""
    outcome = _over_the_service(FakeSpeaking(), response_format=ResponseFormat.OPUS)

    assert isinstance(outcome, Spoken)
    assert ResponseFormat.OPUS.begins_like_container(outcome.audio)


def test_a_wav_of_exactly_the_cap_is_spoken() -> None:
    whole = _over_the_service(FakeSpeaking())
    assert isinstance(whole, Spoken)

    outcome = _over_the_service(FakeSpeaking(), cap_bytes=len(whole.audio))

    assert outcome == whole


def test_a_cap_below_the_wav_is_over_the_cap() -> None:
    outcome = _over_the_service(FakeSpeaking(), cap_bytes=WAV_HEADER_BYTES)

    assert outcome == OverCap()


def test_a_voice_that_is_not_offered_is_refused_with_its_status_code_and_param() -> (
    None
):
    async def scenario() -> SpeakOutcome:
        async with _serving(FakeSpeaking()) as server:
            return await asyncio.to_thread(
                _speak,
                server.base_url,
                voice=VoiceId.RYAN,
                language=VOICES[VoiceId.RYAN].languages[0],
            )

    outcome = asyncio.run(scenario())

    assert isinstance(outcome, Refused)
    assert outcome.refusal == refusal(RefusalCode.UNAVAILABLE, "voice")
    assert outcome.status == REFUSAL_STATUS[RefusalCode.UNAVAILABLE]


@pytest.mark.parametrize("chunks_before_failing", [0, 1])
def test_a_body_the_service_aborted_is_cut(chunks_before_failing: int) -> None:
    """WIRE-90: a body without its end is never a WAV."""
    voice = FailingSpeaking(chunks_before_failing=chunks_before_failing)

    assert _over_the_service(voice) == Cut()


def test_a_voice_that_stalls_leaves_the_call_past_the_deadline() -> None:
    voice = GatedSpeaking()

    began = time.monotonic()
    outcome = _over_the_service(voice, deadline_seconds=DEADLINE_SECONDS)

    assert outcome == PastDeadline()
    assert time.monotonic() - began < DEADLINE_SECONDS + MARGIN_SECONDS


def test_a_request_beyond_the_waiting_places_is_told_busy_with_the_wait() -> None:
    """WIRE-23: one held call, the default waiting places full, then one more."""
    places = Settings().max_waiting_requests

    async def scenario() -> tuple[SpeakOutcome, list[SpeakOutcome]]:
        voice = GatedSpeaking()
        async with _serving(voice) as server:
            held = asyncio.create_task(asyncio.to_thread(_speak, server.base_url))
            assert await asyncio.to_thread(
                voice.entered_first_step.wait, GATE_TIMEOUT_SECONDS
            )
            more = [
                asyncio.create_task(asyncio.to_thread(_speak, server.base_url))
                for _ in range(places + 1)
            ]
            done, waiting = await asyncio.wait(
                more, timeout=GATE_TIMEOUT_SECONDS, return_when=asyncio.FIRST_COMPLETED
            )
            voice.release_all()
            return done.pop().result(), await asyncio.gather(held, *waiting)

    first, later = asyncio.run(scenario())

    assert isinstance(first, Busy)
    assert first.wait.retry_after_seconds == math.ceil(Settings().busy_wait_seconds)
    assert all(isinstance(outcome, Spoken) for outcome in later)


def _speak_with_preference(
    base_url: str, preference: list[VoiceId], instructions: str
) -> PreferenceOutcome:
    pair = valid_pair()
    return speak_with_preference(
        base_url,
        "Hallo",
        preference,
        language=Language.DE,
        instructions=instructions,
        response_format=ResponseFormat.WAV,
        workload_token=pair.workload,
        person_token=pair.person,
        deadline_seconds=GATE_TIMEOUT_SECONDS,
        cap_bytes=CAP_BYTES,
    )


@pytest.mark.parametrize(
    ("offered", "spoke", "instructions_left_out"),
    [
        ([VoiceId.RYAN, VoiceId.THORSTEN], VoiceId.RYAN, False),
        ([VoiceId.THORSTEN], VoiceId.THORSTEN, True),
    ],
)
def test_a_preference_speaks_with_the_first_voice_the_service_offers(
    offered: list[VoiceId], spoke: VoiceId, *, instructions_left_out: bool
) -> None:
    """WIRE-125, WIRE-142: the service offers fewer voices; the tone is not refused."""
    runtime = Runtime(
        {voice: FakeSpeaking() for voice in offered}, memory_probe=lambda: 0
    )

    async def scenario() -> PreferenceOutcome:
        async with serving_over_tcp(runtime) as server:
            return await asyncio.to_thread(
                _speak_with_preference,
                server.base_url,
                [VoiceId.RYAN, VoiceId.THORSTEN],
                "Sprich ruhig",
            )

    outcome = asyncio.run(scenario())

    assert isinstance(outcome, SpokenWithPreference)
    assert outcome.voice == spoke
    assert outcome.instructions_left_out is instructions_left_out


def _streamed(voice: FakeSpeaking | GatedSpeaking) -> list[SpeakStreamItem]:
    async def scenario() -> list[SpeakStreamItem]:
        async with _serving(voice) as server:
            return await asyncio.to_thread(lambda: list(_speak_stream(server.base_url)))

    return asyncio.run(scenario())


def test_the_stream_carries_the_wav_and_ends_with_the_exact_seconds() -> None:
    """WIRE-88: the chunks join to the blocking call's WAV; the end names the length."""
    streamed = _streamed(FakeSpeaking())
    spoken = _over_the_service(FakeSpeaking())
    assert isinstance(spoken, Spoken)

    *chunks, end = streamed
    assert all(isinstance(chunk, bytes) for chunk in chunks)
    assert b"".join(chunks) == spoken.audio
    assert isinstance(end, Finished)
    pcm = spoken.audio.split(b"data", 1)[1][4:]  # after the chunk id and size
    assert end.seconds == len(pcm) // 2 / SPEAK_SAMPLE_RATE


def test_the_stream_hands_over_the_first_audio_while_the_voice_still_renders() -> None:
    """WIRE-90: a player may play the chunks as they arrive."""
    voice = GatedSpeaking(hold_last_chunk=True)

    async def scenario() -> tuple[bool, list[SpeakStreamItem]]:
        async with _serving(voice) as server:
            voice.release_first_step.set()
            stream = _speak_stream(server.base_url)
            first = await asyncio.to_thread(next, stream)
            still_rendering = not voice.release_last_chunk.is_set()
            voice.release_all()
            rest = await asyncio.to_thread(list, stream)
            return still_rendering and isinstance(first, bytes), rest

    first_came_early, rest = asyncio.run(scenario())

    assert first_came_early
    assert isinstance(rest[-1], Finished)


@pytest.mark.parametrize("chunks_before_failing", [0, 1])
def test_a_stream_the_service_aborted_ends_cut_and_never_finished(
    chunks_before_failing: int,
) -> None:
    """WIRE-90: a body without its end is a failure, not a short sound."""
    items = _streamed(FailingSpeaking(chunks_before_failing=chunks_before_failing))

    assert items[-1] == Cut()
    assert not any(isinstance(item, Finished) for item in items)


def test_a_voice_that_is_not_offered_is_refused_before_any_audio() -> None:
    async def scenario() -> list[SpeakStreamItem]:
        async with _serving(FakeSpeaking()) as server:
            return await asyncio.to_thread(
                lambda: list(
                    _speak_stream(
                        server.base_url,
                        voice=VoiceId.RYAN,
                        language=VOICES[VoiceId.RYAN].languages[0],
                    )
                )
            )

    items = asyncio.run(scenario())

    assert [item.refusal for item in items if isinstance(item, Refused)] == [
        refusal(RefusalCode.UNAVAILABLE, "voice")
    ]
    assert len(items) == 1


def test_a_stream_past_the_cap_ends_over_the_cap() -> None:
    async def scenario() -> list[SpeakStreamItem]:
        async with _serving(FakeSpeaking()) as server:
            return await asyncio.to_thread(
                lambda: list(_speak_stream(server.base_url, cap_bytes=WAV_HEADER_BYTES))
            )

    assert asyncio.run(scenario())[-1] == OverCap()

"""The speaking call with an ordered voice preference against a scripted service."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from pydantic import ValidationError

from packages.client.tests.support.scripted_service_support import (
    WAV,
    scripted_service,
)
from speech_client import voice_preference
from speech_client.outcomes import Malformed, OverCap, PastDeadline, Unreachable
from speech_client.speak import Refused
from speech_client.voice_preference import SpokenWithPreference, speak_with_preference
from speech_client.wire import (
    HEALTH_PATH,
    Language,
    RefusalCode,
    ResponseFormat,
    VoiceId,
    refusal,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from packages.client.tests.support.listener_support import RawListener
    from speech_client.voice_preference import PreferenceOutcome

DEADLINE_SECONDS = 30.0
CAP_BYTES = 1024 * 1024
TONE = "calm"
PAST = DEADLINE_SECONDS + 1
UNAVAILABLE = refusal(RefusalCode.UNAVAILABLE, "voice")


def _speak(
    base_url: str,
    preference: Sequence[VoiceId],
    *,
    language: Language | None = Language.DE,
    instructions: str | None = None,
    deadline_seconds: float = DEADLINE_SECONDS,
) -> PreferenceOutcome:
    return speak_with_preference(
        base_url,
        "Hallo",
        preference,
        language=language,
        instructions=instructions,
        response_format=ResponseFormat.WAV,
        workload_token="workload-token",  # noqa: S106 - a stand-in the script never checks
        person_token="person-token",  # noqa: S106 - a stand-in the script never checks
        deadline_seconds=deadline_seconds,
        cap_bytes=CAP_BYTES,
    )


def test_the_first_voice_of_the_preference_that_is_offered_ready_speaks() -> None:
    offered = {VoiceId.KERSTIN: True, VoiceId.THORSTEN: True, VoiceId.RYAN: True}
    with scripted_service(offered) as (base_url, service):
        outcome = _speak(base_url, [VoiceId.RYAN, VoiceId.THORSTEN])

    assert outcome == SpokenWithPreference(
        WAV, VoiceId.RYAN, instructions_left_out=False
    )
    assert [asked["voice"] for asked in service.asked] == [VoiceId.RYAN]


@pytest.mark.parametrize(
    ("preference", "offered", "language", "spoke"),
    [
        pytest.param(
            [VoiceId.RYAN, VoiceId.THORSTEN],
            {VoiceId.THORSTEN: True},
            Language.DE,
            VoiceId.THORSTEN,
            id="not-offered",
        ),
        pytest.param(
            [VoiceId.RYAN, VoiceId.THORSTEN],
            {VoiceId.RYAN: False, VoiceId.THORSTEN: True},
            Language.DE,
            VoiceId.THORSTEN,
            id="not-ready",
        ),
        pytest.param(
            [VoiceId.THORSTEN, VoiceId.NORMAN],
            {VoiceId.THORSTEN: True, VoiceId.NORMAN: True},
            Language.EN,
            VoiceId.NORMAN,
            id="not-in-the-language",
        ),
    ],
)
def test_a_voice_not_offered_ready_in_the_language_is_skipped(
    preference: list[VoiceId],
    offered: dict[VoiceId, bool],
    language: Language,
    spoke: VoiceId,
) -> None:
    with scripted_service(offered) as (base_url, _):
        outcome = _speak(base_url, preference, language=language)

    assert isinstance(outcome, SpokenWithPreference)
    assert outcome.voice == spoke


@pytest.mark.parametrize(
    ("voice", "instructions_left_out"),
    [(VoiceId.RYAN, False), (VoiceId.THORSTEN, True)],
)
def test_instructions_go_only_to_a_voice_that_takes_them(
    voice: VoiceId, *, instructions_left_out: bool
) -> None:
    with scripted_service({voice: True}) as (base_url, service):
        outcome = _speak(base_url, [voice], language=Language.DE, instructions=TONE)

    assert isinstance(outcome, SpokenWithPreference)
    assert outcome.instructions_left_out is instructions_left_out
    assert ("instructions" in service.asked[0]) is not instructions_left_out


def test_instructions_never_given_are_not_reported_left_out() -> None:
    with scripted_service({VoiceId.THORSTEN: True}) as (base_url, _):
        outcome = _speak(base_url, [VoiceId.THORSTEN])

    assert isinstance(outcome, SpokenWithPreference)
    assert not outcome.instructions_left_out


def test_a_voice_that_turns_unavailable_mid_call_hands_over_to_the_next_offered() -> (
    None
):
    """The switch also takes Thorsten away: only a fresh health read shows it."""
    offered = {VoiceId.RYAN: True, VoiceId.THORSTEN: True, VoiceId.KERSTIN: True}
    preference = [VoiceId.RYAN, VoiceId.THORSTEN, VoiceId.KERSTIN]

    def switch(voice: VoiceId) -> None:
        if voice is VoiceId.RYAN:
            del offered[VoiceId.RYAN], offered[VoiceId.THORSTEN]

    with scripted_service(offered) as (base_url, service):
        service.unavailable_on_speak.add(VoiceId.RYAN)
        service.on_speak = switch
        outcome = _speak(base_url, preference, instructions=TONE)

    assert outcome == SpokenWithPreference(
        WAV, VoiceId.KERSTIN, instructions_left_out=True
    )
    assert [asked["voice"] for asked in service.asked] == [
        VoiceId.RYAN,
        VoiceId.KERSTIN,
    ]


def test_a_voice_that_turned_unavailable_is_not_asked_again() -> None:
    offered = {VoiceId.RYAN: True, VoiceId.THORSTEN: True}
    with scripted_service(offered) as (base_url, service):
        service.unavailable_on_speak.update(offered)
        outcome = _speak(base_url, [VoiceId.RYAN, VoiceId.THORSTEN])

    assert outcome == Refused(UNAVAILABLE)
    assert [asked["voice"] for asked in service.asked] == [
        VoiceId.RYAN,
        VoiceId.THORSTEN,
    ]


def test_no_voice_of_the_preference_offered_fails_unavailable_without_a_speak() -> None:
    with scripted_service({VoiceId.KERSTIN: True}) as (base_url, service):
        outcome = _speak(base_url, [VoiceId.RYAN, VoiceId.THORSTEN])

    assert outcome == Refused(UNAVAILABLE)
    assert service.asked == []


def test_a_service_nothing_listens_on_is_unreachable(refused_url: str) -> None:
    assert _speak(refused_url, [VoiceId.THORSTEN]) == Unreachable()


@pytest.mark.parametrize(
    ("answer", "outcome"),
    [
        pytest.param(
            b"HTTP/1.1 503 Service Unavailable\r\nContent-Length: 0\r\n\r\n",
            Malformed(),
            id="status-other-than-200",
        ),
        pytest.param(
            b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\n{}",
            Malformed(),
            id="no-health",
        ),
        pytest.param(
            b"HTTP/1.1 200 OK\r\nContent-Length: 70000\r\n\r\n" + b" " * 70000,
            Malformed(),
            id="body-over-the-limit",
        ),
    ],
)
def test_a_health_that_breaks_the_wire_is_malformed(
    raw_listener: RawListener, answer: bytes, outcome: Malformed
) -> None:
    with raw_listener(answer) as (base_url, requests):
        assert _speak(base_url, [VoiceId.THORSTEN]) == outcome

    assert requests[0].startswith(f"GET {HEALTH_PATH} ".encode())


@pytest.mark.parametrize(
    ("clock", "asked"),
    [
        pytest.param([0.0, 0.0, PAST], [], id="after-the-health-read"),
        pytest.param([0.0, 0.0, 0.0, PAST], [VoiceId.RYAN], id="after-a-speak"),
    ],
)
def test_the_reads_and_speaks_share_one_deadline(
    monkeypatch: pytest.MonkeyPatch, clock: list[float], asked: list[VoiceId]
) -> None:
    offered = {VoiceId.RYAN: True, VoiceId.THORSTEN: True}
    now = iter(clock)
    monkeypatch.setattr(voice_preference, "monotonic", lambda: next(now))
    with scripted_service(offered) as (base_url, service):
        service.unavailable_on_speak.add(VoiceId.RYAN)
        outcome = _speak(base_url, [VoiceId.RYAN, VoiceId.THORSTEN])

    assert outcome == PastDeadline()
    assert [VoiceId(request["voice"]) for request in service.asked] == asked


def test_an_audio_over_the_cap_is_passed_on() -> None:
    with scripted_service({VoiceId.THORSTEN: True}) as (base_url, _):
        outcome = speak_with_preference(
            base_url,
            "Hallo",
            [VoiceId.THORSTEN],
            response_format=ResponseFormat.WAV,
            workload_token="workload-token",  # noqa: S106 - a stand-in the script never checks
            person_token="person-token",  # noqa: S106 - a stand-in the script never checks
            deadline_seconds=DEADLINE_SECONDS,
            cap_bytes=1,
        )

    assert outcome == OverCap()


def test_an_empty_preference_raises() -> None:
    with pytest.raises(ValueError, match=r"preference"):
        _speak("http://127.0.0.1:1", [])


def test_a_defect_of_the_request_raises_before_the_network() -> None:
    with pytest.raises(ValidationError):
        _speak(
            "http://127.0.0.1:1",
            [VoiceId.THORSTEN, VoiceId.RYAN],
            language=None,
        )

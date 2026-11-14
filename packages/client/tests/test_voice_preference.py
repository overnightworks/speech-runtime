"""The speaking call with an ordered voice preference against a scripted service."""

from __future__ import annotations

from time import monotonic
from typing import TYPE_CHECKING

import pytest
from pydantic import ValidationError

from packages.client.tests.support.scripted_service_support import (
    OGG,
    SAMPLE_RATE,
    WAV,
    scripted_service,
)
from speech_client import voice_preference
from speech_client.outcomes import Cut, Malformed, OverCap, PastDeadline, Unreachable
from speech_client.speak import Finished, Refused
from speech_client.voice_preference import (
    SpokenWithPreference,
    VoiceChosen,
    speak_stream_with_preference,
    speak_with_preference,
)
from speech_client.wire import (
    HEALTH_PATH,
    Language,
    RefusalCode,
    ResponseFormat,
    VoiceId,
    refusal,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from packages.client.tests.support.listener_support import RawListener
    from speech_client.voice_preference import (
        PreferenceOutcome,
        PreferenceStreamItem,
    )

DEADLINE_SECONDS = 30.0
CAP_BYTES = 1024 * 1024
TONE = "calm"
PAST = DEADLINE_SECONDS + 1
UNAVAILABLE = refusal(RefusalCode.UNAVAILABLE, "voice")


def _speak(  # noqa: PLR0913 - the arguments a test varies, all keyword-only after the first two
    base_url: str,
    preference: Sequence[VoiceId],
    *,
    language: Language | None = Language.DE,
    instructions: str | None = None,
    deadline_seconds: float = DEADLINE_SECONDS,
    cap_bytes: int = CAP_BYTES,
    clock: Callable[[], float] = monotonic,
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
        cap_bytes=cap_bytes,
        clock=clock,
    )


def _stream(
    base_url: str,
    preference: Sequence[VoiceId],
    *,
    language: Language | None = Language.DE,
    instructions: str | None = None,
) -> list[PreferenceStreamItem]:
    return list(
        speak_stream_with_preference(
            base_url,
            "Hallo",
            preference,
            language=language,
            instructions=instructions,
            workload_token="workload-token",  # noqa: S106 - a stand-in the script never checks
            person_token="person-token",  # noqa: S106 - a stand-in the script never checks
            deadline_seconds=DEADLINE_SECONDS,
            cap_bytes=CAP_BYTES,
        )
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


class _Clock:
    """A clock a test moves past the deadline when the service does something."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def pass_the_deadline(self, *_ignored: object) -> None:
        self.now = PAST


@pytest.mark.parametrize(
    ("passes_on", "asked"),
    [
        pytest.param("health", [], id="after-the-health-read"),
        pytest.param("speak", [VoiceId.RYAN], id="after-a-speak"),
    ],
)
def test_the_reads_and_speaks_share_one_deadline(
    passes_on: str, asked: list[VoiceId]
) -> None:
    offered = {VoiceId.RYAN: True, VoiceId.THORSTEN: True}
    clock = _Clock()
    with scripted_service(offered) as (base_url, service):
        service.unavailable_on_speak.add(VoiceId.RYAN)
        if passes_on == "health":
            service.on_health = clock.pass_the_deadline
        else:
            service.on_speak = clock.pass_the_deadline
        outcome = _speak(base_url, [VoiceId.RYAN, VoiceId.THORSTEN], clock=clock)

    assert outcome == PastDeadline()
    assert [VoiceId(request["voice"]) for request in service.asked] == asked


def test_a_deadline_passed_after_the_voice_is_chosen_ends_before_the_audio(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = _Clock()
    monkeypatch.setattr(voice_preference, "monotonic", clock)
    with scripted_service({VoiceId.RYAN: True}) as (base_url, _):
        stream = speak_stream_with_preference(
            base_url,
            "Hallo",
            [VoiceId.RYAN],
            language=Language.DE,
            workload_token="workload-token",  # noqa: S106 - a stand-in the script never checks
            person_token="person-token",  # noqa: S106 - a stand-in the script never checks
            deadline_seconds=DEADLINE_SECONDS,
            cap_bytes=CAP_BYTES,
        )
        chosen = next(stream)
        clock.pass_the_deadline()
        rest = list(stream)

    assert chosen == VoiceChosen(VoiceId.RYAN, instructions_left_out=False)
    assert rest == [PastDeadline()]


@pytest.mark.parametrize(
    ("deadline_seconds", "cap_bytes"),
    [
        pytest.param(0.0, CAP_BYTES, id="deadline-of-zero"),
        pytest.param(-1.0, CAP_BYTES, id="deadline-below-zero"),
        pytest.param(DEADLINE_SECONDS, 0, id="cap-of-zero"),
        pytest.param(DEADLINE_SECONDS, -1, id="cap-below-zero"),
    ],
)
def test_a_bound_at_or_below_zero_raises_even_when_no_voice_is_offered(
    deadline_seconds: float, cap_bytes: int
) -> None:
    with (
        scripted_service({}) as (base_url, _),
        pytest.raises(ValueError, match=r"above 0"),
    ):
        _speak(
            base_url,
            [VoiceId.THORSTEN],
            deadline_seconds=deadline_seconds,
            cap_bytes=cap_bytes,
        )


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


def test_the_stream_names_the_voice_first_then_the_audio_then_the_exact_seconds() -> (
    None
):
    offered = {VoiceId.THORSTEN: True, VoiceId.RYAN: True}
    with scripted_service(offered) as (base_url, _):
        first, *chunks, end = _stream(
            base_url,
            [VoiceId.THORSTEN, VoiceId.RYAN],
            instructions=TONE,
        )

    assert first == VoiceChosen(VoiceId.THORSTEN, instructions_left_out=True)
    assert b"".join(chunks) == WAV
    assert end == Finished(4 / SAMPLE_RATE)


def test_the_stream_moves_on_when_a_voice_is_unavailable_before_any_chunk() -> None:
    offered = {VoiceId.RYAN: True, VoiceId.THORSTEN: True}
    with scripted_service(offered) as (base_url, service):
        service.unavailable_on_speak.add(VoiceId.RYAN)
        first, *chunks, end = _stream(base_url, [VoiceId.RYAN, VoiceId.THORSTEN])

    assert first == VoiceChosen(VoiceId.THORSTEN, instructions_left_out=False)
    assert b"".join(chunks) == WAV
    assert isinstance(end, Finished)


def test_the_stream_ends_as_the_failure_after_a_chunk_and_tries_no_other_voice() -> (
    None
):
    offered = {VoiceId.RYAN: True, VoiceId.THORSTEN: True}
    with scripted_service(offered) as (base_url, service):
        service.cut_after_first_chunk.add(VoiceId.RYAN)
        first, *chunks, end = _stream(base_url, [VoiceId.RYAN, VoiceId.THORSTEN])

    assert first == VoiceChosen(VoiceId.RYAN, instructions_left_out=False)
    assert b"".join(chunks) == WAV
    assert end == Cut()
    assert [asked["voice"] for asked in service.asked] == [VoiceId.RYAN]


def test_the_stream_with_no_voice_offered_is_the_refusal_alone() -> None:
    with scripted_service({VoiceId.KERSTIN: True}) as (base_url, _):
        items = _stream(base_url, [VoiceId.RYAN, VoiceId.THORSTEN])

    assert items == [Refused(UNAVAILABLE)]


def test_the_stream_raises_at_the_call_not_at_its_first_item() -> None:
    with pytest.raises(ValueError, match=r"preference"):
        speak_stream_with_preference(
            "http://127.0.0.1:1",
            "Hallo",
            [],
            workload_token="workload-token",  # noqa: S106 - a stand-in the script never checks
            person_token="person-token",  # noqa: S106 - a stand-in the script never checks
            deadline_seconds=DEADLINE_SECONDS,
            cap_bytes=CAP_BYTES,
        )


def test_the_blocking_speak_ends_as_the_cut_after_a_chunk_without_another_voice() -> (
    None
):
    offered = {VoiceId.RYAN: True, VoiceId.THORSTEN: True}
    with scripted_service(offered) as (base_url, service):
        service.cut_after_first_chunk.add(VoiceId.RYAN)
        outcome = _speak(base_url, [VoiceId.RYAN, VoiceId.THORSTEN])

    assert outcome == Cut()
    assert [asked["voice"] for asked in service.asked] == [VoiceId.RYAN]


def test_an_ogg_speak_returns_the_audio_judged_by_its_signature() -> None:
    with scripted_service({VoiceId.THORSTEN: True}) as (base_url, service):
        service.audio_format = ResponseFormat.OPUS
        outcome = speak_with_preference(
            base_url,
            "Hallo",
            [VoiceId.THORSTEN],
            response_format=ResponseFormat.OPUS,
            workload_token="workload-token",  # noqa: S106 - a stand-in the script never checks
            person_token="person-token",  # noqa: S106 - a stand-in the script never checks
            deadline_seconds=DEADLINE_SECONDS,
            cap_bytes=CAP_BYTES,
        )

    assert outcome == SpokenWithPreference(
        OGG, VoiceId.THORSTEN, instructions_left_out=False
    )

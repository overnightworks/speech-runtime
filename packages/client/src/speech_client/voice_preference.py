"""The speaking call with an ordered voice preference.

`speak_with_preference` takes the voices an app would speak with, best first,
and speaks with the first one that health (WIRE-125) offers as ready in the
requested language, so an app keeps speaking when the service offers fewer
voices, for example after a switch from the GPU image to the small one. It
returns one `PreferenceOutcome`, a closed union a caller matches with
`assert_never`:

- `SpokenWithPreference`: the audio, the voice that spoke and whether the
  `instructions` were left out. They go only to a voice that takes them
  (WIRE-142); for any other voice they are left out and the outcome says so.
- `Refused`, `Busy`, `Cut`, `PastDeadline`, `OverCap`, `Malformed` and
  `Unreachable`, as for `speak` (`speech_client.speak`, `speech_client.outcomes`).
  `Refused` is `unavailable` for the field `voice` when no voice of the
  preference is offered, or when every offered one answered `unavailable`.
  A health fetch that breaks the wire (a status other than 200, a body that is
  no health or is over 64 KiB) is `Malformed`.

A voice that answers `unavailable` to its speak, for example because the
runtime switched mid-call, hands over to the next voice of the preference: the
call reads health again and speaks with the first later voice offered. The
`deadline_seconds` bound the health reads and the speaks together.

A defect of the caller raises and is no outcome: an empty preference, an input
the request refuses for any voice of the preference (`pydantic.ValidationError`),
and what `speak` raises.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from time import monotonic
from typing import TYPE_CHECKING, Final

from speech_client._transport import checked_origin
from speech_client.health import UnexpectedStatus, fetch_health
from speech_client.outcomes import (
    Busy,
    Cut,
    Malformed,
    OverCap,
    PastDeadline,
    Unreachable,
)
from speech_client.speak import Refused, Spoken, speak
from speech_client.wire import (
    VOICES,
    Health,
    Language,
    RefusalCode,
    ResponseFormat,
    VoiceId,
    refusal,
    speech_request_body,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from speech_client.health import HealthOutcome
    from speech_client.speak import SpeakOutcome

__all__ = [
    "Busy",
    "Cut",
    "Malformed",
    "OverCap",
    "PastDeadline",
    "PreferenceOutcome",
    "Refused",
    "SpokenWithPreference",
    "Unreachable",
    "speak_with_preference",
]

_HEALTH_BODY_LIMIT: Final = 64 * 1024


@dataclass(frozen=True, slots=True)
class SpokenWithPreference:
    """The audio, the voice that spoke it and whether `instructions` were left out."""

    audio: bytes = field(repr=False)
    voice: VoiceId
    instructions_left_out: bool


type PreferenceOutcome = (
    SpokenWithPreference
    | Refused
    | Busy
    | Cut
    | PastDeadline
    | OverCap
    | Malformed
    | Unreachable
)


def _instructions_for(voice: VoiceId, instructions: str | None) -> str | None:
    return instructions if VOICES[voice].takes_instructions else None


def _unhealthy(health: HealthOutcome) -> PreferenceOutcome:
    if isinstance(health, UnexpectedStatus | OverCap):
        return Malformed()
    return health


def _is_unavailable(outcome: SpeakOutcome) -> bool:
    return isinstance(outcome, Refused) and outcome.refusal == refusal(
        RefusalCode.UNAVAILABLE, "voice"
    )


def _next_voice(
    base_url: str,
    remaining: Sequence[VoiceId],
    language: Language | None,
    seconds_left: float,
) -> VoiceId | PreferenceOutcome:
    health = fetch_health(
        base_url, deadline_seconds=seconds_left, cap_bytes=_HEALTH_BODY_LIMIT
    )
    if not isinstance(health, Health):
        return _unhealthy(health)
    voice = _first_offered(health, remaining, language)
    if voice is None:
        return Refused(refusal(RefusalCode.UNAVAILABLE, "voice"))
    return voice


def _seconds_left(
    deadline_seconds: float, began: float, clock: Callable[[], float]
) -> float:
    return deadline_seconds - (clock() - began)


def _first_offered(
    health: Health, preference: Sequence[VoiceId], language: Language | None
) -> VoiceId | None:
    voices = health.speaking.voices
    return next(
        (
            voice
            for voice in preference
            if (offered := voices.get(voice.value)) is not None
            and offered.ready
            and (language is None or language in offered.languages)
        ),
        None,
    )


def _checked_speakers(  # noqa: PLR0913
    base_url: str,
    text: str,
    preference: Sequence[VoiceId],
    *,
    language: Language | None,
    instructions: str | None,
    response_format: ResponseFormat,
    deadline_seconds: float,
    cap_bytes: int,
) -> tuple[VoiceId, ...]:
    """Check every bound and request, then keep the voices that speak `language`."""
    if not preference:
        message = "preference must name at least one voice"
        raise ValueError(message)
    checked_origin(base_url, deadline_seconds, cap_bytes)
    speakers = tuple(
        voice
        for voice in preference
        if language is None or language in VOICES[voice].languages
    )
    for voice in speakers:
        speech_request_body(
            text,
            voice,
            response_format,
            language,
            _instructions_for(voice, instructions),
        )
    return speakers


def _speak_with_voice(  # noqa: PLR0913
    base_url: str,
    text: str,
    voice: VoiceId,
    *,
    language: Language | None,
    instructions: str | None,
    response_format: ResponseFormat,
    workload_token: str,
    person_token: str,
    deadline_seconds: float,
    cap_bytes: int,
) -> SpeakOutcome | SpokenWithPreference:
    voice_instructions = _instructions_for(voice, instructions)
    outcome = speak(
        base_url,
        text,
        voice,
        language=language,
        instructions=voice_instructions,
        response_format=response_format,
        workload_token=workload_token,
        person_token=person_token,
        deadline_seconds=deadline_seconds,
        cap_bytes=cap_bytes,
    )
    if not isinstance(outcome, Spoken):
        return outcome
    return SpokenWithPreference(
        outcome.audio,
        voice,
        instructions_left_out=instructions is not None and voice_instructions is None,
    )


# Every input comes from the call site; only `language` and `instructions` default.
def speak_with_preference(  # noqa: PLR0913
    base_url: str,
    text: str,
    preference: Sequence[VoiceId],
    *,
    language: Language | None = None,
    instructions: str | None = None,
    response_format: ResponseFormat,
    workload_token: str,
    person_token: str,
    deadline_seconds: float,
    cap_bytes: int,
    clock: Callable[[], float] = monotonic,
) -> PreferenceOutcome:
    """Speak `text` with the first voice of `preference` the service offers.

    `language` narrows the voices to those that speak it and is required when a
    voice of the preference speaks several; a voice that does not speak it is
    skipped. `instructions` name the tone, for a voice that takes them.
    `deadline_seconds` bounds the whole call, `cap_bytes` the audio, `clock`
    reads the time the deadline runs on. See the module for the outcomes.
    """
    remaining = _checked_speakers(
        base_url,
        text,
        preference,
        language=language,
        instructions=instructions,
        response_format=response_format,
        deadline_seconds=deadline_seconds,
        cap_bytes=cap_bytes,
    )
    began = clock()
    while True:
        seconds_left = _seconds_left(deadline_seconds, began, clock)
        if seconds_left <= 0:
            return PastDeadline()
        voice = _next_voice(base_url, remaining, language, seconds_left)
        if not isinstance(voice, VoiceId):
            return voice
        seconds_left = _seconds_left(deadline_seconds, began, clock)
        if seconds_left <= 0:
            return PastDeadline()
        outcome = _speak_with_voice(
            base_url,
            text,
            voice,
            language=language,
            instructions=instructions,
            response_format=response_format,
            workload_token=workload_token,
            person_token=person_token,
            deadline_seconds=seconds_left,
            cap_bytes=cap_bytes,
        )
        if not _is_unavailable(outcome):
            return outcome
        remaining = remaining[remaining.index(voice) + 1 :]

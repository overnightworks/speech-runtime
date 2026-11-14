"""The speaking calls with an ordered voice preference.

`speak_stream_with_preference` takes the voices an app would speak with, best
first, and speaks with the first one that health (WIRE-125) offers as ready in
the requested language, so an app keeps speaking when the service offers fewer
voices, for example after a switch from the GPU image to the small one. It
streams the WAV as `speak_stream` does (`speech_client.speak`) and yields, in
this order:

- `VoiceChosen`: the voice that speaks and whether the `instructions` were left
  out. They go only to a voice that takes them (WIRE-142); for any other voice
  they are left out and the item says so. It comes just before the first chunk,
  so a consumer knows the voice before any audio.
- the `bytes` chunks of the WAV.
- `Finished` with the exact seconds of audio, or a failure: `Refused`, `Busy`,
  `Cut`, `PastDeadline`, `OverCap`, `Malformed` or `Unreachable`, as for `speak`
  (`speech_client.outcomes`). A failure before any chunk is the only item.
  `Refused` is `unavailable` for the field `voice` when no voice of the
  preference is offered, or when every offered one answered `unavailable`. A
  health fetch that breaks the wire (a status other than 200, a body that is no
  health or is over 64 KiB) is `Malformed`.

A voice that answers `unavailable` to its speak before any chunk, for example
because the runtime switched mid-call, hands over to the next voice of the
preference: the stream reads health again and speaks with the first later voice
offered. Once a chunk came, a failure ends the stream as that failure and no
other voice is tried. The `deadline_seconds` bound the health reads and the
speaks together and start with the first item asked for.

`speak_with_preference` is that stream read to its end, so both share the voice
choice. It returns one `PreferenceOutcome`, a closed union a caller matches with
`assert_never`: `SpokenWithPreference` (the audio, the voice that spoke and
whether the `instructions` were left out) or one of the failures above. It also
speaks Ogg, which carries no length and so ends without `Finished`.

A defect of the caller raises at the call and is no outcome: an empty
preference, an input the request refuses for any voice of the preference
(`pydantic.ValidationError`), and what `speak` raises.
"""

from __future__ import annotations

from contextlib import closing
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
from speech_client.speak import (
    Finished,
    Refused,
    _chunks,
    _with_length,
)
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
    from collections.abc import Callable, Collection, Generator, Sequence

    from speech_client.health import HealthOutcome
    from speech_client.speak import StreamFailure

__all__ = [
    "Busy",
    "Cut",
    "Finished",
    "Malformed",
    "OverCap",
    "PastDeadline",
    "PreferenceOutcome",
    "PreferenceStreamItem",
    "Refused",
    "SpokenWithPreference",
    "Unreachable",
    "VoiceChosen",
    "speak_stream_with_preference",
    "speak_with_preference",
]

_HEALTH_BODY_LIMIT: Final = 64 * 1024


@dataclass(frozen=True, slots=True)
class SpokenWithPreference:
    """The audio, the voice that spoke it and whether `instructions` were left out."""

    audio: bytes = field(repr=False)
    voice: VoiceId
    instructions_left_out: bool


@dataclass(frozen=True, slots=True)
class VoiceChosen:
    """The voice that speaks and whether `instructions` were left out."""

    voice: VoiceId
    instructions_left_out: bool


type PreferenceStreamItem = VoiceChosen | bytes | Finished | StreamFailure
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


def _is_unavailable(item: bytes | StreamFailure) -> bool:
    return isinstance(item, Refused) and item.refusal == refusal(
        RefusalCode.UNAVAILABLE, "voice"
    )


def _speaks(languages: Collection[Language], language: Language | None) -> bool:
    """Whether a voice with `languages` speaks `language`; the one decider.

    The catalogue answers it before the network, health answers it for the
    voices the service offers now.
    """
    return language is None or language in languages


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
            and _speaks(offered.languages, language)
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
        voice for voice in preference if _speaks(VOICES[voice].languages, language)
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


@dataclass(frozen=True, slots=True)
class _Speaking:
    """The voice that answered with audio, its first chunk and the chunks after it."""

    chosen: VoiceChosen
    first: bytes
    chunks: Generator[bytes | StreamFailure]
    seconds_left: Callable[[], float]

    def resumed(self) -> Generator[bytes | StreamFailure]:
        """The first chunk and the ones after it, unless the deadline passed.

        The first chunk is already read, so no socket read ends it at the
        deadline; the consumer may have held the stream past it.
        """
        if self.seconds_left() <= 0:
            yield PastDeadline()
            return
        yield self.first
        yield from self.chunks


def _choose(  # noqa: PLR0913
    base_url: str,
    text: str,
    speakers: Sequence[VoiceId],
    *,
    language: Language | None,
    instructions: str | None,
    response_format: ResponseFormat,
    workload_token: str,
    person_token: str,
    deadline_seconds: float,
    cap_bytes: int,
    clock: Callable[[], float],
) -> _Speaking | StreamFailure:
    """The first voice that does not answer `unavailable` before any audio."""
    remaining = speakers
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
        voice_instructions = _instructions_for(voice, instructions)
        chunks = _chunks(
            base_url,
            text,
            voice,
            language=language,
            instructions=voice_instructions,
            response_format=response_format,
            workload_token=workload_token,
            person_token=person_token,
            deadline_seconds=seconds_left,
            cap_bytes=cap_bytes,
        )
        first = next(chunks)
        if isinstance(first, bytes):
            return _Speaking(
                VoiceChosen(
                    voice,
                    instructions_left_out=instructions is not None
                    and voice_instructions is None,
                ),
                first,
                chunks,
                lambda: _seconds_left(deadline_seconds, began, clock),
            )
        chunks.close()
        if not _is_unavailable(first):
            return first
        remaining = remaining[remaining.index(voice) + 1 :]


def _stream(  # noqa: PLR0913
    base_url: str,
    text: str,
    speakers: Sequence[VoiceId],
    *,
    language: Language | None,
    instructions: str | None,
    response_format: ResponseFormat,
    workload_token: str,
    person_token: str,
    deadline_seconds: float,
    cap_bytes: int,
    clock: Callable[[], float],
) -> Generator[PreferenceStreamItem]:
    """The one stream: the voice, its chunks and the end a WAV carries."""
    speaking = _choose(
        base_url,
        text,
        speakers,
        language=language,
        instructions=instructions,
        response_format=response_format,
        workload_token=workload_token,
        person_token=person_token,
        deadline_seconds=deadline_seconds,
        cap_bytes=cap_bytes,
        clock=clock,
    )
    if not isinstance(speaking, _Speaking):
        yield speaking
    else:
        with closing(speaking.chunks):
            yield speaking.chosen
            # Only a WAV carries a length to check; Ogg is judged by its signature.
            if response_format is ResponseFormat.WAV:
                yield from _with_length(speaking.resumed())
            else:
                yield from speaking.resumed()


# Every input comes from the call site; only `language` and `instructions` default.
def speak_stream_with_preference(  # noqa: PLR0913
    base_url: str,
    text: str,
    preference: Sequence[VoiceId],
    *,
    language: Language | None = None,
    instructions: str | None = None,
    workload_token: str,
    person_token: str,
    deadline_seconds: float,
    cap_bytes: int,
) -> Generator[PreferenceStreamItem]:
    """Speak `text` with the first voice of `preference` the service offers.

    `language` narrows the voices to those that speak it and is required when a
    voice of the preference speaks several; a voice that does not speak it is
    skipped. `instructions` name the tone, for a voice that takes them.
    `deadline_seconds` bounds the whole stream, `cap_bytes` the audio. See the
    module for the items and for what raises, which happens at the call.
    """
    return _stream(
        base_url,
        text,
        _checked_speakers(
            base_url,
            text,
            preference,
            language=language,
            instructions=instructions,
            response_format=ResponseFormat.WAV,
            deadline_seconds=deadline_seconds,
            cap_bytes=cap_bytes,
        ),
        language=language,
        instructions=instructions,
        response_format=ResponseFormat.WAV,
        workload_token=workload_token,
        person_token=person_token,
        deadline_seconds=deadline_seconds,
        cap_bytes=cap_bytes,
        clock=monotonic,
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

    This is `speak_stream_with_preference` read to its end, so its arguments
    apply; `response_format` adds Ogg. `clock` reads the time the deadline runs
    on and is a seam for tests: leave it at its default.
    """
    items = _stream(
        base_url,
        text,
        _checked_speakers(
            base_url,
            text,
            preference,
            language=language,
            instructions=instructions,
            response_format=response_format,
            deadline_seconds=deadline_seconds,
            cap_bytes=cap_bytes,
        ),
        language=language,
        instructions=instructions,
        response_format=response_format,
        workload_token=workload_token,
        person_token=person_token,
        deadline_seconds=deadline_seconds,
        cap_bytes=cap_bytes,
        clock=clock,
    )
    with closing(items):
        chosen = next(items)
        if not isinstance(chosen, VoiceChosen):
            return chosen
        audio = bytearray()
        for item in items:
            if isinstance(item, bytes):
                audio += item
            elif not isinstance(item, Finished):
                return item
    return SpokenWithPreference(
        bytes(audio), chosen.voice, chosen.instructions_left_out
    )

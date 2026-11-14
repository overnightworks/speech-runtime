"""The speaking call against a raw listener: what the real service never sends."""

from __future__ import annotations

import socket
import struct
import time
from typing import TYPE_CHECKING

import pytest
from pydantic import ValidationError

from speech_client import speak as speak_module
from speech_client.speak import (
    Busy,
    Cut,
    Finished,
    Malformed,
    PastDeadline,
    Spoken,
    Unreachable,
    speak,
    speak_stream,
)
from speech_client.wire import (
    RETRY_AFTER_HEADER,
    SPEECH_PATH,
    VOICES,
    Language,
    RefusalCode,
    ResponseFormat,
    VoiceId,
    refusal,
    speech_request_body,
    speech_request_headers,
)

if TYPE_CHECKING:
    import threading

    from speech_client.speak import SpeakOutcome, SpeakStreamItem
    from support.listener_support import Answer, RawListener

VOICE = VoiceId.THORSTEN
WORKLOAD_TOKEN = "workload-token"  # noqa: S105 - a stand-in the raw listener never checks
PERSON_TOKEN = "person-token"  # noqa: S105 - a stand-in the raw listener never checks
DEADLINE_SECONDS = 0.5
MARGIN_SECONDS = 3.0
GENEROUS_DEADLINE_SECONDS = 30.0
CAP_BYTES = 1024 * 1024
DRIP_SECONDS = 0.02


def _speak(  # noqa: PLR0913 - one keyword per call-site option
    base_url: str,
    *,
    voice: VoiceId = VOICE,
    language: Language | None = None,
    instructions: str | None = None,
    response_format: ResponseFormat = ResponseFormat.WAV,
    deadline_seconds: float = GENEROUS_DEADLINE_SECONDS,
    cap_bytes: int = CAP_BYTES,
) -> SpeakOutcome:
    return speak(
        base_url,
        "Hallo",
        voice,
        language=language,
        instructions=instructions,
        response_format=response_format,
        workload_token=WORKLOAD_TOKEN,
        person_token=PERSON_TOKEN,
        deadline_seconds=deadline_seconds,
        cap_bytes=cap_bytes,
    )


def test_a_port_nothing_listens_on_is_unreachable(refused_url: str) -> None:
    assert _speak(refused_url) == Unreachable()


@pytest.mark.parametrize(
    "options",
    [
        {"deadline_seconds": 0},
        {"deadline_seconds": -1},
        {"cap_bytes": 0},
        {"cap_bytes": -1},
    ],
)
def test_a_defect_of_the_caller_raises(options: dict[str, float]) -> None:
    with pytest.raises(ValueError, match=r"."):
        _speak("http://127.0.0.1:1", **options)


@pytest.mark.parametrize(
    "base_url",
    ["https://127.0.0.1:1", "http://127.0.0.1:1/v1", "127.0.0.1:1", "http://:1"],
)
def test_a_base_url_that_is_not_plain_http_without_a_path_raises(
    base_url: str,
) -> None:
    with pytest.raises(ValueError, match=r"base_url"):
        _speak(base_url)


# What the raw listener answers.


def _chunk(data: bytes) -> bytes:
    return b"%x\r\n%b\r\n" % (len(data), data)


def _head(status: str, *headers: str) -> bytes:
    return "\r\n".join([f"HTTP/1.1 {status}", *headers, "", ""]).encode()


_WAV_HEAD = _head("200 OK", "Content-Type: audio/wav", "Transfer-Encoding: chunked")
_OGG_HEAD = _head("200 OK", "Content-Type: audio/ogg", "Transfer-Encoding: chunked")
_WAV_BODY = b"RIFF\0\0\0\0WAVE"
_OGG_BODY = b"OggS\0\0\0\0"


def _refusal_response(
    status: str, code: RefusalCode, *headers: str, padding: int = 0
) -> bytes:
    body = refusal(code).model_dump_json().encode() + b" " * padding
    return _head(status, f"Content-Length: {len(body)}", *headers) + body


def _sends_then_drips(raw: bytes, drip: bytes) -> Answer:
    def answer(connection: socket.socket, stopped: threading.Event) -> None:
        connection.sendall(raw)
        while not stopped.wait(DRIP_SECONDS):
            connection.sendall(drip)

    return answer


def _resets(connection: socket.socket, stopped: threading.Event) -> None:
    del stopped
    connection.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))


def _closes(connection: socket.socket, stopped: threading.Event) -> None:
    del connection, stopped


@pytest.mark.parametrize(
    ("raw", "drip"),
    [
        pytest.param(b"HTTP/1.1 200 OK\r\n", b"X", id="inside-a-header-line"),
        pytest.param(_WAV_HEAD, b"1", id="inside-a-chunk-size-line"),
        pytest.param(_WAV_HEAD + b"ffff\r\n" + _WAV_BODY, b"\0", id="inside-a-chunk"),
    ],
)
def test_a_server_that_drips_cannot_hold_the_call_past_the_deadline(
    raw: bytes, drip: bytes, raw_listener: RawListener
) -> None:
    """The deadline bounds the whole call, not each read."""
    with raw_listener(_sends_then_drips(raw, drip)) as (base_url, _):
        began = time.monotonic()
        outcome = _speak(base_url, deadline_seconds=DEADLINE_SECONDS)

    assert outcome == PastDeadline()
    assert time.monotonic() - began < DEADLINE_SECONDS + MARGIN_SECONDS


def test_every_outcome_is_exported_from_the_speak_module() -> None:
    """A type-checked consumer imports the outcomes from `speak` publicly."""
    exported = {
        "Spoken",
        "Finished",
        "SpeakStreamItem",
        "speak_stream",
        "Refused",
        "Busy",
        "Cut",
        "PastDeadline",
        "OverCap",
        "Malformed",
        "Unreachable",
        "SpeakOutcome",
        "speak",
    }

    assert exported <= set(speak_module.__all__)


def test_a_deadline_that_has_passed_before_the_connect_is_past_the_deadline(
    raw_listener: RawListener,
) -> None:
    with raw_listener(_closes) as (base_url, _):
        outcome = _speak(base_url, deadline_seconds=1e-9)

    assert outcome == PastDeadline()


@pytest.mark.parametrize(
    "answer",
    [
        pytest.param(_resets, id="reset-before-the-status-line"),
        pytest.param(_closes, id="close-before-the-status-line"),
        pytest.param(_WAV_HEAD + _chunk(_WAV_BODY), id="close-without-the-end"),
        pytest.param(
            _WAV_HEAD + _chunk(_WAV_BODY) + b"0",
            id="close-inside-the-last-chunk-line",
        ),
        pytest.param(
            _WAV_HEAD + _chunk(_WAV_BODY) + b"0\r\n",
            id="close-before-the-final-blank-line",
        ),
        pytest.param(
            _head("422 Unprocessable Entity", "Content-Length: 500") + b"{",
            id="refusal-shorter-than-its-length",
        ),
    ],
)
def test_an_answer_that_ends_before_it_is_whole_is_cut(
    answer: Answer, raw_listener: RawListener
) -> None:
    """WIRE-90: a missing end is a failed request."""
    with raw_listener(answer) as (base_url, _):
        outcome = _speak(base_url)

    assert outcome == Cut()


_CANONICAL = refusal(RefusalCode.INVALID_REQUEST)
_ECHOED = (
    _CANONICAL.model_copy(
        update={
            "error": _CANONICAL.error.model_copy(update={"message": PERSON_TOKEN}),
        },
    )
    .model_dump_json()
    .encode()
)
_BUSY_WITHOUT_WAIT = _refusal_response("429 Too Many Requests", RefusalCode.BUSY)


@pytest.mark.parametrize(
    "raw",
    [
        pytest.param(b"not http at all\r\n\r\n", id="bad-status-line"),
        pytest.param(b"HTTP/1.1 200 " + b"a" * 70_000 + b"\r\n", id="line-too-long"),
        pytest.param(
            _head("200 OK", *("X-Header: 1",) * 101), id="more-than-100-headers"
        ),
        pytest.param(
            _head(
                "200 OK", "Content-Type: application/json", "Transfer-Encoding: chunked"
            )
            + _chunk(b"{}")
            + _chunk(b""),
            id="200-that-is-not-audio-wav",
        ),
        pytest.param(
            _head("200 OK", "Content-Type: audio/wav", "Content-Length: 4") + b"RIFF",
            id="200-that-is-not-chunked",
        ),
        pytest.param(
            _head("422 Unprocessable Entity", "Content-Length: 15")
            + b'{"detail":"no"}',
            id="refusal-outside-the-error-form",
        ),
        pytest.param(
            _refusal_response("401 Unauthorized", RefusalCode.NOT_FOUND),
            id="refusal-whose-code-has-another-status",
        ),
        pytest.param(
            _head("422 Unprocessable Entity", f"Content-Length: {len(_ECHOED)}")
            + _ECHOED,
            id="refusal-with-a-message-that-is-not-the-canonical-one",
        ),
        pytest.param(_BUSY_WITHOUT_WAIT, id="429-without-retry-after"),
        pytest.param(
            _refusal_response(
                "429 Too Many Requests", RefusalCode.BUSY, f"{RETRY_AFTER_HEADER}: soon"
            ),
            id="429-with-a-retry-after-that-is-no-number",
        ),
        pytest.param(
            _refusal_response("500 Internal Server Error", RefusalCode.UNAVAILABLE),
            id="a-status-the-wire-does-not-name",
        ),
        pytest.param(
            _refusal_response(
                "422 Unprocessable Entity",
                RefusalCode.INVALID_REQUEST,
                padding=64 * 1024,
            ),
            id="refusal-body-over-64-kib",
        ),
        pytest.param(
            _WAV_HEAD
            + _chunk(_WAV_BODY)
            + b"0\r\n"
            + b"X-Trailer: "
            + b"a" * 70_000
            + b"\r\n\r\n",
            id="trailer-line-too-long",
        ),
    ],
)
def test_an_answer_that_breaks_the_wire_is_malformed(
    raw: bytes, raw_listener: RawListener
) -> None:
    """WIRE-88: only a chunked audio/wav 200 and the wire's refusals are answers."""
    with raw_listener(raw) as (base_url, _):
        outcome = _speak(base_url)

    assert outcome == Malformed()


def test_a_chunked_wav_with_a_trailer_is_spoken(raw_listener: RawListener) -> None:
    raw = _WAV_HEAD + _chunk(_WAV_BODY) + b"0\r\nX-Trailer: 1\r\n\r\n"
    with raw_listener(raw) as (base_url, _):
        outcome = _speak(base_url)

    assert outcome == Spoken(_WAV_BODY)


@pytest.mark.parametrize(
    ("ordered", "raw"),
    [
        pytest.param(
            ResponseFormat.OPUS,
            _WAV_HEAD + _chunk(_WAV_BODY) + _chunk(b""),
            id="wav-media-type-for-an-opus-order",
        ),
        pytest.param(
            ResponseFormat.WAV,
            _OGG_HEAD + _chunk(_OGG_BODY) + _chunk(b""),
            id="ogg-media-type-for-a-wav-order",
        ),
        pytest.param(
            ResponseFormat.WAV,
            _WAV_HEAD + _chunk(b"not audio at all") + _chunk(b""),
            id="wav-media-type-over-a-body-of-text",
        ),
        pytest.param(
            ResponseFormat.WAV,
            _WAV_HEAD + _chunk(b"RIFF\0\0\0\0AVI ") + _chunk(b""),
            id="riff-that-is-not-wave",
        ),
        pytest.param(
            ResponseFormat.OPUS,
            _OGG_HEAD + _chunk(_WAV_BODY) + _chunk(b""),
            id="ogg-media-type-over-a-wav-body",
        ),
        pytest.param(
            ResponseFormat.OPUS,
            _OGG_HEAD + _chunk(b""),
            id="ogg-media-type-over-an-empty-body",
        ),
    ],
)
def test_a_200_that_is_not_the_ordered_format_is_malformed(
    ordered: ResponseFormat, raw: bytes, raw_listener: RawListener
) -> None:
    """WIRE-88, WIRE-133: the media type and the container signature must match."""
    with raw_listener(raw) as (base_url, _):
        outcome = _speak(base_url, response_format=ordered)

    assert outcome == Malformed()


def test_a_chunked_ogg_with_the_ogg_signature_is_spoken_when_opus_is_ordered(
    raw_listener: RawListener,
) -> None:
    """WIRE-133: an audio/ogg body that begins with OggS is the answer."""
    with raw_listener(_OGG_HEAD + _chunk(_OGG_BODY) + _chunk(b"")) as (
        base_url,
        _,
    ):
        outcome = _speak(base_url, response_format=ResponseFormat.OPUS)

    assert outcome == Spoken(_OGG_BODY)


@pytest.mark.parametrize("ordered", list(ResponseFormat))
def test_the_request_orders_the_format_it_was_given(
    ordered: ResponseFormat, raw_listener: RawListener
) -> None:
    with raw_listener(_WAV_HEAD + _chunk(_WAV_BODY) + _chunk(b"")) as (
        base_url,
        requests,
    ):
        _speak(base_url, response_format=ordered)

    (request,) = requests
    assert request.partition(b"\r\n\r\n")[2] == speech_request_body(
        "Hallo", VOICE, ordered
    )


_VOICE_AND_ITS_LANGUAGE = [(v, lang) for v in VoiceId for lang in VOICES[v].languages]
_VOICES_OF_SEVERAL_LANGUAGES = [v for v in VoiceId if len(VOICES[v].languages) > 1]


@pytest.mark.parametrize(("voice", "language"), _VOICE_AND_ITS_LANGUAGE)
def test_the_request_names_the_language_it_was_given(
    voice: VoiceId, language: Language, raw_listener: RawListener
) -> None:
    with raw_listener(_WAV_HEAD + _chunk(_WAV_BODY) + _chunk(b"")) as (
        base_url,
        requests,
    ):
        _speak(base_url, voice=voice, language=language)

    (request,) = requests
    assert request.partition(b"\r\n\r\n")[2] == speech_request_body(
        "Hallo", voice, ResponseFormat.WAV, language
    )


def test_the_request_names_the_instructions_it_was_given(
    raw_listener: RawListener,
) -> None:
    voice = next(v for v in VoiceId if VOICES[v].takes_instructions)
    language = VOICES[voice].languages[0]
    with raw_listener(_WAV_HEAD + _chunk(_WAV_BODY) + _chunk(b"")) as (
        base_url,
        requests,
    ):
        _speak(base_url, voice=voice, language=language, instructions="Speak angrily")

    (request,) = requests
    assert request.partition(b"\r\n\r\n")[2] == speech_request_body(
        "Hallo", voice, ResponseFormat.WAV, language, "Speak angrily"
    )


@pytest.mark.parametrize("voice", _VOICES_OF_SEVERAL_LANGUAGES)
def test_a_voice_of_several_languages_without_a_language_fails_before_the_network(
    voice: VoiceId, raw_listener: RawListener
) -> None:
    with (
        raw_listener(_WAV_HEAD + _chunk(_WAV_BODY) + _chunk(b"")) as (
            base_url,
            requests,
        ),
        pytest.raises(ValidationError) as refused,
    ):
        _speak(base_url, voice=voice)

    assert [error["loc"] for error in refused.value.errors()] == [("language",)]
    assert requests == []


def test_the_request_is_the_wires_body_and_headers(raw_listener: RawListener) -> None:
    with raw_listener(_WAV_HEAD + _chunk(_WAV_BODY) + _chunk(b"")) as (
        base_url,
        requests,
    ):
        outcome = _speak(base_url)

    assert outcome == Spoken(_WAV_BODY)
    (request,) = requests
    head, _, body = request.partition(b"\r\n\r\n")
    lines = head.decode().split("\r\n")
    assert lines[0] == f"POST {SPEECH_PATH} HTTP/1.1"
    assert body == speech_request_body("Hallo", VOICE, ResponseFormat.WAV)
    sent = {line.partition(": ")[0]: line.partition(": ")[2] for line in lines[1:]}
    expected = speech_request_headers(
        workload_token=WORKLOAD_TOKEN, person_token=PERSON_TOKEN
    )
    assert all(sent[name] == value for name, value in expected.items())


def test_a_429_with_a_whole_second_wait_is_busy(raw_listener: RawListener) -> None:
    raw = _refusal_response(
        "429 Too Many Requests", RefusalCode.BUSY, f"{RETRY_AFTER_HEADER}: 7"
    )

    with raw_listener(raw) as (base_url, _):
        outcome = _speak(base_url)

    assert isinstance(outcome, Busy)
    assert outcome.wait.retry_after_seconds == 7  # noqa: PLR2004 - the header's value


# The streaming call.

SAMPLE_RATE = 22_050
SECONDS_PER_PCM_BYTE = 1 / (2 * SAMPLE_RATE)


def _wav(pcm: bytes, *, with_data: bool = True) -> bytes:
    """A WAV head with an INFO chunk before `data`, as the service sends it."""
    info = struct.pack("<4sI4s", b"LIST", 4 + 8 + 4, b"INFO") + struct.pack(
        "<4sI", b"ICMT", 4
    )
    info += b"AI\0\0"
    fmt = struct.pack(
        "<4sIHHIIHH", b"fmt ", 16, 1, 1, SAMPLE_RATE, SAMPLE_RATE * 2, 2, 16
    )
    data = struct.pack("<4sI", b"data", 0xFFFFFFFF) + pcm if with_data else b""
    return b"RIFF\xff\xff\xff\xffWAVE" + fmt + info + data


def _streamed(raw: bytes, raw_listener: RawListener) -> list[SpeakStreamItem]:
    with raw_listener(raw) as (base_url, _):
        return list(
            speak_stream(
                base_url,
                "Hallo",
                VOICE,
                workload_token=WORKLOAD_TOKEN,
                person_token=PERSON_TOKEN,
                deadline_seconds=GENEROUS_DEADLINE_SECONDS,
                cap_bytes=CAP_BYTES,
            )
        )


@pytest.mark.parametrize("piece_bytes", [1, 7, 1000])
def test_the_stream_ends_with_the_exact_seconds_whatever_the_chunking(
    piece_bytes: int, raw_listener: RawListener
) -> None:
    """The length is counted from the PCM bytes, however the head is split."""
    pcm = b"\x01\x00" * 4410
    wav = _wav(pcm)
    chunks = b"".join(
        _chunk(wav[at : at + piece_bytes]) for at in range(0, len(wav), piece_bytes)
    )

    *pieces, end = _streamed(_WAV_HEAD + chunks + _chunk(b""), raw_listener)

    assert b"".join(pieces) == wav
    assert end == Finished(len(pcm) * SECONDS_PER_PCM_BYTE)


def test_a_stream_without_audio_data_ends_malformed(raw_listener: RawListener) -> None:
    raw = _WAV_HEAD + _chunk(_wav(b"", with_data=False)) + _chunk(b"")

    assert _streamed(raw, raw_listener)[-1] == Malformed()


def test_a_stream_cut_off_after_audio_ends_cut_not_finished(
    raw_listener: RawListener,
) -> None:
    """WIRE-90: the chunks that came stay, and the end is the failure."""
    wav = _wav(b"\x01\x00" * 100)

    items = _streamed(_WAV_HEAD + _chunk(wav), raw_listener)

    assert items == [wav, Cut()]

"""The wire between an app and the speech service.

This module is the one owner of what travels on `POST /v1/audio/speech` and
`POST /v1/audio/transcriptions`: the paths, the requests, the transcription
answer, and the closed list of refusals with their statuses, messages and the
body they are answered in (OpenAI's error form). The service and its callers
import these types and copy none of them.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from http import HTTPStatus
from typing import TYPE_CHECKING, Final, Literal

from pydantic import BaseModel, ConfigDict, field_validator

if TYPE_CHECKING:
    from collections.abc import Mapping

SPEECH_PATH: Final = "/v1/audio/speech"
TRANSCRIPTIONS_PATH: Final = "/v1/audio/transcriptions"
API_VERSION: Final = 1
MODEL: Final = "speech-runtime"
RESPONSE_FORMAT: Final = "wav"
RETRY_AFTER_HEADER: Final = "Retry-After"
CONTENT_TYPE_HEADER: Final = "Content-Type"
WORKLOAD_TOKEN_HEADER: Final = "Speech-Workload-Token"  # noqa: S105 - a header name and no secret
AUTHORIZATION_HEADER: Final = "Authorization"
AUTHENTICATE_HEADER: Final = "WWW-Authenticate"

RequestField = Literal["file", "model", "input", "voice", "language", "response_format"]
_INPUT_WITHOUT_CONTENT: Final = "input must hold a letter or a digit"


class SpeechRequest(BaseModel):
    """The body of `POST /v1/audio/speech`: a strict subset of OpenAI's."""

    model_config = ConfigDict(
        extra="forbid", frozen=True, hide_input_in_errors=True, strict=True
    )

    model: Literal["speech-runtime"]
    input: str
    voice: str
    response_format: Literal["wav"]

    @field_validator("input")
    @classmethod
    def _input_has_content(cls, text: str) -> str:
        if not any(character.isalnum() for character in text):
            raise ValueError(_INPUT_WITHOUT_CONTENT)
        return text

    @field_validator("voice")
    @classmethod
    def _voice_is_named(cls, voice: str) -> str:
        if not voice:
            message = "voice must not be empty"
            raise ValueError(message)
        return voice


class HearingLanguage(StrEnum):
    """The languages the service hears."""

    DE = "de"
    EN = "en"


class TranscriptionRequest(BaseModel):
    """The form fields of `POST /v1/audio/transcriptions` beside its `file`."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    model: Literal["speech-runtime"]
    language: HearingLanguage = HearingLanguage.DE
    response_format: Literal["json"] = "json"


class Transcription(BaseModel):
    """The answer of `POST /v1/audio/transcriptions`."""

    model_config = ConfigDict(frozen=True)

    text: str


def speech_request_body(text: str, voice: str) -> bytes:
    """The JSON a caller posts to speak `text` with the voice `voice`."""
    request = SpeechRequest(
        model=MODEL, input=text, voice=voice, response_format=RESPONSE_FORMAT
    )
    return request.model_dump_json().encode()


def speech_request_headers(*, workload_token: str, person_token: str) -> dict[str, str]:
    """The headers a caller posts with the body: the app's token and the person's."""
    return {
        CONTENT_TYPE_HEADER: "application/json",
        WORKLOAD_TOKEN_HEADER: workload_token,
        AUTHORIZATION_HEADER: f"Bearer {person_token}",
    }


class RefusalCode(StrEnum):
    """Why the service refused a request."""

    UNSUPPORTED_VERSION = "unsupported_version"
    NOT_FOUND = "not_found"
    UNAUTHORIZED = "unauthorized"
    FORBIDDEN = "forbidden"
    NOT_ENABLED = "not_enabled"
    INVALID_REQUEST = "invalid_request"
    TOO_LARGE = "too_large"
    BUSY = "busy"
    UNAVAILABLE = "unavailable"


REFUSAL_STATUS: Final[dict[RefusalCode, HTTPStatus]] = {
    RefusalCode.UNSUPPORTED_VERSION: HTTPStatus.BAD_REQUEST,
    RefusalCode.NOT_FOUND: HTTPStatus.NOT_FOUND,
    RefusalCode.UNAUTHORIZED: HTTPStatus.UNAUTHORIZED,
    RefusalCode.FORBIDDEN: HTTPStatus.FORBIDDEN,
    RefusalCode.NOT_ENABLED: HTTPStatus.FORBIDDEN,
    RefusalCode.INVALID_REQUEST: HTTPStatus.UNPROCESSABLE_ENTITY,
    RefusalCode.TOO_LARGE: HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
    RefusalCode.BUSY: HTTPStatus.TOO_MANY_REQUESTS,
    RefusalCode.UNAVAILABLE: HTTPStatus.SERVICE_UNAVAILABLE,
}

_REFUSAL_MESSAGE: Final[dict[RefusalCode, str]] = {
    RefusalCode.UNSUPPORTED_VERSION: "This API version is not supported.",
    RefusalCode.NOT_FOUND: "There is nothing at this path.",
    RefusalCode.UNAUTHORIZED: (
        "A workload token and the person's token are required, and both must be valid."
    ),
    RefusalCode.FORBIDDEN: "The tokens do not permit this request.",
    RefusalCode.NOT_ENABLED: "This app is not permitted to use this feature.",
    RefusalCode.INVALID_REQUEST: "The request is not valid.",
    RefusalCode.TOO_LARGE: "The request body is too large.",
    RefusalCode.BUSY: "The service is busy; try again.",
    RefusalCode.UNAVAILABLE: "The service cannot serve this request right now.",
}


class RefusalDetail(BaseModel):
    """The `error` object of a refusal."""

    model_config = ConfigDict(frozen=True)

    message: str
    type: Literal["invalid_request_error", "server_error"]
    param: RequestField | None
    code: RefusalCode


class Refusal(BaseModel):
    """The JSON body of every refusal under version 1."""

    model_config = ConfigDict(frozen=True)

    error: RefusalDetail

    @property
    def status(self) -> HTTPStatus:
        """The HTTP status this refusal is answered with."""
        return REFUSAL_STATUS[self.error.code]


def refusal(code: RefusalCode, param: RequestField | None = None) -> Refusal:
    """The refusal for `code`, naming the request field at fault, if any."""
    status = REFUSAL_STATUS[code]
    kind = (
        "server_error"
        if status >= HTTPStatus.INTERNAL_SERVER_ERROR
        else "invalid_request_error"
    )
    return Refusal(
        error=RefusalDetail(
            message=_REFUSAL_MESSAGE[code], type=kind, param=param, code=code
        )
    )


class RetryAfterError(ValueError):
    """A `busy` answer whose `Retry-After` is not a whole number of seconds."""


@dataclass(frozen=True)
class BusyWait:
    """How long the service asks the app to wait after a `busy` refusal."""

    retry_after_seconds: int


def busy_wait(status: int, headers: Mapping[str, str]) -> BusyWait | None:
    """The wait a `busy` answer asks for, or `None` for any other status.

    Raises `RetryAfterError` when a `busy` answer lacks a `Retry-After` of whole
    seconds, at least 1: the service never sends a date.
    """
    if status != REFUSAL_STATUS[RefusalCode.BUSY]:
        return None
    wanted = RETRY_AFTER_HEADER.lower()
    value = next((v for name, v in headers.items() if name.lower() == wanted), None)
    if value is None or not (value.isascii() and value.isdigit()) or int(value) < 1:
        message = (
            f"{RETRY_AFTER_HEADER} must be whole seconds, at least 1, got {value!r}"
        )
        raise RetryAfterError(message)
    return BusyWait(retry_after_seconds=int(value))

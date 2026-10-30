"""The wire between an app and the speech service.

This module is the one owner of what travels on `POST /v1/audio/speech`: the
path, the request, and the closed list of refusals with their statuses,
messages and the body they are answered in (OpenAI's error form). The service
and its callers import these types and copy none of them.
"""

from __future__ import annotations

from enum import StrEnum
from http import HTTPStatus
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, field_validator

SPEECH_PATH: Final = "/v1/audio/speech"
API_VERSION: Final = 1
MODEL: Final = "speech-runtime"
RESPONSE_FORMAT: Final = "wav"
RETRY_AFTER_HEADER: Final = "Retry-After"
CONTENT_TYPE_HEADER: Final = "Content-Type"
WORKLOAD_TOKEN_HEADER: Final = "Speech-Workload-Token"  # noqa: S105 - a header name and no secret
AUTHORIZATION_HEADER: Final = "Authorization"

RequestField = Literal["model", "input", "voice", "response_format"]
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
    INVALID_REQUEST = "invalid_request"
    BUSY = "busy"
    UNAVAILABLE = "unavailable"


REFUSAL_STATUS: Final[dict[RefusalCode, HTTPStatus]] = {
    RefusalCode.UNSUPPORTED_VERSION: HTTPStatus.BAD_REQUEST,
    RefusalCode.NOT_FOUND: HTTPStatus.NOT_FOUND,
    RefusalCode.INVALID_REQUEST: HTTPStatus.UNPROCESSABLE_ENTITY,
    RefusalCode.BUSY: HTTPStatus.TOO_MANY_REQUESTS,
    RefusalCode.UNAVAILABLE: HTTPStatus.SERVICE_UNAVAILABLE,
}

_REFUSAL_MESSAGE: Final[dict[RefusalCode, str]] = {
    RefusalCode.UNSUPPORTED_VERSION: "This API version is not supported.",
    RefusalCode.NOT_FOUND: "There is nothing at this path.",
    RefusalCode.INVALID_REQUEST: "The request is not valid.",
    RefusalCode.BUSY: "The service is busy speaking; try again.",
    RefusalCode.UNAVAILABLE: "The voice is not available.",
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

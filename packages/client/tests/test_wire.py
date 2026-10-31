"""The wire: a `busy` answer's wait, the hearing request, and its refusals."""

from http import HTTPStatus

import pytest
from pydantic import ValidationError

from speech_runtime_client.wire import (
    MODEL,
    REFUSAL_STATUS,
    RETRY_AFTER_HEADER,
    BusyWait,
    HearingLanguage,
    RefusalCode,
    RetryAfterError,
    TranscriptionRequest,
    busy_wait,
    refusal,
)

BUSY = REFUSAL_STATUS[RefusalCode.BUSY]


def test_busy_answer_gives_the_wait_in_seconds() -> None:
    assert busy_wait(BUSY, {RETRY_AFTER_HEADER: "5"}) == BusyWait(retry_after_seconds=5)


def test_header_name_is_matched_case_insensitively() -> None:
    assert busy_wait(int(BUSY), {RETRY_AFTER_HEADER.lower(): "1"}) == BusyWait(1)


@pytest.mark.parametrize(
    "status",
    [s for s in (*REFUSAL_STATUS.values(), HTTPStatus.OK) if s != BUSY],
)
def test_other_statuses_give_no_wait(status: HTTPStatus) -> None:
    assert busy_wait(status, {RETRY_AFTER_HEADER: "5"}) is None


@pytest.mark.parametrize(
    "headers",
    [
        {},
        {RETRY_AFTER_HEADER: "-1"},
        {RETRY_AFTER_HEADER: "1.5"},
        {RETRY_AFTER_HEADER: "Wed, 21 Oct 2026 07:28:00 GMT"},
        {RETRY_AFTER_HEADER: ""},
        {RETRY_AFTER_HEADER: "0"},
    ],
    ids=["missing", "negative", "fraction", "http-date", "empty", "zero"],
)
def test_busy_answer_without_whole_seconds_fails_loud(headers: dict[str, str]) -> None:
    with pytest.raises(RetryAfterError, match=RETRY_AFTER_HEADER):
        busy_wait(BUSY, headers)


def test_app_without_the_hear_permission_is_refused_as_not_enabled() -> None:
    body = refusal(RefusalCode.NOT_ENABLED)

    assert body.status == HTTPStatus.FORBIDDEN
    assert body.error.code == "not_enabled"
    assert body.error.type == "invalid_request_error"


@pytest.mark.parametrize("field", ["file", "model", "language", "response_format"])
def test_refusal_names_a_hearing_request_field(field: str) -> None:
    assert refusal(RefusalCode.INVALID_REQUEST, field).error.param == field


@pytest.mark.parametrize("language", list(HearingLanguage))
def test_hearing_accepts_each_language_of_the_wire(language: HearingLanguage) -> None:
    request = TranscriptionRequest.model_validate(
        {"model": MODEL, "language": language}
    )

    assert request.language is language


@pytest.mark.parametrize(
    ("fields", "named"),
    [
        ({"language": "fr"}, "language"),
        ({"response_format": "text"}, "response_format"),
        ({"model": "whisper-1"}, "model"),
        ({"temperature": "0"}, "temperature"),
    ],
)
def test_transcription_fields_outside_the_wire_are_refused(
    fields: dict[str, str], named: str
) -> None:
    with pytest.raises(ValidationError, match=named):
        TranscriptionRequest.model_validate({"model": MODEL} | fields)

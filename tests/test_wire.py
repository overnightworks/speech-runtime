"""The client reads a `busy` answer's wait from the wire."""

from http import HTTPStatus

import pytest

from speech_runtime_client.wire import (
    REFUSAL_STATUS,
    RETRY_AFTER_HEADER,
    BusyWait,
    RefusalCode,
    RetryAfterError,
    busy_wait,
)

BUSY = REFUSAL_STATUS[RefusalCode.BUSY]


def test_busy_answer_gives_the_wait_in_seconds() -> None:
    assert busy_wait(BUSY, {RETRY_AFTER_HEADER: "5"}) == BusyWait(retry_after_seconds=5)


def test_header_name_is_matched_case_insensitively() -> None:
    assert busy_wait(int(BUSY), {RETRY_AFTER_HEADER.lower(): "0"}) == BusyWait(0)


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
    ],
    ids=["missing", "negative", "fraction", "http-date", "empty"],
)
def test_busy_answer_without_whole_seconds_fails_loud(headers: dict[str, str]) -> None:
    with pytest.raises(RetryAfterError, match=RETRY_AFTER_HEADER):
        busy_wait(BUSY, headers)

"""The speak wire admits only a person call; every other call is refused."""

from __future__ import annotations

import asyncio
import logging
import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pytest
from fastapi.testclient import TestClient
from starlette.status import (
    HTTP_200_OK,
    HTTP_401_UNAUTHORIZED,
    HTTP_403_FORBIDDEN,
    HTTP_422_UNPROCESSABLE_CONTENT,
    HTTP_429_TOO_MANY_REQUESTS,
    HTTP_503_SERVICE_UNAVAILABLE,
)
from webauth.jwt_verification import KEY_SET_REFRESH_COOLDOWN_SECONDS

from speech_runtime_client.wire import (
    AUTHENTICATE_HEADER,
    AUTHORIZATION_HEADER,
    WORKLOAD_TOKEN_HEADER,
    RefusalCode,
    speech_request_headers,
)
from speech_runtime_service.config import Settings
from speech_runtime_service.control import create_control_app
from tests.support.fakes_support import GatedSpeaking
from tests.support.identity_support import (
    ABSENT,
    DEFAULT_KEY,
    MINTING,
    Clocks,
    FakeKeySet,
    altered,
    callers_over,
    signing_key,
    valid_pair,
)
from tests.support.speak_support import (
    SpeakCall,
    app_over,
    assert_call_refusal,
    assert_refusal,
    fake_runtime,
    info_comment_before_data,
    post_speech,
    speak_app,
    speak_body,
    speak_scope,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from httpx2 import Response

    from tests.support.identity_support import SigningKey

UNAUTHORIZED = (HTTP_401_UNAUTHORIZED, RefusalCode.UNAUTHORIZED)
FORBIDDEN = (HTTP_403_FORBIDDEN, RefusalCode.FORBIDDEN)
UNAVAILABLE = (HTTP_503_SERVICE_UNAVAILABLE, RefusalCode.UNAVAILABLE)
OTHER_KEY = signing_key("key-two")
ORDINARY_LIFETIME_SECONDS = 300
PAST_SECONDS = 600


@dataclass
class Issuer:
    """One scenario: an issuer's key set and clocks, and the wire that trusts them."""

    key_set: FakeKeySet = field(default_factory=lambda: FakeKeySet(DEFAULT_KEY))
    clocks: Clocks = field(default_factory=Clocks)

    def __post_init__(self) -> None:
        """Build the wire over the scenario's key set and clocks."""
        self.client = TestClient(
            speak_app(callers=callers_over(self.key_set, self.clocks))
        )

    def tokens(self) -> tuple[str, str]:
        """A valid workload and person token at the scenario's now."""
        pair = valid_pair(self.clocks.wall)
        return pair.workload, pair.person

    def speak(self, workload: str | None, person: str | None) -> Response:
        """POST a valid body with the given tokens; `None` sends no header."""
        headers: dict[str, str] = {}
        if workload is not None:
            headers[WORKLOAD_TOKEN_HEADER] = workload
        if person is not None:
            headers[AUTHORIZATION_HEADER] = f"Bearer {person}"
        return post_speech(self.client, speak_body(), headers)

    def speak_valid(self) -> Response:
        """POST a valid body with a valid pair."""
        return self.speak(*self.tokens())


def _answers(response: Response, expected: tuple[int, RefusalCode]) -> None:
    status, code = expected
    assert_refusal(response, status, code, None)
    challenge = response.headers.get(AUTHENTICATE_HEADER)
    assert challenge == ("Bearer" if status == HTTP_401_UNAUTHORIZED else None)


def test_a_valid_pair_is_served_a_wav_with_its_comment() -> None:
    response = Issuer().speak_valid()

    assert response.status_code == HTTP_200_OK
    assert info_comment_before_data(response.content) is not None


def test_the_bearer_scheme_is_case_insensitive() -> None:
    issuer = Issuer()
    workload, person = issuer.tokens()

    response = post_speech(
        issuer.client,
        speak_body(),
        {WORKLOAD_TOKEN_HEADER: workload, AUTHORIZATION_HEADER: f"bEaReR {person}"},
    )

    assert response.status_code == HTTP_200_OK


def test_basic_authorization_counts_as_no_person_token() -> None:
    issuer = Issuer()
    workload, person = issuer.tokens()

    response = post_speech(
        issuer.client,
        speak_body(),
        {WORKLOAD_TOKEN_HEADER: workload, AUTHORIZATION_HEADER: f"Basic {person}"},
    )

    _answers(response, UNAUTHORIZED)


@pytest.mark.parametrize(
    ("sent", "expected"),
    [
        ("workload only", UNAUTHORIZED),
        ("person only", UNAUTHORIZED),
        ("neither", UNAUTHORIZED),
    ],
)
def test_a_missing_token_is_unauthorized(
    sent: str, expected: tuple[int, RefusalCode]
) -> None:
    issuer = Issuer()
    workload, person = issuer.tokens()
    tokens = {
        "workload only": (workload, None),
        "person only": (None, person),
        "neither": (None, None),
    }[sent]

    _answers(issuer.speak(*tokens), expected)


def _expired(token: str, clocks: Clocks) -> str:
    now = int(clocks.wall())
    return altered(token, iat=now - 2 * PAST_SECONDS, exp=now - PAST_SECONDS)


def _lives_too_long(token: str, clocks: Clocks) -> str:
    now = int(clocks.wall())
    return altered(token, iat=now, exp=now + ORDINARY_LIFETIME_SECONDS + 1)


def _without_role(token: str, _clocks: Clocks) -> str:
    return altered(token, realm_access={"roles": []})


def _unregistered(token: str, _clocks: Clocks) -> str:
    return altered(token, azp="stranger-client", client_id="stranger-client")


def _remove(claim: str) -> Callable[[str, Clocks], str]:
    return lambda token, _clocks: altered(token, **{claim: ABSENT})


def _set(**claims: object) -> Callable[[str, Clocks], str]:
    return lambda token, _clocks: altered(token, **claims)


def _signed_by(key: SigningKey, key_identifier: str) -> Callable[[str, Clocks], str]:
    return lambda token, _clocks: altered(
        token, signed_by=key, key_identifier=key_identifier
    )


@pytest.mark.parametrize(
    "damage",
    [
        _expired,
        _set(aud="another-service"),
        _set(iss="https://elsewhere.test/realm"),
        _signed_by(OTHER_KEY, DEFAULT_KEY.key_identifier),
        _signed_by(DEFAULT_KEY, "unknown-key"),
        _remove("sub"),
        _set(sub=7),
    ],
    ids=[
        "expired",
        "audience",
        "issuer",
        "signature",
        "unknown key",
        "subject missing",
        "subject not a string",
    ],
)
@pytest.mark.parametrize("damaged", ["workload", "person"])
def test_a_token_that_cannot_be_verified_is_unauthorized(
    damaged: str, damage: Callable[[str, Clocks], str]
) -> None:
    issuer = Issuer()
    workload, person = issuer.tokens()
    if damaged == "workload":
        workload = damage(workload, issuer.clocks)
    else:
        person = damage(person, issuer.clocks)

    _answers(issuer.speak(workload, person), UNAUTHORIZED)


def test_a_person_group_claim_of_null_is_unauthorized() -> None:
    issuer = Issuer()
    workload, person = issuer.tokens()

    response = issuer.speak(workload, altered(person, groups=None))

    _answers(response, UNAUTHORIZED)


@pytest.mark.parametrize(
    ("damaged", "damage"),
    [
        ("person", _set(azp="elsewhere-client")),
        ("person", _remove("azp")),
        ("person", _set(groups=["/apps/other", "/services/speech-service"])),
        ("person", _set(groups=["/apps/quiz"])),
        ("person", _set(groups=["/apps/quiz", "/services/speech-service/sub"])),
        ("person", _set(groups=["/apps/quiz", "/services/speech-service2"])),
        ("person", _set(sub=MINTING.workload_subject)),
        ("workload", _without_role),
        ("workload", _set(azp="elsewhere-client")),
        ("workload", _lives_too_long),
        ("workload", _unregistered),
    ],
    ids=[
        "person azp differs",
        "person azp missing",
        "person outside the app group",
        "person outside the service group",
        "person only in a subgroup of the service group",
        "person in a group the service group is a prefix of",
        "same subject",
        "workload without the role",
        "workload azp differs from client id",
        "workload lives too long",
        "app not registered",
    ],
)
def test_tokens_that_do_not_permit_the_request_are_forbidden(
    damaged: str, damage: Callable[[str, Clocks], str]
) -> None:
    issuer = Issuer()
    workload, person = issuer.tokens()
    if damaged == "workload":
        workload = damage(workload, issuer.clocks)
    else:
        person = damage(person, issuer.clocks)

    _answers(issuer.speak(workload, person), FORBIDDEN)


def test_a_person_token_sent_as_the_workload_token_is_forbidden() -> None:
    issuer = Issuer()
    _workload, person = issuer.tokens()

    _answers(issuer.speak(person, person), FORBIDDEN)


def test_a_person_in_a_subgroup_of_the_app_group_is_served() -> None:
    issuer = Issuer()
    workload, person = issuer.tokens()

    response = issuer.speak(
        workload,
        altered(person, groups=["/apps/quiz/players", MINTING.service_group]),
    )

    assert response.status_code == HTTP_200_OK


def test_a_missing_person_token_is_unauthorized_though_the_app_is_unregistered() -> (
    None
):
    issuer = Issuer()
    workload, _person = issuer.tokens()

    response = issuer.speak(_unregistered(workload, issuer.clocks), None)

    _answers(response, UNAUTHORIZED)


def test_the_key_set_being_down_is_unavailable_before_an_unregistered_app() -> None:
    issuer = Issuer()
    issuer.key_set.down = True
    workload, person = issuer.tokens()

    response = issuer.speak(_unregistered(workload, issuer.clocks), person)

    _answers(response, UNAVAILABLE)


def test_a_cached_key_set_serves_while_the_issuer_is_down() -> None:
    issuer = Issuer()
    assert issuer.speak_valid().status_code == HTTP_200_OK
    issuer.key_set.down = True

    assert issuer.speak_valid().status_code == HTTP_200_OK


def test_an_expired_cached_key_set_is_unavailable_while_the_issuer_is_down() -> None:
    issuer = Issuer()
    assert issuer.speak_valid().status_code == HTTP_200_OK
    issuer.key_set.down = True
    issuer.clocks.advance(2 * 60 * 60)

    _answers(issuer.speak_valid(), UNAVAILABLE)


def _rotated(*, cooldown_passed: bool) -> Response:
    issuer = Issuer()
    assert issuer.speak_valid().status_code == HTTP_200_OK
    issuer.key_set.published.append(OTHER_KEY)
    if cooldown_passed:
        issuer.clocks.advance(KEY_SET_REFRESH_COOLDOWN_SECONDS + 1)
    workload, person = issuer.tokens()
    return issuer.speak(
        altered(workload, signed_by=OTHER_KEY), altered(person, signed_by=OTHER_KEY)
    )


def test_a_rotated_key_is_accepted_once_the_refresh_cooldown_has_passed() -> None:
    assert _rotated(cooldown_passed=True).status_code == HTTP_200_OK


def test_a_rotated_key_is_refused_within_the_refresh_cooldown() -> None:
    _answers(_rotated(cooldown_passed=False), UNAUTHORIZED)


def test_identity_is_judged_before_the_body() -> None:
    issuer = Issuer()

    response = post_speech(issuer.client, b"{}", {})

    _answers(response, UNAUTHORIZED)


def test_a_refused_identity_never_reads_the_body() -> None:
    app = speak_app()
    received = False
    sent: list[dict[str, object]] = []

    async def receive() -> dict[str, object]:
        nonlocal received
        received = True
        return {"type": "http.request", "body": b"{}", "more_body": False}

    async def send(message: dict[str, object]) -> None:
        sent.append(message)

    asyncio.run(app(speak_scope({}), receive, send))

    assert sent[0]["status"] == HTTP_401_UNAUTHORIZED
    assert not received


def test_a_valid_pair_with_an_invalid_body_is_unprocessable() -> None:
    issuer = Issuer()
    workload, person = issuer.tokens()
    response = post_speech(
        issuer.client,
        b"{}",
        speech_request_headers(workload_token=workload, person_token=person),
    )

    assert response.status_code == HTTP_422_UNPROCESSABLE_CONTENT


def test_the_busy_wait_starts_after_a_slow_identity_check() -> None:
    held_seconds = 0.4
    busy_wait_seconds = 0.2

    async def scenario() -> SpeakCall:
        voice = GatedSpeaking()
        runtime = fake_runtime(voice)
        holder = SpeakCall(app_over(runtime, busy_wait_seconds=busy_wait_seconds))
        await holder.status()
        await asyncio.to_thread(voice.entered_first_step.wait, 5)
        key_set = FakeKeySet(DEFAULT_KEY)
        key_set.hold = threading.Event()
        slow = create_control_app(
            Settings(busy_wait_seconds=busy_wait_seconds),
            runtime,
            callers_over(key_set),
        )
        waiting = SpeakCall(slow)
        try:
            await asyncio.to_thread(key_set.held.wait, 5)
            await asyncio.sleep(held_seconds)
            key_set.hold.set()
            await waiting.status()
        finally:
            key_set.hold.set()
            voice.release_all()
        await waiting.finish()
        await holder.finish()
        return waiting

    waiting = asyncio.run(scenario())

    assert_call_refusal(waiting, HTTP_429_TOO_MANY_REQUESTS, RefusalCode.BUSY, None)
    assert waiting.started_after >= held_seconds + busy_wait_seconds - 0.05


def test_no_log_record_holds_a_token_a_subject_or_the_input(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG)
    issuer = Issuer()
    workload, person = issuer.tokens()
    secrets = [
        workload,
        person,
        MINTING.person_subject,
        MINTING.workload_subject,
        "SENTINEL-KEY-SET-CREDENTIAL",
    ]
    body = speak_body(text="Zebrakuchen")
    headers = speech_request_headers(workload_token=workload, person_token=person)
    post_speech(issuer.client, body, headers)
    post_speech(issuer.client, body, {})
    post_speech(issuer.client, body, headers | {WORKLOAD_TOKEN_HEADER: person})
    issuer.key_set.down = True
    issuer.clocks.advance(2 * 60 * 60)
    issuer.speak_valid()

    text = "\n".join(
        f"{record.getMessage()} {record.exc_text or ''}" for record in caplog.records
    )

    for secret in [*secrets, "eyJ", "Bearer", "Zebrakuchen"]:
        assert secret not in text


@pytest.mark.parametrize(
    "expected",
    [UNAUTHORIZED, FORBIDDEN, UNAVAILABLE],
    ids=["unauthorized", "forbidden", "unavailable"],
)
def test_every_refusal_is_in_openai_form_with_a_fixed_message(
    expected: tuple[int, RefusalCode],
) -> None:
    issuer = Issuer()
    workload, person = issuer.tokens()
    if expected == FORBIDDEN:
        response = issuer.speak(person, person)
    elif expected == UNAVAILABLE:
        issuer.key_set.down = True
        response = issuer.speak(workload, person)
    else:
        response = issuer.speak(None, None)

    _answers(response, expected)

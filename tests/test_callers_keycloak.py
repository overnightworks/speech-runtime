"""The caller check against tokens a real Keycloak issues.

The `keycloak-token-shapes` CI job serves `tests/fixtures/keycloak/realm.json`
and runs these tests with `-m keycloak`; the default run deselects them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from fastapi.testclient import TestClient
from starlette.status import (
    HTTP_200_OK,
    HTTP_400_BAD_REQUEST,
    HTTP_401_UNAUTHORIZED,
    HTTP_403_FORBIDDEN,
)

from speech_runtime_client.wire import (
    WORKLOAD_TOKEN_HEADER,
    RefusalCode,
    speech_request_headers,
)
from tests.support.identity_support import MINTING
from tests.support.keycloak_support import (
    APP_CLIENT,
    AUDIENCE_SCOPE,
    OTHER_CLIENT,
    ROLELESS_CLIENT,
    UNSCOPED_CLIENT,
    WORKLOAD_ROLE,
    audiences_of,
    claims_of,
    keycloak_callers,
    person_token,
    token_response,
    workload_token,
)
from tests.support.speak_support import (
    assert_refusal,
    info_comment_before_data,
    post_speech,
    speak_app,
    speak_body,
)

if TYPE_CHECKING:
    from httpx2 import Response as ClientResponse

pytestmark = pytest.mark.keycloak

SERVICE_GROUP = MINTING.service_group
APP_GROUP = MINTING.app_group


def test_a_person_token_with_the_scope_carries_the_audience_and_groups() -> None:
    claims = claims_of(person_token("ann"))

    assert MINTING.audience in audiences_of(claims)
    assert {SERVICE_GROUP, APP_GROUP} <= set(claims["groups"])


def test_a_workload_token_with_the_scope_carries_the_audience_and_the_role() -> None:
    claims = claims_of(workload_token(APP_CLIENT))

    assert MINTING.audience in audiences_of(claims)
    assert WORKLOAD_ROLE in claims["realm_access"]["roles"]


def test_a_workload_token_asked_without_the_scope_carries_no_audience() -> None:
    claims = claims_of(workload_token(APP_CLIENT, with_audience=False))

    assert MINTING.audience not in audiences_of(claims)


def test_a_client_without_the_scope_is_refused_when_it_asks_for_it() -> None:
    response = token_response(
        {
            "grant_type": "client_credentials",
            "client_id": UNSCOPED_CLIENT,
            "client_secret": f"{UNSCOPED_CLIENT}-secret",
            "scope": AUDIENCE_SCOPE,
        }
    )

    assert response.status_code == HTTP_400_BAD_REQUEST
    assert response.json()["error"] == "invalid_scope"


@pytest.fixture(scope="module")
def wire() -> TestClient:
    return TestClient(speak_app(callers=keycloak_callers()))


def speak(wire: TestClient, workload: str, person: str | None) -> ClientResponse:
    if person is None:
        headers = {WORKLOAD_TOKEN_HEADER: workload}
    else:
        headers = speech_request_headers(workload_token=workload, person_token=person)
    return post_speech(wire, speak_body(), headers)


def test_the_app_speaks_for_a_real_pair(wire: TestClient) -> None:
    response = speak(wire, workload_token(APP_CLIENT), person_token("ann"))

    assert response.status_code == HTTP_200_OK
    assert info_comment_before_data(response.content) is not None


def test_a_person_in_a_subgroup_of_the_app_group_is_served(wire: TestClient) -> None:
    response = speak(wire, workload_token(APP_CLIENT), person_token("dot"))

    assert response.status_code == HTTP_200_OK


def test_the_real_workload_token_alone_is_unauthorized(wire: TestClient) -> None:
    response = speak(wire, workload_token(APP_CLIENT), None)

    assert_refusal(response, HTTP_401_UNAUTHORIZED, RefusalCode.UNAUTHORIZED, None)


def test_a_workload_asked_without_the_scope_is_unauthorized_beside_a_valid_person(
    wire: TestClient,
) -> None:
    workload = workload_token(APP_CLIENT, with_audience=False)

    response = speak(wire, workload, person_token("ann"))

    assert_refusal(response, HTTP_401_UNAUTHORIZED, RefusalCode.UNAUTHORIZED, None)


def test_a_workload_whose_service_account_lacks_the_role_is_forbidden(
    wire: TestClient,
) -> None:
    response = speak(wire, workload_token(ROLELESS_CLIENT), person_token("ann"))

    assert_refusal(response, HTTP_403_FORBIDDEN, RefusalCode.FORBIDDEN, None)


def test_a_person_token_issued_to_another_client_is_forbidden(wire: TestClient) -> None:
    response = speak(
        wire, workload_token(APP_CLIENT), person_token("ann", client=OTHER_CLIENT)
    )

    assert_refusal(response, HTTP_403_FORBIDDEN, RefusalCode.FORBIDDEN, None)


@pytest.mark.parametrize(
    "user",
    [
        pytest.param("cy", id="outside the app group"),
        pytest.param("ben", id="outside the service group"),
        pytest.param("eli", id="only in a subgroup of the service group"),
    ],
)
def test_a_person_missing_the_app_or_the_service_group_is_forbidden(
    wire: TestClient, user: str
) -> None:
    response = speak(wire, workload_token(APP_CLIENT), person_token(user))

    assert_refusal(response, HTTP_403_FORBIDDEN, RefusalCode.FORBIDDEN, None)

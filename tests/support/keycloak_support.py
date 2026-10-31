"""The throwaway Keycloak realm of `tests/fixtures/keycloak/realm.json`.

The `keycloak-token-shapes` CI job serves the realm; these helpers ask it for
tokens and build the speak wire's callers over its key set.
"""

from __future__ import annotations

import os
import time
from typing import TYPE_CHECKING

import httpx2 as httpx
import jwt

from speech_runtime_service.callers import SpeechCallers
from tests.support.identity_support import MINTING, caller_config

if TYPE_CHECKING:
    from collections.abc import Mapping

REALM = "speech-test"
WORKLOAD_ROLE = "speech-workload"
AUDIENCE_SCOPE = "speech-service-audience"
REQUEST_TIMEOUT_SECONDS = 10

# The realm's clients; each one's secret is `<client>-secret`, each person's
# password `<user>-test-password`.
APP_CLIENT = MINTING.client_id
OTHER_CLIENT = "other-client"
ROLELESS_CLIENT = "roleless-client"
UNSCOPED_CLIENT = "unscoped-client"


def realm_url() -> str:
    """The realm's issuer address on the CI Keycloak."""
    return f"{os.environ['KEYCLOAK_URL']}/realms/{REALM}"


def fetch_key_set_from_test_keycloak(_configured_url: str) -> bytes:
    """The realm's key set, read over plain HTTP.

    The configuration keeps its `https://` address; the CI Keycloak serves no TLS.
    """
    response = httpx.get(
        f"{realm_url()}/protocol/openid-connect/certs", timeout=REQUEST_TIMEOUT_SECONDS
    )
    response.raise_for_status()
    return response.content


def keycloak_callers() -> SpeechCallers:
    """Callers configured for the realm's values, verifying against its real keys."""
    return SpeechCallers.from_config(
        caller_config(
            issuer=realm_url(),
            workload_role=WORKLOAD_ROLE,
        ),
        fetch_key_set=fetch_key_set_from_test_keycloak,
        wall_clock=time.time,
        monotonic_clock=time.monotonic,
    )


def token_response(form: Mapping[str, str]) -> httpx.Response:
    """The realm's token endpoint answer to `form`."""
    return httpx.post(
        f"{realm_url()}/protocol/openid-connect/token",
        data=form,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )


def _issued(form: Mapping[str, str]) -> str:
    response = token_response(form)
    assert response.status_code == httpx.codes.OK, response.text
    return response.json()["access_token"]


def _scope(*, with_audience: bool) -> dict[str, str]:
    return {"scope": AUDIENCE_SCOPE} if with_audience else {}


def workload_token(client: str, *, with_audience: bool = True) -> str:
    """The client's `client_credentials` token, asking for the audience scope or not."""
    return _issued(
        {
            "grant_type": "client_credentials",
            "client_id": client,
            "client_secret": f"{client}-secret",
        }
        | _scope(with_audience=with_audience)
    )


def person_token(
    user: str, client: str = APP_CLIENT, *, with_audience: bool = True
) -> str:
    """`user`'s token as signed in at `client`, asking for the audience scope or not."""
    return _issued(
        {
            "grant_type": "password",
            "client_id": client,
            "client_secret": f"{client}-secret",
            "username": user,
            "password": f"{user}-test-password",
        }
        | _scope(with_audience=with_audience)
    )


def claims_of(token: str) -> Mapping[str, object]:
    """The claims of `token`; the speak routes are what verify the signature."""
    return jwt.decode(token, options={"verify_signature": False})


def audiences_of(claims: Mapping[str, object]) -> set[str]:
    """The `aud` claim as a set, whether Keycloak sent one string or a list."""
    audience = claims.get("aud", [])
    return {audience} if isinstance(audience, str) else set(audience)

"""The proof's throwaway identity mints tokens the service's check can admit."""

from __future__ import annotations

import json
import socket
import ssl
import stat
import time
import tomllib
import urllib.error
import urllib.parse
import urllib.request
from typing import TYPE_CHECKING

import jwt
import pytest

from proof_identity import (
    mint_pair,
    proof_identity,
)

if TYPE_CHECKING:
    from pathlib import Path

    from proof_identity import Minting, ProofIdentity

PEM_PRIVATE_KEY_HEADER = "-----BEGIN"
REQUIRED_WORKLOAD = ["iss", "aud", "azp", "client_id", "sub", "iat", "exp"]
REQUIRED_PERSON = ["iss", "aud", "azp", "sub", "iat", "exp"]
PAST_EXPIRY_SECONDS = 301
WORKLOAD_LIMIT_SECONDS = 300
OWNER_ONLY = stat.S_IRUSR | stat.S_IWUSR


def _fetch_key_set(identity: ProofIdentity, context: ssl.SSLContext) -> jwt.PyJWK:
    with urllib.request.urlopen(  # noqa: S310 - the proof's own https URL
        identity.jwks_url, context=context, timeout=10
    ) as response:
        (key,) = json.load(response)["keys"]
    return jwt.PyJWK.from_dict(key)


def _verify(token: str, key: jwt.PyJWK, minting: Minting, required: list[str]) -> dict:
    return jwt.decode(
        token,
        key,
        algorithms=["RS256"],
        audience=minting.audience,
        issuer=minting.issuer,
        options={"require": required},
    )


def _trusting_only(identity: ProofIdentity) -> ssl.SSLContext:
    return ssl.create_default_context(cafile=str(identity.ca_path))


def test_a_minted_pair_verifies_against_the_key_set_served_over_trusted_https(
    identity: ProofIdentity, minting: Minting
) -> None:
    key = _fetch_key_set(identity, _trusting_only(identity))
    pair = mint_pair(minting)

    workload = _verify(pair.workload, key, minting, REQUIRED_WORKLOAD)
    person = _verify(pair.person, key, minting, REQUIRED_PERSON)

    assert workload["sub"] != person["sub"]
    assert workload["azp"] == person["azp"] == workload["client_id"]
    assert minting.workload_role in workload["realm_access"]["roles"]
    assert {minting.app_group, minting.service_group} <= set(
        person[minting.group_claim]
    )


def test_the_workload_token_lives_at_most_300_seconds(minting: Minting) -> None:
    claims = jwt.decode(
        mint_pair(minting).workload, options={"verify_signature": False}
    )

    assert claims["exp"] - claims["iat"] <= WORKLOAD_LIMIT_SECONDS


def test_a_pair_minted_later_is_valid_when_the_one_minted_before_has_expired(
    identity: ProofIdentity, minting: Minting
) -> None:
    key = _fetch_key_set(identity, _trusting_only(identity))
    now = int(time.time())
    earlier = mint_pair(minting, clock=lambda: now - PAST_EXPIRY_SECONDS)
    later = mint_pair(minting, clock=lambda: now)

    _verify(later.workload, key, minting, REQUIRED_WORKLOAD)
    _verify(later.person, key, minting, REQUIRED_PERSON)
    for expired in (earlier.workload, earlier.person):
        with pytest.raises(jwt.ExpiredSignatureError):
            _verify(expired, key, minting, ["exp"])


def test_the_key_set_fetch_fails_on_the_certificate_without_the_proofs_ca(
    identity: ProofIdentity, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("SSL_CERT_FILE", raising=False)
    context = ssl.create_default_context()

    with pytest.raises(urllib.error.URLError) as failure:
        _fetch_key_set(identity, context)

    assert isinstance(failure.value.reason, ssl.SSLCertVerificationError)


def test_a_client_withholding_the_tls_handshake_does_not_block_the_key_set(
    identity: ProofIdentity,
) -> None:
    address = urllib.parse.urlsplit(identity.jwks_url)
    with socket.create_connection((address.hostname, address.port)):
        key = _fetch_key_set(identity, _trusting_only(identity))

    assert key.key_id


def test_the_identity_directory_and_every_file_in_it_are_private(
    identity: ProofIdentity,
) -> None:
    directory = identity.configuration_path.parent
    files = list(directory.iterdir())

    assert stat.S_IMODE(directory.stat().st_mode) == stat.S_IRWXU
    assert {identity.configuration_path, identity.minting_path, identity.ca_path} <= {
        *files
    }
    assert all(stat.S_IMODE(path.stat().st_mode) == OWNER_ONLY for path in files)


def test_the_service_configuration_holds_no_key_material_and_names_the_voice(
    identity: ProofIdentity, minting: Minting, proof_voice: str
) -> None:
    text = identity.configuration_path.read_text()
    configuration = tomllib.loads(text)

    assert PEM_PRIVATE_KEY_HEADER not in text
    assert minting.private_key_pem not in text
    assert configuration["identity"] == {
        "issuer": minting.issuer,
        "jwks_url": identity.jwks_url,
        "audience": minting.audience,
        "workload_role": minting.workload_role,
        "group_claim": minting.group_claim,
        "service_group": minting.service_group,
        "leeway_seconds": 0,
    }
    assert configuration["apps"] == {
        "proof": {"workload_client": minting.client_id, "voices": [proof_voice]}
    }


def test_leaving_the_context_stops_the_key_server(
    tmp_path: Path, proof_voice: str
) -> None:
    with proof_identity(tmp_path, proof_voice) as identity:
        port = int(identity.jwks_url.split(":")[-1].split("/")[0])

    with (
        pytest.raises(ConnectionRefusedError),
        socket.create_connection(("127.0.0.1", port), timeout=5),
    ):
        pass

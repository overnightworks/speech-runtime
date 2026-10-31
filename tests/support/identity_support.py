"""A throwaway issuer for the speak wire: keys, key-set fakes, tokens and callers.

Tokens come from the stage proof's own minting (`proof_identity`, on the
conftest's `scripts` path); a scenario alters a minted token's claims or
signature instead of restating the claim shapes.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field, replace
from enum import Enum
from functools import cache
from typing import TYPE_CHECKING

import jwt
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from webauth.https_client import KeySetFetchError

from proof_identity import ALGORITHM, Minting, TokenPair, mint_pair
from speech_runtime_client.wire import speech_request_headers
from speech_runtime_service.callers import SpeechCallers
from speech_runtime_service.config import (
    AppConfig,
    CallerConfig,
    IdentityConfig,
    load_caller_config,
)
from speech_runtime_service.voices import VoiceId

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable
    from pathlib import Path

ISSUER = "https://identity.test/realm"
KEY_SET_ADDRESS = "https://identity.test/certs?credential=SENTINEL-KEY-SET-CREDENTIAL"
APP = "quiz"
OTHER_APP = "other"
ALL_VOICES = frozenset(VoiceId)
KEY_BITS = 2048


class Absent(Enum):
    """Remove the claim instead of setting it."""

    CLAIM = "absent"


ABSENT = Absent.CLAIM


@dataclass(frozen=True)
class SigningKey:
    """One issuer key: the private half signs, the public half is published."""

    key_identifier: str
    private_pem: str = field(repr=False)
    public_jwk: dict[str, str]


@cache
def signing_key(key_identifier: str) -> SigningKey:
    """The process's one key for `key_identifier`."""
    private = rsa.generate_private_key(public_exponent=65537, key_size=KEY_BITS)
    private_pem = private.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    ).decode()
    public_jwk = json.loads(jwt.algorithms.RSAAlgorithm.to_jwk(private.public_key()))
    return SigningKey(
        key_identifier,
        private_pem,
        public_jwk | {"kid": key_identifier, "use": "sig", "alg": ALGORITHM},
    )


@dataclass
class Clocks:
    """The wall clock and the monotonic clock of one scenario, moved together."""

    wall_seconds: float = field(default_factory=time.time)
    monotonic_seconds: float = 1_000.0

    def wall(self) -> float:
        """The wall clock's seconds."""
        return self.wall_seconds

    def monotonic(self) -> float:
        """The monotonic clock's seconds."""
        return self.monotonic_seconds

    def advance(self, seconds: float) -> None:
        """Let `seconds` pass on both clocks."""
        self.wall_seconds += seconds
        self.monotonic_seconds += seconds


class FakeKeySet:
    """The issuer's key-set endpoint: publishes keys, can go down or be held."""

    def __init__(self, *published: SigningKey) -> None:
        """Publish `published`; the endpoint starts up and unheld."""
        self.published = list(published)
        self.down = False
        self.fetches = 0
        self.hold: threading.Event | None = None
        self.held = threading.Event()

    def __call__(self, _url: str) -> bytes:
        """Answer the fetch like the issuer: the key set, or a failure."""
        self.fetches += 1
        if self.hold is not None:
            self.held.set()
            self.hold.wait(timeout=10)
        if self.down:
            message = "the key set is down"
            raise KeySetFetchError(message)
        return json.dumps({"keys": [key.public_jwk for key in self.published]}).encode()


def signing_key_of(minting: Minting) -> SigningKey:
    """The key a minting file signs with, as the issuer would publish it."""
    private = serialization.load_pem_private_key(
        minting.private_key_pem.encode(), password=None
    )
    public_jwk = json.loads(jwt.algorithms.RSAAlgorithm.to_jwk(private.public_key()))
    return SigningKey(
        minting.key_identifier,
        minting.private_key_pem,
        public_jwk | {"kid": minting.key_identifier, "use": "sig", "alg": ALGORITHM},
    )


def minting_for(key: SigningKey) -> Minting:
    """The minting facts of the scenario issuer, signing with `key`."""
    return Minting(
        private_key_pem=key.private_pem,
        key_identifier=key.key_identifier,
        issuer=ISSUER,
        audience="speech-service",
        client_id="quiz-client",
        workload_role="workload",
        group_claim="groups",
        app_group=f"/apps/{APP}",
        service_group="/services/speech-service",
        workload_subject="quiz-service-account",
        person_subject="quiz-player",
    )


DEFAULT_KEY = signing_key("key-one")
MINTING = minting_for(DEFAULT_KEY)


OTHER_MINTING = replace(
    MINTING, client_id="other-client", app_group=f"/apps/{OTHER_APP}"
)


def caller_config(
    voices: Iterable[VoiceId] = ALL_VOICES, *, hear: bool = False, **identity: object
) -> CallerConfig:
    """The service configuration that trusts the scenario issuer.

    The one app lists `voices`, by default every catalogue voice, and hears when
    `hear` says so.
    """
    facts = {
        "issuer": MINTING.issuer,
        "jwks_url": KEY_SET_ADDRESS,
        "audience": MINTING.audience,
        "workload_role": MINTING.workload_role,
        "group_claim": MINTING.group_claim,
        "service_group": MINTING.service_group,
    } | identity
    return CallerConfig(
        identity=IdentityConfig(**facts),
        apps={
            APP: AppConfig(
                workload_client=MINTING.client_id,
                voices=frozenset(voices),
                hear=hear,
            )
        },
    )


def callers_of_two_apps(
    quiz_voices: Iterable[VoiceId], other_voices: Iterable[VoiceId]
) -> SpeechCallers:
    """Callers on real clocks over `APP` and `OTHER_APP`, each with its own list."""
    config = caller_config()
    apps = {
        APP: AppConfig(
            workload_client=MINTING.client_id, voices=frozenset(quiz_voices)
        ),
        OTHER_APP: AppConfig(
            workload_client=OTHER_MINTING.client_id, voices=frozenset(other_voices)
        ),
    }
    return SpeechCallers.from_config(
        config.model_copy(update={"apps": apps}),
        fetch_key_set=FakeKeySet(DEFAULT_KEY),
        wall_clock=time.time,
        monotonic_clock=time.monotonic,
    )


def callers_over(
    key_set: FakeKeySet | None = None, clocks: Clocks | None = None
) -> SpeechCallers:
    """Callers that verify against `key_set`, by default the default key's."""
    key_set = key_set or FakeKeySet(DEFAULT_KEY)
    clocks = clocks or Clocks()
    return SpeechCallers.from_config(
        caller_config(),
        fetch_key_set=key_set,
        wall_clock=clocks.wall,
        monotonic_clock=clocks.monotonic,
    )


def callers_trusting(configuration_path: Path, minting: Minting) -> SpeechCallers:
    """Callers from a configuration file; the issuer publishes `minting`'s key."""
    return SpeechCallers.from_config(
        load_caller_config(configuration_path),
        fetch_key_set=FakeKeySet(signing_key_of(minting)),
        wall_clock=time.time,
        monotonic_clock=time.monotonic,
    )


def valid_callers(
    voices: Iterable[VoiceId] = ALL_VOICES, *, hear: bool = False
) -> SpeechCallers:
    """Callers on real clocks that accept `valid_headers()`.

    The app lists `voices` and hears when `hear` says so.
    """
    return SpeechCallers.from_config(
        caller_config(voices, hear=hear),
        fetch_key_set=FakeKeySet(DEFAULT_KEY),
        wall_clock=time.time,
        monotonic_clock=time.monotonic,
    )


def valid_pair(
    clock: Callable[[], float] = time.time, minting: Minting = MINTING
) -> TokenPair:
    """A workload and a person token the scenario issuer vouches for at `clock()`."""
    return mint_pair(minting, clock)


def valid_headers(
    clock: Callable[[], float] = time.time, minting: Minting = MINTING
) -> dict[str, str]:
    """The wire headers of a valid pair, without the content type."""
    pair = valid_pair(clock, minting)
    return speech_request_headers(
        workload_token=pair.workload, person_token=pair.person
    )


def altered(
    token: str,
    *,
    signed_by: SigningKey = DEFAULT_KEY,
    key_identifier: str | None = None,
    **claims: object,
) -> str:
    """`token` with `claims` set (or removed, for `ABSENT`) and signed again."""
    body = jwt.decode(token, options={"verify_signature": False})
    for name, value in claims.items():
        if value is ABSENT:
            body.pop(name, None)
        else:
            body[name] = value
    return jwt.encode(
        body,
        signed_by.private_pem,
        ALGORITHM,
        headers={"kid": key_identifier or signed_by.key_identifier},
    )


def caller_configuration_text(
    jwks_url: str = KEY_SET_ADDRESS,
    *,
    identity: str = "",
    apps: str | None = None,
    voices: Iterable[VoiceId] = ALL_VOICES,
) -> str:
    """The TOML file that trusts the scenario issuer, as the service reads it.

    `identity` adds lines to the identity table; `apps` replaces the app tables;
    the default app lists `voices`.
    """
    registered = (
        f"[apps.{APP}]\nworkload_client = {json.dumps(MINTING.client_id)}\n"
        f"voices = {json.dumps([voice.value for voice in voices])}\n"
        if apps is None
        else apps
    )
    return "\n".join(
        [
            "[identity]",
            f"issuer = {json.dumps(MINTING.issuer)}",
            f"jwks_url = {json.dumps(jwks_url)}",
            f"audience = {json.dumps(MINTING.audience)}",
            f"workload_role = {json.dumps(MINTING.workload_role)}",
            f"group_claim = {json.dumps(MINTING.group_claim)}",
            f"service_group = {json.dumps(MINTING.service_group)}",
            identity,
            "",
            registered,
        ]
    )


def write_caller_configuration(directory: Path, text: str | None = None) -> Path:
    """Write the caller configuration (by default the issuer's) under `directory`."""
    path = directory / "caller-configuration.toml"
    path.write_text(caller_configuration_text() if text is None else text)
    return path

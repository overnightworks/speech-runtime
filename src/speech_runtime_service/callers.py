"""Who may speak: the proof of the two tokens of a request and the service's own policy.

webauth proves the tokens (`ServiceCallerCheck.person_caller`); this module adds
the policy that is the service's: only a person call is admitted, the person
must belong to exactly the configured service group, and an app may name only
the voices its table lists. It logs nothing, and
nothing taken from a token leaves it.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Final

from webauth.oidc_login import AdmissionConfig, NoAdmins
from webauth.service_callers import (
    CallerRefusalClass,
    PersonCaller,
    RegisteredApp,
    ServiceCallerCheck,
    ServiceIdentity,
)

from speech_runtime_client.wire import (
    AUTHORIZATION_HEADER,
    WORKLOAD_TOKEN_HEADER,
    RefusalCode,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from starlette.datastructures import Headers
    from webauth.jwt_verification import KeySetFetch, MonotonicClock
    from webauth.ports import WallClock

    from speech_runtime_service.config import CallerConfig
    from speech_runtime_service.voices import VoiceId

MAX_WORKLOAD_SECONDS: Final = 300
_BEARER_SCHEME: Final = "bearer"
_REFUSAL_CODE: Final = {
    CallerRefusalClass.UNAUTHENTICATED: RefusalCode.UNAUTHORIZED,
    CallerRefusalClass.UNAVAILABLE: RefusalCode.UNAVAILABLE,
    CallerRefusalClass.FORBIDDEN: RefusalCode.FORBIDDEN,
}


class SpeechCallers:
    """The one check of a process, over the configured issuer and apps."""

    def __init__(
        self,
        check: ServiceCallerCheck,
        service_group: str,
        voices_by_app: Mapping[str, frozenset[VoiceId]],
    ) -> None:
        """Hold `check`, the group a person must belong to and each app's voices."""
        self._check = check
        self._service_group = service_group
        self._voices_by_app = voices_by_app

    @classmethod
    def from_config(
        cls,
        config: CallerConfig,
        *,
        fetch_key_set: KeySetFetch,
        wall_clock: WallClock,
        monotonic_clock: MonotonicClock,
    ) -> SpeechCallers:
        """Build the check from the configuration and the clocks and fetch it uses."""
        identity = config.identity
        registered = {
            app.workload_client: RegisteredApp(
                app=name,
                admission=AdmissionConfig(
                    bound_issuer=identity.issuer,
                    login_group=f"/apps/{name}",
                    admin_group=NoAdmins.IN_THIS_APP,
                ),
            )
            for name, app in config.apps.items()
        }
        check = ServiceCallerCheck(
            ServiceIdentity(
                issuer=identity.issuer,
                audience=identity.audience,
                jwks_url=identity.jwks_url,
                workload_role=identity.workload_role,
                group_claim=identity.group_claim,
                max_workload_seconds=MAX_WORKLOAD_SECONDS,
                leeway_seconds=identity.leeway_seconds,
            ),
            registered_apps=registered,
            fetch_key_set=fetch_key_set,
            wall_clock=wall_clock,
            monotonic_clock=monotonic_clock,
        )
        voices_by_app = {name: app.voices for name, app in config.apps.items()}
        return cls(check, identity.service_group, voices_by_app)

    async def admit(self, headers: Headers) -> PersonCaller | RefusalCode:
        """The person call the request's tokens prove, or the refusal to answer."""
        caller = await asyncio.to_thread(
            self._check.person_caller,
            headers.get(WORKLOAD_TOKEN_HEADER),
            _bearer_token(headers.get(AUTHORIZATION_HEADER)),
        )
        if not isinstance(caller, PersonCaller):
            return _REFUSAL_CODE[caller.refusal_class]
        if self._service_group not in caller.groups:
            return RefusalCode.FORBIDDEN
        return caller

    def may_name(self, app: str, voice: VoiceId) -> bool:
        """Whether the configuration lists `voice` for the admitted `app`."""
        return voice in self._voices_by_app[app]


def _bearer_token(authorization: str | None) -> str | None:
    """The token of a `Bearer` authorization; any other scheme counts as none."""
    if authorization is None:
        return None
    scheme, _, token = authorization.partition(" ")
    return token if scheme.lower() == _BEARER_SCHEME else None

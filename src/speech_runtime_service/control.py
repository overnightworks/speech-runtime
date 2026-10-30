"""The private Unix-socket app: health and the speech wire."""

from typing import TYPE_CHECKING

from fastapi import FastAPI

from speech_runtime_service.service import Runtime, mount_health_route
from speech_runtime_service.wire_route import mount_wire_routes

if TYPE_CHECKING:
    from speech_runtime_service.callers import SpeechCallers
    from speech_runtime_service.config import Settings


def create_control_app(
    settings: Settings, runtime: Runtime, callers: SpeechCallers
) -> FastAPI:
    """Build the private app: health and the speech wire, no docs."""
    app = FastAPI(
        title="speech-runtime-private",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    mount_health_route(app, runtime)
    mount_wire_routes(app, runtime, settings, callers)
    return app

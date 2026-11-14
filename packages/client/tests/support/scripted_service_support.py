"""A service over TCP that answers health and speaks from a script.

The script is state a test changes between and during calls: the voices health
offers, and the voices whose speak is refused as unavailable.
"""

from __future__ import annotations

import json
import threading
from contextlib import contextmanager
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import TYPE_CHECKING

from speech_client.wire import (
    REFUSAL_STATUS,
    VOICES,
    RefusalCode,
    SpeakingModel,
    VoiceId,
    refusal,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

POLL_SECONDS = 0.01
WAV = b"RIFF\0\0\0\0WAVE"


def voice_health(voice: VoiceId, *, ready: bool = True) -> dict[str, object]:
    """The health entry of `voice` as the service lists it."""
    catalogue = VOICES[voice]
    return {
        "model": catalogue.model.value,
        "name": catalogue.name,
        "languages": [language.value for language in catalogue.languages],
        "takes_instructions": catalogue.takes_instructions,
        "ready": ready,
        "sample_rate": 24000 if catalogue.model is SpeakingModel.QWEN else 22050,
    }


class ScriptedService:
    """What the service offers and refuses now, and what it was asked."""

    def __init__(self, offered: dict[VoiceId, bool]) -> None:
        """Start with `offered` (voice to ready), refusing nothing."""
        self.offered = offered
        self.unavailable_on_speak: set[VoiceId] = set()
        self.on_speak: Callable[[VoiceId], None] = lambda _voice: None
        self.asked: list[dict[str, object]] = []

    def health_body(self) -> bytes:
        """The health the service answers now."""
        voices = {
            voice.value: voice_health(voice, ready=ready)
            for voice, ready in self.offered.items()
        }
        return json.dumps(
            {
                "speaking": {"ready": True, "voices": voices},
                "hearing": {"model": "whisper", "ready": True},
                "card_memory_mb": 0,
            }
        ).encode()


def _handler(service: ScriptedService) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args: object) -> None:
            del args

        def _send(self, status: HTTPStatus, body: bytes) -> None:
            self.send_response(status)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            self._send(HTTPStatus.OK, service.health_body())

        def do_POST(self) -> None:
            length = int(self.headers["Content-Length"])
            asked = json.loads(self.rfile.read(length))
            service.asked.append(asked)
            voice = VoiceId(asked["voice"])
            service.on_speak(voice)
            if voice in service.unavailable_on_speak:
                body = refusal(RefusalCode.UNAVAILABLE, "voice").model_dump_json()
                self._send(REFUSAL_STATUS[RefusalCode.UNAVAILABLE], body.encode())
                return
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "audio/wav")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            self.wfile.write(b"%x\r\n%b\r\n0\r\n\r\n" % (len(WAV), WAV))

    return Handler


@contextmanager
def scripted_service(
    offered: dict[VoiceId, bool],
) -> Iterator[tuple[str, ScriptedService]]:
    """Serve `offered` (voice to ready) and yield the base URL and the script."""
    service = ScriptedService(offered)
    server = ThreadingHTTPServer(("127.0.0.1", 0), _handler(service))
    thread = threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.01}
    )
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}", service
    finally:
        server.shutdown()
        server.server_close()
        thread.join()

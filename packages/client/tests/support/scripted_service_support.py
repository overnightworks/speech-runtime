"""A service over TCP that answers health and speaks from a script.

The script is state a test changes between and during calls: the voices health
offers, the voices whose speak is refused as unavailable and the voices whose
speak ends after its first chunk without the body's last one. The audio it
speaks is in `audio_format`.
"""

from __future__ import annotations

import json
import struct
import threading
from contextlib import contextmanager
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import TYPE_CHECKING

from speech_client.wire import (
    REFUSAL_STATUS,
    VOICES,
    RefusalCode,
    ResponseFormat,
    SpeakingModel,
    VoiceId,
    refusal,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

POLL_SECONDS = 0.01
SAMPLE_RATE = 22_050


def wav_of(pcm: bytes, *, with_data: bool = True) -> bytes:
    """A WAV head with an INFO chunk before `data`, as the service sends it."""
    info = struct.pack("<4sI4s", b"LIST", 4 + 8 + 4, b"INFO") + struct.pack(
        "<4sI", b"ICMT", 4
    )
    info += b"AI\0\0"
    fmt = struct.pack(
        "<4sIHHIIHH", b"fmt ", 16, 1, 1, SAMPLE_RATE, SAMPLE_RATE * 2, 2, 16
    )
    data = struct.pack("<4sI", b"data", 0xFFFFFFFF) + pcm if with_data else b""
    return b"RIFF\xff\xff\xff\xffWAVE" + fmt + info + data


WAV = wav_of(b"\x01\x00" * 4)
OGG = b"OggS" + bytes(64)


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
        self.cut_after_first_chunk: set[VoiceId] = set()
        self.audio_format = ResponseFormat.WAV
        self.on_health: Callable[[], None] = lambda: None
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
            service.on_health()
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
            self.send_header("Content-Type", service.audio_format.media_type)
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            audio = OGG if service.audio_format is ResponseFormat.OPUS else WAV
            self.wfile.write(b"%x\r\n%b\r\n" % (len(audio), audio))
            if voice in service.cut_after_first_chunk:
                self.close_connection = True
                return
            self.wfile.write(b"0\r\n\r\n")

    return Handler


@contextmanager
def scripted_service(
    offered: dict[VoiceId, bool],
) -> Iterator[tuple[str, ScriptedService]]:
    """Serve `offered` (voice to ready) and yield the base URL and the script."""
    service = ScriptedService(offered)
    server = ThreadingHTTPServer(("127.0.0.1", 0), _handler(service))
    thread = threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": POLL_SECONDS}
    )
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}", service
    finally:
        server.shutdown()
        server.server_close()
        thread.join()

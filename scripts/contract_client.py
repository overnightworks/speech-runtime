"""Drive health and the speak wire over the private socket.

This process must not import the `speech_runtime_service` package. It is the
parallel-lane proof: a caller that only knows the HTTP contract, through the
types of `speech_runtime_client`.
"""

# ruff: noqa: T201 - the client prints its measurements as the command line output

from __future__ import annotations

import argparse
import http.client
import json
import socket
import sys
from array import array
from dataclasses import dataclass, field
from pathlib import Path

from proof_identity import Minting, load_minting, mint_pair
from speech_runtime_client.wire import (
    SPEECH_PATH,
    speech_request_body,
    speech_request_headers,
)

SENTENCE = (
    "Die Kamera zeigt den Vortragenden, während im Hintergrund "
    "die nächste Folie automatisch vorbereitet wird."
)
BYTES_PER_SAMPLE = 2
SILENCE_RMS = 0.01


def _must_not_have_imported_speech() -> None:
    if "speech_runtime_service" in sys.modules:
        message = "contract client imported speech_runtime_service"
        raise SystemExit(message)


class _UnixHttpConnection(http.client.HTTPConnection):
    def __init__(self, path: Path, timeout: float) -> None:
        super().__init__("speech", timeout=timeout)
        self._path = path

    def connect(self) -> None:
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(self.timeout)
        self.sock.connect(str(self._path))


@dataclass(frozen=True)
class _Post:
    body: bytes
    headers: dict[str, str] = field(repr=False)


def _request(
    path: Path, method: str, target: str, post: _Post | None, timeout: float
) -> bytes:
    connection = _UnixHttpConnection(path, timeout)
    try:
        connection.request(
            method,
            target,
            post.body if post else None,
            post.headers if post else {},
        )
        response = connection.getresponse()
        data = response.read()
    except http.client.IncompleteRead:
        message = f"{method} {target} ended before its body was complete"
        raise SystemExit(message) from None
    finally:
        connection.close()
    if response.status != http.client.OK:
        message = f"{method} {target} answered {response.status}"
        raise SystemExit(message)
    return data


def get_json(path: Path, target: str) -> dict[str, object]:
    """GET a path over the speech socket and decode the JSON body."""
    return json.loads(_request(path, "GET", target, None, 30).decode("utf-8"))


def _pcm_from_wav(data: bytes) -> tuple[int, bytes]:
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        message = "speak did not return a WAV"
        raise SystemExit(message)
    rate = int.from_bytes(data[24:28], "little")
    position = 12
    while position + 8 <= len(data):
        chunk_id = data[position : position + 4]
        size = int.from_bytes(data[position + 4 : position + 8], "little")
        payload = position + 8
        if chunk_id == b"data":
            return rate, data[payload:]
        position = payload + size
    message = "WAV has no data chunk"
    raise SystemExit(message)


def speak(path: Path, text: str, voice: str, minting: Minting) -> bytes:
    """POST German text to the speech socket and return the whole WAV bytes.

    Each call mints a fresh pair of tokens, so a long run never presents an
    expired one.
    """
    pair = mint_pair(minting)
    headers = speech_request_headers(
        workload_token=pair.workload, person_token=pair.person
    )
    post = _Post(speech_request_body(text, voice), headers)
    return _request(path, "POST", SPEECH_PATH, post, 60)


def _rms(pcm: bytes) -> float:
    if len(pcm) < BYTES_PER_SAMPLE:
        return 0.0
    samples = array("h")
    samples.frombytes(pcm[: len(pcm) - (len(pcm) % BYTES_PER_SAMPLE)])
    total = sum(sample * sample for sample in samples)
    return (total / len(samples)) ** 0.5 / 32768.0


def _run(path: Path, voice: str, minting: Minting) -> None:
    health = get_json(path, "/health")
    print("contract_health", json.dumps(health, sort_keys=True))
    speaking = health["speaking"]
    if not isinstance(speaking, dict):
        message = "health speaking is not an object"
        raise SystemExit(message)
    if not speaking["ready"]:
        message = "the speaking model is not ready"
        raise SystemExit(message)

    wav = speak(path, SENTENCE, voice, minting)
    rate, pcm = _pcm_from_wav(wav)
    print(
        "contract_speak",
        f"bytes={len(wav)} pcm={len(pcm)} rate={rate} rms={_rms(pcm):.4f}",
    )
    if _rms(pcm) < SILENCE_RMS:
        message = "speak returned silence"
        raise SystemExit(message)


def parse_arguments(arguments: list[str]) -> argparse.Namespace:
    """Read the command line: the socket, the voice and the minting file's path."""
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--minting-file", type=Path, required=True)
    parser.add_argument("socket", type=Path)
    parser.add_argument("voice")
    return parser.parse_args(arguments)


def main() -> None:
    """Drive health and speak against the speech socket path."""
    _must_not_have_imported_speech()
    options = parse_arguments(sys.argv[1:])
    _run(options.socket, options.voice, load_minting(options.minting_file))


if __name__ == "__main__":
    main()

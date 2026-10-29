"""Drive the three speech endpoints over the private socket.

This process must not import the `speech_runtime_service` package. It is the
parallel-lane proof: a caller that only knows the HTTP contract.
"""

from __future__ import annotations

import asyncio
import contextlib
import http.client
import json
import re
import socket
import statistics
import sys
import time
from array import array
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from websockets.asyncio.client import ClientConnection, unix_connect

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

SENTENCE = (
    "Die Kamera zeigt den Vortragenden, während im Hintergrund "
    "die nächste Folie automatisch vorbereitet wird."
)
SECOND = "Die nächste Folie bitte."
FRAME_SECONDS = 0.1
SILENCE_SECONDS = 0.9
MEASURE_REPETITIONS = 5
FINAL_WAIT_SECONDS = 30
BYTES_PER_SAMPLE = 2
SILENCE_RMS = 0.01
NO_FINAL = f"no final within {FINAL_WAIT_SECONDS} s"


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


def _request(
    path: Path, method: str, target: str, body: bytes | None, timeout: float
) -> bytes:
    connection = _UnixHttpConnection(path, timeout)
    try:
        connection.request(method, target, body, {"Content-Type": "application/json"})
        response = connection.getresponse()
        data = response.read()
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


def _resample(pcm: bytes, src_rate: int, dst_rate: int) -> bytes:
    if src_rate == dst_rate:
        return pcm
    samples = array("h")
    samples.frombytes(pcm)
    n_src = len(samples)
    n_dst = max(1, int(n_src * dst_rate / src_rate))
    out = array("h", [0] * n_dst)
    last = max(n_src - 1, 1)
    for index in range(n_dst):
        position = index * last / max(n_dst - 1, 1)
        left = int(position)
        frac = position - left
        right = min(left + 1, n_src - 1)
        value = samples[left] * (1 - frac) + samples[right] * frac
        out[index] = int(max(-32768, min(32767, value)))
    return out.tobytes()


def speak(path: Path, text: str) -> bytes:
    """POST German text to the speech socket and return the WAV bytes."""
    body = json.dumps({"text": text, "language": "de"}).encode("utf-8")
    return _request(path, "POST", "/speak", body, 60)


def _rms(pcm: bytes) -> float:
    if len(pcm) < BYTES_PER_SAMPLE:
        return 0.0
    samples = array("h")
    samples.frombytes(pcm[: len(pcm) - (len(pcm) % BYTES_PER_SAMPLE)])
    total = sum(sample * sample for sample in samples)
    return (total / len(samples)) ** 0.5 / 32768.0


@contextlib.asynccontextmanager
async def _at_most(seconds: float) -> AsyncIterator[None]:
    """Run the block for at most `seconds`; running out of time is not an error."""
    with contextlib.suppress(TimeoutError):
        async with asyncio.timeout(seconds):
            yield


async def _drain(
    websocket: ClientConnection,
    events: list[dict[str, object]],
) -> None:
    raw = await websocket.recv()
    if isinstance(raw, str):
        events.append(json.loads(raw))


def _first_nonempty_partial(
    events: list[dict[str, object]],
) -> dict[str, object] | None:
    for event in events:
        text = event.get("text")
        if event.get("final") is False and isinstance(text, str) and text:
            return event
    return None


def _first_final(events: list[dict[str, object]]) -> dict[str, object] | None:
    for event in events:
        if event.get("final") is True:
            return event
    return None


def frames_remain_unsent(pcm: bytes, next_offset: int) -> bool:
    """True when at least one PCM frame after next_offset has not been sent."""
    return next_offset < len(pcm)


class _FrameObserver(Protocol):
    def note_sent(self) -> None: ...

    def note(self) -> None: ...


async def send_pcm(
    websocket: ClientConnection,
    pcm: bytes,
    sample_rate: int,
    events: list[dict[str, object]],
    observer: _FrameObserver | None = None,
) -> None:
    """Send paced frames, telling the observer at each send and each receive."""
    frame_bytes = int(FRAME_SECONDS * sample_rate * BYTES_PER_SAMPLE)
    for offset in range(0, len(pcm), frame_bytes):
        await websocket.send(pcm[offset : offset + frame_bytes])
        if observer is not None:
            observer.note_sent()
        async with _at_most(0.02):
            await _drain(websocket, events)
        if observer is not None:
            observer.note()
        await asyncio.sleep(FRAME_SECONDS)
        async with _at_most(0.02):
            await _drain(websocket, events)
        if observer is not None:
            observer.note()


async def _hear_utterance(
    websocket: ClientConnection,
    pcm: bytes,
    sample_rate: int,
) -> tuple[list[dict[str, object]], dict[str, object] | None, float, int, bool]:
    started = time.perf_counter()
    events: list[dict[str, object]] = []
    partial: dict[str, object] | None = None
    partial_at = 0.0
    partial_frame = 0
    still_sending = False
    frame_bytes = int(FRAME_SECONDS * sample_rate * BYTES_PER_SAMPLE)
    for frames_sent, offset in enumerate(range(0, len(pcm), frame_bytes), start=1):
        await websocket.send(pcm[offset : offset + frame_bytes])
        next_offset = offset + frame_bytes
        async with _at_most(0.02):
            await _drain(websocket, events)
        await asyncio.sleep(FRAME_SECONDS)
        async with _at_most(0.02):
            await _drain(websocket, events)
        if partial is None:
            found = _first_nonempty_partial(events)
            if found is not None:
                partial = found
                partial_at = time.perf_counter() - started
                partial_frame = frames_sent
                still_sending = frames_remain_unsent(pcm, next_offset)
    silence = bytes(int(SILENCE_SECONDS * sample_rate * BYTES_PER_SAMPLE))
    await send_pcm(websocket, silence, sample_rate, events)
    deadline = time.monotonic() + 8
    while time.monotonic() < deadline:
        if _first_final(events) is not None:
            break
        async with _at_most(0.3):
            await _drain(websocket, events)
    return events, partial, partial_at, partial_frame, still_sending


@dataclass(frozen=True)
class Repetition:
    """One heard utterance, every time counted in seconds from the start of sending."""

    audio_seconds: float
    speech_ended_s: float
    sending_ended_s: float
    partial_at_s: float | None
    final_at_s: float | None
    final_text: str | None

    @property
    def first_partial(self) -> str:
        """Say when the first non-empty partial came, or that none did."""
        if self.partial_at_s is None:
            return "no partial"
        sending = self.partial_at_s <= self.speech_ended_s
        when = "while sending" if sending else "after speech ended"
        return f"{self.partial_at_s:.3f} s {when}"

    @property
    def sending_to_final_s(self) -> float | None:
        """The moment `contract_hear_final` reports: not before the silence is sent."""
        if self.final_at_s is None:
            return None
        return max(self.final_at_s, self.sending_ended_s)

    @property
    def speech_end_to_final_s(self) -> float | None:
        """What a speaker waits for after the last speech frame."""
        if self.final_at_s is None:
            return None
        return self.final_at_s - self.speech_ended_s

    @property
    def final_description(self) -> str:
        """The final text, or the missed gate as a value."""
        return NO_FINAL if self.final_text is None else self.final_text


def summarize(repetitions: list[Repetition]) -> dict[str, object]:
    """Median and worst of the completed repetitions, and how many completed."""
    completed = [item for item in repetitions if item.final_at_s is not None]
    summary: dict[str, object] = {"completed": len(completed)}
    if not completed:
        return summary
    for name in ("sending_to_final_s", "speech_end_to_final_s"):
        values = [getattr(item, name) for item in completed]
        summary[name] = (statistics.median(values), max(values))
    partials = [item.partial_at_s for item in completed if item.partial_at_s]
    if partials:
        summary["partial_s"] = (statistics.median(partials), max(partials))
    return summary


def same_words(heard: str, spoken: str) -> bool:
    """Compare two texts word by word, ignoring case and punctuation."""
    return _words(heard) == _words(spoken)


def _words(text: str) -> list[str]:
    return re.findall(r"\w+", text.casefold())


def final_arrived_in_time(arrived_s: float, sending_ended_s: float) -> bool:
    """True when a final reached the client within the wait after sending ended."""
    return arrived_s - sending_ended_s <= FINAL_WAIT_SECONDS


class _Arrivals:
    """When the first non-empty partial and the first final reached the client."""

    def __init__(self) -> None:
        self.events: list[dict[str, object]] = []
        self.started = time.perf_counter()
        self.partial_at_s: float | None = None
        self.final_at_s: float | None = None
        self.last_sent_s = 0.0
        self.sending_ended_s: float | None = None

    def elapsed(self) -> float:
        return time.perf_counter() - self.started

    def note_sent(self) -> None:
        self.last_sent_s = self.elapsed()

    def note(self) -> None:
        arrived_s = self.elapsed()
        if self.partial_at_s is None and _first_nonempty_partial(self.events):
            self.partial_at_s = arrived_s
        if (
            self.final_at_s is None
            and _first_final(self.events)
            and (
                self.sending_ended_s is None
                or final_arrived_in_time(arrived_s, self.sending_ended_s)
            )
        ):
            self.final_at_s = arrived_s


async def _measure_utterance(
    websocket: ClientConnection, pcm: bytes, sample_rate: int
) -> Repetition:
    arrivals = _Arrivals()
    await send_pcm(websocket, pcm, sample_rate, arrivals.events, arrivals)
    speech_ended_s = arrivals.last_sent_s
    silence = bytes(int(SILENCE_SECONDS * sample_rate * BYTES_PER_SAMPLE))
    await send_pcm(websocket, silence, sample_rate, arrivals.events, arrivals)
    sending_ended_s = arrivals.elapsed()
    arrivals.sending_ended_s = sending_ended_s
    deadline = time.monotonic() + FINAL_WAIT_SECONDS
    while arrivals.final_at_s is None:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        async with _at_most(min(0.3, remaining)):
            await _drain(websocket, arrivals.events)
        arrivals.note()
    final = _first_final(arrivals.events) if arrivals.final_at_s is not None else None
    return Repetition(
        audio_seconds=len(pcm) / 2 / sample_rate,
        speech_ended_s=speech_ended_s,
        sending_ended_s=sending_ended_s,
        partial_at_s=arrivals.partial_at_s,
        final_at_s=arrivals.final_at_s,
        final_text=None if final is None else str(final.get("text", "")),
    )


async def _run_measure(path: Path) -> None:
    """Repeat the first sentence on one socket and print what each repetition cost."""
    health = get_json(path, "/health")
    sample_rate = int(health["sample_rate"])
    rate, pcm = _pcm_from_wav(speak(path, SENTENCE))
    pcm = _resample(pcm, rate, sample_rate)
    repetitions: list[Repetition] = []
    async with unix_connect(
        str(path), uri="ws://speech/hear?language=de", max_size=None
    ) as connection:
        for number in range(1, MEASURE_REPETITIONS + 1):
            repetition = await _measure_utterance(connection, pcm, sample_rate)
            repetitions.append(repetition)
            print(
                f"measure_rep{number}",
                f"audio_s={repetition.audio_seconds:.3f}",
                f"speech_ended_s={repetition.speech_ended_s:.3f}",
                f"sending_ended_s={repetition.sending_ended_s:.3f}",
                f"first_partial={repetition.first_partial!r}",
                f"sending_to_final_s={repetition.sending_to_final_s}",
                f"speech_end_to_final_s={repetition.speech_end_to_final_s}",
                f"same_words={same_words(repetition.final_description, SENTENCE)}",
                json.dumps(repetition.final_description, ensure_ascii=False),
            )
            if repetition.final_at_s is None:
                break
    print("measure_summary", json.dumps(summarize(repetitions), sort_keys=True))


async def _run(path: Path) -> None:
    health = get_json(path, "/health")
    print("contract_health", json.dumps(health, sort_keys=True))
    speaking = health["speaking"]
    hearing = health["hearing"]
    if not isinstance(speaking, dict) or not isinstance(hearing, dict):
        message = "health speaking/hearing is not an object"
        raise SystemExit(message)
    if not speaking["ready"] or not hearing["ready"]:
        message = "models are not ready"
        raise SystemExit(message)
    sample_rate = int(health["sample_rate"])

    wav = speak(path, SENTENCE)
    rate, pcm = _pcm_from_wav(wav)
    pcm = _resample(pcm, rate, sample_rate)
    print(
        "contract_speak",
        f"bytes={len(wav)} pcm={len(pcm)} "
        f"rate={rate}->{sample_rate} rms={_rms(pcm):.4f}",
    )
    if _rms(pcm) < SILENCE_RMS:
        message = "speak returned silence"
        raise SystemExit(message)

    async with unix_connect(
        str(path), uri="ws://speech/hear?language=de", max_size=None
    ) as connection:
        first_started = time.perf_counter()
        (
            first,
            partial,
            partial_at,
            partial_frame,
            still_sending,
        ) = await _hear_utterance(connection, pcm, sample_rate)
        if partial is None or not still_sending:
            message = "no non-empty partial arrived while frames were still being sent"
            raise SystemExit(message)
        print(
            "contract_hear_partial",
            f"t={partial_at:.3f}s",
            f"frame={partial_frame}",
            f"still_sending={still_sending}",
            json.dumps(partial, ensure_ascii=False),
        )
        first_final = _first_final(first)
        if first_final is None:
            message = "no final transcript arrived"
            raise SystemExit(message)
        print(
            "contract_hear_final",
            f"t={time.perf_counter() - first_started:.3f}s",
            json.dumps(first_final, ensure_ascii=False),
        )
        second_wav = speak(path, SECOND)
        rate2, pcm2 = _pcm_from_wav(second_wav)
        pcm2 = _resample(pcm2, rate2, sample_rate)
        second_started = time.perf_counter()
        second, _partial2, _at2, _frame2, _sending2 = await _hear_utterance(
            connection,
            pcm2,
            sample_rate,
        )
        second_final = _first_final(second)
        if second_final is None:
            message = (
                "no second final arrived from a second utterance on the same socket"
            )
            raise SystemExit(message)
        print(
            "contract_hear_second_final",
            f"t={time.perf_counter() - second_started:.3f}s",
            json.dumps(second_final, ensure_ascii=False),
        )


def main() -> None:
    """Drive health, speak, and hear against the speech socket path."""
    _must_not_have_imported_speech()
    arguments = sys.argv[1:]
    measure = arguments[:1] == ["--measure"]
    if len(arguments) != 1 + measure:
        message = "usage: contract_client.py [--measure] SOCKET_PATH"
        raise SystemExit(message)
    run = _run_measure if measure else _run
    asyncio.run(run(Path(arguments[-1])))


if __name__ == "__main__":
    main()

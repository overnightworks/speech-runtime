"""Stage proof: start the service, drive health/speak/hear, then the contract client.

Takes /tmp/probe-stack.lock, checks the 1-minute load, runs one heavy step at a
time, and leaves no process behind. Model weights stay in the Hugging Face and
Piper caches.
"""

from __future__ import annotations

import fcntl
import json
import os
import socket
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from array import array
from pathlib import Path
from typing import TextIO
from urllib.parse import urlparse

from speech.chatterbox import CHATTERBOX_SAMPLE_RATE
from speech.config import CHATTERBOX_SPEAKING_MODEL
from speech.cuda_libs import cuda_library_dirs

SENTENCE = (
    "Die Kamera zeigt den Vortragenden, während im Hintergrund "
    "die nächste Folie automatisch vorbereitet wird."
)
ENGLISH = (
    "The camera shows the presenter while the next slide is "
    "automatically prepared in the background."
)
SECOND = "Die nächste Folie bitte."
LOCK_PATH = Path("/tmp/probe-stack.lock")
GERMAN_WAV_PATH = Path("/tmp/issue-83-voice/german.wav")
MEASURE_REPETITIONS = 5
LOAD_CEILING = 18.0
READY_TIMEOUT_SECONDS = 300.0
PIPER_FIRST_BYTE_LIMIT_S = 0.5
CHATTERBOX_FIRST_BYTE_LIMIT_S = 1.0
PIPER_CACHE = Path.home() / ".cache" / "piper"
HF_CACHE = Path.home() / ".cache" / "huggingface" / "hub"
CHATTERBOX_CACHE = HF_CACHE / "models--ResembleAI--chatterbox"
WHISPER_CACHE = HF_CACHE / "models--Systran--faster-whisper-large-v3"


def _load_1min() -> float:
    return float(Path("/proc/loadavg").read_text(encoding="utf-8").split()[0])


def _wait_for_load() -> None:
    while True:
        load = _load_1min()
        print(f"load_1min={load:.2f}")
        if load <= LOAD_CEILING:
            return
        print(f"load {load:.2f} is above {LOAD_CEILING}, waiting")
        time.sleep(15)


def _acquire_lock(lock: TextIO) -> None:
    while True:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(
                "waiting_for_lock",
                LOCK_PATH,
                f"load_1min={_load_1min():.2f}",
            )
            time.sleep(20)
            continue
        print("lock_acquired", LOCK_PATH)
        return


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _get_json(url: str) -> dict[str, object]:
    with urllib.request.urlopen(url, timeout=5) as response:
        return json.loads(response.read().decode("utf-8"))


def _wait_ready(base: str) -> dict[str, object]:
    deadline = time.monotonic() + READY_TIMEOUT_SECONDS
    last: dict[str, object] | None = None
    while time.monotonic() < deadline:
        try:
            last = _get_json(f"{base}/health")
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            time.sleep(0.5)
            continue
        speaking = last["speaking"]
        hearing = last["hearing"]
        if not isinstance(speaking, dict) or not isinstance(hearing, dict):
            time.sleep(0.5)
            continue
        print(
            "health",
            f"speaking_ready={speaking['ready']}",
            f"hearing_ready={hearing['ready']}",
            f"card_memory_mb={last['card_memory_mb']}",
        )
        if speaking["ready"] is True and hearing["ready"] is True:
            return last
        time.sleep(0.5)
    message = f"models not ready within {READY_TIMEOUT_SECONDS}s: {last}"
    raise SystemExit(message)


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


def _rms(pcm: bytes) -> float:
    if len(pcm) < 2:
        return 0.0
    samples = array("h")
    samples.frombytes(pcm[: len(pcm) - (len(pcm) % 2)])
    total = sum(sample * sample for sample in samples)
    return (total / len(samples)) ** 0.5 / 32768.0


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


def _dechunk(body: bytes) -> bytes:
    out = bytearray()
    position = 0
    while position < len(body):
        line_end = body.find(b"\r\n", position)
        if line_end < 0:
            break
        size_token = body[position:line_end].split(b";", 1)[0]
        size = int(size_token, 16)
        position = line_end + 2
        if size == 0:
            break
        out.extend(body[position : position + size])
        position += size + 2
    return bytes(out)


def _http_body_complete(header: bytes, body: bytes) -> bool:
    lowered = header.lower()
    if b"transfer-encoding: chunked" in lowered:
        return b"\r\n0\r\n\r\n" in body or body == b"0\r\n\r\n"
    for line in header.split(b"\r\n"):
        if line.lower().startswith(b"content-length:"):
            length = int(line.split(b":", 1)[1].strip())
            return len(body) >= length
    return False


def _speak_measured(
    base: str,
    text: str,
    language: str = "de",
) -> tuple[bytes, float, float]:
    parsed = urlparse(base)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 80
    payload = json.dumps({"text": text, "language": language}).encode("utf-8")
    request = (
        b"POST /speak HTTP/1.1\r\n"
        + f"Host: {host}:{port}\r\n".encode()
        + b"Content-Type: application/json\r\n"
        + f"Content-Length: {len(payload)}\r\n".encode()
        + b"Connection: close\r\n"
        + b"\r\n"
        + payload
    )
    sock = socket.create_connection((host, port), timeout=120)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    try:
        start = time.perf_counter()
        sock.sendall(request)
        raw = bytearray()
        first_byte_at = None
        while True:
            piece = sock.recv(4096)
            if not piece:
                break
            raw.extend(piece)
            header, separator, body = bytes(raw).partition(b"\r\n\r\n")
            if separator and body and first_byte_at is None:
                first_byte_at = time.perf_counter()
            if separator and _http_body_complete(header, body):
                break
        end = time.perf_counter()
    finally:
        sock.close()
    if first_byte_at is None:
        message = "speak returned no bytes on the socket"
        raise SystemExit(message)
    header, separator, body = bytes(raw).partition(b"\r\n\r\n")
    if not separator or not header.startswith(b"HTTP/1.1 200"):
        status = header.splitlines()[0] if header else b""
        message = f"speak HTTP status: {status!r}"
        raise SystemExit(message)
    if b"transfer-encoding: chunked" in header.lower():
        body = _dechunk(body)
    return body, first_byte_at - start, end - start


def _nvidia_memory() -> str:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total",
                "--format=csv,noheader",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        return f"unavailable: {error}"
    return completed.stdout.strip()


def _first_byte_limit(model: str) -> float:
    if model == CHATTERBOX_SPEAKING_MODEL:
        return CHATTERBOX_FIRST_BYTE_LIMIT_S
    return PIPER_FIRST_BYTE_LIMIT_S


def _require_speaking(health: dict[str, object], model: str) -> None:
    speaking = health["speaking"]
    if not isinstance(speaking, dict):
        message = f"health speaking is not an object: {speaking!r}"
        raise SystemExit(message)
    if speaking.get("model") != model:
        message = f"speaking.model is {speaking.get('model')!r}, expected {model!r}"
        raise SystemExit(message)
    if model == CHATTERBOX_SPEAKING_MODEL:
        if speaking.get("streams") is not True:
            message = f"Chatterbox must report speaking.streams=true: {speaking!r}"
            raise SystemExit(message)
        if speaking.get("sample_rate") != CHATTERBOX_SAMPLE_RATE:
            message = (
                f"Chatterbox must report sample_rate={CHATTERBOX_SAMPLE_RATE}: "
                f"{speaking!r}"
            )
            raise SystemExit(message)
        return
    if speaking.get("streams") is not False:
        message = f"Piper must report speaking.streams=false: {speaking!r}"
        raise SystemExit(message)


def _prove_one_speak(
    base: str,
    text: str,
    language: str,
    *,
    limit: float,
    label: str,
) -> tuple[bytes, float, float]:
    wav, ttfa, total = _speak_measured(base, text, language=language)
    rate, pcm = _pcm_from_wav(wav)
    duration = (len(pcm) / 2) / rate
    rtf = duration / total if total else 0.0
    print(
        label,
        f"first_byte_s={ttfa:.3f}",
        f"total_s={total:.3f}",
        f"limit_s={limit:.1f}",
        f"duration_s={duration:.3f}",
        f"rtf={rtf:.2f}",
        f"rms={_rms(pcm):.4f}",
        f"rate={rate}",
        f"bytes={len(wav)}",
    )
    if _rms(pcm) < 0.01:
        message = f"{label} produced silence"
        raise SystemExit(message)
    if duration < 1.0:
        message = f"{label} produced too little audio to be a sentence"
        raise SystemExit(message)
    if ttfa > limit:
        message = f"{label} first byte {ttfa:.3f}s exceeded the {limit:.1f}s limit"
        raise SystemExit(message)
    return wav, ttfa, rtf


def _prove_speak_medians(
    base: str,
    text: str,
    language: str,
    *,
    limit: float,
    label: str,
) -> tuple[bytes, float, float]:
    """Speak the same sentence MEASURE_REPETITIONS times; print and return medians."""
    wavs: list[bytes] = []
    ttfas: list[float] = []
    rtfs: list[float] = []
    for rep in range(1, MEASURE_REPETITIONS + 1):
        wav, ttfa, rtf = _prove_one_speak(
            base,
            text,
            language,
            limit=limit,
            label=f"{label}_rep{rep}",
        )
        wavs.append(wav)
        ttfas.append(ttfa)
        rtfs.append(rtf)
    median_ttfa = statistics.median(ttfas)
    median_rtf = statistics.median(rtfs)
    print(
        f"{label}_median",
        f"first_byte_s={median_ttfa:.3f}",
        f"rtf={median_rtf:.2f}",
        f"limit_s={limit:.1f}",
        f"n={MEASURE_REPETITIONS}",
    )
    return wavs[-1], median_ttfa, median_rtf


def _prove_speak(base: str, model: str) -> bytes:
    """Speak German and English MEASURE_REPETITIONS times each.

    `_prove_one_speak` fails the run the moment any single repetition, in
    either language, misses `model`'s first-byte limit; a passing run is
    the proof that every repetition held, not just the medians.
    """
    limit = _first_byte_limit(model)
    german, _ttfa_de, _rtf_de = _prove_speak_medians(
        base,
        SENTENCE,
        "de",
        limit=limit,
        label="speak_wire_de",
    )
    _prove_speak_medians(base, ENGLISH, "en", limit=limit, label="speak_wire_en")
    return german


def _prove_concurrent_speak(base: str, model: str) -> None:
    limit = _first_byte_limit(model)
    results: dict[str, tuple[bytes, float, float]] = {}
    errors: list[str] = []

    def run(key: str, text: str) -> None:
        try:
            results[key] = _speak_measured(base, text, language="de")
        except Exception as error:
            errors.append(f"{key}: {error}")

    first = threading.Thread(target=run, args=("first", SENTENCE), daemon=True)
    second = threading.Thread(target=run, args=("second", SECOND), daemon=True)
    first.start()
    second.start()
    first.join()
    second.join()
    if errors:
        raise SystemExit("concurrent speak failed: " + "; ".join(errors))
    if set(results) != {"first", "second"}:
        message = f"concurrent speak missing a response: {sorted(results)}"
        raise SystemExit(message)
    for key, (wav, ttfa, total) in results.items():
        rate, pcm = _pcm_from_wav(wav)
        duration = (len(pcm) / 2) / rate
        print(
            f"speak_concurrent_{key}",
            f"first_byte_s={ttfa:.3f}",
            f"total_s={total:.3f}",
            f"duration_s={duration:.3f}",
            f"rms={_rms(pcm):.4f}",
            f"rate={rate}",
        )
        if _rms(pcm) < 0.01:
            message = f"concurrent {key} produced silence"
            raise SystemExit(message)
        if duration < 0.4:
            message = f"concurrent {key} produced too little audio"
            raise SystemExit(message)
        if ttfa > limit + 8.0:
            message = (
                f"concurrent {key} first byte {ttfa:.3f}s; "
                f"serialized limit is {limit}s plus 8s wait"
            )
            raise SystemExit(message)
    if results["first"][0] == results["second"][0]:
        raise SystemExit("concurrent speak returned identical bodies")


def _prove_contract(project: Path, base: str, env: dict[str, str]) -> None:
    client = project / "scripts" / "contract_client.py"
    print("contract_client_start")
    completed = subprocess.run(
        [sys.executable, str(client), base],
        check=False,
        cwd="/tmp",
        env=env,
        capture_output=True,
        text=True,
    )
    sys.stdout.write(completed.stdout)
    sys.stderr.write(completed.stderr)
    if completed.returncode != 0:
        message = f"contract client failed: {completed.returncode}"
        raise SystemExit(message)
    for marker in (
        "contract_hear_partial",
        "contract_hear_final",
        "contract_hear_second_final",
    ):
        if marker not in completed.stdout:
            message = f"contract client did not report {marker}"
            raise SystemExit(message)


def main() -> None:
    """Run the stage proof under the probe-stack lock."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    _wait_for_load()
    LOCK_PATH.touch()
    with LOCK_PATH.open("a", encoding="utf-8") as lock:
        print("waiting_for_lock", LOCK_PATH)
        _acquire_lock(lock)
        print("gpu_before", _nvidia_memory())
        port = _free_port()
        env = os.environ.copy()
        speaking_model = os.environ.get(
            "SPEECH_SPEAKING_MODEL",
            CHATTERBOX_SPEAKING_MODEL,
        )
        env["SPEECH_HOST"] = "127.0.0.1"
        env["SPEECH_PORT"] = str(port)
        env["SPEECH_DEVICE"] = "cuda"
        env["SPEECH_SPEAKING_MODEL"] = speaking_model
        env["SPEECH_HEARING_MODEL"] = "Systran/faster-whisper-large-v3"
        env["SPEECH_DEBUG"] = "false"
        env.setdefault("HF_HUB_OFFLINE", "1")
        env.setdefault("TRANSFORMERS_OFFLINE", "1")
        cuda_dirs = ":".join(str(path) for path in cuda_library_dirs())
        if cuda_dirs:
            existing = env.get("LD_LIBRARY_PATH", "")
            env["LD_LIBRARY_PATH"] = (
                f"{cuda_dirs}:{existing}" if existing else cuda_dirs
            )
        project = Path(__file__).resolve().parents[1]
        server = subprocess.Popen(
            [sys.executable, "-m", "speech"],
            cwd=project,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        base = f"http://127.0.0.1:{port}"
        try:
            health = _wait_ready(base)
            print("health_ready", json.dumps(health, sort_keys=True))
            _require_speaking(health, speaking_model)
            print("gpu_both_resident", _nvidia_memory())
            print("whisper_cache", WHISPER_CACHE)
            print("chatterbox_cache", CHATTERBOX_CACHE)
            print("piper_cache", PIPER_CACHE)
            german = _prove_speak(base, speaking_model)
            GERMAN_WAV_PATH.parent.mkdir(parents=True, exist_ok=True)
            GERMAN_WAV_PATH.write_bytes(german)
            print("german_wav", GERMAN_WAV_PATH)
            _prove_concurrent_speak(base, speaking_model)
            _prove_contract(project, base, env)
            print("gpu_after", _nvidia_memory())
            print("caches_kept")
            print("  huggingface:", HF_CACHE)
            print("  chatterbox:", CHATTERBOX_CACHE)
            print("  piper:", PIPER_CACHE)
            print("PROOF_OK")
        finally:
            server.terminate()
            try:
                server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait(timeout=5)
            if server.stdout is not None:
                leftover = server.stdout.read()
                if leftover:
                    print("server_log_tail:")
                    print(leftover[-4000:])
            print("server_exit", server.returncode)
            print("gpu_released", _nvidia_memory())
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
            print("lock_released")


if __name__ == "__main__":
    main()

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
import subprocess
import sys
import time
import urllib.error
import urllib.request
from array import array
from pathlib import Path
from urllib.parse import urlparse

from speech.cuda_libs import cuda_library_dirs

SENTENCE = (
    "Die Kamera zeigt den Vortragenden, während im Hintergrund "
    "die nächste Folie automatisch vorbereitet wird."
)
SECOND = "Die nächste Folie bitte."
LOCK_PATH = Path("/tmp/probe-stack.lock")
LOAD_CEILING = 18.0
READY_TIMEOUT_SECONDS = 180.0
WIRE_FIRST_BYTE_LIMIT_S = 0.5
PIPER_CACHE = Path.home() / ".cache" / "piper"
HF_CACHE = Path.home() / ".cache" / "huggingface" / "hub"
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


def _speak_measured(base: str, text: str) -> tuple[bytes, float, float]:
    parsed = urlparse(base)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 80
    payload = json.dumps({"text": text, "language": "de"}).encode("utf-8")
    request = (
        b"POST /speak HTTP/1.1\r\n"
        + f"Host: {host}:{port}\r\n".encode()
        + b"Content-Type: application/json\r\n"
        + f"Content-Length: {len(payload)}\r\n".encode()
        + b"Connection: close\r\n"
        + b"\r\n"
        + payload
    )
    sock = socket.create_connection((host, port), timeout=60)
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


def _require_piper_batch(health: dict[str, object]) -> None:
    speaking = health["speaking"]
    if not isinstance(speaking, dict) or speaking.get("streams") is not False:
        message = f"Piper must report speaking.streams=false: {speaking!r}"
        raise SystemExit(message)


def _prove_speak(base: str) -> None:
    wav, ttfa, total = _speak_measured(base, SENTENCE)
    rate, pcm = _pcm_from_wav(wav)
    duration = (len(pcm) / 2) / rate
    print(
        "speak_wire",
        f"first_byte_s={ttfa:.3f}",
        f"total_s={total:.3f}",
        f"limit_s={WIRE_FIRST_BYTE_LIMIT_S:.1f}",
        f"duration_s={duration:.3f}",
        f"rms={_rms(pcm):.4f}",
        f"rate={rate}",
        f"bytes={len(wav)}",
    )
    if ttfa > WIRE_FIRST_BYTE_LIMIT_S:
        message = (
            f"first byte on the socket took {ttfa:.3f}s; "
            f"limit is {WIRE_FIRST_BYTE_LIMIT_S}s"
        )
        raise SystemExit(message)
    if _rms(pcm) < 0.01:
        raise SystemExit("speak produced silence")
    if duration < 1.0:
        raise SystemExit("speak produced too little audio to be a sentence")


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
    _wait_for_load()
    LOCK_PATH.touch()
    with LOCK_PATH.open("a", encoding="utf-8") as lock:
        print("waiting_for_lock", LOCK_PATH)
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        print("lock_acquired", LOCK_PATH)
        print("gpu_before", _nvidia_memory())
        port = _free_port()
        env = os.environ.copy()
        env["SPEECH_HOST"] = "127.0.0.1"
        env["SPEECH_PORT"] = str(port)
        env["SPEECH_DEVICE"] = "cuda"
        env["SPEECH_SPEAKING_MODEL"] = "de_DE-thorsten-medium"
        env["SPEECH_HEARING_MODEL"] = "Systran/faster-whisper-large-v3"
        env["SPEECH_DEBUG"] = "false"
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
            _require_piper_batch(health)
            print("gpu_both_resident", _nvidia_memory())
            print("whisper_cache", WHISPER_CACHE)
            print("piper_cache", PIPER_CACHE)
            _prove_speak(base)
            _prove_contract(project, base, env)
            print("gpu_after", _nvidia_memory())
            print("caches_kept")
            print("  huggingface:", HF_CACHE)
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

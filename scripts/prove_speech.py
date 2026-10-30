"""Stage proof: start the service, drive health/speak/hear, then the contract client.

Takes /tmp/probe-stack.lock, checks the 1-minute load, runs one heavy step at a
time, and leaves no process behind. Model weights stay in the Hugging Face and
Piper caches.
"""

# ruff: noqa: T201 - the proof prints its stage lines as the command line output

from __future__ import annotations

import argparse
import contextlib
import fcntl
import json
import logging
import os
import shutil
import socket
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from array import array
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, TextIO

import httpx2
from speech_runtime_provider_contract import CHATTERBOX_SAMPLE_RATE

from speech_runtime_service.config import CHATTERBOX_SPEAKING_MODEL
from speech_runtime_service.cuda_libs import cuda_library_dirs
from speech_runtime_service.pcm import BYTES_PER_SAMPLE

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

SENTENCE = (
    "Die Kamera zeigt den Vortragenden, während im Hintergrund "
    "die nächste Folie automatisch vorbereitet wird."
)
ENGLISH = (
    "The camera shows the presenter while the next slide is "
    "automatically prepared in the background."
)
SECOND = "Die nächste Folie bitte."
LOCK_PATH = Path("/tmp/probe-stack.lock")  # noqa: S108 - the one machine-wide probe lock must keep this exact path whatever TMPDIR says
MEASURE_REPETITIONS = 5
MEASURE_FLAG = {False: [], True: ["--measure"]}
HEARD_MARKERS = (
    "contract_hear_partial",
    "contract_hear_final",
    "contract_hear_second_final",
)
CONTRACT_MARKERS = {
    False: HEARD_MARKERS,
    True: ("measure_rep1", "measure_summary"),
}
LOGGER = logging.getLogger(__name__)
SILENCE_RMS = 0.01
MIN_SPEECH_SECONDS = 0.4
LOAD_CEILING = 18.0
READY_TIMEOUT_SECONDS = 300.0
PIPER_FIRST_BYTE_LIMIT_S = 0.5
CHATTERBOX_FIRST_BYTE_LIMIT_S = 1.0
PIPER_CACHE = Path.home() / ".cache" / "piper"
PROVIDER_ROOT = Path(__file__).resolve().parents[1] / "providers"
HF_CACHE = Path.home() / ".cache" / "huggingface" / "hub"
CHATTERBOX_CACHE = HF_CACHE / "models--ResembleAI--chatterbox"
WHISPER_CACHE = HF_CACHE / "models--Systran--faster-whisper-large-v3"
DEFAULT_HEARING_MODEL = "Systran/faster-whisper-large-v3"


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
    print("waiting_for_lock", LOCK_PATH)
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
    print("lock_acquired", LOCK_PATH)


def get_json(socket_path: Path, target: str) -> dict[str, object]:
    """GET a path over the speech socket and decode the JSON body."""
    transport = httpx2.HTTPTransport(uds=str(socket_path))
    with httpx2.Client(transport=transport, base_url="http://speech") as client:
        return client.get(target, timeout=5).json()


def _wait_ready(socket_path: Path) -> dict[str, object]:
    deadline = time.monotonic() + READY_TIMEOUT_SECONDS
    last: dict[str, object] | None = None
    while time.monotonic() < deadline:
        try:
            last = get_json(socket_path, "/health")
        except httpx2.HTTPError, json.JSONDecodeError:
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
    if len(pcm) < BYTES_PER_SAMPLE:
        return 0.0
    samples = array("h")
    samples.frombytes(pcm[: len(pcm) - (len(pcm) % BYTES_PER_SAMPLE)])
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


def speak_measured(
    socket_path: Path,
    text: str,
    language: str = "de",
) -> tuple[bytes, float, float]:
    """Stream one speak request; return the WAV, time to first audio and total time."""
    payload = json.dumps({"text": text, "language": language}).encode("utf-8")
    request = (
        b"POST /speak HTTP/1.1\r\n"
        b"Host: speech\r\n"
        b"Content-Type: application/json\r\n"
        + f"Content-Length: {len(payload)}\r\n".encode()
        + b"Connection: close\r\n"
        + b"\r\n"
        + payload
    )
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(120)
    sock.connect(str(socket_path))
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


def _query_nvidia(*arguments: str) -> str:
    nvidia = shutil.which("nvidia-smi")
    if nvidia is None:
        return "unavailable: nvidia-smi not found"
    try:
        completed = subprocess.run(  # noqa: S603 - argv is the resolved nvidia-smi and fixed arguments with no shell
            [nvidia, *arguments],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        return f"unavailable: {error}"
    return completed.stdout.strip()


def _nvidia_memory() -> str:
    return _query_nvidia(
        "--query-gpu=memory.used,memory.total", "--format=csv,noheader"
    )


def _nvidia_compute_pids() -> str:
    pids = _query_nvidia("--query-compute-apps=pid", "--format=csv,noheader")
    if pids.startswith("unavailable:"):
        return pids
    return " ".join(pids.split()) or "none"


def peak_rss_mib(status: str) -> float:
    """Read `VmHWM`, the process's peak resident set, from /proc/<pid>/status."""
    for line in status.splitlines():
        if line.startswith("VmHWM:"):
            return int(line.split()[1]) / 1024
    message = "no VmHWM in the process status"
    raise SystemExit(message)


@dataclass(frozen=True)
class _RunningSpeech:
    process: subprocess.Popen[str]
    health: dict[str, object]
    ready_seconds: float
    heard: list[str]

    def peak_rss_mib(self) -> float:
        status = Path(f"/proc/{self.process.pid}/status")
        return peak_rss_mib(status.read_text(encoding="utf-8"))


@contextlib.contextmanager
def _speech_server(
    project: Path, env: dict[str, str], socket_path: Path
) -> Iterator[_RunningSpeech]:
    """Start speech, wait until both models are ready, and stop it on the way out."""
    server = subprocess.Popen(
        [sys.executable, "-m", "speech_runtime_service"],
        cwd=project,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    heard: list[str] = []
    proof_failed = True
    try:
        started = time.monotonic()
        health = _wait_ready(socket_path)
        yield _RunningSpeech(server, health, time.monotonic() - started, heard)
        proof_failed = False
    finally:
        server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait(timeout=5)
        log = server.stdout.read() if server.stdout is not None else ""
        if log:
            print("server_log_tail:")
            print(log[-4000:])
        print("server_exit", server.returncode)
        texts = [SENTENCE, ENGLISH, SECOND, *heard]
        if proof_failed:
            _report_log_text(log, texts)
        else:
            _require_log_without_text(log, texts)


def heard_texts(contract_output: str) -> list[str]:
    """The texts the contract client printed as heard partials and finals."""
    return [
        str(json.loads(line[line.index("{") :])["text"])
        for line in contract_output.splitlines()
        if line.startswith(HEARD_MARKERS)
    ]


def spoken_or_heard_in_log(log: str, texts: Iterable[str]) -> list[str]:
    """The non-blank texts that the service log holds."""
    return [text for text in texts if text.strip() and text in log]


def _leak_message(log: str, texts: Iterable[str]) -> str | None:
    leaked = spoken_or_heard_in_log(log, texts)
    if leaked:
        return f"the service log holds {len(leaked)} spoken or heard text(s)"
    return None


def _report_log_text(log: str, texts: Iterable[str]) -> None:
    """Print a leak without hiding the failure that is already propagating."""
    message = _leak_message(log, texts)
    if message:
        print(message)


def _require_log_without_text(log: str, texts: Iterable[str]) -> None:
    message = _leak_message(log, texts)
    if message:
        raise SystemExit(message)


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
    socket_path: Path,
    text: str,
    language: str,
    *,
    limit: float,
    label: str,
) -> tuple[bytes, float, float]:
    wav, ttfa, total = speak_measured(socket_path, text, language=language)
    rate, pcm = _pcm_from_wav(wav)
    duration = (len(pcm) / BYTES_PER_SAMPLE) / rate
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
    if _rms(pcm) < SILENCE_RMS:
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
    socket_path: Path,
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
            socket_path,
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


def _prove_speak(socket_path: Path, model: str) -> bytes:
    """Speak German and English MEASURE_REPETITIONS times each.

    `_prove_one_speak` fails the run the moment any single repetition, in
    either language, misses `model`'s first-byte limit; a passing run is
    the proof that every repetition held, not just the medians.
    """
    limit = _first_byte_limit(model)
    german, _ttfa_de, _rtf_de = _prove_speak_medians(
        socket_path,
        SENTENCE,
        "de",
        limit=limit,
        label="speak_wire_de",
    )
    _prove_speak_medians(socket_path, ENGLISH, "en", limit=limit, label="speak_wire_en")
    return german


def _prove_concurrent_speak(socket_path: Path, model: str) -> None:
    limit = _first_byte_limit(model)
    results: dict[str, tuple[bytes, float, float]] = {}
    errors: list[str] = []

    def run(key: str, text: str) -> None:
        try:
            results[key] = speak_measured(socket_path, text, language="de")
        except Exception as error:
            LOGGER.exception("concurrent speak %s failed", key)
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
        duration = (len(pcm) / BYTES_PER_SAMPLE) / rate
        print(
            f"speak_concurrent_{key}",
            f"first_byte_s={ttfa:.3f}",
            f"total_s={total:.3f}",
            f"duration_s={duration:.3f}",
            f"rms={_rms(pcm):.4f}",
            f"rate={rate}",
        )
        if _rms(pcm) < SILENCE_RMS:
            message = f"concurrent {key} produced silence"
            raise SystemExit(message)
        if duration < MIN_SPEECH_SECONDS:
            message = f"concurrent {key} produced too little audio"
            raise SystemExit(message)
        if ttfa > limit + 8.0:
            message = (
                f"concurrent {key} first byte {ttfa:.3f}s; "
                f"serialized limit is {limit}s plus 8s wait"
            )
            raise SystemExit(message)
    if results["first"][0] == results["second"][0]:
        message = "concurrent speak returned identical bodies"
        raise SystemExit(message)


def _prove_contract(
    socket_path: Path,
    private_directory: Path,
    env: dict[str, str],
    heard: list[str],
    *,
    measure: bool,
) -> None:
    client = Path(__file__).with_name("contract_client.py")
    print("contract_client_start")
    completed = subprocess.run(  # noqa: S603 - argv is this interpreter and this repository's client script with no shell
        [sys.executable, str(client), *MEASURE_FLAG[measure], str(socket_path)],
        check=False,
        cwd=private_directory,
        env=env,
        capture_output=True,
        text=True,
    )
    sys.stdout.write(completed.stdout)
    sys.stderr.write(completed.stderr)
    if completed.returncode != 0:
        message = f"contract client failed: {completed.returncode}"
        raise SystemExit(message)
    for marker in CONTRACT_MARKERS[measure]:
        if marker not in completed.stdout:
            message = f"contract client did not report {marker}"
            raise SystemExit(message)
    heard.extend(heard_texts(completed.stdout))


def speech_environment(private_directory: Path, speaking_model: str) -> dict[str, str]:
    """Return the process environment pointing at the private directory and model."""
    env = os.environ.copy()
    env["SPEECH_PRIVATE_DIRECTORY"] = str(private_directory)
    env["PRESENTATOR_RUNTIME_UID"] = str(os.geteuid())
    env.setdefault("SPEECH_PROVIDER_ROOT", str(PROVIDER_ROOT))
    env["SPEECH_DEVICE"] = os.environ.get("SPEECH_DEVICE", "cuda")
    if env["SPEECH_DEVICE"] == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""
    state_directory = private_directory / "state"
    state_directory.mkdir(mode=0o700, exist_ok=True)
    env["SPEECH_STATE_DIRECTORY"] = str(state_directory)
    env["SPEECH_SPEAKING_MODEL"] = speaking_model
    env["SPEECH_HEARING_MODEL"] = os.environ.get(
        "SPEECH_HEARING_MODEL", DEFAULT_HEARING_MODEL
    )
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("TRANSFORMERS_OFFLINE", "1")
    cuda_dirs = ":".join(str(path) for path in cuda_library_dirs())
    if cuda_dirs:
        existing = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = f"{cuda_dirs}:{existing}" if existing else cuda_dirs
    return env


def parse_arguments(arguments: list[str]) -> argparse.Namespace:
    """Read the command line before anything starts; `--help` and typos exit here."""
    parser = argparse.ArgumentParser(
        description="Stage proof of the speech service under the probe-stack lock.",
        allow_abbrev=False,
    )
    parser.add_argument(
        "--measure",
        action="store_true",
        help="run the CPU hearing measurement instead of the contract proof",
    )
    return parser.parse_args(arguments)


def main() -> None:
    """Run the stage proof under the probe-stack lock."""
    options = parse_arguments(sys.argv[1:])
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    LOCK_PATH.touch()
    with LOCK_PATH.open("a", encoding="utf-8") as lock:
        _acquire_lock(lock)
        _wait_for_load()
        print("gpu_before", _nvidia_memory())
        private_directory = Path(tempfile.mkdtemp(prefix="speech-"))
        socket_path = private_directory / "speech.sock"
        speaking_model = os.environ.get(
            "SPEECH_SPEAKING_MODEL",
            CHATTERBOX_SPEAKING_MODEL,
        )
        project = Path(__file__).resolve().parents[1]
        try:
            env = speech_environment(private_directory, speaking_model)
            with _speech_server(project, env, socket_path) as running:
                print("health_ready", json.dumps(running.health, sort_keys=True))
                print("load_to_ready_s", f"{running.ready_seconds:.1f}")
                print(
                    "gpu_compute_pids",
                    _nvidia_compute_pids(),
                    f"speech_pid={running.process.pid}",
                )
                _require_speaking(running.health, speaking_model)
                print("gpu_both_resident", _nvidia_memory())
                print("whisper_cache", WHISPER_CACHE)
                print("chatterbox_cache", CHATTERBOX_CACHE)
                print("piper_cache", PIPER_CACHE)
                german = _prove_speak(socket_path, speaking_model)
                german_wav = (
                    Path(tempfile.mkdtemp(prefix="speech-proof-")) / "german.wav"
                )
                german_wav.write_bytes(german)
                print("german_wav", german_wav)
                _prove_concurrent_speak(socket_path, speaking_model)
                _prove_contract(
                    socket_path,
                    private_directory,
                    env,
                    running.heard,
                    measure=options.measure,
                )
                print("peak_rss_mib", f"{running.peak_rss_mib():.0f}")
            print("gpu_after", _nvidia_memory())
            print("caches_kept")
            print("  huggingface:", HF_CACHE)
            print("  chatterbox:", CHATTERBOX_CACHE)
            print("  piper:", PIPER_CACHE)
            print("PROOF_OK")
        finally:
            shutil.rmtree(private_directory)
            print("gpu_released", _nvidia_memory())
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
            print("lock_released")


if __name__ == "__main__":
    main()

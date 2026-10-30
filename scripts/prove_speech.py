"""Stage proof: start the service, drive health/speak/hear, then the contract client.

Takes /tmp/probe-stack.lock, checks the 1-minute load, runs one heavy step at a
time, and leaves no process behind. Model weights stay in the Hugging Face and
Piper caches.
"""

# ruff: noqa: T201 - the proof prints its stage lines as the command line output

from __future__ import annotations

import argparse
import contextlib
import http.client
import json
import logging
import os
import shutil
import signal
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
from typing import TYPE_CHECKING

import httpx2
from speech_runtime_provider_contract import CHATTERBOX_SAMPLE_RATE

from speech_runtime_client.wire import SPEECH_PATH, speech_request_body
from speech_runtime_service.config import CHATTERBOX_SPEAKING_MODEL
from speech_runtime_service.cuda_libs import cuda_library_dirs
from speech_runtime_service.pcm import BYTES_PER_SAMPLE
from speech_runtime_service.service import voice_for_model

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator, Mapping
    from types import FrameType

SENTENCE = (
    "Die Kamera zeigt den Vortragenden, während im Hintergrund "
    "die nächste Folie automatisch vorbereitet wird."
)
SECOND = "Die nächste Folie bitte."
LOCK_PATH = Path("/tmp/probe-stack.lock")  # noqa: S108 - the one machine-wide probe lock must keep this exact path whatever TMPDIR says
LOCK_HELD_ENV = "SPEECH_PROOF_HOLDS_PROBE_LOCK"
LOCK_DIRECTORY_PREFIX = "speech-lock-"
LOCK_WAIT_SECONDS = 1200
FLOCK_TIMED_OUT_STATUS = 1
# The cap and the kill grace are the numbers this script owns inside the fleet
# lock form; together they stay within 1830 s. The grace covers the teardown
# `speech_server` runs after TERM: terminate, wait up to 10 s, then kill and wait
# up to 5 s, so 15 s at worst, with the providers stopped through the server.
# Measured on the CPU (SPEECH_SPEAKING_MODEL=de_DE-thorsten-medium, hearing
# faster-whisper-medium, OMP_NUM_THREADS=4, 30.10.2026): the server exits 0.6 s
# after the proof body ends; the full proof holds the lock for 92 s and
# `--measure` for 84 s, so the cap leaves them far more room than they use.
CAP_SECONDS = 1800
KILL_GRACE_SECONDS = 30
MEASURE_REPETITIONS = 5
SPEAK_TIMEOUT_SECONDS = 120
CONCURRENT_EXTRA_WAIT_SECONDS = 8.0
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


def wait_for_load(
    read_load: Callable[[], float] = _load_1min,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    """Wait for the 1-minute load to fall to the ceiling, inside the lock's cap."""
    while True:
        load = read_load()
        print(f"load_1min={load:.2f}")
        if load <= LOAD_CEILING:
            return
        print(f"load {load:.2f} is above {LOAD_CEILING}, waiting")
        sleep(15)


def fleet_command(arguments: list[str]) -> list[str]:
    """This script under the fleet's one form for taking the probe lock."""
    return [
        "flock",
        "-o",
        "-w",
        str(LOCK_WAIT_SECONDS),
        str(LOCK_PATH),
        "timeout",
        "-k",
        str(KILL_GRACE_SECONDS),
        str(CAP_SECONDS),
        sys.executable,
        str(Path(__file__).resolve()),
        *arguments,
    ]


def run_under_lock(
    arguments: list[str],
    run: Callable[..., subprocess.CompletedProcess[bytes]] = subprocess.run,
) -> int:
    """Run the proof under the lock; a lock still busy after the wait fails loudly."""
    with tempfile.TemporaryDirectory(prefix=LOCK_DIRECTORY_PREFIX) as directory:
        token = Path(directory) / "acquired"
        environment = {**os.environ, LOCK_HELD_ENV: str(token)}
        status = run(fleet_command(arguments), env=environment, check=False).returncode
        lock_was_taken = token.exists()
    if status == FLOCK_TIMED_OUT_STATUS and not lock_was_taken:
        message = (
            f"probe lock {LOCK_PATH} is busy: still held after {LOCK_WAIT_SECONDS} s,"
            " so the proof did not start"
        )
        raise SystemExit(message)
    return status


def announce_lock_held(environment: Mapping[str, str]) -> None:
    """Tell the process that waited for the lock that its child holds it."""
    token = Path(environment[LOCK_HELD_ENV])
    if not token.parent.name.startswith(LOCK_DIRECTORY_PREFIX):
        message = (
            f"{LOCK_HELD_ENV} must name a file in a {LOCK_DIRECTORY_PREFIX} directory"
        )
        raise SystemExit(message)
    token.touch()
    print("lock_acquired", LOCK_PATH)


def prove_under_lock(environment: Mapping[str, str], prove: Callable[[], None]) -> None:
    """Announce the held lock, turn TERM into an exit, and run the proof body."""
    announce_lock_held(environment)
    exit_on_terminate()
    prove()


def _exit_on_signal(received: int, _frame: FrameType | None) -> None:
    raise SystemExit(128 + received)


def exit_on_terminate() -> None:
    """Turn the cap's TERM into an ordinary exit, so the teardown `finally` runs."""
    signal.signal(signal.SIGTERM, _exit_on_signal)


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


def speak_measured(
    socket_path: Path,
    text: str,
    voice: str,
) -> tuple[bytes, float, float]:
    """Speak one request; return the whole WAV, time to first audio and total time.

    The headers leave before synthesis, so the first audio is the first body
    byte. A body the service aborted raises `IncompleteRead` in `http.client`,
    and the proof fails rather than keep the partial WAV.
    """
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(SPEAK_TIMEOUT_SECONDS)
    sock.connect(str(socket_path))
    connection = http.client.HTTPConnection("speech", timeout=SPEAK_TIMEOUT_SECONDS)
    connection.sock = sock
    try:
        start = time.perf_counter()
        connection.request(
            "POST",
            SPEECH_PATH,
            speech_request_body(text, voice),
            {"Content-Type": "application/json"},
        )
        response = connection.getresponse()
        if response.status != http.client.OK:
            message = f"speak HTTP status: {response.status}"
            raise SystemExit(message)
        first = response.read(1)
        first_byte_at = time.perf_counter()
        body = first + response.read()
        end = time.perf_counter()
    except http.client.IncompleteRead:
        message = "speak body ended before it was complete"
        raise SystemExit(message) from None
    finally:
        connection.close()
    if not first:
        message = "speak returned no bytes on the socket"
        raise SystemExit(message)
    return body, first_byte_at - start, end - start


def _voice_id(model: str) -> str:
    voice = voice_for_model(model)
    if voice is None:
        message = f"{model} is not a baseline speaking model"
        raise SystemExit(message)
    return voice.value


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
def speech_server(
    project: Path,
    env: dict[str, str],
    socket_path: Path,
    start: Callable[..., subprocess.Popen[str]] = subprocess.Popen,
    wait_ready: Callable[[Path], dict[str, object]] = _wait_ready,
) -> Iterator[_RunningSpeech]:
    """Start speech, wait until both models are ready, and stop it on the way out."""
    server = start(
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
        health = wait_ready(socket_path)
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
        texts = [SENTENCE, SECOND, *heard]
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
    voice: str,
    *,
    limit: float,
    label: str,
) -> tuple[bytes, float, float]:
    wav, ttfa, total = speak_measured(socket_path, text, voice)
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
    voice: str,
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
            voice,
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
    """Speak German MEASURE_REPETITIONS times.

    `_prove_one_speak` fails the run the moment any single repetition misses
    `model`'s first-byte limit; a passing run is the proof that every
    repetition held, not just the median.
    """
    german, _ttfa, _rtf = _prove_speak_medians(
        socket_path,
        SENTENCE,
        _voice_id(model),
        limit=_first_byte_limit(model),
        label="speak_wire_de",
    )
    return german


def _prove_concurrent_speak(socket_path: Path, model: str) -> None:
    limit = _first_byte_limit(model)
    voice = _voice_id(model)
    results: dict[str, tuple[bytes, float, float]] = {}
    errors: list[str] = []

    def run(key: str, text: str) -> None:
        try:
            results[key] = speak_measured(socket_path, text, voice)
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
        if ttfa > limit + CONCURRENT_EXTRA_WAIT_SECONDS:
            message = (
                f"concurrent {key} first byte {ttfa:.3f}s; "
                f"serialized limit is {limit}s plus "
                f"{CONCURRENT_EXTRA_WAIT_SECONDS}s wait"
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
        [
            sys.executable,
            str(client),
            *MEASURE_FLAG[measure],
            str(socket_path),
            _voice_id(env["SPEECH_SPEAKING_MODEL"]),
        ],
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
    env["SPEECH_BUSY_WAIT_SECONDS"] = str(
        _first_byte_limit(speaking_model) + CONCURRENT_EXTRA_WAIT_SECONDS
    )
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


def _prove(options: argparse.Namespace) -> None:
    wait_for_load()
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
        with speech_server(project, env, socket_path) as running:
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
            german_wav = Path(tempfile.mkdtemp(prefix="speech-proof-")) / "german.wav"
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


def main() -> None:
    """Run the stage proof, taking the probe-stack lock through the fleet form."""
    options = parse_arguments(sys.argv[1:])
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    if LOCK_HELD_ENV not in os.environ:
        print("waiting_for_lock", LOCK_PATH)
        raise SystemExit(run_under_lock(sys.argv[1:]))
    prove_under_lock(os.environ, lambda: _prove(options))


if __name__ == "__main__":
    main()

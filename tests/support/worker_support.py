"""A fake provider worker and the readers of what it observed."""

import ast
import os
import sys
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

_WORKER = r"""#!{python}
import argparse
import os
import signal
import struct
import time
from pathlib import Path

protocol_fd = os.dup(1)
null = os.open(os.devnull, os.O_WRONLY)
os.dup2(null, 1)
os.close(null)
print("provider print noise", flush=True)
os.write(1, b"native provider noise\n")

from speech_runtime_provider_contract import (
    MAX_FRAME_LENGTH, Frame, FrameKind, ProviderFailure, read_frame, write_frame
)

parser = argparse.ArgumentParser()
parser.add_argument("--expected-parent-pid", required=True)
parser.add_argument("--device", required=True)
parser.add_argument("--cache", required=True)
if {provider!r} in ("qwen", "magpie"):
    parser.add_argument("--speaker", required=True)
arguments = parser.parse_args()
cache = Path(arguments.cache)
cache.mkdir(parents=True, exist_ok=True)
(cache / "worker-observed").write_text(
    repr(os.getcwd()) + "\n" + repr(os.environ.copy()) + "\n" +
    repr(os.getpid()) + "\n" + repr(arguments.__dict__), encoding="utf-8"
)
stream = os.fdopen(protocol_fd, "wb", buffering=0)
behavior = {behavior!r}
if behavior == "startup-exit":
    raise SystemExit(1)
if behavior == "startup-oversized":
    stream.write(struct.pack("!I", MAX_FRAME_LENGTH + 1))
    raise SystemExit(1)
if behavior == "startup-failed":
    write_frame(stream, Frame(FrameKind.FAILED, 0, b"\x01"))
    raise SystemExit(1)
if {provider!r} == "magpie":
    rate = 22050
elif {provider!r} == "voxcpm":
    rate = 48000
else:
    rate = 24000
write_frame(stream, Frame(FrameKind.READY, 0, bytes((0, 1)) + rate.to_bytes(4, "big")))
if behavior == "idle-exit":
    while not (cache / "exit-now").exists():
        time.sleep(0.01)
    raise SystemExit(1)
if behavior == "term-resistant":
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
last_terminal = 0
while True:
    frame = read_frame(__import__("sys").stdin.buffer)
    if frame.kind is FrameKind.CANCEL and frame.request_id == last_terminal:
        continue
    if frame.kind is not FrameKind.SYNTHESIZE:
        write_frame(stream, Frame(FrameKind.FAILED, 0, b"\x01"))
        raise SystemExit(1)
    request_id = frame.request_id
    text = frame.payload[1:].decode()
    print("synthesis print noise", flush=True)
    os.write(1, b"synthesis native noise\n")
    with (cache / "request-ids").open("a", encoding="utf-8") as ids:
        ids.write(f"{{request_id}}\n")
    if text == "stale":
        write_frame(stream, Frame(FrameKind.PCM, request_id + 1, b"\x00\x01"))
        continue
    if text == "wrong-direction":
        write_frame(stream, Frame(FrameKind.READY, 0, b"\x00\x01\x00\x00]\xc0"))
        continue
    if text == "unexpected-cancelled":
        write_frame(stream, Frame(FrameKind.CANCELLED, request_id, b""))
        continue
    if text == "failed":
        write_frame(stream, Frame(FrameKind.FAILED, request_id, b"\x01"))
        raise SystemExit(1)
    write_frame(stream, Frame(FrameKind.PCM, request_id, b"\x00\x01"))
    if text == "midstream-exit":
        raise SystemExit(1)
    if text == "wait-cancel":
        cancel = read_frame(__import__("sys").stdin.buffer)
        with (cache / "cancel-observed").open("a", encoding="utf-8") as observed:
            observed.write(f"{{cancel.request_id}}\n")
        write_frame(stream, Frame(FrameKind.CANCELLED, cancel.request_id, b""))
        last_terminal = request_id
        continue
    if behavior == "partial-cancel":
        read_frame(__import__("sys").stdin.buffer)
        (cache / "cancel-observed").touch()
        stream.write(struct.pack("!IBQ", 11, FrameKind.COMPLETE, request_id) + b"x")
        time.sleep(30)
    write_frame(stream, Frame(FrameKind.COMPLETE, request_id, b""))
    if behavior == "duplicate-terminal":
        write_frame(stream, Frame(FrameKind.COMPLETE, request_id, b""))
    last_terminal = request_id
"""


def closed_provider(
    tmp_path: Path, behavior: str = "normal", provider_name: str = "chatterbox"
) -> tuple[Path, Path]:
    root = tmp_path / "providers"
    provider = root / provider_name
    entrypoint = provider / ".venv" / "bin" / f"speech-runtime-{provider_name}-worker"
    entrypoint.parent.mkdir(parents=True, exist_ok=True)
    entrypoint.write_text(
        _WORKER.format(
            python=sys.executable,
            behavior=behavior,
            provider=provider_name,
        ),
        encoding="utf-8",
    )
    entrypoint.chmod(0o700)
    return root, tmp_path / "cache"


def observed(cache: Path) -> tuple[str, dict[str, str], int, dict[str, str]]:
    lines = (cache / "worker-observed").read_text(encoding="utf-8").splitlines()
    return (
        ast.literal_eval(lines[0]),
        ast.literal_eval(lines[1]),
        ast.literal_eval(lines[2]),
        ast.literal_eval(lines[3]),
    )


def assert_reaped(pid: int) -> None:
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)

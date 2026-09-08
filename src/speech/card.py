"""How much of the GPU the process can see."""

from __future__ import annotations

import shutil
import subprocess
from typing import Final

_QUERY: Final = [
    "--query-gpu=memory.used",
    "--format=csv,noheader,nounits",
]


def card_memory_mb() -> int:
    """Used memory on GPU 0 in mebibytes, or 0 when the card cannot be queried."""
    nvidia = shutil.which("nvidia-smi")
    if nvidia is None:
        return 0
    try:
        completed = subprocess.run(
            [nvidia, *_QUERY],
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return 0
    first = completed.stdout.strip().splitlines()
    if not first:
        return 0
    try:
        return int(first[0].split(",")[0].strip())
    except ValueError:
        return 0

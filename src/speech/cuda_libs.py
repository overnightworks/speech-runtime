"""CTranslate2 looks up CUDA 12 libraries at encode time; they live in this venv."""

from __future__ import annotations

import ctypes
import logging
import os
from pathlib import Path

_LOG = logging.getLogger(__name__)

_PACKAGES = ("nvidia.cublas", "nvidia.cudnn")
_LIBRARIES = (
    "libcublas.so.12",
    "libcublasLt.so.12",
    "libcudnn.so.9",
)


def cuda_library_dirs() -> list[Path]:
    """Directories that hold the bundled NVIDIA CUDA 12 libraries."""
    found: list[Path] = []
    for package in _PACKAGES:
        try:
            module = __import__(package, fromlist=["*"])
        except ImportError:
            continue
        lib = Path(module.__file__).resolve().parent / "lib"
        if lib.is_dir():
            found.append(lib)
    return found


def prepare_cuda_libraries() -> None:
    """Put bundled CUDA 12 libs on the loader path and preload Whisper's."""
    directories = cuda_library_dirs()
    if not directories:
        return
    prefix = ":".join(str(path) for path in directories)
    current = os.environ.get("LD_LIBRARY_PATH", "")
    os.environ["LD_LIBRARY_PATH"] = f"{prefix}:{current}" if current else prefix
    for directory in directories:
        for name in _LIBRARIES:
            candidate = directory / name
            if not candidate.exists():
                continue
            try:
                ctypes.CDLL(str(candidate), mode=ctypes.RTLD_GLOBAL)
            except OSError:
                _LOG.debug("could not preload %s", candidate)

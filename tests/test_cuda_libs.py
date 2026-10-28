"""The bundled CUDA library lookup finds regular and namespace NVIDIA wheels."""

from __future__ import annotations

import importlib
import sys
from typing import TYPE_CHECKING

import pytest

from presentator_speech.cuda_libs import cuda_library_dirs

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path
    from types import ModuleType


def _nvidia_modules() -> dict[str, ModuleType]:
    return {
        name: module
        for name, module in sys.modules.items()
        if name.split(".")[0] == "nvidia"
    }


@pytest.fixture
def site(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """An empty import path that no installed nvidia package can shadow."""
    directory = tmp_path / "site"
    directory.mkdir()
    original_modules = _nvidia_modules()
    for name in original_modules:
        del sys.modules[name]
    monkeypatch.setattr(sys, "path", [str(directory)])
    importlib.invalidate_caches()
    yield directory
    for name in _nvidia_modules():
        del sys.modules[name]
    sys.modules.update(original_modules)
    importlib.invalidate_caches()


@pytest.mark.parametrize("init_files", [False, True], ids=["namespace", "regular"])
def test_finds_the_cublas_and_cudnn_library_directories(
    site: Path, *, init_files: bool
) -> None:
    expected = []
    for package in ("cublas", "cudnn"):
        library = site / "nvidia" / package / "lib"
        library.mkdir(parents=True)
        expected.append(library.resolve())
        if init_files:
            (site / "nvidia" / "__init__.py").touch()
            (site / "nvidia" / package / "__init__.py").touch()

    assert cuda_library_dirs() == expected


@pytest.mark.usefixtures("site")
def test_a_missing_package_yields_no_directory() -> None:
    assert cuda_library_dirs() == []

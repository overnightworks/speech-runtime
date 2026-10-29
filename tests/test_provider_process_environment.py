"""A provider process sees the CUDA libraries of its own virtual environment."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pytest

from speech_runtime_service.provider_process import provider_environment

if TYPE_CHECKING:
    from pathlib import Path


def _cuda_library_directory(
    provider_directory: Path, python_directory: str, package: str
) -> Path:
    directory = (
        provider_directory
        / ".venv"
        / "lib"
        / python_directory
        / "site-packages"
        / "nvidia"
        / package
        / "lib"
    )
    directory.mkdir(parents=True)
    return directory


def test_no_cuda_packages_leave_the_library_path_unset(tmp_path: Path) -> None:
    assert "LD_LIBRARY_PATH" not in provider_environment(tmp_path)


@pytest.mark.parametrize("python_directory", ["python3.12", "python3.14"])
def test_library_path_names_the_cuda_packages_of_any_python_three_venv(
    tmp_path: Path, python_directory: str
) -> None:
    cublas = _cuda_library_directory(tmp_path, python_directory, "cublas")
    cudnn = _cuda_library_directory(tmp_path, python_directory, "cudnn")

    environment = provider_environment(tmp_path)

    assert environment["LD_LIBRARY_PATH"] == os.pathsep.join([str(cublas), str(cudnn)])


@pytest.mark.parametrize("visible_devices", ["", "0,1"])
def test_visible_devices_are_passed_through_when_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, visible_devices: str
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible_devices)

    assert provider_environment(tmp_path)["CUDA_VISIBLE_DEVICES"] == visible_devices


def test_visible_devices_stay_unset_when_the_parent_has_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    assert "CUDA_VISIBLE_DEVICES" not in provider_environment(tmp_path)

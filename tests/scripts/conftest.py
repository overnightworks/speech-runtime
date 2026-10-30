"""Fixtures for the proof scripts' tests: one throwaway identity per test."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from proof_identity import Minting, ProofIdentity, load_minting, proof_identity

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path


@pytest.fixture
def proof_voice() -> str:
    return "piper"


@pytest.fixture
def identity(tmp_path: Path, proof_voice: str) -> Iterator[ProofIdentity]:
    with proof_identity(tmp_path, proof_voice) as running:
        yield running


@pytest.fixture
def minting(identity: ProofIdentity) -> Minting:
    return load_minting(identity.minting_path)

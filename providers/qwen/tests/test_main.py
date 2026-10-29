"""The Qwen entrypoint delegates to the one shared worker loop."""

from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import TYPE_CHECKING

from speech_runtime_qwen import __main__ as entrypoint

if TYPE_CHECKING:
    import pytest


def test_entrypoint_loads_without_synthesis_before_ready(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {}
    exit_code = 17
    protocol_descriptor = 9
    sample_rate = 24_000

    def serve_provider(**arguments: object) -> int:
        observed.update(arguments)
        return exit_code

    monkeypatch.setattr(
        entrypoint,
        "_arguments",
        lambda: SimpleNamespace(
            expected_parent_pid=1,
            device="cuda",
            cache="/closed/cache",
            speaker="Ryan",
        ),
    )
    monkeypatch.setattr(entrypoint, "install_parent_death_signal", lambda _pid: None)
    monkeypatch.setattr(entrypoint.os, "dup", lambda _fd: protocol_descriptor)
    monkeypatch.setattr(entrypoint.os, "open", lambda *_args: 10)
    monkeypatch.setattr(entrypoint.os, "dup2", lambda *_args: None)
    monkeypatch.setattr(entrypoint.os, "close", lambda *_args: None)
    monkeypatch.setitem(
        sys.modules,
        "speech_runtime_qwen.model",
        SimpleNamespace(
            load_model=lambda *_args: object(), pcm_chunks=lambda *_args: ()
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "speech_runtime_provider_contract",
        SimpleNamespace(
            QWEN_SAMPLE_RATE=sample_rate,
            ProviderFunctions=lambda load, synthesize: SimpleNamespace(
                load=load,
                synthesize=synthesize,
                prepare=None,
            ),
            serve_provider=serve_provider,
        ),
    )

    assert entrypoint.main() == exit_code

    functions = observed["functions"]
    assert callable(functions.load)
    assert callable(functions.synthesize)
    assert functions.prepare is None
    assert observed["device"] == "cuda"
    assert observed["cache"] == "/closed/cache"
    assert observed["protocol_stdout"] == protocol_descriptor
    assert observed["sample_rate"] == sample_rate

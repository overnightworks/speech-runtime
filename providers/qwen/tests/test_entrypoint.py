"""The Qwen entrypoint delegates to the one shared worker loop."""

from __future__ import annotations

import sys
from types import SimpleNamespace

from presentator_qwen import __main__ as entrypoint


def test_entrypoint_loads_without_synthesis_before_ready(monkeypatch) -> None:
    observed: dict[str, object] = {}

    def serve_provider(**arguments: object) -> int:
        observed.update(arguments)
        return 17

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
    monkeypatch.setattr(entrypoint.os, "dup", lambda _fd: 9)
    monkeypatch.setattr(entrypoint.os, "open", lambda *_args: 10)
    monkeypatch.setattr(entrypoint.os, "dup2", lambda *_args: None)
    monkeypatch.setattr(entrypoint.os, "close", lambda *_args: None)
    monkeypatch.setitem(
        sys.modules,
        "presentator_qwen.model",
        SimpleNamespace(
            load_model=lambda *_args: object(), pcm_chunks=lambda *_args: ()
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "presentator_speech_provider_contract",
        SimpleNamespace(
            ProviderFunctions=lambda load, synthesize: SimpleNamespace(
                load=load,
                synthesize=synthesize,
                prepare=None,
            ),
            serve_provider=serve_provider,
        ),
    )

    assert entrypoint.main() == 17

    functions = observed["functions"]
    assert callable(functions.load)
    assert callable(functions.synthesize)
    assert functions.prepare is None
    assert observed["device"] == "cuda"
    assert observed["cache"] == "/closed/cache"
    assert observed["protocol_stdout"] == 9

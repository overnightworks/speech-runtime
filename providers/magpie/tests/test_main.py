"""The Magpie entrypoint delegates to the existing provider loop."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace
from typing import TYPE_CHECKING

from speech_runtime_magpie import __main__ as entrypoint

if TYPE_CHECKING:
    import pytest


def test_entrypoint_reserves_stdout_before_importing_the_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {}
    exit_code = 17
    protocol_descriptor = 9
    sample_rate = 22_050
    events: list[str] = []

    def serve_provider(**arguments: object) -> int:
        events.append("serve")
        observed.update(arguments)
        return exit_code

    def provider_functions(load: object, synthesize: object) -> SimpleNamespace:
        events.append("contract:functions")
        return SimpleNamespace(load=load, synthesize=synthesize, prepare=None)

    class RecordingModule(ModuleType):
        def __init__(self, name: str, recorded: dict[str, str]) -> None:
            super().__init__(name)
            self._recorded = recorded

        def __getattribute__(self, name: str) -> object:
            recorded = super().__getattribute__("_recorded")
            if name in recorded:
                events.append(recorded[name])
            return super().__getattribute__(name)

    monkeypatch.setattr(
        entrypoint,
        "_arguments",
        lambda: SimpleNamespace(
            expected_parent_pid=1, device="cuda", cache="/cache", speaker="Sofia"
        ),
    )
    monkeypatch.setattr(
        entrypoint,
        "install_parent_death_signal",
        lambda pid: events.append(f"parent-death:{pid}"),
    )
    monkeypatch.setattr(
        entrypoint.os,
        "dup",
        lambda _fd: events.append("stdout:dup") or protocol_descriptor,
    )
    monkeypatch.setattr(
        entrypoint.os, "open", lambda *_args: events.append("stdout:open") or 10
    )
    monkeypatch.setattr(
        entrypoint.os, "dup2", lambda *_args: events.append("stdout:redirect")
    )
    monkeypatch.setattr(
        entrypoint.os, "close", lambda *_args: events.append("stdout:close")
    )
    monkeypatch.setitem(
        sys.modules,
        "speech_runtime_magpie.model",
        RecordingModule(
            "speech_runtime_magpie.model",
            {"load_model": "model:load", "pcm_chunks": "model:pcm"},
        ),
    )
    model_module = sys.modules["speech_runtime_magpie.model"]
    model_module.load_model = lambda *_args: object()
    model_module.pcm_chunks = lambda *_args: ()
    monkeypatch.setitem(
        sys.modules,
        "speech_runtime_provider_contract",
        RecordingModule(
            "speech_runtime_provider_contract",
            {"MAGPIE_SAMPLE_RATE": "contract:rate", "serve_provider": "contract:serve"},
        ),
    )
    contract_module = sys.modules["speech_runtime_provider_contract"]
    contract_module.MAGPIE_SAMPLE_RATE = sample_rate
    contract_module.ProviderFunctions = provider_functions
    contract_module.serve_provider = serve_provider

    assert entrypoint.main() == exit_code
    assert observed["sample_rate"] == sample_rate
    assert observed["protocol_stdout"] == protocol_descriptor
    assert events == [
        "stdout:dup",
        "stdout:open",
        "stdout:redirect",
        "stdout:close",
        "parent-death:1",
        "contract:rate",
        "contract:serve",
        "model:load",
        "model:pcm",
        "contract:functions",
        "serve",
    ]

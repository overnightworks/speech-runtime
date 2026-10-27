"""The Magpie entrypoint delegates to the existing provider loop."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

from presentator_magpie import __main__ as entrypoint


def test_entrypoint_reserves_stdout_before_importing_the_provider(monkeypatch) -> None:
    observed: dict[str, object] = {}
    events: list[str] = []

    def serve_provider(**arguments: object) -> int:
        events.append("serve")
        observed.update(arguments)
        return 17

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
        entrypoint.os, "dup", lambda _fd: events.append("stdout:dup") or 9
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
        "presentator_magpie.model",
        RecordingModule(
            "presentator_magpie.model",
            {"load_model": "model:load", "pcm_chunks": "model:pcm"},
        ),
    )
    model_module = sys.modules["presentator_magpie.model"]
    model_module.load_model = lambda *_args: object()
    model_module.pcm_chunks = lambda *_args: ()
    monkeypatch.setitem(
        sys.modules,
        "presentator_speech_provider_contract",
        RecordingModule(
            "presentator_speech_provider_contract",
            {"MAGPIE_SAMPLE_RATE": "contract:rate", "serve_provider": "contract:serve"},
        ),
    )
    contract_module = sys.modules["presentator_speech_provider_contract"]
    contract_module.MAGPIE_SAMPLE_RATE = 22_050
    contract_module.ProviderFunctions = provider_functions
    contract_module.serve_provider = serve_provider

    assert entrypoint.main() == 17
    assert observed["sample_rate"] == 22_050
    assert observed["protocol_stdout"] == 9
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

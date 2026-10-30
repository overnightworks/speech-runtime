"""The proof's log check rejects a service log that holds spoken or heard text."""

from __future__ import annotations

import json

import pytest

from tests.support.script_support import script_module

PROVE = script_module("prove_speech")
CONTRACT_OUTPUT = "\n".join(
    [
        "contract_health {}",
        "contract_hear_partial t=0.5s still_sending=True "
        + json.dumps({"text": "Die Kamera", "final": False}),
        "contract_hear_final t=2.0s "
        + json.dumps({"text": "Die Kamera zeigt", "final": True}),
        "contract_hear_second_final t=3.0s "
        + json.dumps({"text": "Die nächste Folie", "final": True}, ensure_ascii=False),
    ]
)


def test_heard_texts_are_read_from_the_contract_client_output() -> None:
    assert PROVE.heard_texts(CONTRACT_OUTPUT) == [
        "Die Kamera",
        "Die Kamera zeigt",
        "Die nächste Folie",
    ]


@pytest.mark.parametrize(
    ("log", "texts", "leaked"),
    [
        ("INFO started\nINFO text: Hallo Welt", ["Hallo Welt"], ["Hallo Welt"]),
        ("INFO heard Die Kamera zeigt", ["Die Kamera zeigt"], ["Die Kamera zeigt"]),
        ("INFO started\nINFO speaking done", ["Hallo Welt", "Die Kamera"], []),
        ("INFO started", ["", "  "], []),
    ],
)
def test_the_log_check_reports_only_the_texts_the_log_holds(
    log: str, texts: list[str], leaked: list[str]
) -> None:
    assert PROVE.spoken_or_heard_in_log(log, texts) == leaked

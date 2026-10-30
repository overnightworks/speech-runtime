"""The proof's log check rejects a service log that holds spoken or heard text."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import Mock

import pytest

from tests.support.script_support import script_module

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

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


FLEET_BUDGET_SECONDS = 1830


@pytest.fixture
def restored_terminate_handler() -> Iterator[None]:
    """Put the TERM handler back after a test that lets the proof install its own."""
    previous = signal.getsignal(signal.SIGTERM)
    yield
    signal.signal(signal.SIGTERM, previous)


@pytest.fixture
def terminate_handler() -> Iterator[None]:
    """Install the proof's TERM handler for one test and put the old one back."""
    previous = signal.getsignal(signal.SIGTERM)
    PROVE.exit_on_terminate()
    yield
    signal.signal(signal.SIGTERM, previous)


def run_returning(
    status: int, *, takes_lock: bool = False
) -> Callable[..., subprocess.CompletedProcess[bytes]]:
    """A fake for the re-exec: the flock command ends with this status."""

    def run(
        command: list[str], env: dict[str, str], *, check: bool
    ) -> subprocess.CompletedProcess[bytes]:
        assert not check
        if takes_lock:
            PROVE.announce_lock_held(env)
        return subprocess.CompletedProcess(command, status)

    return run


def test_the_proof_runs_itself_inside_the_fleet_lock_form() -> None:
    command = PROVE.fleet_command(["--measure"])

    assert command[:6] == [
        "flock",
        "-o",
        "-w",
        "1200",
        str(PROVE.LOCK_PATH),
        "timeout",
    ]
    assert command[6:9] == [
        "-k",
        str(PROVE.KILL_GRACE_SECONDS),
        str(PROVE.CAP_SECONDS),
    ]
    assert command[9] == sys.executable
    assert Path(command[10]) == Path(PROVE.__file__)
    assert command[11:] == ["--measure"]


def test_the_cap_and_kill_grace_fit_the_fleet_budget() -> None:
    assert PROVE.CAP_SECONDS + PROVE.KILL_GRACE_SECONDS <= FLEET_BUDGET_SECONDS


def test_a_run_outside_the_lock_hands_over_to_the_fleet_form(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(PROVE.LOCK_HELD_ENV, raising=False)
    started: list[list[str]] = []
    environments: list[dict[str, str]] = []

    def run(
        command: list[str], env: dict[str, str], *, check: bool
    ) -> subprocess.CompletedProcess[bytes]:
        assert not check
        started.append(command)
        environments.append(env)
        return subprocess.CompletedProcess(command, 0)

    assert PROVE.run_under_lock(["--measure"], run) == 0
    assert started == [PROVE.fleet_command(["--measure"])]
    assert PROVE.LOCK_HELD_ENV in environments[0]


def test_a_busy_lock_fails_the_run_and_says_why() -> None:
    with pytest.raises(SystemExit) as failure:
        PROVE.run_under_lock([], run_returning(PROVE.FLOCK_TIMED_OUT_STATUS))

    assert "busy" in str(failure.value)
    assert str(PROVE.LOCK_WAIT_SECONDS) in str(failure.value)


def test_a_failed_proof_under_the_lock_keeps_its_own_exit_status() -> None:
    status = PROVE.FLOCK_TIMED_OUT_STATUS

    assert PROVE.run_under_lock([], run_returning(status, takes_lock=True)) == status


@pytest.mark.usefixtures("terminate_handler")
def test_term_stops_the_speech_server_through_its_teardown(tmp_path: Path) -> None:
    server = Mock(stdout=None, returncode=0)

    with (
        pytest.raises(SystemExit),
        PROVE.speech_server(
            tmp_path,
            {},
            tmp_path / "speech.sock",
            start=lambda *_, **__: server,
            wait_ready=lambda _socket_path: {},
        ),
    ):
        os.kill(os.getpid(), signal.SIGTERM)

    server.terminate.assert_called_once_with()


@pytest.mark.usefixtures("terminate_handler")
def test_a_load_that_never_falls_ends_at_term() -> None:
    with pytest.raises(SystemExit):
        PROVE.wait_for_load(
            read_load=lambda: PROVE.LOAD_CEILING + 1,
            sleep=lambda _: os.kill(os.getpid(), signal.SIGTERM),
        )


@pytest.mark.usefixtures("restored_terminate_handler")
def test_term_during_the_proof_under_the_lock_runs_its_teardown(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    token = tmp_path_factory.mktemp(PROVE.LOCK_DIRECTORY_PREFIX) / "acquired"
    torn_down: list[bool] = []

    def proof_body() -> None:
        try:
            os.kill(os.getpid(), signal.SIGTERM)
        finally:
            torn_down.append(True)

    with pytest.raises(SystemExit):
        PROVE.prove_under_lock({PROVE.LOCK_HELD_ENV: str(token)}, proof_body)

    assert torn_down == [True]
    assert token.exists()


def test_a_lock_marker_outside_a_lock_directory_is_refused(tmp_path: Path) -> None:
    marker = tmp_path / "1"

    with pytest.raises(SystemExit):
        PROVE.announce_lock_held({PROVE.LOCK_HELD_ENV: str(marker)})

    assert not marker.exists()

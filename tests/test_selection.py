"""The durable voice choice has one safe, atomic owner."""

import os
import stat

import pytest

from speech.selection import (
    DurabilityUnconfirmedError,
    SelectionError,
    VoiceSelectionStore,
)
from speech.voices import VoiceId


def test_selection_creates_private_state_and_atomically_restores_a_voice(
    tmp_path,
) -> None:
    store = VoiceSelectionStore(tmp_path / "state", owner_uid=os.geteuid())

    assert store.read() is None
    assert store.write(VoiceId.CHATTERBOX) is True
    assert store.read() is VoiceId.CHATTERBOX
    assert stat.S_IMODE(store.path.stat().st_mode) == 0o600
    assert stat.S_IMODE(store.path.parent.stat().st_mode) == 0o700


def test_selection_refuses_a_present_invalid_token_without_rewriting_it(
    tmp_path,
) -> None:
    directory = tmp_path / "state"
    directory.mkdir(mode=0o700)
    path = directory / "selected-voice"
    path.write_text("future-voice", encoding="utf-8")
    path.chmod(0o600)
    store = VoiceSelectionStore(directory, owner_uid=os.geteuid())

    with pytest.raises(SelectionError):
        store.read()

    assert path.read_text(encoding="utf-8") == "future-voice"


def test_selection_refuses_a_symlink_even_when_its_target_is_private(tmp_path) -> None:
    directory = tmp_path / "state"
    directory.mkdir(mode=0o700)
    target = tmp_path / "target"
    target.write_text(VoiceId.PIPER.value, encoding="utf-8")
    target.chmod(0o600)
    (directory / "selected-voice").symlink_to(target)
    store = VoiceSelectionStore(directory, owner_uid=os.geteuid())

    with pytest.raises(SelectionError):
        store.read()


@pytest.mark.parametrize("step", ["write", "replace"])
def test_selection_keeps_the_previous_choice_when_commit_has_not_happened(
    tmp_path, monkeypatch: pytest.MonkeyPatch, step: str
) -> None:
    store = VoiceSelectionStore(tmp_path / "state", owner_uid=os.geteuid())
    store.write(VoiceId.PIPER)

    if step == "write":
        monkeypatch.setattr(
            "speech.selection.os.write", lambda *_args: (_ for _ in ()).throw(OSError())
        )
    else:
        monkeypatch.setattr(
            "speech.selection.Path.replace",
            lambda *_args: (_ for _ in ()).throw(OSError()),
        )

    with pytest.raises(SelectionError):
        store.write(VoiceId.CHATTERBOX)

    assert store.read() is VoiceId.PIPER


def test_selection_exposes_the_replaced_choice_when_directory_sync_fails(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = VoiceSelectionStore(tmp_path / "state", owner_uid=os.geteuid())
    monkeypatch.setattr(
        store, "_sync_directory", lambda: (_ for _ in ()).throw(OSError())
    )

    with pytest.raises(DurabilityUnconfirmedError):
        store.write(VoiceId.CHATTERBOX)

    assert store.read() is VoiceId.CHATTERBOX

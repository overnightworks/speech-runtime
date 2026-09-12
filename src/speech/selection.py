"""The one durable, private choice of active speaking voice."""

from __future__ import annotations

import os
import stat
import tempfile
from pathlib import Path

from speech.voices import VoiceId

_DIRECTORY_MODE = 0o700
_FILE_MODE = 0o600
_FILE_NAME = "selected-voice"


class SelectionError(RuntimeError):
    """The selection state cannot be safely read or written."""


class DurabilityUnconfirmedError(SelectionError):
    """The replacement is visible, but its directory could not be synced."""


class VoiceSelectionStore:
    """Own a single atomically-replaced non-secret VoiceId file."""

    def __init__(self, directory: Path, *, owner_uid: int) -> None:
        """Remember the directory and effective identity that may own it."""
        self._directory = directory
        self._owner_uid = owner_uid

    @property
    def path(self) -> Path:
        """Name the one file this owner maintains."""
        return self._directory / _FILE_NAME

    def read(self) -> VoiceId | None:
        """Return an absent choice or reject an unsafe/invalid present file."""
        self.preflight()
        try:
            metadata = self.path.lstat()
        except FileNotFoundError:
            return None
        self._validate_regular_file(metadata)
        try:
            return VoiceId(self.path.read_text(encoding="utf-8").strip())
        except (OSError, ValueError) as error:
            raise SelectionError from error

    def write(self, voice: VoiceId) -> bool:
        """Replace the selection and report confirmed directory durability."""
        self.preflight()
        descriptor = -1
        temporary: Path | None = None
        replaced = False
        try:
            descriptor, raw_path = tempfile.mkstemp(
                prefix=".selected-voice-", dir=self._directory
            )
            temporary = Path(raw_path)
            os.fchmod(descriptor, _FILE_MODE)
            os.write(descriptor, voice.value.encode("utf-8"))
            os.fsync(descriptor)
            os.close(descriptor)
            descriptor = -1
            temporary.replace(self.path)
            replaced = True
            self._sync_directory()
        except OSError as error:
            if replaced:
                raise DurabilityUnconfirmedError from error
            raise SelectionError from error
        else:
            return True
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            if temporary is not None and temporary.exists():
                temporary.unlink()

    def preflight(self) -> None:
        """Create or validate the private directory before a Load consumes a model."""
        self._prepare_directory()

    def _prepare_directory(self) -> None:
        self._directory.mkdir(mode=_DIRECTORY_MODE, parents=True, exist_ok=True)
        metadata = self._directory.stat()
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or metadata.st_uid != self._owner_uid
            or stat.S_IMODE(metadata.st_mode) != _DIRECTORY_MODE
        ):
            raise SelectionError

    def _validate_regular_file(self, metadata: os.stat_result) -> None:
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != self._owner_uid
            or stat.S_IMODE(metadata.st_mode) != _FILE_MODE
        ):
            raise SelectionError

    def _sync_directory(self) -> None:
        descriptor = os.open(self._directory, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

"""Download the exact local Qwen snapshot without selecting it."""

import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass, field

from huggingface_hub import snapshot_download
from presentator_speech_provider_contract import (
    QWEN_ARTIFACTS,
    QWEN_MODEL_ID,
    QWEN_REVISION,
)

from speech import voices
from speech.config import Settings
from speech.voices import VoiceDownloadOutcome


def download_qwen(settings: Settings) -> None:
    """Fetch only the pinned Qwen files into the configured Hub cache."""
    if not voices.qwen_worker_is_usable(settings):
        raise RuntimeError
    snapshot_download(
        repo_id=QWEN_MODEL_ID,
        revision=QWEN_REVISION,
        cache_dir=settings.huggingface_cache,
        allow_patterns=list(QWEN_ARTIFACTS),
        token=False,
        force_download=False,
    )
    if voices.VoiceId.QWEN not in voices.installed_voice_ids(settings):
        raise RuntimeError


@dataclass(slots=True)
class _QwenDownload:
    download: Callable[[], None] | None
    worker_is_usable: Callable[[], bool] | None
    artifacts_are_complete: Callable[[], bool]
    publication_lock: threading.Lock
    _guard: threading.Lock = field(default_factory=threading.Lock)
    _state: tuple[bool, bool] = (False, False)

    def acquire(self) -> bool:
        return self._guard.acquire(blocking=False)

    def start(self, may_start: Callable[[], bool]) -> VoiceDownloadOutcome | None:
        release_guard = True
        if not self.acquire():
            return None
        try:
            with self.publication_lock:
                if (
                    not may_start()
                    or self.download is None
                    or self.worker_is_usable is None
                    or not self.worker_is_usable()
                ):
                    return None
                if self.artifacts_are_complete():
                    return VoiceDownloadOutcome.ALREADY_COMPLETE
                self._state = (True, False)
            try:
                threading.Thread(target=self._run, daemon=True).start()
            except RuntimeError:
                self._finish(callback_failed=True)
                return None
            release_guard = False
            return VoiceDownloadOutcome.STARTED
        finally:
            if release_guard:
                self.release()

    def release(self) -> None:
        self._guard.release()

    def state(self) -> tuple[bool, bool]:
        return self._state

    def _run(self) -> None:
        callback_failed = False
        try:
            if self.download is not None:
                self.download()
        except Exception:
            callback_failed = True
        finally:
            try:
                self._finish(callback_failed=callback_failed)
            finally:
                self.release()

    def _finish(self, *, callback_failed: bool) -> None:
        try:
            complete = self.artifacts_are_complete()
        except Exception:
            complete = False
            callback_failed = True
        if callback_failed:
            logging.getLogger(__name__).error("Qwen download failed")
        with self.publication_lock:
            self._state = (False, not complete)

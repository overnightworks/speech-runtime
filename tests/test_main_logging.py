"""The service logs at INFO and holds Piper's input-carrying loggers back."""

import os
import subprocess
import sys

_CHILD = """
import logging
from speech_runtime_service.__main__ import configure_logging

configure_logging()
logging.getLogger("piper.voice").debug("text=%s", "SPOKEN-SENTENCE")
logging.getLogger("piper.phoneme_ids").warning("missing %s", "SPOKEN-PHONEMES")
logging.getLogger("speech_runtime_service.service").info("SERVICE-LINE")
"""


def test_configured_logging_drops_piper_input_and_keeps_service_info() -> None:
    environment = {**os.environ, "SPEECH_DEBUG": "true"}

    completed = subprocess.run(
        [sys.executable, "-c", _CHILD],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert "SERVICE-LINE" in completed.stderr
    assert "SPOKEN" not in completed.stderr

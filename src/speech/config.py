"""What varies by machine; everything else is a constant beside its owner."""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_SPEAKING_MODEL = "de_DE-thorsten-medium"
CHATTERBOX_SPEAKING_MODEL = "ResembleAI/chatterbox"
DEFAULT_HEARING_MODEL = "Systran/faster-whisper-large-v3"
HEAR_SAMPLE_RATE = 16_000


class Settings(BaseSettings):
    """Read from `SPEECH_*` in the environment."""

    model_config = SettingsConfigDict(env_prefix="SPEECH_")

    host: str = "127.0.0.1"
    port: int = 8090
    device: str = "cuda"
    speaking_model: str = DEFAULT_SPEAKING_MODEL
    hearing_model: str = DEFAULT_HEARING_MODEL
    debug: bool = False
    voice_cache: Path = Field(default_factory=lambda: Path.home() / ".cache" / "piper")


def load_settings() -> Settings:
    """Read `SPEECH_*` from the environment."""
    return Settings()

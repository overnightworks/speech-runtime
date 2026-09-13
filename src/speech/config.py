"""What varies by machine; everything else is a constant beside its owner."""

from pathlib import Path

from huggingface_hub import constants as huggingface_hub_constants
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_SPEAKING_MODEL = "de_DE-thorsten-medium"
CHATTERBOX_SPEAKING_MODEL = "ResembleAI/chatterbox"
DEFAULT_HEARING_MODEL = "Systran/faster-whisper-large-v3"
HEAR_SAMPLE_RATE = 16_000
_STATE_DIRECTORY_ERROR = "state directory must be absolute"
_PROVIDER_ROOT_ERROR = "provider root must be absolute"
_SPEAKING_MODEL_ERROR = "speaking model must be a supported baseline"


class Settings(BaseSettings):
    """Read from `SPEECH_*` in the environment."""

    model_config = SettingsConfigDict(env_prefix="SPEECH_", hide_input_in_errors=True)

    host: str = "127.0.0.1"
    port: int = 8090
    device: str = "cuda"
    speaking_model: str = DEFAULT_SPEAKING_MODEL
    hearing_model: str = DEFAULT_HEARING_MODEL
    debug: bool = False
    voice_cache: Path = Field(default_factory=lambda: Path.home() / ".cache" / "piper")
    huggingface_cache: Path = Field(
        default_factory=lambda: Path(huggingface_hub_constants.HF_HUB_CACHE)
    )
    provider_root: Path | None = None
    private_directory: Path = Path("/run/presentator-speech")
    state_directory: Path = Field(
        default_factory=lambda: Path.home() / ".local/state/presentator-speech"
    )
    runtime_uid: int | None = Field(
        default=None,
        gt=0,
        validation_alias="PRESENTATOR_RUNTIME_UID",
    )

    @field_validator("state_directory")
    @classmethod
    def _state_directory_is_absolute(cls, directory: Path) -> Path:
        if not directory.is_absolute():
            raise ValueError(_STATE_DIRECTORY_ERROR)
        return directory

    @field_validator("provider_root")
    @classmethod
    def _provider_root_is_absolute(cls, directory: Path | None) -> Path | None:
        if directory is not None and not directory.is_absolute():
            raise ValueError(_PROVIDER_ROOT_ERROR)
        return directory

    @field_validator("speaking_model")
    @classmethod
    def _speaking_model_is_a_baseline(cls, model: str) -> str:
        if model not in {DEFAULT_SPEAKING_MODEL, CHATTERBOX_SPEAKING_MODEL}:
            raise ValueError(_SPEAKING_MODEL_ERROR)
        return model


def load_settings() -> Settings:
    """Read `SPEECH_*` from the environment."""
    return Settings()

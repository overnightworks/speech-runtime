"""What varies by machine; everything else is a constant beside its owner."""

import tomllib
from pathlib import Path
from typing import Annotated, Self

from huggingface_hub import constants as huggingface_hub_constants
from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    Strict,
    field_validator,
    model_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict
from speech_runtime_provider_contract import (
    QWEN_MODEL_ID,
    MagpieSpeaker,
    QwenSpeaker,
)
from webauth.https_client import is_https_url

from speech_runtime_service.voices import (
    CHATTERBOX_SPEAKING_MODEL,
    DEFAULT_SPEAKING_MODEL,
    VoiceId,
)

DEFAULT_HEARING_MODEL = "Systran/faster-whisper-large-v3"
HEAR_SAMPLE_RATE = 16_000
_STATE_DIRECTORY_ERROR = "state directory must be absolute"
_PROVIDER_ROOT_ERROR = "provider root must be absolute"
_SPEAKING_MODEL_ERROR = "speaking model must be a supported baseline"
_CONFIG_FILE_ERROR = "config file must be absolute"
_KEY_SET_URL_ERROR = "jwks_url must be an https URL"
_SERVICE_GROUP_ERROR = "service_group must be a group path such as /services/name"
_APP_NAME_ERROR = "an app name must be one group path segment"
_VOICE_ERROR = "voices must each be a voice this service offers"
_SHARED_CLIENT_ERROR = "workload_client must belong to one app"
_CONFIG_FILE_UNREADABLE_ERROR = "SPEECH_CONFIG_FILE must name a readable file"


class Settings(BaseSettings):
    """Read from `SPEECH_*` in the environment."""

    model_config = SettingsConfigDict(env_prefix="SPEECH_", hide_input_in_errors=True)

    device: str = "cuda"
    speaking_model: str = DEFAULT_SPEAKING_MODEL
    hearing_model: str = DEFAULT_HEARING_MODEL
    voice_cache: Path = Field(default_factory=lambda: Path.home() / ".cache" / "piper")
    huggingface_cache: Path = Field(
        default_factory=lambda: Path(huggingface_hub_constants.HF_HUB_CACHE)
    )
    provider_root: Path | None = None
    qwen_speaker: QwenSpeaker = QwenSpeaker.RYAN
    magpie_speaker: MagpieSpeaker = MagpieSpeaker.SOFIA
    private_directory: Path = Path("/run/speech-runtime")
    state_directory: Path = Field(
        default_factory=lambda: Path.home() / ".local/state/speech-runtime"
    )
    busy_wait_seconds: float = Field(default=5.0, ge=0, allow_inf_nan=False)
    max_input_characters: int = Field(default=4096, ge=1)
    config_file: Path | None = None
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

    @field_validator("config_file")
    @classmethod
    def _config_file_is_absolute(cls, path: Path | None) -> Path | None:
        if path is not None and not path.is_absolute():
            raise ValueError(_CONFIG_FILE_ERROR)
        return path

    @field_validator("provider_root")
    @classmethod
    def _provider_root_is_absolute(cls, directory: Path | None) -> Path | None:
        if directory is not None and not directory.is_absolute():
            raise ValueError(_PROVIDER_ROOT_ERROR)
        return directory

    @field_validator("speaking_model")
    @classmethod
    def _speaking_model_is_a_baseline(cls, model: str) -> str:
        if model not in {
            DEFAULT_SPEAKING_MODEL,
            CHATTERBOX_SPEAKING_MODEL,
            QWEN_MODEL_ID,
        }:
            raise ValueError(_SPEAKING_MODEL_ERROR)
        return model


def load_settings() -> Settings:
    """Read `SPEECH_*` from the environment."""
    return Settings()


class _FileModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid", frozen=True, hide_input_in_errors=True, strict=True
    )


class IdentityConfig(_FileModel):
    """Who vouches for the callers, and the one group that may use this service."""

    issuer: str
    jwks_url: str = Field(repr=False)
    audience: str
    workload_role: str
    group_claim: str
    service_group: str
    leeway_seconds: int = Field(default=0, ge=0)

    @field_validator("jwks_url")
    @classmethod
    def _key_set_is_served_over_https(cls, url: str) -> str:
        if not is_https_url(url):
            raise ValueError(_KEY_SET_URL_ERROR)
        return url

    @field_validator("service_group")
    @classmethod
    def _service_group_is_a_group_path(cls, group: str) -> str:
        if not group.startswith("/") or "" in group[1:].split("/"):
            raise ValueError(_SERVICE_GROUP_ERROR)
        return group


def _known_voice(name: object) -> VoiceId:
    # Pydantic's enum error lists every catalogue voice; a fixed message does not.
    try:
        return VoiceId(name)
    except ValueError:
        raise ValueError(_VOICE_ERROR) from None


class AppConfig(_FileModel):
    """An app the service admits: its workload client and the voices it may name."""

    workload_client: str
    # TOML yields strings and lists, which strict mode refuses for an enum and a set.
    voices: Annotated[
        frozenset[Annotated[VoiceId, BeforeValidator(_known_voice)]],
        Strict(strict=False),
        Field(min_length=1),
    ]


class CallerConfig(_FileModel):
    """The service's trust in its callers, read from `SPEECH_CONFIG_FILE`."""

    identity: IdentityConfig
    apps: Annotated[dict[str, AppConfig], Field(min_length=1)]

    @field_validator("apps")
    @classmethod
    def _app_names_are_one_segment(
        cls, apps: dict[str, AppConfig]
    ) -> dict[str, AppConfig]:
        if any(not name or "/" in name for name in apps):
            raise ValueError(_APP_NAME_ERROR)
        return apps

    @model_validator(mode="after")
    def _each_client_belongs_to_one_app(self) -> Self:
        clients = [app.workload_client for app in self.apps.values()]
        if len(set(clients)) != len(clients):
            raise ValueError(_SHARED_CLIENT_ERROR)
        return self


def load_caller_config(path: Path) -> CallerConfig:
    """Read the caller configuration; a bad file fails naming a field, not a value."""
    try:
        text = path.read_text()
    except OSError:
        raise ValueError(_CONFIG_FILE_UNREADABLE_ERROR) from None
    return CallerConfig.model_validate(tomllib.loads(text))

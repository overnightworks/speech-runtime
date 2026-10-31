"""The caller configuration: what the service trusts, and how it refuses a bad file."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest
from pydantic import ValidationError

from speech_runtime_service.config import Settings, load_caller_config
from tests.support.identity_support import (
    ALL_VOICES,
    APP,
    ISSUER,
    MINTING,
    caller_config,
    caller_configuration_text,
    callers_over,
    write_caller_configuration,
)

if TYPE_CHECKING:
    from pathlib import Path

SENTINEL = "SENTINEL-KEY-SET-CREDENTIAL"
HTTP_ADDRESS = f"http://identity.test/certs?credential={SENTINEL}"
LARGE_LEEWAY = 100_000_000
QUIZ_TABLE = (
    f'[apps.{APP}]\nworkload_client = "{MINTING.client_id}"\nvoices = ["piper"]\n'
)


def test_the_file_is_read_into_the_identity_and_the_apps(tmp_path: Path) -> None:
    config = load_caller_config(write_caller_configuration(tmp_path))

    assert config.identity.issuer == ISSUER
    assert config.identity.service_group == MINTING.service_group
    assert config.identity.leeway_seconds == 0
    assert config.apps[APP].workload_client == MINTING.client_id
    assert config.apps[APP].voices == ALL_VOICES


def test_an_app_hears_only_when_its_table_says_so(tmp_path: Path) -> None:
    deaf = load_caller_config(write_caller_configuration(tmp_path))
    hearing = load_caller_config(
        write_caller_configuration(
            tmp_path, caller_configuration_text(apps=QUIZ_TABLE + "hear = true\n")
        )
    )

    assert not deaf.apps[APP].hear
    assert hearing.apps[APP].hear


def test_leeway_has_no_upper_cap(tmp_path: Path) -> None:
    text = caller_configuration_text(identity=f"leeway_seconds = {LARGE_LEEWAY}")

    config = load_caller_config(write_caller_configuration(tmp_path, text))

    assert config.identity.leeway_seconds == LARGE_LEEWAY


@pytest.mark.parametrize(
    ("field", "text"),
    [
        ("jwks_url", caller_configuration_text(HTTP_ADDRESS)),
        ("surprise", caller_configuration_text(identity="surprise = 1")),
        ("extra", caller_configuration_text(apps=QUIZ_TABLE + "\n[extra]\n")),
        (
            "workload_client",
            caller_configuration_text(
                apps=QUIZ_TABLE
                + f'\n[apps.second]\nworkload_client = "{MINTING.client_id}"\n'
                + 'voices = ["piper"]\n'
            ),
        ),
        ("apps", caller_configuration_text(apps="")),
        (
            "workload_client",
            caller_configuration_text(
                apps=f'[apps.{APP}]\nworkload_client = 7\nvoices = ["piper"]\n'
            ),
        ),
        ("leeway_seconds", caller_configuration_text(identity="leeway_seconds = -1")),
        (
            "apps",
            caller_configuration_text(
                apps='[apps."a/b"]\nworkload_client = "other"\nvoices = ["piper"]\n'
            ),
        ),
    ],
    ids=[
        "key-set URL is not https",
        "unknown identity key",
        "unknown top-level key",
        "one workload client for two apps",
        "no apps",
        "client id not a string",
        "negative leeway",
        "app name of two segments",
    ],
)
def test_a_bad_file_fails_naming_the_field_and_not_its_value(
    tmp_path: Path, field: str, text: str
) -> None:
    path = write_caller_configuration(tmp_path, text)

    with pytest.raises(ValidationError) as failure:
        load_caller_config(path)

    assert field in str(failure.value)
    assert SENTINEL not in str(failure.value)


@pytest.mark.parametrize(
    "malformed",
    ["", "/", "speech-service", "/services/", "//services", "/services//speech"],
    ids=["empty", "root", "no slash", "trailing slash", "double lead", "double mid"],
)
def test_a_service_group_that_is_not_a_group_path_fails_at_startup(
    tmp_path: Path, malformed: str
) -> None:
    text = caller_configuration_text().replace(
        json.dumps(MINTING.service_group), json.dumps(malformed)
    )
    path = write_caller_configuration(tmp_path, text)

    with pytest.raises(ValidationError) as failure:
        load_caller_config(path)

    assert "service_group" in str(failure.value)


def test_the_key_set_address_stays_out_of_every_repr() -> None:
    config = caller_config(jwks_url=HTTP_ADDRESS.replace("http:", "https:"))

    shown = repr(config) + repr(config.identity) + repr(callers_over())

    assert SENTINEL not in shown


def test_the_configuration_file_must_be_absolute() -> None:
    with pytest.raises(ValidationError):
        Settings(config_file="relative.toml")


@pytest.mark.parametrize(
    ("table", "field"),
    [
        (f'[apps.{APP}]\nworkload_client = "{MINTING.client_id}"\n', "voices"),
        (QUIZ_TABLE.replace('["piper"]', "[]"), "voices"),
        (QUIZ_TABLE.replace('["piper"]', '["alloy"]'), "voices"),
        (QUIZ_TABLE.replace('["piper"]', '"piper"'), "voices"),
        (QUIZ_TABLE + "speak = true\n", "speak"),
        (QUIZ_TABLE + 'hear = "yes"\n', "hear"),
    ],
    ids=["missing", "empty", "unknown voice", "not a list", "speak flag", "hear text"],
)
def test_an_app_table_without_a_valid_voice_list_fails_naming_the_key(
    tmp_path: Path, table: str, field: str
) -> None:
    path = write_caller_configuration(tmp_path, caller_configuration_text(apps=table))

    with pytest.raises(ValidationError) as failure:
        load_caller_config(path)

    assert f"apps.{APP}.{field}" in str(failure.value)

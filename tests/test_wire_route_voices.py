"""An app names only the voices its configuration lists."""

from __future__ import annotations

import asyncio
import json
import logging
import traceback
from typing import TYPE_CHECKING

import pytest
from fastapi.testclient import TestClient
from starlette.status import (
    HTTP_200_OK,
    HTTP_401_UNAUTHORIZED,
    HTTP_403_FORBIDDEN,
    HTTP_422_UNPROCESSABLE_CONTENT,
    HTTP_503_SERVICE_UNAVAILABLE,
)

from speech_runtime_client.wire import RETRY_AFTER_HEADER, RefusalCode
from speech_runtime_service.config import Settings
from speech_runtime_service.control import create_control_app
from speech_runtime_service.service import Runtime, RuntimeDependencies
from speech_runtime_service.voices import VoiceId
from tests.support.fakes_support import FakeHearing, FakeSpeaking, GatedSpeaking
from tests.support.identity_support import (
    APP,
    MINTING,
    OTHER_APP,
    OTHER_MINTING,
    callers_of_two_apps,
    valid_callers,
    valid_headers,
)
from tests.support.speak_support import (
    SpeakCall,
    assert_call_refusal,
    assert_refusal,
    info_comment_before_data,
    post_speech,
    speak_app,
    speak_body,
)

if TYPE_CHECKING:
    from httpx2 import Response

    from speech_runtime_service.callers import SpeechCallers

PIPER = VoiceId.PIPER
CHATTERBOX = VoiceId.CHATTERBOX
SENTENCE = "Zebrakuchen"
FORBIDDEN_VOICE = (HTTP_403_FORBIDDEN, RefusalCode.FORBIDDEN, "voice")


def _runtime_with_active(
    voice: VoiceId, speaking: FakeSpeaking | None = None
) -> Runtime:
    return Runtime(
        speaking or FakeSpeaking(),
        FakeHearing(),
        memory_probe=lambda: 0,
        dependencies=RuntimeDependencies(default_voice=voice),
    )


def _app(
    callers: SpeechCallers, active: VoiceId = PIPER, **settings: float
) -> TestClient:
    return TestClient(
        create_control_app(Settings(**settings), _runtime_with_active(active), callers)
    )


def _speak(
    client: TestClient, voice: VoiceId, headers: dict[str, str] | None = None
) -> Response:
    return post_speech(client, speak_body(voice.value), headers)


def test_an_app_names_a_listed_active_voice_and_is_spoken_to() -> None:
    response = _speak(_app(valid_callers([PIPER])), PIPER)

    assert response.status_code == HTTP_200_OK
    assert info_comment_before_data(response.content) is not None


def test_an_unlisted_voice_is_forbidden_naming_the_voice_even_when_it_is_active() -> (
    None
):
    response = _speak(_app(valid_callers([CHATTERBOX])), PIPER)

    assert_refusal(response, *FORBIDDEN_VOICE)


def test_an_unlisted_voice_that_is_not_active_is_forbidden_and_not_unavailable() -> (
    None
):
    response = _speak(_app(valid_callers([PIPER])), CHATTERBOX)

    assert_refusal(response, *FORBIDDEN_VOICE)


def test_a_listed_voice_that_is_not_active_is_unavailable() -> None:
    response = _speak(_app(valid_callers([PIPER, CHATTERBOX])), CHATTERBOX)

    assert_refusal(
        response, HTTP_503_SERVICE_UNAVAILABLE, RefusalCode.UNAVAILABLE, None
    )


def test_an_unlisted_voice_is_refused_without_waiting_for_the_busy_gate() -> None:
    async def scenario() -> tuple[SpeakCall, SpeakCall]:
        voice = GatedSpeaking()
        runtime = _runtime_with_active(PIPER, voice)
        app = create_control_app(
            Settings(busy_wait_seconds=60), runtime, valid_callers([PIPER])
        )
        holder = SpeakCall(app)
        await holder.status()
        await asyncio.to_thread(voice.entered_first_step.wait, 5)
        refused = SpeakCall(app, speak_body(CHATTERBOX.value))
        try:
            await refused.status()
        finally:
            voice.release_all()
        await refused.finish()
        await holder.finish()
        return holder, refused

    _, refused = asyncio.run(scenario())

    assert_call_refusal(refused, *FORBIDDEN_VOICE)
    assert RETRY_AFTER_HEADER.lower() not in {name.lower() for name in refused.headers}


@pytest.mark.parametrize("active", [PIPER, CHATTERBOX])
def test_each_app_is_judged_by_its_own_list(active: VoiceId) -> None:
    callers = callers_of_two_apps([PIPER], [CHATTERBOX])
    client = _app(callers, active)
    quiz = valid_headers(minting=MINTING)
    other = valid_headers(minting=OTHER_MINTING)

    listed_by_quiz = _speak(client, active, quiz)
    listed_by_other = _speak(client, active, other)

    owner, stranger = (
        (listed_by_quiz, listed_by_other)
        if active is PIPER
        else (listed_by_other, listed_by_quiz)
    )
    assert owner.status_code == HTTP_200_OK
    assert_refusal(stranger, *FORBIDDEN_VOICE)


def test_an_oversized_input_is_unprocessable_before_the_voice_is_checked() -> None:
    client = _app(valid_callers([PIPER]), max_input_characters=5)
    body = json.dumps(
        json.loads(speak_body(CHATTERBOX.value)) | {"input": "far too long"}
    ).encode()

    response = post_speech(client, body)

    assert_refusal(
        response, HTTP_422_UNPROCESSABLE_CONTENT, RefusalCode.INVALID_REQUEST, "input"
    )


def test_missing_tokens_are_unauthorized_before_the_voice_is_checked() -> None:
    response = post_speech(
        TestClient(speak_app(callers=valid_callers([PIPER]))),
        speak_body(CHATTERBOX.value),
        {},
    )

    assert_refusal(response, HTTP_401_UNAUTHORIZED, RefusalCode.UNAUTHORIZED, None)


def test_an_unknown_voice_stays_invalid_and_not_forbidden() -> None:
    response = post_speech(
        TestClient(speak_app(callers=valid_callers([PIPER]))), speak_body("alloy")
    )

    assert_refusal(
        response, HTTP_422_UNPROCESSABLE_CONTENT, RefusalCode.INVALID_REQUEST, "voice"
    )


def test_a_voice_refusal_leaks_no_token_subject_input_app_or_voice_list(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG)
    callers = callers_of_two_apps([PIPER], [PIPER, CHATTERBOX])
    headers = valid_headers()
    body = json.dumps(
        json.loads(speak_body(CHATTERBOX.value)) | {"input": SENTENCE}
    ).encode()

    response = post_speech(_app(callers), body, headers)

    logged = "\n".join(
        f"{record.getMessage()} {record.exc_text or ''}"
        + "".join(traceback.format_exception(*record.exc_info or (None, None, None)))
        for record in caplog.records
    )
    shown = logged + response.text
    leaks = [
        *headers.values(),
        MINTING.person_subject,
        MINTING.workload_subject,
        SENTENCE,
        APP,
        OTHER_APP,
        PIPER.value,
    ]
    assert response.status_code == HTTP_403_FORBIDDEN
    for secret in leaks:
        assert secret not in shown

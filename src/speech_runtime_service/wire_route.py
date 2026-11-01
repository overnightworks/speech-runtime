"""POST /v1/audio/speech, POST /v1/audio/transcriptions and the refusals of the rest."""

from __future__ import annotations

import asyncio
import logging
import math
import time
from typing import TYPE_CHECKING, Final

from pydantic import BaseModel, ValidationError
from python_multipart.exceptions import FormParserError
from starlette.background import BackgroundTask
from starlette.datastructures import UploadFile
from starlette.exceptions import HTTPException
from starlette.requests import Request
from starlette.responses import JSONResponse

from speech_runtime_client.wire import (
    API_VERSION,
    AUTHENTICATE_HEADER,
    RETRY_AFTER_HEADER,
    SPEECH_PATH,
    TRANSCRIPTIONS_PATH,
    HearingLanguage,
    RefusalCode,
    RequestField,
    SpeechRequest,
    Transcription,
    TranscriptionRequest,
    refusal,
)
from speech_runtime_service.failures import log_failure
from speech_runtime_service.recording import (
    RecordingTooLongError,
    UndecodableRecordingError,
    decode_recording,
)
from speech_runtime_service.service import wav_chunks
from speech_runtime_service.voices import VoiceId
from speech_runtime_service.wav_response import ClosingWavResponse

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from fastapi import FastAPI
    from starlette.datastructures import FormData
    from starlette.responses import Response
    from starlette.types import Message, Receive, Scope, Send

    from speech_runtime_service.admission_gate import Lease
    from speech_runtime_service.callers import SpeechCallers
    from speech_runtime_service.config import Settings
    from speech_runtime_service.hearing import HearingEngine
    from speech_runtime_service.service import Runtime, VoiceAdmission

logger = logging.getLogger(__name__)

SPEAKING_LANGUAGE: Final = "de"
_VERSION_PREFIX: Final = "/v{version:int}"
_BEARER_CHALLENGE: Final = "Bearer"


def _refused(
    code: RefusalCode,
    param: RequestField | None = None,
    *,
    headers: dict[str, str] | None = None,
    background: BackgroundTask | None = None,
) -> JSONResponse:
    body = refusal(code, param)
    if code is RefusalCode.UNAUTHORIZED:
        headers = {**(headers or {}), AUTHENTICATE_HEADER: _BEARER_CHALLENGE}
    return JSONResponse(
        body.model_dump(mode="json"),
        status_code=body.status,
        headers=headers,
        background=background,
    )


def _refused_field(
    error: ValidationError, request: type[BaseModel]
) -> RequestField | None:
    """The `request` field the first error names; nothing else of the error leaves."""
    location = error.errors(
        include_url=False, include_context=False, include_input=False
    )[0]["loc"]
    if location and location[0] in request.model_fields:
        return location[0]
    return None


def _parse(raw: bytes) -> SpeechRequest | JSONResponse:
    try:
        return SpeechRequest.model_validate_json(raw)
    except ValidationError as error:
        return _refused(
            RefusalCode.INVALID_REQUEST, _refused_field(error, SpeechRequest)
        )


async def _read_within(request: Request, limit: int) -> bytes | None:
    """The body, or `None` once it exceeds `limit` bytes; the rest is never read."""
    body = bytearray()
    async for chunk in request.stream():
        body += chunk
        if len(body) > limit:
            return None
    return bytes(body)


class _SpeechRoute:
    def __init__(
        self, runtime: Runtime, settings: Settings, callers: SpeechCallers
    ) -> None:
        self._runtime = runtime
        self._settings = settings
        self._callers = callers

    async def speak(self, request: Request) -> Response:
        admitted = await self._callers.admit(request.headers)
        if isinstance(admitted, RefusalCode):
            return _refused(admitted)
        deadline = time.monotonic() + self._settings.busy_wait_seconds
        parsed = await self._checked(request)
        if isinstance(parsed, JSONResponse):
            return parsed
        text, voice = parsed
        if not self._callers.may_name(admitted.app, voice):
            return _refused(RefusalCode.FORBIDDEN, "voice")
        admission = self._admit(voice)
        if isinstance(admission, JSONResponse):
            return admission
        lease = await self._runtime.synthesis_gate.acquire(
            deadline, self._settings.max_waiting_requests
        )
        if lease is None:
            return self._busy()
        chunks = wav_chunks(
            admission.speaking, text, SPEAKING_LANGUAGE, admission.sample_rate
        )
        return ClosingWavResponse(chunks, lease)

    async def _checked(self, request: Request) -> tuple[str, VoiceId] | JSONResponse:
        raw = await _read_within(request, self._settings.max_body_bytes)
        if raw is None:
            return _refused(RefusalCode.TOO_LARGE)
        parsed = _parse(raw)
        if isinstance(parsed, JSONResponse):
            return parsed
        if len(parsed.input) > self._settings.max_input_characters:
            return _refused(RefusalCode.INVALID_REQUEST, "input")
        try:
            return parsed.input, VoiceId(parsed.voice)
        except ValueError:
            return _refused(RefusalCode.INVALID_REQUEST, "voice")

    def _admit(self, voice: VoiceId) -> VoiceAdmission | JSONResponse:
        admission = self._runtime.capture_speaking()
        if admission is None:
            return _refused(
                RefusalCode.UNAVAILABLE,
                background=BackgroundTask(self._runtime.reconcile_failed_active),
            )
        if admission.voice is not voice:
            return _refused(RefusalCode.UNAVAILABLE)
        return admission

    def _busy(self) -> JSONResponse:
        wait = max(1, math.ceil(self._settings.busy_wait_seconds))
        return _refused(RefusalCode.BUSY, headers={RETRY_AFTER_HEADER: str(wait)})


def _replayed(request: Request, body: bytes) -> Request:
    """A request over the same scope whose body is the already read `body`."""
    messages = iter([{"type": "http.request", "body": body, "more_body": False}])

    def receive() -> Awaitable[Message]:
        return asyncio.sleep(0, next(messages, {"type": "http.disconnect"}))

    return Request(request.scope, receive)


def _heard(
    recording: bytes, language: HearingLanguage, max_seconds: float
) -> Callable[[HearingEngine], str]:
    """The job that decodes `recording` and transcribes it."""

    def job(engine: HearingEngine) -> str:
        return engine.transcribe(decode_recording(recording, max_seconds), language)

    return job


class _TranscriptionRoute:
    def __init__(
        self, runtime: Runtime, settings: Settings, callers: SpeechCallers
    ) -> None:
        self._runtime = runtime
        self._settings = settings
        self._callers = callers

    async def transcribe(self, request: Request) -> Response:
        refused = await self._hearing_refusal(request)
        if refused is not None:
            return refused
        deadline = time.monotonic() + self._settings.hearing_busy_wait_seconds
        raw = await _read_within(request, self._settings.max_recording_bytes)
        if raw is None:
            return _refused(RefusalCode.TOO_LARGE)
        parsed = await self._parsed(_replayed(request, raw))
        if isinstance(parsed, JSONResponse):
            return parsed
        recording, fields = parsed
        lease = await self._runtime.hearing.gate.acquire(
            deadline, self._settings.hearing_max_waiting_requests
        )
        if lease is None:
            return self._busy()
        return await self._answer(lease, recording, fields.language)

    async def _hearing_refusal(self, request: Request) -> JSONResponse | None:
        admitted = await self._callers.admit(request.headers)
        if isinstance(admitted, RefusalCode):
            return _refused(admitted)
        if not self._callers.may_hear(admitted.app):
            return _refused(RefusalCode.NOT_ENABLED)
        if self._runtime.hearing.capture() is None:
            return _refused(RefusalCode.UNAVAILABLE)
        return None

    async def _parsed(
        self, request: Request
    ) -> tuple[bytes, TranscriptionRequest] | JSONResponse:
        try:
            async with request.form(max_files=1) as form:
                upload = form.get("file")
                if not isinstance(upload, UploadFile):
                    return _refused(RefusalCode.INVALID_REQUEST, "file")
                fields = _fields(form)
                if isinstance(fields, JSONResponse):
                    return fields
                return await upload.read(), fields
        except HTTPException, FormParserError:
            return _refused(RefusalCode.INVALID_REQUEST, "file")

    async def _answer(
        self, lease: Lease, recording: bytes, language: HearingLanguage
    ) -> Response:
        try:
            job = _heard(recording, language, self._settings.max_recording_seconds)
            text = await asyncio.wrap_future(self._runtime.hearing.submit(lease, job))
        except RecordingTooLongError:
            return _refused(RefusalCode.TOO_LARGE, "file")
        except UndecodableRecordingError:
            return _refused(RefusalCode.INVALID_REQUEST, "file")
        except Exception as error:  # noqa: BLE001 - logged by type and never by message
            log_failure(logger, "hearing failed", error)
            return _refused(RefusalCode.UNAVAILABLE)
        return JSONResponse(Transcription(text=text).model_dump())

    def _busy(self) -> JSONResponse:
        wait = max(1, math.ceil(self._settings.hearing_busy_wait_seconds))
        return _refused(RefusalCode.BUSY, headers={RETRY_AFTER_HEADER: str(wait)})


def _fields(form: FormData) -> TranscriptionRequest | JSONResponse:
    text_fields = {name: value for name, value in form.items() if name != "file"}
    try:
        return TranscriptionRequest.model_validate(text_fields)
    except ValidationError as error:
        return _refused(
            RefusalCode.INVALID_REQUEST, _refused_field(error, TranscriptionRequest)
        )


class _NotARoute:
    """An ASGI app, so the route it serves answers every HTTP method."""

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["path_params"]["version"] == API_VERSION:
            response = _refused(RefusalCode.NOT_FOUND)
        else:
            response = _refused(RefusalCode.UNSUPPORTED_VERSION)
        await response(scope, receive, send)


def mount_wire_routes(
    app: FastAPI, runtime: Runtime, settings: Settings, callers: SpeechCallers
) -> None:
    """Mount the speech and transcription routes, then the catch-alls for the rest."""
    speech = _SpeechRoute(runtime, settings, callers)
    app.add_route(SPEECH_PATH, speech.speak, methods=["POST"], include_in_schema=False)
    hearing = _TranscriptionRoute(runtime, settings, callers)
    app.add_route(
        TRANSCRIPTIONS_PATH,
        hearing.transcribe,
        methods=["POST"],
        include_in_schema=False,
    )
    for path in (_VERSION_PREFIX, f"{_VERSION_PREFIX}/{{rest:path}}"):
        app.add_route(path, _NotARoute(), include_in_schema=False)

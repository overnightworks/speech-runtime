"""The private fixed-phrase sample endpoint shares Runtime synthesis ownership."""

import asyncio
import json
import os
import threading
from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient
from starlette.responses import StreamingResponse

from speech.config import Settings
from speech.pcm import pcm_from_wav
from speech.selection import VoiceSelectionStore
from speech.service import (
    Runtime,
    RuntimeDependencies,
    SampleLanguage,
    create_app,
    create_control_app,
)
from speech.voices import VoiceId, VoiceLoadOutcome
from tests.conftest import FakeHearing, FakeSpeaking
from tests.test_speak import _ASGI_TIMEOUT_SECONDS


class _ClientSendError(RuntimeError):
    """The ASGI client stopped accepting the response before PCM began."""


class _SynthesisError(RuntimeError):
    """The fake provider fails while the sample response owns synthesis."""


class _ResponseConstructionError(RuntimeError):
    """The test-controlled response constructor failed."""


@pytest.mark.parametrize(
    ("language", "phrase", "sample_rate"),
    [
        ("de", "Hallo, ich bin die Stimme deiner Präsentation.", 22_050),
        ("en", "Hello, I am the voice of your presentation.", 24_000),
    ],
)
def test_private_sample_uses_its_fixed_phrase_and_native_rate(
    tmp_path, language: str, phrase: str, sample_rate: int
) -> None:
    speaking = FakeSpeaking()
    speaking.sample_rate = sample_rate
    runtime = Runtime(speaking, FakeHearing())
    settings = Settings(
        voice_cache=tmp_path,
        huggingface_cache=tmp_path,
        PRESENTATOR_RUNTIME_UID=os.geteuid(),
    )

    private = TestClient(create_control_app(settings, runtime)).post(
        f"/voices/piper/sample/{language}"
    )
    public = TestClient(create_app(runtime=runtime)).post("/voices/piper/sample/de")

    assert private.status_code == 200
    assert private.headers["content-type"] == "audio/wav"
    rate, pcm = pcm_from_wav(private.content)
    assert rate == sample_rate
    assert pcm
    assert speaking.heard_text == phrase
    assert speaking.heard_language == language
    assert public.status_code == 404


def test_sample_refuses_a_stale_voice_without_synthesising(tmp_path) -> None:
    speaking = FakeSpeaking()
    runtime = Runtime(speaking, FakeHearing())
    settings = Settings(
        voice_cache=tmp_path,
        huggingface_cache=tmp_path,
        PRESENTATOR_RUNTIME_UID=os.geteuid(),
    )

    response = TestClient(create_control_app(settings, runtime)).post(
        f"/voices/{VoiceId.CHATTERBOX}/sample/en"
    )

    assert response.status_code == 404
    assert speaking.heard_text == ""


def test_sample_waits_for_an_actual_public_speak_stream_then_succeeds(tmp_path) -> None:
    class GatedSpeaking(FakeSpeaking):
        def __init__(self) -> None:
            super().__init__()
            self.entered = threading.Event()
            self.release = threading.Event()

        def pcm_chunks(self, text: str, language: str) -> Iterator[bytes]:
            self.heard_text = text
            self.heard_language = language
            self.entered.set()
            assert self.release.wait(timeout=_ASGI_TIMEOUT_SECONDS)
            yield from super().pcm_chunks(text, language)

    speaking = GatedSpeaking()
    runtime = Runtime(speaking, FakeHearing())
    settings = Settings(
        voice_cache=tmp_path,
        huggingface_cache=tmp_path,
        PRESENTATOR_RUNTIME_UID=os.geteuid(),
    )

    async def scenario() -> int:
        public = create_app(runtime=runtime)
        sent = False
        stopped = asyncio.Event()

        async def receive() -> dict[str, Any]:
            nonlocal sent
            if not sent:
                sent = True
                return {
                    "type": "http.request",
                    "body": json.dumps({"text": "talk", "language": "de"}).encode(),
                    "more_body": False,
                }
            await stopped.wait()
            return {"type": "http.disconnect"}

        async def send(_message: dict[str, Any]) -> None:
            return None

        scope: dict[str, Any] = {
            "type": "http",
            "asgi": {"version": "3.0", "spec_version": "2.3"},
            "http_version": "1.1",
            "method": "POST",
            "path": "/speak",
            "raw_path": b"/speak",
            "query_string": b"",
            "headers": [(b"content-type", b"application/json")],
            "client": ("test", 123),
            "server": ("test", 80),
        }
        speaking_task = asyncio.create_task(public(scope, receive, send))
        assert await asyncio.to_thread(speaking.entered.wait, _ASGI_TIMEOUT_SECONDS)
        waiting = TestClient(create_control_app(settings, runtime)).post(
            "/voices/piper/sample/de"
        )
        speaking.release.set()
        await asyncio.wait_for(speaking_task, timeout=_ASGI_TIMEOUT_SECONDS)
        return waiting.status_code

    assert asyncio.run(scenario()) == 409
    later = TestClient(create_control_app(settings, runtime)).post(
        "/voices/piper/sample/de"
    )
    assert later.status_code == 200


def test_second_sample_refuses_while_the_first_sample_streams(tmp_path) -> None:
    class GatedSpeaking(FakeSpeaking):
        def __init__(self) -> None:
            super().__init__()
            self.entered = threading.Event()
            self.release = threading.Event()

        def pcm_chunks(self, text: str, language: str) -> Iterator[bytes]:
            self.entered.set()
            assert self.release.wait(timeout=_ASGI_TIMEOUT_SECONDS)
            yield from super().pcm_chunks(text, language)

    speaking = GatedSpeaking()
    runtime = Runtime(speaking, FakeHearing())
    settings = Settings(
        voice_cache=tmp_path,
        huggingface_cache=tmp_path,
        PRESENTATOR_RUNTIME_UID=os.geteuid(),
    )
    app = create_control_app(settings, runtime)

    async def scenario() -> int:
        sent = False
        never = asyncio.Event()

        async def receive() -> dict[str, Any]:
            nonlocal sent
            if not sent:
                sent = True
                return {"type": "http.request", "body": b"", "more_body": False}
            await never.wait()
            return {"type": "http.disconnect"}

        async def send(_message: dict[str, Any]) -> None:
            return None

        scope: dict[str, Any] = {
            "type": "http",
            "asgi": {"version": "3.0", "spec_version": "2.3"},
            "http_version": "1.1",
            "method": "POST",
            "path": "/voices/piper/sample/de",
            "raw_path": b"/voices/piper/sample/de",
            "query_string": b"",
            "headers": [],
            "client": ("test", 123),
            "server": ("test", 80),
        }
        first = asyncio.create_task(app(scope, receive, send))
        assert await asyncio.to_thread(speaking.entered.wait, _ASGI_TIMEOUT_SECONDS)
        waiting = TestClient(app).post("/voices/piper/sample/de")
        speaking.release.set()
        await asyncio.wait_for(first, timeout=_ASGI_TIMEOUT_SECONDS)
        return waiting.status_code

    assert asyncio.run(scenario()) == 409


def test_sample_synthesis_error_releases_the_next_sample(tmp_path) -> None:
    class FailingSpeaking(FakeSpeaking):
        def __init__(self) -> None:
            super().__init__()
            self.fails = True

        def pcm_chunks(self, text: str, language: str) -> Iterator[bytes]:
            if self.fails:
                raise _SynthesisError
            yield from super().pcm_chunks(text, language)

    speaking = FailingSpeaking()
    runtime = Runtime(speaking, FakeHearing())
    settings = Settings(
        voice_cache=tmp_path,
        huggingface_cache=tmp_path,
        PRESENTATOR_RUNTIME_UID=os.geteuid(),
    )
    client = TestClient(create_control_app(settings, runtime))

    with pytest.raises(_SynthesisError):
        client.post("/voices/piper/sample/de")
    speaking.fails = False

    assert client.post("/voices/piper/sample/de").status_code == 200


def test_sample_keeps_its_engine_and_rate_when_the_selection_changes(tmp_path) -> None:
    class HeldSpeaking(FakeSpeaking):
        def __init__(self) -> None:
            super().__init__()
            self.entered = threading.Event()
            self.release = threading.Event()

        def pcm_chunks(self, text: str, language: str) -> Iterator[bytes]:
            self.entered.set()
            assert self.release.wait(timeout=_ASGI_TIMEOUT_SECONDS)
            yield from super().pcm_chunks(text, language)

    first = HeldSpeaking()
    first.sample_rate = 22_050
    second = FakeSpeaking()
    second.sample_rate = 24_000
    committed = threading.Event()

    class ObservedSelection(VoiceSelectionStore):
        def write(self, voice: VoiceId) -> None:
            super().write(voice)
            committed.set()

    runtime = Runtime(
        first,
        FakeHearing(),
        dependencies=RuntimeDependencies(
            selection=ObservedSelection(tmp_path / "selection", owner_uid=os.geteuid()),
            engine_factory=lambda _voice: second,
            artifact_checker=lambda _voice: True,
        ),
    )
    settings = Settings(
        voice_cache=tmp_path,
        huggingface_cache=tmp_path,
        PRESENTATOR_RUNTIME_UID=os.geteuid(),
    )
    result: list[bytes] = []

    def sample() -> None:
        result.append(
            TestClient(create_control_app(settings, runtime))
            .post("/voices/piper/sample/de")
            .content
        )

    request = threading.Thread(target=sample)
    request.start()
    assert first.entered.wait(timeout=_ASGI_TIMEOUT_SECONDS)
    outcome: list[VoiceLoadOutcome | None] = []
    finished = threading.Event()

    def switch() -> None:
        outcome.append(runtime.load_voice(VoiceId.CHATTERBOX))
        finished.set()

    switching = threading.Thread(target=switch)
    switching.start()
    assert committed.wait(timeout=_ASGI_TIMEOUT_SECONDS)
    assert not finished.wait(timeout=0.1)
    first.release.set()
    request.join(timeout=_ASGI_TIMEOUT_SECONDS)
    switching.join(timeout=_ASGI_TIMEOUT_SECONDS)
    assert not request.is_alive()
    assert not switching.is_alive()
    assert outcome == [VoiceLoadOutcome.ACTIVATED]
    assert result

    rate, pcm = pcm_from_wav(result[0])
    assert rate == first.sample_rate
    assert pcm
    assert first.heard_text == "Hallo, ich bin die Stimme deiner Präsentation."
    assert second.heard_text == ""


def test_sample_response_construction_failure_releases_admission_for_deselection(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = VoiceSelectionStore(tmp_path / "state", owner_uid=os.geteuid())
    store.write(VoiceId.CHATTERBOX)
    old = FakeSpeaking()
    old.streams = True
    closed: list[int] = []
    old.close = lambda: closed.append(1)
    target = FakeSpeaking()
    runtime = Runtime(
        old,
        FakeHearing(),
        dependencies=RuntimeDependencies(
            selection=store,
            engine_factory=lambda _voice: target,
            artifact_checker=lambda _voice: True,
            default_voice=VoiceId.CHATTERBOX,
        ),
    )
    app = create_control_app(Settings(voice_cache=tmp_path), runtime)
    route = next(
        route
        for route in app.routes
        if getattr(route, "path", None) == "/voices/{voice}/sample/{language}"
    )

    def fail_response_construction(*_args: object, **_kwargs: object) -> None:
        raise _ResponseConstructionError

    monkeypatch.setattr(StreamingResponse, "__init__", fail_response_construction)
    with pytest.raises(_ResponseConstructionError):
        route.endpoint(VoiceId.CHATTERBOX, SampleLanguage.GERMAN)

    outcomes: list[VoiceLoadOutcome | None] = []
    switching = threading.Thread(
        target=lambda: outcomes.append(runtime.load_voice(VoiceId.PIPER))
    )
    switching.start()
    switching.join(timeout=_ASGI_TIMEOUT_SECONDS)
    assert not switching.is_alive()
    assert outcomes == [VoiceLoadOutcome.ACTIVATED]
    assert closed == [1]

    monkeypatch.undo()
    client = TestClient(create_control_app(Settings(voice_cache=tmp_path), runtime))
    response = client.post("/voices/piper/sample/de")
    assert response.status_code == 200


def test_sample_send_failure_before_pcm_releases_the_admitted_voice(tmp_path) -> None:
    async def scenario() -> int:
        runtime = Runtime(FakeSpeaking(), FakeHearing())
        settings = Settings(
            voice_cache=tmp_path,
            huggingface_cache=tmp_path,
            PRESENTATOR_RUNTIME_UID=os.geteuid(),
        )
        app = create_control_app(settings, runtime)
        received = False

        async def receive() -> dict[str, Any]:
            nonlocal received
            if not received:
                received = True
                return {"type": "http.request", "body": b"", "more_body": False}
            await never.wait()
            return {"type": "http.disconnect"}

        async def send(_message: dict[str, Any]) -> None:
            raise _ClientSendError

        never = asyncio.Event()
        scope: dict[str, Any] = {
            "type": "http",
            "asgi": {"version": "3.0", "spec_version": "2.3"},
            "http_version": "1.1",
            "method": "POST",
            "path": "/voices/piper/sample/de",
            "raw_path": b"/voices/piper/sample/de",
            "query_string": b"",
            "headers": [],
            "client": ("test", 123),
            "server": ("test", 80),
        }
        with pytest.raises(_ClientSendError):
            await app(scope, receive, send)
        return TestClient(app).post("/voices/piper/sample/de").status_code

    assert asyncio.run(scenario()) == 200


def test_sample_pre_body_disconnect_releases_the_voice(tmp_path) -> None:
    async def scenario() -> int:
        runtime = Runtime(FakeSpeaking(), FakeHearing())
        settings = Settings(
            voice_cache=tmp_path,
            huggingface_cache=tmp_path,
            PRESENTATOR_RUNTIME_UID=os.geteuid(),
        )
        app = create_control_app(settings, runtime)

        async def receive() -> dict[str, Any]:
            return {"type": "http.disconnect"}

        async def send(_message: dict[str, Any]) -> None:
            return None

        scope: dict[str, Any] = {
            "type": "http",
            "asgi": {"version": "3.0", "spec_version": "2.3"},
            "http_version": "1.1",
            "method": "POST",
            "path": "/voices/piper/sample/de",
            "raw_path": b"/voices/piper/sample/de",
            "query_string": b"",
            "headers": [],
            "client": ("test", 123),
            "server": ("test", 80),
        }
        await app(scope, receive, send)
        return TestClient(app).post("/voices/piper/sample/de").status_code

    assert asyncio.run(scenario()) == 200

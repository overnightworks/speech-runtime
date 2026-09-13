"""The private Unix-socket voice status, download, Load, and sample controls."""

from enum import StrEnum

from fastapi import FastAPI, HTTPException

from speech.config import Settings
from speech.service import Runtime, SynthesisBusyError
from speech.voices import VoiceDownloadOutcome, VoiceId, VoiceLoadOutcome, VoiceSnapshot
from speech.wav_response import ClosingWavResponse


class SampleLanguage(StrEnum):
    """The two fixed phrases the private sample endpoint can synthesize."""

    GERMAN = "de"
    ENGLISH = "en"


_SAMPLE_TEXT: dict[SampleLanguage, str] = {
    SampleLanguage.GERMAN: "Hallo, ich bin die Stimme deiner Präsentation.",
    SampleLanguage.ENGLISH: "Hello, I am the voice of your presentation.",
}


def create_control_app(settings: Settings, runtime: Runtime) -> FastAPI:
    """Build the private control surface over the shared runtime."""
    app = FastAPI(title="presentator-speech-control", docs_url=None, redoc_url=None)

    @app.get("/voices")
    def voices() -> VoiceSnapshot:
        return runtime.snapshot(settings)

    @app.post("/voices/{voice}/load")
    def load(voice: VoiceId) -> dict[str, VoiceLoadOutcome]:
        outcome = runtime.load_voice(voice)
        if outcome is None:
            raise HTTPException(
                status_code=409, detail="voice transition is in progress"
            )
        return {"outcome": outcome}

    @app.post("/voices/qwen3-tts-0.6b/download")
    def download() -> dict[str, VoiceDownloadOutcome]:
        outcome = runtime.start_qwen_download()
        if outcome is None:
            raise HTTPException(status_code=409, detail="Qwen download is unavailable")
        return {"outcome": outcome}

    @app.post("/voices/{voice}/sample/{language}")
    def sample(voice: VoiceId, language: SampleLanguage) -> ClosingWavResponse:
        try:
            voice_sample = runtime.preacquire_sample(voice)
        except SynthesisBusyError:
            raise HTTPException(status_code=409, detail="speech is busy") from None
        if voice_sample is None:
            raise HTTPException(status_code=404, detail="active voice is unavailable")
        chunks = voice_sample.wav_chunks(_SAMPLE_TEXT[language], language.value)
        return ClosingWavResponse(chunks, voice_sample)

    return app

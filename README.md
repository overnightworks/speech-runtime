# Local speech

Audience: the operator starting this process on the home server, and the
co-presenter that will call it.

One process holds a German voice and a German listener resident, and offers
them over HTTP. Nothing in `src/presentator` imports this package. The caller
owns the address.

## Models

| Role | Hugging Face | Licence | Why this one |
| --- | --- | --- | --- |
| Speaking | [`rhasspy/piper-voices`](https://huggingface.co/rhasspy/piper-voices) voice `de_DE-thorsten-medium` | Voice recordings CC0; ONNX weights under the piper-voices MIT repo. The `piper-tts` runtime is `GPL-3.0-or-later`. The HTTP boundary keeps it out of `src/presentator` while the speech program itself carries the GPL obligations. | German is documented and not in question. #70 measured this voice on this 3090 at 0.23 s to first audio on CPU. `thorsten-high` on the same sentence took 1.21 s through this service, over the one-second first-audio target, so medium is the default. The voice does not compete with the listener for the card. |
| Hearing | [`Systran/faster-whisper-large-v3`](https://huggingface.co/Systran/faster-whisper-large-v3) | MIT (CTranslate2 conversion of [`openai/whisper-large-v3`](https://huggingface.co/openai/whisper-large-v3), also MIT) | Already cached on this machine. German is a documented language. Partials are chunked re-decode of a growing buffer, not a native streaming architecture. |

Checked and not chosen:

- **Kokoro / Kokoro-German** (`Tundragoon/Kokoro-German`, Apache-2.0). Official Kokoro has no documented German. The community German fine-tune warns it is undertrained.
- **XTTS-v2** and maintained forks. Weights under the Coqui Public Model License (non-commercial). ADR 0004 already excluded them.
- **Orpheus** (`Thorsten-Voice/tv-orpheus-v1`, Apache-2.0 on the fine-tune; some GGUF conversions inherit Llama 3.2). A 3B SNAC talker, roughly 8 GB resident. First audio for a short sentence is the LLM's first codec frame, not a VITS chunk; not measured on this card, and it would sit on the GPU beside Whisper.
- **Chatterbox Multilingual** (MIT, German documented). #70 could not load its 3.2 GB weights in time; Piper already met the one-second first-audio target without touching the card.
- **Voxtral Mini 4B Realtime** (Apache-2.0, documented German, native streaming, sub-500 ms in the paper). Not cached here, wants vLLM, and ships no server that matches this contract. ADR 0004 still names it as the M3 spike.

The listener is not a streaming-native model. Kyutai STT has no German checkpoint. Voxtral is the current streaming-native candidate and was left for that spike.

## Card

Piper runs on the CPU. Whisper large-v3 float16 on the 3090 is the GPU resident. Together they fit beside the display processes. Songmaker was not using the GPU when this was measured; nothing was stopped.

Measured on this RTX 3090 (24 GB), German sentence of 14 words, `scripts/prove.py`, 08.09.2026:

| What | Number |
| --- | --- |
| Time to first byte of `/speak` (wire, first HTTP body byte) | 0.216 s (`speaking.streams` is `false`; Piper allows the first byte when the sentence is done; limit 0.5 s) |
| Total `/speak` | 0.226 s |
| WAV duration / RMS | 5.515 s / 0.173 (not silence) |
| First partial transcript | 0.968 s (frame 7 of 100 ms frames), while frames were still being sent |
| Final transcript | matched the spoken sentence at 9.354 s |
| Second utterance on the same socket | `"Die nächste Folie bitte."` at 3.027 s, no reload |
| Card before load | 1557 MiB |
| Card with both resident | 5437 MiB |
| Hearing footprint | ~3880 MiB GPU |
| Speaking footprint | 0 MiB GPU (CPU / ONNX) |

CTranslate2 does not bundle CUDA. This project installs `nvidia-cublas-cu12` and `nvidia-cudnn-cu12` and preloads them at start.

Weights stay in:

- Piper voices: `~/.cache/piper/`
- Whisper: `~/.cache/huggingface/hub/models--Systran--faster-whisper-large-v3`

## Contract

- `GET /health` → `{"speaking": {"model", "ready", "streams"}, "hearing": {"model", "ready"}, "sample_rate", "card_memory_mb"}`. Answers while models are still loading, with `ready` false. `speaking.streams` is whether the voice yields PCM while it is still synthesising: `false` for Piper (one completed chunk per sentence, so the whole waveform exists before the first byte leaves), `true` for a model that yields as it goes.
- `POST /speak` with `{"text", "language"}` → chunked `audio/wav`, 16-bit PCM mono. One sentence per request. The first bytes leave as early as the model allows; chunked transfer is kept so a later streaming voice can use it. The WAV header carries Piper's native rate (22 050 Hz for Thorsten).
- `WS /hear?language=de` takes binary frames of raw 16-bit PCM mono at `sample_rate` (16 000 Hz) and sends `{"text", "final"}`. The socket stays open; the model is not reloaded between utterances.

`sample_rate` is the hear rate. Speak is a WAV, so its rate is in the header. A caller that feeds speak output into hear must resample.

## Settings

All `SPEECH_*`:

| Variable | Default | Meaning |
| --- | --- | --- |
| `SPEECH_HOST` | `127.0.0.1` | Bind address |
| `SPEECH_PORT` | `8090` | Bind port |
| `SPEECH_DEVICE` | `cuda` | Device for the hearing model (`cuda` or `cpu`). The voice stays on CPU. |
| `SPEECH_SPEAKING_MODEL` | `de_DE-thorsten-medium` | Piper catalogue name |
| `SPEECH_HEARING_MODEL` | `Systran/faster-whisper-large-v3` | Hugging Face id or faster-whisper size name |
| `SPEECH_DEBUG` | `false` | When true, logs the text of what was spoken or heard. Audio is never logged. |
| `SPEECH_VOICE_CACHE` | `~/.cache/piper` | Where Piper ONNX files are kept |

If a model cannot be loaded the process exits and names which one failed.

## Run

From this directory:

```sh
uv sync --group dev
uv run presentator-speech
```

Stage proof (takes `/tmp/probe-stack.lock`, checks load, one heavy step at a time):

```sh
uv run python scripts/prove.py
```

Checks:

```sh
uv run ruff check src tests scripts
uv run ruff format --check src tests scripts
uv run pytest -q
```

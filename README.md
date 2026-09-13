# Local speech

Audience: the operator starting this process on the home server, and the
co-presenter that will call it.

One process holds a German voice and a German listener resident, and offers
them over HTTP. Nothing in `src/presentator` imports this package. The caller
owns the address. Settings · Voice chooses between downloaded Piper and
Chatterbox; the durable choice is private state, and neither loader downloads
weights.

## Models

| Role | Hugging Face | Licence | Why this one |
| --- | --- | --- | --- |
| Speaking (default) | [`rhasspy/piper-voices`](https://huggingface.co/rhasspy/piper-voices) voice `de_DE-thorsten-medium` | Voice recordings CC0; ONNX weights under the piper-voices MIT repo. The `piper-tts` runtime is `GPL-3.0-or-later`. The HTTP boundary keeps it out of `src/presentator` while the speech program itself carries the GPL obligations. | German is documented and not in question. #70 measured this voice on this 3090 at 0.23 s to first audio on CPU. `thorsten-high` on the same sentence took 1.21 s through this service, over the one-second first-audio target, so medium is the default. The voice does not compete with the listener for the card. |
| Speaking (optional) | [`ResembleAI/chatterbox`](https://huggingface.co/ResembleAI/chatterbox) Multilingual V3, snapshot `5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18` | MIT (`chatterbox-tts` 0.1.7) | Real German, streams PCM while it synthesises, 24 kHz. Installed `from_pretrained` takes only `device` and loads V2; this service loads `t3_mtl23ls_v3.safetensors` from the local Hub cache, pinned to the snapshot it was measured against. `resemble-perth` 1.0.1's `PerthImplicitWatermarker` needs `pkg_resources`, which setuptools stopped shipping at 82; this project pins `setuptools==81.0.0` so Chatterbox's own construction of the real watermarker succeeds instead of silently degrading to `perth`'s no-op. The streamed path calls `s3gen` directly rather than the model's own `generate()`, so it never reaches the line that applies the watermark to the waveform — unmarked audio either way, now for a stated reason rather than a substituted no-op. |
| Hearing | [`Systran/faster-whisper-large-v3`](https://huggingface.co/Systran/faster-whisper-large-v3) | MIT (CTranslate2 conversion of [`openai/whisper-large-v3`](https://huggingface.co/openai/whisper-large-v3), also MIT) | Already cached on this machine. German is a documented language. Partials are chunked re-decode of a growing buffer, not a native streaming architecture. |

Checked and not chosen:

- **Kokoro / Kokoro-German** (`Tundragoon/Kokoro-German`, Apache-2.0). Official Kokoro has no documented German. The community German fine-tune warns it is undertrained.
- **XTTS-v2** and maintained forks. Weights under the Coqui Public Model License (non-commercial). ADR 0004 already excluded them.
- **Orpheus** (`Thorsten-Voice/tv-orpheus-v1`, Apache-2.0 on the fine-tune; some GGUF conversions inherit Llama 3.2). A 3B SNAC talker, roughly 8 GB resident. First audio for a short sentence is the LLM's first codec frame, not a VITS chunk; not measured on this card, and it would sit on the GPU beside Whisper.
- **Qwen3-TTS-0.6B** (`Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice`, Apache-2.0). The next speaking candidate if Chatterbox's voice does not hold up under the operator's ear; it wants FlashAttention-2.
- **Nemotron 3.5 ASR Streaming 0.6B** (`nvidia/nemotron-3.5-asr-streaming-0.6b`). The completed compiled fp16 direct-model experiment on this RTX 3090 missed the partial-transcript target: across five runs, German first/stable-partial median and final WER were 0.949 s / 17.24% at lookahead 0 and 0.879 s / 15.52% at lookahead 6; English was 0.892 s / 6.78% and 1.054 s / 5.08%, respectively. [The durable result](https://github.com/overnightworks/agent-presentator/issues/77#issuecomment-5592496732) is not a `/hear` measurement, socket integration, or joint-residency proof, so faster-whisper remains selected.
- **Voxtral Mini 4B Realtime** (Apache-2.0, documented German, native streaming, sub-500 ms in the paper). Not cached here, wants vLLM, and ships no server that matches this contract. ADR 0004 still names it as the M3 spike.

The listener is not a streaming-native model. Kyutai STT has no German checkpoint. Voxtral is the current streaming-native candidate and was left for that spike.

## Card

Piper runs on the CPU. Chatterbox Multilingual V3 and Whisper large-v3 float16
share the 3090 when Chatterbox is selected. Together they fit beside the
display processes.

Measured on this RTX 3090 (24 GB), one German sentence (14 words) and one
English sentence (15 words), `scripts/prove.py`, 5 repetitions each, median
reported, 08.09.2026. The operator's live speech process (Piper + Whisper,
~6.4 GB) was resident on the card throughout both runs and was not stopped.

Piper default:

| What | Number |
| --- | --- |
| Time to first byte of `/speak` (wire, German, median of 5) | 0.216 s (`speaking.streams` is `false`; Piper allows the first byte when the sentence is done; limit 0.5 s) |
| Time to first byte (English, median of 5) | 0.210 s |
| Real-time factor (German / English, median of 5) | 25.4× / 24.2× |
| WAV RMS | not silence on every repetition |
| First partial transcript | 1.116 s (frame 8 of 100 ms frames), while frames were still being sent |
| Final transcript | matched the spoken sentence at 9.559 s |
| Second utterance on the same socket | `"Die nächste Folie bitte."` at 3.267 s, no reload |
| Card before this process | 6437 MiB |
| Card with both resident (before any `/speak` call) | 10311 MiB |
| Hearing footprint | ~3874 MiB GPU |
| Speaking footprint | 0 MiB GPU (CPU / ONNX) |

`SPEECH_SPEAKING_MODEL=ResembleAI/chatterbox`, same sentences, same card, with
the real `PerthImplicitWatermarker` constructing (`resource_filename` and the
implicit net both need `pkg_resources`, restored by pinning `setuptools`):

| What | Number |
| --- | --- |
| Time to first byte of `/speak` (wire, German, median of 5) | 0.534 s (`speaking.streams` is `true`; limit 1.0 s) |
| Time to first byte (English, median of 5) | 0.523 s |
| Real-time factor (German / English, median of 5) | 0.83× / 0.82× |
| WAV RMS | not silence on every repetition |
| First partial transcript | 0.966 s (frame 7), while frames were still being sent |
| Final transcript | matched the spoken sentence at 7.901 s |
| Second `/speak` while the first streams | both WAVs distinct speech; Runtime serialises request synthesis |
| Card before this process | 6296 MiB |
| Card with Chatterbox + Whisper resident (before any `/speak` call) | 13465 MiB |
| Chatterbox + hearing footprint together | ~7169 MiB GPU (Chatterbox itself ~3.5 GB, matching #70's inference for a 0.5B model beside Whisper's ~3.9 GB) |

`scripts/prove.py` now fails the run the moment any single repetition, in
either language, misses its first-byte limit — a passing run is the proof
that every repetition held, not just the medians. All ten repetitions (5
German, 5 English) stayed under the 1.0 s limit on this run; the worst single
repetition was 0.555 s. Constructing the real watermarker at load time cost no
measurable per-request latency, because the streamed path never calls it.
Load now synthesises a short German warmup so `ready` means the first real
sentence is already in budget — a cold first request before that warmup
existed took 1.362 s. `scripts/prove.py` writes the German WAV of the run it
just proved to `/tmp/issue-83-voice/german.wav` for a listening judgment.

CTranslate2 does not bundle CUDA. This project installs `nvidia-cublas-cu12` and `nvidia-cudnn-cu12` and preloads them at start.

Weights stay in:

- Piper voices: `~/.cache/piper/`
- Chatterbox: `~/.cache/huggingface/hub/models--ResembleAI--chatterbox` (pinned snapshot `5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18`, `CHATTERBOX_REVISION` in the shared provider contract)
- Whisper: `~/.cache/huggingface/hub/models--Systran--faster-whisper-large-v3`

## Contract

- `GET /health` → `{"speaking": {"model", "ready", "streams", "sample_rate"}, "hearing": {"model", "ready"}, "sample_rate", "card_memory_mb"}`. Answers while models are still loading, with `ready` false. `speaking.streams` is whether the voice yields PCM while it is still synthesising: `false` for Piper (one completed chunk per sentence, so the whole waveform exists before the first byte leaves), `true` for a model that yields as it goes. `speaking.sample_rate` is the WAV rate.
- `POST /speak` with `{"text", "language"}` → chunked `audio/wav`, 16-bit PCM mono. One sentence per request. The first bytes leave as early as the model allows. The WAV header carries the voice's native rate (22 050 Hz for Thorsten, 24 000 Hz for Chatterbox), also reported as `speaking.sample_rate`. Runtime serialises request synthesis across both engines. If the client disconnects mid-stream, the response closes its synthesis generator once the ASGI layer's in-flight step returns, releasing the voice for the next request.
- `WS /hear?language=de` takes binary frames of raw 16-bit PCM mono at `sample_rate` (16 000 Hz) and sends `{"text", "final"}`. The socket stays open; the model is not reloaded between utterances.
- `GET /voices` exists only on `speech.sock` in `SPEECH_PRIVATE_DIRECTORY`. It
  reports the five fixed admin catalogue rows, typed recovery detail, and local
  artifact evidence. `POST /voices/{piper|chatterbox}/load` is private too; it
  synchronously loads an already-downloaded baseline, atomically persists the
  choice, and retains loaded baseline engines for the process lifetime.
  `POST /voices/{piper|chatterbox}/sample/{de|en}` returns one fixed WAV only
  when its named voice is still active and ready; contention returns 409. The
  public TCP application has no `/voices` route.

`sample_rate` is the hear rate. Speak is a WAV, so its rate is in the header. A caller that feeds speak output into hear must resample.

## Settings

All `SPEECH_*`:

| Variable | Default | Meaning |
| --- | --- | --- |
| `SPEECH_HOST` | `127.0.0.1` | Bind address |
| `SPEECH_PORT` | `8090` | Bind port |
| `SPEECH_DEVICE` | `cuda` | Device for the hearing model and for Chatterbox (`cuda` or `cpu`). Piper stays on CPU. |
| `SPEECH_SPEAKING_MODEL` | `de_DE-thorsten-medium` | First-run default only, used when `selected-voice` is absent |
| `SPEECH_HEARING_MODEL` | `Systran/faster-whisper-large-v3` | Hugging Face id or faster-whisper size name |
| `SPEECH_DEBUG` | `false` | When true, logs the text of what was spoken or heard. Audio is never logged. |
| `SPEECH_VOICE_CACHE` | `~/.cache/piper` | Where Piper ONNX files are kept |
| `SPEECH_HUGGINGFACE_CACHE` | Provider Hub cache (normally `~/.cache/huggingface/hub`) | The shared local Hub cache for Chatterbox loading and status lookup |
| `SPEECH_PRIVATE_DIRECTORY` | `/run/presentator-speech` | Private directory that owns `speech.sock` |
| `SPEECH_STATE_DIRECTORY` | `~/.local/state/presentator-speech` | Private directory containing the durable selected voice |

When `SPEECH_HUGGINGFACE_CACHE` is absent, `HF_HUB_CACHE` and `HF_HOME` influence the provider default. An explicit `SPEECH_HUGGINGFACE_CACHE` value is already the Hub-cache root.

`PRESENTATOR_RUNTIME_UID` is the same positive runtime UID the co-presenter
uses. Speech creates its private directory at mode 0700 and its socket at 0600;
an existing socket or an unsafe directory refuses startup.

If a model cannot be loaded the process exits and names which one failed.

## Run

From this directory:

```sh
uv sync --group dev
uv run presentator-speech
```

Stage proof (takes `/tmp/probe-stack.lock`, checks load, one heavy step at a
time). It selects Chatterbox when the durable selection is absent:

```sh
uv run python scripts/prove.py
```

Checks:

```sh
uv run ruff check src tests scripts
uv run ruff format --check src tests scripts
uv run pytest -q
```
Chatterbox runs in its own provider virtual environment. The speech service
talks to it only through its private binary protocol; its model libraries do
not enter the service process.

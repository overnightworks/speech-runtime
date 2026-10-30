# Speech runtime

Audience: the operator starting this process on the home server, and
Presentator, which calls it.

One process holds a German voice resident and offers `GET /health` and the
speak wire over HTTP on a private Unix socket whose caller is Presentator. The
caller owns the address. The voice is the one `SPEECH_SPEAKING_MODEL` names:
Piper, Chatterbox, Qwen3-TTS 0.6B, VoxCPM2 or NVIDIA Magpie, from weights that
are already cached. The service stores no choice of its own.

The code was extracted, with its history, from `speech/` in
[overnightworks/presentator](https://github.com/overnightworks/presentator).

## Models

| Role | Hugging Face | Licence | Why this one |
| --- | --- | --- | --- |
| Speaking (default) | [`rhasspy/piper-voices`](https://huggingface.co/rhasspy/piper-voices) voice `de_DE-thorsten-medium` | Voice recordings CC0; ONNX weights under the piper-voices MIT repo. The `piper-tts` runtime is `GPL-3.0-or-later`. The HTTP boundary keeps it out of `src/speech_runtime_service` while the speech program itself carries the GPL obligations. | German is documented and not in question. #70 measured this voice on this 3090 at 0.23 s to first audio on CPU. `thorsten-high` on the same sentence took 1.21 s through this service, over the one-second first-audio target, so medium is the default. The voice does not compete with the listener for the card. |
| Speaking (optional) | [`ResembleAI/chatterbox`](https://huggingface.co/ResembleAI/chatterbox) Multilingual V3, snapshot `5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18` | MIT (`chatterbox-tts` 0.1.7) | Real German, streams PCM while it synthesises, 24 kHz. Installed `from_pretrained` takes only `device` and loads V2; this service loads `t3_mtl23ls_v3.safetensors` from the local Hub cache, pinned to the snapshot it was measured against. `resemble-perth` 1.0.1's `PerthImplicitWatermarker` needs `pkg_resources`, which setuptools stopped shipping at 82; this project pins `setuptools==81.0.0` so Chatterbox's own construction of the real watermarker succeeds instead of silently degrading to `perth`'s no-op. The streamed path calls `s3gen` directly rather than the model's own `generate()`, so it never reaches the line that applies the watermark to the waveform — unmarked audio either way, now for a stated reason rather than a substituted no-op. |
| Speaking (optional) | [`Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice`](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice), snapshot `85e237c12c027371202489a0ec509ded67b5e4b5` | Apache-2.0 (`qwen-tts` 0.1.1) | German and English through one configured built-in preset, initially Ryan. It returns a complete 24 kHz waveform, so `speaking.streams` is false. The isolated provider accepts the repository's `cuda` setting as `cuda:0`, uses bfloat16 and package-default attention, and reads only the pinned local snapshot. Runtime quality and latency remain unmeasured. |
| Speaking (optional) | [`openbmb/VoxCPM2`](https://huggingface.co/openbmb/VoxCPM2), snapshot `32279effe8c19989596f05d353d1447f51d9e915` | Apache-2.0 (`voxcpm` 2.0.3) | German and English text-only streaming at 48 kHz through its isolated provider. It reads only the pinned local snapshot, uses no prompt, reference audio, cloning, or voice-design controls, and has no measured quality, latency, residency, or deployment result. |
| Speaking (optional) | [`nvidia/magpie_tts_multilingual_357m`](https://huggingface.co/nvidia/magpie_tts_multilingual_357m), revision `5023df68bd3f5b5ce6d666a50979bc501af145cc` with NanoCodec revision `fc00890b604aa2de298d2641ffc6c5f6caf8c4d7` | NVIDIA Open Model License (`nemo-toolkit[tts]` 3.0.0) | German and English through one configured fixed voice, initially Sofia, at 22,050 Hz. It restores both pinned archives from the local Hub cache only and also needs `microsoft/wavlm-base-plus` and `google/byt5-small` there, which nothing pins yet (#2). It installs on Python 3.14 with torch 2.15; its provider worker has spoken German on the GPU once (30.10.2026, torch 2.15.0+cu132, NVIDIA driver 595.99.02), and has not yet run through the service. Hearing, quality, and latency against Piper are unmeasured. |

Checked and not chosen:

- **Kokoro / Kokoro-German** (`Tundragoon/Kokoro-German`, Apache-2.0). Official Kokoro has no documented German. The community German fine-tune warns it is undertrained.
- **XTTS-v2** and maintained forks. Weights under the Coqui Public Model License (non-commercial). ADR 0004 already excluded them.
- **Orpheus** (`Thorsten-Voice/tv-orpheus-v1`, Apache-2.0 on the fine-tune; some GGUF conversions inherit Llama 3.2). A 3B SNAC talker, roughly 8 GB resident. First audio for a short sentence is the LLM's first codec frame, not a VITS chunk; not measured on this card.

## Card

Piper runs on the CPU. Chatterbox Multilingual V3 runs on the 3090 when it is
the configured voice and fits beside the display processes. Qwen's source
integration makes no residency or latency claim before the target-machine proof.

Measured on this RTX 3090 (24 GB), one German sentence (14 words) and one
English sentence (15 words), `scripts/prove_speech.py`, 5 repetitions each, median
reported, 08.09.2026.

Piper default:

| What | Number |
| --- | --- |
| Time to first byte of `/speak` (wire, German, median of 5) | 0.216 s (`speaking.streams` is `false`; Piper allows the first byte when the sentence is done; limit 0.5 s) |
| Time to first byte (English, median of 5) | 0.210 s |
| Real-time factor (German / English, median of 5) | 25.4× / 24.2× |
| WAV RMS | not silence on every repetition |
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
| Second `/speak` while the first streams | both WAVs distinct speech; Runtime serialises request synthesis |

`scripts/prove_speech.py` now fails the run the moment any single repetition
misses its first-byte limit — a passing run is the proof
that every repetition held, not just the medians. All ten repetitions (5
German, 5 English) stayed under the 1.0 s limit on this run; the worst single
repetition was 0.555 s. Constructing the real watermarker at load time cost no
measurable per-request latency, because the streamed path never calls it.
Load now synthesises a short German warmup so `ready` means the first real
sentence is already in budget — a cold first request before that warmup
existed took 1.362 s. `scripts/prove_speech.py` writes the German WAV of the run it
just proved to `german.wav` in a private temporary directory the script prints for a listening judgment.

The service installs `nvidia-cublas-cu12` and `nvidia-cudnn-cu12` and preloads them at start.

Weights stay in:

- Piper voices: `~/.cache/piper/`
- Chatterbox: `~/.cache/huggingface/hub/models--ResembleAI--chatterbox` (pinned snapshot `5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18`, `CHATTERBOX_REVISION` in the shared provider contract)
- Qwen: `~/.cache/huggingface/hub/models--Qwen--Qwen3-TTS-12Hz-0.6B-CustomVoice` (pinned snapshot `85e237c12c027371202489a0ec509ded67b5e4b5`)
- VoxCPM2: `~/.cache/huggingface/hub/models--openbmb--VoxCPM2` (pinned snapshot `32279effe8c19989596f05d353d1447f51d9e915`)

## Contract

- `GET /health` → `{"speaking": {"model", "ready", "streams", "sample_rate"}, "card_memory_mb"}`. Answers while the model is still loading, with `ready` false. `speaking.streams` is whether the voice yields PCM while it is still synthesising: `false` for Piper (one completed chunk per sentence, so the whole waveform exists before the first byte leaves), `true` for a model that yields as it goes. `speaking.sample_rate` is the WAV rate.
- `POST /v1/audio/speech` → chunked `audio/wav`, German, with the AI mark. It needs a workload token and the person's token. The request, the tokens, the refusals and the settings that shape them are in [`docs/wire.md`](docs/wire.md).
- `/health` and `/v1/audio/speech` answer only on `speech.sock` in
  `SPEECH_PRIVATE_DIRECTORY`; the process binds no TCP port. The socket serves
  nothing else (no `/docs`, no `/openapi.json`, no voice administration, no
  hearing).

The service never writes spoken text to its log. A speaking failure is logged with its exception types and stack frames, never with an exception message.

## Settings

All `SPEECH_*`:

| Variable | Default | Meaning |
| --- | --- | --- |
| `SPEECH_DEVICE` | `cuda` | Device for the isolated providers. Chatterbox and VoxCPM2 accept `cuda` or `cpu`; Qwen accepts only `cuda` and maps it to `cuda:0`. Piper stays on CPU. |
| `SPEECH_SPEAKING_MODEL` | `de_DE-thorsten-medium` | The one voice the process starts and speaks: `de_DE-thorsten-medium` (Piper), `ResembleAI/chatterbox`, `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice`, `openbmb/VoxCPM2` or `nvidia/magpie_tts_multilingual_357m`. Its weights must already be cached; the process exits naming the model when they are not |
| `SPEECH_VOICE_CACHE` | `~/.cache/piper` | Where Piper ONNX files are kept |
| `SPEECH_HUGGINGFACE_CACHE` | Provider Hub cache (normally `~/.cache/huggingface/hub`) | The shared local Hub cache for provider loading and the artifact check |
| `SPEECH_QWEN_SPEAKER` | `Ryan` | One of Vivian, Serena, Uncle_Fu, Dylan, Eric, Ryan, Aiden, Ono_Anna, or Sohee |
| `SPEECH_MAGPIE_SPEAKER` | `Sofia` | One of Aria, Jason, John, Leo, or Sofia |
| `SPEECH_PRIVATE_DIRECTORY` | `/run/speech-runtime` | Private directory that owns `speech.sock` |
| `SPEECH_BUSY_WAIT_SECONDS` | `5.0` | How long a second speech request waits for the one synthesis before it answers 429 `busy`; at least 0, where 0 refuses at once |
| `SPEECH_MAX_INPUT_CHARACTERS` | `4096` | The longest `input` a speech request may carry; a whole number of at least 1 |
| `SPEECH_CONFIG_FILE` | none, required | Absolute path of the TOML file that names the issuer, the service group, the apps whose tokens the speak route trusts and the voices each app may name; the process does not start without it ([`docs/wire.md`](docs/wire.md)) |

When `SPEECH_HUGGINGFACE_CACHE` is absent, `HF_HUB_CACHE` and `HF_HOME` influence the provider default. An explicit `SPEECH_HUGGINGFACE_CACHE` value is already the Hub-cache root.

`PRESENTATOR_RUNTIME_UID` is the positive runtime UID that Presentator
shares with speech. Speech creates its private directory at mode 0700 and its socket at 0600.
That UID guards the socket: the Presentator container and every process under
it may reach the socket, and the speak route also needs both tokens.
A dead leftover socket is replaced; a socket with a live listener, a foreign
or non-socket path, or an unsafe directory refuses startup.

If the voice cannot be loaded the process exits and names the model.

## Run

From the repository root:

```sh
uv sync --group dev
uv run speech-runtime-service
```

Stage proof (checks load, one heavy step at a time). The script runs itself as
`flock -o -w 1200 /tmp/probe-stack.lock setpriv --pdeathsig TERM timeout -k 30 1800 …`, so it waits at most
20 minutes for `/tmp/probe-stack.lock` and fails the run, saying the lock is busy,
when that wait ends; the load wait and the proof then share the 30-minute cap,
whose TERM stops the speech server and its providers and removes the private
directory. If its caller dies, `setpriv` hands `timeout` a TERM at once, so
the lock stays held only while the stack tears down; a host without `setpriv`
fails the run, naming it. It starts Chatterbox unless `SPEECH_SPEAKING_MODEL` is set and points
`SPEECH_PROVIDER_ROOT` at `providers` unless the caller set it, so each
provider needs its virtual environment first (`uv sync --locked` inside
`providers/<provider>`). `SPEECH_DEVICE` (default `cuda`) passes through; on
`cpu` the card is hidden from speech with an empty `CUDA_VISIBLE_DEVICES`. Weights must already be cached, because
`HF_HUB_OFFLINE=1` is the default and resolves `refs/main`:

```sh
uv run python scripts/prove_speech.py
```

The proof serves its own throwaway key set over local HTTPS and speaks with
fresh tokens minted from it; nothing of it outlives the run.

Every run reports load to ready, the card's compute pids at health ready and
the peak RSS of the speech process, and ends with `PROOF_OK` when health and
speaking held. On the CPU with Piper:

```sh
SPEECH_DEVICE=cpu SPEECH_SPEAKING_MODEL=de_DE-thorsten-medium \
  OMP_NUM_THREADS=4 uv run python scripts/prove_speech.py
```

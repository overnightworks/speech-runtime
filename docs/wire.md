# The speech wire

Audience: an app developer who speaks through this service.

One request speaks one text: `POST /v1/audio/speech` on the private socket
(`speech.sock` in `SPEECH_PRIVATE_DIRECTORY`). It is a strict subset of
OpenAI's speech endpoint, so an OpenAI-shaped client can build the body.
`speech_runtime_client.wire` (the `overnightworks-speech-runtime-client`
distribution) owns the path, the request type and the refusal list; build the
body and read the refusals through it.

## Request

A JSON object with exactly these fields, all required, all strict JSON types:

| Field | Accepted value |
|---|---|
| `model` | `speech-runtime`. Any other value is refused, so a request naming another model is never answered by this one. |
| `input` | Text with at least one letter or digit, at most `SPEECH_MAX_INPUT_CHARACTERS` characters (default 4096; the operator may raise it above OpenAI's 4096). |
| `voice` | The id of the voice that is active and ready: `piper`, `chatterbox`, `qwen3-tts-0.6b`, `voxcpm2` or `nvidia-magpie`. The service speaks with the active voice only; no request switches it. |
| `response_format` | `wav`. OpenAI defaults an absent field to `mp3`, so it is required here. |

The service speaks German only. There is no `language` field, and any field
not listed here (`speed`, `instructions`, `stream_format`, `language`, ...) is
refused.

## Response

`200` with `Content-Type: audio/wav`, chunked: 16-bit mono PCM at the voice's
native rate, which is in the WAV header. Every WAV carries an AI mark, a RIFF
`LIST`/`INFO` `ICMT` chunk before `data`. The status and headers leave before
synthesis, so a synthesis failure cannot change the status: the server aborts
the connection without the chunked terminator. Read the body to its end and
treat a missing end as a failed request; never use a partial WAV.

## Refusals

Every refusal is `Content-Type: application/json` in OpenAI's error form,
`{"error": {"message", "type", "param", "code"}}`. `message` is a fixed
sentence per code and never echoes the request. `type` is
`invalid_request_error` for a 4xx and `server_error` for a 5xx. `param` is one
of `model`, `input`, `voice`, `response_format`, or `null` when no request
field is at fault (an unknown field, a body that is not a JSON object). Apps
read `code`.

| `code` | Status | When | Header |
|---|---|---|---|
| `unsupported_version` | 400 | any method on `/v{n}` or `/v{n}/...` with `n` other than 1 | |
| `not_found` | 404 | any other path or method under version 1 | |
| `invalid_request` | 422 | the body is not a JSON object, a field is missing, unknown or mistyped, a value is not accepted, or `input` is empty, without a letter or digit, or over the limit | |
| `busy` | 429 | the one synthesis is not free within `SPEECH_BUSY_WAIT_SECONDS` | `Retry-After`: the wait in whole seconds, at least 1 |
| `unavailable` | 503 | the named voice is not the active, ready voice (loading, failed, switched away, or shutting down) | |

`n` is read as a decimal integer, so `/v01/...` is version 1. A path that is no
version path (`/v-1/...`, `/vx/...`, `/speak`) gets the app's default 404 with
`{"detail": "Not Found"}`, outside the wire.

## Settings that shape the wire

- `SPEECH_BUSY_WAIT_SECONDS` (default `5.0`, at least 0): how long a request
  waits for the synthesis before `busy`.
- `SPEECH_MAX_INPUT_CHARACTERS` (default `4096`, at least 1): the longest
  `input`.

## Not offered

Version 1 has no hearing and no voice administration on the wire:
`/v1/audio/transcriptions` and `/v1/voices` answer `not_found`. `/hear` and
`/voices*` stay on the private socket, unchanged.

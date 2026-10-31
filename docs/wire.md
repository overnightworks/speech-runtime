# The speech wire

Audience: an app developer who speaks through this service.

One request speaks one text: `POST /v1/audio/speech` on the TCP address
`SPEECH_HOST:SPEECH_PORT`. It is a strict subset of
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
| `voice` | The id of the voice that is active and ready: `piper`, `chatterbox`, `qwen3-tts-0.6b`, `voxcpm2` or `nvidia-magpie`. The voice must be listed for the calling app (`voices` in its configuration) and be the active one. The service speaks with the active voice only; no request switches it. |
| `response_format` | `wav`. OpenAI defaults an absent field to `mp3`, so it is required here. |

The service speaks German only. There is no `language` field, and any field
not listed here (`speed`, `instructions`, `stream_format`, `language`, ...) is
refused.

## Headers

A speak request carries the app's workload token in `Speech-Workload-Token`
and the person's token as `Authorization: Bearer <token>`.
`speech_request_headers` in `speech_runtime_client.wire` builds them with the
content type, and `wire.py` owns the header names. The service answers only
when both tokens are valid and belong together:

- Both come from the configured issuer, name the service in `aud`, and are
  unexpired. The app requests the service's audience scope in its workload
  grant and at the person's sign-in.
- The workload token carries the configured realm role, comes from the app's
  registered client (`azp` and `client_id`) and lives at most 300 seconds.
- The person's token has the same `azp`, a different subject, membership of
  `/apps/<app>` or a group below it, and membership of exactly the configured
  service group (no subgroup of it).

The service checks the headers before it reads the body, and the wait of
`busy` starts only after the tokens are accepted. A catalogue voice the app does
not list is refused as `forbidden` after the body checks and before the active
voice is checked, so the answer never says which voice is active; an unknown
voice stays `invalid_request`. Calls an app makes as itself,
without a person, are not offered in version 1.

## Admission

The scarce resource is the one synthesis a process runs at a time. A request
that finds it taken may wait, within two limits, and anything beyond them is
refused before any work starts:

- `SPEECH_MAX_WAITING_REQUESTS` places for requests that wait for the synthesis.
  A request that finds the synthesis taken and every place taken is refused at
  once with `busy` and `Retry-After`.
- `SPEECH_BUSY_WAIT_SECONDS`: how long a waiting request waits. When it runs out,
  the answer is the same `busy` with `Retry-After`.
- `SPEECH_MAX_BODY_BYTES`: the largest request body. The service stops reading
  at the limit, whatever `Content-Length` says, and answers `too_large` without
  parsing the body. It answers only a caller whose tokens are accepted.

A refused request holds no place, so the next one is served.

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
| `unauthorized` | 401 | a token is missing, malformed, expired, for another issuer or audience, badly signed, or lacks a subject or a valid group claim | `WWW-Authenticate`: `Bearer` |
| `forbidden` | 403 | both tokens are valid but do not permit the request: not a workload token, an unregistered app, a workload token that lives too long, tokens of different clients or of one subject, a person outside the app group or the service group, or a voice the app's configuration does not list (`param` `voice`) | |
| `invalid_request` | 422 | the body is not a JSON object, a field is missing, unknown or mistyped, a value is not accepted, or `input` is empty, without a letter or digit, or over the limit | |
| `too_large` | 413 | the body is over `SPEECH_MAX_BODY_BYTES` | |
| `busy` | 429 | the one synthesis is taken and `SPEECH_MAX_WAITING_REQUESTS` requests already wait (answered at once), or it is not free within `SPEECH_BUSY_WAIT_SECONDS` | `Retry-After`: the wait in whole seconds, at least 1 |
| `unavailable` | 503 | the tokens cannot be checked because the issuer's key set is unreachable and no cached key serves, or the named voice is not the active, ready voice (loading, failed, switched away, or shutting down) | |

`n` is read as a decimal integer, so `/v01/...` is version 1. A path that is no
version path (`/v-1/...`, `/vx/...`, `/speak`) gets the app's default 404 with
`{"detail": "Not Found"}`, outside the wire.

## Settings that shape the wire

- `SPEECH_BUSY_WAIT_SECONDS` (default `5.0`, at least 0): how long a request
  waits for the synthesis before `busy`.
- `SPEECH_MAX_WAITING_REQUESTS` (default `3`, at least 0, where 0 allows no
  waiting): how many requests wait for the synthesis.
- `SPEECH_MAX_BODY_BYTES` (default `32768`, at least 1): the largest body. It
  holds a 4096-character `input` even when every character is escaped as a
  six-byte `\uXXXX`, plus the other fields.
- `SPEECH_MAX_INPUT_CHARACTERS` (default `4096`, at least 1): the longest
  `input`.
- `SPEECH_CONFIG_FILE` (absolute path, required): a TOML file that says whom the
  service trusts. The service does not start without a valid file. The
  `[identity]` table holds `issuer`, `jwks_url` (https; it may carry a
  credential and is never printed), `audience`, `workload_role`, `group_claim`,
  `service_group` (a group path such as `/services/speech-service`) and
  `leeway_seconds` (default 0). Each `[apps.<name>]` table, at least one,
  holds `workload_client`, the client id of that app's workload token; a client
  belongs to one app, and the app group `/apps/<name>` follows from the name; and
  `voices`, required, at least one catalogue id: the voices that app may name.
  An unknown id is refused at start. Unknown keys are refused.

## Health

`GET /health` answers without a token, so a supervisor can poll it:
`{"speaking": {"model", "ready", "streams", "sample_rate"}, "card_memory_mb"}`.
It answers while the voice is still loading, with `ready` false.

## Not offered

Version 1 has no hearing and no voice administration: `/v1/audio/transcriptions`
and `/v1/voices` answer `not_found`, and the service mounts no `/hear` and no
`/voices*` route at all.

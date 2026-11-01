# The speech wire

Audience: an app developer who speaks, or has audio heard, through this
service.

One request speaks one text: `POST /v1/audio/speech` on the TCP address
`SPEECH_HOST:SPEECH_PORT`. It is a strict subset of
OpenAI's speech endpoint, so an OpenAI-shaped client can build the body.
One request hears one recording: `POST /v1/audio/transcriptions`, a strict
subset of OpenAI's transcription endpoint, described under
[Hearing a recording](#hearing-a-recording). Live audio is heard over the
Realtime transcription WebSocket `/v1/realtime?intent=transcription`, described
under [Hearing live](#hearing-live).
`speech_runtime_client.wire` (the `overnightworks-speech-runtime-client`
distribution) owns the paths, the request types, the transcription answer, the
Realtime events and the refusal list; build the body and read the refusals through it.

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

Speaking and hearing each have one scarce resource, the one synthesis and the
one decode a process runs at a time, each under its own limits; a request is
refused only by the limits of its own resource. For speaking, a request that
finds the synthesis taken may wait, within two limits, and anything beyond
them is refused before any work starts:

- `SPEECH_MAX_WAITING_REQUESTS` places for requests that wait for the synthesis.
  A request that finds the synthesis taken and every place taken is refused at
  once with `busy` and `Retry-After`.
- `SPEECH_BUSY_WAIT_SECONDS`: how long a waiting request waits. When it runs out,
  the answer is the same `busy` with `Retry-After`.
- `SPEECH_MAX_BODY_BYTES`: the largest request body. The service stops reading
  at the limit, whatever `Content-Length` says, and answers `too_large` without
  parsing the body. It answers only a caller whose tokens are accepted.

A refused request holds no place, so the next one is served.

## Hearing a recording

`POST /v1/audio/transcriptions` answers the words of one recording. It carries
the same two headers as a speak request and is checked in this order, each step
before the next: the tokens, the app's `hear` permission, whether hearing is
available, the body's size, the form, the decode's admission, the recording's
length, the decoding itself.

The body is `multipart/form-data` with exactly these fields:

| Field | Accepted value |
|---|---|
| `file` | The recording, required, in any container and codec PyAV reads (WAV, MP3, Ogg, FLAC, ...). |
| `model` | `speech-runtime`, required. |
| `language` | `de` or `en`; `de` when absent. Any other value is refused. |
| `response_format` | `json`; `json` when absent. |

Any other field (`prompt`, `temperature`, `timestamp_granularities[]`, ...) is
refused. The answer is `200` with `{"text": "..."}`; the same recording gives
the same text.

An app hears only if its table in `SPEECH_CONFIG_FILE` says `hear = true`;
otherwise the answer is `not_enabled`, after the tokens are accepted. Hearing
is `unavailable` while no model is configured (`SPEECH_HEARING_MODEL`), while
it loads, when its load found no CUDA device or no room on the card, and when a
decode fails.

The limits, each answered without decoding more than it must:

- `SPEECH_MAX_RECORDING_BYTES` bounds the request body. The service stops
  reading at the limit, whatever `Content-Length` says, and answers `too_large`
  before it parses the form.
- `SPEECH_MAX_RECORDING_SECONDS` bounds the recording's length. The decoder
  stops at the limit and the answer is `too_large` with `param` `file`; a
  recording longer than the limit is never transcribed in part.
- `SPEECH_HEARING_MAX_WAITING_REQUESTS` places for requests that wait for the
  one decode, and `SPEECH_HEARING_BUSY_WAIT_SECONDS` for how long they wait. A
  request that finds the decode taken and every place taken is refused at once
  with `busy` and `Retry-After`; so is one whose wait runs out. The decode is
  freed only when it has finished, even if the caller has gone.

Audio that PyAV cannot read, or that holds no sound, is `invalid_request` with
`param` `file`. A body that is not a multipart form, or has no `file`, is
`invalid_request` with `param` `file` as well.

## Hearing live

`WS /v1/realtime?intent=transcription` hears audio as it arrives. It speaks the
events of OpenAI's Realtime transcription session that this service applies, and
nothing else. Both tokens travel in the upgrade request's headers, as on the
other routes; a client sends them from a server, never from a browser.

**Upgrade.** A refusal before the session is an HTTP answer in the refusal form
below, in this order: the tokens, the app's `hear` permission (`not_enabled`), the
`intent` (`invalid_request`, `param` `intent`), whether hearing is available
(`unavailable`), the session bound. `SPEECH_MAX_HEARING_SESSIONS` (default `1`)
sessions run at once; one more is `busy` with `Retry-After`. A wrong version is
`unsupported_version`. An idle session holds its place only, never the decode.

**Setup.** The service sends `session.created` with its default session: PCM16
mono at 24 kHz, `de`, the model `speech-runtime`, server turn detection. The
client may send one `session.update` with `session.type` `transcription` before
its first append; the service answers `session.updated` with the applied session.
Accepted: the format absent or exactly `{"type": "audio/pcm", "rate": 24000}`;
`transcription.model` absent or `speech-runtime`; `transcription.language`
absent, `de` or `en`; `turn_detection` absent or `{"type": "server_vad"}`. Anything
else ends the session with an `error`, among it a second update, an update after
audio, `null` turn detection, `languages`, and every field the service does not
apply (prompt, keywords, noise reduction, include).

**Client events.** Only `session.update`, `input_audio_buffer.append` (base64
PCM16 at 24 kHz; invalid base64 or an odd byte count ends the session) and
`input_audio_buffer.commit`. Any other event type, `input_audio_buffer.clear`
included, is a protocol violation. A client `event_id` is optional; the service
echoes it only in an `error` about that event. A commit ends the open turn at
once; a commit with no audio since the last turn ended is a protocol violation.
With server turn detection a client need not commit.

**Turns.** The service ends a turn after speech followed by 0.7 s of silence, or
at 30 s of audio; the rest of the audio opens the next turn. A session holds at
most one final being heard, one ended turn waiting behind it and the open turn,
so at most 90 s of audio.

**Server events**, in the order one turn produces them:

| Event | When |
|---|---|
| `session.created` | first, on every session |
| `session.updated` | after a valid `session.update` |
| `input_audio_buffer.speech_started` (`item_id`, `audio_start_ms`) | speech first heard in the turn |
| `conversation.item.input_audio_transcription.delta` (`item_id`, `content_index` 0, `delta`) | zero or more, before the commit: the words two consecutive decodes agree on. Provisional. |
| `input_audio_buffer.speech_stopped` (`item_id`, `audio_end_ms`) | silence or the cap ended the turn; not sent after a client commit |
| `input_audio_buffer.committed` (`item_id`, `previous_item_id`) | the turn is closed; `previous_item_id` is `null` for the first |
| `conversation.item.created` (`item`, `previous_item_id`) | right after `committed`; the item is a user message with one `input_audio` part whose `transcript` is `null` |
| `conversation.item.input_audio_transcription.completed` (`item_id`, `content_index` 0, `transcript`, `usage`) | the authoritative text, possibly empty, with `usage` `{"type": "duration", "seconds": ...}`; it replaces the deltas |
| `conversation.item.input_audio_transcription.failed` (`item_id`, `content_index` 0, `error`) | only as a session ends |
| `error` (`error.type`, `error.code`, `error.message`, `error.param`, `error.event_id`) | the session ends after it |

Every server event has a fresh `event_id`. Correlate by `item_id`: OpenAI does not
order completions across items; here, with one final per session, they come in turn
order, but do not depend on it.

**Every announced item ends once.** An item is announced by its `speech_started`,
its first delta or its `committed`. While connected it ends with exactly one
`completed` or one `failed`. When the service ends a session (a decode fails or
the engine is lost, a final waited longer than `SPEECH_HEARING_FINAL_WAIT_SECONDS`,
a second ended turn while one waits, or a protocol violation) it sends one `failed`
per announced unfinished item, then one `error`, then closes. The `error`'s `code`
is `invalid_request` (`param` the field at fault), `busy` or `unavailable`. A client
that disconnects gets nothing.

**Finals first.** A turn's final waits for the one decode ahead of every waiting
recording, behind earlier finals. Its deadline runs from the turn's end for
`SPEECH_HEARING_FINAL_WAIT_SECONDS` (default `60`, above 0), the time behind the
session's own earlier final included; a final past it is never started, even on a
free decode, and the session ends `busy`. A partial that finds the decode taken is
skipped. The decode is freed only when it has finished, even if the client has gone.

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
of `file`, `model`, `input`, `voice`, `language`, `response_format`, `intent`, or `null`
when no request field is at fault (an unknown field, a body that is not a JSON
object). Apps read `code`.

| `code` | Status | When | Header |
|---|---|---|---|
| `unsupported_version` | 400 | any method or upgrade on `/v{n}` or `/v{n}/...` with `n` other than 1 | |
| `not_found` | 404 | any other path, method or upgrade under version 1 | |
| `unauthorized` | 401 | a token is missing, malformed, expired, for another issuer or audience, badly signed, or lacks a subject or a valid group claim | `WWW-Authenticate`: `Bearer` |
| `forbidden` | 403 | both tokens are valid but do not permit the request: not a workload token, an unregistered app, a workload token that lives too long, tokens of different clients or of one subject, a person outside the app group or the service group, or a voice the app's configuration does not list (`param` `voice`) | |
| `not_enabled` | 403 | the app's table does not say `hear = true` (the hearing routes only) | |
| `invalid_request` | 422 | the body is not a JSON object, a field is missing, unknown or mistyped, a value is not accepted, or `input` is empty, without a letter or digit, or over the limit; for a recording: the form or `file` is missing, a field is unknown or not accepted, or the audio cannot be decoded (`param` `file`) | |
| `too_large` | 413 | the body is over `SPEECH_MAX_BODY_BYTES` (a recording: `SPEECH_MAX_RECORDING_BYTES`), or the recording is longer than `SPEECH_MAX_RECORDING_SECONDS` (`param` `file`) | |
| `busy` | 429 | the one synthesis (a recording: the one decode) is taken and its waiting places are taken (answered at once), or it is not free within its wait: `SPEECH_MAX_WAITING_REQUESTS` and `SPEECH_BUSY_WAIT_SECONDS`, for a recording `SPEECH_HEARING_MAX_WAITING_REQUESTS` and `SPEECH_HEARING_BUSY_WAIT_SECONDS` | `Retry-After`: the wait in whole seconds, at least 1 |
| `unavailable` | 503 | the tokens cannot be checked because the issuer's key set is unreachable and no cached key serves, or the named voice is not the active, ready voice (loading, failed, switched away, or shutting down); for a recording: hearing is not loaded or configured, or its decode failed | |

Apps read the `busy` wait with `busy_wait(status, headers)` in
`speech_runtime_client.wire`: it returns the wait in whole seconds, `None` for any
other status, and fails loud on a `busy` without whole seconds, at least 1, in `Retry-After`.

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
  `voices`, required, at least one catalogue id: the voices that app may name;
  and `hear`, `false` when absent: whether that app may have recordings
  heard. An unknown id is refused at start. Unknown keys are refused.
- `SPEECH_HEARING_MODEL` (`large-v3` or `small`, none when absent): the hearing
  model the process loads at start. With none, hearing is `unavailable`.
- `SPEECH_HEARING_BUSY_WAIT_SECONDS` (default `30.0`, at least 0) and
  `SPEECH_HEARING_MAX_WAITING_REQUESTS` (default `3`, at least 0): the
  decode's wait and its waiting places, as for speaking.
- `SPEECH_MAX_HEARING_SESSIONS` (default `1`, at least 1): live sessions at
  once; one more is refused at the upgrade as `busy`.
- `SPEECH_HEARING_FINAL_WAIT_SECONDS` (default `60`, above 0): how long a live
  final may wait for the decode, from its turn's end.
- `SPEECH_MAX_RECORDING_BYTES` (default 25 MiB, at least 1) and
  `SPEECH_MAX_RECORDING_SECONDS` (default `300`, above 0): the largest request
  body and the longest recording.

## Health

`GET /health` answers without a token, so a supervisor can poll it:
`{"speaking": {"model", "ready", "streams", "sample_rate"}, "hearing": {"model", "ready"}, "card_memory_mb"}`.
It answers while the voice is still loading, with `ready` false. `hearing.model`
is the configured hearing model, or `unavailable` when none is configured;
`hearing.ready` is false while it loads and for good when its load found no
CUDA device or no room on the card, which leaves speaking up.

## Not offered

Version 1 has no voice administration: `/v1/voices` answers
`not_found`, and the service mounts no `/hear` and no `/voices*` route at all.

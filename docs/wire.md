# The speech wire

Audience: an app developer who speaks, or has audio heard, through this
service.

**WIRE-01** One request speaks one text: `POST /v1/audio/speech` on the TCP address
`SPEECH_HOST:SPEECH_PORT`. It is a strict subset of
OpenAI's speech endpoint, so an OpenAI-shaped client can build the body.

**WIRE-02** One request hears one recording: `POST /v1/audio/transcriptions`, a strict
subset of OpenAI's transcription endpoint, described under
[Hearing a recording](#hearing-a-recording).

**WIRE-03** Live audio is heard over the
Realtime transcription WebSocket `/v1/realtime?intent=transcription`, described
under [Hearing live](#hearing-live).

`speech_client.wire` (the `overnightworks-speech-client`
distribution) owns the paths, the request types, the transcription answer, the
Realtime events and the refusal list; build the body and read the refusals through it.
`speech_client.speak` makes the speaking call itself, within a total deadline and a
byte cap, and answers with a typed outcome; its module documents them. `speak` and
`speech_request_body` take `language`; one the voice lacks, or none for a voice of
several languages, raises the request's validation error before the network.
`speech_client.voice_preference` speaks with an ordered preference of voices instead of
one: it reads health, speaks with the first voice that `speaking.voices` offers as
`ready` in the requested language, sends `instructions` only to a voice that
`takes_instructions` (WIRE-142) and says in its outcome which voice spoke and whether
the instructions were left out. A speak answered `unavailable` (the service switched
voices mid-call) reads health again and speaks with the next voice; with none offered
the outcome is `unavailable` for `voice`. It adds nothing to the wire.
`speech_client.health` fetches `GET /health` (WIRE-122) within a deadline and a
cap and answers with a typed outcome; the outcomes the calls share live in
`speech_client.outcomes`. `speech_client.hearing` runs the live hearing
session: audio in, typed events out, one closed outcome when it ends, within a total
deadline and an opening bound; a service that does not open the session in time ends
it with `Unreachable`.
`speech_client.relay` relays a browser's binary audio frames to such a session for an
app's server and hands the app the typed events, never the browser; it ends the hop
when the browser leaves, breaks a frame rule or the app ends it, and closes the
browser with a code of its own. `speech_client.utterances` builds the session on top
of the relay and hands the app finished utterances (text, language and audio
positions), optionally the text so far, or the code of a failed item or of an
`error` event (`ServiceFailed`).

## Request

**WIRE-04** A JSON object with exactly these fields, all required but `language` and `instructions`, all strict JSON types:

| ID | Field | Accepted value |
|---|---|---|
| WIRE-05 | `model` | `piper` or `qwen3-tts-1.7b`, the engine of `voice`. Any other value is refused, so a request naming another model is never answered by this one. |
| WIRE-06 | `input` | Text with at least one letter or digit, at most `SPEECH_MAX_INPUT_CHARACTERS` characters (default 4096; the operator may raise it above OpenAI's 4096). |
| WIRE-07 | `voice` | A catalogue voice of `model` as a string: `de_DE-thorsten-medium`, `de_DE-kerstin-medium`, `en_US-norman-medium`, `en_US-ljspeech-medium`, `en_GB-cori-medium` under `piper`; `ryan` under `qwen3-tts-1.7b`. `speaking.voices` in health lists those offered. OpenAI's object form is refused. |
| WIRE-09 | `voice` | A GPU voice that is not loaded yet loads for this request, which waits for the load; with WIRE-143 it is loaded at start, so none is. |
| WIRE-10 | `response_format` | `wav` or `opus`. OpenAI defaults an absent field to `mp3`, so it is required here. |
| WIRE-138 | `language` | An extension to OpenAI's request, which has no such field. `de` or `en`, the codes of WIRE-34; `null` counts as not named. A voice of one language speaks it, and the field is optional but, when named, must be that language. A voice of several languages (`ryan`: `de`, `en`) speaks the language named, and the field is required. A value other than `de` or `en`, or a language the voice does not speak, or no language for a voice of several, is refused. |
| WIRE-141 | `instructions` | OpenAI's field of that name: how to speak the text, for example its tone. A string with at least one letter and at most 500 characters; `null` counts as not named, and a request without it speaks as before. Otherwise it is refused. The voice speaks the text in the manner the instructions name. |
| WIRE-142 | `instructions` | `speaking.voices` in health lists for each voice whether it `takes_instructions`: `ryan` does, the Piper voices do not. A request that names instructions for a voice that takes none is refused, never ignored. |

**WIRE-11** Each voice speaks the languages `speaking.voices` in health lists for it, and
any field not listed here (`speed`, `stream_format`, ...) is refused.
**WIRE-139** The request is checked in the order `model`, `voice`, `language`,
`instructions`: a refused `language` or `instructions` is `invalid_request` with `param`
`language` or `instructions` unless an earlier field is refused first, which names that
field, and it is answered before any `unavailable` (WIRE-103).

## Headers

**WIRE-12** A speak request carries the app's workload token in `Speech-Workload-Token`
and the person's token as `Authorization: Bearer <token>`.
`speech_request_headers` in `speech_client.wire` builds them with the
content type, and `wire.py` owns the header names. The service answers only
when both tokens are valid and belong together:

- **WIRE-13** Both come from the configured issuer, name the service in `aud`, and are
  unexpired. The app requests the service's audience scope in its workload
  grant and at the person's sign-in.
- **WIRE-14** The workload token carries the configured realm role and comes from the app's
  registered client (`azp` and `client_id`).
- **WIRE-15** The person's token has the same `azp`, a different subject, membership of
  `/apps/<app>` or a group below it, and membership of exactly the configured
  service group (no subgroup of it).

An agent with its own account is the person: its app exchanges the agent's token for one
of its own client through Keycloak's standard token exchange, and that token is the
person's token above.

**WIRE-16** The service checks the headers before it reads the body, and **WIRE-17** the wait of
`busy` starts only after the tokens are accepted, or, for a request that
waited for its voice to load, when that load ends. **WIRE-19** An unknown voice, or a voice of another model, is
`invalid_request` with `param` `voice`; a catalogue voice that is not offered is
`unavailable` (WIRE-103). **WIRE-137** An unknown model is `invalid_request` with
`param` `model`; `speech` names hearing's model and is refused here. **WIRE-20** Calls an app makes as itself,
without a person, are not offered in version 1.

## Admission

**WIRE-21** Each voice and hearing have one scarce resource, the one synthesis of that
voice and the one decode a process runs at a time, each under its own limits; a
request is refused only by the limits of its own resource. For speaking, **WIRE-22** a
request that finds its voice's synthesis taken may wait, within two limits that
hold for each voice apart, and anything beyond them is refused before any work
starts:

- **WIRE-23** `SPEECH_MAX_WAITING_REQUESTS` places for requests that wait for the synthesis.
  A request that finds the synthesis taken and every place taken is refused at
  once with `busy` and `Retry-After`.
- **WIRE-24** `SPEECH_BUSY_WAIT_SECONDS`: how long a waiting request waits. When it runs out,
  the answer is the same `busy` with `Retry-After`. **WIRE-25** Requests waiting for a GPU
  voice's load hold no place, and their wait starts when the load ends.
- **WIRE-26** `SPEECH_MAX_BODY_BYTES`: the largest request body. The service stops reading
  at the limit, whatever `Content-Length` says, and answers `too_large` without
  parsing the body. **WIRE-27** It answers only a caller whose tokens are accepted.

**WIRE-28** A refused request holds no place, so the next one is served.
**WIRE-132** A speak request whose caller has gone frees its voice for the next request.

## Hearing a recording

`POST /v1/audio/transcriptions` answers the words of one recording. **WIRE-29** It carries
the same two headers as a speak request and **WIRE-30** is checked in this order, each step
before the next: the tokens, whether hearing is
available, the body's size, the decode's admission, the form, the recording's
length, the decoding itself.

**WIRE-31** The body is `multipart/form-data` with exactly these fields:

| ID | Field | Accepted value |
|---|---|---|
| WIRE-32 | `file` | The recording, required, in any container and codec PyAV reads (WAV, MP3, Ogg, FLAC, ...). |
| WIRE-33 | `model` | `speech`, required. |
| WIRE-34 | `language` | `de` or `en`; `de` when absent. Any other value is refused. |
| WIRE-35 | `response_format` | `json`; `json` when absent. |

**WIRE-36** Any other field (`prompt`, `temperature`, `timestamp_granularities[]`, ...) is
refused. **WIRE-37** The answer is `200` with `{"text": "..."}`; **WIRE-38** the same recording gives
the same text, because the service decodes greedily, at temperature 0.

**WIRE-40** Hearing is `unavailable` while no model is configured
(`SPEECH_HEARING_MODEL`), while it loads, when its load found no CUDA device or
no room on the card, and when a decode fails.

The limits, each answered without decoding more than it must:

- **WIRE-41** `SPEECH_MAX_RECORDING_BYTES` bounds the request body. The service stops
  reading at the limit, whatever `Content-Length` says, and answers `too_large`
  before it parses the form.
- **WIRE-42** `SPEECH_MAX_RECORDING_SECONDS` bounds the recording's length. The decoder
  stops at the limit and the answer is `too_large` with `param` `file`; a
  recording longer than the limit is never transcribed in part.
- **WIRE-43** `SPEECH_HEARING_MAX_WAITING_REQUESTS` places for requests that wait for the
  one decode, and `SPEECH_HEARING_BUSY_WAIT_SECONDS` for how long they wait. A
  request that finds the decode taken and every place taken is refused at once
  with `busy` and `Retry-After`; so is one whose wait runs out. **WIRE-44** The decode is
  freed only when it has finished, even if the caller has gone.

**WIRE-45** Audio that PyAV cannot read, or that holds no sound, is `invalid_request` with
`param` `file`. **WIRE-46** A body that is not a multipart form, or has no `file`, is
`invalid_request` with `param` `file` as well. **WIRE-47** The form is parsed only once the
request holds the decode, so one upload is spooled to disk at a time: while the
service is busy, a malformed form is answered `busy` like any other request, and
`invalid_request` once it reaches the decode.

## Hearing live

**WIRE-48** `WS /v1/realtime?intent=transcription` hears audio as it arrives. **WIRE-49** It speaks the
events of OpenAI's Realtime transcription session that this service applies, and
nothing else. **WIRE-50** Both tokens travel in the upgrade request's headers, as on the
other routes; a client sends them from a server, never from a browser.
**WIRE-140** The tokens are checked once, at the upgrade; a session runs on after either
token expires until it ends, so the app ends a hearing when its right to hear ends.

**Upgrade.** **WIRE-51** A refusal before the session is an HTTP answer in the refusal form
below, in this order: the tokens, the
`intent` (`invalid_request`, `param` `intent`), whether hearing is available
(`unavailable`), the session bound. **WIRE-52** `SPEECH_MAX_HEARING_SESSIONS` (default `1`)
sessions run at once; one more is `busy` with `Retry-After`. **WIRE-53** A wrong version is
`unsupported_version`. **WIRE-54** An idle session holds its place only, never the decode.

**Setup.** **WIRE-55** The service sends `session.created` with its default session: PCM16
mono at 24 kHz, `de`, the model `speech`, server turn detection. **WIRE-56** The
client may send one `session.update` with `session.type` `transcription` before
its first append; the service answers `session.updated` with the applied session.
**WIRE-57** Accepted: the format absent or exactly `{"type": "audio/pcm", "rate": 24000}`;
`transcription.model` absent or `speech`; `transcription.language`
absent, `de` or `en`; `transcription.languages` absent or a non-empty array of
distinct `de` or `en`; `turn_detection` absent or `{"type": "server_vad"}`. **WIRE-58** Anything
else ends the session with an `error`, among it a second update, an update after
audio, `null` turn detection, `language` and `languages` together, an empty or
repeating `languages`, a code the service does not hear (`param` ends with the
element's index, such as `...transcription.languages.1`), an explicit `null` for
either field, and every field the service does not apply (prompt, keywords, noise
reduction, include).

**Languages.** **WIRE-135** `transcription.languages` is a strict, ordered set; its
first element is the main language. A `languages` update is echoed by `session.updated`
with `languages` and without `language`; a `language` update, or one that names neither
field, is echoed with `language` and without `languages`. The service hears each ended
turn, and each partial, in exactly one language of the set: the language the identifier
names, or the main language when it does not know. No language outside the set is ever
chosen. A set of one language, like `language`, is decoded in that language with no
language step. This deviates from OpenAI, where `languages` is a hint: here it is a
strict set.

**Client events.** **WIRE-59** Only `session.update`, `input_audio_buffer.append` (base64
PCM16 at 24 kHz) and
`input_audio_buffer.commit`. **WIRE-60** Invalid base64 or an odd byte count ends the session. **WIRE-61** Any other event type, `input_audio_buffer.clear`
included, is a protocol violation. **WIRE-62** A client `event_id` is optional; the service
echoes it only in an `error` about that event. **WIRE-63** A commit ends the open turn at
once; **WIRE-64** a commit with no audio since the last turn ended is a protocol violation.
**WIRE-65** With server turn detection a client need not commit.

**Turns.** **WIRE-66** The service ends a turn after speech followed by 0.5 s of silence, or
at 30 s of audio; the rest of the audio opens the next turn. **WIRE-67** A session holds at
most two ended turns, both queued to be heard from their turn's end, and the
open turn, so at most 90 s of audio.

**Server events**, in the order one turn produces them:

| ID | Event | When |
|---|---|---|
| WIRE-68 | `session.created` | first, on every session |
| WIRE-69 | `session.updated` | after a valid `session.update` |
| WIRE-70 | `input_audio_buffer.speech_started` (`item_id`, `audio_start_ms`) | speech first heard in the turn; the position counts as below |
| WIRE-71 | `conversation.item.input_audio_transcription.delta` (`item_id`, `content_index` 0, `delta`) | zero or more, before the commit: the words two consecutive decodes agree on. Provisional; a delta names no language, and a partial may be heard in another language than its final. |
| WIRE-72 | `input_audio_buffer.speech_stopped` (`item_id`, `audio_end_ms`) | silence or the cap ended the turn, at the position counted as below; not sent after a client commit |
| WIRE-73 | `input_audio_buffer.committed` (`item_id`, `previous_item_id`) | the turn is closed; `previous_item_id` is `null` for the first |
| WIRE-74 | `conversation.item.created` (`item`, `previous_item_id`) | right after `committed`; the item is a user message with one `input_audio` part whose `transcript` is `null` |
| WIRE-75 | `conversation.item.input_audio_transcription.completed` (`item_id`, `content_index` 0, `transcript`, `usage`, `languages`) | the authoritative text, possibly empty, with `usage` `{"type": "duration", "seconds": ...}` and `languages` (WIRE-136); it replaces the deltas |
| WIRE-76 | `conversation.item.input_audio_transcription.failed` (`item_id`, `content_index` 0, `error`) | only as a session ends |
| WIRE-77 | `error` (`error.type`, `error.code`, `error.message`, `error.param`, `error.event_id`) | the session ends after it |

`audio_start_ms` and `audio_end_ms` (WIRE-70, WIRE-72) count milliseconds of the audio
the service received in this session, from its first byte, in the service's 16 kHz time.
They depend on neither the wall clock nor when an event is sent. Turns are contiguous: a
turn's end is its start plus the turn's audio in whole milliseconds, rounded down once
per turn, and the next turn starts at that end. `audio_start_ms` is where the first
appended audio of the turn that was not silence starts, with no pre-roll, so it lies at
or before the first speech by at most one appended frame.
`audio_end_ms` is the end of all audio in the turn: it includes the silence that ended it
(WIRE-66), or lies at the turn's start plus the cap.

**WIRE-136** `completed.languages` is `[{"code": "<code>"}]` with exactly one element, the
language the turn was decoded in: also for a session of one language, for a turn heard in
the main language because the identifier did not know, and for an empty transcript. It is
never `[]`, which deviates from OpenAI, whose `languages` may come back empty. `failed` and
`error` carry no language.

**WIRE-78** Every server event has a fresh `event_id`. Correlate by `item_id`: OpenAI does not
order completions across items; **WIRE-79** here the second ended turn's `completed` waits
for the first's, so they come in turn order, but do not depend on it.

**Every announced item ends once.** **WIRE-80** An item is announced by its `speech_started`,
its first delta or its `committed`. While connected it ends with exactly one
`completed` or one `failed`. **WIRE-81** When the service ends a session (a decode fails or
the engine is lost, a final waited longer than `SPEECH_HEARING_FINAL_WAIT_SECONDS`,
a third ended turn while two are outstanding, or a protocol violation) it sends one `failed`
per announced unfinished item, then one `error`, then closes. **WIRE-82** The `error`'s `code`
is `invalid_request` (`param` the field at fault), `busy` or `unavailable`. **WIRE-83** A client
that disconnects gets nothing.

**Finals first.** **WIRE-84** A turn's final waits for the one decode ahead of every waiting
recording, behind earlier finals. **WIRE-85** Its deadline runs from the turn's end for
`SPEECH_HEARING_FINAL_WAIT_SECONDS` (default `60`, above 0), the time behind the
session's own earlier final included; a final past it is never started, even on a
free decode, and the session ends `busy`. **WIRE-86** A partial that finds the decode taken is
skipped. **WIRE-87** The decode is freed only when it has finished, even if the client has gone.

## Response

**WIRE-88** For `wav`, `200` with `Content-Type: audio/wav`, chunked: 16-bit mono PCM at the voice's
native rate, which is in the WAV header. **WIRE-89** Every WAV carries an AI mark, a RIFF
`LIST`/`INFO` `ICMT` chunk before `data`. **WIRE-90** The status and headers leave before
synthesis, so a synthesis failure cannot change the status: the server aborts
the connection without the chunked terminator. Read the body to its end and
treat a missing end as a failed request; never use partial audio, except to
play it live: a player may play the chunks as they arrive, and a body that
breaks off stops the playback where it broke and ends it failed.

**WIRE-133** For `opus`, `200` with `Content-Type: audio/ogg`, chunked: Ogg Opus
(RFC 7845), mono at 48 kHz whatever the voice's rate, its pages leaving as the voice
speaks. RFC 7845 names `audio/ogg`; OpenAI lists `audio/opus` among its answer types
and names no container, so the media type deviates from OpenAI's. **WIRE-134** Every
Ogg Opus carries the AI mark as `DESCRIPTION=<mark>` in its OpusTags.

## Refusals

**WIRE-91** Every refusal is `Content-Type: application/json` in OpenAI's error form,
`{"error": {"message", "type", "param", "code"}}`. **WIRE-92** `message` is a fixed
sentence per code and never echoes the request. **WIRE-93** `type` is
`invalid_request_error` for a 4xx and `server_error` for a 5xx. **WIRE-94** `param` is one
of `file`, `model`, `input`, `voice`, `language`, `response_format`, `intent`, or `null`
when no request field is at fault (an unknown field, a body that is not a JSON
object). Apps read `code`. With several faults in one request, which field `param`
names is not part of the contract, but for WIRE-139.

| ID | `code` | Status | When | Header |
|---|---|---|---|---|
| WIRE-95 | `unsupported_version` | 400 | any method or upgrade on `/v{n}` or `/v{n}/...` with `n` other than 1 | |
| WIRE-96 | `not_found` | 404 | any other path, method or upgrade under version 1 | |
| WIRE-97 | `unauthorized` | 401 | a token is missing, malformed, expired, for another issuer or audience, badly signed, or lacks a subject or a valid group claim | `WWW-Authenticate`: `Bearer` |
| WIRE-98 | `forbidden` | 403 | both tokens are valid but do not permit the request: not a workload token, an unregistered app, tokens of different clients or of one subject, a person outside the app group or the service group | |
| WIRE-100 | `invalid_request` | 422 | the body is not a JSON object, a field is missing, unknown or mistyped, a value is not accepted, or `input` is empty, without a letter or digit, or over the limit, or `instructions` are not accepted (WIRE-141, WIRE-142); for a recording: the form or `file` is missing, a field is unknown or not accepted, or the audio cannot be decoded (`param` `file`) | |
| WIRE-101 | `too_large` | 413 | the body is over `SPEECH_MAX_BODY_BYTES` (a recording: `SPEECH_MAX_RECORDING_BYTES`), or the recording is longer than `SPEECH_MAX_RECORDING_SECONDS` (`param` `file`) | |
| WIRE-102 | `busy` | 429 | the one synthesis (a recording: the one decode) is taken and its waiting places are taken (answered at once), or it is not free within its wait: `SPEECH_MAX_WAITING_REQUESTS` and `SPEECH_BUSY_WAIT_SECONDS`, for a recording `SPEECH_HEARING_MAX_WAITING_REQUESTS` and `SPEECH_HEARING_BUSY_WAIT_SECONDS` | `Retry-After`: the wait in whole seconds, at least 1 |
| WIRE-103 | `unavailable` | 503 | the tokens cannot be checked because the issuer's key set is unreachable and no cached key serves, or the named voice cannot speak: it is not offered, the start load still runs, its load failed or found its files missing, its worker died while the request waited, or the service is shutting down (`param` `voice`); for a recording: hearing is not loaded or configured, or its decode failed | |

**WIRE-104** Apps read the `busy` wait with `busy_wait(status, headers)` in
`speech_client.wire`: it returns the wait in whole seconds, `None` for any
other status, and fails loud on a `busy` without whole seconds, at least 1, in `Retry-After`.

**WIRE-105** `n` is read as a decimal integer, so `/v01/...` is version 1. **WIRE-106** A path that is no
version path (`/v-1/...`, `/vx/...`, `/speak`) gets the app's default 404 with
`{"detail": "Not Found"}`, outside the wire.

## Settings that shape the wire

- **WIRE-107** `SPEECH_VOICES` (catalogue voice ids separated by commas; every Piper voice
  when absent, none when set empty): the voices offered. A Piper voice loads at start, a
  GPU voice on its first request (WIRE-09), or at start with WIRE-143. An unknown or
  repeated id, or a named Piper voice whose files are missing or fail to load, ends the
  start.
- **WIRE-143** `SPEECH_LOAD_GPU_VOICES_AT_START` (`true` or `false`, default `false`): with
  `true` every offered GPU voice loads at start, after the Piper voices and before hearing,
  and a GPU voice whose files are missing or fail to load ends the start as a Piper voice's
  do. `speaking.ready` (WIRE-124) then waits for it.
- **WIRE-108** `SPEECH_BUSY_WAIT_SECONDS` (default `5.0`, at least 0): how long a request
  waits for its voice's synthesis before `busy`.
- **WIRE-109** `SPEECH_MAX_WAITING_REQUESTS` (default `3`, at least 0, where 0 allows no
  waiting): how many requests wait for each voice's synthesis.
- **WIRE-110** `SPEECH_MAX_BODY_BYTES` (default `32768`, at least 1): the largest body. It
  holds a 4096-character `input` even when every character is escaped as a
  six-byte `\uXXXX`, plus the other fields.
- **WIRE-111** `SPEECH_MAX_INPUT_CHARACTERS` (default `4096`, at least 1): the longest
  `input`.
- **WIRE-112** `SPEECH_CONFIG_FILE` (absolute path, required): a TOML file that says whom the
  service trusts. The service does not start without a valid file. **WIRE-113** The
  `[identity]` table holds `issuer`, `jwks_url` (https; it may carry a
  credential and is never printed), `audience`, `workload_role`, `group_claim`,
  `service_group` (a group path such as `/services/speech-service`). **WIRE-114** Each `[apps.<name>]` table, at least one,
  holds `workload_client`, the client id of that app's workload token; a client
  belongs to one app, and the app group `/apps/<name>` follows from the name.
  Every registered app may name every offered voice and may hear. **WIRE-116** Unknown keys are refused.
- **WIRE-117** `SPEECH_HEARING_MODEL` (`large-v3` or `small`, none when absent): the hearing
  model the process loads at start. With none, hearing is `unavailable`.
- **WIRE-118** `SPEECH_HEARING_BUSY_WAIT_SECONDS` (default `30.0`, at least 0) and
  `SPEECH_HEARING_MAX_WAITING_REQUESTS` (default `3`, at least 0): the
  decode's wait and its waiting places, as for speaking.
- **WIRE-119** `SPEECH_MAX_HEARING_SESSIONS` (default `1`, at least 1): live sessions at
  once; one more is refused at the upgrade as `busy`.
- **WIRE-120** `SPEECH_HEARING_FINAL_WAIT_SECONDS` (default `60`, above 0): how long a live
  final may wait for the decode, from its turn's end.
- **WIRE-121** `SPEECH_MAX_RECORDING_BYTES` (default 25 MiB, at least 1) and
  `SPEECH_MAX_RECORDING_SECONDS` (default `300`, above 0): the largest request
  body and the longest recording.

## Health

**WIRE-122** `GET /health` answers without a token, so a supervisor can poll it:
`{"speaking": {"ready", "voices": {"<voice id>": {"model", "name", "languages", "takes_instructions", "ready", "sample_rate"}}}, "hearing": {"model", "ready"}, "card_memory_mb"}`.
**WIRE-123** It answers while the models are still loading, with `ready` false.
**WIRE-124** `speaking.ready` is the speaking service's own readiness: the start load has
ended, shutdown has not begun, and every offered Piper voice (with WIRE-143, every offered voice)
can speak. With no such voice offered, it is true once the start load ends. **WIRE-125** `speaking.voices` lists
the offered voices by id. Each reports, loaded or not, its `model` (the request's
`model`), its `name`, its `languages` (the codes it speaks), whether it `takes_instructions` (WIRE-142) and its WAV `sample_rate` (a
Piper voice's is 0 until it is loaded). A client reads each id as a string, so a voice id it
does not know, under a model and in languages it knows, does not break its reading of
health. **WIRE-126** A voice's `ready` is true exactly when a request for it
would be spoken now: never during the start load or once shutdown began, and
never for a GPU voice that is not loaded, failed, or whose worker died.
**WIRE-127** The service builds the answer through `Health` in `speech_client.wire`,
which ignores keys it does not know. **WIRE-128** `hearing.model` is the configured hearing
model, or `unavailable` when none is configured; **WIRE-129** `hearing.ready` is false while
it loads and for good when its load found no CUDA device or no room on the card,
which leaves speaking up.

## Not offered

**WIRE-130** Version 1 has no voice administration: `/v1/voices` answers
`not_found`, and the service mounts no `/hear` and no `/voices*` route at all.
**WIRE-131** The service serves no API documentation: `/docs`, `/redoc` and
`/openapi.json` answer `404`.

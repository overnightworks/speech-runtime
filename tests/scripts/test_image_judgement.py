"""The image proof's judgement: what it admits, what it refuses, and how it can fail."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import jwt
import pytest

import image_judgement as judged
from hearing_figures import to_wire_rate
from image_stack import REPOSITORY, LiveOutcome, Reply
from proof_identity import mint_pair
from speech_client.wire import (
    VOICES,
    Health,
    Language,
    RefusalCode,
    Transcription,
    VoiceId,
    refusal,
)
from speech_service.config import CallerConfig, load_caller_config
from speech_service.hearing_models import HearingModel
from speech_service.pcm import pcm_from_wav, wav_header
from speech_service.voices import PIPER_VOICES

if TYPE_CHECKING:
    from proof_identity import Minting

BAKED_CONFIG = REPOSITORY / "deploy" / "speech.toml"
SAMPLE_RATE = 24_000
SPEECH = Path(__file__).parents[1] / "fixtures" / "speech" / "nein_danke.wav"
# One phrase holds less voiced speech than any model's floor, this many hold more.
LONG_ENOUGH_PHRASES = 4
SILENCE = bytes(64)


@pytest.fixture
def config() -> CallerConfig:
    return load_caller_config(BAKED_CONFIG)


@pytest.fixture(params=sorted(load_caller_config(BAKED_CONFIG).apps))
def app(request: pytest.FixtureRequest) -> str:
    return str(request.param)


def test_the_baked_caller_speaks_a_piper_voice(config: CallerConfig, app: str) -> None:
    baked = judged.baked_caller(config, app)

    assert baked.voice in PIPER_VOICES


def test_the_english_voice_of_the_offer_is_a_piper_voice_that_speaks_english() -> None:
    voice = judged.first_english_voice()

    assert voice in PIPER_VOICES
    assert Language.EN in VOICES[voice].languages


def test_tokens_minted_for_the_baked_caller_carry_its_identity(
    config: CallerConfig, minting: Minting, app: str
) -> None:
    baked = judged.baked_caller(config, app)
    pair = mint_pair(judged.minting_for(config, baked, minting))

    workload = jwt.decode(pair.workload, options={"verify_signature": False})
    person = jwt.decode(pair.person, options={"verify_signature": False}, audience=None)
    assert workload["iss"] == config.identity.issuer
    assert workload["azp"] == config.apps[baked.app].workload_client
    assert config.identity.workload_role in workload["realm_access"]["roles"]
    assert config.identity.service_group in person[config.identity.group_claim]
    assert person["aud"] == config.identity.audience


def marked_wav() -> bytes:
    return wav_header(SAMPLE_RATE) + SILENCE


@pytest.mark.parametrize(
    "wav",
    [
        pytest.param(judged.without_mark(marked_wav()), id="no mark"),
        pytest.param(b"not audio at all", id="not a wav"),
        pytest.param(wav_header(SAMPLE_RATE), id="no audio"),
        pytest.param(wav_header(SAMPLE_RATE)[:-8] + SILENCE, id="no data chunk"),
        pytest.param(
            wav_header(SAMPLE_RATE, 0)
            + b"JUNK"
            + len(SILENCE).to_bytes(4, "little")
            + SILENCE,
            id="empty data chunk before another chunk",
        ),
    ],
)
def test_a_wav_without_the_mark_or_the_audio_fails_the_proof(wav: bytes) -> None:
    with pytest.raises(SystemExit):
        judged.require_marked_wav(wav)


def test_a_marked_wav_with_audio_passes_and_loses_its_mark_when_cut() -> None:
    judged.require_marked_wav(marked_wav())

    assert judged.ai_mark_of(judged.without_mark(marked_wav())) is None


def test_the_proof_fails_when_an_action_that_must_fail_does_not() -> None:
    judged.require_fails(
        lambda: judged.require_marked_wav(b""), "an empty answer", naming="not a WAV"
    )

    with pytest.raises(SystemExit):
        judged.require_fails(lambda: None, "nothing", naming="")


def test_the_proof_fails_when_an_action_fails_for_another_reason() -> None:
    def docker_is_down() -> None:
        message = "docker run failed"
        raise SystemExit(message)

    with pytest.raises(SystemExit, match="not as ConnectionRefusedError"):
        judged.require_fails(
            docker_is_down, "a loopback bind", naming="ConnectionRefusedError"
        )


def test_a_refusal_passes_only_as_the_expected_code_with_its_status() -> None:
    body = refusal(RefusalCode.FORBIDDEN).model_dump_json().encode()
    status = refusal(RefusalCode.FORBIDDEN).status

    reply = Reply(status, body)

    judged.require_refusal(reply, RefusalCode.FORBIDDEN)
    with pytest.raises(SystemExit):
        judged.require_refusal(reply, RefusalCode.UNAUTHORIZED)


def test_a_refusal_asked_for_a_field_passes_only_when_it_names_that_field() -> None:
    named = refusal(RefusalCode.UNAVAILABLE, "voice")
    unnamed = refusal(RefusalCode.UNAVAILABLE)

    judged.require_refusal(
        Reply(named.status, named.model_dump_json().encode()),
        RefusalCode.UNAVAILABLE,
        param="voice",
    )
    unnamed_reply = Reply(unnamed.status, unnamed.model_dump_json().encode())

    with pytest.raises(SystemExit, match="expected voice"):
        judged.require_refusal(unnamed_reply, RefusalCode.UNAVAILABLE, param="voice")


@pytest.mark.parametrize(
    "reply",
    [
        pytest.param(Reply(200, marked_wav()), id="audio"),
        pytest.param(
            Reply(200, refusal(RefusalCode.BUSY).model_dump_json().encode()),
            id="status of another code",
        ),
    ],
)
def test_an_answer_that_is_no_matching_refusal_fails_the_proof(
    reply: Reply,
) -> None:
    with pytest.raises(SystemExit):
        judged.require_refusal(reply, RefusalCode.BUSY)


def test_locked_names_come_from_the_export_for_this_platform_and_its_projects() -> None:
    export = """\
# a comment
-e ./packages/client
Some_Package==1.0 ; python_version >= '3.14'
windows-only==1.0 ; sys_platform == 'win32'
webauth @ git+https://example.invalid/webauth@abc
    # via something
"""

    names = judged.locked_distributions(export, lambda path: f"own-{path[-6:]}")

    assert names == {"own-client", "some-package", "webauth"}


def clean_report() -> judged.ImageReport:
    return judged.ImageReport(
        distributions=["Some_Package"],
        user=1000,
        application_files=["deploy/speech.toml"],
        application_files_with_private_keys=[],
        package_stray_files=[],
        package_files_with_private_keys=[],
        voice_hashes={"voice.onnx": "abc"},
        hearing_model_directory="/opt/hearing/snapshots/revision",
        executables={"git": None},
        nvidia_directories=[],
    )


@pytest.mark.parametrize(
    ("change", "fault"),
    [
        ({"distributions": ["Some_Package", "nvidia-cublas-cu12"]}, "extra"),
        ({"distributions": []}, "missing"),
        ({"user": 0}, "root"),
        ({"application_files": ["deploy/.git/config"]}, "holds"),
        ({"application_files": ["tests/test_x.py"]}, "holds"),
        ({"application_files": ["deploy/token.key"]}, "holds"),
        ({"application_files_with_private_keys": ["a.txt"]}, "private key"),
        ({"executables": {"git": "/usr/bin/git"}}, "carries git"),
        ({"nvidia_directories": ["nvidia"]}, "holds nvidia"),
        (
            {"package_stray_files": ["speech_service/local.txt"]},
            "among the packages",
        ),
        (
            {"package_files_with_private_keys": ["speech_x/a.key"]},
            "private key",
        ),
        ({"voice_hashes": {"voice.onnx": "other"}}, "pinned bytes"),
        ({"hearing_model_directory": None}, "hearing model"),
    ],
)
def test_an_image_that_differs_from_the_lock_is_named_by_its_fault(
    change: dict[str, object], fault: str
) -> None:
    report = judged.ImageReport(**{**vars(clean_report()), **change})

    faults = judged.image_faults(report, {"some-package"}, {"voice.onnx": "abc"})

    assert any(fault in line for line in faults)


def test_an_image_as_locked_has_no_fault() -> None:
    assert not judged.image_faults(
        clean_report(), {"some-package"}, {"voice.onnx": "abc"}
    )


def test_a_log_with_spoken_text_or_a_token_is_named() -> None:
    log = "INFO ok\nspoke: Hallo Welt\nheader abc.def.ghi"

    faults = judged.log_faults(log, ["Hallo Welt"], [], ["abc.def.ghi"])
    assert [fault.split()[0] for fault in faults] == ["spoken", "a"]
    assert not judged.log_faults("INFO ok", ["Hallo Welt"], ["gehört"], ["abc.def.ghi"])


def test_a_log_with_heard_text_is_named_and_fails_the_proof() -> None:
    log = "INFO ok\ntranscript: Guten Tag zusammen"

    assert "heard text" in judged.log_faults(log, [], ["Guten Tag zusammen"], [])[0]
    with pytest.raises(SystemExit, match="heard text"):
        judged.require_log_clean(log, [], ["Guten Tag zusammen"], [])
    judged.require_log_clean("INFO ok", [], ["Guten Tag zusammen"], [])


def test_a_log_with_a_live_partial_that_differs_from_the_final_is_named() -> None:
    turn = LiveOutcome("Guten Tag zusammen", ("Guten Tag zusamm",), None)
    log = "INFO ok\npartial: Guten Tag zusamm"
    heard = judged.live_texts(turn)

    with pytest.raises(SystemExit, match="heard text"):
        judged.require_log_clean(log, [], heard, [])


def test_a_log_names_the_count_of_heard_texts_and_never_their_words() -> None:
    log = "Guten Tag zusammen and Auf Wiedersehen"

    (fault,) = judged.log_faults(log, [], ["Guten Tag zusammen", "Auf Wiedersehen"], [])

    assert fault == "2 heard text(s)"


def test_a_live_turn_shows_its_partials_of_a_word_or_more_and_then_its_final() -> None:
    turn = LiveOutcome("Guten Tag zusammen", ("Guten", " ", "Tag"), None)

    assert judged.live_texts(turn) == ["Guten", "Guten Tag zusammen"]


@pytest.mark.parametrize(
    ("health", "alone"),
    [
        ({"speaking": {"ready": True}, "hearing": {"ready": False}}, True),
        ({"speaking": {"ready": False}, "hearing": {"ready": False}}, False),
        ({"speaking": {"ready": True}, "hearing": {"ready": True}}, False),
    ],
    ids=["speaking alone", "still loading", "both"],
)
def test_only_a_voice_ready_with_hearing_not_ready_is_speaking_alone(
    health: dict[str, dict[str, bool]], *, alone: bool
) -> None:
    assert judged.is_speaking_alone(health) is alone


def health_of(*, speaking: bool, hearing: bool, voices: dict[VoiceId, bool]) -> Health:
    """A `/health` answer with these readinesses and offered voices."""
    return Health.model_validate(
        {
            "speaking": {
                "ready": speaking,
                "voices": {
                    voice.value: {
                        "model": VOICES[voice].model,
                        "name": VOICES[voice].name,
                        "languages": VOICES[voice].languages,
                        "ready": ready,
                        "takes_instructions": VOICES[voice].takes_instructions,
                        "sample_rate": 1,
                    }
                    for voice, ready in voices.items()
                },
            },
            "hearing": {"model": "small", "ready": hearing},
            "card_memory_mb": 0,
        }
    )


def test_an_offer_passes_only_as_exactly_the_voices_health_lists() -> None:
    health = health_of(speaking=True, hearing=True, voices={PIPER_VOICES[0]: True})

    judged.require_offer(health, [PIPER_VOICES[0]])
    with pytest.raises(SystemExit, match="offers"):
        judged.require_offer(health, [PIPER_VOICES[0], VoiceId.RYAN])


@pytest.mark.parametrize(
    ("speaking", "hearing", "gpu_voice_ready"),
    [(True, True, True), (False, True, False), (True, False, False)],
    ids=["gpu voice ready", "speaking not ready", "hearing not ready"],
)
def test_a_gpu_voice_down_needs_it_not_ready_beside_ready_speaking_and_hearing(
    *, speaking: bool, hearing: bool, gpu_voice_ready: bool
) -> None:
    voices = {PIPER_VOICES[0]: True, VoiceId.RYAN: gpu_voice_ready}
    down = health_of(
        speaking=True, hearing=True, voices={**voices, VoiceId.RYAN: False}
    )

    judged.require_gpu_voice_down(down, VoiceId.RYAN)
    health = health_of(speaking=speaking, hearing=hearing, voices=voices)

    with pytest.raises(SystemExit, match="down beside a ready service"):
        judged.require_gpu_voice_down(health, VoiceId.RYAN)


def test_only_the_external_networks_of_a_rendered_compose_file_are_named() -> None:
    rendered = {
        "networks": {
            "a": {"name": "outside", "external": True},
            "b": {"name": "inside"},
        }
    }

    assert judged.external_networks(rendered) == ["outside"]


def test_an_nvidia_distribution_in_the_image_is_named_even_when_the_lock_lists_it() -> (
    None
):
    report = judged.ImageReport(
        **{**vars(clean_report()), "distributions": ["Some_Package", "nvidia-nccl"]}
    )

    faults = judged.image_faults(
        report, {"some-package", "nvidia-nccl"}, {"voice.onnx": "abc"}
    )

    assert faults == ["holds nvidia-nccl"]


SENTENCE = "Die Kamera zeigt den Vortragenden, während die Folie vorbereitet wird."
ENGLISH_SENTENCE = "The camera shows the speaker while the next slide is prepared."


def transcription(text: str, status: int = 200) -> Reply:
    return Reply(status, Transcription(text=text).model_dump_json().encode())


@pytest.mark.parametrize(
    ("heard", "words"),
    [
        pytest.param("die kamera zeigt", ["kamera", "zeigt"], id="case does not count"),
        pytest.param("Die den wird", [], id="short words match by accident"),
        pytest.param("Guten Tag zusammen", [], id="another text"),
        pytest.param("", [], id="nothing"),
    ],
)
def test_the_words_of_the_sentence_a_text_holds_leave_out_the_short_ones(
    heard: str, words: list[str]
) -> None:
    assert judged.heard_words(heard, SENTENCE) == words


def test_a_text_without_a_word_of_the_sentence_fails_the_proof() -> None:
    judged.require_heard("Die Kamera", SENTENCE)

    with pytest.raises(SystemExit, match="no word of the spoken sentence"):
        judged.require_heard("Guten Tag zusammen", SENTENCE)


@pytest.mark.parametrize(
    ("reply", "fault"),
    [
        pytest.param(transcription("Guten Tag zusammen"), "no word", id="wrong text"),
        pytest.param(Reply(403, b"{}"), "answered 403", id="a refusal"),
        pytest.param(Reply(500, b"oops"), "answered 500", id="a server error"),
    ],
)
def test_a_recording_answer_that_is_no_transcription_of_the_sentence_fails(
    reply: Reply, fault: str
) -> None:
    with pytest.raises(SystemExit, match=fault):
        judged.require_transcript(reply, SENTENCE)


def test_a_transcription_with_a_word_of_the_sentence_gives_its_text() -> None:
    assert judged.require_transcript(transcription("die Kamera"), SENTENCE) == (
        "die Kamera"
    )


def too_large(param: str | None) -> Reply:
    body = refusal(RefusalCode.TOO_LARGE, param).model_dump_json().encode()
    return Reply(413, body)


def busy(headers: dict[str, str]) -> Reply:
    body = refusal(RefusalCode.BUSY).model_dump_json().encode()
    return Reply(429, body, headers)


@pytest.mark.parametrize(
    "reply",
    [
        pytest.param(transcription("anything"), id="text"),
        pytest.param(too_large("file"), id="too long"),
    ],
)
def test_a_recording_at_the_limit_may_get_text_or_too_large_naming_the_file(
    reply: Reply,
) -> None:
    judged.require_limit_answer(reply)


@pytest.mark.parametrize(
    ("reply", "fault"),
    [
        pytest.param(Reply(500, b"internal server error"), "no wire refusal", id="500"),
        pytest.param(Reply(200, b"{}"), "no transcription", id="200 without text"),
        pytest.param(too_large(None), "byte limit", id="byte limit at the limit"),
        pytest.param(busy({"Retry-After": "30"}), "expected", id="busy alone"),
    ],
)
def test_a_recording_at_the_limit_that_got_anything_else_fails_the_proof(
    reply: Reply, fault: str
) -> None:
    with pytest.raises(SystemExit, match=fault):
        judged.require_limit_answer(reply)


def test_a_burst_may_also_be_told_busy_with_a_retry_after() -> None:
    judged.require_limit_answer(busy({"retry-after": "30"}), busy_allowed=True)


def test_a_burst_told_busy_without_a_retry_after_fails_the_proof() -> None:
    without_retry_after = busy({})

    with pytest.raises(SystemExit, match="Retry-After"):
        judged.require_limit_answer(without_retry_after, busy_allowed=True)


@pytest.mark.parametrize(
    ("after", "fault"),
    [
        pytest.param(judged.TmpUsage(4096, 100), "spool is left", id="a spool is left"),
        pytest.param(judged.TmpUsage(0, 9), "smaller than one recording", id="small"),
    ],
)
def test_a_tmp_that_cannot_hold_a_recording_or_keeps_one_fails_the_proof(
    after: judged.TmpUsage, fault: str
) -> None:
    before = judged.TmpUsage(0, 100)

    with pytest.raises(SystemExit, match=fault):
        judged.require_tmp_clean(before, after, 10)


def test_a_tmp_back_at_its_start_passes() -> None:
    judged.require_tmp_clean(judged.TmpUsage(4096, 10), judged.TmpUsage(4096, 10), 10)


@pytest.mark.parametrize(
    ("outcome", "fault"),
    [
        pytest.param(LiveOutcome(None, (), "Refused"), "Refused", id="error"),
        pytest.param(
            LiveOutcome("Guten Tag zusammen", ("Guten",), None), "no word", id="text"
        ),
    ],
)
def test_a_live_turn_without_a_final_of_the_sentence_fails_the_proof(
    outcome: LiveOutcome, fault: str
) -> None:
    with pytest.raises(SystemExit, match=fault):
        judged.require_final(outcome, SENTENCE)


@pytest.mark.parametrize(
    ("sentence", "final"),
    [
        pytest.param(SENTENCE, "die Kamera", id="german"),
        pytest.param(ENGLISH_SENTENCE, "the camera", id="english"),
    ],
)
def test_a_live_turn_with_a_final_of_the_sentence_passes(
    sentence: str, final: str
) -> None:
    judged.require_final(LiveOutcome(final, (), None), sentence)


def test_a_final_of_the_german_text_does_not_pass_for_the_english_sentence() -> None:
    with pytest.raises(SystemExit, match="no word of the spoken sentence"):
        judged.require_final(LiveOutcome("die Kamera", (), None), ENGLISH_SENTENCE)


def test_a_final_in_another_language_than_expected_fails_the_proof() -> None:
    turn = LiveOutcome("die Kamera", (), None, language="en")

    with pytest.raises(SystemExit, match="names en, expected de"):
        judged.require_language(turn, Language.DE)
    judged.require_language(turn, Language.EN)


def _clip(phrases: int) -> bytes:
    """`phrases` spoken phrases of a Piper voice, one after the other, as a WAV."""
    rate, phrase = pcm_from_wav(SPEECH.read_bytes())
    pcm = to_wire_rate(phrase, rate) * phrases
    return wav_header(SAMPLE_RATE, len(pcm)) + pcm


def test_a_clip_below_the_voiced_floor_cannot_prove_the_language_step() -> None:
    long_enough = _clip(LONG_ENOUGH_PHRASES)
    too_short = _clip(1)

    judged.require_voiced_floor(long_enough, HearingModel.SMALL)
    with pytest.raises(SystemExit, match="voiced seconds"):
        judged.require_voiced_floor(too_short, HearingModel.SMALL)


@pytest.mark.parametrize(
    ("hearing", "passes", "fault"),
    [
        pytest.param({"model": None, "ready": False}, True, "not ready", id="no model"),
        pytest.param(
            {"model": "small", "ready": False}, True, "not ready", id="loading"
        ),
        pytest.param(
            {"model": "large-v3", "ready": True}, True, "not ready", id="another"
        ),
        pytest.param(
            {"model": "small", "ready": True}, False, "healthcheck", id="check"
        ),
    ],
)
def test_hearing_that_is_not_ready_with_the_model_or_unchecked_fails_the_proof(
    hearing: dict[str, object], *, passes: bool, fault: str
) -> None:
    with pytest.raises(SystemExit, match=fault):
        judged.require_hearing_ready(
            {"hearing": hearing}, "small", healthcheck_passes=passes
        )


def test_hearing_ready_with_the_model_and_a_passing_healthcheck_passes() -> None:
    judged.require_hearing_ready(
        {"hearing": {"model": "small", "ready": True}}, "small", healthcheck_passes=True
    )


@pytest.mark.parametrize(
    ("text", "seconds"),
    [("30s", 30), ("1m30s", 90), ("2m", 120), ("1h", 3600)],
)
def test_a_rendered_compose_duration_is_read_in_seconds(
    text: str, seconds: int
) -> None:
    assert judged.duration_seconds(text) == seconds


STARTED = "2026-11-02T20:00:00.123456789Z"


def check(ended: str, exit_code: int) -> dict[str, object]:
    return {"End": ended, "ExitCode": exit_code}


def test_the_seconds_to_healthy_end_at_the_first_passing_check_after_the_start() -> (
    None
):
    log = [
        check("2026-11-02T19:59:00Z", 0),
        check("2026-11-02T20:00:03Z", 1),
        check("2026-11-02T20:00:07.623456789+00:00", 0),
        check("2026-11-02T20:00:30Z", 0),
    ]

    assert judged.seconds_to_healthy(STARTED, log) == pytest.approx(7.5)


@pytest.mark.parametrize(
    "log",
    [
        pytest.param([], id="no check"),
        pytest.param([check("2026-11-02T19:59:00Z", 0)], id="only before the start"),
        pytest.param([check("2026-11-02T20:00:03Z", 1)], id="only failing"),
    ],
)
def test_a_start_with_no_passing_check_after_it_fails_the_proof(
    log: list[dict[str, object]],
) -> None:
    with pytest.raises(SystemExit, match="no healthcheck passed"):
        judged.seconds_to_healthy(STARTED, log)


def test_a_service_that_was_healthy_after_its_start_period_fails_the_proof() -> None:
    judged.require_ready_within(29.9, "30s")

    with pytest.raises(SystemExit, match="past 30s"):
        judged.require_ready_within(30.1, "30s")

"""The proof scripts speak to a running service with minted tokens and nothing else."""

from __future__ import annotations

import asyncio
import http.server
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

import jwt
import pytest

from speech_client.wire import (
    AUTHORIZATION_HEADER,
    CONTENT_TYPE_HEADER,
    VOICES,
    WORKLOAD_TOKEN_HEADER,
    Health,
    SpeechRequest,
    VoiceId,
)
from speech_service.config import Settings
from speech_service.pcm import pcm_from_wav, wav_header
from speech_service.service import Runtime
from speech_service.voices import PIPER_VOICES
from tests.support.fakes_support import SPEAK_SAMPLE_RATE, FakeSpeaking
from tests.support.identity_support import callers_trusting
from tests.support.script_support import script_module
from tests.support.speak_support import FailingSpeaking
from tests.support.tcp_server_support import serving_over_tcp

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from proof_identity import Minting, ProofIdentity

CLIENT = script_module("contract_client")
PROVE = script_module("prove_speech")
HEARING = script_module("prove_hearing")


def test_no_credential_holder_shows_a_key_or_token_when_represented(
    minting: Minting,
) -> None:
    pair = CLIENT.mint_pair(minting)

    shown = [repr(minting), repr(pair)]

    for text in shown:
        assert "-----BEGIN" not in text
        assert "eyJ" not in text


def test_both_proof_scripts_reach_a_running_speech_over_tcp(
    minting: Minting, identity: ProofIdentity
) -> None:
    async def scenario() -> tuple[object, bytes, bytes]:
        runtime = Runtime({VoiceId.THORSTEN: FakeSpeaking()}, memory_probe=lambda: 0)
        callers = callers_trusting(identity.configuration_path, minting)
        async with serving_over_tcp(runtime, callers) as server:
            return (
                await asyncio.to_thread(PROVE.read_health, server.base_url),
                await asyncio.to_thread(
                    CLIENT.speak, server.base_url, "Hallo", VoiceId.THORSTEN, minting
                ),
                await asyncio.to_thread(
                    PROVE.speak_measured,
                    server.base_url,
                    "Hallo",
                    VoiceId.THORSTEN,
                    minting,
                ),
            )

    health, contract_wav, measured = asyncio.run(scenario())

    assert isinstance(health, Health)
    assert pcm_from_wav(contract_wav)[0] == SPEAK_SAMPLE_RATE
    assert pcm_from_wav(measured[0])[0] == SPEAK_SAMPLE_RATE


def test_prove_points_the_service_at_the_repository_provider_environments(
    monkeypatch: pytest.MonkeyPatch, identity: ProofIdentity
) -> None:
    monkeypatch.delenv("SPEECH_PROVIDER_ROOT", raising=False)

    environment = PROVE.speech_environment(8080, VoiceId.THORSTEN, identity)

    provider_root = Path(environment["SPEECH_PROVIDER_ROOT"])
    assert provider_root.is_absolute()
    assert (provider_root / "qwen" / "pyproject.toml").is_file()


@pytest.mark.parametrize(
    "offered",
    [
        frozenset(PIPER_VOICES),
        frozenset({VoiceId.RYAN}),
        frozenset({*PIPER_VOICES, VoiceId.RYAN}),
    ],
    ids=["piper", "gpu", "both"],
)
def test_prove_speaks_the_gpu_voice_the_offer_names_else_piper(
    offered: frozenset[VoiceId],
) -> None:
    on_demand = offered - set(PIPER_VOICES)

    assert PROVE.proof_voice(offered) is next(iter(on_demand), PROVE.PIPER_PROOF_VOICE)


@pytest.mark.parametrize("voice", list(VoiceId))
def test_prove_offers_the_service_the_voice_it_speaks_beside_a_piper_voice(
    monkeypatch: pytest.MonkeyPatch, identity: ProofIdentity, voice: VoiceId
) -> None:
    monkeypatch.setenv("SPEECH_VOICES", VoiceId.RYAN.value)

    environment = PROVE.speech_environment(8080, voice, identity)

    monkeypatch.setenv("SPEECH_VOICES", environment["SPEECH_VOICES"])
    assert Settings().voices == {PROVE.PIPER_PROOF_VOICE, voice}


def test_prove_keeps_a_provider_root_the_caller_set(
    monkeypatch: pytest.MonkeyPatch, identity: ProofIdentity
) -> None:
    monkeypatch.setenv("SPEECH_PROVIDER_ROOT", "/elsewhere")

    environment = PROVE.speech_environment(8080, VoiceId.THORSTEN, identity)

    assert environment["SPEECH_PROVIDER_ROOT"] == "/elsewhere"


def test_prove_defaults_to_the_card_proof(
    monkeypatch: pytest.MonkeyPatch, identity: ProofIdentity
) -> None:
    for name in ("SPEECH_DEVICE", "CUDA_VISIBLE_DEVICES"):
        monkeypatch.delenv(name, raising=False)

    environment = PROVE.speech_environment(8080, VoiceId.THORSTEN, identity)

    assert environment["SPEECH_DEVICE"] == "cuda"
    assert "CUDA_VISIBLE_DEVICES" not in environment


def test_prove_takes_the_device_and_hides_the_card_on_the_cpu(
    monkeypatch: pytest.MonkeyPatch, identity: ProofIdentity
) -> None:
    monkeypatch.setenv("SPEECH_DEVICE", "cpu")

    environment = PROVE.speech_environment(8080, VoiceId.THORSTEN, identity)

    assert environment["SPEECH_DEVICE"] == "cpu"
    assert environment["CUDA_VISIBLE_DEVICES"] == ""


def test_peak_resident_memory_is_read_from_the_process_status_in_mebibytes() -> None:
    """Change: peak resident memory from the process status."""
    status = "Name:\tpython\nVmPeak:\t 9999 kB\nVmHWM:\t 2048000 kB\nVmRSS:\t 1 kB\n"

    assert PROVE.peak_rss_mib(status) == pytest.approx(2000.0)


@pytest.mark.parametrize("arguments", [["--help"], ["--measure"], ["extra"]])
def test_prove_exits_before_any_start_on_help_or_an_unknown_argument(
    arguments: list[str],
) -> None:
    """Never: --help and an unknown argument never start a proof."""
    with pytest.raises(SystemExit) as exit_info:
        PROVE.parse_arguments(arguments)

    assert (exit_info.value.code == 0) == (arguments == ["--help"])


@pytest.mark.parametrize("chunks_before_failing", [0, 1])
@pytest.mark.parametrize("speak", [CLIENT.speak, PROVE.speak_measured])
def test_a_proof_client_rejects_a_body_the_service_aborted(
    minting: Minting,
    speak: Callable[[str, str, str, Minting], object],
    chunks_before_failing: int,
) -> None:
    async def scenario() -> None:
        voice = FailingSpeaking(chunks_before_failing=chunks_before_failing)
        runtime = Runtime({VoiceId.THORSTEN: voice}, memory_probe=lambda: 0)
        async with serving_over_tcp(runtime) as server:
            with pytest.raises(SystemExit):
                await asyncio.to_thread(
                    speak, server.base_url, "Hallo", VoiceId.THORSTEN, minting
                )

    asyncio.run(scenario())


class _RecordedPost(http.server.BaseHTTPRequestHandler):
    """A request the recording server answered, with the body it carried."""

    body: bytes = b""


@contextmanager
def _recording_speech_service() -> Iterator[tuple[str, list[_RecordedPost]]]:
    """A local HTTP server that answers every POST and keeps its requests."""
    seen: list[_RecordedPost] = []

    class Handler(_RecordedPost):
        protocol_version = "HTTP/1.1"

        def do_POST(self) -> None:
            seen.append(self)
            self.body = self.rfile.read(int(self.headers["Content-Length"]))
            self.send_response(200)
            self.send_header("Content-Type", "audio/wav")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            wav = wav_header(SPEAK_SAMPLE_RATE) + b"\x01\x00" * 4
            self.wfile.write(b"%x\r\n%b\r\n0\r\n\r\n" % (len(wav), wav))

        def log_message(self, *_: object) -> None:
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", seen
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def _unverified_subject(token: str) -> str:
    return jwt.decode(token, options={"verify_signature": False})["sub"]


@pytest.mark.parametrize("speak", [CLIENT.speak, PROVE.speak_measured])
def test_each_speaking_script_sends_the_workload_token_and_the_persons_bearer(
    minting: Minting, speak: Callable[..., object]
) -> None:
    with _recording_speech_service() as (base_url, requests):
        speak(base_url, "Hallo", VoiceId.THORSTEN, minting)

    (headers,) = (request.headers for request in requests)
    scheme, _, person = headers[AUTHORIZATION_HEADER].partition(" ")
    assert headers[CONTENT_TYPE_HEADER] == "application/json"
    assert scheme == "Bearer"
    assert _unverified_subject(headers[WORKLOAD_TOKEN_HEADER]) == (
        minting.workload_subject
    )
    assert _unverified_subject(person) == minting.person_subject


@pytest.mark.parametrize(
    "speak",
    [
        pytest.param(
            lambda url, voice, minting: CLIENT.speak(url, "Hallo", voice, minting),
            id="contract client",
        ),
        pytest.param(
            lambda url, voice, minting: HEARING._speaks(url, minting, voice),  # noqa: SLF001 - the probe's speaking step is the unit under test
            id="hearing probe",
        ),
    ],
)
@pytest.mark.parametrize("voice", list(VoiceId))
def test_a_script_speaks_each_voice_in_a_language_the_service_accepts(
    minting: Minting, voice: VoiceId, speak: Callable[[str, VoiceId, Minting], object]
) -> None:
    with _recording_speech_service() as (base_url, requests):
        speak(base_url, voice, minting)

    (request,) = requests
    spoken = SpeechRequest.model_validate_json(request.body)
    assert spoken.spoken_language == VOICES[voice].languages[0]


def test_the_contract_client_command_passes_the_minting_file_path_and_no_secret(
    minting: Minting, identity: ProofIdentity
) -> None:
    command = PROVE.contract_client_command(
        "http://127.0.0.1:1", VoiceId.THORSTEN, identity.minting_path
    )

    assert str(identity.minting_path) in command
    assert all(
        minting.private_key_pem not in part and "eyJ" not in part for part in command
    )
    options = CLIENT.parse_arguments(command[2:])
    assert options.minting_file == identity.minting_path
    assert options.voice == VoiceId.THORSTEN


def test_prove_hands_the_service_its_configuration_and_the_proofs_ca(
    identity: ProofIdentity,
) -> None:
    environment = PROVE.speech_environment(8080, VoiceId.THORSTEN, identity)

    assert environment["SPEECH_CONFIG_FILE"] == str(identity.configuration_path)
    assert environment["SSL_CERT_FILE"] == str(identity.ca_path)


def test_prove_binds_the_service_to_the_loopback_port_it_picked(
    identity: ProofIdentity,
) -> None:
    port = PROVE.free_local_port()

    environment = PROVE.speech_environment(port, VoiceId.THORSTEN, identity)

    assert environment["SPEECH_HOST"] == "127.0.0.1"
    assert environment["SPEECH_PORT"] == str(port)

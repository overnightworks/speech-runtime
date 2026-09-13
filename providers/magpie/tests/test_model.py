"""Magpie uses the real pinned NeMo class shape without loading weights."""

from __future__ import annotations

import queue
from io import BytesIO
from pathlib import Path
from unittest.mock import create_autospec

import pytest
import torch
from omegaconf import OmegaConf
from presentator_speech_provider_contract import (
    MAX_PCM_BYTES,
    Frame,
    FrameKind,
    MagpieSpeaker,
    read_frame,
    serve_commands,
)

from presentator_magpie import model


def test_load_resolves_exact_local_archives_before_overriding_the_codec(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    resolved = [tmp_path / "model.nemo", tmp_path / "codec.nemo"]
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    downloads: list[dict[str, object]] = []
    configuration = OmegaConf.create({"codecmodel_path": "nvidia/original"})
    restored = create_autospec(model.MagpieTTSModel, instance=True)
    restored.output_sample_rate = 22_050

    def download(**arguments: object) -> str:
        downloads.append(arguments)
        return str(resolved.pop(0))

    monkeypatch.setattr(model, "hf_hub_download", download)

    def restore_from(*arguments: object, **keywords: object) -> object:
        calls.append((arguments, keywords))
        return configuration if keywords.get("return_config") else restored

    monkeypatch.setattr(model.MagpieTTSModel, "restore_from", restore_from)

    assert model.load_model("cuda", tmp_path) is restored
    assert downloads == [
        {
            "repo_id": "nvidia/magpie_tts_multilingual_357m",
            "filename": "magpie_tts_multilingual_357m.nemo",
            "revision": "5023df68bd3f5b5ce6d666a50979bc501af145cc",
            "cache_dir": tmp_path,
            "local_files_only": True,
        },
        {
            "repo_id": "nvidia/nemo-nano-codec-22khz-1.89kbps-21.5fps",
            "filename": "nemo-nano-codec-22khz-1.89kbps-21.5fps.nemo",
            "revision": "fc00890b604aa2de298d2641ffc6c5f6caf8c4d7",
            "cache_dir": tmp_path,
            "local_files_only": True,
        },
    ]
    assert calls[0] == ((tmp_path / "model.nemo",), {"return_config": True})
    assert calls[1][1]["restore_path"] == tmp_path / "model.nemo"
    assert calls[1][1]["override_config_path"].codecmodel_path == str(
        tmp_path / "codec.nemo"
    )
    assert calls[1][1]["map_location"] == torch.device("cpu")
    restored.to.assert_called_once_with("cuda")
    restored.eval.assert_called_once_with()


def test_load_rejects_an_unexpected_native_rate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    configuration = OmegaConf.create({"codecmodel_path": "nvidia/original"})
    restored = create_autospec(model.MagpieTTSModel, instance=True)
    restored.output_sample_rate = 24_000

    monkeypatch.setattr(model, "hf_hub_download", lambda **_arguments: str(tmp_path))
    monkeypatch.setattr(
        model.MagpieTTSModel,
        "restore_from",
        lambda *_arguments, **keywords: (
            configuration if keywords.get("return_config") else restored
        ),
    )

    with pytest.raises(ValueError, match="unexpected Magpie sample rate"):
        model.load_model("cpu", tmp_path)

    restored.to.assert_called_once_with("cpu")
    restored.eval.assert_called_once_with()


@pytest.mark.parametrize(("language", "text"), [("de", "Hallo"), ("en", "Hello")])
def test_synthesis_uses_the_fixed_sofia_voice_and_bounded_pcm(
    language: str, text: str
) -> None:
    generated = create_autospec(model.MagpieTTSModel, instance=True)
    generated.do_tts.return_value = (
        torch.tensor([[0.0, 0.5, 2.0, -2.0] + [0.0] * (MAX_PCM_BYTES // 2)]),
        torch.tensor([4]),
    )

    chunks = list(
        model.pcm_chunks(generated, text, language, speaker=MagpieSpeaker.SOFIA)
    )

    generated.do_tts.assert_called_once_with(
        transcript=text,
        language=language,
        apply_TN=False,
        use_cfg=True,
        speaker_index=4,
    )
    assert b"".join(chunks) == b"\x00\x00\xff\x3f\xff\x7f\x00\x80"
    assert all(len(chunk) <= MAX_PCM_BYTES and len(chunk) % 2 == 0 for chunk in chunks)


def test_synthesis_rejects_an_unsupported_language_before_calling_nemo() -> None:
    generated = create_autospec(model.MagpieTTSModel, instance=True)

    with pytest.raises(ValueError, match="unsupported Magpie language"):
        list(model.pcm_chunks(generated, "Bonjour", "fr", speaker=MagpieSpeaker.SOFIA))

    generated.do_tts.assert_not_called()


def test_synthesis_rejects_an_empty_waveform() -> None:
    generated = create_autospec(model.MagpieTTSModel, instance=True)
    generated.do_tts.return_value = (torch.zeros((1, 1)), torch.tensor([0]))

    with pytest.raises(ValueError, match="malformed Magpie waveform"):
        list(model.pcm_chunks(generated, "Hallo", "de", speaker=MagpieSpeaker.SOFIA))


@pytest.mark.parametrize("sample", [float("nan"), float("inf"), float("-inf")])
def test_synthesis_rejects_nonfinite_waveform_values(sample: float) -> None:
    generated = create_autospec(model.MagpieTTSModel, instance=True)
    generated.do_tts.return_value = (torch.tensor([[sample]]), torch.tensor([1]))

    with pytest.raises(ValueError, match="malformed Magpie waveform"):
        list(model.pcm_chunks(generated, "Hallo", "de", speaker=MagpieSpeaker.SOFIA))


@pytest.mark.parametrize(
    ("language", "samples", "length"),
    [("fr", [0.0], 1), ("de", [0.0], 0), ("de", [float("nan")], 1)],
)
def test_shared_worker_fails_and_exits_for_magpie_validation_errors(
    language: str, samples: list[float], length: int
) -> None:
    generated = create_autospec(model.MagpieTTSModel, instance=True)
    generated.do_tts.return_value = (torch.tensor([samples]), torch.tensor([length]))
    commands: queue.Queue[Frame] = queue.Queue()
    commands.put(Frame(FrameKind.SYNTHESIZE, 7, b"\x01Hallo"))
    stream = BytesIO()

    assert (
        serve_commands(
            commands,
            stream,
            lambda generated, text, _protocol_language: model.pcm_chunks(
                generated, text, language, speaker=MagpieSpeaker.SOFIA
            ),
            generated,
        )
        == 1
    )

    stream.seek(0)
    assert read_frame(stream) == Frame(FrameKind.FAILED, 7, b"\x01")
    assert stream.read() == b""

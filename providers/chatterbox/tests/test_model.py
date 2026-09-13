"""The moved model adapter stays offline and streams with fake collaborators."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from presentator_chatterbox_contract import CHATTERBOX_ARTIFACTS, CHATTERBOX_REVISION

from presentator_chatterbox import model


class _SnapshotObservedError(RuntimeError):
    pass


def test_model_load_asks_only_for_the_pinned_local_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    observed: dict[str, object] = {}

    def snapshot_download(**arguments: object) -> str:
        observed.update(arguments)
        raise _SnapshotObservedError

    monkeypatch.setattr("huggingface_hub.snapshot_download", snapshot_download)
    with pytest.raises(_SnapshotObservedError):
        model.load_model("cpu", tmp_path)
    assert observed == {
        "repo_id": "ResembleAI/chatterbox",
        "repo_type": "model",
        "revision": CHATTERBOX_REVISION,
        "allow_patterns": CHATTERBOX_ARTIFACTS,
        "local_files_only": True,
        "cache_dir": tmp_path,
    }


class _FakeHyperParameters:
    start_text_token = 0
    stop_text_token = 1
    start_speech_token = 2
    stop_speech_token = 99


class _FakeOutput:
    past_key_values = None
    logits = torch.zeros(2, 1, 128)


class _FakeSpeechPosition:
    @staticmethod
    def get_fixed_embedding(_step: int) -> torch.Tensor:
        return torch.zeros(1, 1, 4)


class _FakeT3:
    hp = _FakeHyperParameters()
    compiled = True

    def prepare_input_embeds(self, **_arguments: object) -> tuple[torch.Tensor, None]:
        return torch.zeros(2, 1, 4), None

    def speech_emb(self, token: torch.Tensor) -> torch.Tensor:
        return token.float().unsqueeze(-1).expand(token.shape[0], token.shape[1], 4)

    def patched_model(self, **_arguments: object) -> _FakeOutput:
        return _FakeOutput()

    speech_pos_emb = _FakeSpeechPosition()


class _FakeTokenizer:
    def text_to_tokens(self, _text: str, language_id: str) -> torch.Tensor:
        assert language_id == "de"
        return torch.zeros(1, 3, dtype=torch.long)


class _FakeConditionals:
    t3 = object()


class _FakeModel:
    t3 = _FakeT3()
    tokenizer = _FakeTokenizer()
    conds = _FakeConditionals()
    device = "cpu"


def test_speech_tokens_stream_in_order_with_a_short_trailing_chunk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(model, "STREAM_TOKEN_CHUNK", 3)
    script = iter([10, 11, 12, 13, 14, 15, 16, _FakeT3.hp.stop_speech_token])
    monkeypatch.setattr(
        torch,
        "multinomial",
        lambda *_args, **_kwargs: torch.tensor([[next(script)]]),
    )

    chunks = list(model.speech_token_chunks(_FakeModel(), "Hallo", "de"))

    assert [chunk.tolist() for chunk in chunks] == [
        [10, 11, 12],
        [13, 14, 15],
        [16],
    ]
    assert torch.cat(chunks).tolist() == [10, 11, 12, 13, 14, 15, 16]


def test_float_audio_is_clipped_to_signed_little_endian_pcm() -> None:
    pcm = model.float_to_pcm(np.array([-2.0, -1.0, 0.0, 1.0, 2.0]))
    assert np.frombuffer(pcm, dtype="<i2").tolist() == [
        -32768,
        -32767,
        0,
        32767,
        32767,
    ]

"""Chatterbox language tags and the streamed token-chunking loop."""

import pytest
import torch

from speech.chatterbox import _speech_token_chunks, language_id


@pytest.mark.parametrize(
    ("language", "expected"),
    [
        ("de", "de"),
        ("de-DE", "de"),
        ("en_US", "en"),
        ("EN", "en"),
        ("", "de"),
        ("  fr-FR  ", "fr"),
    ],
)
def test_language_id_is_the_iso_code(language: str, expected: str) -> None:
    assert language_id(language) == expected


class _FakeHyperParams:
    start_text_token = 0
    stop_text_token = 1
    start_speech_token = 2
    stop_speech_token = 99


class _FakeOutput:
    past_key_values = None


class _FakeT3:
    """Stands in for `T3`: fixed shapes, no real weights, already "compiled"."""

    hp = _FakeHyperParams()
    compiled = True  # `_ensure_patched_t3` skips real loading when already set.

    def prepare_input_embeds(self, **_kwargs: object) -> tuple[torch.Tensor, None]:
        return torch.zeros(2, 1, 4), None

    def speech_emb(self, token: torch.Tensor) -> torch.Tensor:
        return token.float().unsqueeze(-1).expand(token.shape[0], token.shape[1], 4)

    def patched_model(self, **_kwargs: object) -> _FakeOutput:
        return _FakeOutput()

    class speech_pos_emb:  # noqa: N801 - mirrors chatterbox's own attribute shape
        @staticmethod
        def get_fixed_embedding(_step: int) -> torch.Tensor:
            return torch.zeros(1, 1, 4)


class _FakeTokenizer:
    def text_to_tokens(self, _text: str, language_id: str) -> torch.Tensor:
        return torch.zeros(1, 3, dtype=torch.long)


class _FakeConds:
    t3 = object()


class _FakeModel:
    t3 = _FakeT3()
    tokenizer = _FakeTokenizer()
    conds = _FakeConds()
    device = "cpu"


def test_speech_token_chunks_stream_in_order_with_the_last_chunk_shorter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The reimplemented decode loop chunks tokens in generation order.

    A scripted token script (not the real model's logits) drives the loop so
    the test proves the chunking, not the language model.
    """
    monkeypatch.setattr("speech.chatterbox.STREAM_TOKEN_CHUNK", 3)
    script = iter([10, 11, 12, 13, 14, 15, 16, _FakeT3.hp.stop_speech_token])

    def fake_next_speech_token(*_args: object, **_kwargs: object) -> torch.Tensor:
        return torch.tensor([[next(script)]])

    monkeypatch.setattr(
        "speech.chatterbox._next_speech_token",
        fake_next_speech_token,
    )

    chunks = list(_speech_token_chunks(_FakeModel(), "Hallo", "de"))

    token_lists = [chunk.tolist() for chunk in chunks]
    assert token_lists == [[10, 11, 12], [13, 14, 15], [16]]
    assert torch.cat(chunks).tolist() == [10, 11, 12, 13, 14, 15, 16]
    assert len(chunks[-1]) < 3, "the trailing remainder ships as its own short chunk"

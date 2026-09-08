"""The resident Chatterbox Multilingual V3 voice, streaming PCM as it synthesises."""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Iterator

_LOG = logging.getLogger(__name__)

CHATTERBOX_SAMPLE_RATE = 24_000
CHATTERBOX_T3_WEIGHTS = "t3_mtl23ls_v3.safetensors"
# Pinned to the snapshot this service was measured against; a moving "main"
# could swap in weights nobody here has timed or listened to.
CHATTERBOX_REVISION = "5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18"
# 25 speech tokens per second of 24 kHz audio; 12 tokens is about half a second.
STREAM_TOKEN_CHUNK = 12
MAX_NEW_TOKENS = 1000
WARMUP_TEXT = "Hallo."


def _checkpoint_dir() -> object:
    """Local Hub snapshot. 0.1.7 from_pretrained takes only device and loads V2."""
    import os
    from pathlib import Path

    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            repo_id="ResembleAI/chatterbox",
            repo_type="model",
            revision=CHATTERBOX_REVISION,
            allow_patterns=[
                "ve.pt",
                CHATTERBOX_T3_WEIGHTS,
                "s3gen.pt",
                "grapheme_mtl_merged_expanded_v1.json",
                "conds.pt",
                "Cangjie5_TC.json",
            ],
            local_files_only=os.environ.get("HF_HUB_OFFLINE") == "1",
            token=os.getenv("HF_TOKEN"),
        )
    )


def _load_v3(device: str) -> object:
    """Load Multilingual V3 weights the installed package has no t3_model flag for."""
    import torch
    from chatterbox.models.s3gen import S3Gen
    from chatterbox.models.t3 import T3
    from chatterbox.models.t3.modules.t3_config import T3Config
    from chatterbox.models.tokenizers import MTLTokenizer
    from chatterbox.models.voice_encoder import VoiceEncoder
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS, Conditionals
    from safetensors.torch import load_file as load_safetensors

    ckpt_dir = _checkpoint_dir()
    map_location = torch.device("cpu") if device in {"cpu", "mps"} else None
    voice_encoder = VoiceEncoder()
    voice_encoder.load_state_dict(
        torch.load(ckpt_dir / "ve.pt", map_location=map_location, weights_only=True)
    )
    voice_encoder.to(device).eval()
    t3 = T3(T3Config.multilingual())
    t3_state = load_safetensors(ckpt_dir / CHATTERBOX_T3_WEIGHTS)
    if "model" in t3_state:
        t3_state = t3_state["model"][0]
    t3.load_state_dict(t3_state)
    t3.to(device).eval()
    s3gen = S3Gen()
    s3gen.load_state_dict(
        torch.load(ckpt_dir / "s3gen.pt", map_location=map_location, weights_only=True)
    )
    s3gen.to(device).eval()
    tokenizer = MTLTokenizer(str(ckpt_dir / "grapheme_mtl_merged_expanded_v1.json"))
    conds = None
    builtin_voice = ckpt_dir / "conds.pt"
    if builtin_voice.exists():
        conds = Conditionals.load(builtin_voice, map_location=map_location).to(device)
    return ChatterboxMultilingualTTS(
        t3,
        s3gen,
        voice_encoder,
        tokenizer,
        device,
        conds=conds,
    )


def language_id(language: str) -> str:
    """ISO 639-1 code Chatterbox expects (`de`, `en`)."""
    token = language.strip().lower().replace("_", "-")
    if not token:
        return "de"
    return token.split("-", 1)[0]


def _float_to_pcm(wav: np.ndarray) -> bytes:
    clipped = np.clip(np.asarray(wav, dtype=np.float64) * 32767.0, -32768, 32767)
    return clipped.astype(np.int16).tobytes()


class ChatterboxSpeaking:
    """One Chatterbox Multilingual V3 model, loaded once and held on the card."""

    streams = True
    sample_rate = CHATTERBOX_SAMPLE_RATE

    def __init__(self, model_name: str, device: str) -> None:
        """Remember the Hub id and the device the weights will occupy."""
        self.model_name = model_name
        self.ready = False
        self._device = device
        self._model = None
        self._lock = threading.Lock()

    def load(self) -> None:
        """Load V3 weights onto the configured device from the local Hub cache."""
        from speech.cuda_libs import prepare_cuda_libraries

        prepare_cuda_libraries()
        _LOG.info("loading speaking model %s on %s", self.model_name, self._device)
        model = _load_v3(self._device)
        self._model = model
        self.sample_rate = int(model.sr)
        for _chunk in _stream_pcm(model, WARMUP_TEXT, "de"):
            pass
        self.ready = True
        _LOG.info("speaking model %s ready", self.model_name)

    def pcm_chunks(self, text: str, language: str) -> Iterator[bytes]:
        """Yield 16-bit mono PCM while Chatterbox is still synthesising."""
        if self._model is None:
            message = "speaking model is not loaded"
            raise RuntimeError(message)
        with self._lock:
            yield from _stream_pcm(self._model, text, language_id(language))


def _stream_pcm(model: object, text: str, language: str) -> Iterator[bytes]:
    import torch
    from chatterbox.models.s3tokenizer import S3_TOKEN_RATE, drop_invalid_tokens

    samples_per_token = int(model.sr) // int(S3_TOKEN_RATE)
    emitted = 0
    accumulated: torch.Tensor | None = None
    for token_chunk in _speech_token_chunks(model, text, language):
        if accumulated is None:
            accumulated = token_chunk
        else:
            accumulated = torch.cat([accumulated, token_chunk])
        clean = drop_invalid_tokens(accumulated).to(model.device)
        if clean.numel() == 0:
            continue
        # Keep the last token's audio back until EOS so the cropped tail never ships.
        wav = _vocode(model, clean)
        ready_until = max(0, len(wav) - samples_per_token)
        new = wav[emitted:ready_until]
        emitted = ready_until
        if new.size:
            yield _float_to_pcm(new)
    if accumulated is None:
        return
    clean = drop_invalid_tokens(accumulated).to(model.device)
    if clean.numel() == 0:
        return
    n_tokens = int(clean.shape[-1])
    st_len = max(1, n_tokens - 1)
    wav = _vocode(model, clean)[: st_len * samples_per_token]
    new = wav[emitted:]
    if new.size:
        yield _float_to_pcm(new)


def _vocode(model: object, speech_tokens: object) -> np.ndarray:
    # Calls s3gen directly rather than the model's own generate(), so the
    # watermark that generate() applies afterward never runs on this audio.
    import torch

    with torch.inference_mode():
        wav, _hidden = model.s3gen.inference(
            speech_tokens=speech_tokens,
            ref_dict=model.conds.gen,
        )
    return wav.squeeze(0).detach().cpu().numpy()


def _ensure_patched_t3(t3: object) -> None:
    if getattr(t3, "compiled", False) and hasattr(t3, "patched_model"):
        return
    from chatterbox.models.t3.inference.t3_hf_backend import T3HuggingfaceBackend

    t3.patched_model = T3HuggingfaceBackend(
        config=t3.cfg,
        llama=t3.tfmr,
        speech_enc=t3.speech_emb,
        speech_head=t3.speech_head,
    )
    t3.compiled = True


def _text_tokens(model: object, text: str, language: str) -> object:
    import torch
    from chatterbox.mtl_tts import punc_norm
    from torch.nn import functional

    t3 = model.t3
    tokens = model.tokenizer.text_to_tokens(
        punc_norm(text),
        language_id=language,
    ).to(model.device)
    tokens = torch.cat([tokens, tokens], dim=0)
    tokens = functional.pad(tokens, (1, 0), value=t3.hp.start_text_token)
    return functional.pad(tokens, (0, 1), value=t3.hp.stop_text_token)


def _next_speech_token(
    output: object,
    generated_ids: object,
    processors: tuple[object, object, object],
) -> object:
    import torch

    repetition_penalty, min_p_warper, top_p_warper = processors
    logits_step = output.logits[:, -1, :]
    cond = logits_step[0:1, :]
    uncond = logits_step[1:2, :]
    logits = cond + 0.5 * (cond - uncond)
    ids_for_proc = generated_ids[:1, ...]
    logits = repetition_penalty(ids_for_proc, logits)
    logits = logits / 0.8
    logits = min_p_warper(ids_for_proc, logits)
    logits = top_p_warper(ids_for_proc, logits)
    return torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1)


def _speech_token_chunks(
    model: object,
    text: str,
    language: str,
) -> Iterator[object]:
    # The pinned wheel's T3.inference() and generate() run to completion before
    # returning a token, and generate() has no way to select the v3 checkpoint
    # this loader wires up; this reimplements the decode loop token by token so
    # a caller can stream speech tokens as they are produced.
    import torch
    from transformers.generation.logits_process import (
        MinPLogitsWarper,
        RepetitionPenaltyLogitsProcessor,
        TopPLogitsWarper,
    )

    t3 = model.t3
    text_tokens = _text_tokens(model, text, language)
    initial = t3.hp.start_speech_token * torch.ones_like(text_tokens[:, :1])
    embeds, _len_cond = t3.prepare_input_embeds(
        t3_cond=model.conds.t3,
        text_tokens=text_tokens,
        speech_tokens=initial,
        cfg_weight=0.5,
    )
    _ensure_patched_t3(t3)
    bos_token = torch.tensor(
        [[t3.hp.start_speech_token]],
        dtype=torch.long,
        device=embeds.device,
    )
    bos_embed = t3.speech_emb(bos_token) + t3.speech_pos_emb.get_fixed_embedding(0)
    bos_embed = torch.cat([bos_embed, bos_embed])
    generated_ids = bos_token.clone()
    processors = (
        RepetitionPenaltyLogitsProcessor(penalty=1.2),
        MinPLogitsWarper(min_p=0.05),
        TopPLogitsWarper(top_p=1.0),
    )
    with torch.inference_mode():
        output = t3.patched_model(
            inputs_embeds=torch.cat([embeds, bos_embed], dim=1),
            past_key_values=None,
            use_cache=True,
            output_attentions=False,
            output_hidden_states=True,
            return_dict=True,
        )
        past = output.past_key_values
        buffer: list[object] = []
        for step in range(MAX_NEW_TOKENS):
            next_token = _next_speech_token(output, generated_ids, processors)
            generated_ids = torch.cat([generated_ids, next_token], dim=1)
            if next_token.view(-1) == t3.hp.stop_speech_token:
                if buffer:
                    yield torch.cat(buffer, dim=1).squeeze(0)
                    buffer = []
                break
            buffer.append(next_token)
            if len(buffer) >= STREAM_TOKEN_CHUNK:
                yield torch.cat(buffer, dim=1).squeeze(0)
                buffer = []
            next_embed = t3.speech_emb(next_token)
            next_embed = next_embed + t3.speech_pos_emb.get_fixed_embedding(step + 1)
            output = t3.patched_model(
                inputs_embeds=torch.cat([next_embed, next_embed]),
                past_key_values=past,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            past = output.past_key_values
        if buffer:
            yield torch.cat(buffer, dim=1).squeeze(0)

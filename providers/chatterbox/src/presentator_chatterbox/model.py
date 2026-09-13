"""The Chatterbox V3 model adapter, local-cache only."""

from collections.abc import Iterator
from pathlib import Path

import numpy as np
from presentator_chatterbox_contract import CHATTERBOX_ARTIFACTS, CHATTERBOX_REVISION

STREAM_TOKEN_CHUNK = 12
MAX_NEW_TOKENS = 1000


def load_model(device: str, cache: Path) -> object:
    """Load the pinned multilingual model from the local Hub cache."""
    import torch
    from chatterbox.models.s3gen import S3Gen
    from chatterbox.models.t3 import T3
    from chatterbox.models.t3.modules.t3_config import T3Config
    from chatterbox.models.tokenizers import MTLTokenizer
    from chatterbox.models.voice_encoder import VoiceEncoder
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS, Conditionals
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file as load_safetensors

    checkpoint = Path(
        snapshot_download(
            repo_id="ResembleAI/chatterbox",
            repo_type="model",
            revision=CHATTERBOX_REVISION,
            allow_patterns=CHATTERBOX_ARTIFACTS,
            local_files_only=True,
            cache_dir=cache,
        )
    )
    map_location = torch.device("cpu") if device in {"cpu", "mps"} else None
    voice_encoder = VoiceEncoder()
    voice_encoder.load_state_dict(
        torch.load(checkpoint / "ve.pt", map_location=map_location, weights_only=True)
    )
    voice_encoder.to(device).eval()
    t3 = T3(T3Config.multilingual())
    state = load_safetensors(checkpoint / "t3_mtl23ls_v3.safetensors")
    if "model" in state:
        state = state["model"][0]
    t3.load_state_dict(state)
    t3.to(device).eval()
    s3gen = S3Gen()
    s3gen.load_state_dict(
        torch.load(
            checkpoint / "s3gen.pt", map_location=map_location, weights_only=True
        )
    )
    s3gen.to(device).eval()
    tokenizer = MTLTokenizer(str(checkpoint / "grapheme_mtl_merged_expanded_v1.json"))
    conds = Conditionals.load(checkpoint / "conds.pt", map_location=map_location)
    conds = conds.to(device)
    return ChatterboxMultilingualTTS(
        t3, s3gen, voice_encoder, tokenizer, device, conds=conds
    )


def pcm_chunks(model: object, text: str, language: str) -> Iterator[bytes]:
    """Yield model output in valid signed 16-bit little-endian PCM chunks."""
    import torch
    from chatterbox.models.s3tokenizer import S3_TOKEN_RATE, drop_invalid_tokens

    samples_per_token = int(model.sr) // int(S3_TOKEN_RATE)
    emitted = 0
    accumulated: torch.Tensor | None = None
    for token_chunk in speech_token_chunks(model, text, language):
        accumulated = (
            token_chunk
            if accumulated is None
            else torch.cat([accumulated, token_chunk])
        )
        clean = drop_invalid_tokens(accumulated).to(model.device)
        if clean.numel() == 0:
            continue
        wav = _vocode(model, clean)
        ready_until = max(0, len(wav) - samples_per_token)
        new = wav[emitted:ready_until]
        emitted = ready_until
        if new.size:
            yield float_to_pcm(new)
    if accumulated is None:
        return
    clean = drop_invalid_tokens(accumulated).to(model.device)
    if clean.numel() == 0:
        return
    wav = _vocode(model, clean)
    ready = wav[: max(1, int(clean.shape[-1]) - 1) * samples_per_token]
    if ready[emitted:].size:
        yield float_to_pcm(ready[emitted:])


def float_to_pcm(wav: np.ndarray) -> bytes:
    """Clip float audio into signed little-endian 16-bit PCM."""
    clipped = np.clip(np.asarray(wav, dtype=np.float64) * 32767.0, -32768, 32767)
    return clipped.astype("<i2").tobytes()


def _vocode(model: object, speech_tokens: object) -> np.ndarray:
    import torch

    with torch.inference_mode():
        wav, _hidden = model.s3gen.inference(
            speech_tokens=speech_tokens, ref_dict=model.conds.gen
        )
    return wav.squeeze(0).detach().cpu().numpy()


def speech_token_chunks(model: object, text: str, language: str) -> Iterator[object]:
    """Yield bounded token chunks from the provider's incremental decoder."""
    import torch
    from chatterbox.models.t3.inference.t3_hf_backend import T3HuggingfaceBackend
    from chatterbox.mtl_tts import punc_norm
    from torch.nn import functional
    from transformers.generation.logits_process import (
        MinPLogitsWarper,
        RepetitionPenaltyLogitsProcessor,
        TopPLogitsWarper,
    )

    t3 = model.t3
    tokens = model.tokenizer.text_to_tokens(punc_norm(text), language_id=language).to(
        model.device
    )
    tokens = torch.cat([tokens, tokens], dim=0)
    tokens = functional.pad(tokens, (1, 0), value=t3.hp.start_text_token)
    tokens = functional.pad(tokens, (0, 1), value=t3.hp.stop_text_token)
    initial = t3.hp.start_speech_token * torch.ones_like(tokens[:, :1])
    embeds, _length = t3.prepare_input_embeds(
        t3_cond=model.conds.t3,
        text_tokens=tokens,
        speech_tokens=initial,
        cfg_weight=0.5,
    )
    if not (getattr(t3, "compiled", False) and hasattr(t3, "patched_model")):
        t3.patched_model = T3HuggingfaceBackend(
            config=t3.cfg,
            llama=t3.tfmr,
            speech_enc=t3.speech_emb,
            speech_head=t3.speech_head,
        )
        t3.compiled = True
    bos = torch.tensor(
        [[t3.hp.start_speech_token]], dtype=torch.long, device=embeds.device
    )
    bos_embed = t3.speech_emb(bos) + t3.speech_pos_emb.get_fixed_embedding(0)
    bos_embed = torch.cat([bos_embed, bos_embed])
    generated = bos.clone()
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
            repetition, minimum, top = processors
            logits = output.logits[:, -1, :]
            logits = logits[:1] + 0.5 * (logits[:1] - logits[1:2])
            logits = repetition(generated[:1], logits) / 0.8
            logits = minimum(generated[:1], logits)
            next_token = torch.multinomial(
                torch.softmax(top(generated[:1], logits), dim=-1), num_samples=1
            )
            generated = torch.cat([generated, next_token], dim=1)
            if next_token.view(-1) == t3.hp.stop_speech_token:
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

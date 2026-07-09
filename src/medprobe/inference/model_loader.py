"""Two-tier model loader. Detects GPU and picks dtype + attention impl accordingly."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

log = logging.getLogger(__name__)


@dataclass
class LoadedModel:
    model: Any
    tokenizer: Any
    device: str
    dtype: torch.dtype
    attn_impl: str
    gpu_name: str
    n_layers: int
    hidden_dim: int


def _detect_gpu_tier() -> tuple[str, torch.dtype, str]:
    if not torch.cuda.is_available():
        return "cpu", torch.float32, "eager"
    name = torch.cuda.get_device_name(0).upper()
    if "H100" in name:
        return name, torch.bfloat16, "flash_attention_2"
    if "A100" in name:
        return name, torch.bfloat16, "flash_attention_2"
    if "A10" in name or "V100" in name or "T4" in name or "L40" in name or "L4" in name:
        return name, torch.float16, "eager"
    # Default: safe fp16 + eager
    return name, torch.float16, "eager"


def load_model(
    hf_id: str,
    hf_token: str | None = None,
    force_dtype: torch.dtype | None = None,
    force_attn_impl: str | None = None,
) -> LoadedModel:
    gpu_name, dtype, attn_impl = _detect_gpu_tier()
    if force_dtype is not None:
        dtype = force_dtype
    if force_attn_impl is not None:
        attn_impl = force_attn_impl

    # Empty string overrides the cached token and triggers 401 on gated repos.
    # Pass None so huggingface_hub falls back to ~/.cache/huggingface/token.
    hf_token = hf_token if hf_token else None

    log.info(
        "Loading %s on %s with dtype=%s attn_impl=%s",
        hf_id, gpu_name, dtype, attn_impl,
    )
    tok = AutoTokenizer.from_pretrained(hf_id, token=hf_token)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # flash-attn might not be installed on A10; fall back gracefully.
    try:
        model = AutoModelForCausalLM.from_pretrained(
            hf_id,
            torch_dtype=dtype,
            device_map="auto",
            token=hf_token,
            attn_implementation=attn_impl,
        )
    except (ImportError, ValueError) as e:
        log.warning("attn_impl=%s failed (%s); retrying with eager", attn_impl, e)
        attn_impl = "eager"
        model = AutoModelForCausalLM.from_pretrained(
            hf_id,
            torch_dtype=dtype,
            device_map="auto",
            token=hf_token,
            attn_implementation="eager",
        )
    model.eval()
    cfg = model.config
    n_layers = getattr(cfg, "num_hidden_layers", getattr(cfg, "n_layer", 0))
    hidden_dim = getattr(cfg, "hidden_size", getattr(cfg, "n_embd", 0))
    device = next(model.parameters()).device.type
    return LoadedModel(
        model=model,
        tokenizer=tok,
        device=device,
        dtype=dtype,
        attn_impl=attn_impl,
        gpu_name=gpu_name,
        n_layers=n_layers,
        hidden_dim=hidden_dim,
    )

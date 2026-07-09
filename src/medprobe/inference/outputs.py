"""Output-level baselines packaged as reusable utilities."""
from __future__ import annotations

from typing import Any

import torch


@torch.inference_mode()
def self_consistency(
    loaded,
    question: str,
    answer: str,
    n_samples: int = 3,
    temperature: float = 0.7,
    max_new_tokens: int = 20,
    template: str | None = None,
) -> dict[str, Any]:
    from .qa_prompt import format_qa_prompt

    tok = loaded.tokenizer
    model = loaded.model
    device = next(model.parameters()).device
    t = template or (
        "Question: {question}\nAnswer: {answer}\n"
        "Is this answer medically correct? Respond with Yes or No."
    )
    prompt = format_qa_prompt(tok, question, answer, template=t)
    enc = tok(prompt, return_tensors="pt").to(device)
    verbals: list[str] = []
    for _ in range(n_samples):
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            pad_token_id=tok.eos_token_id,
        )
        txt = tok.decode(out[0, enc.input_ids.shape[1] :], skip_special_tokens=True).strip()
        first = (txt.split() or [""])[0].lower().strip(".,!?")
        if first.startswith("yes"):
            verbals.append("yes")
        elif first.startswith("no"):
            verbals.append("no")
        else:
            verbals.append("other")
    if not verbals:
        return {"majority": "other", "agreement": 0.0, "samples": verbals}
    majority = max(set(verbals), key=verbals.count)
    agreement = verbals.count(majority) / len(verbals)
    return {"majority": majority, "agreement": agreement, "samples": verbals}

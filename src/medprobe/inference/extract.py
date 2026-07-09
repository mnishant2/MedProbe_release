"""Run generate() with output_hidden_states=True and pull last-question + first-answer tokens."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from .qa_prompt import format_qa_prompt


@dataclass
class ExtractionResult:
    hidden_states: dict[str, np.ndarray]   # key "layer{ℓ}_{position}" → (hidden_dim,)
    generated_text: str
    mean_token_entropy: float
    mean_logprob: float
    verbal_label: str                      # "yes" / "no" / "other" (hard decoded token)
    p_true: float                          # soft P(True): P(Yes)/(P(Yes)+P(No)) at first gen position
    n_layers: int
    hidden_dim: int


_YESNO_CACHE: dict[int, tuple[list[int], list[int]]] = {}


def _yes_no_token_ids(tok) -> tuple[list[int], list[int]]:
    """First-token ids for common 'Yes'/'No' surface forms, cached per tokenizer.
    Used for the Kadavath-style P(True) baseline."""
    key = id(tok)
    if key in _YESNO_CACHE:
        return _YESNO_CACHE[key]
    yes_words = ["Yes", " Yes", "yes", " yes", "YES", " YES"]
    no_words = ["No", " No", "no", " no", "NO", " NO"]

    def first_ids(words: list[str]) -> list[int]:
        ids: set[int] = set()
        for w in words:
            t = tok.encode(w, add_special_tokens=False)
            if t:
                ids.add(t[0])
        return sorted(ids)

    pair = (first_ids(yes_words), first_ids(no_words))
    _YESNO_CACHE[key] = pair
    return pair


@torch.inference_mode()
def extract_one(
    loaded,
    question: str,
    answer: str,
    max_new_tokens: int = 20,
    template: str | None = None,
) -> ExtractionResult:
    model = loaded.model
    tok = loaded.tokenizer
    device = next(model.parameters()).device
    prompt = format_qa_prompt(
        tok, question, answer, template=template or _default_template()
    )
    enc = tok(prompt, return_tensors="pt").to(device)
    input_len = enc.input_ids.shape[1]

    outputs = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        output_hidden_states=True,
        output_scores=True,
        return_dict_in_generate=True,
        pad_token_id=tok.eos_token_id,
    )

    # outputs.hidden_states is a tuple of steps. Step 0 = prompt processing,
    # each subsequent step = one generated token.
    step_0 = outputs.hidden_states[0]           # tuple of (n_layers+1,) tensors
    n_hs_layers = len(step_0)
    step_1 = outputs.hidden_states[1] if len(outputs.hidden_states) > 1 else None

    hs: dict[str, np.ndarray] = {}
    for layer_idx in range(n_hs_layers):
        h_q = step_0[layer_idx][0, -1, :].detach().float().cpu().numpy()
        hs[f"layer{layer_idx}_last_question_token"] = h_q
        if step_1 is not None:
            h_a = step_1[layer_idx][0, -1, :].detach().float().cpu().numpy()
            hs[f"layer{layer_idx}_first_answer_token"] = h_a

    gen_ids = outputs.sequences[0, input_len:]
    gen_text = tok.decode(gen_ids, skip_special_tokens=True).strip()

    entropies: list[float] = []
    logprobs: list[float] = []
    for t_idx, score in enumerate(outputs.scores):
        logits = score[0]                          # (vocab,)
        logp = torch.log_softmax(logits, dim=-1)
        p = logp.exp()
        ent = -(p * logp).sum().item()
        entropies.append(ent)
        if t_idx < gen_ids.shape[0]:
            logprobs.append(logp[gen_ids[t_idx]].item())
    mean_ent = float(np.mean(entropies)) if entropies else math.nan
    mean_logp = float(np.mean(logprobs)) if logprobs else math.nan

    first_word = gen_text.split()[0].lower().strip(".,!?") if gen_text else ""
    if first_word.startswith("yes"):
        verbal = "yes"
    elif first_word.startswith("no"):
        verbal = "no"
    else:
        verbal = "other"

    # Soft P(True), Kadavath et al. 2022: probability mass on "Yes" tokens vs "No"
    # tokens at the first generated position, renormalised over {Yes, No}.
    p_true = 0.5
    if outputs.scores:
        yes_ids, no_ids = _yes_no_token_ids(tok)
        p0 = torch.softmax(outputs.scores[0][0], dim=-1)
        p_yes = float(p0[yes_ids].sum().item()) if yes_ids else 0.0
        p_no = float(p0[no_ids].sum().item()) if no_ids else 0.0
        denom = p_yes + p_no
        if denom > 1e-8:
            p_true = p_yes / denom

    return ExtractionResult(
        hidden_states=hs,
        generated_text=gen_text,
        mean_token_entropy=mean_ent,
        mean_logprob=mean_logp,
        verbal_label=verbal,
        p_true=p_true,
        n_layers=n_hs_layers,
        hidden_dim=next(iter(hs.values())).shape[0],
    )


def _default_template() -> str:
    return (
        "Question: {question}\n"
        "Answer: {answer}\n"
        "Is this answer medically correct? Respond with Yes or No."
    )

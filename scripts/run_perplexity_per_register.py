#!/usr/bin/env python
"""Compute next-token perplexity per (source, register, label) for every variant"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv

from medprobe.config import load_config, model_by_slug, resolve_path
from medprobe.inference.model_loader import load_model
from medprobe.logging_utils import setup_logger

load_dotenv()


PROMPT_TEMPLATE = "Question: {question}\nAnswer: {answer}"


@torch.inference_mode()
def sequence_perplexity(model, tok, text: str, device) -> tuple[float, int, float]:
    """Return (mean NLL per non-pad token, n_tokens, perplexity)."""
    enc = tok(text, return_tensors='pt', truncation=True, max_length=1024).to(device)
    input_ids = enc.input_ids
    if input_ids.shape[1] < 2:
        return float('nan'), 0, float('nan')
    out = model(input_ids=input_ids, labels=input_ids)
    # HuggingFace CausalLMOutput sums per-shift-token NLL and returns the mean.
    # Multiply by (n - 1) to get total NLL and divide explicitly so we match the
    # convention reported in the paper.
    mean_nll = float(out.loss.item())
    n_eff = int(input_ids.shape[1] - 1)
    ppl = float(math.exp(mean_nll)) if mean_nll < 50 else float('inf')
    return mean_nll, n_eff, ppl


def iter_native(facts_path: Path, source_tag: str):
    facts = json.loads(facts_path.read_text())
    for f in facts:
        for label, ans_key in [(1, 'correct_answer'), (0, 'wrong_answer')]:
            ans = f.get(ans_key)
            if ans is None:
                continue
            yield {
                'source': source_tag,
                'register': 'native',
                'label': label,
                'fact_id': f['id'],
                'variant_key': f"{f['id']}__native__{'correct' if label else 'wrong'}",
                'text': PROMPT_TEMPLATE.format(question=f['question'], answer=ans),
            }


def iter_variants(variants_path: Path, source_tag: str):
    if not variants_path.exists():
        return
    variants = json.loads(variants_path.read_text())
    for key, row in variants.items():
        if 'error' in row:
            continue
        yield {
            'source': source_tag,
            'register': row['register'],
            'label': int(row['label']),
            'fact_id': row['fact_id'],
            'variant_key': key,
            'text': PROMPT_TEMPLATE.format(question=row['question'], answer=row['answer']),
        }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--reference-model', default='gemma-2-2b-it',
                    help='Project model slug (configs/models.yaml) used as reference LM.')
    ap.add_argument('--out', default=None,
                    help='Output CSV path. Defaults to outputs/probes/perplexity_per_register.csv.')
    ap.add_argument('--include-medmcqa-sonnet', action='store_true',
                    help='Also score the 800 MedMCQA register variants if present.')
    ap.add_argument('--limit', type=int, default=None,
                    help='Cap total sequences scored, useful for smoke-testing.')
    ap.add_argument('--override', nargs='*', default=[])
    args = ap.parse_args()

    cfg = load_config(overrides=args.override)
    log = setup_logger('perplexity_per_register', resolve_path(cfg, 'logs_dir'))
    ref_info = model_by_slug(cfg, args.reference_model)
    log.info('Reference LM: %s (hf=%s)', ref_info['slug'], ref_info['id'])

    loaded = load_model(ref_info['id'])
    device = next(loaded.model.parameters()).device

    facts_dir = resolve_path(cfg, 'facts_dir')
    variants_dir = resolve_path(cfg, 'variants_dir')

    sources = []
    sources.append(('medqa-native',  iter_native(facts_dir / 'facts.json', 'medqa-native')))
    sources.append(('medmcqa-native', iter_native(facts_dir / 'medmcqa-facts.json', 'medmcqa-native')))
    sources.append(('medqa-sonnet',  iter_variants(variants_dir / 'sonnet' / 'variants.json', 'medqa-sonnet')))
    if args.include_medmcqa_sonnet:
        sources.append(('medmcqa-sonnet', iter_variants(variants_dir / 'medmcqa-sonnet' / 'variants.json', 'medmcqa-sonnet')))

    rows: list[dict] = []
    n_total = 0
    for tag, it in sources:
        n_in_source = 0
        for record in it:
            mean_nll, n_tok, ppl = sequence_perplexity(loaded.model, loaded.tokenizer, record['text'], device)
            rows.append({
                'source':         record['source'],
                'register':       record['register'],
                'label':          record['label'],
                'fact_id':        record['fact_id'],
                'variant_key':    record['variant_key'],
                'n_input_tokens': n_tok,
                'mean_nll':       mean_nll,
                'perplexity':     ppl,
                'reference_model': ref_info['slug'],
            })
            n_in_source += 1
            n_total += 1
            if n_in_source % 200 == 0:
                log.info('  [%s] scored %d sequences (running total %d)', tag, n_in_source, n_total)
            if args.limit is not None and n_total >= args.limit:
                log.info('Reached --limit %d, stopping early.', args.limit)
                break
        log.info('source=%s done, %d sequences', tag, n_in_source)
        if args.limit is not None and n_total >= args.limit:
            break

    out_path = Path(args.out) if args.out else resolve_path(cfg, 'probes_dir') / 'perplexity_per_register.csv'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    log.info('Wrote %d rows to %s', len(rows), out_path)

    # Headline summary printed to the slurm log so the user does not have to
    # parse the CSV to read the result.
    valid = df[np.isfinite(df['perplexity'])]
    print('\n=== Perplexity by (source, register), reference =', ref_info['slug'], '===')
    print(f"{'source':<18} {'register':<14} {'n':>5}  {'mean_ppl':>9}  {'median_ppl':>10}  {'q1':>7}  {'q3':>7}")
    for (src, reg), sub in valid.groupby(['source', 'register']):
        ppls = sub['perplexity'].to_numpy()
        print(f"{src:<18} {reg:<14} {len(sub):>5}  {np.mean(ppls):>9.2f}  {np.median(ppls):>10.2f}  {np.quantile(ppls, 0.25):>7.2f}  {np.quantile(ppls, 0.75):>7.2f}")

    return 0


if __name__ == '__main__':
    raise SystemExit(main())

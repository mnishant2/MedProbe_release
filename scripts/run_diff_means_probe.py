#!/usr/bin/env python
"""Marks & Tegmark (2024) difference-in-means probe."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import roc_auc_score

from medprobe.config import load_config, model_by_slug, resolve_path
from medprobe.logging_utils import setup_logger
from medprobe.probes.ablations import fact_level_split
from medprobe.probes.dataset import load_probe_matrix, load_variants
from medprobe.probes.evaluate import score
from medprobe.probes.train import train_logistic

load_dotenv()


def diff_means_direction(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    """Return (w, b) such that w · h + b is the diff-of-means classifier score.
    w = μ(y=1) − μ(y=0), bias placed so the decision boundary is the midpoint
    of the two class means projected onto w."""
    mu_pos = X[y == 1].mean(axis=0)
    mu_neg = X[y == 0].mean(axis=0)
    w = mu_pos - mu_neg
    norm = np.linalg.norm(w)
    if norm > 0:
        w = w / norm
    midpoint = 0.5 * (mu_pos + mu_neg)
    b = -float(w @ midpoint)
    return w, b


def diff_means_auroc(w: np.ndarray, b: float, X: np.ndarray, y: np.ndarray) -> float:
    if X.shape[0] == 0 or len(set(y)) < 2:
        return float('nan')
    scores = X @ w + b
    return float(roc_auc_score(y, scores))


def best_textbook_layer(activations_dir, variants, model_slug, layers, position, train_register):
    """Pick the best-textbook layer by training a logistic probe at each layer
    and scoring on its own held-out 20% (no bootstrap, just point AUROC). This
    mirrors how the existing train_probes.py picks the best layer."""
    best_layer, best_auroc = None, -1.0
    for layer in layers:
        X, y, metas = load_probe_matrix(
            activations_dir, variants, model_slug, layer, position, registers=[train_register]
        )
        if X.shape[0] == 0:
            continue
        fact_ids_all = [m['fact_id'] for m in metas]
        train_facts, _ = fact_level_split(fact_ids_all, 0.8, 42)
        tr_mask = np.array([m['fact_id'] in train_facts for m in metas])
        te_mask = ~tr_mask
        if te_mask.sum() == 0 or len(set(y[te_mask])) < 2:
            continue
        probe = train_logistic(X[tr_mask], y[tr_mask])
        s = score(probe, X[te_mask], y[te_mask])
        if s.auroc > best_auroc:
            best_auroc = s.auroc
            best_layer = layer
    return best_layer, best_auroc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--generator', default='sonnet')
    ap.add_argument('--out', default=None)
    ap.add_argument('--bootstrap', type=int, default=1000)
    ap.add_argument('--override', nargs='*', default=[])
    args = ap.parse_args()

    cfg = load_config(overrides=args.override)
    log = setup_logger('diff_means_probe', resolve_path(cfg, 'logs_dir'))
    model_info = model_by_slug(cfg, args.model)

    activations_dir = resolve_path(cfg, 'activations_dir')
    variants = load_variants(resolve_path(cfg, 'variants_dir') / args.generator / 'variants.json')

    layers = list(range(
        0 if bool(cfg.layer_sweep.include_embedding) else 1,
        int(model_info['n_layers']) + 1,
        int(cfg.layer_sweep.stride),
    ))
    positions = list(cfg.layer_sweep.positions)
    registers = list(cfg.registers)
    train_register = str(cfg.train_register)

    rows: list[dict] = []
    for position in positions:
        best_layer, _ = best_textbook_layer(
            activations_dir, variants, model_info['slug'], layers, position, train_register
        )
        if best_layer is None:
            log.warning('No best textbook layer for pos=%s, skipping', position)
            continue
        log.info('model=%s pos=%s best_textbook_layer=%d', model_info['slug'], position, best_layer)

        # Train on textbook 80%, evaluate on textbook held-out 20% AND on every other register.
        X_all, y_all, metas_all = load_probe_matrix(
            activations_dir, variants, model_info['slug'], best_layer, position, registers=[train_register]
        )
        fact_ids_all = [m['fact_id'] for m in metas_all]
        train_facts, _ = fact_level_split(fact_ids_all, 0.8, 42)
        tr_mask = np.array([m['fact_id'] in train_facts for m in metas_all])
        te_mask = ~tr_mask

        probe_log = train_logistic(X_all[tr_mask], y_all[tr_mask])
        w_dm, b_dm = diff_means_direction(X_all[tr_mask], y_all[tr_mask])

        for r in registers:
            if r == train_register:
                X_eval, y_eval = X_all[te_mask], y_all[te_mask]
                eval_metas = [m for m, k in zip(metas_all, te_mask) if k]
            else:
                X_eval, y_eval, eval_metas = load_probe_matrix(
                    activations_dir, variants, model_info['slug'], best_layer, position, registers=[r]
                )
            fids = [m['fact_id'] for m in eval_metas]
            s_log = score(probe_log, X_eval, y_eval, fact_ids=fids, bootstrap=args.bootstrap, seed=42)
            auroc_dm = diff_means_auroc(w_dm, b_dm, X_eval, y_eval)
            rows.append({
                'model_slug': model_info['slug'],
                'generator': args.generator,
                'layer': best_layer,
                'position': position,
                'register': r,
                'n': int(X_eval.shape[0]),
                'auroc_logreg':     s_log.auroc,
                'logreg_ci_lo':     s_log.auroc_ci_lo,
                'logreg_ci_hi':     s_log.auroc_ci_hi,
                'auroc_diff_means': auroc_dm,
                'delta_auroc':      s_log.auroc - auroc_dm if not np.isnan(auroc_dm) else float('nan'),
                'method': 'diff_means_vs_logreg',
            })

    out = Path(args.out) if args.out else resolve_path(cfg, 'probes_dir') / 'diff_means_probe.csv'
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    if out.exists():
        prev = pd.read_csv(out)
        df = pd.concat([prev, df], ignore_index=True)
        df = df.drop_duplicates(subset=['model_slug', 'generator', 'layer', 'position', 'register', 'method'], keep='last')
    df.to_csv(out, index=False)
    log.info('Wrote %d rows to %s', len(rows), out)

    # Headline summary
    if rows:
        new_df = pd.DataFrame(rows)
        for position in positions:
            sub = new_df[new_df['position'] == position]
            if sub.empty:
                continue
            mean_log = sub['auroc_logreg'].mean()
            mean_dm  = sub['auroc_diff_means'].mean()
            log.info('[summary] model=%s pos=%s mean AUROC logreg=%.3f diff_means=%.3f gap=%+.3f',
                     model_info['slug'], position, mean_log, mean_dm, mean_log - mean_dm)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

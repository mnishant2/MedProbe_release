#!/usr/bin/env python
"""Cross-dataset robustness evaluation: probe trained on MedQA-textbook,"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from medprobe.config import load_config, model_by_slug, resolve_path
from medprobe.logging_utils import setup_logger
from medprobe.probes.ablations import fact_level_split
from medprobe.probes.dataset import load_probe_matrix, load_variants
from medprobe.probes.evaluate import score
from medprobe.probes.train import train_logistic

load_dotenv()


def layer_sweep_list(n_layers: int, stride: int, include_embedding: bool) -> list[int]:
    start = 0 if include_embedding else 1
    return list(range(start, n_layers + 1, stride))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="model slug from configs/models.yaml")
    ap.add_argument("--medqa-generator", default="sonnet",
                    help="generator name for the MedQA-Sonnet activations directory")
    ap.add_argument("--medmcqa-generator", default="medmcqa-textbook",
                    help="generator name for the MedMCQA activations directory")
    ap.add_argument("--medmcqa-activations-dir", default=None,
                    help="override path for MedMCQA activation root (default: same activations_dir)")
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--out", default=None)
    ap.add_argument("--override", nargs="*", default=[])
    args = ap.parse_args()

    cfg = load_config(overrides=args.override)
    log = setup_logger("cross_dataset_eval", resolve_path(cfg, "logs_dir"))
    model_info = model_by_slug(cfg, args.model)

    # Source 1: MedQA-Sonnet textbook (training + held-out test)
    medqa_act_dir = resolve_path(cfg, "activations_dir")
    medqa_variants_path = resolve_path(cfg, "variants_dir") / args.medqa_generator / "variants.json"
    medqa_variants = load_variants(medqa_variants_path)

    # Source 2: MedMCQA-textbook (eval only)
    medmcqa_act_dir = (
        Path(args.medmcqa_activations_dir) if args.medmcqa_activations_dir
        else medqa_act_dir
    )
    medmcqa_variants_path = resolve_path(cfg, "variants_dir") / args.medmcqa_generator / "variants.json"
    if not medmcqa_variants_path.exists():
        raise FileNotFoundError(
            f"MedMCQA variants not found at {medmcqa_variants_path}. "
            "Run scripts/build_medmcqa_variants.py first."
        )
    medmcqa_variants = load_variants(medmcqa_variants_path)

    layers = layer_sweep_list(
        int(model_info["n_layers"]),
        int(cfg.layer_sweep.stride),
        bool(cfg.layer_sweep.include_embedding),
    )
    positions = list(cfg.layer_sweep.positions)
    train_register = str(cfg.train_register)
    train_fraction = float(cfg.ablations.mixed_register.train_fraction)
    split_seed = int(cfg.ablations.mixed_register.seed)

    rows: list[dict] = []
    for position in positions:
        for layer in layers:
            X_all, y_all, metas_all = load_probe_matrix(
                medqa_act_dir, medqa_variants, model_info["slug"],
                layer, position, registers=[train_register],
            )
            if X_all.shape[0] == 0:
                log.warning("no MedQA activations for L=%d pos=%s, skipping", layer, position)
                continue
            fact_ids_all = [m["fact_id"] for m in metas_all]
            train_facts, _ = fact_level_split(fact_ids_all, train_fraction, split_seed)
            train_mask = np.array([m["fact_id"] in train_facts for m in metas_all])
            test_mask = ~train_mask

            probe = train_logistic(
                X_all[train_mask], y_all[train_mask],
                C=float(cfg.probe.C),
                max_iter=int(cfg.probe.max_iter),
                solver=str(cfg.probe.solver),
            )

            test_metas = [m for m, keep in zip(metas_all, test_mask) if keep]
            s_in = score(
                probe, X_all[test_mask], y_all[test_mask],
                fact_ids=[m["fact_id"] for m in test_metas],
                bootstrap=args.bootstrap, seed=split_seed,
            )
            rows.append({
                "model_slug": model_info["slug"],
                "layer": layer, "position": position,
                "source": "medqa-textbook-test",
                "n": s_in.n,
                "auroc": s_in.auroc,
                "auroc_ci_lo": s_in.auroc_ci_lo,
                "auroc_ci_hi": s_in.auroc_ci_hi,
                "ece": s_in.ece,
            })

            X_med, y_med, metas_med = load_probe_matrix(
                medmcqa_act_dir, medmcqa_variants, model_info["slug"],
                layer, position, registers=["textbook"],
            )
            if X_med.shape[0] == 0:
                log.warning("no MedMCQA activations for L=%d pos=%s", layer, position)
                continue
            s_med = score(
                probe, X_med, y_med,
                fact_ids=[m["fact_id"] for m in metas_med],
                bootstrap=args.bootstrap, seed=split_seed,
            )
            rows.append({
                "model_slug": model_info["slug"],
                "layer": layer, "position": position,
                "source": "medmcqa",
                "n": s_med.n,
                "auroc": s_med.auroc,
                "auroc_ci_lo": s_med.auroc_ci_lo,
                "auroc_ci_hi": s_med.auroc_ci_hi,
                "ece": s_med.ece,
            })
            log.info(
                "L=%d pos=%s  medqa-test=%.3f [%.3f,%.3f]  medmcqa=%.3f [%.3f,%.3f]",
                layer, position,
                s_in.auroc, s_in.auroc_ci_lo, s_in.auroc_ci_hi,
                s_med.auroc, s_med.auroc_ci_lo, s_med.auroc_ci_hi,
            )

    out = Path(args.out) if args.out else resolve_path(cfg, "probes_dir") / "cross_dataset_eval.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    if out.exists():
        prev = pd.read_csv(out)
        df = pd.concat([prev, df], ignore_index=True)
        dedupe = ["model_slug", "layer", "position", "source"]
        df = df.drop_duplicates(subset=dedupe, keep="last")
    df.to_csv(out, index=False)
    log.info("wrote %d rows to %s", len(rows), out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

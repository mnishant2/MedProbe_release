#!/usr/bin/env python
"""Train probes on textbook register + evaluate on all registers × layers × positions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from medprobe.config import load_config, model_by_slug, resolve_path
from medprobe.logging_utils import setup_logger
from medprobe.probes.ablations import fact_level_split
from medprobe.probes.baselines import (
    entropy_auroc,
    load_inference_meta,
    load_self_consistency,
    ptrue_auroc,
    self_consistency_auroc,
    verbal_auroc,
)
from medprobe.probes.dataset import load_probe_matrix, load_variants
from medprobe.probes.evaluate import score
from medprobe.probes.train import train_logistic, train_mlp

load_dotenv()


def layer_sweep_list(n_layers: int, stride: int, include_embedding: bool) -> list[int]:
    start = 0 if include_embedding else 1
    # n_hs_layers = n_layers + 1 (embedding) but we also include the final layer.
    # Here "layer" is an index into outputs.hidden_states[0] (0..n_layers inclusive).
    return list(range(start, n_layers + 1, stride))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--generator", default="sonnet")
    ap.add_argument("--out", default=None)
    ap.add_argument(
        "--bootstrap",
        type=int,
        default=0,
        help="If > 0, run this many fact-level bootstrap resamples per row and emit AUROC 95 pct CI columns. 1000 is standard.",
    )
    ap.add_argument("--with-mlp", action="store_true", help="Additionally train a small MLP probe (method=probe_mlp).")
    ap.add_argument("--with-permutation", action="store_true", help="Additionally train a probe on label-shuffled textbook (method=probe_permuted) as a sanity baseline.")
    ap.add_argument("--with-specialty", action="store_true", help="Additionally emit per-specialty AUROC rows (method=probe_specialty).")
    ap.add_argument("--override", nargs="*", default=[])
    args = ap.parse_args()

    cfg = load_config(overrides=args.override)
    log = setup_logger("train_probes", resolve_path(cfg, "logs_dir"))
    model_info = model_by_slug(cfg, args.model)

    activations_dir = resolve_path(cfg, "activations_dir")
    variants_path = resolve_path(cfg, "variants_dir") / args.generator / "variants.json"
    variants = load_variants(variants_path)
    facts = json.loads((resolve_path(cfg, "facts_dir") / "facts.json").read_text())
    facts_by_id = {f["id"]: f for f in facts}

    layers = layer_sweep_list(
        int(model_info["n_layers"]),
        int(cfg.layer_sweep.stride),
        bool(cfg.layer_sweep.include_embedding),
    )
    positions = list(cfg.layer_sweep.positions)
    registers = list(cfg.registers)
    train_register = str(cfg.train_register)

    rows: list[dict] = []
    inference_meta = load_inference_meta(activations_dir, model_info["slug"])
    sc_meta = load_self_consistency(activations_dir, model_info["slug"])

    train_fraction = float(cfg.ablations.mixed_register.train_fraction)
    split_seed = int(cfg.ablations.mixed_register.seed)

    for position in positions:
        for layer in layers:
            # Load ALL training-register variants, then fact-level 80/20 split so
            # the held-out fraction gives us a real in-distribution textbook AUROC.
            X_all, y_all, metas_all = load_probe_matrix(
                activations_dir,
                variants,
                model_info["slug"],
                layer,
                position,
                registers=[train_register],
            )
            if X_all.shape[0] == 0:
                log.warning("No training data for layer=%d pos=%s, skipping", layer, position)
                continue
            fact_ids_all = [m["fact_id"] for m in metas_all]
            train_facts, test_facts = fact_level_split(fact_ids_all, train_fraction, split_seed)
            train_mask = np.array([m["fact_id"] in train_facts for m in metas_all])
            test_mask = ~train_mask
            X_tr, y_tr = X_all[train_mask], y_all[train_mask]
            probe = train_logistic(
                X_tr, y_tr,
                C=float(cfg.probe.C),
                max_iter=int(cfg.probe.max_iter),
                solver=str(cfg.probe.solver),
            )
            mlp_probe = None
            if args.with_mlp:
                mlp_probe = train_mlp(X_tr, y_tr)
            permuted_probe = None
            if args.with_permutation:
                rng_perm = np.random.default_rng(split_seed + 1)
                y_shuf = rng_perm.permutation(y_tr)
                permuted_probe = train_logistic(
                    X_tr, y_shuf,
                    C=float(cfg.probe.C),
                    max_iter=int(cfg.probe.max_iter),
                    solver=str(cfg.probe.solver),
                )
            # Evaluate: textbook on HELD-OUT facts; other registers on ALL facts
            # (the probe has never seen these surface forms regardless of fact id).
            per_reg: dict[str, object] = {}
            per_reg_meta: dict[str, list] = {}
            test_metas = [m for m, keep in zip(metas_all, test_mask) if keep]
            per_reg[train_register] = score(
                probe, X_all[test_mask], y_all[test_mask],
                fact_ids=[m["fact_id"] for m in test_metas],
                bootstrap=args.bootstrap,
                seed=split_seed,
            )
            per_reg_meta[train_register] = test_metas
            for r in registers:
                if r == train_register:
                    continue
                X_r, y_r, metas_r = load_probe_matrix(
                    activations_dir, variants, model_info["slug"], layer, position, registers=[r]
                )
                per_reg[r] = score(
                    probe, X_r, y_r,
                    fact_ids=[m["fact_id"] for m in metas_r],
                    bootstrap=args.bootstrap,
                    seed=split_seed,
                )
                per_reg_meta[r] = metas_r
            for r, s in per_reg.items():
                rows.append({
                    "model_slug": model_info["slug"],
                    "generator": args.generator,
                    "layer": layer,
                    "position": position,
                    "register": r,
                    "auroc": s.auroc,
                    "auroc_ci_lo": s.auroc_ci_lo,
                    "auroc_ci_hi": s.auroc_ci_hi,
                    "bootstrap_n": s.bootstrap_n,
                    "ece": s.ece,
                    "accuracy": s.accuracy,
                    "f1": s.f1,
                    "n": s.n,
                    "method": "probe",
                })
                # MLP probe rows (parallel to linear)
                if mlp_probe is not None:
                    if r == train_register:
                        Xe, ye = X_all[test_mask], y_all[test_mask]
                        fids = [m["fact_id"] for m, keep in zip(metas_all, test_mask) if keep]
                    else:
                        Xe, ye, mm = load_probe_matrix(
                            activations_dir, variants, model_info["slug"], layer, position, registers=[r]
                        )
                        fids = [m["fact_id"] for m in mm]
                    s_mlp = score(mlp_probe, Xe, ye, fact_ids=fids, bootstrap=args.bootstrap, seed=split_seed)
                    rows.append({
                        "model_slug": model_info["slug"],
                        "generator": args.generator,
                        "layer": layer, "position": position, "register": r,
                        "auroc": s_mlp.auroc, "auroc_ci_lo": s_mlp.auroc_ci_lo, "auroc_ci_hi": s_mlp.auroc_ci_hi,
                        "bootstrap_n": s_mlp.bootstrap_n, "ece": s_mlp.ece,
                        "accuracy": s_mlp.accuracy, "f1": s_mlp.f1, "n": s_mlp.n,
                        "method": "probe_mlp",
                    })
                # Permutation-test rows
                if permuted_probe is not None:
                    if r == train_register:
                        Xe, ye = X_all[test_mask], y_all[test_mask]
                        fids = [m["fact_id"] for m, keep in zip(metas_all, test_mask) if keep]
                    else:
                        Xe, ye, mm = load_probe_matrix(
                            activations_dir, variants, model_info["slug"], layer, position, registers=[r]
                        )
                        fids = [m["fact_id"] for m in mm]
                    s_perm = score(permuted_probe, Xe, ye, fact_ids=fids, bootstrap=args.bootstrap, seed=split_seed)
                    rows.append({
                        "model_slug": model_info["slug"],
                        "generator": args.generator,
                        "layer": layer, "position": position, "register": r,
                        "auroc": s_perm.auroc, "auroc_ci_lo": s_perm.auroc_ci_lo, "auroc_ci_hi": s_perm.auroc_ci_hi,
                        "bootstrap_n": s_perm.bootstrap_n, "ece": s_perm.ece,
                        "accuracy": s_perm.accuracy, "f1": s_perm.f1, "n": s_perm.n,
                        "method": "probe_permuted",
                    })
                # Per-register rarity split at this layer/position (bootstrap too)
                metas = per_reg_meta[r]
                if r == train_register:
                    _X, _y = X_all[test_mask], y_all[test_mask]
                else:
                    _X, _y, _ = load_probe_matrix(
                        activations_dir, variants, model_info["slug"], layer, position, registers=[r]
                    )
                for rarity in ("common", "rare"):
                    mask = [
                        facts_by_id.get(m["fact_id"], {}).get("rarity") == rarity for m in metas
                    ]
                    if not any(mask):
                        continue
                    mask_a = np.array(mask, dtype=bool)
                    sub_fact_ids = [m["fact_id"] for m, keep in zip(metas, mask_a) if keep]
                    sub = score(
                        probe, _X[mask_a], _y[mask_a],
                        fact_ids=sub_fact_ids,
                        bootstrap=args.bootstrap,
                        seed=split_seed,
                    )
                    rows.append({
                        "model_slug": model_info["slug"],
                        "generator": args.generator,
                        "layer": layer,
                        "position": position,
                        "register": r,
                        "rarity": rarity,
                        "auroc": sub.auroc,
                        "auroc_ci_lo": sub.auroc_ci_lo,
                        "auroc_ci_hi": sub.auroc_ci_hi,
                        "bootstrap_n": sub.bootstrap_n,
                        "ece": sub.ece,
                        "accuracy": sub.accuracy,
                        "f1": sub.f1,
                        "n": sub.n,
                        "method": "probe_rarity",
                    })
                if args.with_specialty:
                    specs = sorted({
                        facts_by_id.get(m["fact_id"], {}).get("specialty", "unknown") for m in metas
                    })
                    for spec in specs:
                        if spec == "unknown":
                            continue
                        mask = [
                            facts_by_id.get(m["fact_id"], {}).get("specialty") == spec
                            for m in metas
                        ]
                        mask_a = np.array(mask, dtype=bool)
                        if mask_a.sum() < 8:
                            continue  # too small to bootstrap meaningfully
                        sub_fact_ids = [
                            m["fact_id"] for m, keep in zip(metas, mask_a) if keep
                        ]
                        sub = score(
                            probe, _X[mask_a], _y[mask_a],
                            fact_ids=sub_fact_ids,
                            bootstrap=args.bootstrap,
                            seed=split_seed,
                        )
                        rows.append({
                            "model_slug": model_info["slug"],
                            "generator": args.generator,
                            "layer": layer,
                            "position": position,
                            "register": r,
                            "rarity": spec,  # overload the rarity column for specialty name
                            "auroc": sub.auroc,
                            "auroc_ci_lo": sub.auroc_ci_lo,
                            "auroc_ci_hi": sub.auroc_ci_hi,
                            "bootstrap_n": sub.bootstrap_n,
                            "ece": sub.ece,
                            "accuracy": sub.accuracy,
                            "f1": sub.f1,
                            "n": sub.n,
                            "method": "probe_specialty",
                        })

    # Output-level baselines (per register, layer-independent)
    for r in registers:
        rows.append({
            "model_slug": model_info["slug"],
            "generator": args.generator,
            "layer": -1,
            "position": "output",
            "register": r,
            "auroc": entropy_auroc(inference_meta, variants, r),
            "method": "entropy_baseline",
        })
        rows.append({
            "model_slug": model_info["slug"],
            "generator": args.generator,
            "layer": -1,
            "position": "output",
            "register": r,
            "auroc": ptrue_auroc(inference_meta, variants, r),
            "method": "ptrue_baseline",
        })
        rows.append({
            "model_slug": model_info["slug"],
            "generator": args.generator,
            "layer": -1,
            "position": "output",
            "register": r,
            "auroc": verbal_auroc(inference_meta, variants, r),
            "method": "verbal_baseline",
        })
        if sc_meta:
            rows.append({
                "model_slug": model_info["slug"],
                "generator": args.generator,
                "layer": -1,
                "position": "output",
                "register": r,
                "auroc": self_consistency_auroc(sc_meta, variants, r),
                "method": "self_consistency_baseline",
            })

    out = Path(args.out) if args.out else resolve_path(cfg, "probes_dir") / "probe_results.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    if out.exists():
        prev = pd.read_csv(out)
        df = pd.concat([prev, df], ignore_index=True)
        # Dedupe on (model_slug, generator, layer, position, register, method, rarity)
        dedupe_cols = ["model_slug", "generator", "layer", "position", "register", "method"]
        if "rarity" in df.columns:
            dedupe_cols.append("rarity")
        df = df.drop_duplicates(subset=dedupe_cols, keep="last")
    df.to_csv(out, index=False)
    log.info("Wrote %d rows to %s", len(rows), out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""Mixed-register ablation. Writes outputs/probes/mixed_register_ablation.csv."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from medprobe.config import load_config, model_by_slug, resolve_path
from medprobe.logging_utils import setup_logger
from medprobe.probes.ablations import mixed_register_ablation
from medprobe.probes.dataset import load_variants

load_dotenv()


def layer_sweep_list(n_layers: int, stride: int, include_embedding: bool) -> list[int]:
    start = 0 if include_embedding else 1
    return list(range(start, n_layers + 1, stride))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--generator", default="sonnet")
    ap.add_argument("--out", default=None)
    ap.add_argument("--bootstrap", type=int, default=0)
    ap.add_argument("--override", nargs="*", default=[])
    args = ap.parse_args()

    cfg = load_config(overrides=args.override)
    log = setup_logger("ablations", resolve_path(cfg, "logs_dir"))
    model_info = model_by_slug(cfg, args.model)

    activations_dir = resolve_path(cfg, "activations_dir")
    variants = load_variants(resolve_path(cfg, "variants_dir") / args.generator / "variants.json")

    layers = layer_sweep_list(
        int(model_info["n_layers"]),
        int(cfg.layer_sweep.stride),
        bool(cfg.layer_sweep.include_embedding),
    )
    positions = list(cfg.layer_sweep.positions)
    registers = list(cfg.registers)

    rows: list[dict] = []
    for position in positions:
        for layer in layers:
            per_reg = mixed_register_ablation(
                activations_dir=activations_dir,
                variants=variants,
                model_slug=model_info["slug"],
                layer=layer,
                position=position,
                registers=registers,
                train_fraction=float(cfg.ablations.mixed_register.train_fraction),
                seed=int(cfg.ablations.mixed_register.seed),
                bootstrap=args.bootstrap,
            )
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
                    "accuracy": s.accuracy,
                    "f1": s.f1,
                    "n": s.n,
                    "method": "mixed_register",
                })

    out = Path(args.out) if args.out else resolve_path(cfg, "probes_dir") / "mixed_register_ablation.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    if out.exists():
        prev = pd.read_csv(out)
        df = pd.concat([prev, df], ignore_index=True)
        dedupe_cols = ["model_slug", "generator", "layer", "position", "register", "method"]
        df = df.drop_duplicates(subset=dedupe_cols, keep="last")
    df.to_csv(out, index=False)
    log.info("Wrote %d rows to %s (total %d after merge)", len(rows), out, len(df))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

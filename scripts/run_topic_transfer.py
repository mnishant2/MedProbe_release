#!/usr/bin/env python
"""Specialty-disjoint topic-transfer probe evaluation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from medprobe.config import load_config, model_by_slug, resolve_path
from medprobe.logging_utils import setup_logger
from medprobe.probes.dataset import load_probe_matrix, load_variants
from medprobe.probes.evaluate import score
from medprobe.probes.train import train_logistic

load_dotenv()


def layer_sweep_list(n_layers: int, stride: int, include_embedding: bool) -> list[int]:
    start = 0 if include_embedding else 1
    return list(range(start, n_layers + 1, stride))


def make_specialty_splits(
    specialties: list[str],
    n_splits: int,
    seed: int,
) -> list[tuple[set[str], set[str]]]:
    """Random 50/50 specialty splits. Returns a list of (train_specs, test_specs)."""
    rng = np.random.default_rng(seed)
    sorted_specs = sorted(specialties)
    n = len(sorted_specs)
    n_train = n // 2  # 7 of 14, or 7 of 15 leaving 8 for test
    splits: list[tuple[set[str], set[str]]] = []
    for _ in range(n_splits):
        order = list(rng.permutation(sorted_specs))
        splits.append((set(order[:n_train]), set(order[n_train:])))
    return splits


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--generator", default="sonnet")
    ap.add_argument("--out", default=None)
    ap.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="Number of random specialty splits (each is 50/50 of available specialties).",
    )
    ap.add_argument(
        "--bootstrap",
        type=int,
        default=1000,
        help="Fact-level bootstrap iterations per cell. 1000 is the project standard.",
    )
    ap.add_argument(
        "--in-dist-fraction",
        type=float,
        default=0.8,
        help="Fraction of train-specialty facts used for training; remainder is the in-distribution test set.",
    )
    ap.add_argument("--override", nargs="*", default=[])
    args = ap.parse_args()

    cfg = load_config(overrides=args.override)
    log = setup_logger("topic_transfer", resolve_path(cfg, "logs_dir"))
    model_info = model_by_slug(cfg, args.model)

    activations_dir = resolve_path(cfg, "activations_dir")
    variants_path = resolve_path(cfg, "variants_dir") / args.generator / "variants.json"
    variants = load_variants(variants_path)
    facts = json.loads((resolve_path(cfg, "facts_dir") / "facts.json").read_text())
    facts_by_id = {f["id"]: f for f in facts}

    spec_path = resolve_path(cfg, "facts_dir") / "specialty_map.json"
    specialty_map = json.loads(spec_path.read_text())

    train_register = str(cfg.train_register)
    layers = layer_sweep_list(
        int(model_info["n_layers"]),
        int(cfg.layer_sweep.stride),
        bool(cfg.layer_sweep.include_embedding),
    )
    positions = list(cfg.layer_sweep.positions)

    # Inventory of specialties with enough facts to bootstrap.
    spec_to_facts: dict[str, list[str]] = {}
    for fid, spec in specialty_map.items():
        if spec == "unknown":
            continue
        spec_to_facts.setdefault(spec, []).append(fid)
    eligible_specs = [s for s, fs in spec_to_facts.items() if len(fs) >= 8]
    log.info(
        "Eligible specialties (>=8 facts): %d / %d total",
        len(eligible_specs), len(spec_to_facts),
    )
    log.info(
        "Per-specialty fact counts: %s",
        {s: len(spec_to_facts[s]) for s in eligible_specs},
    )

    splits = make_specialty_splits(
        eligible_specs, n_splits=args.n_splits, seed=int(cfg.project.seed)
    )

    rows: list[dict] = []
    base_seed = int(cfg.project.seed)

    for position in positions:
        for layer in layers:
            # Load all training-register activations once per (layer, position),
            # then mask by specialty-membership inside the split loop.
            X_all, y_all, metas_all = load_probe_matrix(
                activations_dir,
                variants,
                model_info["slug"],
                layer,
                position,
                registers=[train_register],
            )
            if X_all.shape[0] == 0:
                log.warning("No training data at layer=%d pos=%s, skipping", layer, position)
                continue
            fact_ids_all = np.array([m["fact_id"] for m in metas_all])
            spec_of_meta = np.array(
                [specialty_map.get(m["fact_id"], "unknown") for m in metas_all]
            )

            for split_id, (train_specs, test_specs) in enumerate(splits):
                split_seed = base_seed + 1000 * split_id
                rng = np.random.default_rng(split_seed)

                in_train_spec_mask = np.array([s in train_specs for s in spec_of_meta])
                in_test_spec_mask = np.array([s in test_specs for s in spec_of_meta])

                if in_train_spec_mask.sum() == 0 or in_test_spec_mask.sum() == 0:
                    log.warning(
                        "Empty split at split_id=%d layer=%d pos=%s",
                        split_id, layer, position,
                    )
                    continue

                # Within train_specs, fact-level 80/20 split for the in-distribution
                # held-out evaluation. Done at fact level so both polarities of a
                # fact go to the same side.
                train_spec_facts = sorted(set(fact_ids_all[in_train_spec_mask].tolist()))
                rng.shuffle(train_spec_facts)
                n_in_train = int(round(args.in_dist_fraction * len(train_spec_facts)))
                in_train_facts = set(train_spec_facts[:n_in_train])
                in_test_facts = set(train_spec_facts[n_in_train:])

                tr_mask = np.array(
                    [fid in in_train_facts for fid in fact_ids_all]
                ) & in_train_spec_mask
                in_test_mask = np.array(
                    [fid in in_test_facts for fid in fact_ids_all]
                ) & in_train_spec_mask
                out_test_mask = in_test_spec_mask  # all test-specialty facts

                if tr_mask.sum() == 0 or in_test_mask.sum() == 0 or out_test_mask.sum() == 0:
                    continue

                probe = train_logistic(
                    X_all[tr_mask], y_all[tr_mask],
                    C=float(cfg.probe.C),
                    max_iter=int(cfg.probe.max_iter),
                    solver=str(cfg.probe.solver),
                )

                in_dist = score(
                    probe, X_all[in_test_mask], y_all[in_test_mask],
                    fact_ids=fact_ids_all[in_test_mask].tolist(),
                    bootstrap=args.bootstrap,
                    seed=split_seed,
                )
                topic = score(
                    probe, X_all[out_test_mask], y_all[out_test_mask],
                    fact_ids=fact_ids_all[out_test_mask].tolist(),
                    bootstrap=args.bootstrap,
                    seed=split_seed,
                )

                delta = (
                    float("nan") if (np.isnan(in_dist.auroc) or np.isnan(topic.auroc))
                    else in_dist.auroc - topic.auroc
                )

                rows.append({
                    "model_slug": model_info["slug"],
                    "generator": args.generator,
                    "split_id": split_id,
                    "seed": split_seed,
                    "layer": layer,
                    "position": position,
                    "n_train_specs": len(train_specs),
                    "n_test_specs": len(test_specs),
                    "n_train_facts": int(tr_mask.sum() // 2),  # divide by 2 polarities
                    "n_test_facts_in": int(in_test_mask.sum() // 2),
                    "n_test_facts_out": int(out_test_mask.sum() // 2),
                    "in_dist_auroc": in_dist.auroc,
                    "in_dist_ci_lo": in_dist.auroc_ci_lo,
                    "in_dist_ci_hi": in_dist.auroc_ci_hi,
                    "topic_auroc": topic.auroc,
                    "topic_ci_lo": topic.auroc_ci_lo,
                    "topic_ci_hi": topic.auroc_ci_hi,
                    "delta_topic": delta,
                    "method": "topic_transfer",
                })

    out = Path(args.out) if args.out else resolve_path(cfg, "probes_dir") / "topic_transfer.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    if out.exists():
        prev = pd.read_csv(out)
        df = pd.concat([prev, df], ignore_index=True)
        dedupe_cols = ["model_slug", "generator", "split_id", "layer", "position", "method"]
        df = df.drop_duplicates(subset=dedupe_cols, keep="last")
    df.to_csv(out, index=False)
    log.info(
        "Wrote %d topic-transfer rows for model=%s to %s",
        len(rows), model_info["slug"], out,
    )

    # Headline summary in the log: pick best-textbook layer per (model, position)
    # using in_dist_auroc as the proxy, then average delta_topic over splits.
    if rows:
        df_new = pd.DataFrame(rows)
        for position in positions:
            sub = df_new[df_new["position"] == position]
            if sub.empty:
                continue
            mean_by_layer = sub.groupby("layer")["in_dist_auroc"].mean()
            best_layer = int(mean_by_layer.idxmax())
            best_rows = sub[sub["layer"] == best_layer]
            mean_in = best_rows["in_dist_auroc"].mean()
            mean_topic = best_rows["topic_auroc"].mean()
            mean_delta = best_rows["delta_topic"].mean()
            log.info(
                "[summary] model=%s pos=%s best_layer=%d in_dist=%.3f topic=%.3f delta_topic=%.3f (mean over %d splits)",
                model_info["slug"], position, best_layer,
                mean_in, mean_topic, mean_delta, len(best_rows),
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

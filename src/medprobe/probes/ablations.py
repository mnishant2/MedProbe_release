"""Mixed-register training and rarity split ablations."""
from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np

from .dataset import load_probe_matrix
from .evaluate import ProbeScores, score
from .train import train_logistic


def fact_level_split(
    fact_ids_seen: list[str],
    train_fraction: float,
    seed: int,
) -> tuple[set[str], set[str]]:
    uniq = sorted(set(fact_ids_seen))
    rng = random.Random(seed)
    rng.shuffle(uniq)
    n_train = int(round(train_fraction * len(uniq)))
    return set(uniq[:n_train]), set(uniq[n_train:])


def mixed_register_ablation(
    activations_dir: Path,
    variants: dict[str, Any],
    model_slug: str,
    layer: int,
    position: str,
    registers: list[str],
    train_fraction: float = 0.8,
    seed: int = 42,
    bootstrap: int = 0,
) -> dict[str, ProbeScores]:
    X, y, metas = load_probe_matrix(
        activations_dir, variants, model_slug, layer, position, registers=registers
    )
    if X.shape[0] == 0:
        return {r: ProbeScores(float("nan"), float("nan"), float("nan"), 0) for r in registers}
    all_facts = [m["fact_id"] for m in metas]
    train_facts, test_facts = fact_level_split(all_facts, train_fraction, seed)
    train_mask = np.array([m["fact_id"] in train_facts for m in metas])
    test_mask = ~train_mask
    probe = train_logistic(X[train_mask], y[train_mask])
    out: dict[str, ProbeScores] = {}
    for r in registers:
        r_mask = np.array([m["register"] == r for m in metas]) & test_mask
        if r_mask.sum() == 0:
            out[r] = ProbeScores(float("nan"), float("nan"), float("nan"), 0)
            continue
        sub_fact_ids = [m["fact_id"] for m, keep in zip(metas, r_mask) if keep]
        out[r] = score(
            probe, X[r_mask], y[r_mask],
            fact_ids=sub_fact_ids,
            bootstrap=bootstrap,
            seed=seed,
        )
    return out


def rarity_split_scores(
    activations_dir: Path,
    variants: dict[str, Any],
    facts_by_id: dict[str, dict[str, Any]],
    model_slug: str,
    layer: int,
    position: str,
    register: str,
    probe,
) -> dict[str, ProbeScores]:
    X, y, metas = load_probe_matrix(
        activations_dir, variants, model_slug, layer, position, registers=[register]
    )
    out: dict[str, ProbeScores] = {}
    for rarity in ("common", "rare"):
        mask = np.array(
            [facts_by_id.get(m["fact_id"], {}).get("rarity") == rarity for m in metas]
        )
        if mask.sum() == 0:
            out[rarity] = ProbeScores(float("nan"), float("nan"), float("nan"), 0)
            continue
        out[rarity] = score(probe, X[mask], y[mask])
    return out

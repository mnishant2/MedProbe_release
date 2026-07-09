"""Output-level baselines: token entropy, verbal confidence, self-consistency."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score


def load_inference_meta(activations_dir: Path, model_slug: str) -> dict[str, dict[str, Any]]:
    path = activations_dir / model_slug / "results.json"
    if not path.exists():
        return {}
    with path.open() as fh:
        return json.load(fh)


def entropy_auroc(
    results: dict[str, dict[str, Any]],
    variants: dict[str, dict[str, Any]],
    register: str,
) -> float:
    entropies: list[float] = []
    labels: list[int] = []
    for key, row in variants.items():
        if row.get("register") != register:
            continue
        meta = results.get(key)
        if meta is None or "mean_token_entropy" not in meta:
            continue
        entropies.append(-meta["mean_token_entropy"])  # lower entropy → more confident → higher score
        labels.append(int(row["label"]))
    if len(set(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, entropies))


def verbal_auroc(
    results: dict[str, dict[str, Any]],
    variants: dict[str, dict[str, Any]],
    register: str,
) -> float:
    scores: list[float] = []
    labels: list[int] = []
    for key, row in variants.items():
        if row.get("register") != register:
            continue
        meta = results.get(key)
        if meta is None or "verbal_label" not in meta:
            continue
        label_map = {"yes": 1.0, "other": 0.5, "no": 0.0}
        scores.append(label_map.get(meta["verbal_label"], 0.5))
        labels.append(int(row["label"]))
    if len(set(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def ptrue_auroc(
    results: dict[str, dict[str, Any]],
    variants: dict[str, dict[str, Any]],
    register: str,
) -> float:
    """Kadavath et al. (2022) P(True) baseline: soft probability the model places
    on 'Yes' (vs 'No') at the first generated position, used directly as the
    confidence score. Falls back to the hard verbal_label mapping for any variant
    extracted before p_true was recorded."""
    scores: list[float] = []
    labels: list[int] = []
    label_map = {"yes": 1.0, "other": 0.5, "no": 0.0}
    for key, row in variants.items():
        if row.get("register") != register:
            continue
        meta = results.get(key)
        if meta is None:
            continue
        if "p_true" in meta and meta["p_true"] is not None:
            scores.append(float(meta["p_true"]))
        elif "verbal_label" in meta:
            scores.append(label_map.get(meta["verbal_label"], 0.5))
        else:
            continue
        labels.append(int(row["label"]))
    if len(set(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def load_self_consistency(activations_dir: Path, model_slug: str) -> dict[str, dict[str, Any]]:
    p = activations_dir / model_slug / "self_consistency.json"
    if not p.exists():
        return {}
    with p.open() as fh:
        return json.load(fh)


def self_consistency_auroc(
    sc_results: dict[str, dict[str, Any]],
    variants: dict[str, dict[str, Any]],
    register: str,
) -> float:
    """Self-consistency baseline: score each variant by the fraction of samples
    whose majority vote was 'yes'. Higher → more "believed correct"."""
    scores: list[float] = []
    labels: list[int] = []
    for key, row in variants.items():
        if row.get("register") != register:
            continue
        sc = sc_results.get(key)
        if sc is None or "samples" not in sc:
            continue
        samples = sc["samples"]
        if not samples:
            continue
        yes_rate = sum(1 for s in samples if s == "yes") / len(samples)
        scores.append(yes_rate)
        labels.append(int(row["label"]))
    if len(set(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))

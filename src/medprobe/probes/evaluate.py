"""Probe evaluation: AUROC, accuracy, F1 per register, and Δ_r."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from .dataset import load_probe_matrix
from .train import FittedProbe


@dataclass
class ProbeScores:
    auroc: float
    accuracy: float
    f1: float
    n: int
    # Optional bootstrap AUROC 95% CI (2.5% / 97.5% quantiles). NaN when not
    # computed (bootstrap=0) or undefined (single-class sample).
    auroc_ci_lo: float = float("nan")
    auroc_ci_hi: float = float("nan")
    bootstrap_n: int = 0
    # Expected Calibration Error (10-bin equal-width). NaN if not computed.
    ece: float = float("nan")


def _compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error with equal-width probability bins."""
    if len(y_true) == 0:
        return float("nan")
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (y_prob >= lo) & (y_prob < hi if i < n_bins - 1 else y_prob <= hi)
        if mask.sum() == 0:
            continue
        bin_conf = float(y_prob[mask].mean())
        bin_acc = float(y_true[mask].mean())
        ece += (mask.sum() / n) * abs(bin_conf - bin_acc)
    return float(ece)


def score(
    probe: FittedProbe,
    X: np.ndarray,
    y: np.ndarray,
    fact_ids: list[str] | None = None,
    bootstrap: int = 0,
    seed: int = 0,
) -> ProbeScores:
    """Score a probe. If `bootstrap > 0`, resample facts with replacement
    `bootstrap` times and report 2.5/97.5 AUROC quantiles. Resampling is at the
    FACT level (not the variant level) so both the correct/wrong pair of a fact
    enter or leave the sample together — preserves label balance.
    """
    if X.shape[0] == 0:
        return ProbeScores(float("nan"), float("nan"), float("nan"), 0)
    proba = probe.predict_proba(X)
    pred = (proba >= 0.5).astype(int)
    try:
        auroc = float(roc_auc_score(y, proba)) if len(set(y)) > 1 else float("nan")
    except ValueError:
        auroc = float("nan")
    out = ProbeScores(
        auroc=auroc,
        accuracy=float(accuracy_score(y, pred)),
        f1=float(f1_score(y, pred, zero_division=0)),
        n=int(len(y)),
        ece=_compute_ece(y, proba),
    )
    if bootstrap > 0 and fact_ids is not None and not np.isnan(auroc):
        rng = np.random.default_rng(seed)
        fa = np.asarray(fact_ids)
        uniq = np.unique(fa)
        # Pre-index rows by fact_id for fast bootstrap resample
        by_fact: dict[str, np.ndarray] = {
            fid: np.where(fa == fid)[0] for fid in uniq
        }
        boot_aurocs: list[float] = []
        for _ in range(bootstrap):
            sampled = rng.choice(uniq, size=len(uniq), replace=True)
            idx = np.concatenate([by_fact[fid] for fid in sampled])
            y_b, p_b = y[idx], proba[idx]
            if len(set(y_b)) < 2:
                continue
            try:
                boot_aurocs.append(float(roc_auc_score(y_b, p_b)))
            except ValueError:
                continue
        if boot_aurocs:
            arr = np.asarray(boot_aurocs)
            out.auroc_ci_lo = float(np.quantile(arr, 0.025))
            out.auroc_ci_hi = float(np.quantile(arr, 0.975))
            out.bootstrap_n = len(boot_aurocs)
    return out


def sweep_registers(
    activations_dir,
    variants,
    model_slug: str,
    layer: int,
    position: str,
    probe: FittedProbe,
    registers: list[str],
) -> dict[str, ProbeScores]:
    out: dict[str, ProbeScores] = {}
    for r in registers:
        X, y, _ = load_probe_matrix(
            activations_dir, variants, model_slug, layer, position, registers=[r]
        )
        out[r] = score(probe, X, y)
    return out


def delta_from_baseline(
    scores: dict[str, ProbeScores], baseline: str = "textbook"
) -> dict[str, float]:
    base = scores.get(baseline)
    if base is None or np.isnan(base.auroc):
        return {r: float("nan") for r in scores}
    return {r: base.auroc - s.auroc for r, s in scores.items()}

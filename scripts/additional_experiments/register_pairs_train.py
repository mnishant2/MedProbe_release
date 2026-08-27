"""Train probes on pairs of registers, on single registers and on all four, and evaluate every register on the shared held-out split."""
import argparse, csv, sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
OUT = REPO / "outputs/camera_ready/register_matrix"

import numpy as np
from dotenv import load_dotenv
from sklearn.metrics import roc_auc_score
from medprobe.config import load_config, model_by_slug, resolve_path
from medprobe.probes.ablations import fact_level_split
from medprobe.probes.dataset import load_probe_matrix, load_variants
from medprobe.probes.train import train_logistic

load_dotenv()
REGS = ["textbook", "patient", "clinical_note", "colloquial"]
N_BOOT = 1000
SEED = 42

def sweep(n, s, inc):
    return list(range(0 if inc else 1, n + 1, s))

def boot_ci(y, p, fids, rng, n=N_BOOT):
    byf = defaultdict(lambda: ([], []))
    for f, pp, yy in zip(fids, p, y):
        byf[f][0].append(pp); byf[f][1].append(yy)
    facts = list(byf); vals = []
    for _ in range(n):
        s = rng.choice(facts, size=len(facts), replace=True)
        P, Y = [], []
        for f in s:
            P += byf[f][0]; Y += byf[f][1]
        if len(set(Y)) > 1:
            vals.append(roc_auc_score(Y, P))
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))) if vals else (np.nan, np.nan)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--position", default="last_question_token")
    ap.add_argument("--override", nargs="*", default=[])
    args = ap.parse_args()

    cfg = load_config(overrides=args.override)
    mi = model_by_slug(cfg, args.model); slug = mi["slug"]
    act = resolve_path(cfg, "activations_dir")
    variants = load_variants(resolve_path(cfg, "variants_dir") / "sonnet" / "variants.json")
    layers = sweep(int(mi["n_layers"]), int(cfg.layer_sweep.stride), bool(cfg.layer_sweep.include_embedding))
    tf = float(cfg.ablations.mixed_register.train_fraction)
    seed = int(cfg.ablations.mixed_register.seed)
    rng = np.random.default_rng(SEED)
    kw = dict(C=float(cfg.probe.C), max_iter=int(cfg.probe.max_iter), solver=str(cfg.probe.solver))

    # shared fact-level split (from textbook fact ids)
    _, _, m0 = load_probe_matrix(act, variants, slug, layers[0], args.position, registers=["textbook"])
    train_facts, held_facts = fact_level_split([mm["fact_id"] for mm in m0], tf, seed)

    # preload per (layer, register) matrices once
    mat = {}
    for L in layers:
        for reg in REGS:
            X, y, m = load_probe_matrix(act, variants, slug, L, args.position, registers=[reg])
            if X.shape[0]:
                mat[(L, reg)] = (X, y, [mm["fact_id"] for mm in m])

    # training sets: 4 singles, 6 pairs, 1 all-4
    train_sets = [("+".join(c), list(c)) for k in (1, 2, 4)
                  for c in combinations(REGS, k) if k in (1, 2, 4)]
    train_sets = [(name, regs) for name, regs in train_sets if len(regs) in (1, 2, 4)]

    rows = []
    for name, tregs in train_sets:
        # train per layer on union of train-split variants of the training registers
        clfs = {}
        for L in layers:
            Xs, ys = [], []
            for reg in tregs:
                if (L, reg) not in mat:
                    continue
                X, y, fids = mat[(L, reg)]
                keep = np.array([f in train_facts for f in fids])
                if keep.any():
                    Xs.append(X[keep]); ys.append(y[keep])
            if Xs:
                clfs[L] = train_logistic(np.vstack(Xs), np.concatenate(ys), **kw)
        # evaluate on every register's held-out variants, per-cell best layer
        for ereg in REGS:
            best = (-1, None, None)
            for L, clf in clfs.items():
                if (L, ereg) not in mat:
                    continue
                X, y, fids = mat[(L, ereg)]
                keep = np.array([f in held_facts for f in fids])
                if keep.sum() == 0 or len(set(y[keep])) < 2:
                    continue
                p = clf.predict_proba(X[keep])
                a = roc_auc_score(y[keep], p)
                if a > best[0]:
                    best = (a, L, (y[keep], p, [f for f, k in zip(fids, keep) if k]))
            if best[1] is None:
                continue
            a, L, (y, p, fids) = best
            lo, hi = boot_ci(y, p, fids, rng)
            rows.append(dict(model=slug, train_set=name, eval_register=ereg, layer=L,
                             n_eval=len(y) // 2, auroc=round(a, 4), ci_lo=round(lo, 4), ci_hi=round(hi, 4)))
        # mean over eval registers for this train_set
        aus = [r["auroc"] for r in rows if r["train_set"] == name]
        print(f"  train={name:34s} mean eval AUROC={sum(aus)/len(aus):.3f}")

    OUT.mkdir(parents=True, exist_ok=True)
    f = OUT / f"pairwise__{slug}.csv"
    with open(f, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"wrote {f}")

if __name__ == "__main__":
    main()

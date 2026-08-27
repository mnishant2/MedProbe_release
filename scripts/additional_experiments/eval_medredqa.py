#!/usr/bin/env python3
"""Apply the MedQA-textbook probe, without retraining, to the MedRedQA slice and to the Sonnet patient and colloquial registers; reports AUROC with bootstrap CIs and a word-count baseline."""
import argparse, csv, json, os, sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
OUT = REPO / "outputs/camera_ready/medredqa"

import numpy as np
from dotenv import load_dotenv
from sklearn.metrics import roc_auc_score
from medprobe.config import load_config, model_by_slug, resolve_path
from medprobe.probes.ablations import fact_level_split
from medprobe.probes.dataset import load_probe_matrix, load_variants
from medprobe.probes.train import train_logistic

load_dotenv()
N_BOOT = 1000
SEED = 42

# register_source -> (generator dir, register tag inside that variants.json)
CONDITIONS = [
    ("real_patient", "patient-real", "patient_real"),
    ("sonnet_patient", "sonnet", "patient"),
    ("sonnet_colloquial", "sonnet", "colloquial"),
    ("sonnet_patient_lenmatched", "sonnet-lenmatched", "patient"),  # optional control (d)
]

def sweep(n, stride, inc):
    return list(range(0 if inc else 1, n + 1, stride))

def boot_ci(y, p, fids, rng, n=N_BOOT):
    byf = defaultdict(lambda: ([], []))
    for f, pp, yy in zip(fids, p, y):
        byf[f][0].append(pp); byf[f][1].append(yy)
    facts = list(byf)
    vals = []
    for _ in range(n):
        s = rng.choice(facts, size=len(facts), replace=True)
        P, Y = [], []
        for f in s:
            P += byf[f][0]; Y += byf[f][1]
        if len(set(Y)) > 1:
            vals.append(roc_auc_score(Y, P))
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))) if vals else (np.nan, np.nan)

def length_only_auroc(variants, register):
    y, L = [], []
    for e in variants.values():
        if e["register"] == register:
            y.append(e["label"]); L.append(len(e["answer"].split()))
    return float(roc_auc_score(y, L)) if len(set(y)) > 1 else float("nan")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--position", default="last_question_token")
    ap.add_argument("--override", nargs="*", default=[])
    args = ap.parse_args()

    cfg = load_config(overrides=args.override)
    mi = model_by_slug(cfg, args.model); slug = mi["slug"]
    act_root = resolve_path(cfg, "activations_dir")   # PARENT: activations/<generator>/<model>/*.npz
    vdir = resolve_path(cfg, "variants_dir")
    layers = sweep(int(mi["n_layers"]), int(cfg.layer_sweep.stride),
                   bool(cfg.layer_sweep.include_embedding))
    tf = float(cfg.ablations.mixed_register.train_fraction)
    seed = int(cfg.ablations.mixed_register.seed)
    rng = np.random.default_rng(SEED)

    # --- train M_textbook per layer on MedQA sonnet textbook train split (never retrained) ---
    sonnet = load_variants(vdir / "sonnet" / "variants.json")
    clfs, held_facts = {}, None
    for L in layers:
        X, y, m = load_probe_matrix(act_root / "sonnet", sonnet, slug, L, args.position, registers=["textbook"])
        if X.shape[0] == 0:
            continue
        tr_f, he_f = fact_level_split([mm["fact_id"] for mm in m], tf, seed)
        held_facts = he_f
        tr = np.array([mm["fact_id"] in tr_f for mm in m])
        clfs[L] = train_logistic(X[tr], y[tr], C=float(cfg.probe.C),
                                 max_iter=int(cfg.probe.max_iter), solver=str(cfg.probe.solver))

    rows = []
    for source, gen, reg in CONDITIONS:
        vpath = vdir / gen / "variants.json"
        if not vpath.exists():
            print(f"skip {source}: {vpath} not found")
            continue
        V = load_variants(vpath)
        lo_auroc = length_only_auroc(V, reg)
        best = (-1, None, None)
        for L, clf in clfs.items():
            X, y, m = load_probe_matrix(act_root / gen, V, slug, L, args.position, registers=[reg])
            if X.shape[0] == 0:
                continue
            # for the sonnet conditions restrict to the paper's held-out facts; the real-patient
            # set is entirely unseen, so all its facts are used.
            if gen == "sonnet":
                keep = np.array([mm["fact_id"] in held_facts for mm in m])
                X, y, m = X[keep], y[keep], [mm for mm, k in zip(m, keep) if k]
            if len(set(y)) < 2:
                continue
            p = clf.predict_proba(X)   # FittedProbe already returns 1-D P(correct)
            a = roc_auc_score(y, p)
            if a > best[0]:
                best = (a, L, (y, p, [mm["fact_id"] for mm in m]))
        if best[1] is None:
            print(f"skip {source}: no activations")
            continue
        a, L, (y, p, fids) = best
        ci = boot_ci(y, p, fids, rng)
        rows.append(dict(model=slug, register_source=source, auroc=round(a, 4),
                         ci_lo=round(ci[0], 4), ci_hi=round(ci[1], 4), n=len(y),
                         length_only_auroc=round(lo_auroc, 4),
                         probe_minus_length=round(a - lo_auroc, 4), layer=L))
        print(f"  {source:28s} AUROC={a:.3f} [{ci[0]:.3f},{ci[1]:.3f}] n={len(y)} "
              f"length-only={lo_auroc:.3f}  probe-length={a-lo_auroc:+.3f}")

    OUT.mkdir(parents=True, exist_ok=True)
    f = OUT / "r2_2_real_patient_auroc.csv"
    write_header = not f.exists()
    with open(f, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        if write_header:
            w.writeheader()
        w.writerows(rows)
    print(f"\nappended {len(rows)} rows to {f}")

if __name__ == "__main__":
    main()

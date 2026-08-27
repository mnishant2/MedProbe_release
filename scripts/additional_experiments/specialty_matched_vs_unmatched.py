#!/usr/bin/env python3
"""Matched-vs-unmatched specialty check from activations: AUROC of the textbook probe on S-MedQA-matched and unmatched facts with bootstrap CIs, on all variants and on the held-out split."""
import argparse, csv, json, sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
OUT = REPO / "outputs/camera_ready/specialty_selection"

import numpy as np
from dotenv import load_dotenv
from sklearn.metrics import roc_auc_score
from medprobe.config import load_config, model_by_slug, resolve_path
from medprobe.probes.ablations import fact_level_split
from medprobe.probes.dataset import load_probe_matrix, load_variants
from medprobe.probes.train import train_logistic

load_dotenv()
SHIFTED = ["patient", "clinical_note", "colloquial"]
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
    facts = json.loads((resolve_path(cfg, "facts_dir") / "facts.json").read_text())
    spec = {f["id"]: (f.get("specialty") or "unknown") for f in facts}
    layers = sweep(int(mi["n_layers"]), int(cfg.layer_sweep.stride), bool(cfg.layer_sweep.include_embedding))
    tf = float(cfg.ablations.mixed_register.train_fraction)
    seed = int(cfg.ablations.mixed_register.seed)
    rng = np.random.default_rng(SEED)

    # train M_textbook per layer; capture held-out facts
    clfs, held = {}, None
    for L in layers:
        X, y, m = load_probe_matrix(act, variants, slug, L, args.position, registers=["textbook"])
        if X.shape[0] == 0:
            continue
        tr_f, he_f = fact_level_split([mm["fact_id"] for mm in m], tf, seed)
        held = he_f
        tr = np.array([mm["fact_id"] in tr_f for mm in m])
        clfs[L] = train_logistic(X[tr], y[tr], C=float(cfg.probe.C),
                                 max_iter=int(cfg.probe.max_iter), solver=str(cfg.probe.solver))
    if not clfs:
        sys.exit("no textbook activations")

    # cache predictions per (layer, register): fact_id -> {label: prob}
    cache = {}
    for L in clfs:
        for reg in SHIFTED:
            X, y, m = load_probe_matrix(act, variants, slug, L, args.position, registers=[reg])
            if X.shape[0] == 0:
                continue
            p = clfs[L].predict_proba(X)
            d = defaultdict(dict)
            for mm, yy, pp in zip(m, y, p):
                d[mm["fact_id"]][int(yy)] = float(pp)
            cache[(L, reg)] = d

    rows = []
    for eval_set in ["ALL", "HELDOUT"]:
        for subset in ["matched", "unmatched"]:
            def in_subset(fid):
                sp = spec.get(fid, "unknown")
                ok_spec = (sp != "unknown") if subset == "matched" else (sp == "unknown")
                ok_eval = (fid in held) if eval_set == "HELDOUT" else True
                return ok_spec and ok_eval
            # pooled over shifted registers, per-cell best layer chosen on this subset
            for reg in SHIFTED + ["ALL_shifted"]:
                regs = SHIFTED if reg == "ALL_shifted" else [reg]
                best = (-1, None, None)
                for L in clfs:
                    y, p, fids = [], [], []
                    for rr in regs:
                        d = cache.get((L, rr), {})
                        for fid, lab in d.items():
                            if in_subset(fid) and 0 in lab and 1 in lab:
                                y += [1, 0]; p += [lab[1], lab[0]]; fids += [fid, fid]
                    if len(set(y)) > 1:
                        a = roc_auc_score(y, p)
                        if a > best[0]:
                            best = (a, L, (y, p, fids))
                if best[1] is None:
                    continue
                a, L, (y, p, fids) = best
                lo, hi = boot_ci(y, p, fids, rng)
                rows.append(dict(model=slug, eval_set=eval_set, subset=subset, register=reg,
                                 layer=L, n_facts=len(set(fids)), auroc=round(a, 4),
                                 ci_lo=round(lo, 4), ci_hi=round(hi, 4)))
                print(f"  {eval_set:8s} {subset:10s} {reg:14s} L{L} AUROC={a:.3f} [{lo:.3f},{hi:.3f}] nf={len(set(fids))}")

    OUT.mkdir(parents=True, exist_ok=True)
    f = OUT / f"specialty_matched__{slug}.csv"
    with open(f, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"wrote {f}")

if __name__ == "__main__":
    main()

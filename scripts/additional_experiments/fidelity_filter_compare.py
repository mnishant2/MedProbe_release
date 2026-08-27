#!/usr/bin/env python3
"""Fidelity filtering: recompute per-register AUROC and register gaps after dropping facts whose wrong variant falls below a judged fidelity threshold, for several filter definitions and thresholds."""
import csv, glob, json, os, statistics, sys
from collections import defaultdict
import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = os.environ.get("MEDPROBE_ROOT", ".")
PRED = os.path.join(ROOT, "outputs/camera_ready/fidelity_filter")
OUT = PRED
SHIFTED = ["patient", "clinical_note", "colloquial"]
REGISTERS = ["textbook"] + SHIFTED
CUTS = [0.75, 0.85, 1.0]
MODES = ["pooled", "grok", "nano"]
PAPER_GAP = 0.095
TOL = 0.02
N_BOOT = 1000
SEED = 42

def per_judge_fidelity():
    """variant_key(__wrong) -> {judge: factual_fidelity}."""
    out = defaultdict(dict)
    for j in ["grok-fast", "gpt5-nano"]:
        p = os.path.join(ROOT, f"outputs/judge/sonnet/{j}/scores.json")
        if not os.path.exists(p):
            continue
        d = json.load(open(p))
        for k, v in d.items():
            if k.endswith("__wrong") and v.get("factual_fidelity") is not None:
                out[k][j] = v["factual_fidelity"]
    return out

def should_drop(ff_entry, mode, cut):
    """ff_entry = {judge: ff}. Returns True if this fact's wrong variant fails the cut."""
    if not ff_entry:
        return False   # no fidelity info -> keep (unfiltered), warned about separately
    g = ff_entry.get("grok-fast")
    n = ff_entry.get("gpt5-nano")
    vals = [x for x in (g, n) if x is not None]
    if mode == "pooled":
        return (statistics.mean(vals) < cut) if vals else False
    if mode == "grok":
        return (g is not None and g < cut)                # grok judge only
    if mode == "nano":
        return (n is not None and n < cut)                # gpt5-nano judge only
    raise ValueError(mode)

def auroc_pairs(pairs):
    y, p = [], []
    for pc, pw in pairs:
        y += [1, 0]; p += [pc, pw]
    return float(roc_auc_score(y, p)) if len(set(y)) > 1 else float("nan")

def boot_ci(pairs, rng, n=N_BOOT):
    if len(pairs) < 3:
        return float("nan"), float("nan")
    idx = np.arange(len(pairs)); vals = []
    for _ in range(n):
        s = rng.choice(idx, size=len(idx), replace=True)
        y, p = [], []
        for i in s:
            pc, pw = pairs[i]; y += [1, 0]; p += [pc, pw]
        if len(set(y)) > 1:
            vals.append(roc_auc_score(y, p))
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))) if vals else (float("nan"),) * 2

def cell_pairs(bykey, model, layer, reg, drop=frozenset()):
    byfact = defaultdict(dict)
    for (m, L, r, fid, lab), prob in bykey.items():
        if m == model and L == layer and r == reg:
            byfact[fid][lab] = prob
    return [(v[1], v[0]) for f, v in byfact.items() if f not in drop and 0 in v and 1 in v]

def main():
    files = sorted(glob.glob(os.path.join(PRED, "heldout_preds__*.csv")))
    if not files:
        sys.exit(f"No step-1 predictions in {PRED}. Run heldout_predictions.py first")
    bykey = {}
    for f in files:
        for r in csv.DictReader(open(f)):
            bykey[(r["model"], int(r["layer"]), r["register"], r["fact_id"], int(r["label"]))] = float(r["prob"])
    models = sorted({k[0] for k in bykey})
    layers_of = {m: sorted({k[1] for k in bykey if k[0] == m}) for m in models}
    held_facts = {k[3] for k in bykey}
    ff = per_judge_fidelity()
    judged = {k.rsplit("__", 2)[0] for k in ff}
    missing = held_facts - judged
    print(f"models={len(models)} held-out facts={len(held_facts)} judged facts={len(judged)}; "
          f"held-out facts WITHOUT fidelity={len(missing)}")
    if missing:
        print(f"  WARNING: {len(missing)} held-out facts lack fidelity, treated as KEPT in every filter.")

    rng = np.random.default_rng(SEED)
    best_layer = {}
    for m in models:
        for reg in REGISTERS:
            scored = [(auroc_pairs(cell_pairs(bykey, m, L, reg)), L) for L in layers_of[m]]
            scored = [(a, L) for a, L in scored if not np.isnan(a)]
            best_layer[(m, reg)] = max(scored)[1] if scored else layers_of[m][0]

    cell_rows, drop_rows, gap_rows = [], [], []
    filters = ["unfiltered"] + [f"{mode}_FF>={c}" for mode in MODES for c in CUTS]
    for m in models:
        gaps = {}
        for filt in filters:
            aur = {}
            for reg in REGISTERS:
                L = best_layer[(m, reg)]
                drop = set()
                if filt != "unfiltered" and reg in SHIFTED:
                    mode, cut = filt.split("_FF>="); cut = float(cut)
                    for fid in held_facts:
                        if should_drop(ff.get(f"{fid}__{reg}__wrong", {}), mode, cut):
                            drop.add(fid)
                pairs = cell_pairs(bykey, m, L, reg, drop)
                a = auroc_pairs(pairs); lo, hi = boot_ci(pairs, rng)
                aur[reg] = a
                re_a = float("nan")
                if filt != "unfiltered":
                    cand = [auroc_pairs(cell_pairs(bykey, m, LL, reg, drop)) for LL in layers_of[m]]
                    cand = [c for c in cand if not np.isnan(c)]
                    re_a = max(cand) if cand else float("nan")
                cell_rows.append(dict(model=m, register=reg, filter=filt, layer=L, n_facts=len(pairs),
                                      auroc=round(a, 4), ci_lo=round(lo, 4), ci_hi=round(hi, 4),
                                      delta_vs_unfiltered="",
                                      auroc_reargmax=("" if np.isnan(re_a) else round(re_a, 4))))
                if filt != "unfiltered" and reg in SHIFTED:
                    drop_rows.append(dict(model=m, register=reg, filter=filt, dropped=len(drop),
                                          kept=len(pairs), pct_dropped=round(100 * len(drop) / max(1, len(held_facts)), 1)))
            gap = statistics.mean(aur["textbook"] - aur[r] for r in SHIFTED)
            gaps[filt] = gap
            gap_rows.append(dict(model=m, filter=filt, textbook_auroc=round(aur["textbook"], 4),
                                 mean_shifted_auroc=round(statistics.mean(aur[r] for r in SHIFTED), 4),
                                 mean_register_gap=round(gap, 4), delta_vs_unfiltered=""))
        for r in gap_rows:
            if r["model"] == m and r["filter"] != "unfiltered":
                r["delta_vs_unfiltered"] = round(r["mean_register_gap"] - gaps["unfiltered"], 4)

    base = {(r["model"], r["register"]): r["auroc"] for r in cell_rows if r["filter"] == "unfiltered"}
    for r in cell_rows:
        if r["filter"] != "unfiltered":
            r["delta_vs_unfiltered"] = round(r["auroc"] - base[(r["model"], r["register"])], 4)

    def dump(name, rows, fields):
        with open(os.path.join(OUT, name), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields); w.writeheader(); w.writerows(rows)
        print(f"wrote {name} ({len(rows)} rows)")

    dump("heldout_auroc_by_cell_full500.csv", cell_rows,
         ["model", "register", "filter", "layer", "n_facts", "auroc", "ci_lo", "ci_hi", "delta_vs_unfiltered", "auroc_reargmax"])
    dump("register_gap_by_filter_full500.csv", gap_rows,
         ["model", "filter", "textbook_auroc", "mean_shifted_auroc", "mean_register_gap", "delta_vs_unfiltered"])
    dump("per_register_dropped_full500.csv", drop_rows,
         ["model", "register", "filter", "dropped", "kept", "pct_dropped"])

    overall = []
    for filt in filters:
        g = [r["mean_register_gap"] for r in gap_rows if r["filter"] == filt]
        row = dict(filter=filt, mean_register_gap_over_models=round(statistics.mean(g), 4),
                   delta_vs_unfiltered="")
        for reg in REGISTERS:
            a = [r["auroc"] for r in cell_rows if r["filter"] == filt and r["register"] == reg]
            row[f"mean_auroc_{reg}"] = round(statistics.mean(a), 4)
        overall.append(row)
    unf_gap = overall[0]["mean_register_gap_over_models"]
    for row in overall:
        row["delta_vs_unfiltered"] = round(row["mean_register_gap_over_models"] - unf_gap, 4)
    dump("overall_summary_full500.csv", overall, list(overall[0].keys()))

    print("\n=== OVERALL (mean over models) ===")
    for r in overall:
        print(f"  {r['filter']:16s} gap={r['mean_register_gap_over_models']:.4f}  (Δ {r['delta_vs_unfiltered']:+.4f})")

    print(f"\n=== SANITY GATE: unfiltered gap {unf_gap:.4f} vs paper {PAPER_GAP} ===")
    if abs(unf_gap - PAPER_GAP) > TOL:
        print("!" * 70)
        print(f"MISMATCH: unfiltered gap {unf_gap:.4f} not within {TOL} of {PAPER_GAP}. STOPPING.")
        print("  check: position=last_question_token, per-cell layer on UNFILTERED set,")
        print("  held-out 20% facts, M_textbook only (no per-register retrain).")
        print("!" * 70)
        sys.exit(1)
    print("OK, matches the paper.")
    deltas = [abs(r["delta_vs_unfiltered"]) for r in overall if r["filter"] != "unfiltered"]
    print(f"\nVERDICT: across all filter modes/cuts, the mean register gap moves by at most "
          f"{max(deltas):.4f} AUROC.")

if __name__ == "__main__":
    main()

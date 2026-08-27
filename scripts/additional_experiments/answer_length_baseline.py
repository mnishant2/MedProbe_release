#!/usr/bin/env python3
"""Word-count-only baseline: AUROC of a classifier that scores each answer by its length, per variant set and register, on the held-out fact split."""
import json, csv, os, random, statistics
from collections import defaultdict
from sklearn.metrics import roc_auc_score

ROOT = os.environ.get("MEDPROBE_ROOT", ".")
OUT = os.path.join(ROOT, "outputs/camera_ready/length_control")

SETS = {
    "medqa-sonnet": "data/variants/sonnet/variants.json",
    "medqa-gemini": "data/variants/gemini/variants.json",
    "medmcqa-sonnet": "data/variants/medmcqa-sonnet/variants.json",
    "medmcqa-textbook-native": "data/variants/medmcqa-textbook/variants.json",
}

def heldout_medqa():
    v = json.load(open(os.path.join(ROOT, SETS["medqa-sonnet"])))
    uniq = sorted({e["fact_id"] for e in v.values() if e["register"] == "textbook"})
    rng = random.Random(42)
    rng.shuffle(uniq)
    return set(uniq[400:])

def main():
    os.makedirs(OUT, exist_ok=True)
    held = heldout_medqa()
    rows = []
    for name, path in SETS.items():
        p = os.path.join(ROOT, path)
        if not os.path.exists(p):
            print(f"skip {name} (missing)")
            continue
        v = json.load(open(p))
        # For MedQA-sonnet also restrict to the paper's held-out facts (main-table eval set)
        restrict = held if name == "medqa-sonnet" else None
        by = defaultdict(lambda: ([], []))
        for e in v.values():
            if restrict and e["fact_id"] not in restrict:
                continue
            by[e["register"]][0].append(len(e["answer"].split()))
            by[e["register"]][1].append(e["label"])
        for reg, (L, y) in sorted(by.items()):
            if len(set(y)) < 2:
                continue
            au = roc_auc_score(y, L)
            cm = statistics.mean([l for l, yy in zip(L, y) if yy == 1])
            wm = statistics.mean([l for l, yy in zip(L, y) if yy == 0])
            rows.append(dict(variant_set=name, register=reg, n=len(y),
                             length_only_auroc=round(au, 4),
                             mean_words_correct=round(cm, 2), mean_words_wrong=round(wm, 2),
                             heldout_restricted=bool(restrict)))
            print(f"{name:26s} {reg:14s} length-only AUROC={au:.3f}  "
                  f"(correct {cm:.1f}w vs wrong {wm:.1f}w, n={len(y)})")

    with open(os.path.join(OUT, "length_only_auroc.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\nwrote {OUT}/length_only_auroc.csv")

if __name__ == "__main__":
    main()

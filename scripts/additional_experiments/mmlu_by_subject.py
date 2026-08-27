"""Apply the MedQA-textbook probe, without retraining, to MMLU-medical broken down by subject; reports per-subject AUROC and mean question length."""
import argparse, csv, json, sys, statistics
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
OUT = REPO / "outputs/camera_ready/mmlu_medical"

import numpy as np
from dotenv import load_dotenv
from sklearn.metrics import roc_auc_score
from medprobe.config import load_config, model_by_slug, resolve_path
from medprobe.probes.ablations import fact_level_split
from medprobe.probes.dataset import load_probe_matrix, load_variants
from medprobe.probes.train import train_logistic

load_dotenv()

def sweep(n, s, inc):
    return list(range(0 if inc else 1, n + 1, s))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--position", default="last_question_token")
    ap.add_argument("--override", nargs="*", default=[])
    args = ap.parse_args()

    cfg = load_config(overrides=args.override)
    mi = model_by_slug(cfg, args.model); slug = mi["slug"]
    act_root = resolve_path(cfg, "activations_dir")
    vdir = resolve_path(cfg, "variants_dir")
    layers = sweep(int(mi["n_layers"]), int(cfg.layer_sweep.stride), bool(cfg.layer_sweep.include_embedding))
    tf = float(cfg.ablations.mixed_register.train_fraction); seed = int(cfg.ablations.mixed_register.seed)
    kw = dict(C=float(cfg.probe.C), max_iter=int(cfg.probe.max_iter), solver=str(cfg.probe.solver))

    # subject + question length per fact
    facts = json.loads((resolve_path(cfg, "facts_dir") / "mmlu-medical-facts.json").read_text())
    subj = {f["id"]: f["specialty"] for f in facts}
    qlen = {f["id"]: len(f["question"].split()) for f in facts}

    # train M_textbook on MedQA-sonnet textbook train split
    sonnet = load_variants(vdir / "sonnet" / "variants.json")
    clfs = {}
    for L in layers:
        X, y, m = load_probe_matrix(act_root / "sonnet", sonnet, slug, L, args.position, registers=["textbook"])
        if X.shape[0] == 0:
            continue
        tr_f, _ = fact_level_split([mm["fact_id"] for mm in m], tf, seed)
        tr = np.array([mm["fact_id"] in tr_f for mm in m])
        clfs[L] = train_logistic(X[tr], y[tr], **kw)

    # predict on all MMLU variants, cache per layer: fact_id -> {label: prob}
    V = load_variants(vdir / "mmlu-medical" / "variants.json")
    cache = {}
    for L in clfs:
        X, y, m = load_probe_matrix(act_root / "mmlu-medical", V, slug, L, args.position, registers=["textbook"])
        if X.shape[0] == 0:
            continue
        p = clfs[L].predict_proba(X)
        d = defaultdict(dict)
        for mm, yy, pp in zip(m, y, p):
            d[mm["fact_id"]][int(yy)] = float(pp)
        cache[L] = d

    subjects = sorted(set(subj.values()))
    rows = []
    for s in subjects + ["ALL"]:
        fids = [f for f in subj if (s == "ALL" or subj[f] == s)]
        best = -1; bestL = None
        for L in cache:
            yy, pp = [], []
            for f in fids:
                lab = cache[L].get(f, {})
                if 0 in lab and 1 in lab:
                    yy += [1, 0]; pp += [lab[1], lab[0]]
            if len(set(yy)) > 1:
                a = roc_auc_score(yy, pp)
                if a > best:
                    best = a; bestL = L
        ql = statistics.mean(qlen[f] for f in fids)
        rows.append(dict(model=slug, subject=s, n_facts=len(fids),
                         mean_q_words=round(ql, 1), layer=bestL, auroc=round(best, 4)))
        print(f"  {s:22s} n={len(fids):3d} qlen={ql:6.1f}w  AUROC={best:.3f}")

    OUT.mkdir(parents=True, exist_ok=True)
    f = OUT / f"mmlu_by_subject__{slug}.csv"
    with open(f, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"wrote {f}")

if __name__ == "__main__":
    main()

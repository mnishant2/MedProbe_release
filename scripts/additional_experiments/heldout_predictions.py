#!/usr/bin/env python3
"""Per-variant held-out predictions of the textbook probe at every layer, under the main-table protocol, used by fidelity_filter_compare.py and accuracy_f1.py."""
import argparse, csv, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
OUT = REPO / "outputs/camera_ready/fidelity_filter"

import numpy as np
from dotenv import load_dotenv
from medprobe.config import load_config, model_by_slug, resolve_path
from medprobe.probes.ablations import fact_level_split
from medprobe.probes.dataset import load_probe_matrix, load_variants
from medprobe.probes.train import train_logistic

load_dotenv()

def sweep(n_layers, stride, inc_emb):
    return list(range(0 if inc_emb else 1, n_layers + 1, stride))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--generator", default="sonnet")
    ap.add_argument("--position", default="last_question_token")  # paper primary
    ap.add_argument("--override", nargs="*", default=[])
    args = ap.parse_args()

    cfg = load_config(overrides=args.override)
    mi = model_by_slug(cfg, args.model)
    slug = mi["slug"]
    act = resolve_path(cfg, "activations_dir")
    variants = load_variants(resolve_path(cfg, "variants_dir") / args.generator / "variants.json")
    registers = list(cfg.registers)
    train_register = str(cfg.train_register)
    train_fraction = float(cfg.ablations.mixed_register.train_fraction)
    split_seed = int(cfg.ablations.mixed_register.seed)
    layers = sweep(int(mi["n_layers"]), int(cfg.layer_sweep.stride),
                   bool(cfg.layer_sweep.include_embedding))

    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for L in layers:
        # train M_textbook on the textbook 80% train facts at this layer
        Xtb, ytb, mtb = load_probe_matrix(act, variants, slug, L, args.position,
                                          registers=[train_register])
        if Xtb.shape[0] == 0:
            continue
        fact_ids = [m["fact_id"] for m in mtb]
        train_facts, held_facts = fact_level_split(fact_ids, train_fraction, split_seed)
        tr = np.array([m["fact_id"] in train_facts for m in mtb])
        clf = train_logistic(Xtb[tr], ytb[tr], C=float(cfg.probe.C),
                             max_iter=int(cfg.probe.max_iter), solver=str(cfg.probe.solver))

        # predict on the SAME held-out facts, for EVERY register
        for reg in registers:
            Xr, yr, mr = load_probe_matrix(act, variants, slug, L, args.position, registers=[reg])
            keep = np.array([m["fact_id"] in held_facts for m in mr])
            if keep.sum() == 0:
                continue
            probs = clf.predict_proba(Xr[keep])   # FittedProbe already returns 1-D P(correct)
            kept_meta = [m for m, k in zip(mr, keep) if k]
            for m, y, p in zip(kept_meta, yr[keep], probs):
                rows.append(dict(model=slug, layer=L, register=reg, fact_id=m["fact_id"],
                                 variant_key=m.get("variant_key", ""), label=int(y),
                                 prob=float(p)))
        print(f"layer {L}: cumulative rows {len(rows)}", flush=True)

    out = OUT / f"heldout_preds__{slug}.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["model", "layer", "register", "fact_id",
                                           "variant_key", "label", "prob"])
        w.writeheader(); w.writerows(rows)
    print(f"wrote {out} ({len(rows)} rows, {len({r['fact_id'] for r in rows})} held-out facts, "
          f"position={args.position})")

if __name__ == "__main__":
    main()

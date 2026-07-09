#!/usr/bin/env python
"""Calibration experiment + error analysis for the paper."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from medprobe.config import load_config, model_by_slug, resolve_path
from medprobe.logging_utils import setup_logger
from medprobe.probes.ablations import fact_level_split
from medprobe.probes.dataset import load_probe_matrix, load_variants
from medprobe.probes.evaluate import _compute_ece
from medprobe.probes.train import train_logistic

load_dotenv()


def pick_best_layer(
    probe_results: Path, model_slug: str, generator: str
) -> tuple[int, str]:
    """Pick the (layer, position) with the highest AUROC on the textbook register
    (i.e. the held-out in-distribution result)."""
    df = pd.read_csv(probe_results)
    s = df[
        (df.method == "probe")
        & (df.rarity.isna())
        & (df.model_slug == model_slug)
        & (df.generator == generator)
        & (df.register == "textbook")
    ]
    if s.empty:
        raise RuntimeError(f"No probe rows for {model_slug} / {generator}")
    row = s.sort_values("auroc", ascending=False).iloc[0]
    return int(row.layer), str(row.position)


def compute_ece_per_register(
    probe_probs: dict[str, np.ndarray],
    probe_labels: dict[str, np.ndarray],
) -> dict[str, float]:
    return {r: _compute_ece(probe_labels[r], probe_probs[r]) for r in probe_probs}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--generator", default="sonnet")
    ap.add_argument("--layer", type=int, default=None, help="override auto best layer")
    ap.add_argument("--position", default=None, help="override auto best position")
    ap.add_argument("--top-k", type=int, default=8, help="K failures per register to surface")
    ap.add_argument("--override", nargs="*", default=[])
    args = ap.parse_args()

    cfg = load_config(overrides=args.override)
    log = setup_logger("calibrate_and_errors", resolve_path(cfg, "logs_dir"))
    model_info = model_by_slug(cfg, args.model)

    activations_dir = resolve_path(cfg, "activations_dir")
    variants_path = resolve_path(cfg, "variants_dir") / args.generator / "variants.json"
    variants = load_variants(variants_path)

    probe_results_csv = resolve_path(cfg, "probes_dir") / "probe_results.csv"
    if args.layer is None or args.position is None:
        layer, position = pick_best_layer(probe_results_csv, model_info["slug"], args.generator)
        log.info("auto best for %s/%s: layer=%d position=%s", model_info["slug"], args.generator, layer, position)
    else:
        layer, position = args.layer, args.position

    out_dir = resolve_path(cfg, "outputs_dir") / "errors" / model_info["slug"] / args.generator
    out_dir.mkdir(parents=True, exist_ok=True)

    train_fraction = float(cfg.ablations.mixed_register.train_fraction)
    split_seed = int(cfg.ablations.mixed_register.seed)

    # === Load textbook matrix ===
    X_tb, y_tb, metas_tb = load_probe_matrix(
        activations_dir, variants, model_info["slug"], layer, position, registers=["textbook"]
    )
    fact_ids_tb = [m["fact_id"] for m in metas_tb]
    train_facts, held_out_facts = fact_level_split(fact_ids_tb, train_fraction, split_seed)
    train_mask = np.array([m["fact_id"] in train_facts for m in metas_tb])
    held_out_mask = ~train_mask
    log.info("textbook split: train=%d held-out=%d", int(train_mask.sum()), int(held_out_mask.sum()))

    # === Train probe on 80% train facts ===
    probe = train_logistic(X_tb[train_mask], y_tb[train_mask])
    log.info("probe trained")

    # === Further split held-out 20% → calibration (50%) + test (50%) ===
    held_out_fact_ids = sorted({m["fact_id"] for m, keep in zip(metas_tb, held_out_mask) if keep})
    rng = np.random.default_rng(split_seed + 7)
    rng.shuffle(held_out_fact_ids)
    n_calib = len(held_out_fact_ids) // 2
    calib_facts = set(held_out_fact_ids[:n_calib])
    test_facts = set(held_out_fact_ids[n_calib:])
    log.info("calib facts=%d, final test facts=%d", len(calib_facts), len(test_facts))

    calib_mask = np.array([m["fact_id"] in calib_facts for m in metas_tb])
    test_tb_mask = np.array([m["fact_id"] in test_facts for m in metas_tb])

    # Probabilities on calibration set
    p_calib = probe.predict_proba(X_tb[calib_mask])
    y_calib = y_tb[calib_mask]
    log.info("calibration probs: %d", len(p_calib))

    # === Fit Platt (sigmoid) calibration ===
    platt = LogisticRegression(max_iter=1000)
    platt.fit(p_calib.reshape(-1, 1), y_calib)

    # === Fit isotonic calibration ===
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(p_calib, y_calib)

    def calibrated(raw: np.ndarray):
        platted = platt.predict_proba(raw.reshape(-1, 1))[:, 1]
        isotonic = iso.predict(raw)
        return platted, isotonic

    # === Collect per-register predictions ===
    all_rows = []
    raw_per_reg: dict[str, np.ndarray] = {}
    labels_per_reg: dict[str, np.ndarray] = {}
    platt_per_reg: dict[str, np.ndarray] = {}
    iso_per_reg: dict[str, np.ndarray] = {}

    for register in ("textbook", "patient", "clinical_note", "colloquial"):
        if register == "textbook":
            X_r = X_tb[test_tb_mask]
            y_r = y_tb[test_tb_mask]
            metas_r = [m for m, keep in zip(metas_tb, test_tb_mask) if keep]
        else:
            X_r, y_r, metas_r = load_probe_matrix(
                activations_dir, variants, model_info["slug"], layer, position,
                registers=[register],
            )
        raw = probe.predict_proba(X_r)
        platt_p, iso_p = calibrated(raw)
        raw_per_reg[register] = raw
        labels_per_reg[register] = y_r
        platt_per_reg[register] = platt_p
        iso_per_reg[register] = iso_p

        for meta, label, r_raw, r_platt, r_iso in zip(metas_r, y_r, raw, platt_p, iso_p):
            all_rows.append({
                "model_slug": model_info["slug"],
                "generator": args.generator,
                "layer": layer,
                "position": position,
                "register": register,
                "fact_id": meta["fact_id"],
                "variant_key": meta["key"],
                "label": int(label),
                "raw_prob": float(r_raw),
                "platt_prob": float(r_platt),
                "isotonic_prob": float(r_iso),
                "pred_raw":     int(r_raw >= 0.5),
                "correct_raw":  int((r_raw >= 0.5) == label),
            })

    preds_df = pd.DataFrame(all_rows)
    preds_csv = out_dir / "predictions.csv"
    preds_df.to_csv(preds_csv, index=False)
    log.info("wrote predictions: %s  (%d rows)", preds_csv, len(preds_df))

    # === ECE summary ===
    ece_rows = []
    for register in ("textbook", "patient", "clinical_note", "colloquial"):
        ece_raw = _compute_ece(labels_per_reg[register], raw_per_reg[register])
        ece_platt = _compute_ece(labels_per_reg[register], platt_per_reg[register])
        ece_iso = _compute_ece(labels_per_reg[register], iso_per_reg[register])
        from sklearn.metrics import roc_auc_score
        auroc = float(roc_auc_score(labels_per_reg[register], raw_per_reg[register])) if len(set(labels_per_reg[register])) > 1 else float("nan")
        ece_rows.append({
            "register": register,
            "n": len(labels_per_reg[register]),
            "auroc": round(auroc, 3),
            "ece_raw": round(ece_raw, 3),
            "ece_platt": round(ece_platt, 3),
            "ece_isotonic": round(ece_iso, 3),
            "platt_improvement": round(ece_raw - ece_platt, 3),
            "isotonic_improvement": round(ece_raw - ece_iso, 3),
        })
    ece_df = pd.DataFrame(ece_rows)
    ece_csv = out_dir / "ece_summary.csv"
    ece_df.to_csv(ece_csv, index=False)
    log.info("wrote ECE summary: %s\n%s", ece_csv, ece_df.to_string(index=False))

    # === Top-K failures per register ===
    md_lines: list[str] = [
        f"# Top failures — {model_info['slug']} / {args.generator}",
        "",
        f"Probe: logistic regression, layer {layer}, position {position}.",
        f"Sort: highest-confidence wrong predictions (using raw probabilities).",
        "",
    ]
    for register in ("textbook", "patient", "clinical_note", "colloquial"):
        sub = preds_df[preds_df.register == register].copy()
        if sub.empty:
            continue
        # false positives: label=0 but raw_prob high
        fp = sub[(sub.label == 0)].sort_values("raw_prob", ascending=False).head(args.top_k)
        # false negatives: label=1 but raw_prob low
        fn = sub[(sub.label == 1)].sort_values("raw_prob").head(args.top_k)

        md_lines.append(f"## {register} — {args.top_k} false positives (answer is wrong, probe said 'correct')")
        md_lines.append("")
        for _, row in fp.iterrows():
            v = variants[row.variant_key]
            q, a = v.get("question", ""), v.get("answer", "")
            q_short = (q[:120] + "…") if len(q) > 120 else q
            a_short = (a[:180] + "…") if len(a) > 180 else a
            md_lines.append(
                f"- **{row.fact_id}** · raw={row.raw_prob:.3f} · platt={row.platt_prob:.3f} · iso={row.isotonic_prob:.3f}"
            )
            md_lines.append(f"  - Q: {q_short}")
            md_lines.append(f"  - A: {a_short}")
        md_lines.append("")
        md_lines.append(f"## {register} — {args.top_k} false negatives (answer is correct, probe said 'wrong')")
        md_lines.append("")
        for _, row in fn.iterrows():
            v = variants[row.variant_key]
            q, a = v.get("question", ""), v.get("answer", "")
            q_short = (q[:120] + "…") if len(q) > 120 else q
            a_short = (a[:180] + "…") if len(a) > 180 else a
            md_lines.append(
                f"- **{row.fact_id}** · raw={row.raw_prob:.3f} · platt={row.platt_prob:.3f} · iso={row.isotonic_prob:.3f}"
            )
            md_lines.append(f"  - Q: {q_short}")
            md_lines.append(f"  - A: {a_short}")
        md_lines.append("")
    md_path = out_dir / "top_failures.md"
    md_path.write_text("\n".join(md_lines))
    log.info("wrote failure report: %s", md_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

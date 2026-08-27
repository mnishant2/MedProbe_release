#!/usr/bin/env python
"""Inter-rater reliability for two judges on the same generator's variants."""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from omegaconf import OmegaConf
from sklearn.metrics import cohen_kappa_score

from medprobe.config import load_config, resolve_path
from medprobe.logging_utils import setup_logger

load_dotenv()


def _scores_path(cfg, generator: str, judge: str) -> Path:
    return resolve_path(cfg, "outputs_dir") / "judge" / generator / judge / "scores.json"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--generator", required=True)
    ap.add_argument("--judges", nargs=2, required=True, help="two judge profile names to compare")
    ap.add_argument("--override", nargs="*", default=[])
    args = ap.parse_args()

    cfg = load_config(overrides=args.override)
    log = setup_logger("judge_consistency", resolve_path(cfg, "logs_dir"))
    rubric = OmegaConf.to_container(cfg.judge.rubric, resolve=True)
    dims = list(rubric.keys())

    paths = [_scores_path(cfg, args.generator, j) for j in args.judges]
    for p in paths:
        if not p.exists():
            raise FileNotFoundError(p)
    a = json.loads(paths[0].read_text())
    b = json.loads(paths[1].read_text())

    keys = sorted(set(a) & set(b))
    clean_keys = [
        k for k in keys
        if "error" not in a[k] and "error" not in b[k]
        and "composite_score" in a[k] and "composite_score" in b[k]
    ]
    log.info("matched variants: %d  (judge A: %s, B: %s)", len(clean_keys), args.judges[0], args.judges[1])

    # ── Per-variant numeric correlation (composite + dimensions) ──
    rows: list[dict] = []
    corr_results: list[dict] = []
    for field in ["composite_score", *dims]:
        va = np.array([a[k][field] for k in clean_keys], dtype=float)
        vb = np.array([b[k][field] for k in clean_keys], dtype=float)
        if va.std() == 0 or vb.std() == 0:
            pearson = float("nan")
        else:
            pearson = float(np.corrcoef(va, vb)[0, 1])
        corr_results.append({
            "field": field,
            "pearson_r": pearson,
            "mean_a": float(va.mean()),
            "mean_b": float(vb.mean()),
            "delta_mean": float(va.mean() - vb.mean()),
        })

    # ── Per-binary-question agreement (exact match + Cohen's κ) ──
    binary_ids = [f"{d}__{q['id']}" for d, info in rubric.items() for q in info["questions"]]
    all_a: list[int] = []
    all_b: list[int] = []
    per_q: list[dict] = []
    for qid in binary_ids:
        qa = np.array([int(round(a[k].get(qid, 0))) for k in clean_keys])
        qb = np.array([int(round(b[k].get(qid, 0))) for k in clean_keys])
        agree = float((qa == qb).mean())
        if len(set(qa)) > 1 or len(set(qb)) > 1:
            kappa = float(cohen_kappa_score(qa, qb))
        else:
            kappa = float("nan")  # both judges constant → kappa undefined
        per_q.append({
            "question": qid,
            "agreement_rate": agree,
            "cohens_kappa": kappa,
            "mean_a": float(qa.mean()),
            "mean_b": float(qb.mean()),
        })
        all_a.extend(qa.tolist())
        all_b.extend(qb.tolist())
    pooled_agreement = float(np.mean(np.array(all_a) == np.array(all_b)))
    pooled_kappa = float(cohen_kappa_score(all_a, all_b)) if len(set(all_a + all_b)) > 1 else float("nan")

    # ── Disagreements on factual_fidelity (the decisive gate) ──
    ff_disagree: list[dict] = []
    for k in clean_keys:
        ff_a = a[k].get("factual_fidelity", float("nan"))
        ff_b = b[k].get("factual_fidelity", float("nan"))
        if abs(ff_a - ff_b) >= 0.5:  # at least one binary Q disagrees
            ff_disagree.append({
                "key": k,
                "register": a[k].get("register"),
                "label": a[k].get("label"),
                f"ff_{args.judges[0]}": ff_a,
                f"ff_{args.judges[1]}": ff_b,
                f"notes_{args.judges[0]}": a[k].get("notes", "")[:140],
                f"notes_{args.judges[1]}": b[k].get("notes", "")[:140],
            })

    # ── Write report ──
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = resolve_path(cfg, "quality_dir") / f"judge_consistency_{args.generator}_{ts}.md"
    tex_out = resolve_path(cfg, "tables_dir") / f"judge_consistency_{args.generator}_{ts}.csv"
    tex_out.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(corr_results).to_csv(
        tex_out.with_name(f"judge_consistency_{args.generator}_{ts}_corr.csv"), index=False
    )
    pd.DataFrame(per_q).to_csv(
        tex_out.with_name(f"judge_consistency_{args.generator}_{ts}_per_q.csv"), index=False
    )

    lines: list[str] = [
        f"# Judge consistency, {args.generator}  ({args.judges[0]} vs {args.judges[1]})",
        "",
        f"_Matched variants: {len(clean_keys)}. Generated {ts}._",
        "",
        "## Per-field correlation",
        "",
        pd.DataFrame(corr_results).round(3).to_markdown(index=False),
        "",
        "## Pooled agreement on binary questions",
        "",
        f"- Exact-agreement rate: **{pooled_agreement:.3f}**",
        f"- Cohen's kappa:        **{pooled_kappa:.3f}**",
        "",
        "Kappa interpretation (Landis & Koch 1977): <0 poor, 0–0.2 slight, 0.2–0.4 fair, 0.4–0.6 moderate, 0.6–0.8 substantial, 0.8–1.0 near-perfect.",
        "",
        "## Per-question agreement",
        "",
        pd.DataFrame(per_q).round(3).to_markdown(index=False),
        "",
        f"## Factual-fidelity disagreements ({len(ff_disagree)} variants)",
        "",
    ]
    if ff_disagree:
        df_dis = pd.DataFrame(ff_disagree)
        lines.append(df_dis.to_markdown(index=False))
    else:
        lines.append("_No variants where the two judges disagree on factual_fidelity by ≥ 0.5._")
    lines.append("")

    out.write_text("\n".join(lines))
    log.info("report: %s", out)
    log.info(
        "%s vs %s: pooled agreement=%.3f  kappa=%.3f  composite corr=%.3f",
        args.judges[0], args.judges[1],
        pooled_agreement, pooled_kappa,
        next(r["pearson_r"] for r in corr_results if r["field"] == "composite_score"),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

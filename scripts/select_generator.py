#!/usr/bin/env python
"""Combine lexical (Tier 1) + LLM-judge (Tier 2) scores across generators,"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from omegaconf import OmegaConf

from medprobe.config import load_config, resolve_path
from medprobe.data.judge import aggregate
from medprobe.data.quality import compute_quality
from medprobe.logging_utils import setup_logger

load_dotenv()


def _judge_dir(cfg, generator: str, judge_profile: str) -> Path:
    return resolve_path(cfg, "outputs_dir") / "judge" / generator / judge_profile


def _load_variants(cfg, generator: str) -> dict | None:
    p = resolve_path(cfg, "variants_dir") / generator / "variants.json"
    if not p.exists():
        return None
    with p.open() as fh:
        return json.load(fh)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--generators", nargs="+", default=["sonnet", "gpt4o-mini", "gemini"])
    ap.add_argument("--n-facts", type=int, default=None, help="for the report header only")
    ap.add_argument("--judges", nargs="*", default=None, help="single judge profile override per generator")
    ap.add_argument(
        "--multi-judge",
        nargs="*",
        default=None,
        help=(
            "Pool multiple judges per generator. Format: GEN:judge1,judge2 "
            "e.g. --multi-judge sonnet:gpt-4o,gemini-flash gemini:gpt-4o,claude-sonnet "
            "gpt4o-mini:claude-sonnet,gemini-flash"
        ),
    )
    ap.add_argument("--override", nargs="*", default=[])
    args = ap.parse_args()

    multi_map: dict[str, list[str]] = {}
    if args.multi_judge:
        for spec in args.multi_judge:
            if ":" not in spec:
                raise ValueError(f"--multi-judge spec must be GEN:j1,j2 — got {spec!r}")
            gen, js = spec.split(":", 1)
            multi_map[gen] = [j.strip() for j in js.split(",") if j.strip()]

    cfg = load_config(overrides=args.override)
    log = setup_logger("select_generator", resolve_path(cfg, "logs_dir"))

    weights = cfg.judge.selection_weights
    rubric = cfg.judge.rubric

    if args.judges is not None and len(args.judges) != len(args.generators):
        raise ValueError("--judges must have the same length as --generators")

    rows: list[dict] = []
    per_gen_summary: dict[str, dict] = {}
    for i, gen in enumerate(args.generators):
        variants = _load_variants(cfg, gen)
        if variants is None:
            log.warning("No variants for generator %s — skipping", gen)
            continue
        tier1 = compute_quality(variants)

        if gen in multi_map:
            judge_profiles = multi_map[gen]
        elif args.judges:
            judge_profiles = [args.judges[i]]
        else:
            judge_profiles = [str(cfg.judge.route.get(gen, cfg.judge.default_judge))]

        # Load each judge's scores and pool by averaging per-variant scores.
        # This gives us one aggregated summary per generator that already
        # folds in inter-judge disagreement.
        judge_outs: dict[str, dict] = {}
        for jp in judge_profiles:
            scores_path = _judge_dir(cfg, gen, jp) / "scores.json"
            if not scores_path.exists():
                log.warning(
                    "No judge scores for %s @ %s at %s — skipping",
                    gen, jp, scores_path,
                )
                continue
            with scores_path.open() as fh:
                judge_outs[jp] = json.load(fh)
        if not judge_outs:
            log.warning("No judge files for %s — skipping", gen)
            continue
        pooled = _pool_judges(judge_outs, rubric)
        summary = aggregate(pooled, rubric)
        per_gen_summary[gen] = {
            "judge_profile": ",".join(judge_outs.keys()),
            "judge_profiles_list": list(judge_outs.keys()),
            "summary": summary,
            "tier1": tier1,
        }

        # One row per dimension + composite for the flat CSV
        for dim_name, dim_score in summary["dim_means"].items():
            rows.append(
                {
                    "generator": gen,
                    "judge": ",".join(judge_outs.keys()),
                    "dimension": dim_name,
                    "mean": dim_score,
                }
            )
        rows.append(
            {
                "generator": gen,
                "judge": ",".join(judge_outs.keys()),
                "dimension": "composite",
                "mean": summary["composite_score_mean"],
            }
        )
        rows.append(
            {
                "generator": gen,
                "judge": ",".join(judge_outs.keys()),
                "dimension": "factual_fidelity_on_wrong",
                "mean": summary["factual_fidelity_on_wrong"],
            }
        )
        rows.append(
            {
                "generator": gen,
                "judge": ",".join(judge_outs.keys()),
                "dimension": "parse_success_rate",
                "mean": summary["parse_success_rate"],
            }
        )

    if not per_gen_summary:
        log.error("No generators had judge scores — nothing to compare")
        return 3

    # Apply selection weights
    w = OmegaConf.to_container(weights, resolve=True)
    ranked: list[tuple[str, float, dict[str, float]]] = []
    for gen, payload in per_gen_summary.items():
        s = payload["summary"]
        components = {
            "factual_fidelity_on_wrong": _safe(s["factual_fidelity_on_wrong"]),
            "composite_score_mean": _safe(s["composite_score_mean"]),
            "register_authenticity_mean": _safe(s["dim_means"].get("register_authenticity", float("nan"))),
            "parse_success_rate": _safe(s["parse_success_rate"]),
        }
        weighted = sum(components[k] * w.get(k, 0.0) for k in components)
        ranked.append((gen, weighted, components))

    ranked.sort(key=lambda x: x[1], reverse=True)

    _write_report(cfg, args, per_gen_summary, ranked, rows, log)

    winner = ranked[0][0]
    log.info("WINNER: %s (weighted=%.3f)", winner, ranked[0][1])
    return 0


def _safe(x: float) -> float:
    return 0.0 if x != x else float(x)   # NaN → 0


def _pool_judges(judge_outs: dict[str, dict], rubric) -> dict[str, dict]:
    """Average per-variant numeric scores across two or more judges.

    `judge_outs` maps judge_profile -> {variant_key -> score_dict}. Only keys
    present in all judges are kept. For each kept key, numeric fields (composite,
    dimension means, binary Qs) are averaged; non-numeric fields (fact_id, register,
    label, notes) come from the first judge.
    """
    if len(judge_outs) == 1:
        return next(iter(judge_outs.values()))

    dim_keys = list(rubric.keys())
    binary_keys = [f"{d}__{q['id']}" for d, info in rubric.items() for q in info["questions"]]
    numeric_fields = ["composite_score", *dim_keys, *binary_keys]

    # keys present in every judge output with valid rows
    per_judge_keys = [
        {k for k, v in jo.items() if "error" not in v and "composite_score" in v}
        for jo in judge_outs.values()
    ]
    common = sorted(set.intersection(*per_judge_keys))

    pooled: dict[str, dict] = {}
    first_judge = next(iter(judge_outs.values()))
    for k in common:
        rows = [jo[k] for jo in judge_outs.values()]
        new_row: dict = {
            "fact_id": first_judge[k].get("fact_id"),
            "register": first_judge[k].get("register"),
            "label": first_judge[k].get("label"),
            "judge_profile": ",".join(judge_outs.keys()),
        }
        for field in numeric_fields:
            vals = [float(r.get(field, 0.0)) for r in rows if field in r]
            if vals:
                new_row[field] = sum(vals) / len(vals)
        notes = [str(r.get("notes", ""))[:120] for r in rows]
        new_row["notes"] = " | ".join(notes)
        pooled[k] = new_row
    return pooled


def _write_report(cfg, args, per_gen_summary, ranked, rows, log) -> None:
    out_dir = resolve_path(cfg, "tables_dir")
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")

    df = pd.DataFrame(rows)
    csv_path = out_dir / f"generator_selection_{ts}.csv"
    df.to_csv(csv_path, index=False)

    # LaTeX table: generator × (each dimension mean + composite + ff_wrong + parse)
    pivot = df.pivot_table(index="generator", columns="dimension", values="mean")
    keep_order = [
        d for d in (
            "content_preservation",
            "factual_fidelity",
            "register_authenticity",
            "fluency",
            "composite",
            "factual_fidelity_on_wrong",
            "parse_success_rate",
        )
        if d in pivot.columns
    ]
    pivot = pivot[keep_order]
    tex_path = out_dir / f"table_generator_selection_{ts}.tex"
    with tex_path.open("w") as fh:
        fh.write("% Generator selection — LLM-as-judge + weighted criteria\n")
        fh.write(f"% {ts}\n")
        fh.write(pivot.to_latex(float_format="%.3f"))

    md_path = resolve_path(cfg, "quality_dir") / f"generator_selection_{ts}.md"
    md_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = [
        f"# Generator selection — {ts}",
        "",
        f"**Facts (approx):** {args.n_facts or 'from variants'}",
        "",
        "## Per-dimension judge scores",
        "",
        pivot.round(3).to_markdown(),
        "",
        "## Weighted selection",
        "",
        "| rank | generator | weighted | factual_fidelity_on_wrong | composite | register_auth | parse_rate |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for i, (gen, weighted, comps) in enumerate(ranked, start=1):
        star = " ★" if i == 1 else ""
        lines.append(
            f"| {i}{star} | {gen} | {weighted:.3f} | "
            f"{comps['factual_fidelity_on_wrong']:.3f} | "
            f"{comps['composite_score_mean']:.3f} | "
            f"{comps['register_authenticity_mean']:.3f} | "
            f"{comps['parse_success_rate']:.3f} |"
        )

    lines.append("")
    lines.append("## Judge routing (cross-family, anti-egocentric)")
    lines.append("")
    for gen, payload in per_gen_summary.items():
        lines.append(f"- **{gen}** → judged by `{payload['judge_profile']}`")
    lines.append("")
    gate = float(cfg.judge.gates.min_factual_fidelity_on_wrong)
    lines.append(f"## Factual-fidelity gate (target ≥ {gate})")
    lines.append("")
    for gen, payload in per_gen_summary.items():
        ff = payload["summary"]["factual_fidelity_on_wrong"]
        ok = "✓" if (ff == ff and ff >= gate) else "⚠"
        lines.append(f"- {ok} **{gen}** factual_fidelity_on_wrong = {ff:.3f}")

    md_path.write_text("\n".join(lines))

    log.info("CSV:   %s", csv_path)
    log.info("LaTeX: %s", tex_path)
    log.info("MD:    %s", md_path)


if __name__ == "__main__":
    raise SystemExit(main())

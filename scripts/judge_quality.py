#!/usr/bin/env python
"""Run the LLM-as-judge rubric on a generator's variants.json."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv
from omegaconf import OmegaConf

from medprobe.config import load_config, resolve_path
from medprobe.data.judge import aggregate, run_judge
from medprobe.data.openrouter_client import GeneratorProfile, OpenRouterClient
from medprobe.logging_utils import setup_logger

load_dotenv()


def _pick_judge(cfg, generator: str, override: str | None) -> str:
    if override:
        return override
    return str(cfg.judge.route.get(generator, cfg.judge.default_judge))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--generator", required=True, help="variants set to judge (sonnet | gpt4o-mini | gemini | …)")
    ap.add_argument("--judge", default=None, help="override the routed judge profile")
    ap.add_argument("--n-facts", type=int, default=None, help="cap on number of judged facts")
    ap.add_argument("--registers", nargs="*", default=None, help="subset of registers to judge")
    ap.add_argument("--anchor", default="textbook", help="register used as the reference for judging")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=None)
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--variants", default=None, help="override path to variants.json")
    ap.add_argument("--out", default=None, help="override output scores.json path")
    ap.add_argument("--override", nargs="*", default=[])
    args = ap.parse_args()

    cfg = load_config(overrides=args.override)
    log = setup_logger("judge", resolve_path(cfg, "logs_dir"))

    if not cfg.judge.enabled:
        log.error("judge.enabled is false in configs/judge.yaml")
        return 2

    variants_path = (
        Path(args.variants)
        if args.variants
        else resolve_path(cfg, "variants_dir") / args.generator / "variants.json"
    )
    if not variants_path.exists():
        raise FileNotFoundError(f"No variants at {variants_path}")
    with variants_path.open() as fh:
        variants = json.load(fh)

    # Optionally cap to first N facts, downstream `build_judge_jobs` will still
    # pair register variants per fact.
    if args.n_facts is not None:
        fact_ids_seen: list[str] = []
        capped: dict[str, dict] = {}
        for k, v in variants.items():
            fid = v.get("fact_id")
            if fid not in fact_ids_seen:
                if len(fact_ids_seen) >= args.n_facts:
                    continue
                fact_ids_seen.append(fid)
            capped[k] = v
        variants = capped

    judge_profile_name = _pick_judge(cfg, args.generator, args.judge)
    if judge_profile_name not in cfg.judge.profiles:
        raise KeyError(f"Unknown judge profile {judge_profile_name!r}. Known: {list(cfg.judge.profiles)}")

    judge_cfg = OmegaConf.to_container(cfg.judge.profiles[judge_profile_name], resolve=True)
    profile = GeneratorProfile.from_cfg(judge_profile_name, judge_cfg)

    ledger = resolve_path(cfg, "logs_dir") / "cost_ledger.jsonl"
    client = OpenRouterClient(
        base_url=str(cfg.openrouter.base_url),
        api_key_env=str(cfg.openrouter.api_key_env),
        referer=str(cfg.openrouter.referer),
        app_title=str(cfg.openrouter.app_title),
        retry_max_attempts=int(cfg.judge.retry.max_attempts),
        retry_initial_wait=float(cfg.judge.retry.initial_wait_s),
        retry_max_wait=float(cfg.judge.retry.max_wait_s),
        ledger_path=ledger,
    )

    workers = args.workers if args.workers is not None else int(cfg.judge.concurrency.workers)
    out_path = (
        Path(args.out)
        if args.out
        else resolve_path(cfg, "outputs_dir")
        / "judge"
        / args.generator
        / judge_profile_name
        / "scores.json"
    )

    log.info(
        "generator=%s judge=%s variants=%d out=%s",
        args.generator, judge_profile_name, len(variants), out_path,
    )
    judge_out = run_judge(
        variants=variants,
        rubric=cfg.judge.rubric,
        profile=profile,
        client=client,
        out_path=out_path,
        workers=workers,
        resume=not args.no_resume,
        registers_to_judge=list(args.registers) if args.registers else None,
        anchor_register=args.anchor,
        start=args.start,
        end=args.end,
    )

    summary = aggregate(judge_out, cfg.judge.rubric)
    summary_path = out_path.with_name("summary.json")
    summary_path.write_text(json.dumps(summary, indent=2))
    log.info(
        "parse_rate=%.3f composite=%.3f ff_on_wrong=%.3f",
        summary["parse_success_rate"],
        summary["composite_score_mean"],
        summary["factual_fidelity_on_wrong"],
    )
    log.info("Summary: %s", summary_path)

    gate = float(cfg.judge.gates.min_factual_fidelity_on_wrong)
    ff = summary["factual_fidelity_on_wrong"]
    if ff == ff and ff < gate:  # not NaN
        log.warning(
            "Factual fidelity on WRONG answers %.3f is below gate %.3f → consider re-generating",
            ff, gate,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""Generate register variants via OpenRouter. Parallel, resumable, chunkable."""
from __future__ import annotations

import argparse
from pathlib import Path

from dotenv import load_dotenv

from medprobe.config import load_config, resolve_path
from medprobe.data.facts import load_facts
from medprobe.data.generate_variants import generate_variants
from medprobe.data.openrouter_client import GeneratorProfile, OpenRouterClient
from medprobe.logging_utils import setup_logger

load_dotenv()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--generator", default="sonnet", help="name in configs/openrouter.yaml")
    ap.add_argument("--n-facts", type=int, default=None, help="cap on the number of facts to generate")
    ap.add_argument("--registers", nargs="*", default=None, help="subset of registers (default: all)")
    ap.add_argument("--workers", type=int, default=None, help="thread pool size")
    ap.add_argument("--start", type=int, default=0, help="start index in the job list (SLURM array)")
    ap.add_argument("--end", type=int, default=None, help="end index (exclusive) in the job list")
    ap.add_argument("--no-resume", action="store_true", help="disable resume — overwrite existing output")
    ap.add_argument("--facts", default=None, help="override path to facts.json")
    ap.add_argument("--out", default=None, help="override output variants.json path")
    ap.add_argument("--override", nargs="*", default=[], help="OmegaConf dotlist overrides")
    args = ap.parse_args()

    cfg = load_config(overrides=args.override)
    log = setup_logger("generate_variants", resolve_path(cfg, "logs_dir"))

    facts_path = Path(args.facts) if args.facts else (resolve_path(cfg, "facts_dir") / "facts.json")
    if not facts_path.exists():
        raise FileNotFoundError(f"No facts file at {facts_path} — run scripts/build_facts.py first.")
    facts = load_facts(facts_path)
    if args.n_facts is not None:
        facts = facts[: args.n_facts]

    registers = list(args.registers) if args.registers else list(cfg.registers)
    workers = args.workers if args.workers is not None else int(cfg.openrouter.concurrency.workers)

    gen_cfg = cfg.openrouter.generators[args.generator]
    from omegaconf import OmegaConf as OC

    profile = GeneratorProfile.from_cfg(args.generator, OC.to_container(gen_cfg, resolve=True))
    ledger = resolve_path(cfg, "logs_dir") / "cost_ledger.jsonl"
    client = OpenRouterClient(
        base_url=str(cfg.openrouter.base_url),
        api_key_env=str(cfg.openrouter.api_key_env),
        referer=str(cfg.openrouter.referer),
        app_title=str(cfg.openrouter.app_title),
        retry_max_attempts=int(cfg.openrouter.retry.max_attempts),
        retry_initial_wait=float(cfg.openrouter.retry.initial_wait_s),
        retry_max_wait=float(cfg.openrouter.retry.max_wait_s),
        ledger_path=ledger,
    )

    # Per-generator output directory so sonnet + gpt-4o-mini runs never collide.
    variants_dir = resolve_path(cfg, "variants_dir") / args.generator
    out_path = Path(args.out) if args.out else (variants_dir / "variants.json")

    log.info(
        "generator=%s n_facts=%d registers=%s workers=%d out=%s",
        args.generator, len(facts), registers, workers, out_path,
    )
    variants = generate_variants(
        facts=facts,
        registers=registers,
        profile=profile,
        client=client,
        out_path=out_path,
        workers=workers,
        resume=not args.no_resume,
        start=args.start,
        end=args.end,
    )
    n_ok = sum(1 for v in variants.values() if "error" not in v)
    n_err = len(variants) - n_ok
    log.info("Done. %d variants (%d errors). Output: %s", n_ok, n_err, out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

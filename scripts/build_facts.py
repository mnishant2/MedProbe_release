#!/usr/bin/env python
"""Select N facts from the configured dataset, tag rarity, attach one wrong answer."""
from __future__ import annotations

import argparse

from dotenv import load_dotenv

from medprobe.config import load_config, resolve_path
from medprobe.data.facts import build_facts
from medprobe.logging_utils import setup_logger

load_dotenv()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-facts", type=int, default=None, help="override cfg.dataset.n_facts")
    ap.add_argument("--seed", type=int, default=None, help="override cfg.project.seed")
    ap.add_argument("--source", default=None, help="override cfg.dataset.source")
    ap.add_argument("--split", default=None, help="override cfg.dataset.split")
    ap.add_argument("--out", default=None, help="override output facts.json path")
    ap.add_argument("--override", nargs="*", default=[], help="OmegaConf dotlist overrides")
    args = ap.parse_args()

    cfg = load_config(overrides=args.override)
    log = setup_logger("build_facts", resolve_path(cfg, "logs_dir"))

    n = args.n_facts if args.n_facts is not None else int(cfg.dataset.n_facts)
    seed = args.seed if args.seed is not None else int(cfg.project.seed)
    source = args.source or cfg.dataset.source
    split = args.split or cfg.dataset.split
    out = resolve_path(cfg, "facts_dir") / "facts.json"
    if args.out is not None:
        from pathlib import Path

        out = Path(args.out)

    log.info("Building %d facts from %s/%s → %s", n, source, split, out)
    rows = build_facts(
        source_name=source,
        raw_dir=resolve_path(cfg, "raw_dir"),
        out_path=out,
        n_facts=n,
        seed=seed,
        split=split,
    )
    from collections import Counter

    ctr = Counter(r["rarity"] for r in rows)
    log.info("Wrote %d rows. Rarity: %s", len(rows), dict(ctr))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

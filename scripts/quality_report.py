#!/usr/bin/env python
"""Compute quality metrics on a variants.json and write a human-review markdown + CSV."""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from medprobe.config import load_config, resolve_path
from medprobe.data.quality import compute_quality, write_csv_report, write_markdown_report
from medprobe.logging_utils import setup_logger

load_dotenv()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--generator", default="sonnet")
    ap.add_argument("--variants", default=None, help="override path to variants.json")
    ap.add_argument("--override", nargs="*", default=[])
    args = ap.parse_args()

    cfg = load_config(overrides=args.override)
    log = setup_logger("quality_report", resolve_path(cfg, "logs_dir"))

    variants_path = (
        Path(args.variants)
        if args.variants
        else (resolve_path(cfg, "variants_dir") / args.generator / "variants.json")
    )
    if not variants_path.exists():
        raise FileNotFoundError(f"No variants at {variants_path}")
    with variants_path.open() as fh:
        variants = json.load(fh)

    metrics = compute_quality(variants)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    quality_dir = resolve_path(cfg, "quality_dir")
    md = write_markdown_report(
        variants,
        metrics,
        quality_dir / f"quality_pilot_{args.generator}_{ts}.md",
        title=f"MedProbe pilot quality — {args.generator}",
    )
    csv = write_csv_report(metrics, quality_dir / f"quality_pilot_{args.generator}_{ts}.csv")
    (quality_dir / f"quality_metrics_{args.generator}_{ts}.json").write_text(
        json.dumps(metrics, indent=2)
    )
    log.info("Markdown: %s", md)
    log.info("CSV: %s", csv)
    log.info(
        "Pairwise Jaccard: %s",
        {k: round(v, 3) for k, v in metrics["pairwise_jaccard_questions"].items()},
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

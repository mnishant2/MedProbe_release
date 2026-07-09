#!/usr/bin/env python
"""Build MedMCQA "variants" file in the same schema the rest of the pipeline expects."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

from medprobe.config import load_config, resolve_path
from medprobe.data.sources import get_source
from medprobe.logging_utils import setup_logger

load_dotenv()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-facts", type=int, default=500, help="Match the MedQA facts count for symmetric comparison.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split", default="validation", help="MedMCQA split to draw from (test labels are hidden).")
    ap.add_argument("--override", nargs="*", default=[])
    args = ap.parse_args()

    cfg = load_config(overrides=args.override)
    log = setup_logger("build_medmcqa", resolve_path(cfg, "logs_dir"))

    raw_dir = resolve_path(cfg, "raw_dir")
    facts_dir = resolve_path(cfg, "facts_dir")
    variants_dir = resolve_path(cfg, "variants_dir") / "medmcqa-textbook"
    variants_dir.mkdir(parents=True, exist_ok=True)

    src = get_source("medmcqa")
    log.info("downloading MedMCQA into %s", raw_dir)
    src.download(raw_dir)

    log.info("loading %d facts from MedMCQA %s split (seed=%d)", args.n_facts, args.split, args.seed)
    facts = src.load(raw_dir=raw_dir, split=args.split, n=args.n_facts, seed=args.seed)
    log.info("kept %d facts", len(facts))

    # Save facts file with specialty subject pulled from extra
    facts_payload = []
    for f in facts:
        d = f.to_dict()
        # Promote subject to top level for downstream compatibility
        d["specialty"] = (f.extra or {}).get("subject", "unknown")
        facts_payload.append(d)
    facts_path = facts_dir / "medmcqa-facts.json"
    facts_path.write_text(json.dumps(facts_payload, indent=2))
    log.info("wrote %s (%d facts)", facts_path, len(facts_payload))

    # Build variants.json mirroring the MedQA-Sonnet schema. No rewriting:
    # the textbook-register variant IS the native MedMCQA item, used as-is.
    variants: dict[str, dict] = {}
    for f in facts:
        for label_int, ans, tag in [(1, f.correct_answer, "correct"), (0, f.wrong_answer, "wrong")]:
            key = f"{f.id}__textbook__{tag}"
            variants[key] = {
                "fact_id": f.id,
                "register": "textbook",
                "label": label_int,
                "question": f.question,
                "answer": ans,
                "generator": "medmcqa-native",
            }
    out_path = variants_dir / "variants.json"
    out_path.write_text(json.dumps(variants, indent=2))
    log.info("wrote %s (%d variants = %d facts × 2 labels)", out_path, len(variants), len(facts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

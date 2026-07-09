#!/usr/bin/env python
"""Tag data/facts/facts.json with medical-specialty labels from S-MedQA."""
from __future__ import annotations

import argparse
import json
import subprocess
from collections import Counter
from pathlib import Path

from medprobe.config import load_config, resolve_path
from medprobe.logging_utils import setup_logger


def ensure_smedqa(cache_dir: Path) -> Path:
    """Clone S-MedQA into the cache dir if not already there."""
    if (cache_dir / "S-MedQA dataset").exists():
        return cache_dir
    cache_dir.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "clone", "--depth=1",
         "https://github.com/nlp4health-lab/S-MedQA.git", str(cache_dir)],
        check=True,
    )
    return cache_dir


def load_smedqa_specialty_map(cache_dir: Path) -> dict[str, str]:
    ds_dir = cache_dir / "S-MedQA dataset"
    by_q: dict[str, str] = {}
    for split in ("test", "validation", "train"):
        path = ds_dir / f"S-MedQA_{split}.json"
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        for row in data:
            sp = row.get("Specialty")
            q = row.get("Question", "").strip()
            if sp and q:
                by_q[q] = sp
    return by_q


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--facts", default=None)
    ap.add_argument("--cache", default=None, help="Where to clone S-MedQA")
    ap.add_argument("--override", nargs="*", default=[])
    args = ap.parse_args()

    cfg = load_config(overrides=args.override)
    log = setup_logger("tag_specialty", resolve_path(cfg, "logs_dir"))

    facts_path = Path(args.facts) if args.facts else resolve_path(cfg, "facts_dir") / "facts.json"
    cache_dir = Path(args.cache) if args.cache else resolve_path(cfg, "raw_dir") / "S-MedQA"

    log.info("Ensuring S-MedQA is available at %s", cache_dir)
    ensure_smedqa(cache_dir)
    smap = load_smedqa_specialty_map(cache_dir)
    log.info("Loaded %d question→specialty pairs from S-MedQA", len(smap))

    facts = json.loads(facts_path.read_text())
    out_map: dict[str, str] = {}
    tagged = 0
    for f in facts:
        q = f["question"].strip()
        sp = smap.get(q, "unknown")
        f["specialty"] = sp
        out_map[f["id"]] = sp
        if sp != "unknown":
            tagged += 1

    facts_path.write_text(json.dumps(facts, indent=2))
    map_path = facts_path.parent / "specialty_map.json"
    map_path.write_text(json.dumps(out_map, indent=2))

    c = Counter(out_map.values())
    log.info(
        "Tagged %d / %d facts (%.1f%%). Unknown: %d.",
        tagged, len(facts), 100 * tagged / len(facts), len(facts) - tagged,
    )
    log.info("Specialty distribution:")
    for sp, n in c.most_common():
        log.info("  %-35s %d", sp, n)
    log.info("Wrote updated %s and %s", facts_path, map_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

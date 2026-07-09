"""Build the `facts.json` file: N seed facts with rarity tags and one wrong answer each."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .rarity import classify_rarity
from .sources import get_source


def build_facts(
    source_name: str,
    raw_dir: Path,
    out_path: Path,
    n_facts: int,
    seed: int,
    split: str = "test",
) -> list[dict[str, Any]]:
    source = get_source(source_name)
    facts = source.load(raw_dir=raw_dir, split=split, n=n_facts, seed=seed)
    rows: list[dict[str, Any]] = []
    for f in facts:
        d = f.to_dict()
        d["rarity"] = classify_rarity(
            " ".join([f.question, f.correct_answer, f.wrong_answer])
        )
        rows.append(d)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as fh:
        json.dump(rows, fh, indent=2)
    return rows


def load_facts(path: Path) -> list[dict[str, Any]]:
    with path.open() as fh:
        return json.load(fh)

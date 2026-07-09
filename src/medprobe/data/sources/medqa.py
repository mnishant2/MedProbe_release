"""MedQA (US, 4-option English) loader."""
from __future__ import annotations

from pathlib import Path

from .base import BaseSource, RawFact

_HF_ID = "GBaker/MedQA-USMLE-4-options"


class MedQASource(BaseSource):
    name = "medqa"

    def download(self, raw_dir: Path) -> Path:
        from datasets import load_dataset

        raw_dir = raw_dir / "medqa"
        raw_dir.mkdir(parents=True, exist_ok=True)
        load_dataset(_HF_ID, cache_dir=str(raw_dir))
        (raw_dir / ".loaded").write_text(f"{_HF_ID}\n")
        return raw_dir

    def load(
        self,
        raw_dir: Path,
        split: str = "test",
        n: int | None = None,
        seed: int = 42,
    ) -> list[RawFact]:
        import random as _random

        from datasets import load_dataset

        ds = load_dataset(_HF_ID, cache_dir=str(raw_dir / "medqa"))
        ds_split = ds[split]

        rng = _random.Random(seed)
        indices = list(range(len(ds_split)))
        rng.shuffle(indices)
        if n is not None:
            indices = indices[: n * 2]  # overshoot so we can drop malformed rows

        result: list[RawFact] = []
        for i in indices:
            row = ds_split[i]
            options_dict = row.get("options") or {}
            if not isinstance(options_dict, dict):
                continue
            option_strs = [options_dict[k] for k in sorted(options_dict.keys())]
            correct_key = row.get("answer_idx")
            correct_str = options_dict.get(correct_key) or row.get("answer") or ""
            if not correct_str or correct_str not in option_strs:
                # fall back to first option if schema is off
                if not option_strs:
                    continue
                correct_str = row.get("answer", option_strs[0])
            wrong_pool = [o for o in option_strs if o != correct_str]
            if not wrong_pool:
                continue
            wrong = rng.choice(wrong_pool)
            result.append(
                RawFact(
                    id=f"medqa_{split}_{i}",
                    source="medqa",
                    split=split,
                    question=row.get("question", ""),
                    correct_answer=correct_str,
                    wrong_answer=wrong,
                    all_options=option_strs,
                )
            )
            if n is not None and len(result) >= n:
                break
        return result

"""MedMCQA loader for the cross-dataset robustness experiment."""
from __future__ import annotations

from pathlib import Path

from .base import BaseSource, RawFact

_HF_ID = "openlifescienceai/medmcqa"


class MedMCQASource(BaseSource):
    name = "medmcqa"

    def download(self, raw_dir: Path) -> Path:
        from datasets import load_dataset

        raw_dir = raw_dir / "medmcqa"
        raw_dir.mkdir(parents=True, exist_ok=True)
        load_dataset(_HF_ID, cache_dir=str(raw_dir))
        (raw_dir / ".loaded").write_text(f"{_HF_ID}\n")
        return raw_dir

    def load(
        self,
        raw_dir: Path,
        split: str = "validation",
        n: int | None = None,
        seed: int = 42,
    ) -> list[RawFact]:
        """Load up to `n` MedMCQA facts from `split`.

        Defaults to the validation split because MedMCQA's official `test`
        split has hidden labels (only the leaderboard knows them). Validation
        is ~4,200 questions with full labels.
        """
        import random as _random

        from datasets import load_dataset

        ds = load_dataset(_HF_ID, cache_dir=str(raw_dir / "medmcqa"))
        if split not in ds:
            raise KeyError(
                f"medmcqa split {split!r} not found; available: {list(ds.keys())}"
            )
        ds_split = ds[split]

        rng = _random.Random(seed)
        indices = list(range(len(ds_split)))
        rng.shuffle(indices)
        if n is not None:
            indices = indices[: n * 3]  # overshoot to drop malformed/single-correct rows

        result: list[RawFact] = []
        for i in indices:
            row = ds_split[i]
            options = [
                str(row.get("opa") or ""),
                str(row.get("opb") or ""),
                str(row.get("opc") or ""),
                str(row.get("opd") or ""),
            ]
            options = [o.strip() for o in options if o and o.strip()]
            if len(options) < 2:
                continue
            try:
                correct_idx = int(row.get("cop", -1))
            except (TypeError, ValueError):
                continue
            if correct_idx < 0 or correct_idx >= len(options):
                continue
            correct_str = options[correct_idx]
            wrong_pool = [o for o in options if o != correct_str]
            if not wrong_pool:
                continue
            wrong = rng.choice(wrong_pool)

            # subject_name is MedMCQA's own specialty/subject tag; useful for
            # the cross-dataset specialty breakdown in the appendix.
            subject = (row.get("subject_name") or "").strip() or "unknown"

            result.append(
                RawFact(
                    id=f"medmcqa_{split}_{i}",
                    source="medmcqa",
                    split=split,
                    question=str(row.get("question") or ""),
                    correct_answer=correct_str,
                    wrong_answer=wrong,
                    all_options=options,
                    extra={"subject": subject},
                )
            )
            if n is not None and len(result) >= n:
                break
        return result

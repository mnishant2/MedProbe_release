"""Abstract dataset source. Implementations return a list of RawFact."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class RawFact:
    """One medical QA pair, post-source-normalization, pre-register-rewrite."""
    id: str                              # stable id: f"{source}_{split}_{idx}"
    source: str                          # e.g. "medqa"
    split: str                           # e.g. "test"
    question: str
    correct_answer: str
    wrong_answer: str                    # one randomly-chosen wrong option
    all_options: list[str]               # full option list (kept for traceability)
    extra: dict[str, Any] = field(default_factory=dict)  # source-specific metadata (e.g. specialty subject for MedMCQA)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class BaseSource(ABC):
    name: str  # must be unique; matches the registry key

    @abstractmethod
    def download(self, raw_dir: Path) -> Path:
        """Ensure the raw data is on disk under raw_dir. Return path."""

    @abstractmethod
    def load(
        self,
        raw_dir: Path,
        split: str = "test",
        n: int | None = None,
        seed: int = 42,
    ) -> list[RawFact]:
        """Return up to n normalized RawFact objects from the split."""

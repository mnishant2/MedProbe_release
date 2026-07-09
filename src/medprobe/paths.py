"""Single source of truth for filesystem paths."""
from __future__ import annotations

from pathlib import Path


def repo_root() -> Path:
    """Repo root = parent of src/."""
    return Path(__file__).resolve().parents[2]


def ensure(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path

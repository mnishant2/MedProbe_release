"""Dispatches to a source in the registry. Thin wrapper."""
from __future__ import annotations

from pathlib import Path

from .sources import get_source


def download_source(name: str, raw_dir: Path) -> Path:
    source = get_source(name)
    return source.download(raw_dir)

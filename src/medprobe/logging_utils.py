"""Rich-based logger. Writes to console and to outputs/logs/<run>.log."""
from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler

_CONSOLE = Console()


def setup_logger(name: str, logs_dir: Path | None = None, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)
    logger.addHandler(
        RichHandler(console=_CONSOLE, rich_tracebacks=True, show_time=True, show_path=False)
    )
    if logs_dir is not None:
        logs_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        fh = logging.FileHandler(logs_dir / f"{name}-{ts}.log")
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s :: %(message)s"))
        logger.addHandler(fh)
    logger.propagate = False
    return logger


def console() -> Console:
    return _CONSOLE

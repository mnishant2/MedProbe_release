#!/usr/bin/env python
"""Download MedQA test split via the dataset source registry."""
from __future__ import annotations

import argparse
from pathlib import Path

from dotenv import load_dotenv

from medprobe.config import load_config, resolve_path
from medprobe.data.download import download_source
from medprobe.logging_utils import setup_logger

load_dotenv()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=None, help="Dataset source key; defaults to config.dataset.source")
    ap.add_argument("--override", nargs="*", default=[], help="OmegaConf dotlist overrides")
    args = ap.parse_args()

    cfg = load_config(overrides=args.override)
    log = setup_logger("download", resolve_path(cfg, "logs_dir"))
    source = args.source or cfg.dataset.source

    raw_dir = resolve_path(cfg, "raw_dir")
    raw_dir.mkdir(parents=True, exist_ok=True)
    log.info("Downloading source=%s → %s", source, raw_dir)
    out = download_source(source, raw_dir)
    log.info("Done. Data lives at: %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

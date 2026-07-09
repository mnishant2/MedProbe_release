#!/usr/bin/env python
"""Self-consistency baseline: 3 temperature-sampled answers per variant, majority vote."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

from medprobe.config import load_config, model_by_slug, resolve_path
from medprobe.inference.model_loader import load_model
from medprobe.inference.outputs import self_consistency
from medprobe.logging_utils import setup_logger

load_dotenv()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--generator", default="sonnet")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--n-samples", type=int, default=None)
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--override", nargs="*", default=[])
    args = ap.parse_args()

    cfg = load_config(overrides=args.override)
    log = setup_logger("self_consistency", resolve_path(cfg, "logs_dir"))
    model_info = model_by_slug(cfg, args.model)

    variants_path = resolve_path(cfg, "variants_dir") / args.generator / "variants.json"
    variants = json.loads(variants_path.read_text())
    keys = sorted(variants.keys())
    end = args.end if args.end is not None else len(keys)
    keys = keys[args.start : end]

    n_samples = args.n_samples or int(cfg.self_consistency.n_samples)
    temperature = args.temperature or float(cfg.self_consistency.temperature)

    activations_dir = resolve_path(cfg, "activations_dir")
    out_path = activations_dir / model_info["slug"] / "self_consistency.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    existing: dict[str, dict] = {}
    if args.resume and out_path.exists():
        with out_path.open() as fh:
            existing = json.load(fh)

    hf_token = os.environ.get("HF_TOKEN", None) or None
    loaded = load_model(model_info["id"], hf_token=hf_token)
    log.info(
        "Loaded %s on %s; n=%d variants to score (n_samples=%d, temp=%.1f)",
        model_info["id"], loaded.gpu_name, len(keys), n_samples, temperature,
    )
    template = str(cfg.qa_prompt.template)
    max_new = int(cfg.qa_prompt.max_new_tokens)

    for key in tqdm(keys, desc=f"self-consistency[{model_info['slug']}]"):
        if args.resume and key in existing:
            continue
        row = variants[key]
        if "error" in row:
            continue
        try:
            sc = self_consistency(
                loaded,
                question=row["question"],
                answer=row["answer"],
                n_samples=n_samples,
                temperature=temperature,
                max_new_tokens=max_new,
                template=template,
            )
        except Exception as e:  # noqa: BLE001
            log.warning("self_consistency failed on %s: %r", key, e)
            sc = {"error": repr(e)}
        existing[key] = {
            "fact_id": row["fact_id"],
            "register": row["register"],
            "label": row["label"],
            **sc,
        }
        if len(existing) % 50 == 0:
            with out_path.open("w") as fh:
                json.dump(existing, fh, indent=2)

    with out_path.open("w") as fh:
        json.dump(existing, fh, indent=2)
    log.info("Wrote %d rows to %s", len(existing), out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

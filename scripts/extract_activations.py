#!/usr/bin/env python
"""Extract hidden states + output metrics for one model. SLURM-friendly (--start/--end/--resume)."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

from medprobe.config import load_config, model_by_slug, resolve_path
from medprobe.inference.extract import extract_one
from medprobe.inference.io_npz import extraction_path, save_extraction
from medprobe.inference.model_loader import load_model
from medprobe.logging_utils import setup_logger

load_dotenv()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF id or slug from configs/models.yaml")
    ap.add_argument("--generator", default="sonnet", help="which variants set to consume")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=None)
    ap.add_argument("--resume", action="store_true", help="skip variants whose .npz already exists")
    ap.add_argument("--max-new-tokens", type=int, default=None)
    ap.add_argument("--override", nargs="*", default=[])
    args = ap.parse_args()

    cfg = load_config(overrides=args.override)
    log = setup_logger("extract", resolve_path(cfg, "logs_dir"))
    model_info = model_by_slug(cfg, args.model)

    variants_path = resolve_path(cfg, "variants_dir") / args.generator / "variants.json"
    if not variants_path.exists():
        raise FileNotFoundError(f"No variants at {variants_path} — run scripts/generate_variants.py first.")
    with variants_path.open() as fh:
        variants = json.load(fh)
    keys = sorted(variants.keys())
    end = args.end if args.end is not None else len(keys)
    keys = keys[args.start : end]

    activations_dir = resolve_path(cfg, "activations_dir")
    hf_token = os.environ.get("HF_TOKEN", None)
    loaded = load_model(model_info["id"], hf_token=hf_token)
    log.info(
        "Loaded %s (slug=%s) on %s; n_layers=%d hidden=%d",
        model_info["id"], model_info["slug"], loaded.gpu_name, loaded.n_layers, loaded.hidden_dim,
    )

    results_path = activations_dir / model_info["slug"] / "results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict] = {}
    if args.resume and results_path.exists():
        with results_path.open() as fh:
            results = json.load(fh)

    max_new = args.max_new_tokens or int(cfg.qa_prompt.max_new_tokens)
    template = str(cfg.qa_prompt.template)

    for key in tqdm(keys, desc=f"extract[{model_info['slug']}]"):
        row = variants[key]
        if "error" in row:
            continue
        npz_path = extraction_path(activations_dir, model_info["slug"], key)
        if args.resume and npz_path.exists() and key in results:
            continue
        try:
            r = extract_one(
                loaded,
                question=row["question"],
                answer=row["answer"],
                max_new_tokens=max_new,
                template=template,
            )
        except Exception as e:  # noqa: BLE001
            log.warning("extract failed on %s: %r", key, e)
            results[key] = {"error": repr(e)}
            continue
        meta = {
            "fact_id": row["fact_id"],
            "register": row["register"],
            "label": row["label"],
            "generated_text": r.generated_text,
            "mean_token_entropy": r.mean_token_entropy,
            "mean_logprob": r.mean_logprob,
            "verbal_label": r.verbal_label,
            "p_true": r.p_true,
            "n_layers": r.n_layers,
            "hidden_dim": r.hidden_dim,
            "model_id": model_info["id"],
            "model_slug": model_info["slug"],
        }
        save_extraction(npz_path, r.hidden_states, meta)
        results[key] = meta
        # Periodic checkpoint (every 50 items)
        if len(results) % 50 == 0:
            with results_path.open("w") as fh:
                json.dump(results, fh, indent=2)

    with results_path.open("w") as fh:
        json.dump(results, fh, indent=2)
    log.info("Wrote results: %s", results_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Assemble (X, y) matrices for probe training from per-variant .npz files."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from ..inference.io_npz import extraction_path, load_extraction


def load_probe_matrix(
    activations_dir: Path,
    variants: dict[str, Any],
    model_slug: str,
    layer: int,
    position: str,
    registers: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Load hidden states + labels for the given layer/position and return (X, y, meta_rows)."""
    xs: list[np.ndarray] = []
    ys: list[int] = []
    metas: list[dict[str, Any]] = []
    for key, row in variants.items():
        if "error" in row:
            continue
        if registers is not None and row["register"] not in registers:
            continue
        npz = extraction_path(activations_dir, model_slug, key)
        if not npz.exists():
            continue
        hs, _ = load_extraction(npz)
        arr_key = f"layer{layer}_{position}"
        if arr_key not in hs:
            continue
        xs.append(hs[arr_key])
        ys.append(int(row["label"]))
        metas.append({"key": key, **{k: row[k] for k in ("fact_id", "register", "label")}})
    if not xs:
        return np.empty((0, 0)), np.empty((0,)), []
    X = np.stack(xs, axis=0)
    y = np.array(ys, dtype=np.int64)
    return X, y, metas


def load_variants(path: Path) -> dict[str, Any]:
    with path.open() as fh:
        return json.load(fh)


def fact_ids(metas: list[dict[str, Any]]) -> list[str]:
    return [m["fact_id"] for m in metas]

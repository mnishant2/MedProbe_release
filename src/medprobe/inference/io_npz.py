"""Compressed .npz save/load for hidden-state extractions."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def save_extraction(path: Path, hidden_states: dict[str, np.ndarray], meta: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(hidden_states)
    # np.savez cannot store arbitrary python dicts, so we serialize meta to a JSON bytes array
    import json as _json

    payload["__meta__"] = np.frombuffer(_json.dumps(meta).encode("utf-8"), dtype=np.uint8)
    np.savez_compressed(path, **payload)


def load_extraction(path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    import json as _json

    arr = np.load(path, allow_pickle=False)
    meta_bytes = bytes(arr["__meta__"].tobytes())
    meta = _json.loads(meta_bytes.decode("utf-8"))
    hs = {k: arr[k] for k in arr.files if k != "__meta__"}
    return hs, meta


def extraction_path(activations_dir: Path, model_slug: str, variant_key: str) -> Path:
    return activations_dir / model_slug / f"{variant_key}.npz"

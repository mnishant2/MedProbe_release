"""Config loader. Merges default.yaml + models.yaml + openrouter.yaml + probe.yaml"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from .paths import repo_root

_CONFIG_FILES = (
    "default.yaml",
    "models.yaml",
    "openrouter.yaml",
    "probe.yaml",
    "judge.yaml",
)


def load_config(
    config_dir: Path | None = None,
    overrides: list[str] | None = None,
) -> DictConfig:
    """Load + merge all config YAMLs, then apply dotlist overrides.

    overrides is a list like ["dataset.n_facts=20", "project.seed=1"].
    """
    cfg_dir = config_dir or (repo_root() / "configs")
    merged: DictConfig = OmegaConf.create({})
    for fname in _CONFIG_FILES:
        path = cfg_dir / fname
        if not path.exists():
            raise FileNotFoundError(f"Missing config file: {path}")
        merged = OmegaConf.merge(merged, OmegaConf.load(path))
    if overrides:
        merged = OmegaConf.merge(merged, OmegaConf.from_dotlist(overrides))
    # Resolve relative paths against repo root
    root = repo_root()
    for key, val in list(merged.paths.items()):
        p = Path(val)
        if not p.is_absolute():
            merged.paths[key] = str(root / p)
    return merged


def resolve_path(cfg: DictConfig, key: str) -> Path:
    return Path(cfg.paths[key])


def model_by_slug(cfg: DictConfig, slug: str) -> dict[str, Any]:
    for m in cfg.models:
        if m.slug == slug or m.id == slug:
            return OmegaConf.to_container(m, resolve=True)  # type: ignore[return-value]
    raise KeyError(f"Unknown model slug/id: {slug}. Known: {[m.slug for m in cfg.models]}")

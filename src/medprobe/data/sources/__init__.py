"""Dataset source registry. Add a new source by importing + registering here."""
from __future__ import annotations

from .base import BaseSource, RawFact
from .medmcqa import MedMCQASource
from .medqa import MedQASource

_REGISTRY: dict[str, type[BaseSource]] = {
    "medqa": MedQASource,
    "medmcqa": MedMCQASource,
}


def get_source(name: str) -> BaseSource:
    if name not in _REGISTRY:
        raise KeyError(f"Unknown dataset source: {name!r}. Known: {list(_REGISTRY)}")
    return _REGISTRY[name]()


def register_source(name: str, cls: type[BaseSource]) -> None:
    _REGISTRY[name] = cls


__all__ = ["BaseSource", "RawFact", "get_source", "register_source"]

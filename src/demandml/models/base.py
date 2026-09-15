"""The model protocol and registry.

A model is anything with ``fit(X, y, w) -> self`` and ``predict(X) -> array``. Keeping
the surface that small is what lets a seasonal-naive rule and a LightGBM ensemble be
compared by identical code in experiment.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, runtime_checkable

import numpy as np
import polars as pl


@runtime_checkable
class Model(Protocol):
    name: str

    def fit(
        self, X: pl.DataFrame, y: np.ndarray, w: np.ndarray | None = None, **kw: Any
    ) -> "Model": ...

    def predict(self, X: pl.DataFrame) -> np.ndarray: ...


@dataclass
class ModelSpec:
    """A named, parameterised model factory."""

    name: str
    factory: Callable[..., Model]
    params: dict[str, Any] = field(default_factory=dict)
    #: Baselines read raw history columns and need no feature matrix.
    needs_features: bool = True
    #: Free-text note shown in experiment reports.
    note: str = ""

    def build(self, **overrides: Any) -> Model:
        return self.factory(**{**self.params, **overrides})


REGISTRY: dict[str, ModelSpec] = {}


def register(
    name: str, needs_features: bool = True, note: str = "", **params: Any
) -> Callable[[Callable[..., Model]], Callable[..., Model]]:
    def deco(factory: Callable[..., Model]) -> Callable[..., Model]:
        REGISTRY[name] = ModelSpec(
            name=name, factory=factory, params=params, needs_features=needs_features, note=note
        )
        return factory

    return deco


def get_model(name: str, **overrides: Any) -> Model:
    if name not in REGISTRY:
        raise KeyError(f"unknown model {name!r}. Available: {sorted(REGISTRY)}")
    return REGISTRY[name].build(**overrides)


def list_models() -> list[tuple[str, str]]:
    return sorted((s.name, s.note) for s in REGISTRY.values())


class FittedBase:
    """Small shared helper: non-negative integer-ish demand predictions."""

    clip_negative: bool = True

    def _post(self, p: np.ndarray) -> np.ndarray:
        p = np.asarray(p, dtype=np.float64).ravel()
        p = np.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
        return np.clip(p, 0.0, None) if self.clip_negative else p

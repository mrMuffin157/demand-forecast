"""Model zoo. Every model implements the same three-method protocol (see base.py), so
the experiment runner can swap them without knowing what they are."""

from .base import Model, ModelSpec, REGISTRY, register, get_model, list_models
from . import baselines as _baselines  # noqa: F401  (import registers the models)
from . import gbdt as _gbdt  # noqa: F401
from . import linear as _linear  # noqa: F401

__all__ = ["Model", "ModelSpec", "REGISTRY", "register", "get_model", "list_models"]

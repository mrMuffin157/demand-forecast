"""The business assumptions, read from ONE file: `assumptions.toml` at the repo root.

The values shipped there are clearly labelled EXAMPLES for the generated sample data.
Set `DEMANDML_ASSUMPTIONS=/path/to/file.toml` to use your own.
"""
from __future__ import annotations

import os
import tomllib
from pathlib import Path

from demandml import config as C

PATH = Path(os.environ.get("DEMANDML_ASSUMPTIONS", C.ROOT / "assumptions.toml"))

with PATH.open("rb") as _f:
    _A = tomllib.load(_f)

CU: float = float(_A["cost"]["understock"])
CO: float = float(_A["cost"]["overstock"])

HOLDING_RATE: float = float(_A["economics"]["holding_rate"])
RECAPTURE: float = float(_A["economics"]["recapture"])
MARKDOWN_RISK: float = float(_A["economics"]["markdown_risk"])

LEAD_TIME_DAYS: int = int(_A["ordering"]["lead_time_days"])
REVIEW_DAYS: int = int(_A["ordering"]["review_days"])

"""The stores: `dev/market.py` names them, and the feature build must agree.

Each store runs the whole pipeline on its own data directory and lab state; these tests
pin the few places the code itself differs by store (`dev/market.py`). Everything writes
under `tmp_path`.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dev"))


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "dev" / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


MK = _load("market")


def test_current_defaults_to_us_and_follows_the_variable(monkeypatch):
    assert MK.current().key == "us"
    monkeypatch.setenv("DEMANDML_MARKET", " CA ")
    assert MK.current() is MK.MARKETS["ca"]
    assert MK.current().currency == "CAD"


def test_unknown_market_refuses(monkeypatch):
    monkeypatch.setenv("DEMANDML_MARKET", "uk")
    with pytest.raises(SystemExit):
        MK.current()


def test_every_store_trains_on_its_own_countrys_holidays(monkeypatch):
    from demandml import config as C
    assert set(C.MARKET_COUNTRY) == set(MK.MARKETS)
    for key, want in (("us", "US"), ("us2", "US"), ("ca", "CA"), ("au", "AU")):
        monkeypatch.setenv("DEMANDML_MARKET", key)
        assert C.FeatureSpec().country == want == MK.MARKETS[key].country

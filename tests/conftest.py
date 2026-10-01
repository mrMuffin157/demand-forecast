"""Shared test setup.

A scheduled job for a non-default store may run this suite with `DEMANDML_MARKET` set to
that store. Tests are written against the US default and opt into a store explicitly
(`monkeypatch.setenv`), so the variable is cleared for every test -- otherwise a store's
job would fail the default-store tests and stop its own chain.

`DEMANDML_MODELS` is cleared for the same reason: a production job may narrow the board
to a few models, and the suite is written against the full shortlist.
"""

import pytest


@pytest.fixture(autouse=True)
def _us_market_unless_a_test_says_otherwise(monkeypatch):
    monkeypatch.delenv("DEMANDML_MARKET", raising=False)
    monkeypatch.delenv("DEMANDML_MODELS", raising=False)

"""Which store a process serves: `us` (default), `us2`, `ca` or `au`.

    DEMANDML_MARKET=ca  python dev/forecast.py ...

Each store is its own data source, warehouse and currency, so each runs the WHOLE pipeline
separately -- its own data directory (`DEMANDML_DATA`), its own lab state
(`DEMANDML_STATE`) and its own models -- and nothing is summed across stores. This file
only names the stores; the separation comes from those two directories.

The one thing that CAN be shared is training, and only when `DEMANDML_POOL` says so: every
store's models then fit on all stores' rows, with a `store_<key>` marker per row, and still
score and forecast only the store the process serves. The store marker lets one model share
what the stores have in common while still learning each store's own level.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Market:
    key: str
    currency: str

    @property
    def country(self) -> str:
        """Its holiday calendar -- defined in `demandml.config.MARKET_COUNTRY`, because
        the feature build reads it there and cannot import this file."""
        from demandml.config import MARKET_COUNTRY  # noqa: PLC0415
        return MARKET_COUNTRY[self.key]


MARKETS: dict[str, Market] = {
    "us": Market("us", "USD"),
    "ca": Market("ca", "CAD"),
    "au": Market("au", "AUD"),
    "us2": Market("us2", "USD"),   # a second US store with its own catalogue
}


def current() -> Market:
    key = os.environ.get("DEMANDML_MARKET", "us").strip().lower()
    if key not in MARKETS:
        raise SystemExit(f"DEMANDML_MARKET={key!r} is not one of {', '.join(MARKETS)}")
    return MARKETS[key]


def data_dir(key: str, root: str = "/data") -> str:
    """Where a store's data directory lives under a shared root: the default store at
    the root itself, every other store in its own subdirectory."""
    return root if key == "us" else f"{root}/{key}"


def pool_env(root: str = "/data") -> str:
    """The `DEMANDML_POOL` value naming every store's data directory, e.g.
    `us=/data,ca=/data/ca,...` -- what a job sets to train on all of them."""
    return ",".join(f"{k}={data_dir(k, root)}" for k in MARKETS)


def parse_pool(spec: str | None) -> dict[str, str]:
    """`DEMANDML_POOL` -> {store: data dir}. Empty (pooling off) when unset."""
    out: dict[str, str] = {}
    for part in (spec or "").split(","):
        part = part.strip()
        if not part:
            continue
        key, sep, path = part.partition("=")
        key = key.strip().lower()
        if not sep or key not in MARKETS or not path.strip():
            raise SystemExit(f"DEMANDML_POOL entry {part!r} is not <store>=<data dir> "
                             f"with store one of {', '.join(MARKETS)}")
        out[key] = path.strip()
    return out

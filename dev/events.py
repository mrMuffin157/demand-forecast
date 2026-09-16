"""Let a human tell the forecast about something the data cannot know yet.

    uv run python dev/events.py                          # validate + report dev/events.csv
    uv run python dev/events.py --origin 2026-08-10      # what applies at that origin
    uv run python dev/events.py --explain ZA-100-1       # one SKU, event by event

Every model in this folder is fitted on history and can only carry history forward. A
launch, a delisting, a booked wholesale order and a deliberate change of demand level are
all *decisions* that exist in somebody's head months before they exist in the panel. This
file is the one place that knowledge is allowed in.

`dev/forecast.py` applies it automatically. With no events file, or an empty one, the
adjustment is exactly 1.0x + 0 units for every SKU and the forecast is bit-for-bit
what it would have been without this layer -- that is asserted in `tests/test_events.py`.

-----------------------------------------------------------------------------------
WHY THIS IS NOT A MODEL FEATURE
-----------------------------------------------------------------------------------
The obvious design is a `kf_*` column: reconstruct what was known at each origin, feed it
in, let the trees learn the response. That rarely works (see docs/FINDINGS.md). The
typical forward-looking table with real point-in-time history is a promo calendar, and
very few deals are booked anywhere near 180 days ahead -- so at the horizons that drive a
manufacturing decision the training rows contain almost no positive examples, and a model
cannot learn a response it has never seen. Worse, the feature is a live leakage surface:
backfill one event into history and every backtest in this folder improves silently. A
planning table whose rows all carry the same creation date is exactly that failure --
its "historical" values were written with hindsight.

An override applied AFTER the model has no such failure mode. It cannot touch training,
it cannot change a backtest, it is a visible arithmetic step on a visible input file, and
it can be scored against the alternative of not having done it -- which `dev/track.py`
and `dev/event_backtest.py` do.

-----------------------------------------------------------------------------------
THE FILE
-----------------------------------------------------------------------------------
`dev/events.csv`, nine columns, one row per event. It lives in `dev/` rather than `data/`
on purpose: `data/` is gitignored, and this is the one input to the whole pipeline that
is human judgement rather than a derived artefact. It belongs in version control where a
diff shows who changed what.

    scope         `ZA-100-1` (a SKU) | `line:ZA` (a product line) | `all` (catalogue)
    start_date    first day the event affects demand
    end_date      last day. May be blank for a multiplicative event that never ends
                  (a delisting, a permanent repricing). Required for a units event.
    kind          launch | discontinue | promo | price | wholesale | level | other
                  Reporting label only -- the arithmetic is decided by `effect`.
    effect        `x1.15`  multiply demand in the overlap by 1.15
                  `+8000`  add 8,000 units spread evenly across the event window
                  `-500`   remove 500 units the same way
    probability   0-1. How likely is this to actually happen. NOT optional -- see below.
    entered_at    the date this row was written. Immutable. See below.
    author        who is accountable for the number
    note          free text, shown in the report

-----------------------------------------------------------------------------------
THE TWO COLUMNS THAT MAKE IT HONEST
-----------------------------------------------------------------------------------
**`entered_at`** is what stops this becoming a typical hand-maintained forecast table,
whose rows get edited during the period they describe -- scored as recorded, such a table
looks impossibly accurate, and most of it has to be thrown away before it can be compared
with anything. An event is only ever applied to an origin it predates -- `eligible()` enforces
`entered_at <= origin` -- so a row added today cannot improve yesterday's forecast, and
`dev/track.py` can score human input without the scoring being circular. Backdating
`entered_at` defeats this, so `--lock` records a digest of every row alongside when it was
first seen. Afterwards, a locked row whose content changed and a NEW row whose
`entered_at` predates the last lock are both detected and refuse to load. It is a journal,
not a security control -- anyone who can edit `events.csv` can delete the lock file too --
but the edit then shows up in a git diff as a deliberate act rather than passing unnoticed,
which is exactly the difference between this and an edited-in-place forecast table,
where nobody can later say which rows were written with hindsight. A production setup
could replace it with an append-only table and a DB-assigned timestamp.

**`probability`** is mandatory because the one event class with measured outcomes says it
has to be: a sizeable share of planned deals are cancelled and never run (count the
`Canceled` rows in `data/db/deals.parquet` for your own data). A plan is not a fact, and an override that treats it as one buys
stock for launches that slip. The applied effect is always the expected one:

    multiplicative   1 + p * (m - 1)         p=0.5 on x2.0 gives x1.5
    additive         p * units

-----------------------------------------------------------------------------------
THE ARITHMETIC, AND THE ASSUMPTION INSIDE IT
-----------------------------------------------------------------------------------
For a SKU at horizon `h` the forecast window is `(origin, origin + h]` -- opening the day
AFTER the origin, because `units_lag_0` is a feature and counting the origin day would
adjust a day the model was already shown. An event overlapping that window by `d` days:

    multiplicative   factor = 1 + (d / h) * (m_eff - 1)
    additive         delta += units_eff * d / (event duration)

Both pro-rate **flat across days**, which assumes demand is spread evenly over the window.
It is not -- that is the same assumption `policy.interpolate_cumulative` makes between
knots and the same one `track.py` makes on a partial window, and it is wrong in the same
direction for the same reason around a seasonal boundary. It is used here because the
alternative is a within-window demand shape this project does not model, and inventing one
to spend a human's units estimate more precisely would be false precision on top of an
estimate that is already a guess.

Multiplicative events compose (their factors multiply); additive events sum. The base
forecast is scaled first, then the units are added, so a catalogue-level dial does not
also inflate an explicit launch quantity -- a launch estimate of 8,000 units means 8,000
units.

-----------------------------------------------------------------------------------
WHAT IT DELIBERATELY WILL NOT DO
-----------------------------------------------------------------------------------
It will not apply an event whose window has already closed (`end_date <= origin`). The
model has seen those days in its own features; overriding them would double-count
something the panel already recorded.

It will not silently accept an unparseable row. A typo in `effect` or a `probability` of
`110%` raises rather than defaulting, because an override file that quietly drops the row
somebody just added is worse than one that refuses to load.

It will not let you clip demand below zero, and it will not scale beyond `MAX_FACTOR`
without saying so in the report. Both are guards against a misplaced decimal ordering a
warehouse.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: E402
from common import LAB  # noqa: E402

EVENTS = LAB / "events.csv"

#: The append-only journal. See `verify_lock()`.
LOCKFILE = LAB / "events.lock.json"

#: Columns every events file must have. Order is irrelevant; presence is not.
COLUMNS = ("scope", "start_date", "end_date", "kind", "effect", "probability",
           "entered_at", "author", "note")

#: Labels allowed in `kind`. Reporting only -- the maths comes from `effect`. Kept as a
#: closed set so a typo becomes an error rather than a new category nobody notices.
KINDS = ("launch", "discontinue", "promo", "price", "wholesale", "level", "other")

#: A multiplicative event may not scale a window by more than this. A x20 in the file is
#: a misplaced decimal far more often than it is a real forecast, and the difference is
#: several containers. Events above it are rejected, not clipped.
MAX_FACTOR = 10.0

#: An open-ended event (blank `end_date`) runs to this many days past the origin --
#: comfortably past the longest coverage window, so "forever" and "past the horizon"
#: are the same thing in practice.
OPEN_ENDED_DAYS = 3650

_EFFECT = re.compile(r"^\s*(?:([x*])\s*([0-9]*\.?[0-9]+)|([+-])\s*([0-9]*\.?[0-9]+))\s*$",
                     re.IGNORECASE)


class EventError(ValueError):
    """A row in the events file cannot be used. Always names the row."""


class LockError(EventError):
    """The events file has been edited in a way the journal forbids."""


@dataclass(frozen=True)
class Event:
    """One validated row. `multiplier` and `units` are mutually exclusive."""
    scope: str
    start: dt.date
    end: dt.date
    kind: str
    multiplier: float | None
    units: float | None
    probability: float
    entered_at: dt.date
    author: str
    note: str
    row: int

    @property
    def is_multiplicative(self) -> bool:
        return self.multiplier is not None

    @property
    def duration_days(self) -> int:
        return max(1, (self.end - self.start).days + 1)

    def describe(self) -> str:
        eff = (f"x{self.multiplier:g}" if self.is_multiplicative
               else f"{self.units:+,.0f} units")
        return (f"{self.scope} {self.start}..{self.end} {self.kind} {eff} "
                f"p={self.probability:g}")


# --------------------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------------------

def _date(value, field: str, row: int, *, allow_blank: bool = False) -> dt.date | None:
    if value is None or (isinstance(value, str) and not value.strip()):
        if allow_blank:
            return None
        raise EventError(f"row {row}: `{field}` is required")
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    try:
        return dt.date.fromisoformat(str(value).strip()[:10])
    except ValueError as exc:
        raise EventError(f"row {row}: `{field}` = {value!r} is not an ISO date "
                         f"(YYYY-MM-DD)") from exc


def parse_effect(text: str, row: int) -> tuple[float | None, float | None]:
    """`x1.15` -> (1.15, None);  `+8000` -> (None, 8000.0).

    The sigil decides the arithmetic, which is why it is required. A bare `1.15` is
    ambiguous between "multiply by 1.15" and "add 1.15 units" and is rejected rather than
    guessed at -- the two readings differ by several orders of magnitude.
    """
    m = _EFFECT.match(str(text or ""))
    if not m:
        raise EventError(
            f"row {row}: `effect` = {text!r} is not understood. Use `x1.15` to scale "
            f"demand or `+8000` / `-500` to add or remove units.")
    if m.group(1):
        mult = float(m.group(2))
        if mult < 0:
            raise EventError(f"row {row}: `effect` = {text!r} is negative")
        if mult > MAX_FACTOR:
            raise EventError(
                f"row {row}: `effect` = {text!r} scales demand by more than "
                f"{MAX_FACTOR:g}x. If that is genuinely intended, split it into a "
                f"units event where the quantity is explicit.")
        return mult, None
    units = float(m.group(4)) * (-1.0 if m.group(3) == "-" else 1.0)
    return None, units


def parse_row(rec: dict, row: int) -> Event:
    scope = str(rec.get("scope") or "").strip()
    if not scope:
        raise EventError(f"row {row}: `scope` is required "
                         f"(a SKU, `line:<prefix>` or `all`)")

    kind = str(rec.get("kind") or "").strip().lower()
    if kind not in KINDS:
        raise EventError(f"row {row}: `kind` = {rec.get('kind')!r} is not one of "
                         f"{', '.join(KINDS)}")

    mult, units = parse_effect(rec.get("effect"), row)

    start = _date(rec.get("start_date"), "start_date", row)
    end = _date(rec.get("end_date"), "end_date", row, allow_blank=True)
    if end is None:
        if units is not None:
            raise EventError(
                f"row {row}: a units event needs an `end_date`. `{rec.get('effect')}` "
                f"spread over an open-ended window is not a quantity.")
        end = start + dt.timedelta(days=OPEN_ENDED_DAYS)
    if end < start:
        raise EventError(f"row {row}: `end_date` {end} is before `start_date` {start}")

    raw_p = rec.get("probability")
    if raw_p is None or (isinstance(raw_p, str) and not raw_p.strip()):
        raise EventError(
            f"row {row}: `probability` is required. Planned deals are often cancelled; "
            f"an event with no probability is a claim that this one is certain.")
    try:
        p = float(raw_p)
    except (TypeError, ValueError) as exc:
        raise EventError(f"row {row}: `probability` = {raw_p!r} is not a number") from exc
    if not 0.0 <= p <= 1.0:
        raise EventError(f"row {row}: `probability` = {p} is outside 0-1")

    entered = _date(rec.get("entered_at"), "entered_at", row)
    author = str(rec.get("author") or "").strip()
    if not author:
        raise EventError(f"row {row}: `author` is required -- somebody owns this number")

    return Event(scope=scope, start=start, end=end, kind=kind, multiplier=mult,
                 units=units, probability=p, entered_at=entered, author=author,
                 note=str(rec.get("note") or "").strip(), row=row)


def load(path: Path | None = None, *, strict: bool = True,
         check_lock: bool = True) -> list[Event]:
    """Read and validate the events file. A missing file is not an error -- it is the
    default state of this project, and it means "no overrides"."""
    path = Path(path) if path is not None else EVENTS
    if not path.exists():
        return []
    # Everything as text -- this file parses and validates it rather than trusting an
    # inferred schema. `#` starts a comment so a planner can annotate the file in place,
    # and so `events.example.csv` can be copied to `events.csv` and used as it stands.
    raw = pl.read_csv(path, infer_schema_length=0, comment_prefix="#")
    if raw.is_empty():
        return []
    missing = [c for c in COLUMNS if c not in raw.columns]
    if missing:
        raise EventError(f"{path} is missing column(s): {', '.join(missing)}. "
                         f"Required: {', '.join(COLUMNS)}")

    events, problems = [], []
    for i, rec in enumerate(raw.to_dicts(), start=2):     # row 1 is the header
        if not any(str(v or "").strip() for v in rec.values()):
            continue
        try:
            events.append(parse_row(rec, i))
        except EventError as exc:
            problems.append(str(exc))
    if problems and strict:
        raise EventError(f"{path} has {len(problems)} unusable row(s):\n  "
                         + "\n  ".join(problems))
    if check_lock:
        violations = verify_lock(events, load_lock())
        if violations and strict:
            raise LockError(
                f"{path} breaks the append-only journal ({LOCKFILE.name}):\n  "
                + "\n  ".join(violations)
                + "\n\nIf the change is deliberate, re-lock with:  "
                  "uv run python dev/events.py --lock")
    return events


# --------------------------------------------------------------------------------------
# The append-only journal
# --------------------------------------------------------------------------------------
#
# `entered_at` is the column this whole layer's honesty rests on: an event may only be
# applied to an origin it predates, which is what lets `dev/track.py` compare a forecast
# with human input against one without. A CSV in git cannot enforce that on its own --
# nothing stops somebody typing last month's date into a row written today, and the
# resulting forecast would look prescient.
#
# `--lock` records a digest of every row alongside when it was first seen. Afterwards two
# things become detectable:
#
#   * a locked row whose CONTENT changed -- an event rewritten after the fact
#   * a NEW row whose `entered_at` predates the last lock -- a backdated event
#
# It is a journal, not a security control: anyone who can edit `events.csv` can delete
# `events.lock.json` too. What it buys is that the edit shows up in a git diff as a
# deliberate act rather than passing unnoticed, which is the difference between this and
# an edited-in-place forecast table, where nobody can later say which rows are hindsight.
#
# A production setup could replace it with an append-only table with a DB-assigned
# timestamp, which removes the trust assumption entirely.

def row_digest(e: Event) -> str:
    """A stable digest of everything about an event except where it sat in the file."""
    eff = f"x{e.multiplier!r}" if e.is_multiplicative else f"u{e.units!r}"
    payload = "|".join([e.scope, str(e.start), str(e.end), e.kind, eff,
                        repr(e.probability), str(e.entered_at), e.author])
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def load_lock(path: Path | None = None) -> dict:
    path = Path(path) if path is not None else LOCKFILE
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def verify_lock(events: list[Event], lock: dict) -> list[str]:
    """Violations of the journal. Empty list means the file is append-only so far."""
    if not lock or not lock.get("rows"):
        return []
    rows = lock["rows"]
    problems: list[str] = []

    seen = {row_digest(e) for e in events}
    for digest, rec in rows.items():
        if digest not in seen:
            problems.append(
                f"a locked event was changed or removed: {rec.get('describe', digest)} "
                f"(first seen {rec.get('first_seen', '?')})")

    locked_at = lock.get("locked_at")
    if locked_at:
        cutoff = dt.date.fromisoformat(str(locked_at)[:10])
        for e in events:
            if row_digest(e) in rows:
                continue
            if e.entered_at < cutoff:
                problems.append(
                    f"row {e.row}: `entered_at` is {e.entered_at}, before the journal was "
                    f"last locked on {cutoff}, but this row was not in it. An event "
                    f"cannot be entered into the past -- that is the one thing "
                    f"`entered_at` exists to prevent.")
    return problems


def write_lock(events: list[Event], path: Path | None = None) -> Path:
    path = Path(path) if path is not None else LOCKFILE
    path.write_text(json.dumps({
        "locked_at": dt.date.today().isoformat(),
        "rows": {row_digest(e): {"describe": e.describe(),
                                 "entered_at": str(e.entered_at),
                                 "first_seen": dt.date.today().isoformat()}
                 for e in events},
    }, indent=2, sort_keys=True))
    return path


# --------------------------------------------------------------------------------------
# Selection
# --------------------------------------------------------------------------------------

def eligible(events: list[Event], origin: dt.date, horizon: int) -> list[Event]:
    """Events a forecast made at `origin` for `horizon` days is allowed to use.

    Three filters, and the first is the one that matters:

    * `entered_at <= origin` -- it has to have been known. This is what lets `track.py`
      compare a forecast with overrides against one without and have the comparison mean
      something. Without it, re-running an old origin today would apply everything
      learned since and every historical number would improve.
    * the event window has to overlap `(origin, origin + horizon]`. An event that ended
      before the origin is already in the panel the model was fitted on.
    * probability > 0. A zero-probability row is a cancelled plan somebody kept for the
      record, and it contributes nothing.
    """
    lo = origin + dt.timedelta(days=1)
    hi = origin + dt.timedelta(days=horizon)
    return [e for e in events
            if e.entered_at <= origin and e.probability > 0
            and e.start <= hi and e.end >= lo]


def overlap_days(e: Event, origin: dt.date, horizon: int) -> int:
    """Days of `(origin, origin + horizon]` the event covers. Zero if it misses."""
    lo = max(e.start, origin + dt.timedelta(days=1))
    hi = min(e.end, origin + dt.timedelta(days=horizon))
    return max(0, (hi - lo).days + 1)


def _scope_matches(e: Event, skus: pl.Series, lines: pl.Series | None) -> np.ndarray:
    scope = e.scope
    if scope.lower() == "all":
        return np.ones(len(skus), dtype=bool)
    if scope.lower().startswith("line:"):
        want = scope.split(":", 1)[1].strip()
        if lines is None:
            return np.zeros(len(skus), dtype=bool)
        return (lines == want).to_numpy()
    return (skus == scope).to_numpy()


# --------------------------------------------------------------------------------------
# Application
# --------------------------------------------------------------------------------------

def adjust(base: np.ndarray, skus: pl.Series, origin: dt.date, horizon: int,
           events: list[Event], lines: pl.Series | None = None
           ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Apply every eligible event to one horizon's forecast.

    Returns `(adjusted, factor, delta, n_applied)` -- the pieces as well as the answer,
    because whoever is looking at a number that moved is owed the decomposition, and
    `EVENTS.md` prints it.

    `adjusted = base * factor + delta`, floored at zero. Scaling happens before the
    addition so an explicit units estimate arrives at its stated size.
    """
    base = np.asarray(base, dtype=np.float64)
    factor = np.ones(len(base), dtype=np.float64)
    delta = np.zeros(len(base), dtype=np.float64)
    n = np.zeros(len(base), dtype=np.int64)

    for e in eligible(events, origin, horizon):
        d = overlap_days(e, origin, horizon)
        if d <= 0:
            continue
        hit = _scope_matches(e, skus, lines)
        if not hit.any():
            continue
        n += hit
        if e.is_multiplicative:
            # Expected multiplier, then pro-rated to the share of the window it covers.
            m_eff = 1.0 + e.probability * (e.multiplier - 1.0)
            factor[hit] *= 1.0 + (d / horizon) * (m_eff - 1.0)
        else:
            delta[hit] += e.probability * e.units * d / e.duration_days

    return np.clip(base * factor + delta, 0.0, None), factor, delta, n


def lines_for(df: pl.DataFrame, sku_col: str = "sku") -> pl.Series | None:
    """The product-line column `line:` scopes match against, if the frame carries one."""
    for c in ("sku_brand", "brand", "line"):
        if c in df.columns:
            return df.get_column(c).cast(pl.String)
    return None


# --------------------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------------------

def report(events: list[Event], origin: dt.date, horizons: tuple[int, ...],
           path: Path | None) -> str:
    L: list[str] = []
    A = L.append
    A("# Known future events\n")
    A(f"*Generated by `dev/events.py` from `{(path or EVENTS).name}` "
      f"at origin **{origin}**.*\n")

    if not events:
        A("**The events file is empty or absent, so no forecast is being overridden.** "
          "That is the default state of this project and it is a valid one -- every "
          "number in `dev/forecasts/` is the model's own.\n")
        A("To add one, create `dev/events.csv` with these columns:\n")
        A("```")
        A(",".join(COLUMNS))
        A("ZA-100-1,2027-02-01,2027-12-31,launch,+8000,0.9,"
          f"{origin},you,\"new variant\"")
        A("```")
        A("\nRead the module docstring first -- particularly why `entered_at` and "
          "`probability` are not optional.\n")
        return "\n".join(L)

    A(f"{len(events)} event(s) in the file.\n")

    A("## What applies at this origin\n")
    A("An event is applied only if it was entered on or before the origin and its window "
      "overlaps the forecast window. Anything that fails either test is listed below as "
      "*not applied*, with the reason.\n")
    A("| scope | window | kind | effect | p | 30 d | 90 d | 180 d | author |")
    A("|---|---|---|---|---|---|---|---|---|")
    for e in sorted(events, key=lambda e: (e.start, e.scope)):
        cells = []
        for h in horizons:
            d = overlap_days(e, origin, h) if e.entered_at <= origin else 0
            if d <= 0:
                cells.append("—")
            elif e.is_multiplicative:
                m_eff = 1.0 + e.probability * (e.multiplier - 1.0)
                cells.append(f"x{1.0 + (d / h) * (m_eff - 1.0):.3f}")
            else:
                cells.append(f"{e.probability * e.units * d / e.duration_days:+,.0f}")
        eff = (f"x{e.multiplier:g}" if e.is_multiplicative else f"{e.units:+,.0f}")
        end = "open" if (e.end - e.start).days >= OPEN_ENDED_DAYS else str(e.end)
        A(f"| `{e.scope}` | {e.start} → {end} | {e.kind} | `{eff}` | "
          f"{e.probability:.0%} | {cells[0]} | {cells[1]} | {cells[2]} | {e.author} |")
    A("")

    late = [e for e in events if e.entered_at > origin]
    if late:
        A("## Not applied — entered after the origin\n")
        A("These were written down after the forecast was made, so applying them would "
          "be hindsight. They will apply to the next origin.\n")
        for e in late:
            A(f"- `{e.scope}` — entered {e.entered_at}, origin is {origin}")
        A("")

    past = [e for e in events if e.entered_at <= origin and e.end <= origin]
    if past:
        A("## Not applied — already over\n")
        A("The model has these days in its own history, so overriding them would "
          "double-count what the panel already recorded.\n")
        for e in past:
            A(f"- `{e.scope}` — ended {e.end}")
        A("")

    A("## By author\n")
    A("| author | events | multiplicative | units |")
    A("|---|---|---|---|")
    for a in sorted({e.author for e in events}):
        mine = [e for e in events if e.author == a]
        u = sum(e.probability * e.units for e in mine if not e.is_multiplicative)
        A(f"| {a} | {len(mine)} | {sum(e.is_multiplicative for e in mine)} | "
          f"{u:+,.0f} |")
    A("")

    A("## How to read this\n")
    A("- **The effect columns are already expected values.** A `x2.0` at 50% probability "
      "shows as the `x1.5` it will actually be applied as, pro-rated again by how much "
      "of each window the event covers. The 180-day column is smaller than the 30-day "
      "one for the same event because it is diluted across six times as many days.\n")
    A("- **Pro-rating is flat across days.** Demand is not, so an event sitting on a "
      "seasonal peak is under-weighted here. This is the same assumption "
      "`policy.interpolate_cumulative` makes between knots.\n")
    A("- **Nothing here changes a backtest.** Overrides are applied after the model, to "
      "the forward forecast only. `dev/event_backtest.py` is what measures whether "
      "they help.\n")
    return "\n".join(L)


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--events", type=str, default=None,
                    help=f"events CSV (default {EVENTS.relative_to(LAB.parent)})")
    ap.add_argument("--origin", type=str, default=None,
                    help="origin date to evaluate against (default: today)")
    ap.add_argument("--explain", type=str, default=None,
                    help="show every event touching one SKU")
    ap.add_argument("--out", type=str, default=None, help="report path (default dev/EVENTS.md)")
    ap.add_argument("--lock", action="store_true",
                    help="record the current rows in the append-only journal")
    args = ap.parse_args()

    path = Path(args.events) if args.events else EVENTS
    origin = (dt.date.fromisoformat(args.origin) if args.origin else dt.date.today())

    print(f"\n{'=' * 86}\nKNOWN FUTURE EVENTS   origin {origin}\n{'=' * 86}")
    try:
        events = load(path, check_lock=not args.lock)
    except EventError as exc:
        raise SystemExit(f"\n{exc}\n")

    if args.lock:
        lp = write_lock(events)
        print(f"  journal: {len(events)} row(s) locked -> {lp.name}")

    if not events:
        print(f"  no events ({path} is absent or empty) -- forecasts are unmodified")
    else:
        print(f"  {len(events)} event(s) from {path.name}")
        for h in common.HORIZONS:
            ok = eligible(events, origin, h)
            print(f"  {h:>4} d: {len(ok)} applicable")

    if args.explain:
        print(f"\n  events touching {args.explain}:")
        skus = pl.Series("sku", [args.explain])
        for e in events:
            if _scope_matches(e, skus, None).any() or e.scope.lower() == "all":
                for h in common.HORIZONS:
                    d = overlap_days(e, origin, h)
                    print(f"    {e.describe()}  h={h}: {d} overlap day(s)")

    out = Path(args.out) if args.out else LAB / "EVENTS.md"
    out.write_text(report(events, origin, common.HORIZONS, path))
    print(f"\nwrote {out.relative_to(LAB.parent)}")


if __name__ == "__main__":
    main()

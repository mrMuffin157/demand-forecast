"""Does the data still look like the data this pipeline was built on?

    uv run python dev/datacheck.py                 # compare against the last accepted build
    uv run python dev/datacheck.py --accept        # bless the current build as the baseline
    uv run python dev/datacheck.py --strict        # exit non-zero on any WARN as well

Every other check in this project asks whether a MODEL is right. This asks whether the
INPUT is, and it exists because of what happens when the answer is no.

A model bug makes a score worse and somebody notices. A data bug makes the pipeline
produce *plausible* numbers: a units column that arrives scaled by 1,000, a renamed price
field that silently reads as null, a marketplace added so every SKU's history doubles, a
feed that stops updating while still returning rows. None of those raise. All of them
flow through the panel, the features, the models and `policy.py`, and come out the far
end as a purchase order somebody signs.

In a staging workbench that risk is small: the person running it would notice. Pointed at
a live feed on a schedule, nobody is watching, and the first symptom is a container of the
wrong thing.

-----------------------------------------------------------------------------------
HOW IT WORKS
-----------------------------------------------------------------------------------
It fingerprints the built artefacts -- row and SKU counts, date coverage, null rates,
and quantiles of the columns everything downstream depends on -- and compares that
fingerprint against the last one that was explicitly accepted, in
`data/processed/_datacheck_baseline.json`.

Quantiles rather than means because the failure being hunted is a change of SCALE or of
SHAPE, and a median that moves 40x is unambiguous where a mean can be dragged by one
wholesale order. The panel is deliberately right-skewed, so its mean is the least stable
summary it has.

  FAIL   the pipeline's own invariants are broken, or something moved so far that no
         ordinary week explains it. Exits non-zero.
  WARN   moved more than the tolerance but inside the range real seasonality produces.
         Reported; only fatal under `--strict`.
  PASS   inside tolerance.

There is no baseline on a first run and everything is reported as NEW. That is not a
pass -- it means nothing has been checked yet.

-----------------------------------------------------------------------------------
WHAT IT DOES NOT DO
-----------------------------------------------------------------------------------
It does not repair anything, and it does not know which of two builds is correct. A
deliberate change -- a new marketplace, a genuine catalogue expansion -- looks exactly
like a defect from here, so `--accept` exists and a human runs it. The check's job is to
make sure the change is NOTICED, not to decide whether it was wanted.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import LAB  # noqa: E402
from demandml import config as C  # noqa: E402

PANEL = C.PROCESSED / "panel_daily.parquet"
DATASET = C.PROCESSED / "lab_dataset.parquet"
OPEN = C.PROCESSED / "open_origins.parquet"
BASELINE = C.PROCESSED / "_datacheck_baseline.json"

#: Columns whose distribution is fingerprinted. These are the ones a rescale or a
#: silent rename would corrupt without raising anywhere.
WATCHED = ("units", "unit_price", "sale_price", "average_cost", "po_price")

#: Relative move that is a WARN, and the one that is a FAIL. Deliberately wide: this
#: panel is seasonal and a 20% swing in a quantile between builds is an ordinary
#: November. A 3x move is not.
WARN_AT = 0.25
FAIL_AT = 2.00

#: A build whose newest day is older than this is stale, whatever else is true. Longer
#: than a weekly feed's worst case. A frozen local extract (or the generated sample) will
#: trip it, which is expected.
STALE_DAYS = 10

#: Spacing between forecast origins, in days -- `model_lab/prepare_data.py` emits one
#: origin per week. Used to judge how far the dataset may legitimately trail the panel.
ORIGIN_SPACING_DAYS = 7


def fingerprint() -> dict:
    """Summarise the built artefacts. Cheap: quantiles and counts, no full scan needed."""
    if not PANEL.exists():
        raise SystemExit(f"{PANEL} does not exist. Build it:  uv run demandml panel")

    lf = pl.scan_parquet(PANEL)
    cols = set(lf.collect_schema().names())
    watched = [c for c in WATCHED if c in cols]

    aggs = [pl.len().alias("rows"),
            pl.col(C.SKU).n_unique().alias("skus"),
            pl.col(C.DATE).min().alias("date_min"),
            pl.col(C.DATE).max().alias("date_max")]
    for c in watched:
        aggs += [
            pl.col(c).null_count().alias(f"{c}__nulls"),
            pl.col(c).quantile(0.5).alias(f"{c}__p50"),
            pl.col(c).quantile(0.9).alias(f"{c}__p90"),
            pl.col(c).max().alias(f"{c}__max"),
        ]
    if C.TARGET in cols:
        aggs.append((pl.col(C.TARGET) < 0).sum().alias("negative_units"))
    if "is_oos" in cols:
        aggs.append(pl.col("is_oos").cast(pl.Float64).mean().alias("oos_rate"))

    got = lf.select(aggs).collect(engine="streaming").to_dicts()[0]
    fp: dict = {"generated": dt.datetime.now().isoformat(timespec="seconds"),
                "panel": {k: (str(v) if isinstance(v, dt.date) else v)
                          for k, v in got.items()},
                "columns": sorted(cols)}

    for name, path in (("lab_dataset", DATASET), ("open_origins", OPEN)):
        if path.exists():
            d = pl.scan_parquet(path).select(
                pl.len().alias("rows"), pl.col(C.SKU).n_unique().alias("skus"),
                pl.col(C.DATE).max().alias("date_max")).collect().to_dicts()[0]
            d["date_max"] = str(d["date_max"])
            fp[name] = d
    return fp


def invariants(fp: dict) -> list[tuple[str, str, str]]:
    """Checks that need no baseline -- these are true or the build is broken.

    A first run has nothing to compare against, so without these `--accept` on a corrupt
    build would bless it permanently.
    """
    out: list[tuple[str, str, str]] = []
    p = fp["panel"]

    if p.get("negative_units", 0):
        out.append(("FAIL", "negative units",
                    f"{p['negative_units']:,} rows have units < 0"))
    if not p["rows"] or not p["skus"]:
        out.append(("FAIL", "empty panel", "no rows or no SKUs"))

    # The leaky export columns must not have survived into the panel. They are rolling
    # averages computed at row-write time, so a model given one scores beautifully and
    # forecasts nothing.
    leaked = sorted(set(getattr(C, "DAILY_LEAKY_COLS", ())) & set(fp["columns"]))
    if leaked:
        out.append(("FAIL", "leaky columns present",
                    f"{', '.join(leaked)} should be dropped by clean.py"))

    dmax = dt.date.fromisoformat(p["date_max"])
    age = (dt.date.today() - dmax).days
    if age > STALE_DAYS:
        out.append(("WARN", "stale data",
                    f"newest panel day is {dmax} ({age} days old). Expected on a "
                    f"frozen extract or the sample data; on a live feed it means the "
                    f"pull stopped (fatal under --strict)."))

    # The two dataset files must stay disjoint and correctly ordered, which is the whole
    # reason the split exists. `tests/test_forecast_origins.py` asserts this properly;
    # repeated here because that test skips when the data is absent, which is exactly
    # the case in CI.
    if "lab_dataset" in fp and "open_origins" in fp:
        lab = dt.date.fromisoformat(fp["lab_dataset"]["date_max"])
        opn = dt.date.fromisoformat(fp["open_origins"]["date_max"])
        if opn <= lab:
            out.append(("FAIL", "open origins are not newer",
                        f"open_origins ends {opn}, lab_dataset ends {lab} -- a forecast "
                        f"would be answering a question about the past"))

    # The dataset files must keep up with the PANEL. Nothing else here checks this, and
    # it is an easy failure for a scheduled pipeline: a job that rebuilds the panel and
    # the features every run but never re-runs `prepare_data.py` leaves `forecast.py`
    # training on whatever `lab_dataset.parquet` was there first. The forecast origin
    # freezes and the output filename stops changing, yet the order plan still MOVES
    # because the stock position it subtracts is refreshed separately -- so the plan
    # looks alive while the demand underneath it is a fossil.
    #
    # The check above cannot catch that: the two dataset files go stale together, so
    # their ordering stays valid. Neither can the `STALE_DAYS` check, which measures the
    # panel against the wall clock, not the dataset against the panel.
    #
    # On a correct build origins are weekly (prepare_data.ORIGIN_WEEKDAY), so
    # `open_origins` ends on the last origin weekday at or before the panel's newest day
    # -- a gap of 0-6 days (a Monday origin against a Thursday panel end is 3). One
    # skipped rebuild puts it at 7-13, two at 14+. Hence WARN past one origin week and
    # FAIL past two.
    if "open_origins" in fp:
        opn = dt.date.fromisoformat(fp["open_origins"]["date_max"])
        pan = dt.date.fromisoformat(fp["panel"]["date_max"])
        gap = (pan - opn).days
        if gap > 2 * ORIGIN_SPACING_DAYS:
            out.append(("FAIL", "dataset is behind the panel",
                        f"open_origins ends {opn}, the panel ends {pan} ({gap} days "
                        f"behind). The forecast would train on a stale dataset while the "
                        f"panel moved on. Rebuild it:  "
                        f"uv run python model_lab/prepare_data.py"))
        elif gap > ORIGIN_SPACING_DAYS:
            out.append(("WARN", "dataset is behind the panel",
                        f"open_origins ends {opn}, the panel ends {pan} ({gap} days "
                        f"behind) -- more than one origin week, so a rebuild was likely "
                        f"skipped. Run model_lab/prepare_data.py before forecasting."))
    return out


def compare(fp: dict, base: dict | None) -> list[tuple[str, str, str]]:
    out: list[tuple[str, str, str]] = []
    if base is None:
        return [("NEW", "no baseline",
                 "nothing has been accepted yet -- run --accept on a build you trust")]

    new_cols = sorted(set(fp["columns"]) - set(base["columns"]))
    gone = sorted(set(base["columns"]) - set(fp["columns"]))
    if gone:
        out.append(("FAIL", "columns disappeared", ", ".join(gone)))
    if new_cols:
        out.append(("WARN", "new columns", ", ".join(new_cols)))

    a, b = fp["panel"], base["panel"]
    for key in sorted(set(a) & set(b)):
        if key.startswith("date") or key == "generated":
            continue
        old, new = b[key], a[key]
        if not isinstance(old, (int, float)) or not isinstance(new, (int, float)):
            continue
        if old in (0, None):
            continue
        rel = abs(new - old) / abs(old)
        if rel >= FAIL_AT:
            out.append(("FAIL", key, f"{old:,.4g} -> {new:,.4g}  ({rel:+.0%})"))
        elif rel >= WARN_AT:
            out.append(("WARN", key, f"{old:,.4g} -> {new:,.4g}  ({rel:+.0%})"))

    # Time must move forwards. A rebuild that loses days has dropped data.
    if dt.date.fromisoformat(a["date_max"]) < dt.date.fromisoformat(b["date_max"]):
        out.append(("FAIL", "panel went backwards",
                    f"{b['date_max']} -> {a['date_max']}"))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--accept", action="store_true",
                    help="write the current build as the baseline")
    ap.add_argument("--strict", action="store_true",
                    help="exit non-zero on WARN as well as FAIL")
    ap.add_argument("--baseline", type=str, default=None)
    args = ap.parse_args()

    path = Path(args.baseline) if args.baseline else BASELINE
    fp = fingerprint()
    base = json.loads(path.read_text()) if path.exists() else None

    print(f"\n{'=' * 92}\nDATA CHECK   panel {fp['panel']['date_min']} -> "
          f"{fp['panel']['date_max']}   {fp['panel']['rows']:,} rows   "
          f"{fp['panel']['skus']:,} SKUs\n{'=' * 92}")
    if base:
        print(f"  baseline accepted {base.get('generated', '?')}")
    findings = invariants(fp) + compare(fp, base)

    fails = [f for f in findings if f[0] == "FAIL"]
    warns = [f for f in findings if f[0] == "WARN"]
    for level, what, detail in findings:
        print(f"  {level:<5} {what:<32} {detail}")
    if not findings:
        print("  PASS  everything inside tolerance")

    lines = ["# Data check\n",
             f"*Generated by `dev/datacheck.py` at {fp['generated']}.*\n",
             f"Panel: **{fp['panel']['rows']:,} rows**, {fp['panel']['skus']:,} SKUs, "
             f"{fp['panel']['date_min']} → {fp['panel']['date_max']}.\n"]
    if base:
        lines.append(f"Compared against the baseline accepted "
                     f"**{base.get('generated', '?')}**.\n")
    else:
        lines.append("**No baseline has been accepted.** Nothing has been checked; run "
                     "`--accept` on a build you trust.\n")
    if findings:
        lines += ["| level | what | detail |", "|---|---|---|"]
        lines += [f"| `{lv}` | {w} | {d} |" for lv, w, d in findings]
    else:
        lines.append("Everything inside tolerance.\n")
    lines += [
        "\n## How to read this\n",
        f"- **Tolerances are wide on purpose.** A quantile moving {WARN_AT:.0%} between "
        f"builds is an ordinary seasonal week on this panel; {FAIL_AT:.0%} is not "
        "something a week produces.\n",
        "- **A deliberate change looks exactly like a defect from here.** A new "
        "marketplace or a genuine catalogue expansion will FAIL, and should -- the job "
        "is to make the change noticed, not to guess whether it was wanted. Accept it "
        "with `--accept` once you have looked.\n",
        "- **This is about the input, not the model.** Everything else in `dev/` asks "
        "whether a forecast is good. A wrong input produces a plausible forecast, which "
        "is the failure mode nothing else here can see.\n",
    ]
    (LAB / "DATACHECK.md").write_text("\n".join(lines))

    if args.accept:
        path.parent.mkdir(parents=True, exist_ok=True)
        if fails and not args.baseline:
            raise SystemExit(
                f"\nrefusing to accept a build with {len(fails)} FAIL(s). Fix them, or "
                f"pass --baseline explicitly if you really mean it.")
        path.write_text(json.dumps(fp, indent=2))
        print(f"\n  accepted -> {path}")

    print(f"\n{'=' * 92}")
    print(f"  {len(fails)} FAIL, {len(warns)} WARN   -> dev/DATACHECK.md")
    print(f"{'=' * 92}")
    if fails or (args.strict and warns):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

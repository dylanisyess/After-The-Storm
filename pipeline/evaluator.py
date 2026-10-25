"""The intervention evaluator's math, shared by pipeline/12_intervention.py and app.py.

evaluate() compares a beach's tests before and after a project with what the
weather alone would predict, minus the same change at beaches 5+ km away.
12_intervention.py explains the method and makes the inputs:
  ex     one row per beach test: StationCode, date, lat, lon, exceeds (0/1),
         expected (weather-only risk)
  widen  how much to widen the month-resampled range, from the placebo check
"""
import numpy as np
import pandas as pd

CONTROL_KM = 5      # control beaches are at least this far from the beach being checked
MIN_TESTS = 30      # tests needed at the beach in each period
N_BOOT = 1000


def r3(x):
    return None if pd.isna(x) else round(float(x), 3)


def km(lat1, lon1, lat2, lon2):
    """Distance in km along the earth's surface (haversine formula)."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(a))


def evaluate(ex, station, start, end=None, years=2, widen=1.0, seed=0):
    """Before = `years` before `start`; after = `years` after `end` (default: start).
    The work period between start and end is left out.
    Returns a dict of plain numbers, or {"error": ...} when there isn't enough data."""
    start = pd.Timestamp(start)
    end = pd.Timestamp(end) if end else start
    if end < start:
        return {"error": "the end date is before the start date"}
    before = (start - pd.DateOffset(years=years), start)
    after = (end, end + pd.DateOffset(years=years))
    ex = ex[((ex["date"] >= before[0]) & (ex["date"] < before[1]))
            | ((ex["date"] > after[0]) & (ex["date"] <= after[1]))].copy()
    ex["period"] = np.where(ex["date"] < start, "before", "after")
    ex["resid"] = ex["exceeds"] - ex["expected"]   # what happened minus what the weather predicts

    here = ex.groupby("StationCode")[["lat", "lon"]].median()
    if station not in here.index:
        return {"error": f"no tests at {station} in these years"}
    dist = km(*here.loc[station], here["lat"], here["lon"])
    tgt = ex[ex["StationCode"] == station]
    ctl = ex[ex["StationCode"].isin(dist[dist >= CONTROL_KM].index)]
    n = tgt["period"].value_counts()
    if n.get("before", 0) < MIN_TESTS or n.get("after", 0) < MIN_TESTS:
        return {"error": f"too few tests at this beach: {n.get('before', 0)} before and "
                         f"{n.get('after', 0)} after (need {MIN_TESTS} each)"}

    # Per calendar month and period: sum of residuals and number of tests, beach (_t) and controls (_c).
    def by_month(d):
        return d.groupby([d["date"].dt.to_period("M"), "period"])["resid"].agg(["sum", "size"])
    m = (by_month(tgt).add_suffix("_t")
         .join(by_month(ctl).add_suffix("_c"), how="outer").fillna(0))

    def change(mm):
        """(beach after - before) - (controls after - before), from monthly sums."""
        s = mm.groupby(level="period").sum()
        t = s["sum_t"] / s["size_t"]
        c = s["sum_c"] / s["size_c"]
        return (t["after"] - t["before"]) - (c["after"] - c["before"])

    # Resample whole months within each period, keeping each month's beach and control tests together.
    rng = np.random.default_rng(seed)
    arrs = {p: m.xs(p, level="period")[["sum_t", "size_t", "sum_c", "size_c"]].to_numpy() for p in ["before", "after"]}
    boot = {}
    for p, a in arrs.items():
        pick = rng.integers(0, len(a), size=(N_BOOT, len(a)))
        boot[p] = a[pick].sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        t = boot["after"][:, 0] / boot["after"][:, 1] - boot["before"][:, 0] / boot["before"][:, 1]
        c = boot["after"][:, 2] / boot["after"][:, 3] - boot["before"][:, 2] / boot["before"][:, 3]
    draws = (t - c)[~np.isnan(t - c)]

    def cell(d):
        return {"tests": int(len(d)), "unsafe_rate": r3(d["exceeds"].mean()),
                "expected_from_weather": r3(d["expected"].mean())}
    est = change(m)
    se = draws.std()                    # spread of the month-resampled estimates
    lo, hi = est - 1.96 * widen * se, est + 1.96 * widen * se
    # Monthly residuals for a chart: beach vs. controls (controls averaged over all their tests).
    monthly = (m.assign(beach=m["sum_t"] / m["size_t"].where(m["size_t"] > 0),
                        others=m["sum_c"] / m["size_c"].where(m["size_c"] > 0))
               .reset_index().rename(columns={"date": "month"}))
    # Say what changed AFTER the date, never that the project caused it (one beach,
    # one date: something else could have changed at the same time).
    verdict = "cleaner after" if hi < 0 else "dirtier after" if lo > 0 else "no clear change after"
    return {
        "station": station, "start": str(start.date()), "end": str(end.date()), "years": years,
        "before_window": [str(before[0].date()), str(before[1].date())],
        "after_window": [str(after[0].date()), str(after[1].date())],
        "before": cell(tgt[tgt["period"] == "before"]), "after": cell(tgt[tgt["period"] == "after"]),
        "control_beaches": int(ctl["StationCode"].nunique()),
        "control_change": r3(ctl.loc[ctl.period == "after", "resid"].mean() - ctl.loc[ctl.period == "before", "resid"].mean()),
        "change_beyond_weather": r3(est), "ci95": [r3(lo), r3(hi)], "se": float(se), "widen": widen,
        "verdict": verdict,
        "monthly": [{"month": str(r.month), "period": r.period, "beach": r3(r.beach), "beach_tests": int(r.size_t),
                     "others": r3(r.others)} for r in monthly.itertuples()],
    }

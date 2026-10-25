"""Step 11: beach diagnosis and the 72-hour advisory check (no model needed).

Uses beach tests from 2010 on, plus the daily rain record. For each beach:
  A. Dry vs. after-rain unsafe rates, and the "chronic" label, from the LAST 5
     YEARS only, so a beach that was fixed isn't still called chronic. Everything
     about recovery after rain uses all years, because it needs many storms.
  B. Trigger: the smallest storm that makes 1 in 4 tests unsafe in the first
     day or so after rain.
  C. Recovery: unsafe rate by days since the last day with 0.1 in or more of
     rain, compared with the beach's own normal wet-season level, and checked
     against LA County's advisory (stay out for 72 hours after 0.1 in of rain).
  D. What kind of fix the pattern points to (a clue, not proof of the source).

Counting days: rain is recorded as daily totals (local calendar days).
  day 0 = 0.1 in+ fell on the sample day itself (often overnight, before sampling)
  day k = the last day with 0.1 in+ was k days before the sample day
Because we only know the day of the rain, not the hour, a day-3 test was taken
roughly 48-96 hours after the rain stopped, so day 3 straddles the 72-hour line.
Days 0-2 are inside the advisory; days 4 and later are after it ends.

95% ranges come from a bootstrap that resamples whole storms (and, for dry
days, whole weeks), because tests taken after the same storm are not
independent of each other.
Output: app_data/diagnosis.json
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
P, OUT = ROOT / "processed", ROOT / "app_data"
WET_MM = 2.5           # 0.1 inch: a "rain day" (same cutoff as days_dry in 05)
MM_PER_IN = 25.4
NORMAL_DAYS = 8        # 8+ days since rain counts as dry weather
STORM_BINS = [0.1, 0.25, 0.5, 1.0, np.inf]                  # storm size, inches
STORM_LABELS = ["0.1-0.25 in", "0.25-0.5 in", "0.5-1 in", "1 in or more"]
# Groups of days after rain used to classify each beach (see classify()).
BINS = {"days 0-1": (0, 1), "days 2-3": (2, 3), "days 4-6": (4, 6), "days 7-10": (7, 10)}
MIN_N = 15             # a group needs 15+ tests before we draw a conclusion from it
RECENT_YEARS = 5       # dry/after-rain rates and the chronic label use the last 5 years
N_BOOT = 1000
# MODEL_ID is stamped so check_model_ids.py treats this like every other app file;
# this step itself uses only test results and rain, not the model.
MODEL_ID = json.loads((ROOT / "model" / "metrics.json").read_text())["model_id"]


def r3(x):
    return None if pd.isna(x) else round(float(x), 3)


# ---------------- Rain: days since rain and storm size, per rain cell ----------------
rain = pd.read_csv(P / "rain_daily.csv", parse_dates=["date"]).sort_values("date")
cells = []
for (la, lo), g in rain.groupby(["cell_lat", "cell_lon"]):
    s = g.set_index("date")["precip_mm"].asfreq("D")
    wet = s >= WET_MM
    # A storm = a run of back-to-back rain days. Its size is the total rain over the run.
    run_id = (wet & ~wet.shift(1, fill_value=False)).cumsum().where(wet)
    run_total = s.groupby(run_id).transform("sum").where(wet)
    last_wet = s.index.to_series().where(wet).ffill()    # includes today (day 0)
    cells.append(pd.DataFrame({
        "cell_lat": la, "cell_lon": lo, "date": s.index,
        "rain_today": s.values,
        "days_since_rain": (s.index.to_series() - last_wet).dt.days.values,
        "storm_date": last_wet.values,                     # last rain day of the storm
        "storm_in": (run_total.ffill() / MM_PER_IN).values}))  # size of that storm
rain_f = pd.concat(cells)

tt = pd.read_csv(P / "training_table.csv", parse_dates=["date"])
tt = tt[tt["water"] == "ocean"].merge(rain_f, on=["cell_lat", "cell_lon", "date"], how="left")
tt = tt.dropna(subset=["rain_today", "days_since_rain"])   # skip gaps in the rain record
d = tt["days_since_rain"]
tt["normal"] = (d >= NORMAL_DAYS) & (tt["wet_season"] == 1)  # dry weather, Nov-Apr
tt["dry"] = d >= NORMAL_DAYS                                 # dry weather, any month
tt["advisory"] = d <= 3                                      # during the rain advisory
RECENT_START = tt["date"].max() - pd.DateOffset(years=RECENT_YEARS)
recent = tt["date"] >= RECENT_START
tt["dry"] &= recent           # the dry rate and chronic label: last 5 years only
tt["advisory"] &= recent      # same years, so dry vs. after-rain compares like with like
# Cluster for the bootstrap: tests after the same storm move together; dry-day
# tests are grouped by calendar week.
tt["cluster"] = np.where(d <= 10, "storm " + tt["storm_date"].dt.strftime("%Y-%m-%d"),
                         "week " + tt["date"].dt.strftime("%G-%V"))
st = pd.read_csv(P / "stations.csv").set_index("StationCode")
print(f"{len(tt):,} beach tests at {tt['StationCode'].nunique()} beaches, "
      f"{tt['date'].min().date()} to {tt['date'].max().date()}")


def boot_rates(g, groups, seed=0):
    """Unsafe rate for each named group of tests (dict name -> True/False mask),
    plus N_BOOT bootstrap copies made by resampling whole clusters. Returns
    (rates, n, boot) where boot has one row per bootstrap copy."""
    rng = np.random.default_rng(seed)
    codes, cl = np.unique(g["cluster"], return_inverse=True)
    y = g["exceeds"].to_numpy()
    # per cluster: unsafe count and test count in each group
    unsafe = np.column_stack([np.bincount(cl, weights=y * m.to_numpy(), minlength=len(codes)) for m in groups.values()])
    n = np.column_stack([np.bincount(cl, weights=m.to_numpy(), minlength=len(codes)) for m in groups.values()])
    pick = rng.integers(0, len(codes), size=(N_BOOT, len(codes)))
    with np.errstate(invalid="ignore", divide="ignore"):
        boot = unsafe[pick].sum(1) / n[pick].sum(1)
        rates = unsafe.sum(0) / n.sum(0)
    return rates, n.sum(0).astype(int), boot


def ci(x):
    x = x[~np.isnan(x)]
    return [r3(np.percentile(x, 2.5)), r3(np.percentile(x, 97.5))] if len(x) else [None, None]


def classify(rate, n, lo, normal_rate):
    """Sort a beach by how long it stays unsafe after rain. 'Clearly above' means
    the 95% range of (group rate - normal rate) is entirely above zero.
    lo = lower end of that range for each group."""
    if any(n[b] < MIN_N for b in ["days 0-1", "days 2-3", "days 4-6"]) or n["normal"] < 2 * MIN_N:
        return "not enough tests"
    near = {b: lo[b] <= 0 and rate[b] - normal_rate < 0.05 for b in ["days 2-3", "days 4-6"]}  # within 5 points
    if lo["days 0-1"] <= 0:
        return "little change after rain"
    if lo["days 4-6"] > 0:
        return "still unsafe after 72 hours"
    if near["days 2-3"] and near["days 4-6"]:
        return "back to normal in about a day"
    if lo["days 2-3"] > 0 and near["days 4-6"]:
        return "back to normal within 72 hours"
    return "unclear (too few tests to tell)"


FIXES = {
    "chronic": "Often unsafe even in dry weather. That points to a source that flows without rain, "
               "such as dry-weather runoff from a storm drain, a creek or lagoon mouth, birds, or "
               "leaking sewer pipes. A usual first step is source-tracking tests (for example, DNA "
               "markers for human sewage); fixes that match include diverting dry-weather drain "
               "flow to the sewer and repairing sewer lines.",
    "slow": "Stays unsafe for days after rain. That points to runoff that keeps draining from a "
            "large area (a creek, river or lagoon) after the storm ends. Fixes that match include "
            "capturing or soaking in stormwater upstream, and a longer advisory at this beach.",
    "fast": "Unsafe right after rain, then clears quickly. That points to the first rush of "
            "runoff from nearby storm drains. Fixes that match include capturing or diverting "
            "that first flush near the beach. This beach recovers faster than most after rain.",
    "typical": "Unsafe after rain and back to normal within the 72-hour advisory. That points to "
               "storm runoff; fixes that match include capturing or soaking in stormwater before "
               "it reaches the beach.",
    "clean": "Rarely unsafe, in dry weather or after rain. No fix is indicated by this data.",
}
RECOVERY_TO_FIX = {"still unsafe after 72 hours": "slow", "back to normal in about a day": "fast",
                   "back to normal within 72 hours": "typical", "little change after rain": "clean"}
BIG_TIDE_FT = 6.5      # daily tide range for a "big spring tide" (same as the forecast reason)


# ---------------- Per beach ----------------
rows = []
for code, g in tt.groupby("StationCode"):
    if len(g) < 100:
        continue
    d = g["days_since_rain"]
    groups = {"normal": g["normal"], "dry": g["dry"], "advisory": g["advisory"]}
    groups |= {b: d.between(a, z) for b, (a, z) in BINS.items()}
    groups |= {f"day {k}": d == k for k in range(11)}
    rates, n, boot = boot_rates(g, groups, seed=len(rows))
    # int(): plain Python numbers, so the JSON file stores counts as numbers, not text
    rate, nn = dict(zip(groups, rates)), {k: int(v) for k, v in zip(groups, n)}
    bt = dict(zip(groups, boot.T))
    excess = {b: bt[b] - bt["normal"] for b in BINS}           # rate above normal, per group
    lo = {b: np.nanpercentile(x, 2.5) if nn[b] else np.nan for b, x in excess.items()}
    recovery = classify(rate, nn, lo, rate["normal"])

    # B. Trigger: unsafe rate in days 0-1 by storm size; smallest size where 1 in 4
    # tests were unsafe, and every bigger size (with 8+ tests) was too.
    fresh = g[d <= 1]
    sizes = pd.cut(fresh["storm_in"], STORM_BINS, labels=STORM_LABELS, right=False)
    by_size = fresh.groupby(sizes, observed=False)["exceeds"].agg(["mean", "size"])
    ok = by_size[by_size["size"] >= 8]
    trigger = None
    for lab in ok.index:
        if (ok.loc[lab:, "mean"] >= 0.25).all():
            trigger = lab
            break
    if rate["dry"] >= 0.25:
        trigger = "unsafe 1 in 4 times even without rain"

    # Big tides after rain: unsafe rate on days 3-7 with a big spring tide vs. other days 3-7.
    late = g[d.between(3, 7)]
    big = late["tide_range"] >= BIG_TIDE_FT
    tide = {"big_tide_rate": r3(late.loc[big, "exceeds"].mean()), "big_tide_tests": int(big.sum()),
            "other_rate": r3(late.loc[~big, "exceeds"].mean()), "other_tests": int((~big).sum())}

    chronic = rate["dry"] >= 0.10 and nn["dry"] >= 2 * MIN_N
    fix = "chronic" if chronic else RECOVERY_TO_FIX.get(recovery)
    # Does risk peak later than the first day or so? (rate in days 2-3 clearly above days 0-1)
    later = bt["days 2-3"] - bt["days 0-1"]
    s = st.loc[code]
    rows.append({
        "station": code, "name": s["name"], "lat": s["lat"], "lon": s["lon"],
        "tests": int(len(g)), "first": str(g["date"].min().date()), "last": str(g["date"].max().date()),
        "dry_unsafe_rate": r3(rate["dry"]), "dry_tests": nn["dry"],
        "advisory_unsafe_rate": r3(rate["advisory"]), "advisory_tests": nn["advisory"],
        "normal_wet_season_rate": r3(rate["normal"]), "normal_tests": nn["normal"],
        "groups": {b: {"unsafe_rate": r3(rate[b]), "tests": nn[b], "above_normal": r3(rate[b] - rate["normal"]),
                       "above_normal_ci95": ci(excess[b])} for b in BINS},
        "by_day": [{"day": k, "unsafe_rate": r3(rate[f"day {k}"]), "tests": nn[f"day {k}"]} for k in range(11)],
        "by_storm_size": [{"storm": lab, "unsafe_rate": r3(r.mean_), "tests": int(r.size_)}
                          for lab, r in by_size.rename(columns={"mean": "mean_", "size": "size_"}).iterrows()],
        "trigger": trigger,
        "recovery": recovery,
        "after_rain_tides": tide,
        "later_peak": bool(nn["days 2-3"] >= MIN_N and nn["days 0-1"] >= MIN_N and np.nanpercentile(later, 2.5) > 0),
        "chronic": bool(chronic),
        "fix_category": fix,
        "fix_text": FIXES.get(fix),
        # chronic beaches that also stay unsafe after storms get both explanations
        "also_after_rain": FIXES["slow"] if chronic and recovery == "still unsafe after 72 hours" else None,
    })
diag = pd.DataFrame(rows)

# ---------------- Region: recovery curve, by storm size ----------------
ws = tt[tt["wet_season"] == 1]
normal_all = ws.loc[ws["normal"], "exceeds"].mean()
curve = (ws[ws["days_since_rain"] <= 10].groupby("days_since_rain")["exceeds"].agg(["mean", "size"]))
sizes = pd.cut(ws["storm_in"], [0.1, 0.5, 1.0, np.inf], labels=["0.1-0.5 in", "0.5-1 in", "1 in or more"], right=False)
by_storm = (ws[ws["days_since_rain"] <= 10].groupby([sizes, "days_since_rain"], observed=True)["exceeds"]
            .agg(["mean", "size"]).reset_index())
region = {
    "normal_wet_season_rate": r3(normal_all),
    "curve": [{"day": int(k), "unsafe_rate": r3(r["mean"]), "tests": int(r["size"])} for k, r in curve.iterrows()],
    "by_storm_size": {lab: [{"day": int(r.days_since_rain), "unsafe_rate": r3(r["mean"]), "tests": int(r["size"])}
                            for _, r in grp.iterrows()]
                      for lab, grp in by_storm.groupby("storm_in", observed=True)},
}

# Recovery groups pooled (for the app's headline chart). Chronic beaches are left out
# of "fast" and "typical" (back to normal = back to bad there) but kept in "slow".
rec = diag.set_index("station")
members = {
    "fast": rec.index[(rec["recovery"] == "back to normal in about a day") & ~rec["chronic"]],
    "typical": rec.index[(rec["recovery"] == "back to normal within 72 hours") & ~rec["chronic"]],
    "slow": rec.index[rec["recovery"] == "still unsafe after 72 hours"],
}
region["recovery_groups"] = {}
for grp, codes in members.items():
    w = ws[ws["StationCode"].isin(codes)]
    curves = {}
    for size, sub in [("all storms", w), ("1 in or more", w[w["storm_in"] >= 1])]:
        c = sub[sub["days_since_rain"] <= 10].groupby("days_since_rain")["exceeds"].agg(["mean", "size"])
        curves[size] = [{"day": int(k), "unsafe_rate": r3(r["mean"]), "tests": int(r["size"])} for k, r in c.iterrows()]
    region["recovery_groups"][grp] = {"beaches": len(codes), "stations": list(codes),
                                      "normal_wet_season_rate": r3(w.loc[w["normal"], "exceeds"].mean()),
                                      "curves": curves}

late = ws[ws["days_since_rain"].between(3, 7)]
region["after_rain_tides"] = late.groupby(late["tide_range"] >= BIG_TIDE_FT)["exceeds"].agg(["mean", "size"]) \
    .rename(index={True: "big_tide", False: "other"}).round(3).to_dict("index")
# The advisory check is about rain, so beaches that are often unsafe even in dry
# weather (chronic) are counted separately: "back to normal" there means back to bad.
counts = diag.loc[~diag["chronic"], "recovery"].value_counts().to_dict()
counts["chronic (counted separately)"] = int(diag["chronic"].sum())
out = {
    "model_id": MODEL_ID,
    "period": f"{tt['date'].min().date()} to {tt['date'].max().date()}",
    "recent_period": f"{RECENT_START.date()} to {tt['date'].max().date()}",
    "definitions": {
        "day": "days since the last day with 0.1 in or more of rain (day 0 = rain on the sample day)",
        "dry": f"{NORMAL_DAYS}+ days since 0.1 in of rain, any month, last {RECENT_YEARS} years",
        "normal": f"{NORMAL_DAYS}+ days since 0.1 in of rain, November-April (the comparison for recovery)",
        "advisory": f"days 0-3 (LA County advises staying out for 72 hours after 0.1 in of rain), last {RECENT_YEARS} years",
        "clearly_above": "the 95% range of (rate - normal rate) is entirely above zero; bootstrap over whole storms",
        "trigger": "smallest storm size where 1 in 4 tests on days 0-1 were unsafe (8+ tests), and every bigger size too",
        "chronic": f"unsafe on 10%+ of dry-weather tests in the last {RECENT_YEARS} years (30+ dry tests)",
        "big_tide": f"daily tide range of {BIG_TIDE_FT} ft or more (Boehm & Weisberg 2005)",
        "near_normal": "within 5 points of normal and not clearly above it",
        "min_tests": MIN_N,
    },
    "recovery_counts": counts,
    "region": region,
    # blanks become null, not NaN (NaN isn't valid JSON, and 08/09 copy fix_category on)
    "stations": diag.astype(object).where(diag.notna(), None).to_dict("records"),
}
(OUT / "diagnosis.json").write_text(json.dumps(out, indent=1, default=str))

# ---------------- Print a summary ----------------
pd.set_option("display.width", 200)
print(f"\nRegion, wet season: normal (8+ dry days) {normal_all:.1%}")
print(curve.assign(mean=curve["mean"].round(3)).T.to_string())
print("\nBy storm size, unsafe rate by day:")
print(by_storm.pivot(index="storm_in", columns="days_since_rain", values="mean").round(2).to_string())
print("\nAfter rain (days 3-7, wet season), big tide vs other:", region["after_rain_tides"])
print("\nRecovery groups:", counts)
print("Fix categories:", diag["fix_category"].value_counts(dropna=False).to_dict())
print("Beaches with a later peak (days 2-3 clearly above days 0-1):",
      diag.loc[diag["later_peak"], "name"].tolist())

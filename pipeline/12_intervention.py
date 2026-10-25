"""Step 12: intervention evaluator. Was a beach cleaner (or dirtier) after a
given date, beyond what the weather explains?

  python3 pipeline/12_intervention.py SMB-MC-2 2012-06-01 2013-03-31   (station, work start, work end)
  python3 pipeline/12_intervention.py SMB-3-3 2018-01-01 2018-12-31 --years 3
  python3 pipeline/12_intervention.py --demo    (known cases + placebo check -> app_data/)

Method (same idea as the post-fire analysis in 07; the comparison itself is in
pipeline/evaluator.py, which app.py also uses):
  1. Expected risk from WEATHER ONLY: a model that knows the beach, rain (including
     the sample day, since this looks back in time), days since rain, tides and
     season, but NOT recent test results or the date. The forecast model can't be
     used here: its "last test" and past_rate inputs follow the beach's own recent
     results, so after a real improvement its expected risk drops too and the
     improvement disappears from the residuals.
     Each year's expected risk comes from a model trained on the other years, so
     no test is scored by a model that saw it.
  2. Residual = what happened (1 unsafe / 0 safe) minus expected risk.
  3. Change = (beach's residual after - before) minus (the same change at control
     beaches 5+ km away), which removes region-wide swings like drought years.
  4. 95% range: resample whole calendar months (1,000 times), keeping each month's
     target and control tests together, then WIDEN it. On random beaches and
     dates with no known project, the plain range called about a quarter of them
     a real change instead of about 5%: beaches drift from year to year for
     reasons we don't measure (sand, berms, birds, sampling). So we widen every
     range by the factor that makes only 5% of random dates look like a change
     (calibrate()), and check that on a second, separate set of random dates.
The years during the work itself (start to end) are left out. Results say
"cleaner after" or "no clear change after", never "because of": one beach and
one date can't rule out something else changing at the same time.
Outputs: processed/weather_expected.csv and evaluator_calibration.json (cached);
         with --demo, app_data/interventions.json and app_data/evaluator_data.json
         (every beach test with its weather-only risk, so the app can run evaluate())
"""
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
import evaluator as ev           # pipeline/evaluator.py: the comparison itself
from evaluator import r3, km, N_BOOT

ROOT = Path(__file__).resolve().parent.parent
P, OUT = ROOT / "processed", ROOT / "app_data"
CACHE = P / "weather_expected.csv"
CALIB = P / "evaluator_calibration.json"
FEATURES = ["station_id", "rain_today", "rain_1d", "rain_3d", "rain_7d", "rain_30d", "days_dry",
            "tide_high", "tide_low", "tide_range", "wet_season"]
# Same settings as the forecast model (06), so every rule rests on 100+ tests.
PARAMS = dict(learning_rate=0.05, max_iter=400, max_leaf_nodes=31, min_samples_leaf=100,
              l2_regularization=1.0, random_state=0)
MODEL_ID = json.loads((ROOT / "model" / "metrics.json").read_text())["model_id"]
DAY0 = pd.Timestamp("2010-01-01")   # evaluator_data.json stores dates as days since this


def build_expected():
    """Weather-only expected risk for every beach test, leaving out one year at a time."""
    tt = pd.read_csv(P / "training_table.csv", parse_dates=["date"])
    tt = tt[tt["water"] == "ocean"]
    rain = pd.read_csv(P / "rain_daily.csv", parse_dates=["date"]).rename(columns={"precip_mm": "rain_today"})
    tt = tt.merge(rain, on=["cell_lat", "cell_lon", "date"], how="left")
    tt = tt.dropna(subset=["rain_today", "rain_30d"]).copy()   # full rain record only
    tt["station_id"] = tt["StationCode"].astype("category").cat.codes  # beach as a category
    tt["expected"] = np.nan
    for year in sorted(tt["date"].dt.year.unique()):
        held = tt["date"].dt.year == year
        m = HistGradientBoostingClassifier(categorical_features=[0], **PARAMS)
        m.fit(tt.loc[~held, FEATURES], tt.loc[~held, "exceeds"])
        tt.loc[held, "expected"] = m.predict_proba(tt.loc[held, FEATURES])[:, 1]
        print(f"  expected risk for {year}: {held.sum():,} tests")
    out = tt[["StationCode", "date", "lat", "lon", "exceeds", "expected", "rain_3d"]]
    out.to_csv(CACHE, index=False, float_format="%.4f")
    return out


def load_expected(rebuild=False):
    if rebuild or not CACHE.exists():
        print("Building weather-only expected risk (about a minute)...")
        return build_expected()
    return pd.read_csv(CACHE, parse_dates=["date"])


def evaluate(ex, station, start, end=None, years=2, widen=None, seed=0):
    """evaluator.evaluate() with the calibrated widening factor filled in."""
    if widen is None:
        widen = json.loads(CALIB.read_text())["widen"]
    return ev.evaluate(ex, station, start, end, years, widen, seed)


def show(r):
    if "error" in r:
        print("  ", r["error"])
        return
    b, a = r["before"], r["after"]
    print(f"  {r['station']}: before {r['before_window'][0]}..{r['before_window'][1]}, "
          f"after {r['after_window'][0]}..{r['after_window'][1]}")
    print(f"    unsafe {b['unsafe_rate']:.1%} -> {a['unsafe_rate']:.1%}  "
          f"(weather alone expects {b['expected_from_weather']:.1%} -> {a['expected_from_weather']:.1%}; "
          f"{b['tests']} / {a['tests']} tests)")
    print(f"    change beyond weather and region: {r['change_beyond_weather']:+.1%} "
          f"(95% range {r['ci95'][0]:+.1%} to {r['ci95'][1]:+.1%}) -> {r['verdict']}")


# Known projects (sources in review/causes_results.md). Dates are the work period,
# which is left out of the comparison.
# "short" finishes the app's sentence "<beach> was cleaner after ...".
DEMOS = [
    {"label": "Malibu Lagoon restoration", "beach": "Surfrider Beach", "station": "SMB-MC-2",
     "start": "2012-06-01", "end": "2013-03-31", "short": "the Malibu Lagoon restoration (2012-13)"},
    {"label": "Santa Monica Pier stormwater tank (Clean Beaches Project, completed 2018)",
     "beach": "Santa Monica Pier", "station": "SMB-3-3", "start": "2018-01-01", "end": "2018-12-31",
     "short": "the 2018 stormwater tank"},
    {"label": "Santa Monica Pier new bird netting (August 2023)", "beach": "Santa Monica Pier",
     "station": "SMB-3-3", "start": "2023-08-01", "end": "2023-08-31", "short": "the new bird netting (August 2023)"},
]


def placebo(ex, n=200, seed=1, widen=None):
    """Run the evaluator at random beaches and dates where we know of no project.
    If the 95% ranges are honest, about 5% should exclude zero (a few more if
    real, unknown changes happened). Also returns each run's estimate / spread."""
    rng = np.random.default_rng(seed)
    codes = ex["StationCode"].value_counts()
    codes = codes[codes >= 300].index.to_numpy()
    days = pd.date_range("2012-06-01", "2024-06-01", freq="D")
    hits, done, z = 0, 0, []
    while done < n:
        r = evaluate(ex, rng.choice(codes), rng.choice(days), widen=widen, seed=done)
        if "error" in r:
            continue
        done += 1
        hits += r["ci95"][1] < 0 or r["ci95"][0] > 0
        z.append(r["change_beyond_weather"] / r["se"])
    return {"runs": n, "share_excluding_zero": r3(hits / n)}, np.array(z)


def calibrate(ex):
    """Find the widening factor on one set of random dates (seed 1), then check it
    on a different set (seed 2). Saved to processed/evaluator_calibration.json."""
    raw, z = placebo(ex, n=300, seed=1, widen=1.0)
    widen = round(float(np.percentile(np.abs(z), 95) / 1.96), 2)
    check, _ = placebo(ex, n=300, seed=2, widen=widen)
    out = {"widen": widen, "plain_range_share_excluding_zero": raw["share_excluding_zero"],
           "check_share_excluding_zero": check["share_excluding_zero"], "runs": 300}
    CALIB.write_text(json.dumps(out, indent=1))
    print(f"Calibration: plain ranges called {raw['share_excluding_zero']:.0%} of random dates a change; "
          f"widened x{widen}; on a fresh set: {check['share_excluding_zero']:.0%}")
    return out


def post_fire_check(ex):
    """07's Palisades comparison again, with weather-only residuals instead of the
    forecast model's (2023 on, burn-adjacent <= 1 km vs. far >= 10 km)."""
    pf = json.loads((OUT / "post_fire.json").read_text())
    burn = [s["station"] for s in pf["burn_stations"]]
    gj = json.loads((P / "burn_areas_2025.geojson").read_text())
    pal = next(f for f in gj["features"] if f["properties"]["FIRE_NAME"].upper() == "PALISADES")["geometry"]
    rings = pal["coordinates"] if pal["type"] == "Polygon" else [r for poly in pal["coordinates"] for r in poly]
    v = np.vstack([np.array(r)[:, :2] for r in rings])
    d = ex[ex["date"] >= "2023-01-01"].copy()
    here = d.groupby("StationCode")[["lat", "lon"]].median()
    dist = pd.Series({c: km(r.lat, r.lon, v[:, 1], v[:, 0]).min() for c, r in here.iterrows()})
    d = d[d["StationCode"].isin(burn) | d["StationCode"].isin(dist[dist >= 10].index)]
    d["resid"] = d["exceeds"] - d["expected"]
    d["group"] = np.where(d["StationCode"].isin(burn), "burn", "control")
    d["period"] = np.where(d["date"] >= "2025-01-07", "after", "before")
    d["wet"] = d["rain_3d"] >= 2.54

    def did(x):
        m = x.groupby(["group", "period"])["resid"].mean()
        return (m["burn", "after"] - m["burn", "before"]) - (m["control", "after"] - m["control", "before"])

    def boot(x, seed=0):  # resample whole stations, as in 07
        rng = np.random.default_rng(seed)
        by = {k: g for k, g in x.groupby("StationCode")}
        b = list(x.loc[x.group == "burn", "StationCode"].unique())
        c = list(x.loc[x.group == "control", "StationCode"].unique())
        out = [did(pd.concat([by[s] for s in list(rng.choice(b, len(b))) + list(rng.choice(c, len(c)))]))
               for _ in range(N_BOOT)]
        return [r3(np.percentile(out, 2.5)), r3(np.percentile(out, 97.5))]
    return {k: {"estimate": r3(did(x)), "ci95": boot(x), "forecast_model_result": pf["result"][k]}
            for k, x in [("overall", d), ("wet_days", d[d.wet]), ("dry_days", d[~d.wet])]}


def export_for_app(ex, calib):
    """Every beach test with its weather-only risk, in a compact form for app.py:
    parallel lists (station number, day number, unsafe 0/1, risk in thousandths)."""
    codes = sorted(ex["StationCode"].unique())
    idx = {c: i for i, c in enumerate(codes)}
    where = ex.groupby("StationCode")[["lat", "lon"]].median().round(5)
    data = {
        "model_id": MODEL_ID, "day0": str(DAY0.date()), "widen": calib["widen"], "calibration": calib,
        "stations": [{"station": c, "lat": float(where.at[c, "lat"]), "lon": float(where.at[c, "lon"])} for c in codes],
        "s": [idx[c] for c in ex["StationCode"]],
        "d": ((ex["date"] - DAY0).dt.days).astype(int).tolist(),
        "y": ex["exceeds"].astype(int).tolist(),
        "e": (ex["expected"] * 1000).round().astype(int).tolist(),
    }
    (OUT / "evaluator_data.json").write_text(json.dumps(data, separators=(",", ":")))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("station", nargs="?")
    ap.add_argument("start", nargs="?")
    ap.add_argument("end", nargs="?")
    ap.add_argument("--years", type=int, default=2)
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--rebuild", action="store_true", help="rebuild the weather-only expected risk")
    ap.add_argument("--calibrate", action="store_true", help="redo the placebo calibration only")
    a = ap.parse_args()
    ex = load_expected(a.rebuild)
    if a.rebuild or a.calibrate or not CALIB.exists():   # the widening factor depends on the expected risk
        calibrate(ex)
    if not a.demo:
        show(evaluate(ex, a.station, a.start, a.end, a.years))
    else:
        results = []
        for dm in DEMOS:
            print(dm["label"])
            r = evaluate(ex, dm["station"], dm["start"], dm["end"])
            show(r)
            results.append({**dm, **r})
        pl = json.loads(CALIB.read_text())
        print(f"\nPlacebo check: plain ranges {pl['plain_range_share_excluding_zero']:.0%} of random dates "
              f"looked like a change; widened x{pl['widen']}: {pl['check_share_excluding_zero']:.0%} (target 5%)")
        pfc = post_fire_check(ex)
        print("\nPost-fire, weather-only residuals vs. forecast-model residuals (07):")
        for k, v in pfc.items():
            print(f"  {k}: {v['estimate']:+.3f} {v['ci95']}   (07: {v['forecast_model_result']['estimate']:+.3f} "
                  f"{v['forecast_model_result']['ci95']})")
        (OUT / "interventions.json").write_text(json.dumps(
            {"model_id": MODEL_ID, "method": __doc__.split("Method")[1].split("Outputs")[0].strip(),
             "demos": results, "placebo": pl, "post_fire_weather_only": pfc}, indent=1, default=str))
        export_for_app(ex, pl)

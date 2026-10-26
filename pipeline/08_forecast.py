"""Step 8: beach risk for today and the next 3 days -> app_data/forecast.json

  python pipeline/08_forecast.py                    live: forecast rain + tides (needs internet)
  python pipeline/08_forecast.py --as-of 2023-01-09 replay a past day from saved data

Live mode pulls rain for the last 92 days and the next 4 from the Open-Meteo
Forecast API, and tide predictions from NOAA. The forecast models see less rain
than ERA5 (what the model was trained on), so live rain is multiplied by each
weather cell's scale factor from 04_fetch_weather.py. Replay mode uses the rain that
actually fell (a "perfect forecast") and also records what the samples showed,
so you can check the app against reality or demo a past storm.

Features are built exactly as in training (05_build_table.py). The last test
result counts only if it would have been back from the lab by the issue date.

"What's driving this risk": for each day we also ask the model what the risk would be
if one group of inputs (rain, tides, the latest test, season, track record) were at
this beach's typical value instead, with everything else unchanged. Typical = the
beach's median over its tests from the training years (2010-2022). The difference, in
percentage points, is how much that group is pushing today's risk up or down.

The app (app.py) reruns this step by itself when forecast.json is older than today.
"""
import argparse, json, os, sys, urllib.request
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
import warnings
import joblib
import numpy as np
import pandas as pd
import sklearn
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

ROOT = Path(__file__).resolve().parent.parent
P, OUT = ROOT / "processed", ROOT / "app_data"
OUT.mkdir(exist_ok=True)
LAB_LAG_DAYS, DAYS_AHEAD, WET_MM = 2, 3, 2.5
LA = ZoneInfo("America/Los_Angeles")  # "today" means today in LA, even on a server in another time zone

ap = argparse.ArgumentParser()
ap.add_argument("--as-of", help="replay this past date (YYYY-MM-DD) from saved data")
args = ap.parse_args()
live = args.as_of is None
d0 = pd.Timestamp(datetime.now(LA).date() if live else args.as_of)
days = [d0 + pd.Timedelta(days=k) for k in range(DAYS_AHEAD + 1)]

# ---- Model ----
# Never retrain here: a retrain must also redo 07-10 so every app file uses the same
# model. If the saved model doesn't fit this computer, stop and say what to do.
bundle_path = ROOT / "model" / "beach_model.joblib"
try:
    bundle = joblib.load(bundle_path)
except Exception:  # missing, or saved by a scikit-learn that can't read it
    bundle = {}
if bundle.get("sklearn_version") != sklearn.__version__ or "model_id" not in bundle:
    sys.exit(f"The saved model was made with scikit-learn {bundle.get('sklearn_version', '?')}, but this "
             f"computer has {sklearn.__version__} (or the model is missing/old). Either install the matching "
             "version (pip install -r requirements.txt) or rebuild everything: bash pipeline/rebuild.sh")
model, FEATURES, thr = bundle["model"], bundle["features"], bundle["threshold"]

def get_json(url):
    req = urllib.request.Request(url, headers={"User-Agent": "water-risk-student-app"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)

# ---- Stations active in the past year ----
st = pd.read_csv(P / "stations.csv")
st = st[st["water"] == "ocean"]
lab = pd.read_csv(P / "labels.csv", parse_dates=["SampleDate"])
lab = lab[lab["StationCode"].isin(st["StationCode"])]
known = lab[lab["SampleDate"] <= d0 - pd.Timedelta(days=LAB_LAG_DAYS)]  # same lab lag as training
last = known.sort_values("SampleDate").groupby("StationCode").tail(1).set_index("StationCode")
past = known.groupby("StationCode")["exceeds"].agg(["count", "sum"])  # results known so far, how many unsafe
active = last.index[last["SampleDate"] >= d0 - pd.Timedelta(days=365)]
st = st[st["StationCode"].isin(active)].set_index("StationCode")

# ---- Daily rain per weather cell ----
rain = pd.read_csv(P / "rain_daily.csv", parse_dates=["date"])
cells = sorted(set(zip(st["cell_lat"], st["cell_lon"])))
series = {}
for la, lo in cells:
    s = rain[(rain.cell_lat == la) & (rain.cell_lon == lo)].set_index("date")["precip_mm"]
    series[(la, lo)] = s[s.index < d0] if not live else s
source = "observed rain (replay)"
if live:
    url = ("https://api.open-meteo.com/v1/forecast?latitude=" + ",".join(str(c[0]) for c in cells)
           + "&longitude=" + ",".join(str(c[1]) for c in cells)
           + f"&daily=precipitation_sum&past_days=92&forecast_days={DAYS_AHEAD + 1}"
           + "&timezone=America%2FLos_Angeles")
    resp = get_json(url)
    resp = resp if isinstance(resp, list) else [resp]
    scale = json.loads((P / "rain_scale.json").read_text())
    for c, r in zip(cells, resp):
        recent = pd.Series(r["daily"]["precipitation_sum"], index=pd.to_datetime(r["daily"]["time"]),
                           dtype=float) * scale[f"{c[0]},{c[1]}"]
        old = series[c][series[c].index < recent.index.min()]
        series[c] = pd.concat([old, recent])
    source = "Open-Meteo forecast, scaled to match ERA5"
else:
    for c in cells:  # replay: use what actually fell
        s = rain[(rain.cell_lat == c[0]) & (rain.cell_lon == c[1])].set_index("date")["precip_mm"]
        series[c] = s[s.index <= days[-1]]

def rain_feats(s, d):
    """Rain features for day d, built like 05: only days BEFORE d count."""
    # The exact 30 calendar days before d. A day missing from the data becomes blank,
    # including days past the end of the data, so a window can't silently slide back in time.
    month = s.reindex(pd.date_range(d - pd.Timedelta(days=30), d - pd.Timedelta(days=1)))
    # skipna=False: a window with a missing day gives a blank, never a guessed 0
    sums = {f"rain_{n}d": month.iloc[-n:].sum(skipna=False) for n in (1, 3, 7, 30)}
    wet = s[(s.index < d) & (s >= WET_MM)]
    # no known wet day at all: unknown (blank), as in 05, not "a year of dry weather"
    sums["days_dry"] = min((d - wet.index.max()).days, 365) if len(wet) else np.nan
    return sums

# ---- Tides ----
if live:
    rows = []
    for ts in st["tide_station"].astype(str).unique():
        u = ("https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"
             f"?begin_date={days[0]:%Y%m%d}&end_date={days[-1]:%Y%m%d}&station={ts}"
             "&product=predictions&datum=MLLW&time_zone=lst_ldt&interval=hilo&units=english"
             "&application=water_risk_student_app&format=json")
        rows += [(int(ts), p["t"], float(p["v"])) for p in get_json(u)["predictions"]]
    tides = pd.DataFrame(rows, columns=["tide_station", "time", "height_ft"])
else:
    tides = pd.read_csv(P / "tides_hilo.csv")
tides["date"] = pd.to_datetime(tides["time"]).dt.normalize()
tide_day = tides.groupby(["tide_station", "date"])["height_ft"].agg(["max", "min"])

# Each beach's pattern from 11_beach_diagnosis.py: chronic | slow | typical | fast | clean (or None).
# rebuild.sh runs 11 before this step.
diag = {d["station"]: d["fix_category"] for d in json.loads((OUT / "diagnosis.json").read_text())["stations"]}
inch = lambda mm: f"{mm / 25.4:.2f} in"

# ---- Each beach's typical inputs (for "What's driving this risk") ----
# The inputs in groups a reader can picture. Rain inputs move together (3 days of rain is
# also in the 7-day total), so they are swapped together; same for tides and the last test.
GROUPS = {"rain": ["rain_1d", "rain_3d", "rain_7d", "rain_30d", "days_dry"],
          "tides": ["tide_high", "tide_low", "tide_range"],
          "last test": ["last_exceeds", "last_log_value", "days_since_last"],
          "season": ["wet_season"],
          "track record": ["past_rate", "recent_rate"]}  # lat and lon never change for a beach, so they are left out
tt = pd.read_csv(P / "training_table.csv", parse_dates=["date"])
tt = tt[(tt["water"] == "ocean") & (tt["date"] < "2023-01-01")].dropna(subset=GROUPS["rain"]).copy()
tt["past_rate"] = (tt["past_unsafe"] + bundle["prior_weight"] * bundle["global_rate"]) / (tt["past_n"] + bundle["prior_weight"])
tt["past_share"] = tt["past_unsafe"] / tt["past_n"]  # not a model input; only for the words shown
inputs = [c for g in GROUPS.values() for c in g]
typical = tt.groupby("StationCode")[inputs + ["past_share"]].median()
typical_all = tt[inputs + ["past_share"]].median()  # for a beach with no tests in 2010-2022

# ---- Predict ----
out_st, X_rows, keys = [], [], []
for code, s in st.iterrows():
    lk = last.loc[code]
    n, unsafe = past.loc[code, "count"], past.loc[code, "sum"]
    # same formula as add_past_rate() in 06_train.py
    past_rate = (unsafe + bundle["prior_weight"] * bundle["global_rate"]) / (n + bundle["prior_weight"])
    mine = known[known["StationCode"] == code]
    for d in days:
        # Past-year track record, as in 05's recent_rate(): results sampled in the 365 days before d that are
        # back from the lab (all of `known` is), blank if fewer than 10.
        year = mine[mine["SampleDate"] >= d - pd.Timedelta(days=365)]["exceeds"]
        recent_share = year.mean() if len(year) >= 10 else np.nan
        f = rain_feats(series[(s.cell_lat, s.cell_lon)], d)
        hi, lo = tide_day.loc[(s.tide_station, d)] if (s.tide_station, d) in tide_day.index else (np.nan, np.nan)
        f.update({"tide_high": hi, "tide_low": lo, "tide_range": hi - lo, "month": d.month,
                  "wet_season": int(d.month in [11, 12, 1, 2, 3, 4]),
                  "last_exceeds": lk["exceeds"], "last_log_value": np.log10(max(lk["value"], 1)),
                  "days_since_last": (d - lk["SampleDate"]).days,
                  "past_rate": past_rate, "recent_rate": recent_share, "past_share": unsafe / n, "past_n": n,
                  "lat": s.lat, "lon": s.lon})
        X_rows.append(f); keys.append((code, d))
X = pd.DataFrame(X_rows)
# The model never trained on blank rain, so stop rather than forecast from a gap.
missing = X[["rain_1d", "rain_3d", "rain_7d", "rain_30d", "days_dry"]].isna().any(axis=1)
if missing.any():
    gaps = sorted({f"{k[1]:%Y-%m-%d}" for k, m in zip(keys, missing) if m})
    sys.exit(f"Rain data is missing for {missing.sum()} station-days ({', '.join(gaps)}). For a replay, "
             "check that processed/rain_daily.csv covers the 30 days before each forecast day.")
X["risk"] = model.predict_proba(X[FEATURES])[:, 1]

# Same rows, with one group of inputs at the beach's typical values (and once with all of them).
typ = pd.DataFrame([typical.loc[k[0]] if k[0] in typical.index else typical_all for k in keys]).reset_index(drop=True)
def risk_with(cols):
    X2 = X.copy()
    X2[cols] = typ[cols].to_numpy()
    return model.predict_proba(X2[FEATURES])[:, 1]
X["typical_risk"] = risk_with(inputs)
for g, cols in GROUPS.items():
    X[f"push_{g}"] = X["risk"] - risk_with(cols)  # > 0: this group pushes the risk above a typical day

def describe(g, r, t):
    """Today's value of a group of inputs, and the beach's typical value, in words."""
    if g == "rain":
        return (f"{inch(r.rain_3d)} over the 3 days before, {inch(r.rain_7d)} over the week before",
                f"{inch(t.rain_3d)} over 3 days, {inch(t.rain_7d)} over the week")
    if g == "tides":
        return f"{r.tide_range:.1f} ft between high and low tide", f"{t.tide_range:.1f} ft"
    if g == "last test":
        word = "over" if r.last_exceeds == 1 else "under"
        return (f"{10 ** r.last_log_value:,.0f} bacteria per 100 mL ({word} the limit), {r.days_since_last:.0f} days old",
                f"{10 ** t.last_log_value:,.0f} per 100 mL, {t.days_since_last:.0f} days old")
    if g == "season":
        season = lambda w: "rainy season (Nov to Apr)" if w == 1 else "dry season (May to Oct)"
        return season(r.wet_season), season(round(t.wet_season))
    year = "" if pd.isna(r.recent_rate) else f"; {r.recent_rate:.0%} in the past year"
    typ = "" if pd.isna(t.recent_rate) else f"; {t.recent_rate:.0%} in a typical past year"
    return f"unsafe on {r.past_share:.0%} of {int(r.past_n)} past tests{year}", f"{t.past_share:.0%}{typ}"

actual = lab.set_index(["StationCode", "SampleDate"])
for code, s in st.iterrows():
    lk = last.loc[code]
    fc = []
    for i in [i for i, k in enumerate(keys) if k[0] == code]:
        r, d = X.iloc[i], keys[i][1]
        level = "high" if r.risk >= 0.5 else "caution" if r.risk >= thr else "low"
        t = typ.iloc[i]
        drivers = [{"input": g, "points": round(float(r[f"push_{g}"]) * 100, 1),
                    "now": describe(g, r, t)[0], "typical": describe(g, r, t)[1]} for g in GROUPS]
        drivers.sort(key=lambda x: -abs(x["points"]))  # biggest push (up or down) first
        item = {"date": f"{d:%Y-%m-%d}", "risk": round(float(r.risk), 3), "level": level,
                "typical_risk": round(float(r.typical_risk), 3), "drivers": drivers}
        if not live and (code, d) in actual.index:  # replay: what the water actually showed
            a = actual.loc[(code, d)]
            a = a.iloc[0] if isinstance(a, pd.DataFrame) else a
            item["actual"] = {"value": float(a["value"]), "unsafe": bool(a["exceeds"])}
        fc.append(item)
    out_st.append({"station": code, "name": s["name"], "lat": s.lat, "lon": s.lon,
                   "type": diag.get(code), "last_test": {"date": f"{lk['SampleDate']:%Y-%m-%d}",
                   "value": float(lk["value"]), "unsafe": bool(lk["exceeds"])}, "forecast": fc})

res = {"model_id": bundle["model_id"], "issued": f"{d0:%Y-%m-%d}", "mode": "live" if live else "replay",
       "made_at": datetime.now(LA).strftime("%Y-%m-%d %H:%M"),  # LA time
       "rain_source": source,
       "levels": {"low": f"risk under {thr:.2f}", "caution": f"{thr:.2f} to 0.50 (the warning threshold "
                  "tested on 2023-2026)", "high": "0.50 or more: unsafe more likely than not"},
       "standard": "enterococcus over 104 MPN/100 mL", "stations": out_st}
name = "forecast.json" if live else f"replay_{d0:%Y-%m-%d}.json"
# Write to a temporary file, then swap it in: if this step is stopped halfway, the app
# keeps reading the last complete forecast instead of half a file.
tmp = OUT / (name + ".tmp")
tmp.write_text(json.dumps(res, indent=1))
os.replace(tmp, OUT / name)
lv = pd.Series([s["forecast"][0]["level"] for s in out_st]).value_counts().to_dict()
print(f"Wrote app_data/{name}: {len(out_st)} beaches, day-1 levels {lv}")

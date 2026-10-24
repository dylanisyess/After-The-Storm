"""Step 8: beach risk for today and the next 3 days -> app_data/forecast.json

  python pipeline/08_forecast.py                    live: forecast rain + tides (needs internet)
  python pipeline/08_forecast.py --as-of 2023-01-09 replay a past day from saved data

Live mode pulls rain for the last 92 days and the next 4 from the Open-Meteo
Forecast API, and tide predictions from NOAA. Replay mode uses the rain that
actually fell (a "perfect forecast") and also records what the samples showed,
so you can check the app against reality or demo a past storm.

Features are built exactly as in training (05_build_table.py). The last test
result counts only if it would have been back from the lab by the issue date.
"""
import argparse, json, subprocess, sys, urllib.request
from datetime import date, timedelta
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

ap = argparse.ArgumentParser()
ap.add_argument("--as-of", help="replay this past date (YYYY-MM-DD) from saved data")
args = ap.parse_args()
live = args.as_of is None
d0 = pd.Timestamp(date.today() if live else args.as_of)
days = [d0 + pd.Timedelta(days=k) for k in range(DAYS_AHEAD + 1)]

# ---- Model (retrain once if this computer has a different scikit-learn) ----
bundle_path = ROOT / "model" / "beach_model.joblib"
try:
    bundle = joblib.load(bundle_path)
except Exception:  # missing, or saved by a different scikit-learn that can't read it
    bundle = {}
if bundle.get("sklearn_version") != sklearn.__version__:
    print("Training the model for this computer's scikit-learn version (about 15 s)...")
    subprocess.run([sys.executable, str(ROOT / "pipeline" / "06_train.py")], check=True,
                   stdout=subprocess.DEVNULL)
    bundle = joblib.load(bundle_path)
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
known_cut = d0 - pd.Timedelta(days=LAB_LAG_DAYS) if not live else d0 - pd.Timedelta(days=1)
known = lab[lab["SampleDate"] <= known_cut]
last = known.sort_values("SampleDate").groupby("StationCode").tail(1).set_index("StationCode")
active = last.index[last["SampleDate"] >= d0 - pd.Timedelta(days=365)]
st = st[st["StationCode"].isin(active)].set_index("StationCode")

# ---- Daily rain per weather cell ----
rain = pd.read_csv(P / "rain_daily.csv", parse_dates=["date"])
rain["precip_mm"] = rain["precip_mm"].fillna(0)
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
    for c, r in zip(cells, resp):
        recent = pd.Series(r["daily"]["precipitation_sum"], index=pd.to_datetime(r["daily"]["time"]),
                           dtype=float).fillna(0)
        old = series[c][series[c].index < recent.index.min()]
        series[c] = pd.concat([old, recent])
    source = "Open-Meteo forecast"
else:
    for c in cells:  # replay: use what actually fell
        s = rain[(rain.cell_lat == c[0]) & (rain.cell_lon == c[1])].set_index("date")["precip_mm"]
        series[c] = s[s.index <= days[-1]]

def rain_feats(s, d):
    s = s.asfreq("D", fill_value=0)
    before = s[s.index < d]
    wet = before[before >= WET_MM]
    return {"rain_1d": before.iloc[-1:].sum(), "rain_3d": before.iloc[-3:].sum(),
            "rain_7d": before.iloc[-7:].sum(), "rain_30d": before.iloc[-30:].sum(),
            "days_dry": min((d - wet.index.max()).days, 365) if len(wet) else 365}

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

# ---- Sewage spills reaching water within 10 km in the prior week ----
sp = pd.read_csv(P / "spills_la.csv", parse_dates=["start"])
sp = sp[sp["surface_gal"] > 0]
def km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(a))

hot = {h["station"]: h["type"] for h in json.loads((OUT / "hotspots.json").read_text())["stations"]} \
    if (OUT / "hotspots.json").exists() else {}
inch = lambda mm: f"{mm / 25.4:.2f} in"

# ---- Predict ----
out_st, X_rows, keys = [], [], []
for code, s in st.iterrows():
    lk = last.loc[code]
    for d in days:
        f = rain_feats(series[(s.cell_lat, s.cell_lon)], d)
        hi, lo = tide_day.loc[(s.tide_station, d)] if (s.tide_station, d) in tide_day.index else (np.nan, np.nan)
        w = sp[(sp["start"] < d) & (sp["start"] >= d - pd.Timedelta(days=7))]
        w = w[km(s.lat, s.lon, w["lat"].values, w["lon"].values) <= 10] if len(w) else w
        f.update({"tide_high": hi, "tide_low": lo, "tide_range": hi - lo, "month": d.month,
                  "wet_season": int(d.month in [11, 12, 1, 2, 3, 4]),
                  "last_exceeds": lk["exceeds"], "last_log_value": np.log10(max(lk["value"], 1)),
                  "days_since_last": (d - lk["SampleDate"]).days,
                  "spill_log_gal_7d": np.log10(w["surface_gal"].sum() + 1), "spill_count_7d": len(w),
                  "station_rate": bundle["station_rate"].get(code, bundle["global_rate"]),
                  "lat": s.lat, "lon": s.lon})
        X_rows.append(f); keys.append((code, d))
X = pd.DataFrame(X_rows)
X["risk"] = model.predict_proba(X[FEATURES])[:, 1]

actual = lab.set_index(["StationCode", "SampleDate"])
for code, s in st.iterrows():
    lk = last.loc[code]
    fc = []
    for i in [i for i, k in enumerate(keys) if k[0] == code]:
        r, d = X.iloc[i], keys[i][1]
        reasons = []
        if r.rain_3d >= 2.54:
            reasons.append(f"{inch(r.rain_3d)} of rain in the 3 days before")
        elif r.rain_7d >= 12.7:
            reasons.append(f"{inch(r.rain_7d)} of rain in the past week")
        if r.last_exceeds == 1:
            reasons.append(f"last test ({lk['SampleDate']:%b %d}) was over the limit")
        if hot.get(code) == "chronic":
            reasons.append("this beach is often unsafe even in dry weather")
        elif r.station_rate >= 0.15:
            reasons.append(f"history: unsafe on {r.station_rate:.0%} of past tests")
        if r.spill_count_7d > 0:
            reasons.append("a sewage spill reached water within 10 km this week")
        if not reasons:
            reasons.append("dry weather and a clean recent record")
        level = "high" if r.risk >= 0.5 else "caution" if r.risk >= thr else "low"
        item = {"date": f"{d:%Y-%m-%d}", "risk": round(float(r.risk), 3), "level": level, "reasons": reasons}
        if not live and (code, d) in actual.index:  # replay: what the water actually showed
            a = actual.loc[(code, d)]
            a = a.iloc[0] if isinstance(a, pd.DataFrame) else a
            item["actual"] = {"value": float(a["value"]), "unsafe": bool(a["exceeds"])}
        fc.append(item)
    out_st.append({"station": code, "name": s["name"], "lat": s.lat, "lon": s.lon,
                   "type": hot.get(code), "last_test": {"date": f"{lk['SampleDate']:%Y-%m-%d}",
                   "value": float(lk["value"]), "unsafe": bool(lk["exceeds"])}, "forecast": fc})

res = {"issued": f"{d0:%Y-%m-%d}", "mode": "live" if live else "replay", "rain_source": source,
       "levels": {"low": f"risk under {thr:.2f}", "caution": f"{thr:.2f} to 0.50 (the warning threshold "
                  "tested on 2023-2026)", "high": "0.50 or more: unsafe more likely than not"},
       "standard": "enterococcus over 104 MPN/100 mL", "stations": out_st}
name = "forecast.json" if live else f"replay_{d0:%Y-%m-%d}.json"
(OUT / name).write_text(json.dumps(res, indent=1))
lv = pd.Series([s["forecast"][0]["level"] for s in out_st]).value_counts().to_dict()
print(f"Wrote app_data/{name}: {len(out_st)} beaches, day-1 levels {lv}")

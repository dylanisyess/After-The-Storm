"""Step 7: the three "why" analyses behind the app's Causes view.

A. Hotspots: is each beach chronically dirty, or only after rain?
B. Sewage spills: were beaches near big spills dirtier than the model expected?
C. Post-fire: did beaches below the 2025 Palisades burn area get dirtier than
   the model expected, compared with beaches far away?

B and C use the model's predictions for 2023 onward (it was trained only on
2010-2022, so it never saw these spills or the fire). "Residual" = what
happened (1 unsafe / 0 safe) minus the predicted risk. A positive residual means
dirtier than rain, tide, season and station history explain.
Outputs: app_data/hotspots.json, app_data/spills.json, app_data/post_fire.json
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
P, OUT = ROOT / "processed", ROOT / "app_data"
OUT.mkdir(exist_ok=True)
FIRE_DATE = pd.Timestamp("2025-01-07")
RAIN_MM = 2.54  # 0.1 inch, the LA County rain-advisory trigger

def km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(a))

def r3(x):
    return None if pd.isna(x) else round(float(x), 3)

st = pd.read_csv(P / "stations.csv")
beach = st[st["water"] == "ocean"].set_index("StationCode")
tt = pd.read_csv(P / "training_table.csv", parse_dates=["date"])
tt = tt[tt["water"] == "ocean"]
pred = pd.read_csv(ROOT / "model" / "test_predictions.csv", parse_dates=["date"])
pred["resid"] = pred["exceeds"] - pred["risk"]

# ---------------- A. Hotspots (last 5 years) ----------------
recent = tt[tt["date"] >= "2021-10-01"].copy()
recent["wet"] = recent["rain_3d"] >= RAIN_MM
recent["dry"] = (recent["rain_3d"] < RAIN_MM) & (recent["days_dry"] >= 3)
rows = []
for code, g in recent.groupby("StationCode"):
    if len(g) < 50:
        continue
    dry, wet = g[g["dry"]], g[g["wet"]]
    dr = dry["exceeds"].mean() if len(dry) >= 20 else np.nan
    wr = wet["exceeds"].mean() if len(wet) >= 10 else np.nan
    if pd.notna(dr) and dr >= 0.10:
        kind = "chronic"        # unsafe 1 in 10 dry days or more: a persistent source
    elif pd.notna(wr) and wr >= 0.30:
        kind = "rain-driven"    # mostly fine when dry, often unsafe after rain: runoff
    else:
        kind = "usually clean"
    s = beach.loc[code]
    rows.append({"station": code, "name": s["name"], "lat": s["lat"], "lon": s["lon"],
                 "samples": int(len(g)), "unsafe_rate": r3(g["exceeds"].mean()),
                 "dry_unsafe_rate": r3(dr), "wet_unsafe_rate": r3(wr), "type": kind})
hot = pd.DataFrame(rows).sort_values("unsafe_rate", ascending=False)
(OUT / "hotspots.json").write_text(json.dumps({
    "period": "2021-10-01 to latest", "dry_definition": "under 0.1 in of rain in the past 3 days and 3+ dry days",
    "wet_definition": "0.1 in or more of rain in the past 3 days",
    "rules": {"chronic": "unsafe on 10%+ of dry days", "rain-driven": "under 10% when dry, 30%+ after rain"},
    "stations": hot.to_dict("records")}, indent=1))
print("A. Hotspots:", hot["type"].value_counts().to_dict())
print(hot.head(6)[["station", "name", "unsafe_rate", "dry_unsafe_rate", "wet_unsafe_rate", "type"]].to_string(index=False))

# ---------------- B. Sewage spills, 2023 onward ----------------
sp = pd.read_csv(P / "spills_la.csv", parse_dates=["start"])
sp = sp[(sp["start"] >= "2023-01-01") & (sp["surface_gal"] >= 10_000)].copy()
sp["day"] = sp["start"].dt.normalize()
base = pred.groupby("StationCode")["resid"].mean()  # each station's usual residual
events, all_after = [], []
for s_ in sp.itertuples():
    d = km(s_.lat, s_.lon, beach["lat"].values, beach["lon"].values)
    near = beach.index[d <= 10]
    w = pred[pred["StationCode"].isin(near) & (pred["date"] > s_.day)
             & (pred["date"] <= s_.day + pd.Timedelta(days=7))]
    if len(w) == 0:
        continue
    excess = (w["resid"] - w["StationCode"].map(base)).mean()
    all_after.append(w.assign(excess=w["resid"] - w["StationCode"].map(base)))
    events.append({"spill_id": str(s_.spill_id), "date": str(s_.day.date()), "location": s_.location,
                   "lat": s_.lat, "lon": s_.lon, "surface_gallons": int(s_.surface_gal),
                   "beach_samples_next_7_days": int(len(w)), "stations": int(w["StationCode"].nunique()),
                   "observed_unsafe_rate": r3(w["exceeds"].mean()),
                   "predicted_unsafe_rate": r3(w["risk"].mean()),
                   "excess_vs_usual": r3(excess)})
ev = pd.DataFrame(events).sort_values("surface_gallons", ascending=False)
aa = pd.concat(all_after).drop_duplicates(["StationCode", "date"])
summary = {"spills_with_beach_samples": int(len(ev)), "beach_samples_after_spills": int(len(aa)),
           "observed_unsafe_rate": r3(aa["exceeds"].mean()), "predicted_unsafe_rate": r3(aa["risk"].mean()),
           "average_excess_vs_usual": r3(aa["excess"].mean())}
(OUT / "spills.json").write_text(json.dumps({"rule": "spills since 2023 sending 10,000+ gallons to surface water; beach samples within 10 km over the next 7 days",
                                             "summary": summary, "events": ev.to_dict("records")}, indent=1))
print("\nB. Spills:", summary)
print(ev.head(6)[["date", "location", "surface_gallons", "beach_samples_next_7_days",
                  "observed_unsafe_rate", "predicted_unsafe_rate", "excess_vs_usual"]].to_string(index=False))

# ---------------- C. Post-fire (Palisades) ----------------
gj = json.loads((P / "burn_areas_2025.geojson").read_text())
pal = next(f for f in gj["features"] if f["properties"]["FIRE_NAME"].upper() == "PALISADES")["geometry"]
rings = pal["coordinates"] if pal["type"] == "Polygon" else [r for poly in pal["coordinates"] for r in poly]
v = np.vstack([np.array(r)[:, :2] for r in rings])
dist = pd.Series({c: km(r.lat, r.lon, v[:, 1], v[:, 0]).min() for c, r in beach.iterrows()})
burn_st = dist[dist <= 1].index      # directly below the burn area
ctrl_st = dist[dist >= 10].index     # far enough to be unaffected
pf = pred[pred["StationCode"].isin(burn_st.union(ctrl_st))].copy()
pf["group"] = np.where(pf["StationCode"].isin(burn_st), "burn", "control")
pf["period"] = np.where(pf["date"] >= FIRE_DATE, "after", "before")
pf["wet"] = pf["rain_3d"] >= RAIN_MM

def did(d):
    m = d.groupby(["group", "period"])["resid"].mean()
    return (m["burn", "after"] - m["burn", "before"]) - (m["control", "after"] - m["control", "before"])

def boot(d, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    by = {k: g for k, g in d.groupby("StationCode")}
    b_ids, c_ids = list(d.loc[d.group == "burn", "StationCode"].unique()), list(d.loc[d.group == "control", "StationCode"].unique())
    out = []
    for _ in range(n):  # resample whole stations, so one noisy beach can't drive the result
        pick = list(rng.choice(b_ids, len(b_ids))) + list(rng.choice(c_ids, len(c_ids)))
        out.append(did(pd.concat([by[s] for s in pick])))
    return [r3(np.percentile(out, 2.5)), r3(np.percentile(out, 97.5))]

def cell(d):
    return {"samples": int(len(d)), "observed_unsafe_rate": r3(d["exceeds"].mean()),
            "predicted_unsafe_rate": r3(d["risk"].mean()), "mean_residual": r3(d["resid"].mean())}

result = {"overall": {"estimate": r3(did(pf)), "ci95": boot(pf)}}
# Same comparison within rainy days only, and within dry days only (both periods).
for label, sub in [("wet_days", pf[pf.wet]), ("dry_days", pf[~pf.wet])]:
    result[label] = {"estimate": r3(did(sub)), "ci95": boot(sub), "samples": int(len(sub))}
pf["month"] = pf["date"].dt.to_period("M").astype(str)
monthly = (pf.groupby(["month", "group"])["resid"].mean().unstack().round(3).reset_index()
             .rename(columns={"burn": "burn_residual", "control": "control_residual"}))
post = {
    "method": "difference-in-differences on model residuals (observed minus predicted), burn-adjacent vs. far beaches, before vs. after 2025-01-07",
    "burn_stations": [{"station": c, "name": beach.loc[c, "name"], "km_from_burn": round(float(dist[c]), 2)} for c in burn_st],
    "control_station_count": int(len(ctrl_st)),
    "cells": {f"{g}_{p}": cell(pf[(pf.group == g) & (pf.period == p)]) for g in ["burn", "control"] for p in ["before", "after"]},
    "result": result,
    "eaton_note": "No monitoring station lies within 5 km of the Eaton burn area; its runoff reaches the ocean via the LA River, so it is not analyzed here.",
    "monthly": monthly.to_dict("records")}
(OUT / "post_fire.json").write_text(json.dumps(post, indent=1, default=str))
print(f"\nC. Post-fire: {len(burn_st)} burn-adjacent stations, {len(ctrl_st)} controls")
for k, c in post["cells"].items():
    print(" ", k, c)
print("  difference-in-differences:", result)

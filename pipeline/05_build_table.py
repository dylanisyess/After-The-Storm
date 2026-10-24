"""Step 5: join everything into one training table (one row per bacteria sample).

Every feature uses only information available BEFORE the sample day, except
tide predictions, which are known in advance:
  rain_*        rain in the 1, 3, 7 and 30 days before the sample day (mm)
  days_dry      days since the last day with >= 2.5 mm (0.1 in) of rain
  tide_*        predicted high/low/range on the sample day (ft)
  month, wet_season
  last_*        the station's most recent result that was KNOWN by the sample
                day: taken from samples at least 2 days earlier, because lab
                culture results take about a day to come back
  spill_*       sewage reaching surface water within 10 km in the prior 7 days
Output: processed/training_table.csv
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
P = ROOT / "processed"
LAB_LAG_DAYS = 2
SPILL_KM, SPILL_DAYS = 10, 7

lab = pd.read_csv(P / "labels.csv", parse_dates=["SampleDate"])
st = pd.read_csv(P / "stations.csv")
lab = lab.merge(st[["StationCode", "water", "cell_lat", "cell_lon", "tide_station"]], on="StationCode")
lab = lab[["StationCode", "SampleDate", "water", "cell_lat", "cell_lon", "tide_station",
           "TargetLatitude", "TargetLongitude", "value", "exceeds"]].rename(
    columns={"SampleDate": "date", "TargetLatitude": "lat", "TargetLongitude": "lon"})

# ---- Rain: rolling sums ending the day BEFORE each date ----
rain = pd.read_csv(P / "rain_daily.csv", parse_dates=["date"]).sort_values("date")
rain["precip_mm"] = rain["precip_mm"].fillna(0)
feats = []
for (la, lo), g in rain.groupby(["cell_lat", "cell_lon"]):
    s = g.set_index("date")["precip_mm"].asfreq("D", fill_value=0)
    prev = s.shift(1)  # yesterday and earlier only
    wet = (s >= 2.5)
    last_wet = s.index.to_series().where(wet).ffill().shift(1)
    feats.append(pd.DataFrame({
        "cell_lat": la, "cell_lon": lo, "date": s.index,
        "rain_1d": prev.values,
        "rain_3d": prev.rolling(3).sum().values,
        "rain_7d": prev.rolling(7).sum().values,
        "rain_30d": prev.rolling(30).sum().values,
        "days_dry": (s.index.to_series() - last_wet).dt.days.clip(upper=365).values}))
rain_f = pd.concat(feats)
lab = lab.merge(rain_f, on=["cell_lat", "cell_lon", "date"], how="left")

# ---- Tides: daily predicted extremes ----
t = pd.read_csv(P / "tides_hilo.csv", parse_dates=["time"])
t["date"] = t["time"].dt.normalize()
tide_f = t.groupby(["tide_station", "date"]).agg(
    tide_high=("height_ft", "max"), tide_low=("height_ft", "min")).reset_index()
tide_f["tide_range"] = tide_f["tide_high"] - tide_f["tide_low"]
lab = lab.merge(tide_f, on=["tide_station", "date"], how="left")

# ---- Season ----
lab["month"] = lab["date"].dt.month
lab["wet_season"] = lab["month"].isin([11, 12, 1, 2, 3, 4]).astype(int)

# ---- Last result KNOWN by the sample day (lab lag) ----
lab = lab.sort_values("date").reset_index(drop=True)
known = lab[["StationCode", "date", "value", "exceeds"]].copy()
known["known_date"] = known["date"] + pd.Timedelta(days=LAB_LAG_DAYS)
known = known.rename(columns={"date": "last_sample_date", "value": "last_value",
                              "exceeds": "last_exceeds"}).sort_values("known_date")
lab = pd.merge_asof(lab, known, left_on="date", right_on="known_date", by="StationCode",
                    direction="backward", allow_exact_matches=True)
lab["last_log_value"] = np.log10(lab["last_value"].clip(lower=1))
lab["days_since_last"] = (lab["date"] - lab["last_sample_date"]).dt.days

# ---- Sewage spills near each station in the prior week ----
sp = pd.read_csv(P / "spills_la.csv", parse_dates=["start"])
sp = sp[sp["surface_gal"] > 0].copy()
sp["day"] = sp["start"].dt.normalize()
def km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(a))
days = pd.date_range("2009-12-01", lab["date"].max())
rows = []
for _, s_ in st.iterrows():
    near = sp[km(s_.lat, s_.lon, sp["lat"].values, sp["lon"].values) <= SPILL_KM]
    gal = near.groupby("day")["surface_gal"].sum().reindex(days, fill_value=0)
    cnt = near.groupby("day").size().reindex(days, fill_value=0)
    rows.append(pd.DataFrame({"StationCode": s_.StationCode, "date": days,
        "spill_gal_7d": gal.shift(1).rolling(SPILL_DAYS).sum().values,
        "spill_count_7d": cnt.shift(1).rolling(SPILL_DAYS).sum().values}))
lab = lab.merge(pd.concat(rows), on=["StationCode", "date"], how="left")
lab["spill_log_gal_7d"] = np.log10(lab["spill_gal_7d"].fillna(0) + 1)

lab = lab.drop(columns=["known_date", "last_value", "last_sample_date"])
lab.to_csv(P / "training_table.csv", index=False)
print(f"{len(lab):,} rows x {lab.shape[1]} columns")
print("missing share per feature:")
print(lab.isna().mean().round(3)[lambda s: s > 0].to_string())

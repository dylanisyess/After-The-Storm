"""Step 5: join everything into one training table (one row per bacteria sample).

Every feature uses only information available BEFORE the sample day, except
tide predictions, which are known in advance:
  rain_*        rain in the 1, 3, 7 and 30 days before the sample day (mm).
                Left blank if the rain record has a gap (never guessed as 0).
  days_dry      days since the last day with >= 2.5 mm (0.1 in) of rain
  tide_*        predicted high/low/range on the sample day (ft)
  month, wet_season
  last_*        the station's most recent result that was KNOWN by the sample
                day: taken from samples at least 2 days earlier, because lab
                culture results take about a day to come back
  past_n, past_unsafe   how many of the station's results were known by the
                sample day (same 2-day rule), and how many were unsafe.
                06 turns these into past_rate.
Two more columns describe today's warning signs. They are only used to compare
the model with signs, never as model features:
  sign_1d       the latest enterococcus result that was back by the sample day
                was unsafe. LA County posts a sign as soon as the lab result
                comes back, so signs use a 1-day lag (the app uses 2: public
                data takes longer to appear online).
  sign3_1d      the same, but failing ANY of the three state ocean standards
                (enterococcus 104, fecal coliform 400, total coliform 10,000,
                or total coliform 1,000 when fecal/total > 0.1), which is what
                counties actually post on.
Output: processed/training_table.csv
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
P = ROOT / "processed"
LAB_LAG_DAYS, SIGN_LAG_DAYS = 2, 1

lab = pd.read_csv(P / "labels.csv", parse_dates=["SampleDate"])
st = pd.read_csv(P / "stations.csv")
lab = lab.merge(st[["StationCode", "water", "cell_lat", "cell_lon", "tide_station"]], on="StationCode")
lab = lab[["StationCode", "SampleDate", "water", "cell_lat", "cell_lon", "tide_station",
           "TargetLatitude", "TargetLongitude", "value", "exceeds"]].rename(
    columns={"SampleDate": "date", "TargetLatitude": "lat", "TargetLongitude": "lon"})

# ---- Rain: rolling sums ending the day BEFORE each date ----
rain = pd.read_csv(P / "rain_daily.csv", parse_dates=["date"]).sort_values("date")
feats = []
for (la, lo), g in rain.groupby(["cell_lat", "cell_lon"]):
    s = g.set_index("date")["precip_mm"].asfreq("D")  # a missing day stays blank (NaN)
    prev = s.shift(1)  # yesterday and earlier only
    wet = (s >= 2.5)
    last_wet = s.index.to_series().where(wet).ffill().shift(1)
    feats.append(pd.DataFrame({
        "cell_lat": la, "cell_lon": lo, "date": s.index,
        "rain_1d": prev.values,
        # a sum over a window with a blank day is blank too (rolling's default)
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


def last_known(samples, known, lag_days):
    """For each sample, attach the station's most recent row of `known` taken at
    least lag_days before the sample day. merge_asof finds the latest row whose
    known_date is on or before the sample date, separately for each station."""
    k = known.assign(known_date=known["date"] + pd.Timedelta(days=lag_days)).drop(columns="date")
    out = pd.merge_asof(samples.sort_values("date"), k.sort_values("known_date"),
                        left_on="date", right_on="known_date", by="StationCode",
                        direction="backward", allow_exact_matches=True)
    return out.drop(columns="known_date")


def recent_rate(samples, hist, lag_days, days=365, min_tests=10):
    """Past-year track record: for each sample, the share unsafe among the station's results
    sampled in the `days` before it AND back from the lab by then (sampled at least lag_days
    before). Blank (NaN) if fewer than min_tests such results. 08 and 10 use the same function.
    For each station, dates are sorted, so searchsorted finds the window's first and last result,
    and a running total of unsafe results gives the count inside the window."""
    out = pd.Series(np.nan, index=samples.index)
    for code, idx in samples.groupby("StationCode").groups.items():
        h = hist[hist["StationCode"] == code].sort_values("date")
        if h.empty:
            continue
        dates = h["date"].values
        unsafe = np.concatenate([[0], np.cumsum(h["exceeds"].values)])
        d = samples.loc[idx, "date"].values
        hi = np.searchsorted(dates, d - np.timedelta64(lag_days, "D"), side="right")  # sampled on or before d - lag
        lo = np.searchsorted(dates, d - np.timedelta64(days, "D"), side="left")       # sampled on or after d - 365
        n = hi - lo
        rate = (unsafe[hi] - unsafe[lo]) / np.maximum(n, 1)
        out.loc[idx] = np.where(n >= min_tests, rate, np.nan)
    return out


# ---- Last result and station history KNOWN by the sample day (lab lag) ----
hist = lab[["StationCode", "date", "value", "exceeds"]].sort_values(["StationCode", "date"])
hist = hist.assign(
    last_sample_date=hist["date"],
    past_n=hist.groupby("StationCode").cumcount() + 1,            # results up to and including this one
    past_unsafe=hist.groupby("StationCode")["exceeds"].cumsum(),  # how many of them were unsafe
).rename(columns={"value": "last_value", "exceeds": "last_exceeds"})
lab = last_known(lab, hist, LAB_LAG_DAYS)
lab[["past_n", "past_unsafe"]] = lab[["past_n", "past_unsafe"]].fillna(0)  # nothing known yet
lab["recent_rate"] = recent_rate(lab, hist.rename(columns={"last_exceeds": "exceeds"}), LAB_LAG_DAYS)
lab["last_log_value"] = np.log10(lab["last_value"].clip(lower=1))
lab["days_since_last"] = (lab["date"] - lab["last_sample_date"]).dt.days

# ---- What today's warning signs showed (for comparison only) ----
b = pd.read_csv(P / "bacteria_extra.csv",  # written by 02
                usecols=["StationCode", "SampleDate", "Analyte", "value"],
                parse_dates=["SampleDate"])
coli = (b[b["Analyte"].str.startswith("Coliform")]
        .pivot_table(index=["StationCode", "SampleDate"], columns="Analyte", values="value")
        .reset_index().rename(columns={"SampleDate": "date", "Coliform, Fecal": "fc", "Coliform, Total": "tc"}))
post = lab.loc[lab["water"] == "ocean", ["StationCode", "date", "exceeds"]].merge(
    coli, on=["StationCode", "date"], how="left")
post["sign3"] = ((post["exceeds"] == 1) | (post["fc"] > 400) | (post["tc"] > 10000)
                 | ((post["tc"] > 1000) & (post["fc"] / post["tc"] > 0.1))).astype(int)
post = post.rename(columns={"exceeds": "sign_1d", "sign3": "sign3_1d"})
lab = last_known(lab, post[["StationCode", "date", "sign_1d", "sign3_1d"]], SIGN_LAG_DAYS)

lab = lab.drop(columns=["last_value", "last_sample_date"]).sort_values(["date", "StationCode"])
lab.to_csv(P / "training_table.csv", index=False)
print(f"{len(lab):,} rows x {lab.shape[1]} columns")
print("missing share per column:")
print(lab.isna().mean().round(3)[lambda s: s > 0].to_string())

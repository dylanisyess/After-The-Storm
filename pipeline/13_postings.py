"""Step 13: real beach postings (Beach Watch) -> app_data/postings.json

  python3 pipeline/13_postings.py

Input: the state's "Beach Posting and Closures - Advisories" CSV from
https://data.ca.gov/dataset/beach-water-quality-postings-and-closures (Beach Watch),
saved as beachwatch/beach-advisories.csv, plus beachwatch/beach-monitoring-stations.csv
from the same page (raw downloads, not in git). The first run trims them to the LA and
Ventura rows we use, processed/postings_la.csv and processed/station_county.csv (in git),
so later runs work without the raw files.

What it makes:
1. Each beach's official advisory history: postings (a sign after a failed test),
   closures and rain advisories, with start and end dates.
2. Each station's county (Los Angeles or Ventura), for the app's county filter.
3. A separate check of the forecast against these REAL postings, from 2023 to the
   last record, with the same equal-warnings method as step 10: give the model the
   same number of warnings as the real postings, and compare the share of unsafe
   tested days each one caught. The 95% range comes from resampling whole beaches.
   It reads the saved test predictions (model/test_predictions.csv); nothing is retrained.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW, P, OUT = ROOT / "beachwatch", ROOT / "processed", ROOT / "app_data"
MODEL_ID = json.loads((ROOT / "model" / "metrics.json").read_text())["model_id"]
OUR_COUNTIES = {"Los Angeles": "Los Angeles", "Long Beach City": "Los Angeles", "Ventura": "Ventura"}
SPLIT, REPS = "2023-01-01", 1000   # same test start and number of resamples as step 10

# ---------------- 1. Read (and the first time, trim) the Beach Watch files ----------------
if (RAW / "beach-advisories.csv").exists():
    a = pd.read_csv(RAW / "beach-advisories.csv", encoding="utf-8-sig", low_memory=False)
    a = a[a["CountyName"].isin(OUR_COUNTIES)]
    a = a[["Advisory id", "Station_Name", "HistoricalStationName", "AgencyStationIdentifier", "CountyName",
           "AdvisoryType", "DateofAdvisory", "DateOpened", "AdvisoryCause", "Source"]]
    a.to_csv(P / "postings_la.csv", index=False)
    s = pd.read_csv(RAW / "beach-monitoring-stations.csv", encoding="utf-8-sig")
    s = s[s["CountyName"].isin(OUR_COUNTIES)][["Station_Name", "AgencyStationIdentifier", "CountyName"]]
    s.to_csv(P / "beachwatch_stations_la.csv", index=False)
a = pd.read_csv(P / "postings_la.csv")
bw_stations = pd.read_csv(P / "beachwatch_stations_la.csv")
for c in ("DateofAdvisory", "DateOpened"):
    a[c] = pd.to_datetime(a[c], format="mixed", errors="coerce")
a = a.dropna(subset=["DateofAdvisory"])
through = a["DateofAdvisory"].max()  # the last record in the file

ours = pd.read_csv(P / "stations.csv")  # every station in the app
codes = set(ours["StationCode"])


def our_code(row):
    """Match a Beach Watch station to our station code. Beach Watch names it in up to
    three columns (current name, older name, agency's own ID), e.g. 'B-22' or 'MdRH-2'."""
    for c in ("Station_Name", "HistoricalStationName", "AgencyStationIdentifier"):
        v = str(row[c]).strip()
        for guess in (v, v.upper()):
            if guess in codes:
                return guess
    return None


a["code"] = a.apply(our_code, axis=1)

# ---------------- 2. Health department and county of every station ----------------
# Beach Watch's "CountyName" is really the health department that posts the beach: LA County
# Public Health ("Los Angeles"), the City of Long Beach's own health department ("Long Beach City"),
# or Ventura County. Taken from Beach Watch where the station is listed; otherwise from the nearest listed station.
dept = {}
for df in (bw_stations.assign(code=bw_stations.apply(
        lambda r: next((g for g in (str(r["Station_Name"]), str(r["AgencyStationIdentifier"]).upper()) if g in codes), None),
        axis=1)), a.dropna(subset=["code"])):
    for code, c in zip(df["code"], df["CountyName"]):
        if code and code not in dept:
            dept[code] = c
known = ours[ours["StationCode"].isin(dept)]
source = {}
for _, r in ours.iterrows():
    if r["StationCode"] in dept:
        source[r["StationCode"]] = "Beach Watch"
        continue
    km = np.hypot((known["lat"] - r["lat"]) * 111, (known["lon"] - r["lon"]) * 92)  # rough km
    dept[r["StationCode"]] = dept[known.iloc[int(km.argmin())]["StationCode"]]
    source[r["StationCode"]] = "nearest listed station"
county = {code: OUR_COUNTIES[d] for code, d in dept.items()}
print("counties:", pd.Series(county).value_counts().to_dict(),
      f"({sum(v == 'Beach Watch' for v in source.values())} from Beach Watch)")

# General advisories (mostly rain advisories) are filed under ONE placeholder station per health
# department, e.g. "All_LosAngeles_County_Beaches". They cover every beach that department posts,
# so each one is copied to all of its stations.
everywhere = a["Station_Name"].astype(str).str.startswith("All_")
copies = [a[everywhere & (a["CountyName"] == d)].assign(code=code, AdvisoryCause="County-wide advisory")
          for code, d in dept.items()]
print(f"{len(a):,} LA/Ventura advisories through {through:%Y-%m-%d}: {a['code'].notna().sum():,} matched a station, "
      f"{everywhere.sum()} county-wide (copied to every station of that health department)")
own_records = set(a["code"].dropna())  # stations Beach Watch lists by name (the check uses only these)
a = pd.concat([a.dropna(subset=["code"])] + copies, ignore_index=True)

# ---------------- 3. Each beach's history ----------------
history = {}
for code, g in a.sort_values("DateofAdvisory").groupby("code"):
    history[code] = [[r.AdvisoryType, f"{r.DateofAdvisory:%Y-%m-%d}",
                      None if pd.isna(r.DateOpened) else f"{r.DateOpened:%Y-%m-%d}",
                      None if pd.isna(r.AdvisoryCause) else str(r.AdvisoryCause)]
                     for r in g.itertuples()]

# ---------------- 4. Forecast vs. real postings ----------------
t = pd.read_csv(ROOT / "model" / "test_predictions.csv", parse_dates=["date"])
# Only beaches Beach Watch lists by name: for the others, "no record" may just mean another station name.
t = t[(t["date"] >= SPLIT) & (t["date"] <= through) & t["StationCode"].isin(own_records)].reset_index(drop=True)


def posted(types):
    """True for each tested beach-day when an advisory of these types was up at the start of
    that day: issued on an EARLIER day (a sign put up later on the test day came after the
    sample was taken) and not yet removed (removed on the test day itself still counts, since
    the water was sampled in the morning, before the passing result came back)."""
    on = np.zeros(len(t), bool)
    ad = a[a["AdvisoryType"].isin(types)]
    end = ad["DateOpened"].fillna(through + pd.Timedelta(days=1))
    by_code = {c: (ad["DateofAdvisory"][ad["code"] == c].values, end[ad["code"] == c].values) for c in ad["code"].unique()}
    for i, (c, d) in enumerate(zip(t["StationCode"], t["date"].values)):
        if c in by_code:
            s, e = by_code[c]
            on[i] = ((s < d) & (e >= d)).any()
    return on


def stats(y, risk, rule):
    """Same as step 10: the model's riskiest beach-days, the same count as the rule's warnings."""
    m = risk >= np.quantile(risk, 1 - rule.mean())
    return {"share": rule.mean(), "caught": rule[y == 1].mean(), "model_caught": m[y == 1].mean(),
            "gain": m[y == 1].mean() - rule[y == 1].mean()}


y, risk = t["exceeds"].to_numpy(), t["risk"].to_numpy()
rows_of = [np.where(t["StationCode"].to_numpy() == c)[0] for c in t["StationCode"].unique()]
check = {}
for key, types in {"postings": ["Posting", "Closure"], "postings_plus_rain": ["Posting", "Closure", "Rain"]}.items():
    rule = posted(types)
    point = stats(y, risk, rule)
    rng, reps = np.random.default_rng(0), []
    for _ in range(REPS):  # resample whole beaches with replacement
        k = np.concatenate([rows_of[i] for i in rng.integers(len(rows_of), size=len(rows_of))])
        reps.append(stats(y[k], risk[k], rule[k]))
    reps = pd.DataFrame(reps)
    check[key] = {n: [round(float(v), 4), round(float(reps[n].quantile(.025)), 4), round(float(reps[n].quantile(.975)), 4)]
                  for n, v in point.items()}
    c = check[key]
    print(f"{key:20s} warns on {c['share'][0]:.1%} of tested days; caught {c['caught'][0]:.1%} vs model "
          f"{c['model_caught'][0]:.1%}; gain {c['gain'][0] * 100:+.1f} points "
          f"({c['gain'][1] * 100:+.1f} to {c['gain'][2] * 100:+.1f})")

out = {"model_id": MODEL_ID, "records_through": f"{through:%Y-%m-%d}",
       "source": "California State Water Resources Control Board, Beach Watch: Beach Posting and Closures - Advisories "
                 "(data.ca.gov/dataset/beach-water-quality-postings-and-closures)",
       "columns": ["type", "start", "end", "cause"], "stations": history,
       "county": county, "county_source": source,
       "check": {"period": [SPLIT, f"{through:%Y-%m-%d}"], "tested_beach_days": len(t),
                 "beaches": int(t["StationCode"].nunique()), "unsafe_days": int(y.sum()), **check}}
(OUT / "postings.json").write_text(json.dumps(out, separators=(",", ":")))
print(f"wrote app_data/postings.json ({len(history)} stations with records; check on {len(t):,} tested beach-days)")

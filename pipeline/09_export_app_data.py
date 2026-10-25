"""Step 9: station list, recent history and map layers for the app -> app_data/

  stations.json         every station: location, beach or inland, record, last test, hotspot type,
                        in_model, and on_map: false for stormwater devices (BMPs, catch basins,
                        Oxford Basin) and inland stations with fewer than 20 samples.
                        Every beach stays on the map.
  history.json          each station's samples from the past 2 years (for charts);
                        the ocean study sites below get their whole record
  burn_areas_2025.geojson, model_metrics.json   copied for the app

Ocean study sites: 13 shoreline sites sampled in short studies (2015-2018) under
non-beach programs, so 02 files them as freshwater and grades some on E. coli.
Here, for the app only, they are shown as beaches graded on enterococcus. They
stay out of the model (in_model false, no forecast); nothing upstream changes.
"""
import json, shutil
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
P, OUT = ROOT / "processed", ROOT / "app_data"
MIN_SAMPLES = 20
STORMWATER = r"\bBMP\b|catch[ _]?basin|oxford basin"  # treatment/detention devices, not places people swim
OCEAN_STUDY_SITES = [
    "LACSDS7", "HYPCB2",                                                  # Outer Cabrillo Beach
    "SCCWRPCB01", "SCCWRPCB02", "SCCWRPCB04", "SCCWRPCB05", "SCCWRPCB06",  # Inner Cabrillo Beach study
    "SMB-5-2", "SMB-5-4", "SMB-6-1", "SMB-6-2", "SMB-6-3", "SMB-6-4"]     # Manhattan/Hermosa/Redondo shoreline

st = pd.read_csv(P / "stations.csv", parse_dates=["first", "last"])
st["in_model"] = st["water"] == "ocean"  # decided before relabeling: the study sites stay out
lab = pd.read_csv(P / "labels.csv", parse_dates=["SampleDate"])[["StationCode", "SampleDate", "value", "exceeds"]]
hot = {h["station"]: h["type"] for h in json.loads((OUT / "hotspots.json").read_text())["stations"]}
cut = lab["SampleDate"].max() - pd.Timedelta(days=730)

# ---- Study sites: swap their results for enterococcus results, show them as beaches ----
b = pd.read_csv(P / "bacteria_la.csv", usecols=["StationCode", "SampleDate", "Analyte", "value"],
                parse_dates=["SampleDate"])
ent = b[b["StationCode"].isin(OCEAN_STUDY_SITES) & (b["Analyte"] == "Enterococcus")].drop(columns="Analyte")
ent = ent.assign(exceeds=(ent["value"] > 104).astype(int))
lab = pd.concat([lab[~lab["StationCode"].isin(OCEAN_STUDY_SITES)], ent]).sort_values("SampleDate")
site = st["StationCode"].isin(OCEAN_STUDY_SITES)
summary = ent.groupby("StationCode").agg(n=("value", "size"), first=("SampleDate", "min"),
                                         last=("SampleDate", "max"), exceed_rate=("exceeds", "mean"))
for col in summary.columns:
    st.loc[site, col] = st.loc[site, "StationCode"].map(summary[col])
st.loc[site, "water"], st.loc[site, "analyte"] = "ocean", "Enterococcus"

last = lab.groupby("StationCode").tail(1).set_index("StationCode")
st["on_map"] = (st["water"] == "ocean") | (
    (st["n"] >= MIN_SAMPLES) & ~st["name"].str.contains(STORMWATER, case=False, regex=True))

stations = [{"station": r.StationCode, "name": r.name, "lat": round(r.lat, 5), "lon": round(r.lon, 5),
             "water": r.water, "standard": "enterococcus > 104" if r.analyte == "Enterococcus" else "E. coli > 320",
             "samples": int(r.n), "first": f"{r.first:%Y-%m-%d}", "last": f"{r.last:%Y-%m-%d}",
             "unsafe_rate": round(r.exceed_rate, 3), "type": hot.get(r.StationCode),
             "in_model": bool(r.in_model), "on_map": bool(r.on_map),
             "note": (f"short-term study site ({r.first:%Y}-{r.last:%Y}); no forecast"
                      if r.StationCode in OCEAN_STUDY_SITES else None),
             "last_test": {"date": f"{last.loc[r.StationCode, 'SampleDate']:%Y-%m-%d}",
                           "value": float(last.loc[r.StationCode, "value"]),
                           "unsafe": bool(last.loc[r.StationCode, "exceeds"])}}
            for r in st.itertuples()]
(OUT / "stations.json").write_text(json.dumps(stations))

recent = lab[(lab["SampleDate"] >= cut) | lab["StationCode"].isin(OCEAN_STUDY_SITES)]
history = {code: [[f"{d:%Y-%m-%d}", float(v), int(e)] for d, v, e in zip(g.SampleDate, g.value, g.exceeds)]
           for code, g in recent.groupby("StationCode")}
(OUT / "history.json").write_text(json.dumps({"columns": ["date", "value", "unsafe"], "stations": history}))

shutil.copy(P / "burn_areas_2025.geojson", OUT / "burn_areas_2025.geojson")
shutil.copy(ROOT / "model" / "metrics.json", OUT / "model_metrics.json")
print(f"{len(stations)} stations ({int(st.on_map.sum())} on the map, {int(site.sum())} ocean study sites "
      f"shown as beaches), history for {len(history)} stations since {cut:%Y-%m-%d}")

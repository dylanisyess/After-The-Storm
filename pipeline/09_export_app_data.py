"""Step 9: station list, recent history and map layers for the app -> app_data/

  stations.json         every station: location, beach or inland, record, last test, hotspot type
  history.json          each station's samples from the past 2 years (for charts)
  burn_areas_2025.geojson, model_metrics.json   copied for the app
"""
import json, shutil
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
P, OUT = ROOT / "processed", ROOT / "app_data"
st = pd.read_csv(P / "stations.csv", parse_dates=["first", "last"])
lab = pd.read_csv(P / "labels.csv", parse_dates=["SampleDate"]).sort_values("SampleDate")
hot = {h["station"]: h["type"] for h in json.loads((OUT / "hotspots.json").read_text())["stations"]}
last = lab.groupby("StationCode").tail(1).set_index("StationCode")

stations = [{"station": r.StationCode, "name": r.name, "lat": round(r.lat, 5), "lon": round(r.lon, 5),
             "water": r.water, "standard": "enterococcus > 104" if r.analyte == "Enterococcus" else "E. coli > 320",
             "samples": int(r.n), "first": f"{r.first:%Y-%m-%d}", "last": f"{r.last:%Y-%m-%d}",
             "unsafe_rate": round(r.exceed_rate, 3), "type": hot.get(r.StationCode),
             "in_model": r.water == "ocean",
             "last_test": {"date": f"{last.loc[r.StationCode, 'SampleDate']:%Y-%m-%d}",
                           "value": float(last.loc[r.StationCode, "value"]),
                           "unsafe": bool(last.loc[r.StationCode, "exceeds"])}}
            for r in st.itertuples()]
(OUT / "stations.json").write_text(json.dumps(stations))

cut = lab["SampleDate"].max() - pd.Timedelta(days=730)
recent = lab[lab["SampleDate"] >= cut]
history = {code: [[f"{d:%Y-%m-%d}", float(v), int(e)] for d, v, e in zip(g.SampleDate, g.value, g.exceeds)]
           for code, g in recent.groupby("StationCode")}
(OUT / "history.json").write_text(json.dumps({"columns": ["date", "value", "unsafe"], "stations": history}))

shutil.copy(P / "burn_areas_2025.geojson", OUT / "burn_areas_2025.geojson")
shutil.copy(ROOT / "model" / "metrics.json", OUT / "model_metrics.json")
print(f"{len(stations)} stations, history for {len(history)} stations since {cut:%Y-%m-%d}")

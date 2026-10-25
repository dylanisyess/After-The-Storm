"""Step 2: build the station list (one row per station) used by every later step.

Thresholds: enterococcus single-sample 104 MPN/100 mL (CA ocean beach posting
standard); E. coli statistical threshold value 320 cfu/100 mL (CA freshwater).
"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
THRESH = {"Enterococcus": 104, "E. coli": 320}

b = pd.read_csv(ROOT / "processed" / "bacteria_la.csv", parse_dates=["SampleDate"])
b = b[b["Analyte"].isin(THRESH)]  # coliforms are only used to score warning signs (05)
# Grade each station on ONE standard: beaches report both bacteria, but the
# E. coli 320 limit is a freshwater standard. Keep each station's main analyte.
# BeachWatch stations are ocean beaches (enterococcus); others are inland (E. coli when sampled).
g = b.groupby("StationCode")
beach = g["DataSource"].agg(lambda s: (s == "BeachWatch").any())
has_ecoli = g["Analyte"].agg(lambda s: (s == "E. coli").any())
main = pd.Series(["Enterococcus" if bw or not ec else "E. coli" for bw, ec in zip(beach, has_ecoli)], index=beach.index)
b = b[b["Analyte"] == b["StationCode"].map(main)].copy()
b["exceeds"] = (b["value"] > b["Analyte"].map(THRESH)).astype(int)
st = (b.groupby("StationCode")
        .agg(name=("StationName", "first"), lat=("TargetLatitude", "median"),
             lon=("TargetLongitude", "median"), analyte=("Analyte", "first"),
             n=("value", "size"), first=("SampleDate", "min"), last=("SampleDate", "max"),
             exceed_rate=("exceeds", "mean"))
        .reset_index())
st["water"] = st["StationCode"].map(beach).map({True: "ocean", False: "freshwater"})
# Weather grid cell (ERA5 is 0.25 deg): stations in the same cell share one rain series.
st["cell_lat"] = ((st["lat"] / 0.25).round() * 0.25).round(2)
st["cell_lon"] = ((st["lon"] / 0.25).round() * 0.25).round(2)
# Nearest NOAA tide station: Santa Monica (9410840) or Los Angeles/San Pedro (9410660).
TIDE = {"9410840": (34.0083, -118.5), "9410660": (33.72, -118.272)}
st["tide_station"] = st.apply(lambda r: min(TIDE, key=lambda k: (r.lat - TIDE[k][0])**2 + (r.lon - TIDE[k][1])**2), axis=1)
st.to_csv(ROOT / "processed" / "stations.csv", index=False)
b.to_csv(ROOT / "processed" / "labels.csv", index=False)

print(st.groupby("water").agg(stations=("n", "size"), samples=("n", "sum"),
                              exceed=("exceed_rate", "mean")).round(3).to_string())
print("stations with >=100 samples:", (st.n >= 100).sum())
print("overall exceedance:", b.groupby("Analyte")["exceeds"].mean().round(3).to_dict())
print("lat range", st.lat.min(), st.lat.max(), "lon range", st.lon.min(), st.lon.max())
print(st.sort_values("n", ascending=False).head(8)[["StationCode", "name", "n", "exceed_rate"]].to_string(index=False))

"""Step 3: LA-region sewage spills (2007-present) and 2025 Palisades/Eaton burn perimeters.

Spills: merges the pre-June-2023 SSO.txt and post-June-2023 Cat1-2-3-Spills.txt,
keeping certified reports from active LA-region (Region 4) enrollees, latest
version per spill. Output: processed/spills_la.csv
Fires: processed/burn_areas_2025.geojson (Palisades and Eaton only).
"""
import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
read = lambda f: pd.read_csv(ROOT / f, sep="\t", dtype=str, encoding="latin-1",
                             on_bad_lines="skip", low_memory=False)

old = read("SSO.txt")
old = old[(old["REGION"] == "4") & (old["STATUS"] == "Active") & (old["STEP"] == "Certified")]
old = pd.DataFrame({
    "spill_id": old["SPILL ID"], "start": old["START DT"],
    "lat": old["LATTITUDE DECIMAL DEGREES"], "lon": old["LONGITUDE DECIMAL DEGREES"],
    "volume_gal": old["SPILL VOL"], "surface_gal": old["SPILL VOL REACH SURF"],
    "location": old["SPILL LOC NAME"], "source_file": "SSO.txt"})

new = read("Cat1-2-3-Spills.txt")
new = new[(new["REGION"] == "4") & (new["STATUS"] == "Active")
          & new["SPILL_STATUS"].isin(["Certified", "Amended"])]
new["ver"] = pd.to_numeric(new["SPILL_REPORT_VERSION"], errors="coerce")
new = new.sort_values("ver").drop_duplicates("SPILL_EVENT_ID", keep="last")
new = pd.DataFrame({
    "spill_id": new["SPILL_EVENT_ID"], "start": new["ESTIMATED_SPILL_START_DATE_AND_TIME"],
    "lat": new["LATITUDE"], "lon": new["LONGITUDE"],
    "volume_gal": new["ESTIMATED_TOTAL_SPILL_VOLUME_EXITING_THE_SYSTEM_(GAL)"],
    "surface_gal": new["ESTIMATED_SPILL_VOLUME_THAT_DISCHARGED_TO_SURFACE_WATERS_(GAL)"],
    "location": new["SPILL_LOCATION_NAME"], "source_file": "Cat1-2-3-Spills.txt"})

sp = pd.concat([old, new], ignore_index=True)
sp["start"] = pd.to_datetime(sp["start"], errors="coerce", format="mixed")
for c in ["lat", "lon", "volume_gal", "surface_gal"]:
    sp[c] = pd.to_numeric(sp[c].str.replace(",", ""), errors="coerce")
sp["surface_gal"] = sp["surface_gal"].fillna(0)
sp = sp.dropna(subset=["start", "lat", "lon"])
sp = sp[sp["lat"].between(32, 36) & sp["lon"].between(-121, -116)]  # drop bad coordinates
sp.to_csv(ROOT / "processed" / "spills_la.csv", index=False)
print(f"{len(sp):,} LA spills, {sp.start.min().date()} to {sp.start.max().date()}; "
      f"{(sp.surface_gal > 0).sum():,} reached surface water, "
      f"{sp.surface_gal.sum()/1e6:.1f}M gal total to surface water")

gj = json.loads(next(ROOT.glob("*Fire_Perimeters*.geojson")).read_text())
keep = [f for f in gj["features"]
        if str(f["properties"].get("YEAR_")) == "2025"
        and str(f["properties"].get("FIRE_NAME", "")).upper() in {"PALISADES", "EATON"}]
for f in keep:
    p = f["properties"]
    print(p.get("FIRE_NAME"), p.get("ALARM_DATE"), round(float(p.get("GIS_ACRES") or 0)), "acres")
(ROOT / "processed" / "burn_areas_2025.geojson").write_text(
    json.dumps({"type": "FeatureCollection", "features": keep}))

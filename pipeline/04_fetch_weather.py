"""Step 4 (run on your own computer): download daily rain and tides.

Uses only the Python standard library, so no installs are needed.
    python pipeline/04_fetch_weather.py

Rain: Open-Meteo Historical Weather API (ERA5 reanalysis), one daily series per
0.25-degree weather cell that contains a monitoring station.
Tides: NOAA CO-OPS high/low tide predictions for Santa Monica and Los Angeles.
Forecast-model rain: live forecasts (08) can't use ERA5, which arrives ~5 days
late, so they use Open-Meteo's forecast models instead. Those models see about
25% less rain at our coastal grid points than ERA5, which the model was trained
on. We download the forecast models' archive (from 2021-03-23) and save one
scale factor per weather cell, ERA5 total / forecast total, measured on
2021-2022 only (before the test years). 08 multiplies live rain by it.

Open-Meteo's free tier counts a long date range as many calls, so this script
paces itself and waits when rate-limited. It saves each cell as it finishes:
if it stops, just run it again and it picks up where it left off.
"""
import csv, json, time, urllib.request, urllib.error
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "processed"
CACHE = OUT / "rain_cells"
CACHE.mkdir(parents=True, exist_ok=True)
START, END = "2010-01-01", time.strftime("%Y-%m-%d", time.localtime(time.time() - 86400))


def get_json(url, tries=12):
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "water-risk-student-app"})
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code == 429:  # rate limited: wait it out
                wait = 120 * (i + 1)
                print(f"  rate limited, waiting {wait // 60} min (attempt {i + 1}/{tries})...", flush=True)
                time.sleep(wait)
            else:
                raise
        except urllib.error.URLError as e:
            print(f"  network error ({e.reason}), retrying in 30 s...", flush=True)
            time.sleep(30)
    raise RuntimeError("gave up after repeated rate limits; run the script again later")


# ---- Rain ----
with open(OUT / "stations.csv", newline="") as f:
    cells = sorted({(float(r["cell_lat"]), float(r["cell_lon"])) for r in csv.DictReader(f)})
print(f"Rain: {len(cells)} weather cells, {START} to {END}")
for i, (lat, lon) in enumerate(cells, 1):
    path = CACHE / f"{lat}_{lon}.csv"
    if path.exists():
        print(f"[{i}/{len(cells)}] {lat},{lon} already done")
        continue
    print(f"[{i}/{len(cells)}] {lat},{lon} downloading...", flush=True)
    url = ("https://archive-api.open-meteo.com/v1/archive"
           f"?latitude={lat}&longitude={lon}&start_date={START}&end_date={END}"
           "&daily=precipitation_sum&timezone=America%2FLos_Angeles&models=era5")
    d = get_json(url)["daily"]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cell_lat", "cell_lon", "date", "precip_mm"])
        for day, mm in zip(d["time"], d["precipitation_sum"]):
            w.writerow([lat, lon, day, "" if mm is None else mm])
    time.sleep(65)  # stay under the per-minute limit

with open(OUT / "rain_daily.csv", "w", newline="") as out:
    w = csv.writer(out)
    w.writerow(["cell_lat", "cell_lon", "date", "precip_mm"])
    for p in sorted(CACHE.glob("*.csv")):
        with open(p, newline="") as f:
            next(f)
            w.writerows(csv.reader(f))
print("Rain saved to processed/rain_daily.csv")

# ---- Tides (high/low predictions; NOAA allows 10 years per request) ----
rows = []
for station in ["9410840", "9410660"]:
    for a, b in [("20100101", "20141231"), ("20150101", "20191231"),
                 ("20200101", "20241231"), ("20250101", END.replace("-", ""))]:
        url = ("https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"
               f"?begin_date={a}&end_date={b}&station={station}&product=predictions"
               "&datum=MLLW&time_zone=lst_ldt&interval=hilo&units=english"
               "&application=water_risk_student_app&format=json")
        data = get_json(url)
        if "predictions" not in data:
            raise RuntimeError(f"NOAA error for {station} {a}-{b}: {data}")
        for p in data["predictions"]:
            rows.append([station, p["t"], p["type"], p["v"]])
        print(f"Tides: station {station} {a[:4]}-{b[:4]} done", flush=True)
        time.sleep(2)
with open(OUT / "tides_hilo.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["tide_station", "time", "type", "height_ft"])
    w.writerows(rows)
print("Tides saved to processed/tides_hilo.csv")

# ---- Forecast-model rain archive + scale factors (beach weather cells only) ----
FC_START, FIT_END = "2021-03-23", "2022-12-31"
with open(OUT / "stations.csv", newline="") as f:
    ocean_cells = sorted({(float(r["cell_lat"]), float(r["cell_lon"]))
                          for r in csv.DictReader(f) if r["water"] == "ocean"})
fc_rows, scale = [], {}
for lat, lon in ocean_cells:
    url = ("https://historical-forecast-api.open-meteo.com/v1/forecast"
           f"?latitude={lat}&longitude={lon}&start_date={FC_START}&end_date={END}"
           "&daily=precipitation_sum&timezone=America%2FLos_Angeles")
    d = get_json(url)["daily"]
    fc = {day: mm for day, mm in zip(d["time"], d["precipitation_sum"])}
    fc_rows += [[lat, lon, day, "" if mm is None else mm] for day, mm in fc.items()]
    # Compare with ERA5 on days both have, 2021-2022 only
    with open(CACHE / f"{lat}_{lon}.csv", newline="") as f:
        era = {r["date"]: r["precip_mm"] for r in csv.DictReader(f)}
    both = [day for day in fc if day <= FIT_END and fc[day] is not None and era.get(day, "") != ""]
    era_total, fc_total = sum(float(era[day]) for day in both), sum(fc[day] for day in both)
    scale[f"{lat},{lon}"] = round(era_total / fc_total, 3)
    print(f"Forecast-model rain {lat},{lon}: scale factor {scale[f'{lat},{lon}']}", flush=True)
    time.sleep(20)
with open(OUT / "rain_forecast_model.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["cell_lat", "cell_lon", "date", "precip_mm"])
    w.writerows(fc_rows)
(OUT / "rain_scale.json").write_text(json.dumps(scale, indent=1))
print("Saved processed/rain_forecast_model.csv and rain_scale.json. All done!")

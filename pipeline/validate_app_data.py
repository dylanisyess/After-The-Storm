"""Check app_data/ before the daily forecast is committed (used by .github/workflows/daily-forecast.yml).

  python3 pipeline/validate_app_data.py

Stops with an error (exit code 1) unless:
  1. every file the app reads is present,
  2. every one of them is valid JSON,
  3. forecast.json is today's (Los Angeles date), and for every beach:
     - exactly 4 forecast days, in a row, starting today;
     - station IDs appear only once;
     - every risk is a probability between 0 and 1;
     - every level matches the thresholds (low under the warning threshold, caution up to 0.50,
       high from 0.50), the same rule as 08_forecast.py.
The model-ID check is pipeline/check_model_ids.py (the workflow runs both).
"""
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

DATA = Path(__file__).resolve().parent.parent / "app_data"
NEEDED = ["forecast.json", "replay_2023-01-09.json", "stations.json", "history.json", "diagnosis.json",
          "interventions.json", "evaluator_data.json", "post_fire.json", "spills.json", "final_metrics.json",
          "model_metrics.json", "burn_areas_2025.geojson", "postings.json", "calibration.json"]

problems = []
files = {}
for name in NEEDED:
    path = DATA / name
    if not path.exists():
        problems.append(f"missing: {name}")
        continue
    try:
        files[name] = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as e:  # not valid JSON (for example, a half-written file)
        problems.append(f"not valid JSON: {name} ({e})")

fc = files.get("forecast.json")
if fc:
    today = f"{datetime.now(ZoneInfo('America/Los_Angeles')):%Y-%m-%d}"
    if fc.get("issued") != today:
        problems.append(f"forecast.json is for {fc.get('issued')}, not today ({today})")
    if not fc.get("stations"):
        problems.append("forecast.json has no beaches")
    days = [f"{datetime.strptime(today, '%Y-%m-%d') + timedelta(days=k):%Y-%m-%d}" for k in range(4)]
    codes = [s.get("station") for s in fc.get("stations", [])]
    if len(codes) != len(set(codes)):
        problems.append(f"forecast.json: station IDs repeat ({len(codes) - len(set(codes))} duplicates)")
    thr = files.get("final_metrics.json", {}).get("threshold")
    if thr is None:
        problems.append("final_metrics.json has no warning threshold to check levels against")

    def level(risk, t):  # the same rule as 08_forecast.py
        return "high" if risk >= 0.5 else "caution" if risk >= t else "low"

    for s in fc.get("stations", []):
        f = s.get("forecast", [])
        if [d.get("date") for d in f] != days:
            problems.append(f"{s['station']}: dates are {[d.get('date') for d in f]}, not 4 days in a row from today")
        for d in f:
            for key in ("risk", "typical_risk"):
                v = d.get(key)
                if key in d and not (isinstance(v, (int, float)) and 0 <= v <= 1):
                    problems.append(f"{s['station']} {d.get('date')}: {key} = {v!r} is not between 0 and 1")
            r = d.get("risk")
            # The file rounds risk to 3 decimals, so allow half a thousandth either side of each threshold.
            if thr is not None and isinstance(r, (int, float)) and d.get("level") not in {
                    level(r + e, thr) for e in (-0.0005, 0, 0.0005)}:
                problems.append(f"{s['station']} {d.get('date')}: level {d.get('level')!r} doesn't match risk {r}")

for p in problems:
    print("PROBLEM", p)
print(f"{len(files)} of {len(NEEDED)} files read." + (" All good." if not problems else ""))
sys.exit(1 if problems else 0)

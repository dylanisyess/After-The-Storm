"""Check app_data/ before the daily forecast is committed (used by .github/workflows/daily-forecast.yml).

  python3 pipeline/validate_app_data.py

Stops with an error (exit code 1) unless:
  1. every file the app reads is present,
  2. every one of them is valid JSON,
  3. forecast.json is today's (Los Angeles date) and every beach has today + 3 days.
The model-ID check is pipeline/check_model_ids.py (the workflow runs both).
"""
import json
import sys
from datetime import datetime
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
    short = [s["station"] for s in fc.get("stations", []) if len(s.get("forecast", [])) != 4]
    if short:
        problems.append(f"forecast.json: {len(short)} beaches don't have today + 3 days (e.g. {short[0]})")

for p in problems:
    print("PROBLEM", p)
print(f"{len(files)} of {len(NEEDED)} files read." + (" All good." if not problems else ""))
sys.exit(1 if problems else 0)

"""Check that every file in app_data/ came from the saved model (same model ID).

  python3 pipeline/check_model_ids.py      (rebuild.sh runs this at the end)

Each pipeline step stamps "model_id" into the files it writes. A file with a
different or missing ID was left over from an older model: rerun pipeline/rebuild.sh.
"""
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
expected = json.loads((ROOT / "model" / "metrics.json").read_text())["model_id"]

bad = 0
for f in sorted((ROOT / "app_data").glob("*.*json")):  # .json and .geojson
    found = json.loads(f.read_text()).get("model_id")
    ok = found == expected
    bad += not ok
    print(f"{'OK  ' if ok else 'BAD '} {f.name:28s} {found}")
print(f"\nSaved model: {expected}. " + ("All app files match." if not bad else f"{bad} file(s) don't match."))
sys.exit(1 if bad else 0)

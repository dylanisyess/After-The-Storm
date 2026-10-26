"""Step 14: is the forecast's risk honest? -> app_data/calibration.json

  python3 pipeline/14_calibration.py

Reads the saved test predictions (model/test_predictions.csv: every beach test from
2023 on, which the model never saw while learning) and groups them by predicted risk.
If the model is well calibrated, tests it gave a 20% risk were unsafe about 20% of the
time. For each group: how many tests, the average predicted risk, the share actually
unsafe, and a 95% range for that share (Wilson interval, good for small groups).
Nothing is retrained.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MODEL_ID = json.loads((ROOT / "model" / "metrics.json").read_text())["model_id"]
EDGES = [0, .05, .10, .15, .20, .30, .40, .50, .60, .80, 1.0]  # narrower where most tests are


def wilson(k, n, z=1.96):
    """95% range for a share k/n (Wilson score interval)."""
    p = k / n
    mid = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return mid - half, mid + half


t = pd.read_csv(ROOT / "model" / "test_predictions.csv", parse_dates=["date"])
t["bin"] = pd.cut(t["risk"], EDGES, include_lowest=True)
bins = []
for b, g in t.groupby("bin", observed=True):
    lo, hi = wilson(g["exceeds"].sum(), len(g))
    bins.append({"from": b.left if b.left > 0 else 0.0, "to": b.right, "tests": len(g),
                 "predicted": round(g["risk"].mean(), 4), "actual": round(g["exceeds"].mean(), 4),
                 "actual_low": round(lo, 4), "actual_high": round(hi, 4)})
    print(f"{b}: {len(g):6,} tests, predicted {g['risk'].mean():.1%}, actually unsafe {g['exceeds'].mean():.1%}")

out = {"model_id": MODEL_ID,
       "period": [f"{t['date'].min():%Y-%m-%d}", f"{t['date'].max():%Y-%m-%d}"],
       "tests": len(t), "mean_predicted": round(t["risk"].mean(), 4), "mean_actual": round(t["exceeds"].mean(), 4),
       "brier": round(((t["risk"] - t["exceeds"]) ** 2).mean(), 4), "bins": bins}
(ROOT / "app_data" / "calibration.json").write_text(json.dumps(out, indent=1))
print(f"all: predicted {out['mean_predicted']:.1%}, actual {out['mean_actual']:.1%}; wrote app_data/calibration.json")

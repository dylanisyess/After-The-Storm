"""Step 6: train the beach risk model and compare it to today's warning rules.

Split by TIME: train on 2010-2022, test on 2023 onward (never seen in training).
Baselines (what beachgoers get today):
  sign       : the posted result: last known sample exceeded -> warn
  sign+rain  : sign, OR at least 0.1 in of rain in the past 3 days (the LA County
               72-hour rain advisory)
The model is compared at the SAME number of warning days as sign+rain, so the
comparison is fair: same amount of warning, how many unsafe days does each catch?
Outputs: model/beach_model.joblib, model/metrics.json, model/test_predictions.csv
"""
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = Path(__file__).resolve().parent.parent
SPLIT = "2023-01-01"
FEATURES = ["rain_1d", "rain_3d", "rain_7d", "rain_30d", "days_dry",
            "tide_high", "tide_low", "tide_range", "month", "wet_season",
            "last_exceeds", "last_log_value", "days_since_last",
            "spill_log_gal_7d", "spill_count_7d", "station_rate", "lat", "lon"]

df = pd.read_csv(ROOT / "processed" / "training_table.csv", parse_dates=["date"])
df = df[df["water"] == "ocean"].copy()
train, test = df[df["date"] < SPLIT].copy(), df[df["date"] >= SPLIT].copy()

# Station history from TRAINING years only (smoothed toward the overall rate).
g = train["exceeds"].mean()
agg = train.groupby("StationCode")["exceeds"].agg(["sum", "count"])
rate = ((agg["sum"] + 10 * g) / (agg["count"] + 10)).rename("station_rate")
for d in (train, test):
    d["station_rate"] = d["StationCode"].map(rate).fillna(g)

model = HistGradientBoostingClassifier(learning_rate=0.05, max_iter=400, max_leaf_nodes=31,
                                       l2_regularization=1.0, random_state=0)
model.fit(train[FEATURES], train["exceeds"])
test["risk"] = model.predict_proba(test[FEATURES])[:, 1]

y = test["exceeds"].values
def score(warn):
    warn = np.asarray(warn).astype(bool)
    return {"warning_day_share": round(warn.mean(), 3),
            "unsafe_days_caught": round(warn[y == 1].mean(), 3),
            "warnings_that_were_right": round(y[warn].mean(), 3)}

sign = test["last_exceeds"].fillna(0) == 1
sign_rain = sign | (test["rain_3d"].fillna(0) >= 2.54)
# Warn on the riskiest X% of days, where X matches each baseline's warning share.
cut_sign = np.quantile(test["risk"], 1 - sign.mean())
cut = np.quantile(test["risk"], 1 - sign_rain.mean())
res = {
    "rows": {"train": len(train), "test": len(test), "test_unsafe_share": round(y.mean(), 3)},
    "sign": score(sign), "model_same_budget_as_sign": score(test["risk"] >= cut_sign),
    "sign_plus_rain": score(sign_rain), "model_same_budget": score(test["risk"] >= cut),
    "model_roc_auc": round(roc_auc_score(y, test["risk"]), 3),
    "model_avg_precision": round(average_precision_score(y, test["risk"]), 3),
    "risk_threshold": round(float(cut), 4)}

sub = test.sample(min(20000, len(test)), random_state=0)
imp = permutation_importance(model, sub[FEATURES], sub["exceeds"], scoring="roc_auc",
                             n_repeats=3, random_state=0)
res["feature_importance"] = dict(sorted(zip(FEATURES, imp.importances_mean.round(4)),
                                        key=lambda kv: -kv[1]))

(ROOT / "model").mkdir(exist_ok=True)
import sklearn
joblib.dump({"model": model, "features": FEATURES, "station_rate": rate.to_dict(),
             "global_rate": g, "threshold": float(cut), "sklearn_version": sklearn.__version__}, ROOT / "model" / "beach_model.joblib")
(ROOT / "model" / "metrics.json").write_text(json.dumps(res, indent=2))
test[["StationCode", "date", "value", "exceeds", "risk"] + FEATURES].to_csv(
    ROOT / "model" / "test_predictions.csv", index=False)
print(json.dumps(res, indent=2))

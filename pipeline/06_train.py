"""Step 6: train the beach risk model and compare it to today's warning rules.

Split by TIME: train on 2010-2022, test on 2023 onward (never seen in training).
Baselines (what beachgoers get today; see 05 for the 1-day sign lag):
  sign            : the latest enterococcus result was over the limit -> warn
  sign_plus_rain  : sign, OR at least 0.1 in of rain in the past 3 days (the LA
                    County 72-hour rain advisory)
  sign3, sign3_plus_rain : the same, using all three state ocean standards
The model is compared at the SAME number of warnings as each baseline, so the
comparison is fair: same amount of warning, how many unsafe days does each catch?
The app's warning threshold gives the same number of warnings as sign_plus_rain.
As a check, we also fix a threshold using 2022 only and apply it unchanged to 2023+.
Outputs: model/beach_model.joblib, model/metrics.json, model/test_predictions.csv
"""
import hashlib, json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = Path(__file__).resolve().parent.parent
SPLIT = "2023-01-01"
RAIN = ["rain_1d", "rain_3d", "rain_7d", "rain_30d", "days_dry"]
# No "month": with only two off-season storms in 2010-2022 it let the model invent an
# October-storm effect (review/forecast_puzzle.md). wet_season keeps the seasonal signal.
FEATURES = RAIN + ["tide_high", "tide_low", "tide_range", "wet_season",
                   "last_exceeds", "last_log_value", "days_since_last", "past_rate", "recent_rate", "lat", "lon"]
# min_samples_leaf=100: every rule the model learns must rest on 100+ samples, not a few rare days.
PARAMS = dict(learning_rate=0.05, max_iter=400, max_leaf_nodes=31, min_samples_leaf=100,
              l2_regularization=1.0, random_state=0)
PRIOR_WEIGHT = 10


def add_past_rate(d, overall):
    """Share of the station's already-known results that were unsafe, pulled
    toward the overall rate as if every station had 10 extra 'average' tests,
    so a station with only a few results doesn't get an extreme rate."""
    d["past_rate"] = (d["past_unsafe"] + PRIOR_WEIGHT * overall) / (d["past_n"] + PRIOR_WEIGHT)


def baselines(d):
    """Today's warning rules as True/False per row (see 05 for the sign columns)."""
    sign, sign3 = d["sign_1d"].fillna(0) == 1, d["sign3_1d"].fillna(0) == 1
    rain = d["rain_3d"] >= 2.54
    return {"sign": sign, "sign_plus_rain": sign | rain, "sign3": sign3, "sign3_plus_rain": sign3 | rain}


df = pd.read_csv(ROOT / "processed" / "training_table.csv", parse_dates=["date"])
df = df[df["water"] == "ocean"].dropna(subset=RAIN).copy()  # only days with a full rain record
train, test = df[df["date"] < SPLIT].copy(), df[df["date"] >= SPLIT].copy()
g = train["exceeds"].mean()
for d in (train, test):
    add_past_rate(d, g)

model = HistGradientBoostingClassifier(**PARAMS)
model.fit(train[FEATURES], train["exceeds"])
test["risk"] = model.predict_proba(test[FEATURES])[:, 1]

y = test["exceeds"].values
def score(warn):
    warn = np.asarray(warn).astype(bool)
    return {"warning_day_share": round(warn.mean(), 3),
            "unsafe_days_caught": round(warn[y == 1].mean(), 3),
            "warnings_that_were_right": round(y[warn].mean(), 3)}

# Model ID: a fingerprint of the model's test predictions. Every file made from this model
# carries it, so a file left over from an older model is easy to spot.
model_id = hashlib.sha256(test["risk"].values.tobytes()).hexdigest()[:10]
res = {"model_id": model_id,
       "rows": {"train": len(train), "test": len(test), "test_unsafe_share": round(y.mean(), 3)},
       "note": "shares are of TESTED beach-days (2023 onward), not calendar days",
       "baselines": {}}
for name, warn in baselines(test).items():
    # Warn on the riskiest X% of days, where X matches this baseline's warning share.
    cut_b = np.quantile(test["risk"], 1 - warn.mean())
    res["baselines"][name] = {"rule": score(warn), "model_same_budget": score(test["risk"] >= cut_b)}
cut = float(np.quantile(test["risk"], 1 - baselines(test)["sign_plus_rain"].mean()))
res.update({"model_roc_auc": round(roc_auc_score(y, test["risk"]), 3),
            "model_avg_precision": round(average_precision_score(y, test["risk"]), 3),
            "risk_threshold": round(cut, 4), "model_at_threshold": score(test["risk"] >= cut)})

# Check: a threshold fixed BEFORE seeing 2023+. Train on 2010-2021, find the risk
# level that gives sign_plus_rain's warning share on 2022, apply it to 2023+.
dev, val = train[train["date"] < "2022-01-01"].copy(), train[train["date"] >= "2022-01-01"].copy()
g_dev = dev["exceeds"].mean()
for d in (dev, val):
    add_past_rate(d, g_dev)
val_risk = HistGradientBoostingClassifier(**PARAMS).fit(dev[FEATURES], dev["exceeds"]).predict_proba(val[FEATURES])[:, 1]
cut_2022 = float(np.quantile(val_risk, 1 - baselines(val)["sign_plus_rain"].mean()))
res["threshold_set_on_2022"] = {"threshold": round(cut_2022, 4), **score(test["risk"] >= cut_2022)}

sub = test.sample(min(20000, len(test)), random_state=0)
imp = permutation_importance(model, sub[FEATURES], sub["exceeds"], scoring="roc_auc",
                             n_repeats=3, random_state=0)
res["feature_importance"] = dict(sorted(zip(FEATURES, imp.importances_mean.round(4)),
                                        key=lambda kv: -kv[1]))

(ROOT / "model").mkdir(exist_ok=True)
joblib.dump({"model": model, "features": FEATURES, "global_rate": g, "prior_weight": PRIOR_WEIGHT,
             "threshold": cut, "sklearn_version": sklearn.__version__, "model_id": model_id},
            ROOT / "model" / "beach_model.joblib")
(ROOT / "model" / "metrics.json").write_text(json.dumps(res, indent=2))
test[["StationCode", "date", "value", "exceeds", "risk", "sign_1d", "sign3_1d", "past_n", "past_unsafe"]
     + FEATURES].to_csv(ROOT / "model" / "test_predictions.csv", index=False)
print(json.dumps(res, indent=2))

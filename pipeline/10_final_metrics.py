"""Step 10: the numbers to quote -> review/final_metrics.md

Everything is measured on the test years (2023 onward), which the model never
trained on. The model is compared with what beachgoers get today, at the SAME
number of warnings (as in 06_train.py):
  signs                    latest enterococcus result over 104 (posted 1 day after sampling)
  signs + rain advisory    the same, plus LA County's 72-hour advisory after 0.1 in of rain
  three-indicator signs    any of the 3 state ocean standards failed (what counties post on)
  three-indicator + rain   the same, plus the rain advisory
95% ranges come from a bootstrap: resample whole beaches 1,000 times and
recompute everything, so the range shows how much a result depends on which
beaches happened to be in the data.

Three scenarios, all scored with the saved model (no retraining):
  A. Test set: features exactly as in training (ERA5 rain).
  B. Estimated live mode: the model gets rain from the forecast models that live
     mode uses (Open-Meteo archive, scaled per cell like 08_forecast.py). The rain
     advisory keeps ERA5 rain, standing in for the county's rain gauges.
  C. Day +3: forecast issued 3 days ahead, so the newest known test result is at
     least 5 days old. Rain is what actually fell (a perfect rain forecast), so
     this is a best case. Signs are what is posted on the day itself.
"""
import json, sys
from datetime import date
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
P = ROOT / "processed"
SPLIT, LAB_LAG_DAYS, AHEAD, REPS = "2023-01-01", 2, 3, 1000
RAIN = ["rain_1d", "rain_3d", "rain_7d", "rain_30d", "days_dry"]
BASELINES = {"sign": "Signs", "sign_plus_rain": "Signs + rain advisory",
             "sign3": "Three-indicator signs", "sign3_plus_rain": "Three-indicator signs + rain advisory"}

bundle = joblib.load(ROOT / "model" / "beach_model.joblib")
if bundle.get("sklearn_version") != sklearn.__version__ or "model_id" not in bundle:  # same check as 08
    sys.exit("The saved model doesn't match this computer's scikit-learn. Run: bash pipeline/rebuild.sh")
model, FEATURES, thr, MODEL_ID = bundle["model"], bundle["features"], bundle["threshold"], bundle["model_id"]
metrics = json.loads((ROOT / "model" / "metrics.json").read_text())


def add_past_rate(d):  # same formula as 06_train.py
    w, g = bundle["prior_weight"], bundle["global_rate"]
    d["past_rate"] = (d["past_unsafe"] + w * g) / (d["past_n"] + w)


def rain_features(daily):  # same code as 05_build_table.py
    feats = []
    for (la, lo), g in daily.groupby(["cell_lat", "cell_lon"]):
        s = g.set_index("date")["precip_mm"].asfreq("D")
        prev = s.shift(1)
        last_wet = s.index.to_series().where(s >= 2.5).ffill().shift(1)
        feats.append(pd.DataFrame({
            "cell_lat": la, "cell_lon": lo, "date": s.index, "rain_1d": prev.values,
            "rain_3d": prev.rolling(3).sum().values, "rain_7d": prev.rolling(7).sum().values,
            "rain_30d": prev.rolling(30).sum().values,
            "days_dry": (s.index.to_series() - last_wet).dt.days.clip(upper=365).values}))
    return pd.concat(feats)


def last_known(samples, known, lag_days):  # same as 05_build_table.py
    k = known.assign(known_date=known["date"] + pd.Timedelta(days=lag_days)).drop(columns="date")
    out = pd.merge_asof(samples.sort_values("date"), k.sort_values("known_date"),
                        left_on="date", right_on="known_date", by="StationCode",
                        direction="backward", allow_exact_matches=True)
    return out.drop(columns="known_date")


def recent_rate(samples, hist, lag_days, days=365, min_tests=10):
    """Same as 05_build_table.py. Past-year track record: for each sample, the share unsafe among the station's results
    sampled in the `days` before it AND back from the lab by then (sampled at least lag_days
    before). Blank (NaN) if fewer than min_tests such results. 08 and 10 use the same function.
    For each station, dates are sorted, so searchsorted finds the window's first and last result,
    and a running total of unsafe results gives the count inside the window."""
    out = pd.Series(np.nan, index=samples.index)
    for code, idx in samples.groupby("StationCode").groups.items():
        h = hist[hist["StationCode"] == code].sort_values("date")
        if h.empty:
            continue
        dates = h["date"].values
        unsafe = np.concatenate([[0], np.cumsum(h["exceeds"].values)])
        d = samples.loc[idx, "date"].values
        hi = np.searchsorted(dates, d - np.timedelta64(lag_days, "D"), side="right")  # sampled on or before d - lag
        lo = np.searchsorted(dates, d - np.timedelta64(days, "D"), side="left")       # sampled on or after d - 365
        n = hi - lo
        rate = (unsafe[hi] - unsafe[lo]) / np.maximum(n, 1)
        out.loc[idx] = np.where(n >= min_tests, rate, np.nan)
    return out


# ---------------- Build the three scenarios ----------------
df = pd.read_csv(P / "training_table.csv", parse_dates=["date"])
test = df[(df["water"] == "ocean") & (df["date"] >= SPLIT)].dropna(subset=RAIN).copy()
test["rain_3d_observed"] = test["rain_3d"]  # what the rain advisory is based on
add_past_rate(test)

A = test.copy()

fc = pd.read_csv(P / "rain_forecast_model.csv", parse_dates=["date"])
scale = json.loads((P / "rain_scale.json").read_text())
fc["precip_mm"] *= [scale[f"{la},{lo}"] for la, lo in zip(fc["cell_lat"], fc["cell_lon"])]
B = test.drop(columns=RAIN).merge(rain_features(fc), on=["cell_lat", "cell_lon", "date"], how="left")
B = B.dropna(subset=RAIN)

lab = pd.read_csv(P / "labels.csv", parse_dates=["SampleDate"]).sort_values(["StationCode", "SampleDate"])
hist = pd.DataFrame({"StationCode": lab["StationCode"], "date": lab["SampleDate"],
                     "last_sample_date": lab["SampleDate"], "last_exceeds": lab["exceeds"],
                     "last_log_value": np.log10(lab["value"].clip(lower=1)),
                     "past_n": lab.groupby("StationCode").cumcount() + 1,
                     "past_unsafe": lab.groupby("StationCode")["exceeds"].cumsum()})
C = test.drop(columns=["last_exceeds", "last_log_value", "days_since_last", "past_n", "past_unsafe"])
C = last_known(C, hist, LAB_LAG_DAYS + AHEAD)
C[["past_n", "past_unsafe"]] = C[["past_n", "past_unsafe"]].fillna(0)
C["days_since_last"] = (C["date"] - C["last_sample_date"]).dt.days
C = C.reset_index(drop=True)
C["recent_rate"] = recent_rate(C, lab.rename(columns={"SampleDate": "date"}), LAB_LAG_DAYS + AHEAD)  # 3 days ahead
add_past_rate(C)

for d in (A, B, C):
    d["risk"] = model.predict_proba(d[FEATURES])[:, 1]


# ---------------- Numbers + bootstrap ranges ----------------
def stats(y, risk, rules):
    """Model alone at the app threshold, then model vs each rule at the same number of warnings."""
    w = risk >= thr
    out = {"auc": roc_auc_score(y, risk), "share": w.mean(), "caught": w[y == 1].mean(), "right": y[w].mean()}
    for name, rule in rules.items():
        m = risk >= np.quantile(risk, 1 - rule.mean())  # model's riskiest days, same count as the rule
        out.update({f"{name}.share": rule.mean(), f"{name}.caught": rule[y == 1].mean(),
                    f"{name}.right": y[rule].mean(), f"{name}.model_caught": m[y == 1].mean(),
                    f"{name}.model_right": y[m].mean()})
        out[f"{name}.gain"] = out[f"{name}.model_caught"] - out[f"{name}.caught"]
    return out


def evaluate(d):
    sign, sign3 = (d["sign_1d"].fillna(0) == 1).values, (d["sign3_1d"].fillna(0) == 1).values
    rain = (d["rain_3d_observed"] >= 2.54).values
    rules = {"sign": sign, "sign_plus_rain": sign | rain, "sign3": sign3, "sign3_plus_rain": sign3 | rain}
    y, risk = d["exceeds"].values, d["risk"].values
    point = stats(y, risk, rules)
    codes = d["StationCode"].values
    rows_of = [np.where(codes == c)[0] for c in np.unique(codes)]
    rng, reps = np.random.default_rng(0), []
    for _ in range(REPS):  # resample whole beaches with replacement
        k = np.concatenate([rows_of[i] for i in rng.integers(len(rows_of), size=len(rows_of))])
        reps.append(stats(y[k], risk[k], {n: r[k] for n, r in rules.items()}))
    reps = pd.DataFrame(reps)
    return {key: (val, reps[key].quantile(0.025), reps[key].quantile(0.975)) for key, val in point.items()}, d


pct = lambda v: f"{100 * v[0]:.1f}% ({100 * v[1]:.1f} to {100 * v[2]:.1f})"
pts = lambda v: f"{100 * v[0]:+.1f} points ({100 * v[1]:+.1f} to {100 * v[2]:+.1f})"
auc = lambda v: f"{v[0]:.3f} ({v[1]:.3f} to {v[2]:.3f})"


def section(title, intro, res, d):
    lines = [f"## {title}", "", intro, "",
             f"{len(d):,} tested beach-days at {d['StationCode'].nunique()} beaches, "
             f"{d['exceeds'].mean():.1%} of them unsafe.", "",
             "| Model | Value (95% range) |", "|---|---|",
             f"| ROC-AUC | {auc(res['auc'])} |",
             f"| Warns on (risk ≥ {thr:.3f}) | {pct(res['share'])} of tested days |",
             f"| Unsafe days caught | {pct(res['caught'])} |",
             f"| Warnings that were right | {pct(res['right'])} |", "",
             "| Today's rule | Rule warns on | Rule catches | Model catches, same number of warnings | Model minus rule | Rule's warnings right | Model's warnings right |",
             "|---|---|---|---|---|---|---|"]
    for name, label in BASELINES.items():
        r = lambda k: res[f"{name}.{k}"]
        lines.append(f"| {label} | {pct(r('share'))} | {pct(r('caught'))} | {pct(r('model_caught'))} | "
                     f"**{pts(r('gain'))}** | {pct(r('right'))} | {pct(r('model_right'))} |")
    return lines + [""]


resA, A = evaluate(A)
resB, B = evaluate(B)
resC, C = evaluate(C)
print("A", {k: round(v[0], 3) for k, v in resA.items() if "gain" in k or k == "auc"})
print("B", {k: round(v[0], 3) for k, v in resB.items() if "gain" in k or k == "auc"})
print("C", {k: round(v[0], 3) for k, v in resC.items() if "gain" in k or k == "auc"})

# ---------------- Write the report ----------------
f = lambda v: f"{100 * v[0]:.0f}%"
diff = lambda v: f"{100 * v[0]:+.1f} points; 95% range {100 * v[1]:+.1f} to {100 * v[2]:+.1f}"
quote = [
    f"- On {len(A):,} tested beach-days from {A['date'].min():%B %Y} to {A['date'].max():%B %Y} that the model "
    f"never saw in training, it caught **{f(resA['sign_plus_rain.model_caught'])} of unsafe days**, compared with "
    f"{f(resA['sign_plus_rain.caught'])} for warning signs plus LA County's rain advisory, with the same number "
    f"of warnings ({f(resA['sign_plus_rain.share'])} of tested days). Difference: {diff(resA['sign_plus_rain.gain'])}.",
    f"- Against warning signs alone, at the same number of warnings: {f(resA['sign.model_caught'])} vs "
    f"{f(resA['sign.caught'])} of unsafe days ({diff(resA['sign.gain'])}).",
    f"- Against signs based on all three state standards, plus the rain advisory: "
    f"{f(resA['sign3_plus_rain.model_caught'])} vs {f(resA['sign3_plus_rain.caught'])} ({diff(resA['sign3_plus_rain.gain'])}).",
    f"- Estimated live accuracy, using the rain data the live app actually gets: "
    f"{f(resB['sign_plus_rain.model_caught'])} vs {f(resB['sign_plus_rain.caught'])} for signs + rain advisory "
    f"({diff(resB['sign_plus_rain.gain'])}).",
    f"- Three days ahead, assuming a good rain forecast: {f(resC['sign_plus_rain.model_caught'])} of unsafe days "
    f"caught vs {f(resC['sign_plus_rain.caught'])} for the signs + rain advisory posted on the day itself "
    f"({diff(resC['sign_plus_rain.gain'])}). The range includes zero, so say \"about as good as\", not \"better than\".",
    f"- ROC-AUC {auc(resA['auc'])}.",
]
t22 = metrics["threshold_set_on_2022"]
md = ([f"# Final metrics", "",
       f"Generated by `pipeline/10_final_metrics.py` on {date.today():%Y-%m-%d} from the saved model "
       f"(model ID `{MODEL_ID}`; "
       f"trained on {metrics['rows']['train']:,} beach samples, 2010-2022). Test period: "
       f"{A['date'].min():%Y-%m-%d} to {A['date'].max():%Y-%m-%d}, never used for training.", "",
       "**How to read this.** Percentages are shares of *tested beach-days* (days a beach was sampled), not "
       "calendar days. \"Unsafe\" means enterococcus over 104 MPN/100 mL. Each rule is compared with the model "
       "warning on the same number of days, so the question is: with the same amount of warning, which catches "
       "more unsafe days? Ranges in parentheses are 95% ranges from resampling whole beaches 1,000 times. "
       "Retraining the model can shift results by about ±0.5 points, so don't quote more precision than that.", "",
       "**Today's rules.** *Signs*: a warning is posted when the latest enterococcus result is over the limit, one "
       "day after sampling (when the lab result comes back). *Rain advisory*: LA County's advice to avoid the water "
       "for 72 hours after 0.1 inch of rain. *Three-indicator signs*: a warning when any of the three state ocean "
       "standards fails (enterococcus, fecal coliform, total coliform), which is what counties post on. The model "
       "uses results at least 2 days old, because public data takes longer to appear online.", "",
       "## Quotable summary", ""] + quote + [""]
      + section("A. Test set (2023 onward)", "The model with the same inputs it was trained on (ERA5 rain).", resA, A)
      + section("B. Estimated live mode",
                "The live app can't use ERA5 rain (it arrives about 5 days late), so it uses Open-Meteo's forecast "
                "models, scaled per weather cell to match ERA5 (scale factors fitted on 2021-2022 only). Here the "
                "model gets that archived forecast-model rain for every test day; the rain advisory keeps ERA5 rain "
                "as a stand-in for the county's rain gauges. This estimates today's (day 0) forecast; rain forecasts "
                "for later days add more error.", resB, B)
      + section("C. Day +3 forecast",
                "A forecast issued 3 days ahead: the model only knows test results at least 5 days old. Today's "
                "rules use what is posted on the day itself. Rain is the rain that actually fell (a perfect rain "
                "forecast), so this is a best case.", resC, C)
      + ["## Threshold fixed in advance", "",
         f"The app warns when risk ≥ {thr:.3f}, chosen to give the same number of warnings as signs + rain "
         "advisory on the test years (no test results are used to choose it). As a check, a threshold set using "
         f"2022 only ({t22['threshold']:.3f}) and applied unchanged to 2023 onward warns on "
         f"{100 * t22['warning_day_share']:.1f}% of tested days and catches {100 * t22['unsafe_days_caught']:.1f}% of "
         f"unsafe days ({100 * t22['warnings_that_were_right']:.1f}% of warnings right).", "",
         "## Caveats", "",
         "- Results are for beach-days that were actually sampled. After a bad result, beaches are often "
         "resampled, so unsafe days are over-represented compared with all calendar days.",
         "- The sign baselines are reconstructed from the data, not from records of actual postings.",
         "- Live mode (B) is an estimate from archived forecast-model rain. Real 1-3 day rain forecasts are less "
         "accurate than that, so later forecast days will do somewhat worse than B and C suggest.", ""])
out = ROOT / "review" / "final_metrics.md"
out.parent.mkdir(exist_ok=True)
out.write_text("\n".join(md))
print("Wrote", out.relative_to(ROOT))

# ---------------- Same numbers for the app ----------------
# app.py reads its quoted numbers from here, so they can't drift from this report.
# Each number is [value, low, high]: the result and its 95% range. Keys match the
# report code above, e.g. "sign_plus_rain.gain" = model minus signs + rain advisory.
pack = lambda res: {k: [round(float(x), 6) for x in v] for k, v in res.items()}  # 6 places, so the app rounds exactly like the report
app_json = {
    "model_id": MODEL_ID,
    "generated": f"{date.today():%Y-%m-%d}",
    "test_period": [f"{A['date'].min():%Y-%m-%d}", f"{A['date'].max():%Y-%m-%d}"],
    "tested_beach_days": len(A), "beaches": int(A["StationCode"].nunique()),
    "train_samples": metrics["rows"]["train"], "threshold": round(float(thr), 4),
    "scenarios": {"test": pack(resA), "live": pack(resB), "day3": pack(resC)},
}
out = ROOT / "app_data" / "final_metrics.json"
out.write_text(json.dumps(app_json, indent=1))
print("Wrote", out.relative_to(ROOT))

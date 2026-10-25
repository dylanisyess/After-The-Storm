# Pipeline review of d3a4efb

Reviewed commit `d3a4efb` (`model builds and initial app`) against its parent. Scope: `pipeline/`, `requirements.txt`, `CLAUDE.md`, and the committed `model/` and `app_data/` outputs. No app UI files were reviewed. All execution used the `git archive HEAD` copy in `/tmp/waterapp-review`, with Python from `/home/dylan/waterenv` (pandas 3.1.0, scikit-learn 1.9.1). This report is the only project file written.

**Result:** three medium-priority findings and one low-priority temporal-consistency finding. The reproducible model-dependent outputs are consistent with the saved model; no stale committed output was found among those that could be verified. Full audit coverage and live-forecast provenance could not be established; see limitations below.

## Findings

### 1. [P2 / Medium] New dependencies on an uncommitted intermediate break steps 05 and 09 in a clean checkout

**Evidence:** `pipeline/05_build_table.py:99–100` and `pipeline/09_export_app_data.py:36–37` now read `processed/bacteria_la.csv`. That file is absent from HEAD and explicitly ignored. Both scripts fail with `FileNotFoundError` in the isolated archive. Previously, these steps could operate on the committed processed inputs. Step 09 fails before exporting anything, including the copied model metrics, so the documented `08` → `09` refresh and post-retraining export sequence cannot complete from this checkout. Step 05 cannot reconstruct the new three-indicator sign baseline from `labels.csv`, which retains only one analyte per station.

**Suggested fix:** commit compact derived inputs containing the coliform observations needed by 05 and the study-site enterococcus observations needed by 09, and have those scripts consume them. Alternatively, distribute/version the complete intermediate with explicit retrieval instructions. Preserve the ability to run steps 05 onward without the raw downloads; do not silently replace unavailable coliform measurements with safe results.

### 2. [P2 / Medium] Forecast rain windows still turn missing coverage into apparently valid features

**Evidence:** `pipeline/08_forecast.py:94–101` calls `asfreq("D")` only over the existing series bounds, then sums its last N rows. It neither extends the series through the day before the target nor requires N observations. Using the committed function with a series containing only `2023-01-01: 10.0` and target `2023-01-05` returns **10.0 for all of `rain_1d`, `rain_3d`, `rain_7d`, and `rain_30d`**. January 1 is outside both the one-day and three-day windows. An empty preceding slice also sums to zero, and no known wet day produces `days_dry = 365` even without sufficient history.

This is reachable when a replay extends beyond the saved weather coverage or a returned series is truncated. `skipna=False` correctly propagates missing values inside existing windows, but does not fix missing boundaries or short windows. Training step 05 instead requires complete rolling windows, so the claimed training/forecast feature equivalence is incomplete.

**Suggested fix:** reindex to explicit calendar windows ending at `d - 1 day`, require the complete 1/3/7/30 observations, and leave incomplete results missing. Match step 05's unknown `days_dry` behavior rather than interpreting absent history as a year of dry weather. Check coverage before emitting a forecast.

### 3. [P2 / Medium, inherited] Automatic retraining can still leave app artifacts from different models

**Evidence:** `pipeline/08_forecast.py:38–48` retrains via 06 when loading fails or the installed scikit-learn version differs. Step 06 overwrites the model, metrics, and test predictions; step 08 subsequently writes only the requested forecast or replay. Existing `spills.json`, `post_fire.json`, other forecast/replay files, `model_metrics.json`, and `review/final_metrics.md` remain untouched. Even following the normal `08` → `09` refresh only updates a subset of these files. Pinning scikit-learn avoids the usual version mismatch on a correctly installed environment, but the automatic fallback remains, and `CLAUDE.md` still advertises it.

This is an inherited failure path, not evidence that the currently committed artifacts are mixed. It conflicts with the new instruction in `CLAUDE.md` to regenerate all downstream artifacts after every retrain, because the forecast command can retrain implicitly.

**Suggested fix:** make forecasting fail with an actionable model/environment mismatch message and require an explicit rebuild command, or have that command regenerate and publish the full artifact set together. Include a model hash/run identifier in generated files so partial refreshes are detectable.

### 4. [P3 / Low] The smoothing prior still uses future labels within training years

**Evidence:** `pipeline/06_train.py:51–53` computes `g` from all 2010–2022 training outcomes and inserts it into every training row's `past_rate` through line 38. For example, station `15000` on `2010-04-05` has `past_n = past_unsafe = 0`, but receives `past_rate = 0.10194078154948312`, a rate computed using outcomes through 2022. There are 78 modeled training rows with no known station history; rows with some history also receive this future-informed prior with decreasing weight.

The station-specific counts themselves are correctly lagged. This is **not leakage of 2023+ outcomes into the fitted model**, and no material accuracy inflation was demonstrated. It does, however, leave an exception to the requested strict rule that every feature use only information available before its sample day. The former implementation also used a full-training-period global prior; the new expanding station history fixes the much larger station-specific lookahead but retains this component.

**Suggested fix:** use a fixed external prior established before the modeled period, or calculate an expanding global prior from results available at the same two-day cutoff. Define an explicit fallback for the beginning of the record and share the calculation across training and forecasting.

## Verification results

- **Date parsing (01):** the new `format="mixed"` expression correctly parses both `2023-01-10` and `2023-01-10 08:58:00` and normalizes both to midnight. The actual recovery percentage could not be measured without the raw downloads.
- **Historical features (05/06):** executing 05 through the feature-building section reproduced all 130,326 committed rows and their shared columns within floating-point tolerance. The minimum last-result age is two days; station/date labels are unique. Every saved test-row `past_rate` matches the committed smoothing formula exactly. The prior exception is finding 4.
- **Signs:** independently reconstructing the one-day enterococcus baseline from `labels.csv` produced zero mismatches on ocean rows. The coliform comparisons implement the thresholds stated in the code, but their source measurements cannot be checked without the missing intermediate.
- **Rain scaling (04/08):** all ten committed scale factors exactly match rounded ERA5-total / forecast-model-total ratios on common nonmissing days from 2021-03-23 through 2022-12-31. No test-year rain enters those fits. Step 08 multiplies live rain by the corresponding factor and leaves replay ERA5 rain unscaled.
- **Forecast lag (08):** both live and replay branches use `issue_date - 2 days` for last results and station history, keeping those results fixed across the four forecast dates. The committed forecasts satisfy this cutoff. Both replays regenerate byte for byte.
- **Model and metrics:** the saved model reproduces all 30,262 committed test probabilities exactly. Independently computed ROC-AUC is `0.8588948305531889` and average precision is `0.57266275127976`, matching `metrics.json` after rounding. `app_data/model_metrics.json` is byte-identical to `model/metrics.json`.
- **Final report:** rerunning 10 regenerated `review/final_metrics.md` byte for byte, including all three scenarios and bootstrap intervals. Its test-set numbers agree with `metrics.json`. The +5.0-point gain uses unrounded rates, so it need not equal subtraction of the separately rounded 66.2% and 61.1%. The 66% estimated-live figure is the equal-warning-budget comparison; at the deployed threshold the report correctly lists 62.8% for live-mode recall.
- **Other app artifacts:** rerunning 07 reproduced `hotspots.json`, `spills.json`, and `post_fire.json` byte for byte. The burn geometry matches its processed input. All 142 non-study station histories match the committed labels; all non-study `last_test` entries match as well. The 13 study-site histories are internally consistent with their exported counts and latest values, but their full source verification needs the missing bacteria intermediate.

## Limitations and interpretation

- `review/claude_review.md` does not exist in HEAD, and `git log --all -- review/claude_review.md` returned no history. Permission to read a local copy or another committed reference was requested, but none was supplied during this review. Consequently, this checks the fixes described in the request, commit, and `CLAUDE.md`; it cannot certify every item in the original audit.
- The live `forecast.json` does not include a model identifier, its feature matrix, or the weather API responses used to generate it. Its structure and lab cutoff can be checked, but its exact predictions cannot be independently reproduced from committed inputs. Fetching today's API responses would not establish what the earlier run consumed. Therefore **every app file coming from the same model run remains unproven**, despite the exact matches for all reproducible model-dependent artifacts.
- Historical replays deliberately use observed future rain for later forecast days. Step 10 also labels its day +3 scenario as a perfect-rain-forecast best case. These are retrospective simulations, not leakage-free reconstructions of actual forecasts issued on those dates. Their hotspot descriptions additionally come from the latest full-period hotspot export; that affects explanations, not model features or probabilities.
- No scripts ran in the main project folder. No raw-download steps or live API calls were run, and no model was retrained. The checks used the committed model so that output consistency could be assessed against the actual reviewed run.

# Beach water risk app (Congressional App Challenge 2026, CA-28)

Student project by Dylan. Submission deadline: Monday Oct 26, 2026, 12:00 PM ET
(9:00 AM Pacific); confirm on Rep. Chu's page. AI use must be disclosed in the
submission, so keep a short log of what was AI-generated. Dylan must be able to
explain every part, so prefer simple, readable code with comments.

## What exists (done)
Data pipeline in `pipeline/` (run order and outputs in `pipeline/README.md`):
- LA-region beach bacteria samples 2010 to Oct 2026 (CA Water Boards Safe to Swim
  data), daily rain (Open-Meteo ERA5), tides (NOAA), sewage spills (CIWQS),
  2025 Palisades/Eaton burn perimeters (CAL FIRE).
- Model: scikit-learn HistGradientBoostingClassifier predicting P(enterococcus >
  104 MPN/100 mL) at beach stations. Trained 2010-2022, tested 2023-2026.
  Quotable numbers with 95% ranges: `review/final_metrics.md` (made by
  `pipeline/10_final_metrics.py`). Model ID `40ebd86751` (Oct 24 rebuild). Test
  set: ROC-AUC 0.86; at the same number of warnings, catches 66% of unsafe days
  vs 61% for signs + 72-hour rain advisory (+4.7 points, 95% range +1.8 to +8.2),
  and 53% vs 46% for signs alone. Estimated live mode: 65% vs 61% (+3.6, range
  +0.7 to +7.6). Day +3: 62% vs 61% (range includes zero: "about as good as").
- Decisions already made (ask Dylan before changing): beaches only (inland E.
  coli stations have too few samples to model); time-based split; last test
  result lagged 2 days for lab turnaround; rain features use days before the
  sample only; warning threshold = model risk 0.161 (same warning budget as
  signs + rain rule on the test years; a threshold set on 2022 alone gives
  similar results). No `month` feature (`wet_season` only) and
  min_samples_leaf=100: with month, the model invented an October-storm effect
  from 2 off-season storms (see `review/forecast_puzzle.md`). A "big spring
  tide" reason shows on caution/high days when the daily tide range is 6.5 ft+
  (Boehm & Weisberg 2005).
- Audit fixes (Oct 24, 2026, AI-assisted): date parsing in 01 no longer drops
  ~2% of samples; station history is `past_rate` (share of the station's
  results already back from the lab, smoothed with 10 'average' tests), not a
  frozen 2010-2022 rate; spill features removed (no effect); missing rain is
  left blank, never 0; signs are scored with a 1-day lag and also with all
  three state standards; live rain is scaled per cell to match ERA5
  (`processed/rain_scale.json`); scikit-learn pinned to 1.9.1.
- Codex review fixes (Oct 24, 2026, AI-assisted; `review/codex_review.md`):
  02 writes `processed/bacteria_extra.csv` (7.8 MB: beach coliforms + non-beach
  enterococcus), which 05 and 09 read, so the 76 MB `bacteria_la.csv` stays out
  of git; 08 builds rain windows from exact calendar days and stops if rain is
  missing; 08 and 10 stop with a message instead of retraining; every app file
  carries `model_id` (fingerprint of the model's test predictions).
- Known limitation (accepted): `past_rate` is smoothed toward the 2010-2022
  overall unsafe rate, computed once from all training years, so early training
  rows use a tiny bit of later information (about 6% of `past_rate` on average;
  1.2% of rows have < 10 known results). No 2023+ data is involved.

- Causes analyses (Oct 24, 2026, AI-assisted; results in `review/causes_results.md`):
  11 = beach diagnosis + 72-hour advisory check (no model); 12 = intervention
  evaluator using a weather-only expected risk (leave-one-year-out), DiD vs beaches
  5+ km away, month bootstrap widened x1.74 by placebo calibration. The math is in
  `pipeline/evaluator.py`, shared by 12 and app.py. Dylan's decisions (Oct 24):
  recovery uses all years, but the dry rate, after-rain rate and chronic label use
  the last 5 years; `hotspots.json` retired (08 and 09 read `diagnosis.json`; 11
  runs before 08 in rebuild.sh); verdicts say "cleaner after" / "no clear change
  after", never "because of". Oct 25: app name "After the Rain" (final); the
  advisory headline leads with the 7 slow beaches ("2 of them are also often unsafe
  in dry weather"); the 29 fast beaches are described as "recover faster than most"
  (never "longer than the data needs"), with their days 2-3 rate vs. normal; the
  line "Advisories are precautions; this is not advice to swim during one." stays
  on Advisory check; project dates are labeled "approximate, from news reports".

## App data contract (`app_data/`, all JSON)
Every file has a top-level `model_id`; `python3 pipeline/check_model_ids.py`
checks they all match `model/metrics.json`, and the app warns if they don't.
- `forecast.json`: from `python pipeline/08_forecast.py` (live; needs internet).
  `{issued, mode, rain_source, levels, standard, stations:[{station, name, lat,
  lon, type, last_test:{date,value,unsafe}, forecast:[{date, risk, level
  (low|caution|high), reasons:[str]}]}]}`, today + 3 days.
- `replay_<date>.json`: same shape for a past day, plus `forecast[].actual
  {value, unsafe}` when a sample was taken. `replay_2023-01-09.json` is a big
  storm (good for the demo video).
- `stations.json`: `{model_id, stations:[...]}`, all 311 stations incl. inland (`water`: ocean|freshwater,
  `in_model`, `on_map`, `note`, `unsafe_rate`, `type`, `last_test`). `on_map`
  is false for stormwater devices (BMP, catch basin, Oxford Basin) and inland
  stations with < 20 samples; every beach is on the map. 13 shoreline study
  sites (Cabrillo, Manhattan/Hermosa/Redondo; sampled 2015-2018) are shown as
  beaches graded on enterococcus, but only in 09: `in_model` false, no
  forecast, `note` says so; the model and metrics still treat them as inland.
  `history.json` has their full record.
- `history.json`: `{columns:[date,value,unsafe], stations:{code:[[...]]}}`, past 2 years.
- `type` in forecast/stations files = the beach's pattern from `diagnosis.json`:
  chronic | slow | typical | fast | clean, or null (too few tests).
- `spills.json`: big spills since 2023 vs nearby beach samples, actual vs predicted.
- `post_fire.json`: Palisades difference-in-differences on model residuals,
  `monthly` series for a chart. Result: no measurable overall change; after
  rain +8.6 points but the 95% range (-5.7 to +22) includes zero. Eaton: no nearby stations.
- `final_metrics.json`: from 10; every quoted number as [value, low, high] per scenario
  (`test`, `live`, `day3`), keys as in the report code (e.g. `sign_plus_rain.gain`).
- `diagnosis.json` (11): `{period, recent_period, definitions, recovery_counts,
  region:{curve, by_storm_size, recovery_groups:{slow|typical|fast:{beaches,
  normal_wet_season_rate, curves:{"all storms"|"1 in or more":[{day, unsafe_rate,
  tests}]}}}, after_rain_tides}, stations:[{station, dry_unsafe_rate,
  advisory_unsafe_rate, normal_wet_season_rate, groups{days 0-1|2-3|4-6|7-10},
  by_day, by_storm_size, trigger, recovery, chronic, fix_category, fix_text,
  also_after_rain, after_rain_tides}]}`.
- `interventions.json` (12 `--demo`): `{method, demos:[{label, beach, short, station,
  before, after, change_beyond_weather, ci95, verdict, monthly}], placebo,
  post_fire_weather_only}`.
- `evaluator_data.json` (12 `--demo`, ~2 MB): every beach test with its weather-only
  risk as parallel lists `s` (station index), `d` (days since `day0`), `y` (0/1),
  `e` (risk x 1000), plus `stations`, `widen`, `calibration`. The app runs
  `evaluator.evaluate()` on it.
- `burn_areas_2025.geojson`, `model_metrics.json` (copy of `model/metrics.json`:
  `baselines.{sign, sign_plus_rain, sign3, sign3_plus_rain}` each with `rule`
  and `model_same_budget`; `model_at_threshold`; `threshold_set_on_2022`).

## The app
`app.py` (Streamlit, AI-assisted; redesigned Oct 24-25, 2026 around causes for
agencies and policymakers): `streamlit run app.py`. Name "After the Rain"
(`APP_NAME` in app.py). Theme in
`.streamlit/config.toml`. Pages, in order (left menu): Home (landing: hero with space for assets/hero.jpg,
the problem, 3 findings from app_data, question cards, who it's for, trust strip), Advisory check,
Beach diagnosis, Did it work? (live evaluator), After the fires, Forecast (keeps
the Jan 2023 replay toggle), Methods (model-vs-signs, definitions, spills). Design
rules: each page opens with a SHORT headline (one idea, one number), details on the
line below, plain one-line notes for "normal level", "points" and "surprise"; then
annotated charts, then quiet number tiles; no emoji; chip legends; every chart
has a "Show the numbers" table. Styling (Oct 25, from Dylan's refs in `design_refs/`):
left menu in Harbor blue (st.navigation position="sidebar", icons); Gabarito headlines +
Public Sans body (Google Fonts via config.toml); spacing scale --s1..--s7 (4-48 px);
white panels (`panel()`), tinted tiles (`cards()`, 4 across, 2x2 at medium widths,
no decorative sun since Oct 25); one chart
theme (`after_rain`, horizontal grid only; draw charts with `show_chart()`); line names
at the line ends, no duplicate legend; Carto Positron maps zoomed to the coast;
one name per beach group everywhere (`GROUPS`: Unsafe longer than 72 hours, Back to
normal within 72 hours, Recover faster than most, Often unsafe in dry weather); forecast cards carry a risk bar. Station labels are "Place (CODE)"; SMB-MC-2 is named Surfrider Beach (`KNOWN_PLACES`). Methods reads `final_metrics.json`, so rerunning 10
updates it. Read only from `app_data/`; refresh the forecast by running
`08_forecast.py` then `09_export_app_data.py`. To retrain, always use
`bash pipeline/rebuild.sh` (06, 07, 11, 08 live and every replay, 09, 10, 12, then
the model-ID check), never 06 alone. Screenshots (phone 390 px, desktop 1440 px) in
`screenshots/`, made by `review/screenshots.py`.

## Running (use WSL, not Windows Python)
Windows Smart App Control blocks pandas/numpy DLLs on this laptop, so all
Python runs inside WSL (Ubuntu) on the same folder, /mnt/c/Users/dylan/GitHub/waterapp.
Virtual environment lives in the Linux home folder:
`source ~/waterenv/bin/activate` before any python/pip/streamlit command.
Use `python3` and forward slashes, e.g. `python3 pipeline/08_forecast.py`.
If the installed scikit-learn doesn't match the saved model, 08 and 10 stop with a
message: install requirements.txt or run `bash pipeline/rebuild.sh` (~3 min, needs internet).

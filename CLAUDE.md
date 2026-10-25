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
  `pipeline/10_final_metrics.py`). Test set: ROC-AUC 0.86; at the same number
  of warnings, catches 66% of unsafe days vs 61% for signs + 72-hour rain
  advisory (+5.0 points, 95% range +1.9 to +8.6), and 53% vs 46% for signs alone.
  Estimated live mode: 66% vs 61%. Day +3: 63% vs 61% (range includes zero).
- Decisions already made (ask Dylan before changing): beaches only (inland E.
  coli stations have too few samples to model); time-based split; last test
  result lagged 2 days for lab turnaround; rain features use days before the
  sample only; warning threshold = model risk 0.159 (same warning budget as
  signs + rain rule on the test years; a threshold set on 2022 alone gives
  similar results).
- Audit fixes (Oct 24, 2026, AI-assisted): date parsing in 01 no longer drops
  ~2% of samples; station history is `past_rate` (share of the station's
  results already back from the lab, smoothed with 10 'average' tests), not a
  frozen 2010-2022 rate; spill features removed (no effect); missing rain is
  left blank, never 0; signs are scored with a 1-day lag and also with all
  three state standards; live rain is scaled per cell to match ERA5
  (`processed/rain_scale.json`); scikit-learn pinned to 1.9.1.

## App data contract (`app_data/`, all JSON)
- `forecast.json`: from `python pipeline/08_forecast.py` (live; needs internet).
  `{issued, mode, rain_source, levels, standard, stations:[{station, name, lat,
  lon, type, last_test:{date,value,unsafe}, forecast:[{date, risk, level
  (low|caution|high), reasons:[str]}]}]}`, today + 3 days.
- `replay_<date>.json`: same shape for a past day, plus `forecast[].actual
  {value, unsafe}` when a sample was taken. `replay_2023-01-09.json` is a big
  storm (good for the demo video).
- `stations.json`: all 311 stations incl. inland (`water`: ocean|freshwater,
  `in_model`, `on_map`, `note`, `unsafe_rate`, `type`, `last_test`). `on_map`
  is false for stormwater devices (BMP, catch basin, Oxford Basin) and inland
  stations with < 20 samples; every beach is on the map. 13 shoreline study
  sites (Cabrillo, Manhattan/Hermosa/Redondo; sampled 2015-2018) are shown as
  beaches graded on enterococcus, but only in 09: `in_model` false, no
  forecast, `note` says so; the model and metrics still treat them as inland.
  `history.json` has their full record.
- `history.json`: `{columns:[date,value,unsafe], stations:{code:[[...]]}}`, past 2 years.
- `hotspots.json`: per beach `type` = chronic | rain-driven | usually clean,
  with dry/wet unsafe rates since Oct 2021.
- `spills.json`: big spills since 2023 vs nearby beach samples, actual vs predicted.
- `post_fire.json`: Palisades difference-in-differences on model residuals,
  `monthly` series for a chart. Result: no measurable overall change; after
  rain +7 points but the 95% range includes zero. Eaton: no nearby stations.
- `burn_areas_2025.geojson`, `model_metrics.json` (copy of `model/metrics.json`:
  `baselines.{sign, sign_plus_rain, sign3, sign3_plus_rain}` each with `rule`
  and `model_same_budget`; `model_at_threshold`; `threshold_set_on_2022`).

## Next: build the app
Framework: Dylan to choose (Streamlit = fastest; React + Leaflet = his usual
stack). Views: map of beaches colored by today's level with a 3-day forecast;
station page with history chart and plain-language reasons; Causes tab
(hotspots, spills); post-fire case study; "How it works" page with the
model-vs-signs comparison. Read only from `app_data/`; refresh it by running
`08_forecast.py` then `09_export_app_data.py`. After any retrain (06), rerun
07, 08 (live and replays), 09 and 10 so every app file uses the same model.

## Running (use WSL, not Windows Python)
Windows Smart App Control blocks pandas/numpy DLLs on this laptop, so all
Python runs inside WSL (Ubuntu) on the same folder, /mnt/c/Users/dylan/GitHub/waterapp.
Virtual environment lives in the Linux home folder:
`source ~/waterenv/bin/activate` before any python/pip/streamlit command.
Use `python3` and forward slashes, e.g. `python3 pipeline/08_forecast.py`.
The first forecast run retrains the model to match the local scikit-learn version (~15 s).

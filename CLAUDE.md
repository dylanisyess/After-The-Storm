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
  Results in `model/metrics.json`: ROC-AUC 0.85; at the same number of warning
  days, catches 51% of unsafe days vs 40% for posted signs, and 64% vs 59% for
  signs plus the 72-hour rain advisory.
- Decisions already made (ask Dylan before changing): beaches only (inland E.
  coli stations have too few samples to model); time-based split; last test
  result lagged 2 days for lab turnaround; rain features use days before the
  sample only; warning threshold = model risk 0.166 (same warning budget as
  signs + rain rule).

## App data contract (`app_data/`, all JSON)
- `forecast.json`: from `python pipeline/08_forecast.py` (live; needs internet).
  `{issued, mode, rain_source, levels, standard, stations:[{station, name, lat,
  lon, type, last_test:{date,value,unsafe}, forecast:[{date, risk, level
  (low|caution|high), reasons:[str]}]}]}`, today + 3 days.
- `replay_<date>.json`: same shape for a past day, plus `forecast[].actual
  {value, unsafe}` when a sample was taken. `replay_2023-01-09.json` is a big
  storm (good for the demo video).
- `stations.json`: all 291 stations incl. inland (`water`: ocean|freshwater,
  `in_model`, `unsafe_rate`, `type`, `last_test`).
- `history.json`: `{columns:[date,value,unsafe], stations:{code:[[...]]}}`, past 2 years.
- `hotspots.json`: per beach `type` = chronic | rain-driven | usually clean,
  with dry/wet unsafe rates since Oct 2021.
- `spills.json`: big spills since 2023 vs nearby beach samples, actual vs predicted.
- `post_fire.json`: Palisades difference-in-differences on model residuals,
  `monthly` series for a chart. Result: no measurable overall change; after
  rain +8 points but the 95% range includes zero. Eaton: no nearby stations.
- `burn_areas_2025.geojson`, `model_metrics.json`.

## Next: build the app
Framework: Dylan to choose (Streamlit = fastest; React + Leaflet = his usual
stack). Views: map of beaches colored by today's level with a 3-day forecast;
station page with history chart and plain-language reasons; Causes tab
(hotspots, spills); post-fire case study; "How it works" page with the
model-vs-signs comparison. Read only from `app_data/`; refresh it by running
`08_forecast.py` then `09_export_app_data.py`.

## Running (use WSL, not Windows Python)
Windows Smart App Control blocks pandas/numpy DLLs on this laptop, so all
Python runs inside WSL (Ubuntu) on the same folder, /mnt/c/Users/dylan/GitHub/waterapp.
Virtual environment lives in the Linux home folder:
`source ~/waterenv/bin/activate` before any python/pip/streamlit command.
Use `python3` and forward slashes, e.g. `python3 pipeline/08_forecast.py`.
The first forecast run retrains the model to match the local scikit-learn version (~15 s).

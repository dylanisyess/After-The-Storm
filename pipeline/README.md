# Water risk pipeline

Run in order from the `waterapp` folder (pip install pandas scikit-learn):

| Step | Script | Where it runs | Output |
| --- | --- | --- | --- |
| 1 | `01_filter_bacteria.py` | anywhere with pandas | `bacteria_la.csv` (LA-region enterococcus, E. coli, fecal and total coliform results, 2010+) |
| 2 | `02_stations.py` | anywhere with pandas | `stations.csv`, `labels.csv` (one standard per station, `exceeds` = 0/1) |
| 3 | `03_spills_fires.py` | anywhere with pandas | `spills_la.csv`, `burn_areas_2025.geojson` |
| 4 | `04_fetch_weather.py` | your computer (needs internet, no installs) | `rain_daily.csv`, `tides_hilo.csv`, `rain_forecast_model.csv`, `rain_scale.json` (live-rain correction) |
| 5 | `05_build_table.py` | anywhere with pandas | `training_table.csv` (one row per sample, all features) |
| 6 | `06_train.py` | needs scikit-learn | `model/beach_model.joblib`, `model/metrics.json`, `model/test_predictions.csv` |
| 7 | `07_analysis.py` | anywhere | `app_data/hotspots.json`, `spills.json`, `post_fire.json` |
| 8 | `08_forecast.py` | your computer (needs internet) | `app_data/forecast.json`; `--as-of YYYY-MM-DD` writes `app_data/replay_<date>.json` from saved data |
| 9 | `09_export_app_data.py` | anywhere | `app_data/stations.json`, `history.json`, map layer, metrics |
| 10 | `10_final_metrics.py` | needs scikit-learn | `review/final_metrics.md` (numbers to quote, with 95% ranges) |

Steps 1-5 write to `processed/`.

Standards: enterococcus > 104 MPN/100 mL at beaches (BeachWatch stations);
E. coli > 320 cfu/100 mL at inland stations.

Model: beach stations only (inland stations have too few samples). Trained on
2010-2022, tested on 2023 onward. See `model/metrics.json` and
`review/final_metrics.md` for results. Use scikit-learn 1.9.1
(`requirements.txt`) so the saved model loads without retraining.

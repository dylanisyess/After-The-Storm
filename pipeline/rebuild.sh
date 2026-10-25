#!/usr/bin/env bash
# Retrain the model and rebuild EVERY file made from it (steps 06-10), so all app
# files share one model ID. Run inside WSL from the waterapp folder:
#   source ~/waterenv/bin/activate && bash pipeline/rebuild.sh
# Needs internet (the live forecast). Takes a few minutes.
set -e                   # stop at the first step that fails
cd "$(dirname "$0")/.."  # run from the waterapp folder wherever it's called from

echo "06 train";            python3 pipeline/06_train.py > /dev/null   # prints a lot; results are in model/metrics.json
echo "07 analysis";         python3 pipeline/07_analysis.py > /dev/null
echo "08 live forecast";    python3 pipeline/08_forecast.py
for f in app_data/replay_*.json; do   # redo every replay that exists, e.g. replay_2023-01-09.json
    day="${f#app_data/replay_}"; day="${day%.json}"
    echo "08 replay $day";  python3 pipeline/08_forecast.py --as-of "$day"
done
echo "09 export";           python3 pipeline/09_export_app_data.py
echo "10 final metrics";    python3 pipeline/10_final_metrics.py
python3 pipeline/check_model_ids.py

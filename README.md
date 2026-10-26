# After the Storm

**Which LA and Ventura County beaches stay unsafe after rain, why, and whether cleanup projects worked**, built from 16 years of California State Water Board beach tests.

Built by Dylan Lin for the 2026 Congressional App Challenge (CA-28).

## What it does

Search any beach, then choose what you want to know:

| Page | Question it answers |
| --- | --- |
| **Forecast** | How likely is a water test to be over the safe limit today and over the next 3 days, and what's driving that estimate? |
| **After-rain patterns** | Is LA County's 72-hour rain advisory long enough at every beach? |
| **Beach profile** | Is a beach unsafe mainly after rain or even in dry weather, how fast does it recover, and what does its official posting history show? |
| **Compare beaches** | How do 2 or 3 beaches stack up side by side? |
| **Cleanup projects** | Was a beach cleaner after a cleanup project, once weather and other beaches are accounted for? |
| **Fire impacts** | Did bacteria change at beaches below the 2025 Palisades burn area? |
| **Methods** | How every number was made, how it was tested, and its limits. |

"Unsafe" means a test over 104 enterococcus bacteria per 100 mL, California's standard for ocean beaches.

## How it works

- **Data:** California State Water Boards beach bacteria tests (2010-2026), Beach Watch posting and closure records, Open-Meteo rainfall, NOAA tide predictions, CIWQS sewage spill reports, and CAL FIRE burn perimeters.
- **Forecast model:** gradient-boosted decision trees (scikit-learn), trained on 2010-2022 and tested only on 2023-2026. Inputs: rain over the past day, 3 days, week and month; days since rain; tides; season; each beach's long-term and past-year track record; and the latest test result once it would be public (2 days after sampling).
- **Testing:** with the same number of warnings, the forecast is compared with warning signs plus LA County's rain advisory, and separately with what counties actually posted. Full results and uncertainty ranges are on the Methods page.
- **Cleanup checks:** observed minus weather-expected unsafe rates, before vs. after a project, compared with beaches 5+ km away, with ranges calibrated on 300 random placebo dates.

## Run it locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The data pipeline that builds `processed/`, `model/` and `app_data/` is described in [`pipeline/README.md`](pipeline/README.md). A GitHub Actions workflow refreshes the forecast daily.

## AI use

This project was built with help from AI coding assistants (Claude Code and Codex) and Claude. What was AI-assisted is listed in [`AI_USE_LOG.md`](AI_USE_LOG.md).

## Credits

- Data: California State Water Resources Control Board, Open-Meteo (ERA5), NOAA CO-OPS, CAL FIRE.
- Spring-tide effect: Boehm, A. B., & Weisberg, S. B. (2005), *Environmental Science & Technology* 39: 5575-5583.
- Home photo: Venti Views on Unsplash.

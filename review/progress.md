# Progress log

Phase 1 (Tiers 1 and 2): done and committed before this log was started.

## Phase 2 (Oct 25, 2026, evening; AI-assisted with Claude Code)

### 1. Forecast day selector + phone map zoom (done)
- Forecast page has a "Day" picker (Today, Mon 26, ...). The headline count, the level chips,
  the map colors, the beach card and "What's driving this risk" all follow the picked day; the
  day cards outline the picked day. Days after today say "N days ahead, so less sure than today's".
- Phone map: Streamlit can't measure the screen, so `is_phone()` reads the browser's User-Agent
  ("Mobile", "iPhone", "Android"). On phones maps open at zoom 7.35 (`PHONE_ZOOM`), which fits
  Rincon to Long Beach (and Avalon) in a 360 px map. Fire map uses 8.0 on phones.
- Checked: phone + desktop load with no errors; clicking "Tue 27" changed the headline (12 -> 6
  beaches), card, picked day card and drivers together.

### 2. Real posting records (done)
- Source: data.ca.gov/dataset/beach-water-quality-postings-and-closures (lab.data.ca.gov didn't respond;
  same dataset). Raw files in `beachwatch/` (gitignored). Records end **March 6, 2026** (file updated Oct 1).
- Dylan approved the fields and matching plan (Station_Name / HistoricalStationName / AgencyStationIdentifier;
  129 of 132 forecast beaches match; DPH 125, DPH 126, SMB-1-02 don't).
- New `pipeline/13_postings.py` -> `app_data/postings.json` (history per station, county per station,
  the check). Added to rebuild.sh (after 10), pipeline/README.md, CLAUDE.md.
- (a) Beach profile: "Official posting history (records through March 6, 2026)" panel: days under
  advisory per year (signs / closures / rain), latest 15 advisories, Show the numbers, and a clear
  no-records message.
- (b) Methods: "Checked against real postings" expander, separate from the headline numbers.
  Result shown to Dylan first: 2023 to Mar 6 2026, 24,872 tested beach-days, 131 beaches. Same number of
  warnings (15.5% of days): real postings caught 34.8% of unsafe days, the forecast 54.4%, +19.6 points
  (95% range +15.2 to +24.9). With recorded rain advisories: +19.4 (+15.0 to +24.5). Dylan chose the strict
  rule (posted before the test day); counting same-day signs gives about +15.
- Checked: all pages, phone + desktop, no errors, no sideways scroll; model-ID check passes.

### Feature freeze (midnight)
Reached 12:26 AM while waiting on the item-2 questions, so items 3-7 were NOT started:
3 evidence/reliability + calibration chart, 4 exports, 5 county filter, 6 compare beaches,
7 GitHub Actions daily update. (Step 13 already stores each station's county, for item 5 later.)
Screenshots retaken (phone screenshots now send an iPhone User-Agent so the phone map zoom shows).

## Phase 2b (freeze lifted until 2:30 AM, Oct 26)

### Fix: county-wide rain advisories (done, 12:55 AM)
- Beach Watch DOES include the general rain advisories, but each health department files them under one
  placeholder station (`All_LosAngeles_County_Beaches` 165 records, `All_Long_Beach_City_Beaches` 89,
  `All_Ventura_County_Beaches` 112). Step 13 had dropped them as unmatched, so the earlier "+ rain" row
  (+19.4) was wrong. 13 now copies each county-wide record to every station of that department (and limits
  the check to beaches Beach Watch lists by name, as before).
- New results: signs counties actually posted (+ closures) 36.2% vs forecast 55.6%, +19.4 (+15.2 to +24.6);
  those signs + the counties' rain advisories 64.5% vs 71.1%, +6.5 (+3.6 to +9.8).
- Methods section renamed "Compared with what counties actually posted", now a two-row table; it says the
  second row is the fair match for the headline (+4.7), which is unchanged. Beach profile caption updated.

### 3. Exports (done, ~1:00 AM)
- `csv_button()` (app.py): a small "Download CSV" button (doesn't rerun the page). `numbers()` now takes a
  file name and puts the button under every chart's "Show the numbers" table: after-rain curves, a beach's
  days-since-rain bars, its posting history per year, the cleanup project's monthly surprise, the fire chart.
- Extra buttons for the main tables: all beaches' patterns (After-rain patterns and Beach profile, with test
  counts), the picked day's forecast for every beach (Forecast, under the map), a beach's past tests, and the
  Methods comparison chart.
- Cleanup projects: "Download a one-page report" (`project_report()`): an HTML page with the verdict, work
  dates, before/after windows and test counts, comparison beaches, rates vs. weather-only, change, 95% range,
  assumptions, sources (news links per project from `PROJECT_SOURCES`) and the model ID. Prints to exactly one
  Letter page (checked by printing to PDF).
- Checked: clicked 5 download buttons in a headless browser and opened the files; all pages pass at phone and
  desktop widths.

### 4. Calibration and reliability (done, ~1:05 AM)
- New `pipeline/14_calibration.py` -> `app_data/calibration.json` (reads the saved test predictions, nothing
  retrained; in rebuild.sh after 13). Methods panel "Does the risk mean what it says?": predicted risk vs. share
  actually unsafe, 10 groups, dot size = tests, 95% Wilson ranges, Show the numbers + CSV.
  Result (2023-2026, 30,262 tests): the forecast averaged 11.5% and 14.0% were unsafe, so the risk runs a little
  low, about 3 to 6 points in the 5-50% groups (the page says so, with the gap computed from the data).
- Sample counts / not-enough-data: `MIN_TESTS = 15` (step 11's group minimum); `rate_n()` shows
  "41% (34 tests)" or "Not enough tests (6)". Used in the After-rain slow-beach table; Beach profile tiles show
  "Too few tests" under 15 (`enough()`); storm-size bars go pale under 8 tests (like the days-since-rain chart).
  Forecast card adds "Track record: N tests since YEAR" and, when the latest test is over 30 days old (5 beaches
  today, e.g. Will Rogers: 272 days), "Latest test is N days old, so this estimate leans on rain, tides and the
  beach's track record". Cleanup projects headline adds "Based on N tests before and M after".
- Checked: all pages, phone + desktop, no errors; the stale warning and table counts render.

### 5. Compare beaches (done, ~1:08 AM)
- New page "Compare beaches" (/compare, after Beach profile in the menu; `compare_page()`). Pick 2 or 3
  forecast beaches (starts with the beach picked on Forecast/Beach profile, or the riskiest today, plus its 2
  nearest). Headline: the riskiest of them today and how far apart they are (up to N km).
- One tile per beach (today's risk, level color); a side-by-side table (one block per measure on phones): risk
  today, latest test and its age ("over 30 days old" flag), after-rain group, unsafe on days 2-3 and 4-6 after
  rain, normal level, dry-weather rate (all with test counts / "Not enough tests"), all tests, distance to the
  others (`km_apart()`, haversine). CSV download; a small map zoomed to fit the picked beaches.
- Checked: phone + desktop, no errors; removing beaches down to 2 still compares, down to 1 asks for 2 or 3.
- Not added to the Home question cards (6 cards in a 3 x 2 grid); say if you want it there.

### 6. Daily GitHub Actions update (done, ~1:10 AM; started before 2:00)
- `.github/workflows/daily-forecast.yml`: runs at 17:00 and 18:00 UTC; a first step lets only the run at
  10 AM or later LA time, on a day with no forecast yet, go ahead (summer: the 17:00 run; winter: 18:00), so it
  runs once a day year-round even if GitHub starts late. "Run workflow" button also works.
  Steps: install requirements.txt (Python 3.14, scikit-learn 1.9.1) -> `08_forecast.py` ->
  new `pipeline/validate_app_data.py` (all 14 app files present, valid JSON, forecast.json is today's LA date
  with today + 3 days for every beach) -> `check_model_ids.py` -> commit and push app_data/forecast.json only.
  Any failing step stops the run before the commit.
- Inputs: 08 reads model/beach_model.joblib, processed/{stations,labels,rain_daily,tides_hilo,training_table}.csv,
  processed/rain_scale.json and app_data/diagnosis.json; all in git (no LFS). Tested by cloning the last
  commit into a scratch folder and running 08 there: it worked and changed only forecast.json. Then
  validation + model-ID check passed, and the commit step committed only forecast.json. Gate tested
  (9 AM skip, 10 AM go, already-done skip). YAML parses.
- NOT tested on GitHub: this repo has no GitHub remote yet. It needs the repo pushed, and the 4 new files
  committed (validate_app_data.py, postings.json, calibration.json, the workflow), or validation will
  stop at "missing: postings.json".

### Wrap-up (1:20 AM)
All Phase 2b items done before the 2:30 AM stop. Final pass: all 8 pages load with no errors and no sideways
scroll at phone (390 px) and desktop (1440 px); model-ID check and validate_app_data.py pass. Screenshots
retaken (Compare is now 3_, so later pages moved up one number; the old numbered files were replaced).
Nothing committed. Headline numbers unchanged.

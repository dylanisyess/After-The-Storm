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

## Home beach search rebuild (Oct 26)
- Bug report: picking a beach in the Home search cleared the box and stayed on Home. I couldn't reproduce it
  in a headless browser on any of the three running servers (8501, 8599, 8600), at desktop or phone width.
  The old code jumped pages from inside the search box's change (`st.switch_page`), which can lose a race with
  another rerun in a real browser. The new flow never does that: the search only shows step 2, and each answer
  is a plain link (`st.page_link(..., query_params={"beach": CODE})`).
- Found and fixed a real ?beach= bug: `chosen_beach()` let a picker's old value beat the link, so a second link
  in the same session could open the previous beach. New `link_beach(page)` applies each new link once;
  `remember_beach(code, page)` marks the beach the page shows, so it isn't mistaken for a new link.
- Home step 2 (`beach_questions()`): one line (today's level and risk, or "No forecast: <reason>"; plus its
  pattern from GROUPS, e.g. "Often unsafe in dry weather"), then answer links shown only when the page has data:
  Forecast (always; "What do its past tests show?" if no forecast), Beach profile (diagnosed beaches), Compare
  (forecast beaches; the 2 nearest are prefilled), Cleanup projects (100+ tests; its known project if it has one,
  otherwise "No known cleanup project at X yet" until real dates are entered, so no made-up dates get a verdict),
  Fire impacts (within 1 km of the Palisades burn outline, same measure as step 07; `km_to_burn()`).
- Every destination reads ?beach=: Forecast and Beach profile (picker), Compare (beach + 2 nearest), Cleanup
  projects (preset or custom view), Fire impacts (a note on where the beach fits, and a ring on the map).
- "Explore the findings" -> "See what we found", a link that scrolls to the findings on Home (#what-we-found).
  Question cards: 7, incl. "How does a beach compare with its neighbors?" (4 then 3 across; 1 per row on phones).
- Tested: Santa Monica Pier (SMB-3-3), Malibu Beach DHS001 (no forecast, below the burn) and Solimar Beach
  (10000) at phone and desktop widths: search stays on Home with the summary, and every answer link opens its
  page on that beach (desktop returns Home through the menu, so it's one session). 0 failures. All pages
  pass; screenshots retaken.

## Phase 3 (Oct 26, 2:17-2:47 AM; stop was 3:15)
1. Freshness: `todays(s)` and `day_name(date, today, replay)`. "Today" only ever labels today's LA date. If the
   file has no today: Forecast headline "No forecast for today. Latest: <date>", Compare states its date
   ("on Thursday, October 22") and adds the same note, Home says "No forecast for today (latest: ...)".
   Tested by temporarily shifting the file's dates 4 days back (real file restored; validator passes).
2. Validator (`pipeline/validate_app_data.py`): exactly 4 consecutive dates from today, unique station IDs,
   risk and typical_risk within 0-1, levels match the thresholds (08's rule; +-0.0005 for rounding). Tested on a
   copy with 4 planted errors: all caught; the real file passes.
3. Mismatch warning (`past_year()`): past-365-day unsafe share from history.json, 20+ tests, more than twice
   the estimate -> "This beach failed X% of tests in the past year (N tests), more than this estimate suggests."
   Santa Monica Pier: 44% of 230 tests vs 18% estimate (9% under the old model). 11 beaches today.
4. Driver panel: "How the estimate changes with different inputs"; lines "With typical tides, the estimate
   would be about 5% (now 9%)"; caption says each line changes one input group on its own. Methods text updated.
5. "Rarely unsafe, or too few tests" split into "Historically low failure rate" (15 beaches: little change
   after rain) and "Not enough tests to classify" (58: too few tests, or ranges too wide). Map, legends,
   tables, sorting notes.
6. Beach profile: notes say the normal level uses all years since 2010 (rainy season) and dry-weather rates use
   tests since 2021 (all seasons). New "Unsafe share by year" chart (every test, from evaluator_data.json; pale
   under 10 tests; numbers + CSV). Santa Monica Pier: ~10% (2016-18) -> 57% (2025).
7. Retrain with `recent_rate` (started 2:23, before 2:40): past-365-day unsafe share among results back from the
   lab (2-day lag; 5-day for day +3), blank under 10 results; in 05 (`recent_rate()`), 06 (FEATURES), 08, 10.
   Same split. Old model backed up first. `rebuild.sh` + model-ID check passed at 2:28 (model e5207cd9bd).
   Before -> after (final_metrics, 95% ranges): test +4.7 (+1.8 to +8.2) -> +7.0 (+4.0 to +10.0);
   live +3.6 -> +5.6 (+2.4 to +8.7); day +3 +0.5 (-3.1 to +5.5) -> +4.4 (+1.0 to +8.1); AUC 0.859 -> 0.868.
   All improved, so KEPT. Caution threshold 0.161 -> 0.195 (same warning budget).
8. Forecast card: "Compare nearby beaches" link (?beach=). Stations sharing a beach name are labeled
   "(monitoring site k/n, CODE)" in pickers and tables (code order; on-map stations only).
- Checked: all pages at phone + desktop, no errors; Home flow test (3 beaches x 2 widths) 0 failures;
  validator + model-ID check pass. Screenshots retaken (the replay desktop shot is cut off at the bottom:
  the script measured the page before it finished drawing; the page itself is fine).

# Overnight summary (Oct 24-25, 2026)

*Work done with AI assistance (Claude Code); logged in AI_USE_LOG.md. Nothing was committed.*

## Update, Oct 25: your decisions applied
- **Name:** "After the Rain" kept.
- **Headline (Advisory check):** now leads with the slow beaches: "**7 beaches** stay unsafe
  longer than LA County's 72-hour rain advisory (2 of them are also often unsafe in dry weather)."
  The support line gives:
  - the slow beaches' 29% on days 4-6 vs. 10% normal, after storms of 1 inch or more
  - "Another 29 beaches recover faster than most: 5% of tests 2 to 3 days after rain were
    unsafe, vs. 4% normally (8% after storms of 1 inch or more)."

  The cards say "stay unsafe longer than 72 hours" and "recover faster than most", with the
  days 2-3 rate next to the normal rate. The slow-beach table has a new "Often unsafe in dry
  weather" column, which marks Pico-Kenter and SMB-2-7.
- **Precaution line**: "Advisories are precautions; this is not advice to swim during one."
  It sits under the headline on Advisory check. It also appears on Beach diagnosis whenever a
  beach's finding says it is "back to normal" after rain.
- **"Longer than the data needs" is gone everywhere.** The fix text for fast beaches in
  `diagnosis.json` now says "This beach recovers faster than most after rain" (step 11 rerun;
  model IDs still match).
- **Project dates:** the Did it work? page shows "Project dates: approximate, from news
  reports." under the project picker whenever a known project is selected, and in the sources list.
- `review/causes_results.md` updated to match. Every page was retested with no errors, and
  all 14 screenshots were retaken.

Decisions 1, 2 and 3 below are settled; 4 and 5 still stand.

## Status: done, tested, screenshots taken
- **Rebuild passed.** `bash pipeline/rebuild.sh` ran cleanly, and every app file
  carries model ID `40ebd86751`, the same model as before (the retrain gives the
  identical model).
- **Every page tested** in a real browser (headless Chromium, driven by Playwright),
  with no errors or warnings. The test also tried:
  - the storm-size switch on Advisory check
  - six different beaches on Beach diagnosis, covering every pattern and the not-enough-tests case
  - all three known projects on Did it work?, plus a custom beach and a 3-year window
  - the replay switch on Forecast
- **Screenshots** of all six pages plus the forecast replay, at phone (390 px) and desktop
  (1440 px) widths, are in `screenshots/` (14 files). To retake them, see `review/screenshots.py`.

## Bugs found and fixed while testing
1. Missing fix categories were written as `NaN` (not valid JSON) into `diagnosis.json`,
   and from there into `forecast.json`, `stations.json` and the replays. 11 now writes `null`.
2. Test counts in `diagnosis.json` were saved as text ("41"), not numbers, which crashed
   Beach diagnosis. 11 now writes plain numbers.
3. Layout:
   - The app name was hidden under the top bar.
   - The beach picker, the evaluator controls and the replay switch came before the finding sentence.
   - The line labels on the advisory chart piled up on top of each other.
   - The day axis read "0.0, 1.0".
   - The evaluator sentence rounded 6.5 to "6" while its card said 6.5.
   All fixed. Every page now opens with its finding.

## What I built
1. **Your two decisions**
   - `11_beach_diagnosis.py`: the dry-weather rate, the advisory-period rate and the chronic
     label use the last 5 years (Oct 2021 to Oct 2026); recovery after rain still uses all
     years. **20 beaches are chronic** (19 before). New: four Long Beach stations (10-18% of dry
     tests), Pico-Kenter (12%) and SMB-2-7 (14%). Surfrider is still chronic (12.5%).
     Santa Monica Pier is the worst at 43%.
   - `hotspots.json` retired: 07 no longer makes it (file deleted), and 08 and 09 read
     `diagnosis.json`. The "often unsafe even in dry weather" forecast reason now uses the
     5-year chronic label. Station `type` is now chronic | slow | typical | fast | clean, or
     empty when there are too few tests. `rebuild.sh` runs 11 right after 07 and runs 12 at the end.
2. **Evaluator shared with the app.** The math is in `pipeline/evaluator.py`, used by both
   `12_intervention.py` and `app.py`. Step 12 also writes `app_data/evaluator_data.json`
   (2 MB: every beach test with its weather-only risk), so the app checks any beach and dates
   live, in under a second. Verdicts say "cleaner after", "dirtier after" or "no clear change
   after", never "because of".
3. **App redesign** (`app.py`), pages in your order, with a top navigation bar:
   1. **Advisory check**: "LA County's 72-hour rain advisory is too short for **7 beaches**,
      and longer than the data needs at **29**." The page has:
      - cards for each group, plus the big-tide rate (20% vs 11%)
      - recovery curves for the three groups, with the advisory shaded, each group's normal as a dashed line, and an "All storms / 1 inch or more" switch
      - a map colored by group
      - a table of the 7 slow beaches, and the sorting rules
   2. **Beach diagnosis**: pick a beach (it starts on Pico-Kenter, the slowest). It shows:
      - a one-sentence finding written for that beach
      - cards: dry-weather rate, advisory rate, rain trigger, recovery group
      - bars by days since rain, with the advisory band and the beach's normal
      - "what the pattern points to" (a clue, not proof)
      - storm-size bars, and a table of all beaches
   3. **Did it work?**: three known projects preloaded, or any beach and dates. "Surfrider
      Beach was cleaner after the Malibu Lagoon restoration (2012-13): **23.3 points** fewer
      unsafe tests than weather and other beaches predict (95% range 6.8 to 39.8)." It also has:
      - before and after cards
      - a monthly surprise chart, with the work period shaded and the before/after averages labeled
      - the 95% range drawn against "No change", with the placebo-check explanation
   4. **After the fires**: same result, findings first, plus the weather-only cross-check.
   5. **Forecast**: same logic, replay switch kept, minimal bordered cards. "2 of 132 beaches
      are at caution or high risk today."
   6. **Methods**:
      - model vs. signs chart, and forecast details
      - definitions for the new analyses
      - the sewage-spill check (moved here from the old Causes tab)
      - limits, data sources, and AI use
   Design rules followed:
   - one finding sentence per page, with its number in bold
   - no emoji (the tab icon is a plain material "water" symbol)
   - 1 px bordered cards
   - colors: the first three slots of the reference chart palette (blue, orange, aqua), plus navy and grey
   - dot-and-word legends
   - a "Show the numbers" table under each chart
   I couldn't run the palette checker (Node isn't installed), but these three slots are the
   palette's pre-checked colorblind-safe set.
4. `review/causes_results.md` updated with the 5-year numbers.

## Needs your decision
1. **App name.** I'm using **After the Rain** for now (`APP_NAME` near the top of app.py). Options:
   - **After the Rain** (my pick): names the headline finding, what happens in the days
     after rain, and still fits the fire and forecast pages.
   - **Shoreline Check**: plain and official-sounding; fits the "did it work?" idea.
   - **Runoff Report**: says what most of the problem is; reads like a report for agencies.
2. **Pico-Kenter and SMB-2-7 are now both chronic and slow.** The headline counts them
   among the 7 beaches still unsafe after 72 hours, and their diagnosis shows both
   explanations. Keep them in that count, or report "5 + 2 chronic"?
3. **Project dates.** The evaluator's three examples use dates from news articles. City or
   Heal the Bay records would be stronger before you submit.
4. **The screenshot setup is a workaround.** Headless Chromium needed four system
   libraries that require sudo to install. I unpacked them into a temporary folder for this
   run. On your machine, run `sudo python3 -m playwright install-deps chromium` once
   (the note is in `review/screenshots.py`). Playwright isn't in `requirements.txt`; it is
   only needed for screenshots.
5. **Small things you may want to change:**
   - On phones, the "Normal: 14%" label on the beach chart can touch a bar.
   - The slow-beach table scrolls sideways on phones.
   - The top bar turns into a menu button on phones (Streamlit's default).

## Files changed (not committed)
- **App:** `app.py`, `.streamlit/config.toml`
- **Pipeline:** `pipeline/07, 08, 09, 11, 12`, the new `pipeline/evaluator.py`, `rebuild.sh`, `README.md`
- **App data:** regenerated `app_data/*`, new `evaluator_data.json`, deleted `hotspots.json`
- **Review:** `review/causes_results.md`, `review/screenshots.py`, this file
- **Other:** `screenshots/`, `CLAUDE.md`, `AI_USE_LOG.md`

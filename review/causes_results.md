# Causes analyses: beach diagnosis, 72-hour advisory check, intervention evaluator

*Done October 24-25, 2026, with AI assistance (Claude Code); see AI_USE_LOG.md.
Updated Oct 25 after Dylan's decisions: dry-weather rates and the chronic label from the last 5 years; a headline
that leads with the 7 slow beaches; the precaution line; project dates labeled approximate.*
Made by `pipeline/11_beach_diagnosis.py` (`app_data/diagnosis.json`) and
`pipeline/12_intervention.py` (`app_data/interventions.json`), shown in the app's
Advisory check, Beach diagnosis and Did it work? pages. Data: 126,006 beach tests at
160 beaches, January 2010 to October 2026, with daily ERA5 rain.

**Two time windows.** Recovery after rain uses **all years** (it needs many storms).
The dry-weather rate, the rate during rain advisories and the **chronic** label use
the **last 5 years** (Oct 2021 to Oct 2026), so a beach that was fixed isn't still
called chronic.

**How days are counted.** Rain is a daily total, so "day 0" means 0.1 in or more fell
on the test day, and "day 3" means the last rain day was 3 days earlier. Since we
don't know the hour the rain stopped, a day-3 test was taken about 48-96 hours
after, so it straddles the 72-hour line. Days 0-2 are inside the advisory; day 4
and later are after it ends.

**"Normal"** for a beach = its unsafe rate in the wet season (November-April) on
days with no 0.1 in rain for 8+ days (all years). Region-wide that is 10.1%.

## 1. The 72-hour advisory check

**Headline: 7 beaches stay unsafe longer than LA County's 72-hour rain advisory (2 of
them are also often unsafe in dry weather).** After storms of 1 inch or more, they were
still unsafe on 29% of tests 4 to 6 days later (their normal: 10%). Another 29 beaches
recover faster than most: 5% of tests 2 to 3 days after rain were unsafe, vs. 4% normally
(8% after storms of 1 inch or more).

*Advisories are precautions; this is not advice to swim during one.*

Unsafe rate by days since rain, all beaches, wet season (normal = 10%):

| Storm size | day 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8-10 |
|---|---|---|---|---|---|---|---|---|---|
| 0.1-0.5 in | 31% | 29% | 17% | 11% | 9% | 14% | 8% | 7% | 9-14% |
| 0.5-1 in | 58% | 45% | 21% | 12% | 12% | 11% | 14% | 7% | 7-10% |
| 1 in or more | 67% | 61% | 37% | 24% | 18% | 15% | 15% | 14% | 12-14% |

After small storms the region as a whole is back to normal by about day 3. After storms of 1 inch or
more it is still above normal a week later.

Beach by beach (rules below):

| Group | Beaches | What the data shows |
|---|---|---|
| **Stay unsafe longer than 72 hours** | **7** | Clearly above normal on days 4-6. After 1 in+ storms: 29% unsafe on days 4-6 and 19% on days 7-10 (their normal: 10%). 2 of them are also often unsafe in dry weather |
| Back to normal within 72 hours | 17 | Still clearly unsafe on days 2-3, normal by days 4-6. Mostly Long Beach (LA River mouth). After 1 in+ storms: 11% on days 4-6 (normal 6%) |
| **Recover faster than most** (back to normal in about a day) | **29** | Days 2-3 and 4-6 within 5 points of normal. Days 2-3: 5% unsafe vs. 4% normally (8% after 1 in+ storms). Dockweiler, Hermosa, Venice, Santa Monica State Beach, Will Rogers, Silverstrand, most of Malibu |
| Little change after rain | 15 | Rarely unsafe at all (Ventura and Palos Verdes coves) |
| Unclear | 26 | Too few tests to separate the groups with 95% confidence |
| Not enough tests | 32 | Fewer than 15 tests in a group (Avalon, many Ventura beaches) |
| *Chronic (not in the fast or 72-hour groups)* | *20* | *Unsafe on 10%+ of dry days since 2021. For them "back to normal" means back to bad* |

The 7 beaches that stay unsafe longer than 72 hours:

| Beach | Days 4-6 | Normal | Days 3-7, big tide vs. other days |
|---|---|---|---|
| Pico-Kenter outfall, Santa Monica (SMB-3-4), also chronic | 22% | 14% | 43% vs 18% |
| SMB-2-7 (near Santa Monica Canyon), also chronic | 21% | 13% | 31% vs 22% |
| San Buenaventura State Beach (19000) | 16% | 3% | 35% vs 10% |
| Malibu (DHS007B) | 14% | 2% | 27% vs 10% |
| SMB-2-10 (Playa del Rey, near Ballona Creek) | 11% | 3% | 21% vs 6% |
| Solimar Beach (10000) | 11% | 3% | 24% vs 5% |
| Promenade Park, Ventura (14000) | 10% | 2% | 36% vs 3% |

All seven sit next to a big storm drain, creek or river mouth (for the SMB codes the
location is inferred from coordinates). After rain, they fail far more often on big
spring-tide days (range 6.5 ft or more). Region-wide, on days 3-7 after rain: 20% on
big-tide days vs. 11% on other days.

**What this supports, stated cautiously:** one fixed 72-hour rule is too short for a few
beaches next to large drains and river mouths, especially after storms of 1 inch or more
and on spring tides. About two dozen open-coast beaches recover faster than most. A
rule that depends on storm size and on the beach would match the data better.
This is a pattern in 16 years of tests, not a health study. The faster beaches still
had a little leftover risk after the biggest storms (8% vs. 4%). Advisories are
precautions; this is not advice to swim during one.

**Rules** (fixed before looking at beach names):
- "Clearly above normal" means the 95% range of (rate - normal) is entirely above zero. The range comes from a bootstrap that resamples whole storms, because tests after the same storm aren't independent.
- Each group of days needs 15+ tests.
- "Within 5 points of normal" also requires that the rate is not clearly above normal.

## 2. Beach diagnosis

For every beach, `diagnosis.json` has:
- the dry-weather and advisory-period unsafe rates (last 5 years)
- the unsafe rate on each of days 0-10 after rain
- rates by storm size
- the **trigger**: the smallest storm size where 1 in 4 tests on days 0-1 were unsafe, and every bigger size too
- the recovery group above, and the big-tide check
- a **fix category**, cautiously worded

| Fix category | Beaches | Pattern | Points to (not proof) |
|---|---|---|---|
| chronic | 20 | Unsafe on 10%+ of dry days since 2021 | A source that flows without rain (dry-weather drain flow, a creek or lagoon mouth, birds, sewer leaks). First step: source tracking (e.g., human-sewage DNA markers) |
| slow | 5 (+2 chronic) | Unsafe past 72 hours | A large area that keeps draining after the storm. Matching fixes: upstream capture or infiltration, and a longer advisory |
| typical | 17 | Clears within 72 hours | Storm runoff. Matching fix: capture it before it reaches the beach |
| fast | 29 | Clears in about a day | The first flush from nearby drains. Matching fix: first-flush capture. Recovers faster than most |
| clean | 15 | Little change | No fix indicated |

**Triggers:** at 25 beaches, 0.1-0.25 in of rain already makes 1 in 4 tests fail (Long
Beach, Santa Monica Pier and Pico-Kenter, Manhattan, Hermosa). At 33 beaches it takes 1 in or
more. At 6, 1 in 4 tests fail even without rain.

**Chronic beaches, last 5 years:**

| Area | Station | Unsafe share of dry-weather tests, since Oct 2021 |
|---|---|---|
| Santa Monica Pier | SMB-3-3 | 43% |
| Mother's Beach, Marina del Rey | MDRH-1/2/3 | 23-39% |
| Cabrillo Beach | CB-02 | 35% |
| Cabrillo Beach | CB-01 | 11% |
| Long Beach | B-9, B-64, B-11, B-56 | 10-18% |
| Surfrider Beach | SMB-MC-2 | 12.5% |

Six Santa Monica Bay stations without place names also qualify, along with Pico-Kenter and SMB-2-7.

Most of these are enclosed or semi-enclosed water. Changes from the all-years version:
- Four Long Beach stations, Pico-Kenter and SMB-2-7 are newly chronic: they were dirtier in dry weather over the last 5 years than over 2010-2026 as a whole.
- Surfrider is still chronic, even though it is much cleaner than before 2013 (see section 4).

## 3. The "delayed peak" puzzle

This doesn't change the conclusion in `review/forecast_puzzle.md`: no beach has a real
delayed peak. At none of them is the rate on days 2-3 clearly above days 0-1. At the
three Ventura puzzle beaches pooled, days 4-6 minus days 2-3 is +3 points (95% range -5 to +12).

What this analysis adds:
- Solimar and Promenade Park are two of the seven beaches that stay unsafe longer than 72 hours.
- At those beaches, the failures after storms come on big-tide days. On days 3-7 after rain at the three Ventura beaches, 32% of tests were unsafe with a big tide vs. 6% without.
- So a forecast that rises a few days after a storm when a spring tide arrives is right for those beaches. The fixed model's 29-33% on the Oct 26 spring tide matches that 32%; the old model's 43-56% did not.

## 4. Intervention evaluator

Use it in the app's *Did it work?* page, or from the command line:
`python3 pipeline/12_intervention.py STATION START END [--years N]`.
It compares N years before START with N years after END, leaving out the work period.
The math lives in `pipeline/evaluator.py`, which both the app and step 12 use.

**Method:** the same idea as the post-fire analysis.
- Residual = what happened minus the expected risk.
- Change = (beach after - before) - (control beaches 5+ km away, after - before).

One difference: the expected risk comes from a **weather-only model**. It knows the beach,
rain (including the test day), days since rain, tides and season. Each test is scored by
a model trained without that test's year. The forecast model can't be reused here: its
"last test" and `past_rate` inputs follow the beach's own recent results. After a real
improvement, its expected risk falls too, and the improvement disappears from the residuals.

**The ranges had to be widened.** Resampling months gave ranges that were too narrow.
On 300 random beaches and dates with no known project, 25% looked like a real change
instead of about 5%. Beaches drift from year to year for reasons we don't measure
(sand, berms, birds, sampling). So every range is widened 1.74x, the factor that brings
random dates down to 5%. On a second, separate set of 300 random dates, that gave 4%.
The widening was calibrated for 2-year windows, so ranges for other window lengths are approximate.

**Wording:** results say "cleaner after", "dirtier after" or "no clear change after",
never "because of". One beach and one date can't rule out something else changing at the same time.

**Known projects:**

| Project | Unsafe before → after | Weather alone predicts | Change beyond weather and region | Verdict |
|---|---|---|---|---|
| Malibu Lagoon restoration (Surfrider Beach SMB-MC-2; June 2012 - March 2013) | 45.8% → 17.5% | 19.8% → 16.9% | **-23.3 points (-39.8 to -6.8)** | Cleaner after |
| Santa Monica Pier stormwater tank (completed 2018) | 8.2% → 16.5% | 24.1% → 24.9% | +6.5 (-2.8 to +15.8) | No clear change after |
| Santa Monica Pier new bird netting (Aug 2023) | 38.1% → 50.4% | 25.5% → 21.2% | +13.6 (-1.2 to +28.5) | No clear change after |

The Pier tank mainly handles stormwater, while the Pier's problem is mostly dry weather
(43% of dry-weather tests since 2021), so no change after it is not surprising.
Project dates: approximate, from news reports (sources below). The app shows the same label.

**Check of the post-fire result:** rerunning 07's Palisades comparison with weather-only
residuals gives the same answer: overall +1.0 points (-2.7 to +6.6), and +10.7 after rain
(-4.6 to +24.6). So the forecast model's history inputs were not hiding a fire effect.

## Sources for project dates (approximate, from news reports)
- Malibu Lagoon restoration completed March 2013:
  [The Inertia](https://www.theinertia.com/surf/malibu-lagoon-project-manager-responds-to-allegations-that-lagoon-restoration-is-ruining-surfrider-beach/),
  [SMDP](https://www.smdp.com/malibu-lagoon-reopens-after-massive-reconstruction/)
- Santa Monica Pier Clean Beaches Project (2018) and new netting (August 2023):
  [Santa Monica Lookout, Aug 2023](https://santamonicalookout.com/ssm_site/the_lookout/news/News-2023/August-2023/08_21_2023_Water_Quality_Under_Pier_Gives_Santa_Monica_Beaches_a_Bum_Rap.html)

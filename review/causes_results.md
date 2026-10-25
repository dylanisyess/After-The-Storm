# Causes analyses: beach diagnosis, 72-hour advisory check, intervention evaluator

*Done October 24, 2026, with AI assistance (Claude Code); see AI_USE_LOG.md.*
Made by `pipeline/11_beach_diagnosis.py` (`app_data/diagnosis.json`) and
`pipeline/12_intervention.py` (`app_data/interventions.json`). Data: 126,006 beach
tests at 160 beaches, January 2010 to October 2026, with daily ERA5 rain.

**How days are counted.** Rain is a daily total, so "day 0" means 0.1 in or more fell
on the test day, and "day 3" means the last rain day was 3 days earlier. Since we
don't know the hour the rain stopped, a day-3 test was taken about 48-96 hours
after, so it straddles the 72-hour line. Days 0-2 are inside the advisory; day 4
and later are after it ends.

**"Normal"** for a beach = its unsafe rate in the wet season (November-April) on
days with no 0.1 in rain for 8+ days. Region-wide that is 10.1%.

## 1. The 72-hour advisory check

**Headline: after small storms, the region is back to normal by day 3; after
storms of 1 inch or more, beaches are still above normal a week later.**

Unsafe rate by days since rain, all beaches, wet season (normal = 10%):

| Storm size | day 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8-10 |
|---|---|---|---|---|---|---|---|---|---|
| 0.1-0.5 in | 31% | 29% | 17% | 11% | 9% | 14% | 8% | 7% | 9-14% |
| 0.5-1 in | 58% | 45% | 21% | 12% | 12% | 11% | 14% | 7% | 7-10% |
| 1 in or more | 67% | 61% | 37% | 24% | 18% | 15% | 15% | 14% | 12-14% |

Beach by beach (rules below), the 125 beaches that are not unsafe much of the time in dry weather:

| Group | Beaches | What the data shows |
|---|---|---|
| Back to normal in about a day | 28 | Unsafe right after rain; days 2-3 and 4-6 within 5 points of normal. Even after 1 in+ storms, days 2-3 were 8% unsafe (normal 3%). Dockweiler, Hermosa, Venice, Santa Monica State Beach, Will Rogers, Silverstrand, most of Malibu |
| Back to normal within 72 hours | 17 | Still clearly unsafe on days 2-3, normal by days 4-6. Mostly Long Beach (LA River mouth) |
| **Still unsafe after 72 hours** | **7** | Clearly above normal on days 4-6. After 1 in+ storms: 29% unsafe on days 4-6, 19% on days 7-10 (normal 10%) |
| Little change after rain | 15 | Rarely unsafe at all (Ventura and Palos Verdes coves) |
| Unclear | 27 | Too few tests to separate the groups with 95% confidence |
| Not enough tests | 31 | Fewer than 15 tests in a group (Avalon, many Ventura beaches) |
| *Chronic (counted separately)* | *19* | *Unsafe on 10%+ of dry days. "Back to normal" there means back to bad, so they aren't counted in the advisory check* |

The 7 beaches still unsafe after 72 hours, with their rate on days 4-6 vs. normal:

| Beach | Days 4-6 | Normal | Days 3-7, big tide vs. other days |
|---|---|---|---|
| Pico-Kenter outfall, Santa Monica (SMB-3-4) | 22% | 15% | 43% vs 18% |
| SMB-2-7 (near Santa Monica Canyon) | 21% | 13% | 31% vs 22% |
| San Buenaventura State Beach (19000) | 16% | 3% | 35% vs 10% |
| Malibu (DHS007B) | 14% | 2% | 27% vs 10% |
| SMB-2-10 (Playa del Rey, near Ballona Creek) | 11% | 3% | 21% vs 6% |
| Solimar Beach (10000) | 11% | 3% | 24% vs 5% |
| Promenade Park, Ventura (14000) | 10% | 2% | 36% vs 3% |

All seven sit next to a big storm drain, creek or river mouth (the location for the SMB codes is
inferred from coordinates). After rain, they fail far more often on big spring tide
days (range 6.5 ft or more). Region-wide, days 3-7 after rain: 20% on big-tide days vs. 11%.

**What this supports (cautious):** one fixed 72-hour rule is too short for a few
beaches next to large drains and river mouths, especially after storms of 1 inch or more and on
spring tides, and longer than the data needs at about two dozen open-coast beaches.
A rule that depends on storm size and on the beach would match the data better.
This is a pattern in 16 years of tests, not a health study; the "about a day"
beaches still had small leftover risk after the biggest storms (8% vs. 3%).

**Rules** (fixed before looking at beach names): "clearly above normal" = the 95% range
of (rate - normal) is entirely above zero, from a bootstrap that resamples whole
storms (tests after the same storm aren't independent). Each group of days needs
15+ tests. "Within 5 points of normal" also requires not clearly above.

## 2. Beach diagnosis

For every beach `diagnosis.json` has: dry and advisory-period unsafe rates; the
unsafe rate on days 0-10 after rain (for a chart); rates by storm size; the
**trigger** (the smallest storm size where 1 in 4 tests on days 0-1 were unsafe,
and every bigger size too); the recovery group above; the big-tide check; and a
**fix category** with cautious wording:

| Fix category | Beaches | Pattern | Points to (not proof) |
|---|---|---|---|
| chronic | 19 | Unsafe 10%+ of dry days | A source that flows without rain (dry-weather drain flow, creek/lagoon mouth, birds, sewer leaks). First step: source tracking (e.g. human-sewage DNA markers) |
| slow | 7 | Unsafe past 72 hours | A large area that keeps draining after the storm; upstream capture/infiltration; a longer advisory |
| typical | 17 | Clears within 72 hours | Storm runoff; capture before it reaches the beach |
| fast | 28 | Clears in about a day | The first flush from nearby drains; first-flush capture; advisory may be longer than needed |
| clean | 15 | Little change | No fix indicated |

Triggers: at 27 beaches, 0.1-0.25 in of rain already makes 1 in 4 tests fail (all of
Long Beach, Santa Monica Pier and Pico-Kenter, Manhattan, Hermosa); at 35 it takes 1 in or more.
Most chronic beaches are enclosed or semi-enclosed water: Mother's Beach in
Marina del Rey (MDRH-1/2/3, 22-33% of dry days), Cabrillo Beach (CB-02, 32%), Santa
Monica Pier (20%).

## 3. The "delayed peak" puzzle

This doesn't change the conclusion in `review/forecast_puzzle.md`. No beach has a real delayed peak:
at none of them is the rate on days 2-3 clearly above days 0-1, and at the three
Ventura puzzle beaches pooled, days 4-6 vs days 2-3 is +3 points (95% range -5 to +12).
What it adds: Solimar and Promenade Park are two of the seven "still unsafe after
72 hours" beaches, and there the post-storm failures come on big-tide days (days
3-7 after rain at the three Ventura beaches: 32% unsafe with a big tide vs. 6% without). So a
forecast that rises a few days after a storm when a spring tide arrives is right
for those beaches. The fixed model's 29-33% on the Oct 26 spring tide matches that 32%.
The old model's 43-56% did not.

## 4. Intervention evaluator

`python3 pipeline/12_intervention.py STATION START END [--years N]` compares N
years before START with N years after END (the work period is left out).

**Method:** same idea as the post-fire analysis. Residual = what happened minus
expected risk; change = (beach after - before) - (control beaches 5+ km away, after
- before). One difference: expected risk comes from a **weather-only model**
(beach, rain including the test day, days since rain, tides, season), trained
leaving out the test's own year. The forecast model can't be reused here: its
"last test" and `past_rate` inputs follow the beach's own recent results, so
after a real improvement its expected risk falls too and the improvement
disappears from the residuals.

**The ranges had to be widened.** Resampling months gave ranges that were too
narrow: on 300 random beaches and dates with no known project, 25% looked like a
real change instead of about 5%. Beaches drift from year to year for reasons we
don't measure (sand, berms, birds, sampling). So every range is widened by 1.74x,
the factor that brings random dates down to 5%. On a second, separate set of 300
random dates it gave 4%. (Calibrated for 2-year windows; with `--years 3` the ranges are approximate.)

**Known projects:**

| Project | Unsafe before → after | Weather alone expected | Change beyond weather and region | Verdict |
|---|---|---|---|---|
| Malibu Lagoon restoration (Surfrider Beach SMB-MC-2; June 2012 - March 2013) | 45.8% → 17.5% | 19.8% → 16.9% | **-23 points (-40 to -7)** | Cleaner |
| Santa Monica Pier stormwater tank (completed 2018) | 8.2% → 16.5% | 24.1% → 24.9% | +7 (-3 to +16) | No clear change |
| Santa Monica Pier new bird netting (Aug 2023) | 38.1% → 50.4% | 25.5% → 21.2% | +14 (-1 to +29) | No clear change |

The Malibu result is a before/after at one beach, so it shows the beach got cleaner
at that time beyond weather and regional trends; it can't rule out something else
changing at the same time. The Pier tank mainly handles stormwater, while
the Pier's problem is mostly dry weather (chronic, 20% of dry days), so no change after it is
not surprising. Project dates come from news reports and are approximate (sources below).

**Check of the post-fire result:** rerunning 07's Palisades comparison with
weather-only residuals gives the same answer (overall +1.0 points, -2.7 to +6.6;
after rain +10.7, -4.6 to +24.6), so the history inputs were not hiding a fire effect.

## Decisions for Dylan

1. **Period for the diagnosis.** It uses all years 2010-2026, because recovery needs many
   storms. But beaches change: Surfrider Beach is "chronic" on all years, yet
   much cleaner since 2013. Option: recovery from all years, dry rate and chronic label from the
   last 5 years (like `hotspots.json`).
2. `diagnosis.json` covers what `hotspots.json` (07) does with finer groups. Keep
   both, or retire hotspots?
3. Real project dates: the evaluator is only as good as the dates you give it.
   Agency sources (city or Heal the Bay project records) would be better than news articles.

## Sources for project dates
- Malibu Lagoon restoration completed March 2013:
  [The Inertia](https://www.theinertia.com/surf/malibu-lagoon-project-manager-responds-to-allegations-that-lagoon-restoration-is-ruining-surfrider-beach/),
  [SMDP](https://www.smdp.com/malibu-lagoon-reopens-after-massive-reconstruction/)
- Santa Monica Pier Clean Beaches Project (2018) and new netting (August 2023):
  [Santa Monica Lookout, Aug 2023](https://santamonicalookout.com/ssm_site/the_lookout/news/News-2023/August-2023/08_21_2023_Water_Quality_Under_Pier_Gives_Santa_Monica_Beaches_a_Bum_Rap.html)

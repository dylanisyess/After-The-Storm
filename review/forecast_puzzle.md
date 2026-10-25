# The October forecast puzzle

*Investigation done October 24, 2026, with AI assistance (Claude Code); see AI_USE_LOG.md.*

## The problem

On October 24, 2026 the live forecast looked backwards at Ventura and Malibu
beaches. A storm on October 21-22 dropped 1.1 inches of rain on Ventura and
2.9 inches on Malibu. Two days later, the app said **Low risk** at those
beaches. Then, with no more rain coming, the risk **climbed** and peaked two
days later before falling again:

| Beach | Oct 24 | Oct 25 | Oct 26 | Oct 27 |
|---|---|---|---|---|
| Solimar Beach (10000) | 4% | 16% | **43%** | 25% |
| Promenade Park (14000) | 5% | 19% | **56%** | 28% |
| Malibu (DHS010) | 15% | 23% | **44%** | 18% |

Was this real, or a bug?

## Step 1: Is it a bug in how the live forecast builds its inputs?

I rebuilt the exact inputs the forecast used. The rebuilt predictions matched
the published forecast to every decimal. I then checked them against how the
training data was built:

- **Rain:** each window (1, 3, 7 and 30 days) covered the right days before the
  forecast day. The storm was Oct 21 (16 mm in Ventura) and Oct 22 (12 mm).
- **Tides:** live and training tides come from the same NOAA request (same
  datum, units and time zone).

So the inputs were built correctly. Over the four days, only four inputs
changed: rain in the past 3 days (falling to zero), days since rain (2 to 5),
the tides (growing toward a spring tide), and days since the last test.

## Step 2: Which input causes the jump?

I took the Oct 24 inputs and swapped in **one** of Oct 26's inputs at a time:

| Solimar, Oct 24 inputs with Oct 26's... | Risk |
|---|---|
| nothing (as published) | 4% |
| rain in the past 3 days | 3% |
| days since rain | 4% |
| days since last test | 4% |
| **tides** | **55%** |

The tides cause almost the whole jump. Oct 26 had a big spring tide (7.3 ft
between high and low, versus 5.7 ft on Oct 24).

I also changed only the month. Labelling the same Oct 26 inputs as January
instead of October cut the risk in half (43% to 20% at Solimar). So the model
was combining two things: **spring tide** and **October**.

## Step 3: What does the real data say?

At the 36 beaches in question, from 2010 to 2026, I grouped every water test by
how many days it came after the last rainy day of a storm of 1 inch or more:

| Days after the storm | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8-14 | dry weather |
|---|---|---|---|---|---|---|---|---|---|
| Tests that were unsafe | 45% | 14% | 18% | 11% | 11% | 10% | 4% | 4% | 1.6% |

**There is no delayed peak in the real data.** Risk is highest the day after the
rain and then falls off.

Tides do matter, though. In the week after big storms, tests on days with a
tide range of 6.5 ft or more were unsafe more often (34-52%) than on days with
small tides (about 16%). On all other days, big-tide days were also worse (4.8%
vs 1.7%). Three to six days after a storm with a big tide, 20% of tests were
unsafe.

This matches published research. Boehm and Weisberg (2005, *Environmental
Science & Technology* 39: 5575-5583) studied 60 Southern California beaches and
found that during spring tides (around new and full moons) a beach was about
**twice as likely** to fail water-quality standards, especially as the tide goes
out. They found this even at beaches without storm drains or creeks, and
suggested sources such as beach sand, decaying plant material and polluted
groundwater that big tides wash out.

## The answer: partly real, mostly the model guessing

- **Real:** big spring tides raise the risk, and the data and the research both
  show it.
- **Not real:** a peak 4 days after the storm at 44-56%. The data says about 20%.

Why did the model overreact? The situation on October 26 (a few days after a
big storm, with a big spring tide, in October) **never happened in the
training data**. In 2010-2022, only two storms of 1 inch or more outside the
wet season (November-April) had beach tests in the week after them (October
2010 and July 2014). A decision-tree model can still give an answer for
a situation it has never seen. Here it combined a small "October" pattern
learned from very few storms with the tide pattern, and the answer came out
far too high.

## The fix

1. **Removed the "month" input.** The model keeps a simpler "wet season or
   not" input, which has plenty of data behind it.
2. **Required every rule the model learns to rest on at least 100 tests**
   (`min_samples_leaf=100`), so it can't build rules on a handful of unusual days.
3. **Added a "big spring tide" reason** to the forecast when the tide range is
   6.5 ft or more and the risk level is caution or high, so the app explains tide-driven risk honestly instead of
   pointing at old rain.

Same October 24 inputs, before and after the fix:

| Beach | Old model (Oct 24-27) | New model (Oct 24-27) |
|---|---|---|
| Solimar (10000) | 4%, 16%, 43%, 25% | 2%, 10%, 29%, 9% |
| Promenade Park (14000) | 5%, 19%, 56%, 28% | 3%, 14%, 33%, 10% |
| Malibu (DHS010) | 15%, 23%, 44%, 18% | 10%, 19%, 33%, 11% |

The spring-tide bump is still there (it's real), but it's much smaller and
closer to what the data shows.

## What it cost

I retrained and re-tested on 2023-2026 (data the model never saw):

| | Before | After |
|---|---|---|
| Unsafe days caught vs. signs + rain advisory (same number of warnings) | 66% vs 61% (+5.0 points, range +1.9 to +8.6) | 66% vs 61% (+4.7, range +1.8 to +8.2) |
| Estimated live mode | 66% vs 61% (+4.7, range +1.5 to +8.4) | 65% vs 61% (+3.6, range +0.7 to +7.6) |
| 3 days ahead | 63% vs 61% (+1.4) | 62% vs 61% (+0.5) |
| ROC-AUC | 0.859 | 0.859 |

The main result barely moved (retraining alone moves it by about ±0.5 points).
The estimated live result dropped about 1 point. I chose to accept that: a
forecast that sends people the wrong signal in an unusual situation is worse
than one that scores slightly higher on average.

## What I learned

- A good average score doesn't mean every forecast makes sense. I only found
  this by looking at a real day's forecast and asking whether it was
  believable.
- To find the cause, change one input at a time.
- Check the model against the raw data. Here the data showed no delayed peak,
  which proved the peak came from the model and not from the beaches.
- Machine-learning models can give confident answers in situations they have
  never seen. Simpler inputs and rules that need more evidence make them more
  cautious.

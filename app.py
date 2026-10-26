"""After the Rain: LA and Ventura County beach water, what makes it unsafe and what
fixes it (Congressional App Challenge 2026).

Run inside WSL:  source ~/waterenv/bin/activate && streamlit run app.py

The app READS the files in app_data/ (made by the pipeline; see pipeline/README.md).
One exception: when forecast.json is older than today (LA time), the Forecast page
reruns pipeline/08_forecast.py to make today's. The cleanup-project math lives in
pipeline/evaluator.py, so the app and pipeline/12_intervention.py give the same answer.

Every page opens with a short headline (one idea, one number) and the details on
the line below, then annotated charts, then quiet number tiles. Pages, in order (left menu):
  Home                what the app is for, three key findings, a question for each page
  After-rain patterns does LA County's 72-hour rain advisory fit each beach?
  Beach profile       one beach: dry vs. after rain, rain trigger, recovery, what fix it points to
  Compare beaches     2 or 3 beaches side by side: risk today, latest test, recovery, distance apart
  Cleanup projects    a beach before vs. after a project, beyond what the weather explains
  Fire impacts        did beaches below the 2025 Palisades burn get dirtier?
  Forecast            today + 3 days for every beach (with the January 2023 storm replay)
  Methods             how the model and the analyses work, and their limits

Links can name a beach: /forecast?beach=B-22 or /beach-profile?beach=B-22.
"""

import base64
import html
import json
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import altair as alt
import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st

ROOT = Path(__file__).parent
DATA = ROOT / "app_data"
sys.path.insert(0, str(ROOT / "pipeline"))
from evaluator import evaluate  # noqa: E402  (pipeline/evaluator.py, shared with step 12)

APP_NAME = "After the Rain"
TAGLINE = "LA and Ventura County beaches: what makes the water unsafe, and what fixes it"
LOGO = ROOT / "assets" / "logo"   # a raindrop with rain, the sun and waves
st.set_page_config(page_title=APP_NAME, page_icon=str(LOGO / "after-the-rain-icon-64.png"), layout="wide")


def logo_img(file, size):
    """The logo as an <img> tag. The SVG is pasted into the page as text (base64), because st.html
    can't link to files in our folder. The drop sits in a square with empty space at the sides (about
    a fifth of the width each), so a negative margin pulls that space in and the drop lines up with the text."""
    data = base64.b64encode((LOGO / file).read_bytes()).decode()
    side = round(size * 0.2)
    return (f'<img class="logo" src="data:image/svg+xml;base64,{data}" width="{size}" height="{size}" alt="" '
            f'style="margin: 0 -{side}px">')


# ---------------------------------------------------------------------------
# Shared settings: colors, names, and the numbers we quote
# ---------------------------------------------------------------------------

# Forecast risk levels: one status color each, always shown next to its word
# (so the app works for colorblind users). "size" makes riskier dots bigger on the map.
LEVELS = {
    "low":     {"word": "Low risk",  "hex": "#0ca30c", "rgb": [12, 163, 12],  "size": 6},
    "caution": {"word": "Caution",   "hex": "#fab219", "rgb": [250, 178, 25], "size": 8},
    "high":    {"word": "High risk", "hex": "#d03b3b", "rgb": [208, 59, 59],  "size": 10},
}
GREY = {"hex": "#9aa0a6", "rgb": [154, 160, 166]}  # sites with no forecast
NAVY = "#22314F"     # Harbor: chart annotations, and chronic beaches on the maps
MUTED = "#4E5B72"    # secondary text, "other beaches" lines (5.9:1 contrast on the tinted tiles)
BLUE = "#2a78d6"     # chart slot 1
ORANGE = "#eb6834"   # chart slot 2
AQUA = "#1baf7a"     # chart slot 3

# How each beach recovers after rain (from diagnosis.json). Chronic beaches are often
# unsafe even in dry weather, so for them "back to normal" means back to bad: they get
# their own color unless they also stay unsafe past 72 hours.
# Each group has ONE name, used everywhere: cards, chart labels, legends, map, tables.
GROUPS = {
    "slow":    {"word": "Unsafe longer than 72 hours",     "hex": ORANGE,    "rgb": [235, 104, 52]},
    "typical": {"word": "Back to normal within 72 hours",  "hex": AQUA,      "rgb": [27, 175, 122]},
    "fast":    {"word": "Recover faster than most",        "hex": BLUE,      "rgb": [42, 120, 214]},
    "chronic": {"word": "Often unsafe in dry weather",     "hex": "#8E3B76", "rgb": [142, 59, 118]},  # plum: unlike the blues
    "other":   {"word": "Rarely unsafe, or too few tests", "hex": "#B8C2CC", "rgb": [184, 194, 204]},
}

# One-line plain explanations, shown wherever these words appear.
PRECAUTION = "Advisories are precautions; this is not advice to swim during one."
NORMAL_NOTE = ("Normal level: how often a beach's tests fail in the rainy season (November to April) "
               "after 8 or more days without rain.")
POINTS_NOTE = "Points are percentage points: going from 20% of tests unsafe to 17% is 3 points fewer."
# Where the data comes from, and which rule every beach is compared with.
COVERAGE = "LA and Ventura County beaches"
ADVISORY_NOTE = ("Every beach, including Ventura County's, is compared with LA County's 72-hour rain advisory. "
                 + PRECAUTION)
# The evaluator's known projects: dates are from news articles, not agency records.
DATES_NOTE = "Project dates: approximate, from news reports."

# Offshore receiving-water sites near Palos Verdes: not beaches and not inland, so never mapped.
HIDE_FROM_MAP = {"Peninsula-RW1", "Peninsula-RW2"}
# Map dot sizes are in screen pixels, so dots stay tappable at any zoom.
# pydeck reads plain strings as formulas, so a fixed word must be wrapped in pdk.types.String.
PIXELS = pdk.types.String("pixels")
# A few station codes stand for a place the state data doesn't spell out.
# MDRH = Marina del Rey Harbor (inside the harbor, so it would be wrong to call it "near Venice Beach").
CODE_PLACES = {"MDRH": "Marina del Rey Harbor"}
# Stations best known by another name. SMB-MC-2 sits at the Malibu Creek mouth, on Surfrider Beach.
KNOWN_PLACES = {"SMB-MC-2": "Surfrider Beach, Malibu"}


# Styling the theme (.streamlit/config.toml) can't do. Spacing uses one scale
# (--s1 to --s7: 4, 8, 12, 16, 24, 32, 48 px) so gaps stay consistent everywhere.
# Hierarchy: white panels with a hairline hold charts; quiet tinted tiles hold numbers.
st.html("""
<style>
  :root {
    --s1: 4px; --s2: 8px; --s3: 12px; --s4: 16px; --s5: 24px; --s6: 32px; --s7: 48px;
    --ink: #1A2440; --ink-2: #3B4A63; --muted: #4E5B72; --harbor: #22314F;
    --surface: #FFFFFF; --tide: #E8EEF6; --line: #DCE4EE; --sun-strong: #F6C453;
    --display: 'Gabarito', sans-serif; --text: 'Public Sans', sans-serif;
  }
  [data-testid="stHeader"] { background: transparent; }
  [data-testid="stMainBlockContainer"] { max-width: 1080px; padding: var(--s6) var(--s6) var(--s7);
                                         container-type: inline-size; }  /* lets the tile grid follow the page width */

  /* left menu: wordmark first, then the page links */
  [data-testid="stSidebarContent"] { display: flex; flex-direction: column; }
  [data-testid="stSidebarHeader"] { order: -2; }
  [data-testid="stSidebarUserContent"] { order: -1; padding-top: 0; padding-bottom: var(--s3); }
  [data-testid="stSidebarNav"] { padding-top: 0; }
  [data-testid="stSidebarNavLink"] { border-radius: 10px; padding: var(--s2) var(--s3); margin: 2px var(--s2); }
  [data-testid="stSidebarNavLink"] span { font-size: 1rem; }
  .side-brand { display: flex; align-items: center; gap: var(--s3); font-family: var(--display); font-weight: 800;
                font-size: 1.6rem; line-height: 1; color: #FFFFFF; letter-spacing: -.01em; }
  .logo { flex: none; }
  .side-tag { font-size: .875rem; line-height: 1.4; color: #B9C6DC; margin-top: var(--s3); }
  .brand-main { display: none; }

  /* the top of every page */
  .kicker  { font-size: .95rem; font-weight: 500; color: #2366C7; margin: 0 0 var(--s2); }
  .finding { font-family: var(--display); font-weight: 600; font-size: clamp(1.7rem, 1.2rem + 1.7vw, 2.55rem);
             line-height: 1.1; letter-spacing: -.015em; color: var(--ink); max-width: 21em; margin: 0; }
  .finding b { font-weight: 800; }
  .lede    { font-size: 1.0625rem; line-height: 1.55; color: var(--ink-2); max-width: 62ch; margin: var(--s4) 0 0; }
  .note    { font-size: .9375rem; line-height: 1.5; color: var(--muted); max-width: 62ch; margin: var(--s3) 0 0; }
  .hero    { margin: 0 0 var(--s6); }

  /* numbers: quiet tiles, no border; the small colored key matches the chart below */
  /* 4 tiles across on wide pages, 2 x 2 at medium widths, 1 per row on phones */
  .cards   { display: grid; grid-template-columns: repeat(4, 1fr); gap: var(--s3); margin: 0 0 var(--s5); }
  @container (max-width: 860px) { .cards { grid-template-columns: repeat(2, 1fr); } }
  @container (max-width: 480px) { .cards { grid-template-columns: 1fr; } }
  .card    { background: var(--tide); border-radius: 12px; padding: var(--s4); color: var(--ink); }
  .card .key { width: 28px; height: 4px; border-radius: 2px; background: var(--c); margin-bottom: var(--s3); }
  .card .v { font-family: var(--display); font-weight: 700; font-size: 1.95rem; line-height: 1.05; }
  .card .v-small { display: block; font-size: 1.2rem; line-height: 1.2; }
  .card .l { font-size: .95rem; line-height: 1.35; margin-top: var(--s2); }
  .card .n { font-size: .8125rem; line-height: 1.4; color: var(--muted); margin-top: var(--s2); }

  /* panels: white, hairline, larger corners than the tiles inside the page */
  [class*="st-key-panel-"] { background: var(--surface); border: 1px solid var(--line) !important;
                              border-radius: 16px !important; padding: var(--s5) !important; }
  .ptitle { font-family: var(--display); font-weight: 600; font-size: 1.2rem; line-height: 1.25; color: var(--ink);
            margin: 0 0 var(--s1); }
  [data-testid="stExpander"] details { border-radius: 12px; border-color: var(--line); background: var(--surface); }
  /* Streamlit draws captions see-through, which made them too faint (2.7:1); solid muted text is 5.9:1 or more */
  [data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] * { color: var(--muted); opacity: 1 !important; }

  /* forecast: a status panel and four day cards with a risk bar, like surf-condition cards */
  .box     { background: var(--surface); border: 1px solid var(--line); border-radius: 16px; padding: var(--s5);
             color: var(--ink); margin: var(--s1) 0 var(--s3); }
  .box .where { font-family: var(--display); font-size: 1.2rem; font-weight: 600; }
  .box .code  { font-family: var(--text); font-size: .85rem; color: var(--muted); margin-left: var(--s2); font-weight: 400; }
  .box .level { font-family: var(--display); font-size: 2.2rem; font-weight: 700; line-height: 1.1; margin: var(--s2) 0 0; }
  .box .small { font-size: .9375rem; line-height: 1.45; color: var(--ink-2); margin-top: var(--s2); }
  .meter   { height: 6px; border-radius: 3px; background: #E3E9F1; overflow: hidden; margin: var(--s3) 0; }
  .meter i { display: block; height: 100%; width: var(--w); background: var(--c); border-radius: 3px; }
  .days    { display: grid; grid-template-columns: repeat(4, 1fr); gap: var(--s2); margin: 0 0 var(--s4); }
  .day     { background: var(--surface); border: 1px solid var(--line); border-radius: 12px; padding: var(--s3);
             color: var(--ink); }
  .day b   { display: block; font-size: .8125rem; font-weight: 500; color: var(--muted); }
  .day .lv { font-family: var(--display); font-weight: 600; font-size: 1.05rem; margin-top: var(--s1); }
  .day .meter { margin: var(--s2) 0 var(--s1); height: 4px; }
  .day .pct { font-size: .875rem; color: var(--ink-2); }
  .day .act { font-size: .8125rem; margin-top: var(--s2); color: var(--ink-2); }
  .day.picked { border: 2px solid var(--harbor); padding: calc(var(--s3) - 1px); }  /* the day chosen above the map */

  .box .part { border-top: 1px solid var(--line); margin-top: var(--s4); padding-top: var(--s4); }
  .box .part:first-of-type { border-top: 0; margin-top: var(--s3); padding-top: 0; }
  .box h5  { font-family: var(--text); font-size: .875rem; font-weight: 700; color: var(--ink); margin: 0 0 var(--s1);
             padding: 0; }
  .box .old { font-size: .875rem; color: var(--muted); margin-top: var(--s1); }
  .drivers { list-style: none; padding: 0; margin: var(--s3) 0 0; }
  .drivers li { display: grid; grid-template-columns: 5.5rem 1fr; gap: var(--s3); padding: var(--s2) 0;
                border-top: 1px solid var(--line); font-size: .9375rem; line-height: 1.4; }
  .drivers .p { font-family: var(--display); font-weight: 700; font-size: 1.3rem; line-height: 1.1; }
  .drivers .p small { display: block; font-family: var(--text); font-weight: 400; font-size: .8125rem; color: var(--muted); }
  .drivers .t { color: var(--muted); font-size: .875rem; }

  /* a table that becomes one block per row on narrow screens (no sideways scrolling) */
  .rows { width: 100%; border-collapse: collapse; font-size: .9375rem; color: var(--ink); }
  .rows th { text-align: left; font-weight: 600; font-size: .8125rem; color: var(--muted); padding: var(--s2);
             border-bottom: 1px solid var(--line); vertical-align: bottom; }
  .rows td { padding: var(--s2); border-bottom: 1px solid var(--line); }
  .rows .num { text-align: right; font-variant-numeric: tabular-nums; }
  @container (max-width: 640px) {
    .rows thead { display: none; }
    .rows tr { display: block; padding: var(--s3) 0; border-bottom: 1px solid var(--line); }
    .rows td { display: flex; justify-content: space-between; gap: var(--s3); border: 0; padding: 2px 0; text-align: right; }
    .rows td::before { content: attr(data-label); color: var(--muted); text-align: left; }
    .rows td:first-child { font-weight: 600; text-align: left; padding-bottom: var(--s1); }
    .rows td:first-child::before { content: none; }
  }

  /* legends: a colored dot next to its words */
  .chips { display: flex; flex-wrap: wrap; gap: var(--s2) var(--s4); margin: var(--s1) 0 var(--s2); }
  .chip  { font-size: .875rem; color: var(--ink-2); white-space: nowrap; }
  .dot   { display: inline-block; width: 10px; height: 10px; border-radius: 50%; background: var(--c);
           margin-right: var(--s2); vertical-align: 0; }

  /* home page */
  .home-hero .title-row { display: flex; align-items: center; gap: var(--s3); margin: 0 0 var(--s4); }
  .home-hero h1 { font-family: var(--display); font-weight: 800; font-size: clamp(2.6rem, 1.8rem + 3vw, 4rem);
                  line-height: 1; letter-spacing: -.02em; color: var(--ink); margin: 0; padding: 0; }
  .home-hero p  { font-size: 1.2rem; line-height: 1.5; color: var(--ink-2); max-width: 34ch; margin: 0; }
  .photo        { aspect-ratio: 4 / 3; border-radius: 16px; background: var(--tide); display: flex; align-items: center;
                  justify-content: center; color: var(--muted); font-size: .9rem; text-align: center; padding: var(--s4); }
  .section-h    { font-family: var(--display); font-weight: 700; font-size: 1.55rem; color: var(--ink);
                  margin: var(--s7) 0 var(--s4); }
  .problem p    { font-size: 1.15rem; line-height: 1.45; color: var(--ink); margin: 0 0 var(--s2); max-width: 60ch; }
  .big3         { display: grid; grid-template-columns: repeat(3, 1fr); gap: var(--s3); }
  @container (max-width: 760px) { .big3 { grid-template-columns: 1fr; } }
  .big3 .card .v { font-size: 3rem; }
  /* question cards: the whole card is the link, with an arrow at the end */
  [class*="st-key-qcard-"] a { display: flex; align-items: center; gap: var(--s3); min-height: 6.5rem; height: 100%;
                               background: var(--surface); border: 1px solid var(--line); border-radius: 16px;
                               padding: var(--s4) var(--s5); text-decoration: none; }
  [class*="st-key-qcard-"] a:hover { border-color: #2366C7; }
  [class*="st-key-qcard-"] a:focus-visible { outline: 3px solid #2366C7; outline-offset: 2px; }
  [class*="st-key-qcard-"] a * { white-space: normal !important; overflow: visible !important; text-overflow: clip !important; }
  [class*="st-key-qcard-"] a p { font-family: var(--display); font-weight: 600; font-size: 1.15rem; line-height: 1.3;
                                 color: var(--ink); }
  [class*="st-key-qcard-"] a::after { content: "→"; margin-left: auto; padding-left: var(--s2); font-size: 1.3rem;
                                      color: #2366C7; }
  /* home actions: a beach search (then a line about the beach and its questions) and a filled button */
  .beach-line   { font-size: 1rem; line-height: 1.5; color: var(--ink-2); margin: var(--s1) 0 var(--s2); }
  .beach-line .code { font-size: .85rem; color: var(--muted); margin-left: var(--s2); }
  .st-key-ask { gap: var(--s2); margin-bottom: var(--s3); }
  .st-key-ask a { background: var(--surface); border: 1px solid var(--line); border-radius: 10px;
                  padding: var(--s2) var(--s3); min-height: 2.75rem; }
  .st-key-ask a:hover { border-color: #2366C7; }
  .st-key-ask a * { white-space: normal !important; overflow: visible !important; text-overflow: clip !important; }
  .st-key-ask a p { color: var(--ink); font-weight: 500; }
  a.explore { display: inline-flex; gap: var(--s2); align-items: center; background: #2366C7; color: #FFFFFF;
              border-radius: 10px; padding: var(--s2) var(--s4); min-height: 2.5rem; font-weight: 600;
              text-decoration: none; margin-top: var(--s2); }
  a.explore:hover { background: #1D57AB; color: #FFFFFF; }
  a.explore:focus-visible { outline: 3px solid #2366C7; outline-offset: 2px; }
  #what-we-found { scroll-margin-top: var(--s7); }
  [data-testid="stMain"] { scroll-behavior: smooth; }
  .who p        { font-size: 1rem; line-height: 1.5; color: var(--ink-2); margin: 0 0 var(--s2); max-width: 70ch; }
  .who b        { color: var(--ink); }
  .trust        { display: grid; grid-template-columns: repeat(4, 1fr); gap: var(--s5); background: var(--tide);
                  border-radius: 16px; padding: var(--s5); margin-top: var(--s7); }
  @container (max-width: 860px) { .trust { grid-template-columns: repeat(2, 1fr); } }
  @container (max-width: 480px) { .trust { grid-template-columns: 1fr; } }
  .trust a      { color: #2366C7; font-weight: 600; }
  [data-testid="stImage"] img { border-radius: 16px; }
  .trust h4     { font-family: var(--text); font-size: .875rem; font-weight: 700; color: var(--ink); margin: 0 0 var(--s1);
                  padding: 0; }
  .trust p      { font-size: .875rem; line-height: 1.45; color: var(--ink-2); margin: 0; }

  @media (max-width: 767px) {
    [data-testid="stMainBlockContainer"] { padding: var(--s5) var(--s4) var(--s7); }
    .brand-main { display: flex; align-items: center; gap: var(--s2); font-family: var(--display);
                  font-weight: 800; font-size: 1.15rem; color: var(--ink); margin: calc(-1 * var(--s2)) 0 var(--s5);
                  padding-left: 40px; }  /* sits beside the menu button, not under it */
    [class*="st-key-panel-"] { padding: var(--s4) !important; }
    .days { grid-template-columns: repeat(2, 1fr); }
    .box .level { font-size: 1.8rem; }
  }
</style>
""")


# Charts: one look for all of them. Text in the body font, quiet grey axes and
# hairline grids, no box around the plot. (Each chart still sets its own colors.)
@alt.theme.register("after_rain", enable=True)
def after_rain_theme():
    font = "Public Sans, sans-serif"
    axis = {"labelColor": MUTED, "titleColor": MUTED, "labelFontSize": 12, "titleFontSize": 12,
            "titleFontWeight": 500, "gridColor": "#E6ECF3", "domainColor": "#C9D3E0", "tickColor": "#C9D3E0"}
    return {"config": {"font": font, "background": "transparent", "view": {"stroke": None},
                       "axis": axis, "axisX": {"grid": False}, "text": {"font": font}}}  # horizontal grid lines only


def show_chart(chart, width="stretch"):
    """Draw an Altair chart with the app's own chart theme (not Streamlit's)."""
    st.altair_chart(chart, width=width, theme=None)


_panels = iter(range(1000))  # this file reruns top to bottom on every click, so numbering restarts each time


def panel(parent=st):
    """A white bordered panel for a chart or a block of text (styled by the st-key-panel-* rule above)."""
    return parent.container(border=True, key=f"panel-{next(_panels)}")


def title(text):
    """A panel's title, in the headline font."""
    st.html(f'<div class="ptitle">{html.escape(text)}</div>')


# ---------------------------------------------------------------------------
# Loading data and small helpers
# ---------------------------------------------------------------------------

LA = ZoneInfo("America/Los_Angeles")


def la_today():
    """Today's date in Los Angeles, as text like '2026-10-25' (the server may be in another time zone)."""
    return f"{datetime.now(LA):%Y-%m-%d}"


def file_version(name):
    """When a file in app_data/ last changed. It is part of every cache key below, so when the
    pipeline (or the forecast refresh) rewrites a file, the app reads the new one."""
    return (DATA / name).stat().st_mtime_ns


@st.cache_data(ttl="1h", max_entries=40, show_spinner=False)
def read_json(name, version):
    return json.loads((DATA / name).read_text(encoding="utf-8"))


def load_json(name):
    """Read one file from app_data/ (cached until the file changes, and for at most an hour)."""
    return read_json(name, file_version(name))


def clean_name(code, name):
    """'10000-Solimar Beach, Ventura' -> 'Solimar Beach, Ventura'."""
    if name.startswith(code + "-"):
        name = name[len(code) + 1:]
    return name.strip()


@st.cache_data(ttl="1h", show_spinner=False)
def station_table(version):
    """All 311 stations from stations.json, with a readable 'place' name.

    About 40 beaches only have a code (like 'SMB-1-12') in the state data.
    For those we name the closest ocean beach that does have a place name,
    e.g. 'Near Malibu Beach', if it is within 2 km. (Ocean only, so we never
    borrow the name of a storm basin or a river study site.)
    """
    df = pd.DataFrame(load_json("stations.json")["stations"])
    df["place"] = [clean_name(c, n) for c, n in zip(df["station"], df["name"])]
    has_place = df["place"].str.contains("[a-z]")  # real names have lower-case letters; codes don't
    named = df[has_place & (df["water"] == "ocean")]
    for i in df.index[~has_place]:
        code = df.at[i, "station"]
        if code.split("-")[0] in CODE_PLACES:
            df.at[i, "place"] = CODE_PLACES[code.split("-")[0]]
            continue
        # rough distance in km (1 degree of latitude = 111 km; of longitude = 92 km at LA's latitude)
        km = (((named["lat"] - df.at[i, "lat"]) * 111) ** 2 + ((named["lon"] - df.at[i, "lon"]) * 92) ** 2) ** 0.5
        df.at[i, "place"] = f"Near {named.at[km.idxmin(), 'place']}" if km.min() < 2 else code  # else just the code
    df = df.set_index("station")
    for code, name in KNOWN_PLACES.items():  # same name in every picker, headline and table
        if code in df.index:
            df.at[code, "place"] = name
    return df


STATIONS = station_table(file_version("stations.json"))


@st.cache_data(ttl="1h", show_spinner=False)
def model_ids(versions):
    """The model_id stamped on each app_data file. `versions` (file names and change times)
    is only the cache key, so the check reruns when any file changes."""
    return {name: load_json(name).get("model_id") for name, _ in versions}


def check_model_ids():
    """Warn if some app_data files came from a different model run than others.
    Every pipeline step stamps "model_id"; pipeline/rebuild.sh remakes them all together."""
    ids = model_ids(tuple((f.name, file_version(f.name)) for f in sorted(DATA.glob("*.*json"))))
    if len(set(ids.values())) > 1:
        st.warning("Some data files came from different model runs, so numbers may not match. "
                   "Run `bash pipeline/rebuild.sh`. " + ", ".join(f"{n}: {i}" for n, i in sorted(ids.items())))


def place(code):
    return STATIONS.at[code, "place"] if code in STATIONS.index else code


def label(code):
    """Name plus code, e.g. 'Solimar Beach, Ventura (10000)' (just the code if it has no name)."""
    return place(code) if place(code) == code else f"{place(code)} ({code})"


def code_tag(code):
    """The station code, shown small after the name (skipped when the name already is the code)."""
    return "" if place(code) == code else f'<span class="code">{html.escape(code)}</span>'


def limit_of(code):
    """The state limit for this station: 104 for enterococcus (ocean), 320 for E. coli (fresh water)."""
    return float(STATIONS.at[code, "standard"].split(">")[1])


def pct(x):
    return "n/a" if x is None or pd.isna(x) else f"{x * 100:.0f}%"


MIN_TESTS = 15  # fewer tests than this and a rate jumps around by chance (the same minimum as step 11's groups)


def rate_n(rate, n):
    """A rate with the number of tests behind it, e.g. '41% (34 tests)', or 'Not enough tests (6)'."""
    if not n or n < MIN_TESTS or rate is None:
        return f"Not enough tests ({n or 0})"
    return f"{pct(rate)} ({n:,} tests)"


def pts(x, sign=True):
    """0.071 -> '+7.1' (percentage points)."""
    return f"{x * 100:+.1f}" if sign else f"{abs(x) * 100:.1f}"


def finding(kicker, sentence, lede=None, notes=()):
    """The top of every page: a small label, a SHORT headline with one idea and one
    number (bold the number with <b>), the details on the line below, and short notes
    underneath (plain explanations of words like "points", or a caution)."""
    notes = [notes] if isinstance(notes, str) else [n for n in notes if n]
    st.html(f'<div class="hero"><div class="kicker">{html.escape(kicker)}</div><div class="finding">{sentence}</div>'
            + (f'<div class="lede">{lede}</div>' if lede else "")
            + "".join(f'<div class="note">{n}</div>' for n in notes) + "</div>")


def cards(items):
    """A row of quiet number tiles: [(big value, label, small note or None[, color]), ...].
    The optional color draws a short key bar that matches that group's color in the chart."""
    out = ""
    for v, l, n, *color in items:
        key = f'<div class="key" style="--c:{color[0]}"></div>' if color else ""
        out += (f'<div class="card">{key}<div class="v">{v}</div><div class="l">{l}</div>'
                + (f'<div class="n">{n}</div>' if n else "") + "</div>")
    st.html(f'<div class="cards">{out}</div>')


def meter(share, color):
    """A thin bar filled to `share` (0-1): the forecast's risk %, drawn."""
    return f'<div class="meter"><i style="--w:{share * 100:.0f}%; --c:{color}"></i></div>'


def chips(items):
    """A legend row: a colored dot and its word for each item: [(hex, text), ...].
    Chart legends get cut off on phones, so every chart uses these instead."""
    spans = "".join(f'<span class="chip"><span class="dot" style="--c:{c}"></span>{html.escape(t)}</span>'
                    for c, t in items)
    return f'<div class="chips">{spans}</div>'


def csv_button(df, name, label="Download CSV"):
    """A small button that saves `df` as a spreadsheet file named after-the-rain_<name>.csv.
    on_click="ignore" means clicking it doesn't rerun the page."""
    st.download_button(label, df.to_csv(index=False).encode("utf-8"), file_name=f"after-the-rain_{name}.csv",
                       mime="text/csv", icon=":material/download:", type="tertiary", on_click="ignore",
                       key=f"csv-{name}")


def numbers(df, name, **config):
    """Every chart has a table twin, for screen readers and anyone who wants the exact values,
    and a button to download the same numbers as a CSV file (`name` names the file)."""
    with st.expander("Show the numbers"):
        st.dataframe(df, hide_index=True, width="stretch", column_config=config)
    csv_button(df, name)


def is_phone():
    """True on a phone. Streamlit can't measure the screen, but phone browsers say "Mobile",
    "iPhone" or "Android" in the User-Agent header they send with every request."""
    agent = st.context.headers.get("User-Agent", "")
    return any(word in agent for word in ("Mobile", "iPhone", "Android"))


# On a phone the map is only about 360 px wide. The beaches run 1.33 degrees of longitude
# (Rincon to Long Beach); at zoom 7.35 a 360 px map shows about 1.5 degrees, so they all fit.
PHONE_ZOOM = 7.35


def base_map(layers, lat=33.98, lon=-118.78, zoom=7.9, tooltip=None, height=400, phone_zoom=PHONE_ZOOM, **kw):
    """A plain grey map (so the colored dots stand out), zoomed to the coast from Ventura to Long Beach."""
    if is_phone():
        zoom = phone_zoom
    deck = pdk.Deck(layers=layers, initial_view_state=pdk.ViewState(latitude=lat, longitude=lon, zoom=zoom),
                    map_provider="carto", map_style=pdk.map_styles.CARTO_LIGHT,  # Carto 'Positron': quiet grey, no API key
                    tooltip={"html": tooltip, "style": {"backgroundColor": "white", "color": NAVY, "fontSize": "13px", "borderRadius": "8px",
                                       "fontFamily": "Public Sans, sans-serif"}}
                    if tooltip else None)
    st.pydeck_chart(deck, height=height, **kw)


# ---- diagnosis.json (step 11) ----
DIAG = load_json("diagnosis.json")
DIAG_BY = {s["station"]: s for s in DIAG["stations"]}
RECENT_YEAR = DIAG["recent_period"][:4]


def group_of(s):
    """Which color group a diagnosed beach belongs to (see GROUPS)."""
    if s["recovery"] == "still unsafe after 72 hours":
        return "slow"
    if s["chronic"]:
        return "chronic"
    return {"back to normal in about a day": "fast", "back to normal within 72 hours": "typical"}.get(s["recovery"], "other")


def group_rates(s):
    """The numbers behind a beach's group, so a name like "back to normal" always comes with
    its actual rate, e.g. 'days 4 to 6 after rain: 8% unsafe; normal level 6%'."""
    g = group_of(s)
    normal = pct(s["normal_wet_season_rate"])
    if g in ("slow", "typical"):
        return f"days 4 to 6 after rain: {pct(s['groups']['days 4-6']['unsafe_rate'])} unsafe; normal level {normal}"
    if g == "fast":
        return f"days 2 to 3 after rain: {pct(s['groups']['days 2-3']['unsafe_rate'])} unsafe; normal level {normal}"
    return f"dry weather: {pct(s['dry_unsafe_rate'])} of tests unsafe since {RECENT_YEAR}"


def link_beach(page):
    """The beach named in the link (?beach=CODE), but only the first time this page sees it.
    After that the page's own picker wins, so picking another beach isn't undone by the link.
    A new link (say, from the Home page's buttons) names a different beach, so it applies again."""
    code = st.query_params.get("beach")
    if code and st.session_state.get(f"link_used_{page}") != code:
        st.session_state[f"link_used_{page}"] = code
        return code
    return None


def chosen_beach(key, options, default):
    """Pick the beach a page shows, before its picker (with this widget key) is drawn.
    In order: a beach named in a new link (?beach=CODE); the picker's current value; the
    beach last picked on another page (they share it); `default`."""
    for code in (link_beach(key), st.session_state.get(key), st.session_state.get("shared_beach")):
        if code in options:
            st.session_state[key] = code
            return
    st.session_state[key] = default


def remember_beach(code, page):
    """Put the chosen beach in the link (so it can be shared) and pass it to the other pages.
    `page` is the same name given to link_beach(), so this beach doesn't count as a new link."""
    st.query_params["beach"] = code
    st.session_state["shared_beach"] = code
    st.session_state[f"link_used_{page}"] = code


def rows_table(columns, rows):
    """A plain table that turns into one compact block per row on phones, so it never
    scrolls sideways. columns: [(heading, is_number)]; rows: lists of text (already formatted)."""
    head = "".join(f'<th class="{"num" if num else ""}">{html.escape(h)}</th>' for h, num in columns)
    body = "".join("<tr>" + "".join(f'<td class="{"num" if num else ""}" data-label="{html.escape(h)}">{v}</td>'
                                    for (h, num), v in zip(columns, r)) + "</tr>" for r in rows)
    st.html(f'<table class="rows"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>')


def pooled(curve, lo, hi):
    """Unsafe rate over days lo..hi of a curve [{day, unsafe_rate, tests}], weighted by tests."""
    sel = [c for c in curve if lo <= c["day"] <= hi and c["unsafe_rate"] is not None]
    n = sum(c["tests"] for c in sel)
    return sum(c["unsafe_rate"] * c["tests"] for c in sel) / n if n else None


def advisory_band(top, text="72-hour advisory"):
    """A light shaded band over days 0-3 (inside the advisory) with its label.
    Day 3 is roughly 48-96 hours after the rain, so it straddles the 72-hour line."""
    band = pd.DataFrame({"x": [-0.5], "x2": [3.5], "y": [top], "label": [text]})
    rect = alt.Chart(band).mark_rect(color="#eef2f6").encode(x="x:Q", x2="x2:Q")
    words = alt.Chart(band).mark_text(align="left", baseline="top", dx=4, dy=4, color=MUTED, fontSize=11).encode(
        x="x:Q", y=alt.value(0), text="label:N")
    return rect + words


def day_axis(right=10.5):
    """Days 0-10 since rain. `right` past 10.5 leaves empty room for labels at the line ends."""
    return alt.X("day:Q", title="Days since 0.1 in or more of rain", scale=alt.Scale(domain=[-0.5, right], nice=False),
                 axis=alt.Axis(values=list(range(11)), format="d", grid=False))


DAY_AXIS = day_axis()


def two_lines(name):
    """Split a group name into two lines near the middle (same words, just wrapped)."""
    words = name.split()
    cut = min(range(1, len(words)), key=lambda i: abs(len(" ".join(words[:i])) - len(name) / 2))
    return [" ".join(words[:cut]), " ".join(words[cut:])]


def end_labels(df, key, value, top, gap_share=0.13):
    """Put each line's name just right of its last point, nudged apart so no two
    labels overlap (at least `gap_share` of the chart height between them)."""
    last = df[df["day"] == df["day"].max()].sort_values(value).reset_index(drop=True)
    ys, gap = [], top * gap_share
    for v in last[value]:
        ys.append(max(v, ys[-1] + gap) if ys else v)
    return last.assign(label_y=ys, lines=["\n".join(two_lines(g)) for g in last[key]])  # \n = line break


# ---------------------------------------------------------------------------
# Page 1: After-rain patterns (the 72-hour advisory check)
# ---------------------------------------------------------------------------

def advisory_page():
    reg = DIAG["region"]
    rg = reg["recovery_groups"]
    n = {g: rg[g]["beaches"] for g in ("slow", "typical", "fast")}
    slow_big = pooled(rg["slow"]["curves"]["1 in or more"], 4, 6)
    fast_big = pooled(rg["fast"]["curves"]["1 in or more"], 2, 3)
    typical_46 = pooled(rg["typical"]["curves"]["all storms"], 4, 6)
    tide = reg["after_rain_tides"]

    slow_chronic = sum(DIAG_BY[c]["chronic"] for c in rg["slow"]["stations"])
    also_dry = (f"{slow_chronic} of them {'is' if slow_chronic == 1 else 'are'} also often unsafe in dry weather."
                if slow_chronic else "")
    fast_23 = pooled(rg["fast"]["curves"]["all storms"], 2, 3)
    fast_normal = rg["fast"]["normal_wet_season_rate"]

    # Headline, then two short lines (not one long paragraph), then the chart.
    finding("After-rain patterns",
            f"<b>{n['slow']} beaches</b> stay unsafe longer than LA County's 72-hour rain advisory.",
            f"After storms of 1 inch or more, they were still unsafe on <b>{pct(slow_big)}</b> of tests 4 to 6 days "
            f"later. Their normal level is {pct(rg['slow']['normal_wet_season_rate'])}. {also_dry}",
            notes=[NORMAL_NOTE, ADVISORY_NOTE])

    with panel():
        size = st.segmented_control("Storms", ["All storms", "1 inch or more"], default="All storms",
                                    key="adv_size") or "All storms"
        key = "all storms" if size == "All storms" else "1 in or more"
        rows = []
        for g in ("slow", "typical", "fast"):
            for c in rg[g]["curves"][key]:
                rows.append({"group": GROUPS[g]["word"], "day": c["day"],
                             "unsafe": c["unsafe_rate"] * 100, "tests": c["tests"],
                             "normal": rg[g]["normal_wet_season_rate"] * 100})
        df = pd.DataFrame(rows)
        title(f"Share of tests unsafe, by days since rain ({size.lower()})")
        # No separate legend: each line is named at its right end, in its own color.
        dom = [GROUPS[g]["word"] for g in ("slow", "typical", "fast")]
        color = alt.Color("group:N", scale=alt.Scale(domain=dom, range=[GROUPS[g]["hex"] for g in ("slow", "typical", "fast")]),
                          legend=None)
        top = max(df["unsafe"].max(), 10)
        lines = alt.Chart(df).mark_line(strokeWidth=2, point=alt.OverlayMarkDef(size=45, filled=True)).encode(
            x=day_axis(right=14.8), y=alt.Y("unsafe:Q", title="% of tests unsafe", scale=alt.Scale(domain=[0, top * 1.08])),
            color=color,
            tooltip=[alt.Tooltip("group:N", title="Beaches"), alt.Tooltip("day:Q", title="Days since rain"),
                     alt.Tooltip("unsafe:Q", title="% unsafe", format=".0f"), alt.Tooltip("tests:Q", title="Tests")])
        # Each group's normal level (wet-season days with no rain for 8+ days): thin dashed line, same color.
        normals = alt.Chart(df.drop_duplicates("group")).mark_rule(strokeDash=[4, 4], strokeWidth=1, opacity=.7).encode(
            y="normal:Q", color=color)
        # Each line's name, in the empty space right of day 10 (two short lines of text each).
        ends = end_labels(df, "group", "unsafe", top)
        names = alt.Chart(ends).mark_text(align="left", dx=8, fontSize=10.5, fontWeight="bold", lineHeight=12,
                                          lineBreak="\n").encode(
            x="day:Q", y="label_y:Q", text="lines:N", color=color)
        show_chart(alt.layer(advisory_band(top), normals, lines, names).properties(height=330), width="stretch")
        st.caption("Shaded: days 0 to 3, inside the 72-hour advisory (we only know the day of the rain, not the hour, "
                   "so day 3 is roughly 48 to 96 hours later). Dashed lines: each group's normal level. Beaches often "
                   "unsafe in dry weather are left out of the lower two lines.")
        numbers(df[["group", "day", "unsafe", "tests", "normal"]], f"after-rain-curves_{key.replace(' ', '-')}",
                group="Beaches", day="Days since rain", unsafe=st.column_config.NumberColumn("% unsafe", format="%.1f"),
                tests="Tests", normal=st.column_config.NumberColumn("Normal %", format="%.1f"))

    with panel():
        title("Tides matter too")
        st.markdown(f"Three to seven days after rain, **{pct(tide['big_tide']['mean'])}** of tests were unsafe on big "
                    f"spring-tide days, vs. **{pct(tide['other']['mean'])}** on other days (all beaches). That fits a lagoon "
                    "or storm drain that keeps emptying runoff as the tide goes out.")

    with st.expander("The three groups in numbers"):
        cards([
            (n["slow"], GROUPS["slow"]["word"], group_line("slow", rg)
             + (f"; {slow_chronic} also often unsafe in dry weather" if slow_chronic else ""), GROUPS["slow"]["hex"]),
            (n["typical"], GROUPS["typical"]["word"], group_line("typical", rg), GROUPS["typical"]["hex"]),
            (n["fast"], GROUPS["fast"]["word"], group_line("fast", rg), GROUPS["fast"]["hex"]),
        ])
        st.markdown(f"The {n['fast']} beaches that recover faster than most were unsafe on {pct(fast_23)} of tests 2 to 3 "
                    f"days after rain, vs. {pct(fast_normal)} normally ({pct(fast_big)} after storms of 1 inch or more).")

    # Map: every diagnosed beach colored by its group.
    rows = []
    for s in DIAG["stations"]:
        g = group_of(s)
        rows.append({"station": s["station"], "lat": s["lat"], "lon": s["lon"], "place": place(s["station"]),
                     "group": GROUPS[g]["word"], "color": GROUPS[g]["rgb"], "size": 8 if g in ("slow", "chronic") else 6,
                     "detail": group_rates(s)[0].upper() + group_rates(s)[1:],
                     "order": ["other", "fast", "typical", "chronic", "slow"].index(g)})
    m = pd.DataFrame(rows).sort_values("order")  # slow beaches drawn last, on top
    with panel():
        title("Where they are")
        base_map([pdk.Layer("ScatterplotLayer", data=m, get_position=["lon", "lat"], get_fill_color="color",
                            get_radius="size", radius_units=PIXELS, stroked=True, get_line_color=[255, 255, 255],
                            line_width_min_pixels=1, pickable=True)],
                 tooltip="<b>{place}</b><br/>{group}<br/>{detail}")
        st.html(chips([(GROUPS[g]["hex"], GROUPS[g]["word"]) for g in GROUPS]))

    # The slow beaches, one row each (one block each on a phone).
    slow = [s for s in DIAG["stations"] if group_of(s) == "slow"]
    slow.sort(key=lambda s: -s["groups"]["days 4-6"]["unsafe_rate"])
    with panel():
        title(f"The {len(slow)} beaches that stay unsafe longer than 72 hours")
        st.markdown("All sit next to a big storm drain, creek or river mouth. After rain they fail far more often on "
                    "big spring-tide days.")
        rows_table([("Beach", False), ("Often unsafe in dry weather", False), ("Days 4-6 after rain", True),
                    ("Normal", True), ("Days 3-7, big tide", True), ("Days 3-7, other tides", True)],
                   [[html.escape(label(s["station"])), "Yes" if s["chronic"] else "No",
                     rate_n(s["groups"]["days 4-6"]["unsafe_rate"], s["groups"]["days 4-6"]["tests"]),
                     pct(s["normal_wet_season_rate"]),
                     rate_n(s["after_rain_tides"]["big_tide_rate"], s["after_rain_tides"]["big_tide_tests"]),
                     rate_n(s["after_rain_tides"]["other_rate"], s["after_rain_tides"]["other_tests"])]
                    for s in slow])
        st.caption(f"In brackets: the number of tests behind each rate. Rates from fewer than {MIN_TESTS} tests "
                   "are not shown.")
        csv_button(diagnosis_table(), "all-beaches-after-rain", "Download all beaches (CSV)")

    counts = DIAG["recovery_counts"]
    with st.expander("How beaches were sorted, and the ones we couldn't sort"):
        st.markdown(
            "- **Normal** = the beach's unsafe rate in the wet season (November to April) on days with no 0.1 in rain "
            "for 8+ days.\n"
            "- **Clearly above normal** = the 95% range of (rate minus normal) is entirely above zero. Ranges come "
            "from resampling whole storms 1,000 times, because tests after the same storm aren't independent.\n"
            f"- **{GROUPS['slow']['word']}**: clearly above normal on days 4 to 6.\n"
            f"- **{GROUPS['fast']['word']}**: days 2-3 and 4-6 both within 5 points of normal and not clearly above.\n"
            f"- **{GROUPS['typical']['word']}**: clearly above normal on days 2-3, near normal on days 4-6 "
            f"({pct(typical_46)} unsafe on days 4 to 6 vs. {pct(rg['typical']['normal_wet_season_rate'])} normally).\n"
            f"- **{GROUPS['chronic']['word']}**: unsafe on 10% or more of dry-weather tests since {RECENT_YEAR}.\n"
            f"- Not sorted: {counts.get('unclear (too few tests to tell)', 0)} beaches where the ranges were too wide to "
            f"tell, {counts.get('not enough tests', 0)} with fewer than 15 tests in a group (Avalon and many Ventura "
            f"beaches), and {counts.get('little change after rain', 0)} that rarely fail at all.\n"
            f"- This is a pattern in 16 years of tests at {COVERAGE}, not a health study. It supports a rule that "
            "depends on storm size and on the beach; it doesn't prove any single day was safe.")


def diagnosis_table():
    """Every diagnosed beach in one table (rates in %, with the number of tests behind each)."""
    t = pd.DataFrame([{"Beach": label(x["station"]), "Station": x["station"], "Pattern": GROUPS[group_of(x)]["word"],
                       "Dry weather": x["dry_unsafe_rate"], "Dry tests": x["dry_tests"],
                       "During advisory": x["advisory_unsafe_rate"], "Advisory tests": x["advisory_tests"],
                       "Days 4-6": x["groups"]["days 4-6"]["unsafe_rate"], "Days 4-6 tests": x["groups"]["days 4-6"]["tests"],
                       "Normal": x["normal_wet_season_rate"], "Trigger": x["trigger"] or ""} for x in DIAG["stations"]])
    for c in ["Dry weather", "During advisory", "Days 4-6", "Normal"]:
        t[c] = (t[c].astype(float) * 100).round(1)
    return t.sort_values("Days 4-6", ascending=False)


def group_line(g, rg):
    """One line of numbers for a whole group, from the region's pooled curves."""
    curve, normal = rg[g]["curves"]["all storms"], pct(rg[g]["normal_wet_season_rate"])
    if g == "fast":
        return f"days 2 to 3 after rain: {pct(pooled(curve, 2, 3))} unsafe; normal level {normal}"
    if g == "typical":
        return (f"days 2 to 3: {pct(pooled(curve, 2, 3))} unsafe; days 4 to 6: {pct(pooled(curve, 4, 6))}; "
                f"normal level {normal}")
    return f"days 4 to 6 after rain: {pct(pooled(curve, 4, 6))} unsafe; normal level {normal}"


# ---------------------------------------------------------------------------
# Page 2: Beach profile
# ---------------------------------------------------------------------------

def diagnosis_sentence(s):
    """(headline, details) for one beach: the headline has one idea and one number;
    everything else goes in the details line below it."""
    p = html.escape(place(s["station"]))
    g = s["groups"]
    norm = pct(s["normal_wet_season_rate"])
    dry = pct(s["dry_unsafe_rate"])
    d23, d46 = pct(g["days 2-3"]["unsafe_rate"]), pct(g["days 4-6"]["unsafe_rate"])
    if s["recovery"] == "still unsafe after 72 hours":
        return (f"{p} is still unsafe on <b>{d46}</b> of tests 4 to 6 days after rain.",
                f"That is past the 72-hour advisory; its normal level is {norm}."
                + (f" It is also often unsafe in dry weather: {dry} of dry-weather tests since {RECENT_YEAR}."
                   if s["chronic"] else ""))
    if s["chronic"]:
        return (f"{p} is unsafe on <b>{dry}</b> of dry-weather tests.",
                f"Tests since {RECENT_YEAR}, on days with no rain for 8 or more days.")
    return {
        "back to normal in about a day":
            (f"{p} is back to normal about a day after rain: <b>{d23}</b> unsafe on days 2 to 3.",
             f"Its normal level is {norm}."),
        "back to normal within 72 hours":
            (f"{p} clears within the 72-hour advisory: <b>{d46}</b> unsafe on days 4 to 6.",
             f"{d23} of tests were unsafe on days 2 to 3; its normal level is {norm}."),
        "little change after rain":
            (f"{p} is rarely unsafe: <b>{dry}</b> of dry-weather tests.",
             f"{pct(s['advisory_unsafe_rate'])} during rain advisories; tests since {RECENT_YEAR}."),
    }.get(s["recovery"],
          (f"{p} has too few tests after rain to tell how fast it recovers.",
           f"In dry weather since {RECENT_YEAR}, {dry} of its tests were unsafe."))


def enough(rate, n):
    """A big tile number: the rate, or 'Too few tests' (in smaller type) when fewer than MIN_TESTS."""
    return pct(rate) if n and n >= MIN_TESTS else '<span class="v-small">Too few tests</span>'


ADVISORY_WORDS = {"Posting": "Warning sign", "Closure": "Closed", "Rain": "Rain advisory"}


def posting_history(code):
    """The beach's OFFICIAL advisory record from the state's Beach Watch database (pipeline step 13):
    days under a warning sign, closure or rain advisory each year, and the latest advisories."""
    pj = load_json("postings.json")
    through = pd.Timestamp(pj["records_through"])
    with panel():
        title(f"Official posting history (records through {through:%B} {through.day}, {through.year})")
        rows = pj["stations"].get(code)
        if not rows:
            st.markdown("No Beach Watch posting records for this station. That can mean it was never posted, or that "
                        "the county records it under another station name.")
            return
        df = pd.DataFrame(rows, columns=pj["columns"])
        df["start"] = pd.to_datetime(df["start"])
        df["end"] = pd.to_datetime(df["end"])
        # Days each advisory was up. One still open at the end of the records counts up to that date.
        df["days"] = ((df["end"].fillna(through + pd.Timedelta(days=1)) - df["start"]).dt.days).clip(lower=1)
        df["kind"] = df["type"].map(ADVISORY_WORDS).fillna(df["type"])
        df["year"] = df["start"].dt.year
        recent = df[df["year"] >= 2010]  # the same years as the test data
        posted = recent[recent["type"] == "Posting"]
        n_signs, n_closed, n_rain = len(posted), (recent["type"] == "Closure").sum(), (recent["type"] == "Rain").sum()
        st.markdown(f"Since 2010: **{n_signs} warning {'sign' if n_signs == 1 else 'signs'}** "
                    f"({posted['days'].sum():,} days posted), {n_closed} {'closure' if n_closed == 1 else 'closures'} "
                    f"and {n_rain} recorded rain {'advisory' if n_rain == 1 else 'advisories'}.")
        per_year = recent.groupby(["year", "kind"], as_index=False)["days"].sum()
        kinds = [w for w in ADVISORY_WORDS.values() if w in set(per_year["kind"])]
        colors = {"Warning sign": ORANGE, "Closed": LEVELS["high"]["hex"], "Rain advisory": BLUE}
        st.html(chips([(colors[k], k) for k in kinds]))
        bars = alt.Chart(per_year).mark_bar(cornerRadiusTopLeft=2, cornerRadiusTopRight=2).encode(
            x=alt.X("year:O", title=None, axis=alt.Axis(labelAngle=0, values=list(range(2010, through.year + 1, 2)))),
            y=alt.Y("days:Q", title="Days under advisory", stack=True),
            color=alt.Color("kind:N", scale=alt.Scale(domain=kinds, range=[colors[k] for k in kinds]), legend=None),
            tooltip=[alt.Tooltip("year:O", title="Year"), alt.Tooltip("kind:N", title=" "),
                     alt.Tooltip("days:Q", title="Days")])
        show_chart(bars.properties(height=200), width="stretch")
        st.caption(f"{through.year} is partial: the state's records end {through:%B} {through.day}. Source: State "
                   "Water Board, Beach Watch (data.ca.gov). Rain advisories cover all of a health department's beaches "
                   "at once, so they appear for every beach it covers. Days can overlap when a sign and a rain "
                   "advisory are up together.")
        with st.expander("Latest advisories"):
            last = df.sort_values("start", ascending=False).head(15)
            rows_table([("Started", False), ("Type", False), ("Lifted", False), ("Days", True), ("Cause", False)],
                       [[f"{r.start:%b %d, %Y}", r.kind, "still open" if pd.isna(r.end) else f"{r.end:%b %d, %Y}",
                         str(r.days), html.escape(r.cause or "")] for r in last.itertuples()])
        numbers(per_year.rename(columns={"kind": "type"}), f"postings-per-year_{code}", year=st.column_config.NumberColumn("Year", format="%d"),
                type="Advisory", days="Days")


def diagnosis_page():
    # Start on the beach in the link, or the one picked on Forecast, or else the beach that
    # stays unsafe longest after the advisory.
    slow = [s for s in DIAG["stations"] if group_of(s) == "slow"]
    first = max(slow, key=lambda s: s["groups"]["days 4-6"]["unsafe_rate"])["station"] if slow else DIAG["stations"][0]["station"]
    codes = sorted(DIAG_BY, key=place)
    chosen_beach("diag_beach", codes, first)
    top = st.container()  # the finding sits above the picker, but is written after it (it depends on the choice)
    code = st.selectbox("Beach", codes, key="diag_beach", format_func=label,
                        help="Type to search. The link in your browser's address bar opens this beach.")
    remember_beach(code, "diag_beach")
    s = DIAG_BY[code]
    g = group_of(s)
    with top:
        head, details = diagnosis_sentence(s)
        finding(f"Beach profile, station {code}", head, details,  # many beaches have several stations
                notes=[NORMAL_NOTE, PRECAUTION if s["recovery"].startswith("back to normal") else None])

    with panel():
        title("Share of tests unsafe, by days since rain (all years)")
        df = pd.DataFrame(s["by_day"])
        df["unsafe"] = df["unsafe_rate"].astype(float) * 100
        df["few"] = np.where(df["tests"] < 8, "Fewer than 8 tests", "8 or more tests")
        normal = s["normal_wet_season_rate"] * 100
        top = max(df["unsafe"].max(), normal, 10)
        bars = alt.Chart(df).mark_bar(size=16, color=BLUE, cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
            x=day_axis(right=12.3), y=alt.Y("unsafe:Q", title="% of tests unsafe", scale=alt.Scale(domain=[0, top * 1.12])),
            opacity=alt.Opacity("few:N", scale=alt.Scale(domain=["8 or more tests", "Fewer than 8 tests"], range=[1, .35]),
                                legend=None),
            tooltip=[alt.Tooltip("day:Q", title="Days since rain"), alt.Tooltip("unsafe:Q", title="% unsafe", format=".0f"),
                     alt.Tooltip("tests:Q", title="Tests")])
        line = pd.DataFrame({"y": [normal], "x": [-0.5], "x2": [10.5], "label": [f"Normal\nlevel {normal:.0f}%"]})
        rule = alt.Chart(line).mark_rule(strokeDash=[4, 4], color=NAVY).encode(y="y:Q", x="x:Q", x2="x2:Q")
        # The label sits past the last bar, where nothing can cover it.
        rule_text = alt.Chart(line).mark_text(align="left", baseline="middle", dx=6, color=NAVY, fontSize=11,
                                              lineHeight=13, lineBreak="\n").encode(y="y:Q", x="x2:Q", text="label:N")
        show_chart((advisory_band(top) + bars + rule + rule_text).properties(height=290), width="stretch")
        st.caption("Shaded: inside the 72-hour advisory. Dashed line: this beach's normal level. "
                   "Pale bars have fewer than 8 tests, so they jump around by chance.")
        numbers(df[["day", "unsafe", "tests"]], f"by-day-since-rain_{code}", day="Days since rain",
                unsafe=st.column_config.NumberColumn("% unsafe", format="%.1f"), tests="Tests")

    posting_history(code)

    trig = s["trigger"] or "no storm size reached 1 in 4"
    tide = s["after_rain_tides"]
    with st.expander("This beach in numbers"):
        cards([
            (enough(s["dry_unsafe_rate"], s["dry_tests"]), f"of dry-weather tests unsafe (since {RECENT_YEAR})",
             f"{s['dry_tests']} tests"),
            (enough(s["advisory_unsafe_rate"], s["advisory_tests"]), f"unsafe during rain advisories (since {RECENT_YEAR})",
             f"days 0 to 3 after rain, {s['advisory_tests']} tests"),
            (html.escape(trig), "rain that makes 1 in 4 tests fail", "storm total, tests on days 0 to 1"),
            ('<span class="v-small">' + GROUPS[g]["word"] + "</span>", "its group on the After-rain patterns page",
             group_rates(s), GROUPS[g]["hex"]),
        ])

    with st.expander("What the pattern points to"):
        st.markdown(s["fix_text"] or "There aren't enough tests after rain to point to a kind of fix.")
        if s.get("also_after_rain"):
            st.markdown(s["also_after_rain"])
        st.caption("A pattern in the test results is a clue, not proof of the source. Source-tracking tests "
                   "(for example, DNA markers for human sewage) can confirm it.")

    with st.expander("Days 0 to 1 after rain, by storm size"):
        st.caption("Dashed line: 1 in 4 tests unsafe. Pale bars have fewer than 8 tests.")
        ss = pd.DataFrame(s["by_storm_size"])
        ss["unsafe"] = ss["unsafe_rate"].astype(float) * 100
        ss["few"] = np.where(ss["tests"] < 8, "Fewer than 8 tests", "8 or more tests")
        sb = alt.Chart(ss).mark_bar(size=22, color=BLUE, cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
            opacity=alt.Opacity("few:N", scale=alt.Scale(domain=["8 or more tests", "Fewer than 8 tests"], range=[1, .35]),
                                legend=None),
            x=alt.X("storm:N", sort=list(ss["storm"]), title="Storm total", axis=alt.Axis(labelAngle=0)),
            y=alt.Y("unsafe:Q", title="% unsafe", scale=alt.Scale(domain=[0, 100])),
            tooltip=[alt.Tooltip("storm:N", title="Storm"), alt.Tooltip("unsafe:Q", title="% unsafe", format=".0f"),
                     alt.Tooltip("tests:Q", title="Tests")])
        q = alt.Chart(pd.DataFrame({"y": [25]})).mark_rule(strokeDash=[4, 4], color=NAVY).encode(y="y:Q")
        show_chart((sb + q).properties(height=210), width="stretch")
        if tide["big_tide_tests"] >= 10:
            st.caption(f"Tides matter too: 3 to 7 days after rain, {pct(tide['big_tide_rate'])} of tests were unsafe on "
                       f"big spring-tide days ({tide['big_tide_tests']} tests) vs. {pct(tide['other_rate'])} on other days.")

    with st.expander("All beaches"):
        t = diagnosis_table()
        fmt = st.column_config.NumberColumn(format="%.0f%%")
        st.dataframe(t, hide_index=True, width="stretch",
                     column_config={c: fmt for c in ["Dry weather", "During advisory", "Days 4-6", "Normal"]})
        csv_button(t, "all-beaches-profile")
        st.caption(f"Dry weather and during-advisory rates use tests since {RECENT_YEAR}, so a beach that was fixed "
                   "isn't still called chronic. Days 4-6 and normal use all years since 2010, because recovery needs many storms.")


# ---------------------------------------------------------------------------
# Compare beaches: 2 or 3 beaches side by side
# ---------------------------------------------------------------------------

def km_apart(a, b):
    """Straight-line distance between two stations in km (haversine: distance on a sphere)."""
    la1, lo1, la2, lo2 = np.radians([STATIONS.at[a, "lat"], STATIONS.at[a, "lon"], STATIONS.at[b, "lat"], STATIONS.at[b, "lon"]])
    h = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return 2 * 6371 * np.arcsin(np.sqrt(h))


def nearest(code, options, n):
    """The n stations in `options` closest to `code` (not counting itself)."""
    return sorted((c for c in options if c != code), key=lambda c: km_apart(code, c))[:n]


def compare_page():
    fc = load_json("forecast.json")
    fc_by_code = {s["station"]: s for s in fc["stations"]}
    codes = sorted(fc_by_code, key=place)
    today = la_today()
    # Start with the beach in a new link (?beach=CODE), else the one picked on another page (or the
    # riskiest today), plus its 2 nearest beaches.
    link = link_beach("cmp")
    if link in fc_by_code or "cmp_beaches" not in st.session_state:
        first = link if link in fc_by_code else st.session_state.get("shared_beach")
        if first not in fc_by_code:
            first = max(codes, key=lambda c: fc_by_code[c]["forecast"][today_index(fc_by_code[c], today)]["risk"])
        st.session_state["cmp_beaches"] = [first] + nearest(first, codes, 2)
    top = st.container()  # the finding sits above the picker, but is written after it
    picked = st.multiselect("Beaches (2 or 3)", codes, key="cmp_beaches", max_selections=3, format_func=label,
                            help="Type to search. Beaches with a forecast are listed.")
    if picked:
        remember_beach(picked[0], "cmp")
    if len(picked) < 2:
        with top:
            finding("Compare beaches", "Pick 2 or 3 beaches to compare.",
                    "Each one's risk today, latest test, recovery after rain and dry-weather record, side by side.")
        return

    day = {c: fc_by_code[c]["forecast"][today_index(fc_by_code[c], today)] for c in picked}
    riskiest = max(picked, key=lambda c: day[c]["risk"])
    pairs = [(a, b) for i, a in enumerate(picked) for b in picked[i + 1:]]
    with top:
        finding("Compare beaches",
                f"{html.escape(place(riskiest))} has the highest risk today: <b>{pct(day[riskiest]['risk'])}</b>.",
                f"{len(picked)} beaches, up to {max(km_apart(a, b) for a, b in pairs):.1f} km apart. Each one's risk "
                "today, latest test, recovery after rain and dry-weather record.",
                notes=[NORMAL_NOTE])

    # One tile per beach: today's risk, in its level color.
    cards([(pct(day[c]["risk"]), html.escape(label(c)),
            f"{LEVELS[day[c]['level']]['word']} today ({nice_date(day[c]['date'])})", LEVELS[day[c]["level"]]["hex"])
           for c in picked])

    # Side by side: one row per measure, one column per beach (one block per measure on a phone).
    def latest(c):
        t = fc_by_code[c]["last_test"]
        age = (pd.Timestamp(today) - pd.Timestamp(t["date"])).days
        return (f"{t['value']:,.0f} per 100 mL, {'over' if t['unsafe'] else 'under'} the limit; "
                f"{pd.Timestamp(t['date']):%b %d, %Y} ({ago(t['date'], today)})" + ("; over 30 days old" if age > 30 else ""))

    def diag(c, what):
        s = DIAG_BY.get(c)
        if s is None:
            return "Not enough tests to tell"
        if what == "group":
            return GROUPS[group_of(s)]["word"]
        if what == "dry":
            return rate_n(s["dry_unsafe_rate"], s["dry_tests"])
        if what == "normal":
            return pct(s["normal_wet_season_rate"])
        g = s["groups"][what]
        return rate_n(g["unsafe_rate"], g["tests"])

    measures = [
        ("Forecast risk today", lambda c: f"{pct(day[c]['risk'])} ({LEVELS[day[c]['level']]['word']})"),
        ("Latest test", latest),
        ("After-rain recovery", lambda c: diag(c, "group")),
        ("Unsafe 2 to 3 days after rain", lambda c: diag(c, "days 2-3")),
        ("Unsafe 4 to 6 days after rain", lambda c: diag(c, "days 4-6")),
        ("Normal level", lambda c: diag(c, "normal")),
        (f"Unsafe in dry weather (since {RECENT_YEAR})", lambda c: diag(c, "dry")),
        ("All tests", lambda c: f"{STATIONS.at[c, 'samples']:,} since {STATIONS.at[c, 'first'][:4]}"),
        ("Distance to the others", lambda c: "; ".join(f"{km_apart(c, o):.1f} km to {o}" for o in picked if o != c)),
    ]
    with panel():
        title("Side by side")
        rows_table([("", False)] + [(label(c), False) for c in picked],
                   [[html.escape(m)] + [html.escape(f(c)) for c in picked] for m, f in measures])
        st.caption(f"In brackets: the number of tests behind each rate; rates from fewer than {MIN_TESTS} tests "
                   "are not shown. Recovery and normal level use all years since 2010; dry weather uses recent years. "
                   + PRECAUTION)
        table = pd.DataFrame({label(c): [f(c) for _, f in measures] for c in picked}, index=[m for m, _ in measures])
        csv_button(table.reset_index(names="measure"), "compare_" + "_".join(picked))

    with panel():
        title("Where they are")
        m = pd.DataFrame([{"station": c, "lat": STATIONS.at[c, "lat"], "lon": STATIONS.at[c, "lon"], "place": place(c),
                           "status": f"{LEVELS[day[c]['level']]['word']} ({pct(day[c]['risk'])})",
                           "color": LEVELS[day[c]["level"]]["rgb"]} for c in picked])
        span = max(km_apart(a, b) for a, b in pairs)
        zoom = 12 if span < 3 else 10.5 if span < 15 else 9 if span < 50 else 7.8  # closer beaches, closer map
        base_map([pdk.Layer("ScatterplotLayer", data=m, get_position=["lon", "lat"], get_fill_color="color",
                            get_radius=9, radius_units=PIXELS, stroked=True, get_line_color=[11, 37, 69],
                            line_width_min_pixels=2, pickable=True)],
                 lat=m["lat"].mean(), lon=m["lon"].mean(), zoom=zoom, phone_zoom=zoom - 0.8, height=300,
                 tooltip="<b>{place}</b><br/>{status}")
        st.html(chips([(LEVELS[k]["hex"], LEVELS[k]["word"]) for k in LEVELS]))


# ---------------------------------------------------------------------------
# Page 3: Cleanup projects (the intervention evaluator)
# ---------------------------------------------------------------------------

@st.cache_resource(ttl="1h")
def eval_table(version):
    """Every beach test with its weather-only expected risk (from step 12), as a table.
    `version` (the file's change time) is only the cache key."""
    d = load_json("evaluator_data.json")
    st_ = pd.DataFrame(d["stations"])
    s = np.array(d["s"])
    return pd.DataFrame({
        "StationCode": st_["station"].to_numpy()[s],
        "date": pd.Timestamp(d["day0"]) + pd.to_timedelta(d["d"], unit="D"),
        "lat": st_["lat"].to_numpy()[s], "lon": st_["lon"].to_numpy()[s],
        "exceeds": d["y"], "expected": np.array(d["e"]) / 1000})


@st.cache_data(ttl="1h", show_spinner=False)
def run_eval(station, start, end, years, version):
    return evaluate(eval_table(version), station, start, end, years, load_json("evaluator_data.json")["widen"])


# News articles for each known project's dates (see DATES_NOTE). Keys start like the project labels.
PROJECT_SOURCES = {
    "Malibu Lagoon": [("The Inertia", "https://www.theinertia.com/surf/malibu-lagoon-project-manager-responds-to-allegations-that-lagoon-restoration-is-ruining-surfrider-beach/"),
                      ("SMDP", "https://www.smdp.com/malibu-lagoon-reopens-after-massive-reconstruction/")],
    "Santa Monica Pier": [("Santa Monica Lookout", "https://santamonicalookout.com/ssm_site/the_lookout/news/News-2023/August-2023/08_21_2023_Water_Quality_Under_Pier_Gives_Santa_Monica_Beaches_a_Bum_Rap.html")],
}


def project_report(r, code, preset, years, calib):
    """A one-page report of one cleanup-project check, as a web page (open it, then print or
    save as PDF). Everything in it comes from the evaluator's result `r`."""
    b, a = r["before"], r["after"]
    est, (lo, hi) = r["change_beyond_weather"], r["ci95"]
    what = preset["label"] if preset else "Beach and dates chosen by the user"
    news = next((v for k, v in PROJECT_SOURCES.items() if preset and preset["label"].startswith(k)), [])
    e = html.escape
    row = lambda k, v: f"<tr><th>{e(k)}</th><td>{v}</td></tr>"
    window = lambda w: f"{pd.Timestamp(w[0]):%b %d, %Y} to {pd.Timestamp(w[1]):%b %d, %Y}"
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>After the Rain: cleanup project check, {e(place(code))}</title>
<style>
  @page {{ size: letter; margin: 0.5in; }}
  body {{ font-family: "Public Sans", Arial, sans-serif; color: #1A2440; font-size: 11pt; line-height: 1.4;
          max-width: 7.3in; margin: 24px auto; padding: 0 16px; }}
  h1 {{ font-size: 18pt; margin: 0 0 4px; }} h2 {{ font-size: 12pt; margin: 12px 0 2px; }}
  .sub {{ color: #4E5B72; margin: 0 0 12px; }} .verdict {{ font-size: 14pt; font-weight: 700; margin: 8px 0; }}
  table {{ border-collapse: collapse; width: 100%; }} th, td {{ text-align: left; padding: 3px 8px 3px 0;
          border-bottom: 1px solid #DCE4EE; vertical-align: top; }} th {{ width: 42%; font-weight: 600; }}
  ul {{ margin: 4px 0; padding-left: 18px; }} li {{ margin: 2px 0; }} .small {{ font-size: 9pt; color: #4E5B72; }}
  /* last, so it wins over the rules above when printing */
  @media print {{ body {{ margin: 0; padding: 0; font-size: 10pt; }} h1 {{ font-size: 16pt; }} }}
</style></head><body>
<h1>Cleanup project check: {e(place(code))} ({e(code)})</h1>
<p class="sub">{e(what)}. Made by After the Rain on {datetime.now(LA):%B %d, %Y}.</p>
<p class="verdict">{e(r["verdict"].capitalize())}: {pts(est)} points of unsafe tests beyond what the weather and
other beaches explain (95% range {pts(lo)} to {pts(hi)}).</p>
<h2>Dates and samples</h2><table>
{row("Work started / finished", f"{pd.Timestamp(r['start']):%b %d, %Y} / {pd.Timestamp(r['end']):%b %d, %Y}" + (" (approximate, from news reports)" if preset else ""))}
{row("Before period", f"{window(r['before_window'])}: {b['tests']} tests")}
{row("After period", f"{window(r['after_window'])}: {a['tests']} tests")}
{row("Comparison beaches", f"{r['control_beaches']} beaches more than 5 km away")}
</table>
<h2>Result</h2><table>
{row("Unsafe before", f"{pct(b['unsafe_rate'])} of tests (weather alone predicts {pct(b['expected_from_weather'])})")}
{row("Unsafe after", f"{pct(a['unsafe_rate'])} of tests (weather alone predicts {pct(a['expected_from_weather'])})")}
{row("Change at the comparison beaches", f"{pts(r['control_change'])} points (subtracted out)")}
{row("Change beyond weather and region", f"<b>{pts(est)} points</b>")}
{row("95% range", f"{pts(lo)} to {pts(hi)} points" + ("" if years == 2 else " (approximate: the widening was set for 2-year comparisons)"))}
</table>
<h2>Assumptions</h2><ul>
<li>Expected risk from weather only: a model that knows the beach, rain (including the test day), days since rain,
tides and season, but not recent test results or the date. Each year is scored by a model trained on the other years.</li>
<li>Change = (beach after minus before) minus (beaches 5+ km away, after minus before), using observed minus expected.
The months of the work itself are left out.</li>
<li>95% range: resampling whole months 1,000 times, widened {calib['widen']}x so that only
{pct(calib['check_share_excluding_zero'])} of {calib['runs']} random beaches and dates look like a change (5% is the target).</li>
<li>"Cleaner after" is not "because of": one beach and one date can't rule out something else changing at the same time.
{e(POINTS_NOTE)}</li></ul>
<h2>Sources</h2><ul>
<li>Water tests: California State Water Boards, Safe to Swim (enterococcus over 104 per 100 mL counts as unsafe).</li>
<li>Rain: Open-Meteo (ERA5). Tides: NOAA.</li>
{"".join(f'<li>Project dates: <a href="{e(u)}">{e(n)}</a> ({e(u.split("/")[2])})</li>' for n, u in news)}</ul>
<p class="small">After the Rain is a student project for the 2026 Congressional App Challenge, not an official
health advisory. Model {e(str(load_json("final_metrics.json").get("model_id", "")))}.</p>
</body></html>"""


CUSTOM = "Choose my own beach and dates"


def dates_set():
    """Someone typed real dates, so the page can show its check (see "ev_waiting")."""
    st.session_state.pop("ev_waiting", None)


def use_preset():
    """When someone picks a known project, fill in its beach and dates."""
    dates_set()
    demos = {d["label"]: d for d in load_json("interventions.json")["demos"]}
    d = demos.get(st.session_state["ev_preset"])
    if d:
        st.session_state.update(ev_beach=d["station"], ev_start=pd.Timestamp(d["start"]).date(),
                                ev_end=pd.Timestamp(d["end"]).date())


def evaluator_page():
    iv = load_json("interventions.json")
    demos = {d["label"]: d for d in iv["demos"]}
    calib = load_json("evaluator_data.json")["calibration"]
    version = file_version("evaluator_data.json")
    ex = eval_table(version)
    counts = ex["StationCode"].value_counts()
    codes = sorted(counts[counts >= 100].index, key=place)
    link = link_beach("ev")
    if link:  # a link names a beach: its known project if it has one, else "choose my own" on that beach
        demo = next((d for d in demos.values() if d["station"] == link), None)
        if demo:
            st.session_state["ev_preset"] = demo["label"]
            use_preset()
        elif link in codes:
            # Placeholder dates in the middle of the beach's record, so there are tests before and after.
            mid = ex.loc[ex["StationCode"] == link, "date"].median().to_period("M").start_time.date()
            st.session_state.update(ev_preset=CUSTOM, ev_beach=link, ev_start=mid, ev_end=mid, ev_waiting=True)
    if "ev_preset" not in st.session_state:
        st.session_state["ev_preset"] = next(iter(demos))
        use_preset()

    top = st.container()  # the finding sits above the controls, but is written after them
    if link and link not in codes:
        st.caption(f"{place(link)} has too few tests for this check (it needs 100), so it isn't listed.")
    with panel():
        st.selectbox("Known project", list(demos) + [CUSTOM], key="ev_preset", on_change=use_preset)
        if st.session_state["ev_preset"] in demos:
            st.caption(DATES_NOTE)
        else:
            st.caption("Set the dates the work started and finished. The check compares the years before and after.")
        c1, c2, c3 = st.columns([3, 2, 2])
        code = c1.selectbox("Beach", codes, key="ev_beach", format_func=label)
        remember_beach(code, "ev")
        lo_d, hi_d = ex["date"].min().date(), ex["date"].max().date()
        start = c2.date_input("Work started", key="ev_start", min_value=lo_d, max_value=hi_d, on_change=dates_set)
        end = c3.date_input("Work finished", key="ev_end", min_value=lo_d, max_value=hi_d, on_change=dates_set)
        years = st.segmented_control("Years to compare before and after", [1, 2, 3], default=2, key="ev_years") or 2

    if st.session_state.get("ev_waiting"):  # opened from a link for a beach with no known project
        with top:
            finding("Cleanup projects", f"No known cleanup project at {html.escape(place(code))} yet.",
                    "Set the dates the work started and finished, and this page checks whether the beach was cleaner "
                    "after, beyond what the weather explains.")
        return

    preset = demos.get(st.session_state["ev_preset"])
    if preset and (code, str(start), str(end)) != (preset["station"], preset["start"], preset["end"]):
        preset = None  # the fields were changed by hand
    r = run_eval(code, str(start), str(end), years, version)
    if "error" in r:
        with top:
            finding("Cleanup projects", f"Not enough data to check {html.escape(place(code))} for these dates.",
                    html.escape(r["error"]) + ". Try a beach with more tests, or dates further from the start or end "
                    "of the record.")
        return

    name = place(code)  # the same name as in the Beach picker
    who = html.escape(name)
    what = html.escape(preset["short"] if preset else f"{pd.Timestamp(end):%B %Y}")
    est, (lo, hi) = r["change_beyond_weather"], r["ci95"]
    whole = lambda x: f"{abs(x) * 100:.1f}"   # one decimal, the same as the cards below
    signed = lambda x: f"{x * 100:+.1f}"
    # Headline: one idea and one number. The range and the method go on the line below.
    if r["verdict"] == "cleaner after":
        sentence = f"{who} was cleaner after {what}: <b>{whole(est)} points</b> fewer unsafe tests."
        detail = f"Fewer than the weather and other beaches predict (95% range {whole(hi)} to {whole(lo)} points)."
    elif r["verdict"] == "dirtier after":
        sentence = f"{who} was dirtier after {what}: <b>{whole(est)} points</b> more unsafe tests."
        detail = f"More than the weather and other beaches predict (95% range {whole(lo)} to {whole(hi)} points)."
    else:
        sentence = f"{who} showed no clear change after {what}: <b>{signed(est)} points</b>."
        detail = f"95% range {signed(lo)} to {signed(hi)} points, within normal year-to-year swings."
    detail += f" Based on {r['before']['tests']} tests before and {r['after']['tests']} after."
    with top:
        finding("Cleanup projects", sentence, detail, notes=[POINTS_NOTE])

    b, a = r["before"], r["after"]

    with panel():
        title("Monthly surprise: unsafe tests minus what the weather predicts")
        st.caption("Surprise: each month, the share of tests that were unsafe minus the share that rain, tides and "
                   "season predict. Above zero means dirtier than the weather explains.")
        st.html(chips([(BLUE, name), (MUTED, "Other beaches, 5+ km away")]))
        m = pd.DataFrame(r["monthly"])
        m["month"] = pd.to_datetime(m["month"])
        long = pd.concat([m.assign(who=name, v=m["beach"] * 100),
                          m.assign(who="Other beaches", v=m["others"] * 100)]).dropna(subset=["v"])
        color = alt.Color("who:N", scale=alt.Scale(domain=[name, "Other beaches"], range=[BLUE, MUTED]), legend=None)
        lines = alt.Chart(long).mark_line(strokeWidth=1.5, point=alt.OverlayMarkDef(size=28, filled=True)).encode(
            x=alt.X("month:T", title=None), y=alt.Y("v:Q", title="Points"),
            color=color, detail="period:N",
            tooltip=[alt.Tooltip("month:T", title="Month", format="%b %Y"), alt.Tooltip("who:N", title=" "),
                     alt.Tooltip("v:Q", title="Surprise (points)", format="+.0f")])
        zero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color="#c3cfd9").encode(y="y:Q")
        work = pd.DataFrame({"x": [pd.Timestamp(r["start"])], "x2": [pd.Timestamp(r["end"]) + pd.Timedelta(days=1)],
                             "label": ["Work"]})
        work_band = alt.Chart(work).mark_rect(color="#eef2f6").encode(x="x:T", x2="x2:T")
        work_text = alt.Chart(work).mark_text(align="left", baseline="top", dx=3, dy=4, color=MUTED, fontSize=11).encode(
            x="x:T", y=alt.value(0), text="label:N")
        # The beach's average surprise in each period, as a flat line with its value.
        tw = pd.DataFrame(r["monthly"])
        avg = []
        for per, (w0, w1) in [("before", r["before_window"]), ("after", r["after_window"])]:
            sub = tw[tw["period"] == per].dropna(subset=["beach"])
            v = (sub["beach"] * sub["beach_tests"]).sum() / sub["beach_tests"].sum() * 100
            avg.append({"x": pd.Timestamp(w0), "x2": pd.Timestamp(w1), "v": v,
                        "label": f"{'Before' if per == 'before' else 'After'}: {v:+.0f}"})
        avg = pd.DataFrame(avg)
        avg_rule = alt.Chart(avg).mark_rule(color=NAVY, strokeWidth=2).encode(x="x:T", x2="x2:T", y="v:Q")
        # Labels sit in the top corner of each period, clear of the monthly points.
        avg_text = alt.Chart(avg).mark_text(align="left", baseline="top", dx=4, dy=4, color=NAVY, fontSize=11,
                                            fontWeight="bold").encode(x="x:T", y=alt.value(0), text="label:N")
        show_chart((work_band + work_text + zero + lines + avg_rule + avg_text).properties(height=320), width="stretch")
        st.caption("Above zero = more unsafe tests than rain, tides and season predict for this beach. Navy lines: the "
                   "beach's average surprise before and after. Months with few tests jump around.")
        numbers(long[["month", "who", "v"]].rename(columns={"v": "points"}), f"monthly-surprise_{code}",
                month=st.column_config.DateColumn("Month", format="MMM YYYY"), who=" ",
                points=st.column_config.NumberColumn("Surprise (points)", format="%+.1f"))

    st.download_button("Download a one-page report", project_report(r, code, preset, years, calib).encode("utf-8"),
                       file_name=f"after-the-rain_project-check_{code}_{start}.html", mime="text/html",
                       icon=":material/description:", on_click="ignore", key="report",
                       help="A printable page with the dates, sample counts, change, 95% range, sources and "
                            "assumptions. Open it in a browser, then print or save as PDF.")

    with st.expander("Before and after, in numbers"):
        st.markdown(f"This compares the beach before and after, adjusts for the weather, and subtracts the change at "
                    f"{r['control_beaches']} beaches more than 5 km away. It shows what changed after the date; it "
                    "can't rule out something else changing at the same time.")
        cards([
            (pct(b["unsafe_rate"]), f"unsafe before ({r['before_window'][0][:4]} to {r['before_window'][1][:4]})",
             f"weather alone predicts {pct(b['expected_from_weather'])}; {b['tests']} tests"),
            (pct(a["unsafe_rate"]), f"unsafe after ({r['after_window'][0][:4]} to {r['after_window'][1][:4]})",
             f"weather alone predicts {pct(a['expected_from_weather'])}; {a['tests']} tests"),
            (f"{pts(r['control_change'])}", "points of change at other beaches", "subtracted out"),
            (f"{pts(est)}", "points of change beyond weather and region", f"95% range {pts(lo)} to {pts(hi)}"),
        ])

    with st.expander("How sure is this?"):
        iv_df = pd.DataFrame({"est": [est * 100], "lo": [lo * 100], "hi": [hi * 100], "y": ["Change"]})
        span = max(abs(lo), abs(hi), 0.05) * 100 * 1.15
        x = alt.X("lo:Q", title="Points of change beyond weather and region", scale=alt.Scale(domain=[-span, span]))
        rng = alt.Chart(iv_df).mark_rule(strokeWidth=3, color=BLUE).encode(x=x, x2="hi:Q", y=alt.Y("y:N", title=None, axis=None))
        dot = alt.Chart(iv_df).mark_point(filled=True, size=120, color=BLUE).encode(x="est:Q", y=alt.Y("y:N", axis=None))
        z = pd.DataFrame({"x": [0], "label": ["No change"]})
        zr = alt.Chart(z).mark_rule(color=NAVY, strokeDash=[4, 4], strokeWidth=1.5).encode(x="x:Q")
        zt = alt.Chart(z).mark_text(align="left", dx=4, y=6, baseline="top", color=NAVY, fontSize=11).encode(
            x="x:Q", text="label:N")
        show_chart((zr + zt + rng + dot).properties(height=90), width="stretch")
        st.markdown(
            f"The line is the 95% range. If it crosses zero, the change can't be told apart from normal swings. Beaches drift "
            f"from year to year for reasons we don't measure (sand, berms, birds, sampling), so a simple range was too "
            f"narrow: it called {pct(calib['plain_range_share_excluding_zero'])} of {calib['runs']} random beaches and "
            f"dates a real change. The ranges here are widened {calib['widen']}x, which brought a second set of "
            f"{calib['runs']} random dates down to {pct(calib['check_share_excluding_zero'])} (5% is the target).")
        if years != 2:
            st.caption("The widening was set for 2-year comparisons, so with "
                       f"{years} years the range is approximate.")

    with st.expander("Known projects and sources"):
        st.markdown("\n".join(f"- **{d['label']}** ({d['beach']}): {d['verdict']}, {pts(d['change_beyond_weather'])} "
                              f"points (95% range {pts(d['ci95'][0])} to {pts(d['ci95'][1])})" for d in iv["demos"]))
        st.markdown(f"{DATES_NOTE} Sources: "
                    + ", ".join(f"[{n}]({u})" for links in PROJECT_SOURCES.values() for n, u in links) + ".")


# ---------------------------------------------------------------------------
# Page 4: Fire impacts
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def palisades_points(version):
    """Every corner point of the Palisades burn outline, as (lon, lat) rows (same as pipeline step 07)."""
    gj = load_json("burn_areas_2025.geojson")
    pal = next(f for f in gj["features"] if f["properties"]["FIRE_NAME"].upper() == "PALISADES")["geometry"]
    rings = pal["coordinates"] if pal["type"] == "Polygon" else [r for poly in pal["coordinates"] for r in poly]
    return np.vstack([np.array(r)[:, :2] for r in rings])


def km_to_burn(code):
    """Distance in km from a station to the nearest point of the Palisades burn outline. Like step 07,
    which calls stations within 1 km 'below the burn area'. (1 degree of latitude = 111 km; of longitude
    = 92 km at LA's latitude.)"""
    v = palisades_points(file_version("burn_areas_2025.geojson"))
    lat, lon = STATIONS.at[code, "lat"], STATIONS.at[code, "lon"]
    return float(np.hypot((v[:, 1] - lat) * 111, (v[:, 0] - lon) * 92).min())


def fires_page():
    pf = load_json("post_fire.json")
    res, cells_ = pf["result"], pf["cells"]
    o, w = res["overall"], res["wet_days"]
    link = st.query_params.get("beach")
    link = link if link in STATIONS.index else None
    finding("Fire impacts", "No clear change in measured bacteria.",
            f"Beaches below the 2025 Palisades burn: <b>{pts(o['estimate'])} points</b> compared with what the model "
            f"expected (95% range {pts(o['ci95'][0])} to {pts(o['ci95'][1])}). After rain: {pts(w['estimate'])} points, "
            f"but that range ({pts(w['ci95'][0])} to {pts(w['ci95'][1])}) also includes zero.",
            notes=[POINTS_NOTE])
    if link:  # the page was opened for one beach (?beach=CODE): say where it fits in
        studied = {b["station"] for b in pf["burn_stations"]}
        km = km_to_burn(link)
        far = "less than 0.1 km" if km < 0.1 else f"{km:.1f} km"
        if link in studied:
            note = (f"<b>{html.escape(place(link))}</b> is {far} from the Palisades burn area: one of the "
                    f"{len(studied)} stations below the burn in this comparison. The result is for all {len(studied)} "
                    "together; one beach alone has too few tests after the fire to tell.")
        elif km <= 1:
            note = (f"<b>{html.escape(place(link))}</b> is {far} from the Palisades burn area, but has too few "
                    "tests to be part of this comparison.")
        else:
            note = (f"<b>{html.escape(place(link))}</b> is {km:.0f} km from the Palisades burn area, so it isn't one of "
                    "the stations below the burn (within 1 km) compared here.")
        with panel():
            st.html(f"<div>{note}</div>")

    m = pd.DataFrame(pf["monthly"])
    m["month"] = pd.to_datetime(m["month"])
    long = pd.concat([m.assign(group="Beaches near the burn", surprise=m["burn_residual"] * 100),
                      m.assign(group="All other beaches", surprise=m["control_residual"] * 100)])
    with panel():
        title("Monthly surprise: unsafe rate minus what the model expected")
        st.caption("Surprise: each month, the share of tests that were unsafe minus the share the forecast model "
                   "expected from rain, tides and each beach's history. Above zero means dirtier than expected.")
        st.html(chips([(ORANGE, "Beaches near the burn"), (MUTED, "All other beaches")]))
        lines = alt.Chart(long).mark_line(point=alt.OverlayMarkDef(size=30, filled=True), strokeWidth=1.5).encode(
            x=alt.X("month:T", title=None), y=alt.Y("surprise:Q", title="Points"),
            color=alt.Color("group:N", scale=alt.Scale(domain=["Beaches near the burn", "All other beaches"],
                                                       range=[ORANGE, MUTED]), legend=None),
            tooltip=[alt.Tooltip("month:T", title="Month", format="%b %Y"), alt.Tooltip("group:N", title=" "),
                     alt.Tooltip("surprise:Q", title="Surprise (points)", format="+.1f")])
        zero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color="#c3cfd9").encode(y="y:Q")
        fire = pd.DataFrame({"x": [pd.Timestamp("2025-01-07")], "label": ["Fire, Jan 7, 2025"]})
        fire_rule = alt.Chart(fire).mark_rule(strokeDash=[4, 4], color=NAVY).encode(x="x:T")
        fire_text = alt.Chart(fire).mark_text(align="left", dx=4, y=8, color=NAVY, fontSize=11).encode(x="x:T", text="label:N")
        show_chart((zero + lines + fire_rule + fire_text).properties(height=300), width="stretch")
        st.caption("Above zero = more unsafe tests than the forecast model expected from rain, tides and the beach's "
                   "history. Monthly values jump around because some months have few tests near the burn.")
        numbers(long[["month", "group", "surprise"]], "fire-monthly-surprise", month=st.column_config.DateColumn("Month", format="MMM YYYY"),
                group=" ", surprise=st.column_config.NumberColumn("Surprise (points)", format="%+.1f"))

    with st.expander("The numbers"):
        cards([(f"{pts(res[k]['estimate'])}", t, f"95% range {pts(res[k]['ci95'][0])} to {pts(res[k]['ci95'][1])}")
               for k, t in [("overall", "points, all days"), ("wet_days", "points, after rain"), ("dry_days", "points, dry days")]]
              + [(len(pf["burn_stations"]), "stations next to the burn", f"vs. {pf['control_station_count']} beaches 10+ km away")])
        st.markdown(f"After rain, beaches near the burn did {pts(w['estimate'])} points worse than expected, which is what "
                    "washed-in ash would look like, but the range includes zero: there were few rainy days after the "
                    "fire to learn from.")

    with st.expander("How we checked"):
        st.markdown(
            "1. The forecast model predicts each test from rain, tides and the beach's history. It knows nothing about fires.\n"
            "2. For every test since 2023 we measured the **surprise**: did the beach fail more often than expected?\n"
            "3. We compared the surprise **before vs. after** the fire, for beaches **next to the burn vs. 10+ km away**. "
            "The far beaches show what changed for other reasons (weather, labs). This is a *difference-in-differences*.\n"
            "4. The 95% range comes from resampling whole beaches 1,000 times.\n\n"
            f"Before the fire, beaches near the burn were already {pts(cells_['burn_before']['mean_residual'])} points "
            f"above expected, and {pts(cells_['burn_after']['mean_residual'])} after; other beaches moved from "
            f"{pts(cells_['control_before']['mean_residual'])} to {pts(cells_['control_after']['mean_residual'])}.")
        pfc = load_json("interventions.json")["post_fire_weather_only"]["overall"]
        st.markdown(f"As a check, the same comparison using the weather-only expected risk from the *Cleanup projects* page "
                    f"gives {pts(pfc['estimate'])} points (95% range {pts(pfc['ci95'][0])} to {pts(pfc['ci95'][1])}): "
                    "the same answer.")

    with panel():
        title("Burn areas and the beach stations next to them")
        near = pd.DataFrame(pf["burn_stations"])
        near = near.assign(lat=[STATIONS.at[c, "lat"] for c in near["station"]],
                           lon=[STATIONS.at[c, "lon"] for c in near["station"]],
                           place=[place(c) for c in near["station"]])
        base_map([pdk.Layer("GeoJsonLayer", data=load_json("burn_areas_2025.geojson"), get_fill_color=[235, 104, 52, 90],
                            get_line_color=[235, 104, 52], line_width_min_pixels=1),
                  pdk.Layer("ScatterplotLayer", data=near, get_position=["lon", "lat"], get_fill_color=[11, 37, 69],
                            get_radius=6, radius_units=PIXELS, stroked=True, get_line_color=[255, 255, 255],
                            line_width_min_pixels=1, pickable=True)]
                 + ([pdk.Layer("ScatterplotLayer",  # a ring around the beach named in the link
                               data=pd.DataFrame([{"lat": STATIONS.at[link, "lat"], "lon": STATIONS.at[link, "lon"]}]),
                               get_position=["lon", "lat"], filled=False, stroked=True, get_line_color=[11, 37, 69],
                               line_width_min_pixels=3, get_radius=13, radius_units=PIXELS)] if link else []),
                 lat=34.1, lon=-118.35, zoom=8.4, phone_zoom=8.0, height=320, tooltip="<b>{place}</b><br/>{km_from_burn} km from the burn area")
        st.html(chips([(ORANGE, "2025 burn areas (Palisades, left; Eaton, right)"), (NAVY, "Stations studied")]))
        st.caption(pf["eaton_note"] + " Fire perimeters: CAL FIRE.")


# ---------------------------------------------------------------------------
# Page 5: Forecast
# ---------------------------------------------------------------------------

@st.cache_resource
def refresh_state():
    """Shared by every visitor to this server: a lock, so only one forecast refresh runs at
    a time, and when the last try failed, so a broken connection isn't retried on every click."""
    return {"lock": threading.Lock(), "failed_at": 0.0, "error": None}


def refresh_forecast():
    """If forecast.json is older than today (LA time), rerun pipeline/08_forecast.py to make
    today's (it needs the internet for rain and tides; it takes about 10 seconds).
    08 only replaces the file when it finishes, so if it fails the last good forecast stays.
    Returns an error message if the forecast couldn't be updated, or None."""
    state = refresh_state()
    with state["lock"]:
        if load_json("forecast.json")["issued"] >= la_today():
            return None  # already today's (maybe another visitor just refreshed it)
        if time.time() - state["failed_at"] < 30 * 60:
            return state["error"]  # failed in the last 30 minutes: don't try again yet
        with st.spinner("Making today's forecast (about 10 seconds)..."):
            try:
                run = subprocess.run([sys.executable, str(ROOT / "pipeline" / "08_forecast.py")], cwd=ROOT,
                                     capture_output=True, text=True, timeout=180)
                error = None if run.returncode == 0 else (run.stderr or run.stdout).strip().splitlines()[-1]
            except Exception as e:  # e.g. it took too long
                error = str(e)
        state["failed_at"], state["error"] = (time.time(), error) if error else (0.0, None)
        return error


def today_index(s, day):
    """Which entry of a beach's forecast list is for `day` ('2026-10-25'); 0 if the list doesn't have it."""
    dates = [d["date"] for d in s["forecast"]]
    return dates.index(day) if day in dates else 0


def nice_date(day, year=False):
    """'2026-10-25' -> 'Sunday, October 25' (plus the year if asked)."""
    d = pd.Timestamp(day)
    return f"{d:%A, %B} {d.day}" + (f", {d.year}" if year else "")


def ago(day, today):
    """How long before `today` a date was, in words: 'today', '1 day ago', '6 days ago'."""
    n = (pd.Timestamp(today) - pd.Timestamp(day)).days
    return "today" if n <= 0 else "1 day ago" if n == 1 else f"{n} days ago"


def on_map_click():
    """When someone taps a dot on the map, select that beach in the picker."""
    objects = st.session_state["map"].selection.get("objects", {})
    for layer in ("beaches", "other_sites"):
        if objects.get(layer):
            st.session_state["beach"] = objects[layer][0]["station"]


def no_forecast_reason(code):
    """Short explanation for a site that has no forecast."""
    s = STATIONS.loc[code]
    if isinstance(s["note"], str):  # past study sites have a note; missing notes are blank (NaN)
        return s["note"].capitalize()
    if s["water"] == "freshwater":
        return "River or creek site (tested for E. coli); forecasts cover ocean beaches only"
    return "No tests in the past year, so no forecast"


def beach_map(fc_by_code, selected, day):
    """Map of beaches colored by the level on `day`, plus small grey dots for sites with no forecast."""
    rows = []
    for code, s in fc_by_code.items():
        d = s["forecast"][today_index(s, day)]
        lv = LEVELS[d["level"]]
        rows.append({"station": code, "lat": s["lat"], "lon": s["lon"], "place": place(code),
                     "status": f"{lv['word']} ({pct(d['risk'])})", "color": lv["rgb"], "size": lv["size"]})
    beaches = pd.DataFrame(rows)
    # Everything else that belongs on the map: past study sites, beaches with no recent tests,
    # and inland river/creek sites. They are small and grey so they never hide an active beach.
    others = STATIONS[STATIONS["on_map"] & ~STATIONS.index.isin(list(fc_by_code)) & ~STATIONS.index.isin(HIDE_FROM_MAP)]
    others = others.reset_index().assign(status=lambda d: [no_forecast_reason(c) for c in d["station"]])
    layers = [
        pdk.Layer("ScatterplotLayer", id="other_sites", data=others[["station", "lat", "lon", "place", "status"]],
                  get_position=["lon", "lat"], get_fill_color=GREY["rgb"], get_radius=4, radius_units=PIXELS,
                  pickable=True),
        pdk.Layer("ScatterplotLayer", id="beaches", data=beaches, get_position=["lon", "lat"],
                  get_fill_color="color", get_radius="size", radius_units=PIXELS,
                  stroked=True, get_line_color=[255, 255, 255], line_width_min_pixels=1, pickable=True),
    ]
    if selected in STATIONS.index:  # a navy ring around the selected beach
        ring = pd.DataFrame([{"lat": STATIONS.at[selected, "lat"], "lon": STATIONS.at[selected, "lon"]}])
        layers.append(pdk.Layer("ScatterplotLayer", data=ring, get_position=["lon", "lat"], filled=False,
                                stroked=True, get_line_color=[11, 37, 69], line_width_min_pixels=3,
                                get_radius=14, radius_units=PIXELS))
    base_map(layers, lat=33.86, lon=-118.78, zoom=7.8, height=430, tooltip="<b>{place}</b><br/>{status}",
             on_select=on_map_click, selection_mode="single-object", key="map")
    st.html(chips([(LEVELS[k]["hex"], LEVELS[k]["word"]) for k in LEVELS] + [(GREY["hex"], "No forecast")]))
    table = pd.DataFrame([{"date": day, "station": c, "beach": place(c),
                           "risk_percent": round(s["forecast"][today_index(s, day)]["risk"] * 100, 1),
                           "level": LEVELS[s["forecast"][today_index(s, day)]["level"]]["word"],
                           "last_test_date": s["last_test"]["date"], "last_test_value": s["last_test"]["value"],
                           "lat": s["lat"], "lon": s["lon"]} for c, s in fc_by_code.items()])
    csv_button(table.sort_values("risk_percent", ascending=False), f"forecast_{day}", "Download this day's forecast (CSV)")


def status_box(code, s, i, today):
    """The beach's card in three labeled parts: the model's estimate for the day, the latest
    measured result (and how old it is), and the beach's after-rain pattern."""
    d = s["forecast"][i]
    lv = LEVELS[d["level"]]
    last = s["last_test"]
    result = "over the limit" if last["unsafe"] else "under the limit"
    # The beach's group, by the same name as on the After-rain patterns page, with its numbers.
    if code in DIAG_BY:
        g = group_of(DIAG_BY[code])
        pattern = f"<b>{GROUPS[g]['word']}</b>: {group_rates(DIAG_BY[code])}."
    else:
        pattern = "Not enough tests after rain to tell how this beach recovers."
    # How much the estimate can lean on: the latest test's age and the beach's whole record.
    age = (pd.Timestamp(today) - pd.Timestamp(last["date"])).days
    stale = (f'<div class="old"><b>Latest test is {age} days old</b>, so this estimate leans on rain, tides and '
             "the beach's track record.</div>") if age > 30 else ""
    record = (f'<div class="old">Track record: {STATIONS.at[code, "samples"]:,} tests since '
              f'{STATIONS.at[code, "first"][:4]}.</div>' if code in STATIONS.index else "")
    st.html(f"""
    <div class="box">
      <div class="where">{html.escape(place(code))}{code_tag(code)}</div>
      <div class="part">
        <h5>Model estimate for {nice_date(d['date'])}</h5>
        <div class="level"><span class="dot" style="--c:{lv['hex']}; width:14px; height:14px"></span>{lv['word']}</div>
        {meter(d['risk'], lv['hex'])}
        <div><b>{pct(d['risk'])}</b> chance a water test would be over the safe limit</div>
      </div>
      <div class="part">
        <h5>Latest measured result</h5>
        <div>{pd.Timestamp(last['date']):%b %d, %Y} ({ago(last['date'], today)}): {last['value']:,.0f} bacteria per
          100 mL, {result} of {limit_of(code):.0f}.</div>
        <div class="old">A test from an earlier day doesn't measure today's water. Lab results take a day or more.</div>
        {stale}
        {record}
      </div>
      <div class="part">
        <h5>After-rain pattern for this beach</h5>
        <div class="small">{pattern}</div>
      </div>
    </div>""")


def drivers_box(d):
    """What's driving this risk: the model's risk on a typical day at this beach, then the
    3 inputs that move today's risk most (made by pipeline/08_forecast.py)."""
    if "drivers" not in d:  # a forecast file from before this was added
        return
    items = "".join(
        f'<li><span class="p">{x["points"]:+.0f}<small>points</small></span><span>{html.escape(x["input"].capitalize())}: '
        f'{html.escape(x["now"])}<br><span class="t">Typical here: {html.escape(x["typical"])}</span></span></li>'
        for x in d["drivers"][:3])
    with panel():
        title("What's driving this risk")
        st.html(f'<div>On a typical day at this beach the model says <b>{pct(d["typical_risk"])}</b>. '
                f'For {nice_date(d["date"])} it says <b>{pct(d["risk"])}</b>. The inputs that move it most:</div>'
                f'<ul class="drivers">{items}</ul>')
        st.caption("Each line: the model's risk minus its risk with that input set to this beach's typical value "
                   "(its median over 2010 to 2022 tests), everything else unchanged. Inputs overlap, so the lines "
                   "don't add up exactly. " + POINTS_NOTE)


def day_name(date, first, replay):
    """'Today' for the first day of a live forecast, else e.g. 'Mon 26'."""
    d = pd.Timestamp(date)
    return "Today" if date == first and not replay else f"{d:%a} {d.day}"


def day_boxes(s, first_i, shown_date, replay):
    """Small cards for today and the next days (the day picked above has a navy outline).
    In replay mode, also what tests found."""
    out = []
    first = s["forecast"][first_i]["date"]
    for d in s["forecast"][first_i:]:
        lv = LEVELS[d["level"]]
        actual = ""
        if "actual" in d:  # only in replay files, on days the beach was sampled
            a = d["actual"]
            actual = f'<div class="act">Tested: <b>{"unsafe" if a["unsafe"] else "OK"}</b> ({a["value"]:,.0f})</div>'
        picked = " picked" if d["date"] == shown_date else ""
        out.append(f'<div class="day{picked}"><b>{day_name(d["date"], first, replay)}</b><div class="lv">{lv["word"]}</div>'
                   f'{meter(d["risk"], lv["hex"])}<div class="pct">{pct(d["risk"])}</div>{actual}</div>')
    st.html(f'<div class="days">{"".join(out)}</div>')


def grey_box(code):
    """Card for a site without a forecast (past study site, inactive beach, or inland site)."""
    s = STATIONS.loc[code]
    last = s["last_test"]
    st.html(f"""
    <div class="box">
      <div class="where">{html.escape(s['place'])}{code_tag(code)}</div>
      <div class="level"><span class="dot" style="--c:{GREY['hex']}; width:16px; height:16px"></span>No forecast</div>
      <div>{html.escape(no_forecast_reason(code))}</div>
      <div class="small">{s['samples']:,} tests from {s['first'][:4]} to {s['last'][:4]};
        unsafe on {pct(s['unsafe_rate'])} of them. Last test {pd.Timestamp(last['date']):%b %d, %Y}.</div>
    </div>""")


def history_chart(code):
    """Past test results on a log scale, with a dashed line at the safe limit and unsafe tests in red."""
    hist = load_json("history.json")
    rows = hist["stations"].get(code)
    if not rows:
        st.caption("No test results in the past two years.")
        return
    df = pd.DataFrame(rows, columns=hist["columns"])
    df["date"] = pd.to_datetime(df["date"])
    df["shown"] = df["value"].clip(lower=1)  # a log scale can't show 0, so draw tiny values at 1
    df["result"] = df["unsafe"].map({1: "Over the limit", 0: "Under the limit"})
    limit = limit_of(code)
    st.html(chips([(BLUE, "Under the limit"), (LEVELS["high"]["hex"], "Over the limit")]))
    result_scale = alt.Scale(domain=["Under the limit", "Over the limit"], range=[BLUE, LEVELS["high"]["hex"]])
    points = alt.Chart(df).mark_point(filled=True, size=55, opacity=0.85).encode(
        x=alt.X("date:T", title=None),
        y=alt.Y("shown:Q", scale=alt.Scale(type="log"), title="Bacteria per 100 mL (log scale)",
                axis=alt.Axis(values=[1, 10, 100, 1000, 10000, 100000], format=",")),  # each line = 10x more
        color=alt.Color("result:N", scale=result_scale, legend=None),
        shape=alt.Shape("result:N", scale=alt.Scale(domain=result_scale.domain, range=["circle", "diamond"]), legend=None),
        tooltip=[alt.Tooltip("date:T", title="Date", format="%b %d, %Y"),
                 alt.Tooltip("value:Q", title="Bacteria per 100 mL", format=",.0f"),
                 alt.Tooltip("result:N", title="Result")])
    line = pd.DataFrame({"y": [limit], "label": [f"Safe limit: {limit:.0f}"]})
    rule = alt.Chart(line).mark_rule(strokeDash=[4, 4], color=NAVY).encode(y="y:Q")
    rule_label = alt.Chart(line).mark_text(align="left", dx=4, dy=-7, color=NAVY, fontSize=11).encode(
        y="y:Q", x=alt.value(0), text="label:N")
    show_chart((points + rule + rule_label).properties(height=260), width="stretch")
    first, last = df["date"].min(), df["date"].max()
    st.caption(f"{len(df)} tests from {first:%b %Y} to {last:%b %Y}, {df['unsafe'].sum()} over the limit. "
               "Hover or tap a dot for details.")
    csv_button(df[["date", "value", "unsafe"]].assign(date=df["date"].dt.date), f"past-tests_{code}")


def forecast_page():
    top = st.container()  # the finding sits above the replay switch, but is written after it
    replay = st.toggle("Replay the January 2023 storm",
                       help="Shows what this app would have said on Jan 9, 2023, a few days before huge storms, "
                            "and what the water tests actually found.")
    error = None if replay else refresh_forecast()
    fc = load_json("replay_2023-01-09.json" if replay else "forecast.json")
    fc_by_code = {s["station"]: s for s in fc["stations"]}
    # The day to show: today in LA. In a replay, or if the file is too old to have today, its first day.
    today = fc["issued"] if replay else la_today()
    any_beach = fc["stations"][0]
    first_i = today_index(any_beach, today)
    first = any_beach["forecast"][first_i]["date"]  # today, or the file's first day if it is too old to have today

    # The day picker: today and the next days. The map, the counts, the beach card and
    # "What's driving this risk" below all show the picked day.
    dates = [d["date"] for d in any_beach["forecast"][first_i:]]
    day = st.segmented_control("Day", dates, default=first, key=f"fc_day_{'replay' if replay else 'live'}",
                               format_func=lambda d: day_name(d, first, replay)) or first
    levels = pd.Series([s["forecast"][today_index(s, day)]["level"] for s in fc["stations"]]).value_counts()
    warned = levels.get("caution", 0) + levels.get("high", 0)
    if replay:
        made_on = "that day" if day == first else f"on {nice_date(first)}"
        when = (f"{nice_date(day, year=True)}: the forecast as it would have looked {made_on}, "
                "using the rain that actually fell (a best case). Day cards show real test results where a beach was sampled.")
    elif fc["issued"] >= today:
        made = pd.Timestamp(fc.get("made_at", fc["issued"]))
        ahead = (pd.Timestamp(day) - pd.Timestamp(today)).days
        when = (f"Forecast for {nice_date(day)}, made {made:%b} {made.day} at {made.hour % 12 or 12}:{made:%M %p} "
                "(LA time): each beach's chance that a water test would be over the safe limit."
                + ("" if ahead == 0 else f" {ahead} day{'s' if ahead > 1 else ''} ahead, so less sure than today's."))
    else:
        when = (f"Last updated {nice_date(fc['issued'])}. Today's forecast couldn't be made, so this is the last "
                f"successful run, shown for {nice_date(day)}.")
    when_word = ("" if replay else " today") if day == first else f" on {nice_date(day)}"
    with top:
        finding("Forecast" + (" replay" if replay else ""),
                f"<b>{warned} of {len(fc['stations'])} beaches</b> {'were' if replay else 'are'} at caution or high "
                f"risk{when_word}.", when)
        if error:
            st.caption(f"Couldn't update the forecast: {error}")
    st.html(chips([(LEVELS[k]["hex"], f"{levels.get(k, 0)} {v['word']}") for k, v in LEVELS.items()]))

    # The beach picker lists forecast beaches first (A to Z), then sites without a forecast.
    beach_codes = sorted(fc_by_code, key=place)
    other_codes = sorted((c for c in STATIONS.index[STATIONS["on_map"]]
                          if c not in fc_by_code and c not in HIDE_FROM_MAP), key=place)
    options = beach_codes + other_codes
    riskiest = max(fc_by_code, key=lambda c: fc_by_code[c]["forecast"][today_index(fc_by_code[c], day)]["risk"])
    chosen_beach("beach", options, riskiest)

    left, right = st.columns([5, 6], gap="large")
    with left:  # on a phone this column comes first, so the answer is at the top
        code = st.selectbox("Beach", options, key="beach",
                            format_func=lambda c: label(c) + ("" if c in fc_by_code else " (no forecast)"),
                            help="Type to search, or tap a dot on the map. Starts on the highest-risk beach. "
                                 "The link in your browser's address bar opens this beach.")
        remember_beach(code, "beach")
        if code in fc_by_code:
            s = fc_by_code[code]
            i = today_index(s, day)
            status_box(code, s, i, today)
            day_boxes(s, today_index(s, first), day, replay)
        else:
            grey_box(code)
    with right:
        beach_map(fc_by_code, code, day)

    if code in fc_by_code:
        drivers_box(s["forecast"][i])

    with panel():
        title(f"Past test results: {place(code)}")
        if replay:
            st.caption("History shows recent years, not the replay date.")
        history_chart(code)

    caution = f"{load_json('final_metrics.json')['threshold'] * 100:.0f}%"  # the warning level, set in 06
    st.caption(f"Low risk: under {caution}. Caution: {caution} to 50% (the level where this app would post a "
               "warning). High risk: 50% or more, unsafe more likely than not. Safe limit: 104 enterococcus "
               "bacteria per 100 mL of ocean water (California standard). Rain from the Open-Meteo forecast.")


# ---------------------------------------------------------------------------
# Page 6: Methods
# ---------------------------------------------------------------------------

def gain(v):
    """[0.05, 0.019, 0.086] -> '+5.0 points (95% range +1.9 to +8.6)'."""
    return f"{v[0] * 100:+.1f} points (95% range {v[1] * 100:+.1f} to {v[2] * 100:+.1f})"


def methods_page():
    # Model numbers come from app_data/final_metrics.json, written by pipeline/10_final_metrics.py
    # together with review/final_metrics.md, so they always match. Each is [value, low, high].
    fm = load_json("final_metrics.json")
    test, live, day3 = fm["scenarios"]["test"], fm["scenarios"]["live"], fm["scenarios"]["day3"]
    start, end = (pd.Timestamp(d) for d in fm["test_period"])
    finding("Methods",
            f"The forecast catches <b>{pct(test['sign_plus_rain.model_caught'][0])}</b> of unsafe days.",
            f"Warning signs plus the rain advisory catch {pct(test['sign_plus_rain.caught'][0])}, with the same number "
            f"of warnings. Difference: {gain(test['sign_plus_rain.gain'])}, on {fm['tested_beach_days']:,} tested "
            f"beach-days at {fm['beaches']} {COVERAGE} from {start:%B %Y} to {end:%B %Y} that the model never saw "
            "while learning.",
            notes=[POINTS_NOTE])

    names = {"sign": "Signs", "sign_plus_rain": "Signs + rain advisory",
             "sign3": "Signs (3 tests)", "sign3_plus_rain": "Signs (3 tests) + rain"}
    rows = []
    for key, name in names.items():
        rows.append({"rule": name, "who": "Today's rule", "caught": test[f"{key}.caught"][0] * 100})
        rows.append({"rule": name, "who": "This app's forecast", "caught": test[f"{key}.model_caught"][0] * 100})
    bars_df = pd.DataFrame(rows)
    with panel():
        title("Unsafe days caught, with the same number of warnings")
        st.html(chips([(ORANGE, "Today's rule"), (BLUE, "This app's forecast")]))
        who_scale = alt.Scale(domain=["Today's rule", "This app's forecast"], range=[ORANGE, BLUE])
        base = alt.Chart(bars_df).encode(
            y=alt.Y("rule:N", sort=list(names.values()), title=None, axis=alt.Axis(labelLimit=200)),
            yOffset=alt.YOffset("who:N", sort=who_scale.domain))
        bars = base.mark_bar(cornerRadiusEnd=3, height=12).encode(
            x=alt.X("caught:Q", title="% of unsafe days caught", scale=alt.Scale(domain=[0, 100])),
            color=alt.Color("who:N", scale=who_scale, legend=None),
            tooltip=[alt.Tooltip("rule:N", title="Compared with"), alt.Tooltip("who:N", title=" "),
                     alt.Tooltip("caught:Q", title="% caught", format=".0f")])
        labels = base.mark_text(align="left", dx=4, color=NAVY, fontSize=11).encode(
            x="caught:Q", text=alt.Text("caught:Q", format=".0f"))
        show_chart((bars + labels).properties(height=280), width="stretch")
        csv_button(bars_df.round(1).rename(columns={"caught": "percent_of_unsafe_days_caught"}), "forecast-vs-rules")
        st.caption("\"3 tests\" = signs posted when any of the three state ocean tests fails (enterococcus, "
                   f"fecal coliform, total coliform), which is what counties post on. Against signs alone: "
                   f"{gain(test['sign.gain'])}.")

    # Calibration (pipeline step 14): when the forecast says 20%, are about 20% of those tests unsafe?
    cal = load_json("calibration.json")
    cb = pd.DataFrame(cal["bins"])
    for c in ("predicted", "actual", "actual_low", "actual_high"):
        cb[c] = cb[c] * 100
    cb["group"] = [f"{f * 100:.0f} to {t * 100:.0f}%" for f, t in zip(cb["from"], cb["to"])]
    middle = cb[(cb["from"] >= .05) & (cb["to"] <= .5)]
    gap = (middle["actual"] - middle["predicted"]).abs()  # how far off the risk is, in points
    with panel():
        title("Does the risk mean what it says? Predicted risk vs. how often tests were unsafe")
        st.markdown(f"Each dot is a group of test days from {cal['period'][0][:4]} to {cal['period'][1][:4]} with a "
                    "similar forecast risk. On the dashed line, the risk would be exactly right. Over all "
                    f"{cal['tests']:,} tests the forecast averaged **{pct(cal['mean_predicted'])}** and "
                    f"**{pct(cal['mean_actual'])}** were unsafe: the risk runs a little "
                    f"{'low' if cal['mean_actual'] > cal['mean_predicted'] else 'high'}, by about {gap.min():.0f} to "
                    f"{gap.max():.0f} points in the groups from 5% to 50%.")
        axis = alt.Scale(domain=[0, 100])
        diag = alt.Chart(pd.DataFrame({"x": [0, 100], "y": [0, 100]})).mark_line(
            strokeDash=[4, 4], color=NAVY, strokeWidth=1).encode(x="x:Q", y="y:Q")
        ranges = alt.Chart(cb).mark_rule(color=BLUE, strokeWidth=2).encode(
            x=alt.X("predicted:Q", title="Forecast risk (%)", scale=axis),
            y=alt.Y("actual_low:Q", title="Tests actually unsafe (%)", scale=axis), y2="actual_high:Q")
        dots = alt.Chart(cb).mark_point(filled=True, color=BLUE, opacity=1).encode(
            x="predicted:Q", y="actual:Q", size=alt.Size("tests:Q", legend=None, scale=alt.Scale(range=[40, 260])),
            tooltip=[alt.Tooltip("group:N", title="Forecast risk"), alt.Tooltip("tests:Q", title="Tests", format=","),
                     alt.Tooltip("predicted:Q", title="Average forecast %", format=".1f"),
                     alt.Tooltip("actual:Q", title="Actually unsafe %", format=".1f")])
        show_chart((diag + ranges + dots).properties(height=320), width="stretch")
        st.caption("Bigger dots have more tests. The short lines are 95% ranges for the share unsafe; groups with "
                   "few tests have wider ranges. Tests the model never saw while learning.")
        numbers(cb[["group", "tests", "predicted", "actual", "actual_low", "actual_high"]], "calibration",
                group="Forecast risk", tests="Tests", predicted=st.column_config.NumberColumn("Average forecast %", format="%.1f"),
                actual=st.column_config.NumberColumn("Actually unsafe %", format="%.1f"),
                actual_low=st.column_config.NumberColumn("95% range low", format="%.1f"),
                actual_high=st.column_config.NumberColumn("95% range high", format="%.1f"))

    d3 = day3["sign_plus_rain.gain"]
    d3_verdict = ("the whole range is above zero, so it does better than today's signs" if d3[1] > 0 else
                  "the range includes zero, so the 3-day forecast is about as good as today's signs, not better")
    caution = f"{fm['threshold'] * 100:.0f}%"
    with st.expander("The forecast", expanded=True):
        st.markdown(
            f"- **Where:** {COVERAGE}, with tests from the California State Water Boards. Comparisons with today's rule "
            "use LA County's 72-hour rain advisory for every beach, including Ventura County's.\n"
            "- **What it looks at:** the beach's track record (matters most), the last test result once it would be "
            "public (2 days after sampling), rain in the day, 3 days, week and month before, days since rain, tides "
            "(beaches fail about twice as often on big spring tides; Boehm & Weisberg, 2005), the season, and location.\n"
            f"- **How it learned:** gradient boosting (scikit-learn) on {fm['train_samples']:,} beach tests from 2010 to "
            f"2022, graded only on {start.year} to {end.year}.\n"
            f"- **Levels:** caution starts at {caution}, the level that gives the same number of warnings as signs plus "
            "the rain advisory; high is 50% or more.\n"
            f"- **Ranking:** a score of {test['auc'][0]:.2f} for telling unsafe days from safe ones (1.0 is perfect, 0.5 a coin flip).\n"
            f"- **Live forecast:** uses forecast rain, so estimated {pct(live['sign_plus_rain.model_caught'][0])} vs "
            f"{pct(live['sign_plus_rain.caught'][0])}: {gain(live['sign_plus_rain.gain'])}.\n"
            f"- **3 days ahead:** {pct(day3['sign_plus_rain.model_caught'][0])} vs {pct(day3['sign_plus_rain.caught'][0])} "
            f"for signs and the rain advisory on the day itself: {gain(d3)}; {d3_verdict}.\n"
            "- **Live rain:** the app remakes the forecast each day from the Open-Meteo rain forecast. Forecast rain runs "
            "lower than ERA5, the rain record the model learned from, so each weather cell's forecast is scaled to match "
            "ERA5 over past years.\n"
            "- **What's driving this risk:** the forecast page reruns the model with one group of inputs (rain, tides, "
            "the latest test, season, track record) set to the beach's typical value, and shows the 3 biggest changes.")

    # A separate check against the state's REAL posting records (pipeline step 13). It doesn't
    # replace the numbers above, which use signs rebuilt from the test results.
    pj = load_json("postings.json")
    ck = pj["check"]
    pc, (c0, c1) = ck["postings"], (pd.Timestamp(d) for d in ck["period"])
    pr = ck["postings_plus_rain"]
    with st.expander("Compared with what counties actually posted"):
        st.markdown(
            f"The comparison above rebuilds warning signs from the test results. As a separate check, we used the "
            f"state's records of what counties **actually posted** (Beach Watch), {c0:%B %Y} to {c1:%B} {c1.day}, "
            f"{c1.year} (where the records end): {ck['tested_beach_days']:,} tested beach-days at {ck['beaches']} "
            "beaches. Each row gives the forecast the same number of warnings as the real advisories.")
        rows_table([("Compared with", False), ("Warns on", True), ("Real advisories caught", True),
                    ("Forecast caught", True), ("Difference", False)],
                   [[name, pct(r["share"][0]), pct(r["caught"][0]), pct(r["model_caught"][0]), gain(r["gain"])]
                    for name, r in [("Signs counties actually posted (and closures)", pc),
                                    ("Those signs plus the counties' rain advisories", pr)]])
        st.markdown(
            "- An advisory counts if it was up at the start of the test day (issued on an earlier day).\n"
            "- Rain advisories are issued for all of a health department's beaches at once (LA County, the City of "
            "Long Beach, Ventura County), so each one counts for every beach that department covers.\n"
            "- The second row is the fairer match for the main comparison above (signs plus the 72-hour rain "
            f"advisory, {gain(test['sign_plus_rain.gain'])}). The main numbers stay the headline.")

    defs = DIAG["definitions"]
    calib = load_json("evaluator_data.json")["calibration"]
    with st.expander("After-rain patterns and beach profiles"):
        st.markdown(
            f"- **Days since rain:** {defs['day']}. Rain is a daily total, so a day-3 test came roughly 48 to 96 hours "
            "after the rain.\n"
            f"- **Normal:** {defs['normal']}.\n"
            f"- **Dry weather:** {defs['dry']}. **{GROUPS['chronic']['word']}:** {defs['chronic']}.\n"
            f"- **Trigger:** {defs['trigger']}.\n"
            f"- **Clearly above:** {defs['clearly_above']}. Groups need {defs['min_tests']}+ tests.\n"
            "- Fix categories describe what a pattern points to, not a proven source.")
    with st.expander("Cleanup projects"):
        st.markdown(
            "- **Expected risk from weather only:** a model that knows the beach, rain (including the test day), days "
            "since rain, tides and season, but not recent test results or the date. The forecast model can't be used: its "
            "recent-results inputs would follow a real improvement and hide it. Each year is scored by a model trained on "
            "the other years.\n"
            "- **Change** = (beach after minus before) minus (beaches 5+ km away, after minus before), using "
            "observed minus expected.\n"
            f"- **95% range:** resampling whole months, widened {calib['widen']}x so that only "
            f"{pct(calib['check_share_excluding_zero'])} of {calib['runs']} random beaches and dates look like a change.\n"
            "- Results say **cleaner after** or **no clear change after**, never **because of**: one beach and one date "
            "can't rule out something else changing at the same time.")

    sp = load_json("spills.json")
    sm = sp["summary"]
    with st.expander("Checked and ruled out: sewage spills"):
        st.markdown(
            f"Since 2023, {sm['spills_with_beach_samples']} large sewage spills (10,000+ gallons reaching rivers, drains or "
            f"the ocean) had {sm['beach_samples_after_spills']} beach tests within 10 km in the following week. "
            f"{pct(sm['observed_unsafe_rate'])} of those tests were unsafe; the model, which knows nothing about spills, "
            f"expected {pct(sm['predicted_unsafe_rate'])}. Several spills happened during big storms, when rain alone made "
            "the water unsafe, so the model doesn't use spill data.")

    with st.expander("Limits"):
        st.markdown(
            "- Results count days a beach was actually tested, and beaches are often retested after a bad result.\n"
            "- The main comparison rebuilds warning signs from the test data; the check against what counties "
            f"actually posted (above) only runs to {pd.Timestamp(pj['records_through']):%B %Y}, where the state's records end.\n"
            "- Rain forecasts get less accurate further ahead, so later forecast days are less certain.\n"
            "- The daily forecast uses fresh rain and tides, but test results only up to the last data download "
            f"({pd.Timestamp(DIAG['period'][-10:]):%B %d, %Y}); the forecast card says how old each beach's latest "
            "result is.\n"
            "- This is a student project, not an official health advisory. Always follow posted signs.")
    with st.expander("Data and AI use"):
        st.markdown(
            f"Data: California State Water Boards (Safe to Swim) tests at {COVERAGE}, Open-Meteo (ERA5 rain), NOAA (tides), CIWQS (sewage "
            "spills), CAL FIRE (burn areas), State Water Board Beach Watch (real posting records). Parts of the code and analysis were written with AI help (Claude Code); "
            "every AI-assisted step is listed in the project's AI use log.")


# ---------------------------------------------------------------------------
# Home
# ---------------------------------------------------------------------------

HERO_PHOTO = ROOT / "assets" / "hero.jpg"


def beach_questions(code):
    """Step 2 of the Home search: one line about the beach, then a link for each question this beach
    has data for. Each link opens that page with ?beach=CODE, and the page starts on this beach."""
    fc = {s["station"]: s for s in load_json("forecast.json")["stations"]}
    facts = []
    if code in fc:
        d = fc[code]["forecast"][today_index(fc[code], la_today())]
        lv = LEVELS[d["level"]]
        facts.append(f'<span class="dot" style="--c:{lv["hex"]}"></span><b>{lv["word"]} today</b>: '
                     f'{pct(d["risk"])} chance a test is over the limit')
    else:
        reason = no_forecast_reason(code).replace(", so no forecast", "")  # don't say "no forecast" twice
        facts.append(f"<b>No forecast</b>: {html.escape(reason[0].lower() + reason[1:])}")
    if code in DIAG_BY:
        facts.append(GROUPS[group_of(DIAG_BY[code])]["word"])
    st.html(f'<div class="beach-line"><b>{html.escape(place(code))}</b>{code_tag(code)}<br>{" · ".join(facts)}</div>')

    ex = eval_table(file_version("evaluator_data.json"))
    counts = ex["StationCode"].value_counts()
    asks = [("Is it safe this week?" if code in fc else "What do its past tests show?", "forecast", True),
            ("Why is it unsafe, and how fast does it recover after rain?", "diagnosis", code in DIAG_BY),
            ("How does it compare to nearby beaches?", "compare", code in fc),
            ("Did a cleanup project help here?", "evaluator", counts.get(code, 0) >= 100),
            ("Was it affected by the 2025 fires?", "fires", km_to_burn(code) <= 1)]
    with st.container(key="ask"):
        for text, page, has_data in asks:
            if has_data:
                st.page_link(PAGES[page], label=text, icon=PAGES[page].icon, query_params={"beach": code},
                             width="stretch")


def home_page():
    fm = load_json("final_metrics.json")
    test = fm["scenarios"]["test"]
    test_years = f"{fm['test_period'][0][:4]}–{fm['test_period'][1][:4]}"  # e.g. 2023–2026
    n_slow = DIAG["region"]["recovery_groups"]["slow"]["beaches"]
    n_chronic = sum(s["chronic"] for s in DIAG["stations"])
    first_year, last_year = DIAG["period"][:4], DIAG["period"][-10:-6]

    left, right = st.columns([6, 5], gap="large", vertical_alignment="center")
    with left:
        st.html(f'<div class="home-hero"><div class="title-row">{logo_img("after-the-rain-logo.svg", 72)}'
                f'<h1>{APP_NAME}</h1></div><p>See which {COVERAGE} stay unsafe after rain, why, and whether '
                f'cleanup projects worked, from {int(last_year) - int(first_year)} years of state water tests.</p></div>')
        # Two actions: look up one beach (step 1: search; step 2: what do you want to know?),
        # or jump down to the findings.
        codes = sorted((c for c in STATIONS.index[STATIONS["on_map"]] if c not in HIDE_FROM_MAP), key=place)
        pick = st.selectbox("Check a beach", codes, index=None, format_func=label, key="home_beach",
                            placeholder="Type a beach name")
        if pick:
            beach_questions(pick)
        st.html('<a class="explore" href="#what-we-found" target="_self">See what we found <span>↓</span></a>')
    with right:
        if HERO_PHOTO.exists():
            st.image(str(HERO_PHOTO), width="stretch")
        else:  # the space is kept so the layout doesn't change once a photo is added
            st.html('<div class="photo">Photo goes here: add assets/hero.jpg</div>')

    st.html('<div class="section-h">The problem</div><div class="problem">'
            "<p>Rain washes bacteria from streets and storm drains into the ocean, and swimmers can get sick.</p>"
            "<p>LA County gives every beach the same rule: stay out for 72 hours after rain.</p>"
            "<p>But beaches recover at different speeds, and lab results take a day or more to come back.</p></div>")

    st.html('<div class="section-h" id="what-we-found">What we found</div>'
            f'<div class="big3">'
            f'<div class="card"><div class="v">{n_slow}</div><div class="l">beaches stay unsafe longer than '
            "LA County's 72-hour rain advisory</div></div>"
            f'<div class="card"><div class="v">{n_chronic}</div><div class="l">beaches are often unsafe even in '
            f"dry weather, so rain isn't their only problem</div></div>"
            f'<div class="card"><div class="v">{pct(test["sign_plus_rain.model_caught"][0])}</div><div class="l">'
            f"In testing on {test_years} data, the forecast caught {pct(test['sign_plus_rain.model_caught'][0])} of "
            f"unsafe tested days, vs. {pct(test['sign_plus_rain.caught'][0])} for warning signs plus the rain "
            "advisory, with the same number of warnings.</div></div></div>")

    # One card per question; the whole card is a link to the page that answers it.
    st.html('<div class="section-h">Start with a question</div>')
    questions = [
        ("Is 72 hours the right wait for every beach?", "advisory"),
        ("Why is this beach unsafe, and what would fix it?", "diagnosis"),
        ("How does a beach compare with its neighbors?", "compare"),
        ("Was a beach cleaner after a cleanup project?", "evaluator"),
        ("Did the 2025 fires make beaches dirtier?", "fires"),
        ("Which beaches are risky today?", "forecast"),
        ("How do you know?", "methods"),
    ]
    for r, row in enumerate((questions[:4], questions[4:])):  # 4 cards, then 3 (one per row on a phone)
        for c, (col, (q, page)) in enumerate(zip(st.columns(4), row)):
            with col.container(key=f"qcard-{r * 4 + c}"):
                st.page_link(PAGES[page], label=q, icon=PAGES[page].icon, width="stretch")

    st.html('<div class="section-h">Who it\'s for</div><div class="who">'
            "<p><b>City and county water staff</b>: see which beaches need a longer advisory, and what kind of fix "
            "each one points to.</p>"
            "<p><b>Lawmakers and their staff</b>: check whether a cleanup project was followed by cleaner water.</p>"
            "<p><b>Beachgoers</b>: see each beach's risk today and for the next 3 days.</p></div>")

    st.html('<div class="trust">'
            f'<div><h4>Data</h4><p>California State Water Boards tests at {COVERAGE} ({first_year} to {last_year}), '
            "Open-Meteo rain, NOAA tides, CIWQS sewage spills, CAL FIRE burn areas.</p></div>"
            "<div><h4>What unsafe means</h4><p>A water test over 104 enterococcus bacteria per 100 mL, California's "
            "standard for ocean beaches. Every beach is compared with LA County's 72-hour rain advisory.</p></div>"
            '<div><h4>How it works</h4><p><a href="methods" target="_self">Read the methods</a>: how each number '
            "was made, and its limits.</p></div>"
            "<div><h4>Built by Dylan Lin</h4><p>For the 2026 Congressional App Challenge. AI help is listed "
            "in the methods.</p></div></div>")


# ---------------------------------------------------------------------------
# Page frame
# ---------------------------------------------------------------------------

# The left menu (like a weather map's menu): wordmark, tagline, then one link per page.
# On a phone the menu folds away, so a small wordmark shows at the top of the page instead.
st.sidebar.html(f'<div class="side-brand">{logo_img("after-the-rain-logo-on-dark.svg", 40)}{APP_NAME}</div>'
                f'<div class="side-tag">{TAGLINE}</div>')
st.html(f'<div class="brand-main">{logo_img("after-the-rain-logo.svg", 28)}{APP_NAME}</div>')
check_model_ids()
PAGES = {  # also used by the home page's question cards and actions
    "home": st.Page(home_page, title="Home", icon=":material/home:", url_path="home", default=True),
    "advisory": st.Page(advisory_page, title="After-rain patterns", icon=":material/rainy:", url_path="after-rain"),
    "diagnosis": st.Page(diagnosis_page, title="Beach profile", icon=":material/beach_access:", url_path="beach-profile"),
    "compare": st.Page(compare_page, title="Compare beaches", icon=":material/view_column:", url_path="compare"),
    "evaluator": st.Page(evaluator_page, title="Cleanup projects", icon=":material/compare_arrows:",
                         url_path="cleanup-projects"),
    "fires": st.Page(fires_page, title="Fire impacts", icon=":material/local_fire_department:", url_path="fire-impacts"),
    "forecast": st.Page(forecast_page, title="Forecast", icon=":material/partly_cloudy_day:", url_path="forecast"),
    "methods": st.Page(methods_page, title="Methods", icon=":material/menu_book:", url_path="methods"),
}
st.navigation(list(PAGES.values()), position="sidebar").run()

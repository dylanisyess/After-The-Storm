"""After the Rain: LA beach water, what makes it unsafe and what fixes it
(Congressional App Challenge 2026).

Run inside WSL:  source ~/waterenv/bin/activate && streamlit run app.py

The app only READS the files in app_data/ (made by the pipeline; see
pipeline/README.md). The intervention evaluator's math lives in
pipeline/evaluator.py, so the app and pipeline/12_intervention.py give the same answer.

Every page opens with a short headline (one idea, one number) and the details on
the line below, then annotated charts, then quiet number tiles. Pages, in order (left menu):
  Home              what the app is for, three key findings, a question for each page
  Advisory check    does LA County's 72-hour rain advisory fit each beach?
  Beach diagnosis   one beach: dry vs. after rain, rain trigger, recovery, what fix it points to
  Did it work?      a beach before vs. after a project, beyond what the weather explains
  After the fires   did beaches below the 2025 Palisades burn get dirtier?
  Forecast          today + 3 days for every beach (with the January 2023 storm replay)
  Methods           how the model and the analyses work, and their limits
"""

import html
import json
import sys
from pathlib import Path

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
TAGLINE = "LA beach water: what makes it unsafe, and what fixes it"
st.set_page_config(page_title=APP_NAME, page_icon=":material/water:", layout="wide")


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
MUTED = "#5A6880"    # secondary text, "other beaches" lines
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
    --ink: #1A2440; --ink-2: #3B4A63; --muted: #5A6880; --harbor: #22314F;
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
  .mark { flex: none; width: 26px; height: 26px; border-radius: 50%;
          background: linear-gradient(var(--sun-strong) 0 54%, #6FA8F0 54% 100%); }
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
  [data-testid="stCaptionContainer"] { color: var(--muted); }

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

  /* legends: a colored dot next to its words */
  .chips { display: flex; flex-wrap: wrap; gap: var(--s2) var(--s4); margin: var(--s1) 0 var(--s2); }
  .chip  { font-size: .875rem; color: var(--ink-2); white-space: nowrap; }
  .dot   { display: inline-block; width: 10px; height: 10px; border-radius: 50%; background: var(--c);
           margin-right: var(--s2); vertical-align: 0; }

  /* home page */
  .home-hero h1 { font-family: var(--display); font-weight: 800; font-size: clamp(2.6rem, 1.8rem + 3vw, 4rem);
                  line-height: 1; letter-spacing: -.02em; color: var(--ink); margin: 0 0 var(--s4); padding: 0; }
  .home-hero p  { font-size: 1.2rem; line-height: 1.5; color: var(--ink-2); max-width: 34ch; margin: 0; }
  .photo        { aspect-ratio: 4 / 3; border-radius: 16px; background: var(--tide); display: flex; align-items: center;
                  justify-content: center; color: var(--muted); font-size: .9rem; text-align: center; padding: var(--s4); }
  .section-h    { font-family: var(--display); font-weight: 700; font-size: 1.55rem; color: var(--ink);
                  margin: var(--s7) 0 var(--s4); }
  .problem p    { font-size: 1.15rem; line-height: 1.45; color: var(--ink); margin: 0 0 var(--s2); max-width: 60ch; }
  .big3         { display: grid; grid-template-columns: repeat(3, 1fr); gap: var(--s3); }
  @container (max-width: 760px) { .big3 { grid-template-columns: 1fr; } }
  .big3 .card .v { font-size: 3rem; }
  .q            { font-family: var(--display); font-weight: 600; font-size: 1.15rem; line-height: 1.3; color: var(--ink);
                  min-height: 2.6em; }  /* two lines, so cards in a row line up */
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
    .brand-main .mark { width: 18px; height: 18px; }
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
    axis = {"labelColor": "#5A6880", "titleColor": "#5A6880", "labelFontSize": 12, "titleFontSize": 12,
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

@st.cache_data
def load_json(name):
    """Read one file from app_data/ (cached, so each file is read only once)."""
    return json.loads((DATA / name).read_text(encoding="utf-8"))


def clean_name(code, name):
    """'10000-Solimar Beach, Ventura' -> 'Solimar Beach, Ventura'."""
    if name.startswith(code + "-"):
        name = name[len(code) + 1:]
    return name.strip()


@st.cache_data
def station_table():
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


STATIONS = station_table()


def check_model_ids():
    """Warn if some app_data files came from a different model run than others.
    Every pipeline step stamps "model_id"; pipeline/rebuild.sh remakes them all together."""
    ids = {f.name: json.loads(f.read_text(encoding="utf-8")).get("model_id") for f in DATA.glob("*.*json")}
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


def numbers(df, **config):
    """Every chart has a table twin, for screen readers and anyone who wants the exact values."""
    with st.expander("Show the numbers"):
        st.dataframe(df, hide_index=True, width="stretch", column_config=config)


def base_map(layers, lat=33.98, lon=-118.78, zoom=7.9, tooltip=None, height=400, **kw):
    """A plain grey map (so the colored dots stand out), zoomed to the coast from Ventura to Long Beach."""
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
# Page 1: Advisory check
# ---------------------------------------------------------------------------

def advisory_page():
    reg = DIAG["region"]
    rg = reg["recovery_groups"]
    n = {g: rg[g]["beaches"] for g in ("slow", "typical", "fast")}
    slow_big = pooled(rg["slow"]["curves"]["1 in or more"], 4, 6)
    fast_big = pooled(rg["fast"]["curves"]["1 in or more"], 2, 3)
    tide = reg["after_rain_tides"]

    slow_chronic = sum(DIAG_BY[c]["chronic"] for c in rg["slow"]["stations"])
    also_dry = (f"{slow_chronic} of them {'is' if slow_chronic == 1 else 'are'} also often unsafe in dry weather."
                if slow_chronic else "")
    fast_23 = pooled(rg["fast"]["curves"]["all storms"], 2, 3)
    fast_normal = rg["fast"]["normal_wet_season_rate"]

    finding("Advisory check",
            f"<b>{n['slow']} beaches</b> stay unsafe longer than LA County's 72-hour rain advisory.",
            f"{also_dry} After storms of 1 inch or more, they were still unsafe on <b>{pct(slow_big)}</b> of tests "
            f"4 to 6 days later (their normal level: {pct(rg['slow']['normal_wet_season_rate'])}). Another "
            f"{n['fast']} beaches recover faster than most: {pct(fast_23)} of tests 2 to 3 days after rain were "
            f"unsafe, vs. {pct(fast_normal)} normally ({pct(fast_big)} after storms of 1 inch or more).",
            notes=[NORMAL_NOTE, PRECAUTION])

    cards([
        (n["slow"], GROUPS["slow"]["word"], "clearly above their normal level 4 to 6 days after rain"
         + (f"; {slow_chronic} also often unsafe in dry weather" if slow_chronic else ""), GROUPS["slow"]["hex"]),
        (n["typical"], GROUPS["typical"]["word"], "still unsafe on days 2 to 3, normal by days 4 to 6",
         GROUPS["typical"]["hex"]),
        (n["fast"], GROUPS["fast"]["word"], f"days 2 to 3 after rain: {pct(fast_23)} unsafe vs. "
         f"{pct(fast_normal)} normally", GROUPS["fast"]["hex"]),
        (f"{pct(tide['big_tide']['mean'])} vs {pct(tide['other']['mean'])}", "unsafe on big spring-tide days vs. other days",
         "3 to 7 days after rain, all beaches"),
    ])

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
                   "so day 3 is roughly 48 to 96 hours later). Dashed lines: each group's normal level, from wet-season "
                   "days with no rain for 8+ days. Beaches often unsafe in dry weather are left out of the lower two lines.")
        numbers(df[["group", "day", "unsafe", "tests", "normal"]],
                group="Beaches", day="Days since rain", unsafe=st.column_config.NumberColumn("% unsafe", format="%.1f"),
                tests="Tests", normal=st.column_config.NumberColumn("Normal %", format="%.1f"))

    # Map: every diagnosed beach colored by its group.
    rows = []
    for s in DIAG["stations"]:
        g = group_of(s)
        d46 = s["groups"]["days 4-6"]["unsafe_rate"]
        rows.append({"station": s["station"], "lat": s["lat"], "lon": s["lon"], "place": place(s["station"]),
                     "group": GROUPS[g]["word"], "color": GROUPS[g]["rgb"], "size": 8 if g in ("slow", "chronic") else 6,
                     "detail": f"Days 4-6 after rain: {pct(d46)} unsafe; normal {pct(s['normal_wet_season_rate'])}",
                     "order": ["other", "fast", "typical", "chronic", "slow"].index(g)})
    m = pd.DataFrame(rows).sort_values("order")  # slow beaches drawn last, on top
    with panel():
        title("Where they are")
        base_map([pdk.Layer("ScatterplotLayer", data=m, get_position=["lon", "lat"], get_fill_color="color",
                            get_radius="size", radius_units=PIXELS, stroked=True, get_line_color=[255, 255, 255],
                            line_width_min_pixels=1, pickable=True)],
                 tooltip="<b>{place}</b><br/>{group}<br/>{detail}")
        st.html(chips([(GROUPS[g]["hex"], GROUPS[g]["word"]) for g in GROUPS]))

    # The slow beaches, one row each.
    slow = [s for s in DIAG["stations"] if group_of(s) == "slow"]
    slow.sort(key=lambda s: -s["groups"]["days 4-6"]["unsafe_rate"])
    st.markdown(f"#### The {len(slow)} beaches that stay unsafe longer than 72 hours")
    st.markdown("All sit next to a big storm drain, creek or river mouth" + (f" ({also_dry.rstrip('.')})" if also_dry else "")
                + ". After rain they fail far more often on big spring-tide days, which fits a drain or lagoon that "
                "keeps emptying as the tide goes out.")
    t = pd.DataFrame([{"Beach": label(s["station"]),
                       "Often unsafe in dry weather": "Yes" if s["chronic"] else "No",
                       "Days 4-6 after rain": s["groups"]["days 4-6"]["unsafe_rate"] * 100,
                       "Normal": s["normal_wet_season_rate"] * 100,
                       "Days 3-7, big tide": (s["after_rain_tides"]["big_tide_rate"] or 0) * 100,
                       "Days 3-7, other tides": (s["after_rain_tides"]["other_rate"] or 0) * 100} for s in slow])
    fmt = st.column_config.NumberColumn(format="%.0f%%")
    st.dataframe(t, hide_index=True, width="stretch",
                 column_config={c: fmt for c in t.columns if c not in ("Beach", "Often unsafe in dry weather")})

    counts = DIAG["recovery_counts"]
    with st.expander("How beaches were sorted, and the ones we couldn't sort"):
        st.markdown(
            "- **Normal** = the beach's unsafe rate in the wet season (November to April) on days with no 0.1 in rain "
            "for 8+ days.\n"
            "- **Clearly above normal** = the 95% range of (rate minus normal) is entirely above zero. Ranges come "
            "from resampling whole storms 1,000 times, because tests after the same storm aren't independent.\n"
            f"- **{GROUPS['slow']['word']}**: clearly above normal on days 4 to 6.\n"
            f"- **{GROUPS['fast']['word']}**: days 2-3 and 4-6 both within 5 points of normal and not clearly above.\n"
            f"- **{GROUPS['typical']['word']}**: clearly above normal on days 2-3, near normal on days 4-6.\n"
            f"- **{GROUPS['chronic']['word']}**: unsafe on 10% or more of dry-weather tests since {RECENT_YEAR}.\n"
            f"- Not sorted: {counts.get('unclear (too few tests to tell)', 0)} beaches where the ranges were too wide to "
            f"tell, {counts.get('not enough tests', 0)} with fewer than 15 tests in a group (Avalon and many Ventura "
            f"beaches), and {counts.get('little change after rain', 0)} that rarely fail at all.\n"
            "- This is a pattern in 16 years of tests, not a health study. It supports a rule that depends on storm "
            "size and on the beach; it doesn't prove any single day was safe.")


# ---------------------------------------------------------------------------
# Page 2: Beach diagnosis
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


def diagnosis_page():
    # Start on the beach that stays unsafe longest after the advisory.
    slow = [s for s in DIAG["stations"] if group_of(s) == "slow"]
    first = max(slow, key=lambda s: s["groups"]["days 4-6"]["unsafe_rate"])["station"] if slow else DIAG["stations"][0]["station"]
    codes = sorted(DIAG_BY, key=place)
    if st.session_state.get("diag_beach") not in codes:
        st.session_state["diag_beach"] = first
    top = st.container()  # the finding sits above the picker, but is written after it (it depends on the choice)
    code = st.selectbox("Beach", codes, key="diag_beach", format_func=label,
                        help="Type to search. Starts on the beach that stays unsafe longest after rain.")
    s = DIAG_BY[code]
    g = group_of(s)
    with top:
        head, details = diagnosis_sentence(s)
        finding(f"Beach diagnosis, station {code}", head, details,  # many beaches have several stations
                notes=[NORMAL_NOTE, PRECAUTION if s["recovery"].startswith("back to normal") else None])

    trig = s["trigger"] or "no storm size reached 1 in 4"
    tide = s["after_rain_tides"]
    cards([
        (pct(s["dry_unsafe_rate"]), f"of dry-weather tests unsafe (since {RECENT_YEAR})", f"{s['dry_tests']} tests"),
        (pct(s["advisory_unsafe_rate"]), f"unsafe during rain advisories (since {RECENT_YEAR})",
         f"days 0 to 3 after rain, {s['advisory_tests']} tests"),
        (html.escape(trig), "rain that makes 1 in 4 tests fail", "storm total, tests on days 0 to 1"),
        ('<span class="v-small">' + GROUPS[g]["word"] + "</span>", "its group on the Advisory check page",
         f"normal level: {pct(s['normal_wet_season_rate'])}", GROUPS[g]["hex"]),
    ])

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
        numbers(df[["day", "unsafe", "tests"]], day="Days since rain",
                unsafe=st.column_config.NumberColumn("% unsafe", format="%.1f"), tests="Tests")

    left, right = st.columns(2, gap="medium")
    with panel(left):
        title("What the pattern points to")
        st.markdown(s["fix_text"] or "There aren't enough tests after rain to point to a kind of fix.")
        if s.get("also_after_rain"):
            st.markdown(s["also_after_rain"])
        st.caption("A pattern in the test results is a clue, not proof of the source. Source-tracking tests "
                   "(for example, DNA markers for human sewage) can confirm it.")
    with panel(right):
        title("Days 0 to 1 after rain, by storm size")
        st.caption("Dashed line: 1 in 4 tests unsafe.")
        ss = pd.DataFrame(s["by_storm_size"])
        ss["unsafe"] = ss["unsafe_rate"].astype(float) * 100
        sb = alt.Chart(ss).mark_bar(size=22, color=BLUE, cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
            x=alt.X("storm:N", sort=list(ss["storm"]), title="Storm total", axis=alt.Axis(labelAngle=0)),
            y=alt.Y("unsafe:Q", title="% unsafe", scale=alt.Scale(domain=[0, 100])),
            tooltip=[alt.Tooltip("storm:N", title="Storm"), alt.Tooltip("unsafe:Q", title="% unsafe", format=".0f"),
                     alt.Tooltip("tests:Q", title="Tests")])
        q = alt.Chart(pd.DataFrame({"y": [25]})).mark_rule(strokeDash=[4, 4], color=NAVY).encode(y="y:Q")
        show_chart((sb + q).properties(height=210), width="stretch")
        if tide["big_tide_tests"] >= 10:
            st.caption(f"Big spring tides: 3 to 7 days after rain, {pct(tide['big_tide_rate'])} of tests were unsafe on "
                       f"big-tide days ({tide['big_tide_tests']} tests) vs. {pct(tide['other_rate'])} on other days.")

    with st.expander("All beaches"):
        t = pd.DataFrame([{"Beach": label(x["station"]), "Pattern": GROUPS[group_of(x)]["word"],
                           "Dry weather": x["dry_unsafe_rate"], "During advisory": x["advisory_unsafe_rate"],
                           "Days 4-6": x["groups"]["days 4-6"]["unsafe_rate"], "Normal": x["normal_wet_season_rate"],
                           "Trigger": x["trigger"] or ""} for x in DIAG["stations"]])
        for c in ["Dry weather", "During advisory", "Days 4-6", "Normal"]:
            t[c] = t[c].astype(float) * 100
        fmt = st.column_config.NumberColumn(format="%.0f%%")
        st.dataframe(t.sort_values("Days 4-6", ascending=False), hide_index=True, width="stretch",
                     column_config={c: fmt for c in ["Dry weather", "During advisory", "Days 4-6", "Normal"]})
        st.caption(f"Dry weather and during-advisory rates use tests since {RECENT_YEAR}, so a beach that was fixed "
                   "isn't still called chronic. Days 4-6 and normal use all years since 2010, because recovery needs many storms.")


# ---------------------------------------------------------------------------
# Page 3: Did it work? (intervention evaluator)
# ---------------------------------------------------------------------------

@st.cache_resource
def eval_table():
    """Every beach test with its weather-only expected risk (from step 12), as a table."""
    d = load_json("evaluator_data.json")
    st_ = pd.DataFrame(d["stations"])
    s = np.array(d["s"])
    return pd.DataFrame({
        "StationCode": st_["station"].to_numpy()[s],
        "date": pd.Timestamp(d["day0"]) + pd.to_timedelta(d["d"], unit="D"),
        "lat": st_["lat"].to_numpy()[s], "lon": st_["lon"].to_numpy()[s],
        "exceeds": d["y"], "expected": np.array(d["e"]) / 1000})


@st.cache_data
def run_eval(station, start, end, years):
    return evaluate(eval_table(), station, start, end, years, load_json("evaluator_data.json")["widen"])


def use_preset():
    """When someone picks a known project, fill in its beach and dates."""
    demos = {d["label"]: d for d in load_json("interventions.json")["demos"]}
    d = demos.get(st.session_state["ev_preset"])
    if d:
        st.session_state.update(ev_beach=d["station"], ev_start=pd.Timestamp(d["start"]).date(),
                                ev_end=pd.Timestamp(d["end"]).date())


def evaluator_page():
    iv = load_json("interventions.json")
    demos = {d["label"]: d for d in iv["demos"]}
    calib = load_json("evaluator_data.json")["calibration"]
    ex = eval_table()
    counts = ex["StationCode"].value_counts()
    codes = sorted(counts[counts >= 100].index, key=place)
    if "ev_preset" not in st.session_state:
        st.session_state["ev_preset"] = next(iter(demos))
        use_preset()

    top = st.container()  # the finding sits above the controls, but is written after them
    with panel():
        st.selectbox("Known project", list(demos) + ["Choose my own beach and dates"], key="ev_preset",
                     on_change=use_preset)
        if st.session_state["ev_preset"] in demos:
            st.caption(DATES_NOTE)
        c1, c2, c3 = st.columns([3, 2, 2])
        code = c1.selectbox("Beach", codes, key="ev_beach", format_func=label)
        lo_d, hi_d = ex["date"].min().date(), ex["date"].max().date()
        start = c2.date_input("Work started", key="ev_start", min_value=lo_d, max_value=hi_d)
        end = c3.date_input("Work finished", key="ev_end", min_value=lo_d, max_value=hi_d)
        years = st.segmented_control("Years to compare before and after", [1, 2, 3], default=2, key="ev_years") or 2

    preset = demos.get(st.session_state["ev_preset"])
    if preset and (code, str(start), str(end)) != (preset["station"], preset["start"], preset["end"]):
        preset = None  # the fields were changed by hand
    r = run_eval(code, str(start), str(end), years)
    if "error" in r:
        with top:
            finding("Did it work?", f"Not enough data to check {html.escape(place(code))} for these dates.",
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
    with top:
        finding("Did it work?", sentence,
                f"{detail} This compares the beach before and after, adjusts for the weather, and subtracts the "
                f"change at {r['control_beaches']} beaches more than 5 km away. It shows what changed after the "
                "date; it can't rule out something else changing at the same time.",
                notes=[POINTS_NOTE])

    b, a = r["before"], r["after"]
    cards([
        (pct(b["unsafe_rate"]), f"unsafe before ({r['before_window'][0][:4]} to {r['before_window'][1][:4]})",
         f"weather alone predicts {pct(b['expected_from_weather'])}; {b['tests']} tests"),
        (pct(a["unsafe_rate"]), f"unsafe after ({r['after_window'][0][:4]} to {r['after_window'][1][:4]})",
         f"weather alone predicts {pct(a['expected_from_weather'])}; {a['tests']} tests"),
        (f"{pts(r['control_change'])}", "points of change at other beaches", "subtracted out"),
        (f"{pts(est)}", "points of change beyond weather and region", f"95% range {pts(lo)} to {pts(hi)}"),
    ])

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
        numbers(long[["month", "who", "v"]].rename(columns={"v": "points"}),
                month=st.column_config.DateColumn("Month", format="MMM YYYY"), who=" ",
                points=st.column_config.NumberColumn("Surprise (points)", format="%+.1f"))

    with panel():
        title("How sure is this?")
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
                    "[The Inertia](https://www.theinertia.com/surf/malibu-lagoon-project-manager-responds-to-allegations-that-lagoon-restoration-is-ruining-surfrider-beach/), "
                    "[SMDP](https://www.smdp.com/malibu-lagoon-reopens-after-massive-reconstruction/), "
                    "[Santa Monica Lookout](https://santamonicalookout.com/ssm_site/the_lookout/news/News-2023/August-2023/08_21_2023_Water_Quality_Under_Pier_Gives_Santa_Monica_Beaches_a_Bum_Rap.html).")


# ---------------------------------------------------------------------------
# Page 4: After the fires
# ---------------------------------------------------------------------------

def fires_page():
    pf = load_json("post_fire.json")
    res, cells_ = pf["result"], pf["cells"]
    o, w = res["overall"], res["wet_days"]
    finding("After the fires",
            f"Beaches below the 2025 Palisades burn showed no measurable change (<b>{pts(o['estimate'])} points</b>).",
            f"95% range {pts(o['ci95'][0])} to {pts(o['ci95'][1])} points. After rain they did {pts(w['estimate'])} "
            f"points worse than expected, which is what washed-in ash would look like, but that range "
            f"({pts(w['ci95'][0])} to {pts(w['ci95'][1])}) includes zero. There were few rainy days after the fire to "
            "learn from.",
            notes=[POINTS_NOTE])
    cards([(f"{pts(res[k]['estimate'])}", t, f"95% range {pts(res[k]['ci95'][0])} to {pts(res[k]['ci95'][1])}")
           for k, t in [("overall", "points, all days"), ("wet_days", "points, after rain"), ("dry_days", "points, dry days")]]
          + [(len(pf["burn_stations"]), "stations next to the burn", f"vs. {pf['control_station_count']} beaches 10+ km away")])

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
        numbers(long[["month", "group", "surprise"]], month=st.column_config.DateColumn("Month", format="MMM YYYY"),
                group=" ", surprise=st.column_config.NumberColumn("Surprise (points)", format="%+.1f"))

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
        st.markdown(f"As a check, the same comparison using the weather-only expected risk from the *Did it work?* page "
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
                            line_width_min_pixels=1, pickable=True)],
                 lat=34.1, lon=-118.35, zoom=8.4, height=320, tooltip="<b>{place}</b><br/>{km_from_burn} km from the burn area")
        st.html(chips([(ORANGE, "2025 burn areas (Palisades, left; Eaton, right)"), (NAVY, "Stations studied")]))
        st.caption(pf["eaton_note"] + " Fire perimeters: CAL FIRE.")


# ---------------------------------------------------------------------------
# Page 5: Forecast
# ---------------------------------------------------------------------------

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


def beach_map(fc_by_code, selected):
    """Map of beaches colored by today's level, plus small grey dots for sites with no forecast."""
    rows = []
    for code, s in fc_by_code.items():
        today = s["forecast"][0]
        lv = LEVELS[today["level"]]
        rows.append({"station": code, "lat": s["lat"], "lon": s["lon"], "place": place(code),
                     "status": f"{lv['word']} today ({pct(today['risk'])})", "color": lv["rgb"], "size": lv["size"]})
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
    base_map(layers, lat=33.92, lon=-118.78, zoom=7.8, height=430, tooltip="<b>{place}</b><br/>{status}",
             on_select=on_map_click, selection_mode="single-object", key="map")
    st.html(chips([(LEVELS[k]["hex"], LEVELS[k]["word"]) for k in LEVELS] + [(GREY["hex"], "No forecast")]))


def status_box(code, s):
    """Today's level, risk %, last test, and the beach's pattern from the diagnosis."""
    today = s["forecast"][0]
    lv = LEVELS[today["level"]]
    last = s["last_test"]
    result = "over the limit" if last["unsafe"] else "under the limit"
    # The beach's group, by the same name as on the Advisory check page and its map.
    kind = f"After rain: {GROUPS[group_of(DIAG_BY[code])]['word'].lower()}" if code in DIAG_BY else ""
    st.html(f"""
    <div class="box">
      <div class="where">{html.escape(place(code))}{code_tag(code)}</div>
      <div class="level"><span class="dot" style="--c:{lv['hex']}; width:14px; height:14px"></span>{lv['word']}</div>
      {meter(today['risk'], lv['hex'])}
      <div><b>{pct(today['risk'])}</b> chance a water test would be over the safe limit</div>
      <div class="small">Last test {pd.Timestamp(last['date']):%b %d, %Y}: {last['value']:,.0f} bacteria per 100 mL
        (limit {limit_of(code):.0f}), {result}</div>
      {f'<div class="small">{html.escape(kind)}</div>' if kind else ''}
    </div>""")


def day_boxes(s, replay):
    """Four small cards: today and the next 3 days. In replay mode, also what tests found."""
    out = []
    for i, d in enumerate(s["forecast"]):
        lv = LEVELS[d["level"]]
        day = pd.Timestamp(d["date"])
        name = "Today" if i == 0 and not replay else f"{day:%a} {day.day}"
        actual = ""
        if "actual" in d:  # only in replay files, on days the beach was sampled
            a = d["actual"]
            actual = f'<div class="act">Tested: <b>{"unsafe" if a["unsafe"] else "OK"}</b> ({a["value"]:,.0f})</div>'
        out.append(f'<div class="day"><b>{name}</b><div class="lv">{lv["word"]}</div>{meter(d["risk"], lv["hex"])}'
                   f'<div class="pct">{pct(d["risk"])}</div>{actual}</div>')
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


def forecast_page():
    top = st.container()  # the finding sits above the replay switch, but is written after it
    replay = st.toggle("Replay the January 2023 storm",
                       help="Shows what this app would have said on Jan 9, 2023, a few days before huge storms, "
                            "and what the water tests actually found.")
    fc = load_json("replay_2023-01-09.json" if replay else "forecast.json")
    fc_by_code = {s["station"]: s for s in fc["stations"]}
    issued = pd.Timestamp(fc["issued"])
    levels = pd.Series([s["forecast"][0]["level"] for s in fc["stations"]]).value_counts()
    warned = levels.get("caution", 0) + levels.get("high", 0)
    date_text = f"{issued:%A, %B} {issued.day}" + (f", {issued.year}" if replay else "")  # e.g. "Saturday, October 24"
    with top:
        finding("Forecast" + (" replay" if replay else ""),
                f"<b>{warned} of {len(fc['stations'])} beaches</b> {'were' if replay else 'are'} at caution or high "
                f"risk{'' if replay else ' today'}.",
                (f"{date_text}: the forecast as it would have looked that day, using the rain that actually fell "
                 "(a best case). Day cards show real test results where a beach was sampled.") if replay else
                f"{date_text}. Each beach's chance that a water test would be over the safe limit, today and the "
                f"next 3 days. Rain from {html.escape(fc['rain_source'])}.")
    st.html(chips([(LEVELS[k]["hex"], f"{levels.get(k, 0)} {v['word']}") for k, v in LEVELS.items()]))

    # The beach picker lists forecast beaches first (A to Z), then sites without a forecast.
    beach_codes = sorted(fc_by_code, key=place)
    other_codes = sorted((c for c in STATIONS.index[STATIONS["on_map"]]
                          if c not in fc_by_code and c not in HIDE_FROM_MAP), key=place)
    options = beach_codes + other_codes
    if st.session_state.get("beach") not in options:  # first visit, or switched replay on/off
        st.session_state["beach"] = max(fc_by_code, key=lambda c: fc_by_code[c]["forecast"][0]["risk"])

    left, right = st.columns([5, 6], gap="large")
    with left:  # on a phone this column comes first, so the answer is at the top
        code = st.selectbox("Beach", options, key="beach",
                            format_func=lambda c: label(c) + ("" if c in fc_by_code else " (no forecast)"),
                            help="Type to search, or tap a dot on the map. Starts on the highest-risk beach.")
        if code in fc_by_code:
            s = fc_by_code[code]
            status_box(code, s)
            day_boxes(s, replay)
            title("What the forecast is based on")
            st.markdown("\n".join(f"- {r[0].upper() + r[1:]}" for r in s["forecast"][0]["reasons"]))
        else:
            grey_box(code)
    with right:
        beach_map(fc_by_code, code)

    with panel():
        title(f"Past test results: {place(code)}")
        if replay:
            st.caption("History shows recent years, not the replay date.")
        history_chart(code)

    caution = f"{load_json('final_metrics.json')['threshold'] * 100:.0f}%"  # the warning level, set in 06
    st.caption(f"Low risk: under {caution}. Caution: {caution} to 50% (the level where this app would post a "
               "warning). High risk: 50% or more, unsafe more likely than not. Safe limit: 104 enterococcus "
               "bacteria per 100 mL of ocean water (California standard).")


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
            f"beach-days at {fm['beaches']} beaches from {start:%B %Y} to {end:%B %Y} that the model never saw while "
            "learning.",
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
        st.caption("\"3 tests\" = signs posted when any of the three state ocean tests fails (enterococcus, "
                   f"fecal coliform, total coliform), which is what counties post on. Against signs alone: "
                   f"{gain(test['sign.gain'])}.")

    d3 = day3["sign_plus_rain.gain"]
    d3_verdict = ("the whole range is above zero, so it does better than today's signs" if d3[1] > 0 else
                  "the range includes zero, so the 3-day forecast is about as good as today's signs, not better")
    caution = f"{fm['threshold'] * 100:.0f}%"
    st.markdown("#### The forecast")
    st.markdown(
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
        f"for signs and the rain advisory on the day itself: {gain(d3)}; {d3_verdict}.")

    defs = DIAG["definitions"]
    calib = load_json("evaluator_data.json")["calibration"]
    st.markdown("#### Advisory check and beach diagnosis")
    st.markdown(
        f"- **Days since rain:** {defs['day']}. Rain is a daily total, so a day-3 test came roughly 48 to 96 hours "
        "after the rain.\n"
        f"- **Normal:** {defs['normal']}.\n"
        f"- **Dry weather:** {defs['dry']}. **{GROUPS['chronic']['word']}:** {defs['chronic']}.\n"
        f"- **Trigger:** {defs['trigger']}.\n"
        f"- **Clearly above:** {defs['clearly_above']}. Groups need {defs['min_tests']}+ tests.\n"
        "- Fix categories describe what a pattern points to, not a proven source.")
    st.markdown("#### Did it work?")
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
    st.markdown("#### Checked and ruled out: sewage spills")
    st.markdown(
        f"Since 2023, {sm['spills_with_beach_samples']} large sewage spills (10,000+ gallons reaching rivers, drains or "
        f"the ocean) had {sm['beach_samples_after_spills']} beach tests within 10 km in the following week. "
        f"{pct(sm['observed_unsafe_rate'])} of those tests were unsafe; the model, which knows nothing about spills, "
        f"expected {pct(sm['predicted_unsafe_rate'])}. Several spills happened during big storms, when rain alone made "
        "the water unsafe, so the model doesn't use spill data.")

    st.markdown("#### Limits")
    st.markdown(
        "- Results count days a beach was actually tested, and beaches are often retested after a bad result.\n"
        "- Warning signs were rebuilt from the test data, not from records of real postings.\n"
        "- Rain forecasts get less accurate further ahead, so later forecast days are less certain.\n"
        "- This is a student project, not an official health advisory. Always follow posted signs.")
    st.markdown("#### Data and AI use")
    st.markdown(
        "Data: California State Water Boards (Safe to Swim), Open-Meteo (ERA5 rain), NOAA (tides), CIWQS (sewage "
        "spills), CAL FIRE (burn areas). Parts of the code and analysis were written with AI help (Claude Code); "
        "every AI-assisted step is listed in the project's AI use log.")


# ---------------------------------------------------------------------------
# Home
# ---------------------------------------------------------------------------

HERO_PHOTO = ROOT / "assets" / "hero.jpg"


def home_page():
    fm = load_json("final_metrics.json")["scenarios"]["test"]
    n_slow = DIAG["region"]["recovery_groups"]["slow"]["beaches"]
    n_chronic = sum(s["chronic"] for s in DIAG["stations"])
    first_year, last_year = DIAG["period"][:4], DIAG["period"][-10:-6]

    left, right = st.columns([6, 5], gap="large", vertical_alignment="center")
    with left:
        st.html(f'<div class="home-hero"><h1>{APP_NAME}</h1><p>See which LA beaches stay unsafe after rain, why, '
                f'and whether cleanup projects worked, from {int(last_year) - int(first_year)} years of state '
                'water tests.</p></div>')
    with right:
        if HERO_PHOTO.exists():
            st.image(str(HERO_PHOTO), width="stretch")
        else:  # the space is kept so the layout doesn't change once a photo is added
            st.html('<div class="photo">Photo goes here: add assets/hero.jpg</div>')

    st.html('<div class="section-h">The problem</div><div class="problem">'
            "<p>Rain washes bacteria from streets and storm drains into the ocean, and swimmers can get sick.</p>"
            "<p>LA County gives every beach the same rule: stay out for 72 hours after rain.</p>"
            "<p>But beaches recover at different speeds, and lab results take a day or more to come back.</p></div>")

    st.html('<div class="section-h">What we found</div>'
            f'<div class="big3">'
            f'<div class="card"><div class="v">{n_slow}</div><div class="l">beaches stay unsafe longer than '
            "LA County's 72-hour rain advisory</div></div>"
            f'<div class="card"><div class="v">{n_chronic}</div><div class="l">beaches are often unsafe even in '
            f"dry weather, so rain isn't their only problem</div></div>"
            f'<div class="card"><div class="v">{pct(fm["sign_plus_rain.model_caught"][0])}</div><div class="l">of '
            f"unsafe days caught by the forecast, vs. {pct(fm['sign_plus_rain.caught'][0])} for warning signs "
            "plus the rain advisory</div></div></div>")

    # One card per question, each opening the page that answers it.
    st.html('<div class="section-h">Start with a question</div>')
    questions = [
        ("Is 72 hours the right wait for every beach?", "advisory"),
        ("Why is this beach unsafe, and what would fix it?", "diagnosis"),
        ("Was a beach cleaner after a cleanup project?", "evaluator"),
        ("Did the 2025 fires make beaches dirtier?", "fires"),
        ("Which beaches are risky today?", "forecast"),
        ("How do you know?", "methods"),
    ]
    for row in (questions[:3], questions[3:]):
        for col, (q, page) in zip(st.columns(3), row):
            with panel(col):
                st.html(f'<div class="q">{html.escape(q)}</div>')
                st.page_link(PAGES[page])  # the page's own icon and name

    st.html('<div class="section-h">Who it\'s for</div><div class="who">'
            "<p><b>City and county water staff</b>: see which beaches need a longer advisory, and what kind of fix "
            "each one points to.</p>"
            "<p><b>Lawmakers and their staff</b>: check whether a cleanup project was followed by cleaner water.</p>"
            "<p><b>Beachgoers</b>: see each beach's risk today and for the next 3 days.</p></div>")

    st.html('<div class="trust">'
            f'<div><h4>Data</h4><p>California State Water Boards beach tests ({first_year} to {last_year}), '
            "Open-Meteo rain, NOAA tides, CIWQS sewage spills, CAL FIRE burn areas.</p></div>"
            "<div><h4>What unsafe means</h4><p>A water test over 104 enterococcus bacteria per 100 mL, California's "
            "standard for ocean beaches.</p></div>"
            '<div><h4>How it works</h4><p><a href="methods" target="_self">Read the methods</a>: how each number '
            "was made, and its limits.</p></div>"
            "<div><h4>Built by Dylan Lin</h4><p>For the 2026 Congressional App Challenge. AI help is listed "
            "in the methods.</p></div></div>")


# ---------------------------------------------------------------------------
# Page frame
# ---------------------------------------------------------------------------

# The left menu (like a weather map's menu): wordmark, tagline, then one link per page.
# On a phone the menu folds away, so a small wordmark shows at the top of the page instead.
st.sidebar.html(f'<div class="side-brand"><span class="mark"></span>{APP_NAME}</div>'
                f'<div class="side-tag">{TAGLINE}</div>')
st.html(f'<div class="brand-main"><span class="mark"></span>{APP_NAME}</div>')
check_model_ids()
PAGES = {  # also used by the home page's question cards
    "home": st.Page(home_page, title="Home", icon=":material/home:", url_path="home", default=True),
    "advisory": st.Page(advisory_page, title="Advisory check", icon=":material/rainy:", url_path="advisory"),
    "diagnosis": st.Page(diagnosis_page, title="Beach diagnosis", icon=":material/beach_access:", url_path="diagnosis"),
    "evaluator": st.Page(evaluator_page, title="Did it work?", icon=":material/compare_arrows:", url_path="did-it-work"),
    "fires": st.Page(fires_page, title="After the fires", icon=":material/local_fire_department:", url_path="fires"),
    "forecast": st.Page(forecast_page, title="Forecast", icon=":material/partly_cloudy_day:", url_path="forecast"),
    "methods": st.Page(methods_page, title="Methods", icon=":material/menu_book:", url_path="methods"),
}
st.navigation(list(PAGES.values()), position="sidebar").run()

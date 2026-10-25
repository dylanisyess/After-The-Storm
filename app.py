"""Beach water risk app (Congressional App Challenge 2026).

Run inside WSL:  source ~/waterenv/bin/activate && streamlit run app.py

The app only READS the JSON files in app_data/. To refresh them, run
pipeline/08_forecast.py and then pipeline/09_export_app_data.py.

Tabs:
  Forecast        map + one beach's status card, 3-day forecast and test history
  Causes          which beaches are chronic vs. rain-driven, and the sewage-spill check
  After the Fires did beaches near the 2025 Palisades fire get worse?
  How It Works    the model compared with today's warning signs
"""

import html
import json
from pathlib import Path

import altair as alt
import pandas as pd
import pydeck as pdk
import streamlit as st

DATA = Path(__file__).parent / "app_data"

st.set_page_config(page_title="Beach Water Forecast", page_icon="🌊", layout="wide")


# ---------------------------------------------------------------------------
# Shared settings: colors, names, and the numbers we quote
# ---------------------------------------------------------------------------

# One color per risk level, used everywhere (map dots, cards, legends).
# Each color always appears next to its word, so the app works for colorblind users.
# "tint" is a pale version for card backgrounds; "size" makes riskier dots bigger on the map.
LEVELS = {
    "low":     {"word": "Low risk",  "hex": "#0ca30c", "tint": "#e8f6e8", "rgb": [12, 163, 12],  "size": 6},
    "caution": {"word": "Caution",   "hex": "#fab219", "tint": "#fff4d9", "rgb": [250, 178, 25], "size": 8},
    "high":    {"word": "High risk", "hex": "#d03b3b", "tint": "#fbe8e8", "rgb": [208, 59, 59],  "size": 10},
}
GREY = {"hex": "#9aa0a6", "tint": "#eef0f2", "rgb": [154, 160, 166]}  # sites with no forecast
NAVY = "#0b2545"
BLUE = "#2a78d6"    # chart color for "this app's model" / "under the limit"
ORANGE = "#eb6834"  # chart color for the thing we compare against

# Offshore receiving-water sites near Palos Verdes: not beaches and not inland, so never mapped.
HIDE_FROM_MAP = {"Peninsula-RW1", "Peninsula-RW2"}

# Map dot sizes are in screen pixels, so dots stay tappable at any zoom.
# pydeck reads plain strings as formulas, so a fixed word must be wrapped in pdk.types.String.
PIXELS = pdk.types.String("pixels")

# A few station codes stand for a place the state data doesn't spell out.
# MDRH = Marina del Rey Harbor (inside the harbor, so it would be wrong to call it "near Venice Beach").
CODE_PLACES = {"MDRH": "Marina del Rey Harbor"}

# Plain-language labels for the beach types in hotspots.json.
TYPE_TEXT = {
    "chronic": "Chronic: often unsafe even in dry weather",
    "rain-driven": "Rain-driven: mostly unsafe after rain",
    "usually clean": "Usually clean",
}


# A little CSS for the things the theme can't do: the colored status cards and chips.
st.html("""
<style>
  [data-testid="stMainBlockContainer"] { max-width: 1200px; padding-top: 2rem; }
  .hero  { border-radius: 18px; padding: 18px 20px; background: var(--tint);
           border-left: 10px solid var(--c); color: #0b2545; }
  .hero .where { font-size: 1.15rem; font-weight: 600; }
  .hero .code  { font-size: .8rem; opacity: .65; margin-left: 6px; }
  .hero .level { font-size: 2.4rem; font-weight: 800; line-height: 1.1; margin: 6px 0 2px; }
  .hero .small { font-size: .95rem; opacity: .85; margin-top: 4px; }
  .days  { display: flex; flex-wrap: wrap; gap: 8px; margin: 12px 0 4px; }
  .day   { flex: 1 1 70px; border-radius: 14px; padding: 10px 6px; text-align: center;
           background: var(--tint); border-top: 6px solid var(--c); color: #0b2545; }
  .day b { display: block; font-size: .85rem; }
  .day .lv  { font-weight: 700; font-size: .95rem; }
  .day .pct { font-size: .85rem; opacity: .8; }
  .day .act { font-size: .75rem; margin-top: 4px; }
  .chips { display: flex; flex-wrap: wrap; gap: 8px; margin: 4px 0 8px; }
  .chip  { border-radius: 999px; padding: 3px 12px; background: #ffffff;
           border: 1px solid #d6e2ec; font-size: .9rem; color: #0b2545; }
  .dot   { display: inline-block; width: 11px; height: 11px; border-radius: 50%;
           background: var(--c); margin-right: 6px; vertical-align: -1px; }
</style>
""")


# ---------------------------------------------------------------------------
# Loading data
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
    return df.set_index("station")


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
    """Name plus code, e.g. 'Solimar Beach, Ventura · 10000' (just the code if it has no name)."""
    return place(code) if place(code) == code else f"{place(code)} · {code}"


def code_tag(code):
    """The station code, shown small after the name (skipped when the name already is the code)."""
    return "" if place(code) == code else f'<span class="code">{html.escape(code)}</span>'


def limit_of(code):
    """The state limit for this station: 104 for enterococcus (ocean), 320 for E. coli (fresh water)."""
    return float(STATIONS.at[code, "standard"].split(">")[1])


def pct(x):
    return f"{x * 100:.0f}%"


def chips(items):
    """A row of small rounded labels, each with a colored dot and its word: [(hex, text), ...]."""
    spans = "".join(f'<span class="chip"><span class="dot" style="--c:{c}"></span>{html.escape(t)}</span>'
                    for c, t in items)
    return f'<div class="chips">{spans}</div>'


# ---------------------------------------------------------------------------
# Forecast tab
# ---------------------------------------------------------------------------

def on_map_click():
    """When someone taps a dot on the map, select that beach in the picker."""
    objects = st.session_state["map"].selection.get("objects", {})
    for layer in ("beaches", "other_sites"):
        if objects.get(layer):
            st.session_state["beach"] = objects[layer][0]["station"]


def beach_map(fc_by_code, selected):
    """Map of beaches colored by today's level, plus small grey dots for sites with no forecast."""
    # Beaches with a forecast: color and size come from today's level.
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
    # A navy ring around the selected beach.
    if selected in STATIONS.index:
        ring = pd.DataFrame([{"lat": STATIONS.at[selected, "lat"], "lon": STATIONS.at[selected, "lon"]}])
        layers.append(pdk.Layer("ScatterplotLayer", data=ring, get_position=["lon", "lat"], filled=False,
                                stroked=True, get_line_color=[11, 37, 69], line_width_min_pixels=3,
                                get_radius=14, radius_units=PIXELS))

    deck = pdk.Deck(
        layers=layers,
        initial_view_state=pdk.ViewState(latitude=33.9, longitude=-118.7, zoom=7.3),
        map_provider="carto", map_style=pdk.map_styles.CARTO_LIGHT,  # clean light map, no API key needed
        tooltip={"html": "<b>{place}</b><br/>{status}",
                 "style": {"backgroundColor": "white", "color": NAVY, "fontSize": "13px"}},
    )
    st.pydeck_chart(deck, height=430, on_select=on_map_click, selection_mode="single-object", key="map")
    legend = [(LEVELS[k]["hex"], LEVELS[k]["word"]) for k in LEVELS] + [(GREY["hex"], "No forecast (past study or inland)")]
    st.html(chips(legend))


def no_forecast_reason(code):
    """Short explanation for a site that has no forecast."""
    s = STATIONS.loc[code]
    if isinstance(s["note"], str):  # past study sites have a note; missing notes are blank (NaN)
        return s["note"].capitalize()
    if s["water"] == "freshwater":
        return "River or creek site (tested for E. coli); forecasts cover ocean beaches only"
    return "No tests in the past year, so no forecast"


def hero_card(code, s):
    """Big status card for one beach: today's level, risk %, last test."""
    today = s["forecast"][0]
    lv = LEVELS[today["level"]]
    last = s["last_test"]
    result = "over the limit" if last["unsafe"] else "under the limit"
    kind = TYPE_TEXT.get(s["type"], "")
    st.html(f"""
    <div class="hero" style="--c:{lv['hex']}; --tint:{lv['tint']}">
      <div class="where">{html.escape(place(code))}{code_tag(code)}</div>
      <div class="level">{lv['word']}</div>
      <div><b>{pct(today['risk'])}</b> chance a water test would be over the safe limit</div>
      <div class="small">Last test {pd.Timestamp(last['date']):%b %d, %Y}: {last['value']:,.0f} bacteria per 100 mL
        (limit {limit_of(code):.0f}), {result}</div>
      {f'<div class="small">{html.escape(kind)}</div>' if kind else ''}
    </div>""")


def day_cards(s, replay):
    """Row of 4 small cards: today and the next 3 days. In replay mode, also what tests found."""
    cards = []
    for i, d in enumerate(s["forecast"]):
        lv = LEVELS[d["level"]]
        day = pd.Timestamp(d["date"])
        label = "Today" if i == 0 and not replay else f"{day:%a} {day.day}"
        actual = ""
        if "actual" in d:  # only in replay files, on days the beach was sampled
            a = d["actual"]
            actual = f'<div class="act">Tested: <b>{"unsafe" if a["unsafe"] else "OK"}</b> ({a["value"]:,.0f})</div>'
        cards.append(f'<div class="day" style="--c:{lv["hex"]}; --tint:{lv["tint"]}"><b>{label}</b>'
                     f'<div class="lv">{lv["word"]}</div><div class="pct">{pct(d["risk"])}</div>{actual}</div>')
    st.html(f'<div class="days">{"".join(cards)}</div>')


def grey_card(code):
    """Card for a site without a forecast (past study site, inactive beach, or inland site)."""
    s = STATIONS.loc[code]
    last = s["last_test"]
    st.html(f"""
    <div class="hero" style="--c:{GREY['hex']}; --tint:{GREY['tint']}">
      <div class="where">{html.escape(s['place'])}{code_tag(code)}</div>
      <div class="level">No forecast</div>
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

    result_scale = alt.Scale(domain=["Under the limit", "Over the limit"], range=[BLUE, LEVELS["high"]["hex"]])
    points = alt.Chart(df).mark_point(filled=True, size=70, opacity=0.85).encode(
        x=alt.X("date:T", title=None),
        y=alt.Y("shown:Q", scale=alt.Scale(type="log"), title="Bacteria per 100 mL (log scale)",
                axis=alt.Axis(values=[1, 10, 100, 1000, 10000, 100000], format=",")),  # each line = 10x more
        color=alt.Color("result:N", scale=result_scale, legend=alt.Legend(title=None, orient="top")),
        shape=alt.Shape("result:N", scale=alt.Scale(domain=result_scale.domain, range=["circle", "diamond"])),
        tooltip=[alt.Tooltip("date:T", title="Date", format="%b %d, %Y"),
                 alt.Tooltip("value:Q", title="Bacteria per 100 mL", format=",.0f"),
                 alt.Tooltip("result:N", title="Result")],
    )
    line = pd.DataFrame({"y": [limit], "label": [f"Safe limit: {limit:.0f}"]})
    rule = alt.Chart(line).mark_rule(strokeDash=[6, 4], color=NAVY).encode(y="y:Q")
    rule_label = alt.Chart(line).mark_text(align="left", dx=4, dy=-7, color=NAVY, fontSize=12).encode(
        y="y:Q", x=alt.value(0), text="label:N")
    st.altair_chart((points + rule + rule_label).properties(height=280), width="stretch")
    first, last = df["date"].min(), df["date"].max()
    st.caption(f"{len(df)} tests from {first:%b %Y} to {last:%b %Y}, {df['unsafe'].sum()} over the limit. "
               "Hover or tap a dot for details.")


def forecast_tab():
    replay = st.toggle("Replay the January 2023 storm",
                       help="Shows what this app would have said on Jan 9, 2023, a few days before huge storms, "
                            "and what the water tests actually found.")
    fc = load_json("replay_2023-01-09.json" if replay else "forecast.json")
    fc_by_code = {s["station"]: s for s in fc["stations"]}
    issued = pd.Timestamp(fc["issued"])

    if replay:
        st.info(f"**Replay:** the forecast as it would have looked on {issued:%A, %b %d, %Y}. "
                "It uses the rain that actually fell, so it is a best case. Day cards show real test results "
                "where a beach was sampled.")
    else:
        st.caption(f"Forecast for {issued:%A, %B %d, %Y} · rain from {fc['rain_source']}")

    # Today's totals, e.g. "120 Low risk · 10 Caution · 2 High risk".
    counts = pd.Series([s["forecast"][0]["level"] for s in fc["stations"]]).value_counts()
    st.html(chips([(LEVELS[k]["hex"], f"{counts.get(k, 0)} {v['word']}") for k, v in LEVELS.items()]))

    # The beach picker lists forecast beaches first (A to Z), then sites without a forecast.
    beach_codes = sorted(fc_by_code, key=place)
    other_codes = sorted((c for c in STATIONS.index[STATIONS["on_map"]]
                          if c not in fc_by_code and c not in HIDE_FROM_MAP), key=place)
    options = beach_codes + other_codes
    if st.session_state.get("beach") not in options:  # first visit, or switched replay on/off
        st.session_state["beach"] = max(fc_by_code, key=lambda c: fc_by_code[c]["forecast"][0]["risk"])

    left, right = st.columns([5, 6], gap="large")
    with left:  # on a phone this column comes first, so the answer is at the top
        code = st.selectbox(
            "Beach", options, key="beach",
            format_func=lambda c: label(c) + ("" if c in fc_by_code else " (no forecast)"),
            help="Type to search, or tap a dot on the map. Starts on today's highest-risk beach.")
        if code in fc_by_code:
            s = fc_by_code[code]
            hero_card(code, s)
            day_cards(s, replay)
            st.markdown("**What today's forecast is based on**")
            st.markdown("\n".join(f"- {r[0].upper() + r[1:]}" for r in s["forecast"][0]["reasons"]))
        else:
            grey_card(code)
    with right:
        beach_map(fc_by_code, code)

    with st.container(border=True):
        st.markdown(f"**Past test results: {place(code)}**")
        if replay:
            st.caption("History shows recent years, not the replay date.")
        history_chart(code)

    caution = f"{load_json('final_metrics.json')['threshold'] * 100:.0f}%"  # the warning level, set in 06
    st.caption(f"Low risk: under {caution} · Caution: {caution} to 50% (the level where this app would post a "
               "warning) · High risk: 50% or more, unsafe more likely than not. Safe limit: 104 enterococcus "
               "bacteria per 100 mL of ocean water (California standard).")


# ---------------------------------------------------------------------------
# Causes tab
# ---------------------------------------------------------------------------

def causes_tab():
    hot = load_json("hotspots.json")
    df = pd.DataFrame(hot["stations"])
    df["beach"] = [label(c) for c in df["station"]]
    df["dry"] = df["dry_unsafe_rate"] * 100
    df["wet"] = df["wet_unsafe_rate"] * 100

    st.subheader("Why some beaches fail")
    st.markdown("We sorted every beach by how often it failed in **dry weather** versus **after rain**, "
                f"using tests since {pd.Timestamp(hot['period'].split(' ')[0]):%B %Y}. This splits beaches into three kinds:")

    cols = st.columns(3)
    blurbs = {
        "chronic": "Unsafe on 10% or more of dry days. Something local (storm drains, birds, a pier, "
                   "poor circulation) keeps bacteria high even without rain.",
        "rain-driven": "Usually fine when dry, but unsafe on 30%+ of days after rain, when runoff "
                       "washes streets and drains into the ocean.",
        "usually clean": "Rarely unsafe, even after rain.",
    }
    for col, (kind, text) in zip(cols, blurbs.items()):
        with col.container(border=True):
            st.metric(TYPE_TEXT[kind].split(":")[0], f"{(df['type'] == kind).sum()} beaches")
            st.caption(text)

    # Scatter: each dot is a beach. Further right = dirtier in dry weather; higher = dirtier after rain.
    with st.container(border=True):
        st.markdown("**Each dot is a beach**")
        # Legends are drawn as chips (like the map's) because chart legends get cut off on phones.
        st.html(chips([(ORANGE, "Chronic"), (BLUE, "Rain-driven"), ("#1baf7a", "Usually clean")]))
        kind_scale = alt.Scale(domain=["chronic", "rain-driven", "usually clean"], range=[ORANGE, BLUE, "#1baf7a"])
        dots = alt.Chart(df).mark_circle(size=80, opacity=0.8, stroke="white", strokeWidth=1).encode(
            x=alt.X("dry:Q", title="% of dry days unsafe"),
            y=alt.Y("wet:Q", title="% of days after rain unsafe"),
            color=alt.Color("type:N", scale=kind_scale, legend=None),
            tooltip=[alt.Tooltip("beach:N", title="Beach"), alt.Tooltip("dry:Q", title="Dry days unsafe %", format=".0f"),
                     alt.Tooltip("wet:Q", title="After rain unsafe %", format=".0f"),
                     alt.Tooltip("samples:Q", title="Tests")],
        )
        cutoff = alt.Chart(pd.DataFrame({"x": [10]})).mark_rule(strokeDash=[4, 4], color="#8a99a8").encode(x="x:Q")
        st.altair_chart((dots + cutoff).properties(height=340), width="stretch")
        st.caption("Dashed line: 10% of dry days. Beaches to the right of it are chronic. "
                   f"Dry = {hot['dry_definition']}. After rain = {hot['wet_definition']}.")

    st.markdown("**Beaches most often unsafe in dry weather**")
    worst = df[df["type"] == "chronic"].sort_values("dry", ascending=False).head(10)
    st.dataframe(
        worst[["beach", "dry", "wet", "samples"]], hide_index=True, width="stretch",
        column_config={
            "beach": "Beach",
            "dry": st.column_config.ProgressColumn("Dry days unsafe", format="%d%%", min_value=0, max_value=100),
            "wet": st.column_config.ProgressColumn("After rain unsafe", format="%d%%", min_value=0, max_value=100),
            "samples": st.column_config.NumberColumn("Tests"),
        })

    st.divider()
    spills_section()


def spills_section():
    sp = load_json("spills.json")
    sm = sp["summary"]
    st.subheader("Do sewage spills make beaches worse?")
    st.markdown(
        f"Since 2023, **{sm['spills_with_beach_samples']} large sewage spills** (10,000+ gallons reaching "
        f"rivers, drains or the ocean) had beach tests within 10 km in the following week, "
        f"**{sm['beach_samples_after_spills']} tests** in all. "
        f"**{pct(sm['observed_unsafe_rate'])}** of those tests were unsafe. Our model, which knows nothing "
        f"about spills, expected **{pct(sm['predicted_unsafe_rate'])}**.")
    st.markdown(
        "So the beaches were about as dirty as rain, tides and their own history already predicted. Several of "
        "these spills happened during big storms (January 2023, February 2024), when the rain alone made the "
        "water unsafe. That is why the model doesn't use spill data: it didn't make the forecasts better.")

    ev = pd.DataFrame(sp["events"])
    ev["where"] = ev["location"].str.replace("\x96", "–")  # fix a garbled dash in the state data
    # Some locations start with a sewer ID, like "1438-0079 (4100 Admiralty Way)": keep the address.
    ev["where"] = ev["where"].str.replace(r"^[\d-]+ \((.*)\)$", r"\1", regex=True)
    ev["label"] = [f"{pd.Timestamp(d):%b} {pd.Timestamp(d).day}, {d[:4]} · {w[:20]}"  # full date: two spills share a sewer
                   for d, w in zip(ev["date"], ev["where"])]
    # Long format: one row for "what tests found" and one for "what the model expected".
    long = pd.concat([
        ev.assign(what="Tests found", rate=ev["observed_unsafe_rate"] * 100),
        ev.assign(what="Model expected", rate=ev["predicted_unsafe_rate"] * 100),
    ])
    order = list(ev.sort_values("date")["label"])
    with st.container(border=True):
        st.markdown("**Unsafe tests in the week after each spill**")
        st.html(chips([(ORANGE, "Tests found"), (BLUE, "Model expected")]))
        base = alt.Chart(long).encode(y=alt.Y("label:N", sort=order, title=None, axis=alt.Axis(labelLimit=170)))
        bar = base.mark_rule(color="#c3cfd9", strokeWidth=2).encode(x="min(rate):Q", x2="max(rate):Q")
        pts = base.mark_point(filled=True, size=90).encode(
            x=alt.X("rate:Q", title="% of tests unsafe", scale=alt.Scale(domain=[0, 100])),
            color=alt.Color("what:N", scale=alt.Scale(domain=["Tests found", "Model expected"], range=[ORANGE, BLUE]),
                            legend=None),
            shape=alt.Shape("what:N", scale=alt.Scale(domain=["Tests found", "Model expected"], range=["circle", "square"]),
                            legend=None),
            tooltip=[alt.Tooltip("where:N", title="Spill"), alt.Tooltip("date:N", title="Date"),
                     alt.Tooltip("surface_gallons:Q", title="Gallons", format=","),
                     alt.Tooltip("beach_samples_next_7_days:Q", title="Beach tests"),
                     alt.Tooltip("what:N", title=" "), alt.Tooltip("rate:Q", title="% unsafe", format=".0f")],
        )
        st.altair_chart((bar + pts).properties(height=360), width="stretch")
        st.caption("Some spills had only 3 or 4 tests nearby, so single spills can look better or worse by chance. "
                   "Spill records: California CIWQS sanitary sewer overflow database.")


# ---------------------------------------------------------------------------
# After the Fires tab
# ---------------------------------------------------------------------------

def fires_tab():
    pf = load_json("post_fire.json")
    res, cells = pf["result"], pf["cells"]
    pts = lambda x: f"{x * 100:+.1f}"  # 0.071 -> "+7.1"

    st.subheader("Did the 2025 fires make beaches dirtier?")
    st.markdown(
        "The Palisades fire (January 7, 2025) burned right down to the coast at Malibu and Will Rogers State "
        "Beach. Ash and burned soil can wash into the ocean when it rains, so we checked whether the "
        f"**{len(pf['burn_stations'])} beach stations next to the burn** got worse than "
        f"the **{pf['control_station_count']} other beaches**.")

    with st.expander("How we checked (in plain words)"):
        st.markdown(
            "1. The model predicts each test from rain, tides and the beach's history. It knows nothing about fires.\n"
            "2. For every test we measured the **surprise**: did the beach fail more often than the model expected?\n"
            "3. We compared the surprise **before vs. after** the fire, for beaches **near the burn vs. everywhere "
            "else**. The other beaches show what changed for reasons unrelated to the fire (weather, labs). "
            "Scientists call this a *difference-in-differences*.\n"
            "4. The 95% range comes from resampling beaches many times. If it includes zero, we can't tell "
            "the change apart from chance.")

    # The three results as big numbers.
    c1, c2, c3 = st.columns(3)
    for col, key, title in [(c1, "overall", "All days"), (c2, "wet_days", "After rain"), (c3, "dry_days", "Dry days")]:
        r = res[key]
        with col.container(border=True):
            st.metric(title, f"{pts(r['estimate'])} points")
            st.caption(f"95% range {pts(r['ci95'][0])} to {pts(r['ci95'][1])}")

    st.markdown(
        "**The honest answer: we found no measurable change.** Overall the change is about zero. After rain, "
        f"beaches near the burn did {pts(res['wet_days']['estimate'])} points worse than expected, which is what "
        "you'd expect if ash washed in, but the range runs from a drop to a big rise, so it could just be chance. "
        "There were few rainy days after the fire to learn from.")
    st.markdown(
        f"Beaches near the burn were already failing more than the model expected before the fire "
        f"({pts(cells['burn_before']['mean_residual'])} points), and {pts(cells['burn_after']['mean_residual'])} "
        f"after. But other beaches also moved, from {pts(cells['control_before']['mean_residual'])} to "
        f"{pts(cells['control_after']['mean_residual'])}, so only the difference between the two changes "
        f"({pts(res['overall']['estimate'])} points) could be due to the fire.")

    # Monthly surprise chart, burn-area beaches vs. all other beaches.
    m = pd.DataFrame(pf["monthly"])
    m["month"] = pd.to_datetime(m["month"])
    long = pd.concat([
        m.assign(group="Beaches near the burn", surprise=m["burn_residual"] * 100),
        m.assign(group="All other beaches", surprise=m["control_residual"] * 100),
    ])
    with st.container(border=True):
        st.markdown("**Monthly surprise: unsafe rate minus what the model expected**")
        st.html(chips([(ORANGE, "Beaches near the burn"), (BLUE, "All other beaches")]))
        lines = alt.Chart(long).mark_line(point=alt.OverlayMarkDef(size=40), strokeWidth=2).encode(
            x=alt.X("month:T", title=None),
            y=alt.Y("surprise:Q", title="Percentage points"),
            color=alt.Color("group:N", scale=alt.Scale(domain=["Beaches near the burn", "All other beaches"],
                                                       range=[ORANGE, BLUE]),
                            legend=None),
            tooltip=[alt.Tooltip("month:T", title="Month", format="%b %Y"), alt.Tooltip("group:N", title=" "),
                     alt.Tooltip("surprise:Q", title="Surprise (points)", format="+.1f")],
        )
        zero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color="#8a99a8").encode(y="y:Q")
        fire = pd.DataFrame({"x": [pd.Timestamp("2025-01-07")], "label": ["Fires"]})
        fire_rule = alt.Chart(fire).mark_rule(strokeDash=[6, 4], color=NAVY).encode(x="x:T")
        fire_text = alt.Chart(fire).mark_text(align="left", dx=4, y=8, color=NAVY).encode(x="x:T", text="label:N")
        st.altair_chart((zero + lines + fire_rule + fire_text).properties(height=320), width="stretch")
        st.caption("Above zero = more unsafe tests than the model expected. Monthly values jump around because "
                   "some months have few tests near the burn.")

    # Map of the burn areas with the stations we studied.
    with st.container(border=True):
        st.markdown("**Burn areas and the beach stations next to them**")
        near = pd.DataFrame(pf["burn_stations"])
        near = near.assign(lat=[STATIONS.at[c, "lat"] for c in near["station"]],
                           lon=[STATIONS.at[c, "lon"] for c in near["station"]],
                           place=[place(c) for c in near["station"]])
        deck = pdk.Deck(
            layers=[
                pdk.Layer("GeoJsonLayer", data=load_json("burn_areas_2025.geojson"), get_fill_color=[235, 104, 52, 90],
                          get_line_color=[235, 104, 52], line_width_min_pixels=1),
                pdk.Layer("ScatterplotLayer", data=near, get_position=["lon", "lat"], get_fill_color=[11, 37, 69],
                          get_radius=6, radius_units=PIXELS, stroked=True, get_line_color=[255, 255, 255],
                          line_width_min_pixels=1, pickable=True),
            ],
            initial_view_state=pdk.ViewState(latitude=34.1, longitude=-118.35, zoom=8.4),
            map_provider="carto", map_style=pdk.map_styles.CARTO_LIGHT,
            tooltip={"html": "<b>{place}</b><br/>{km_from_burn} km from the burn area",
                     "style": {"backgroundColor": "white", "color": NAVY}},
        )
        st.pydeck_chart(deck, height=340)
        st.html(chips([("#eb6834", "2025 burn areas (Palisades, left; Eaton, right)"), (NAVY, "Stations studied")]))
        st.caption(pf["eaton_note"] + " Fire perimeters: CAL FIRE.")


# ---------------------------------------------------------------------------
# How It Works tab
# ---------------------------------------------------------------------------

def gain(v):
    """[0.05, 0.019, 0.086] -> '+5.0 points (95% range +1.9 to +8.6)'."""
    return f"{v[0] * 100:+.1f} points (95% range {v[1] * 100:+.1f} to {v[2] * 100:+.1f})"


def how_tab():
    # All numbers on this page come from app_data/final_metrics.json, written by
    # pipeline/10_final_metrics.py together with review/final_metrics.md, so they always match.
    # Each number is [value, low, high]: the result and its 95% range.
    fm = load_json("final_metrics.json")
    test, live, day3 = fm["scenarios"]["test"], fm["scenarios"]["live"], fm["scenarios"]["day3"]
    start, end = (pd.Timestamp(d) for d in fm["test_period"])

    st.subheader("Better warnings with the same number of warnings")
    st.markdown(
        "Today, LA beaches get a warning sign when a water test comes back over the limit, plus LA County's "
        "advice to stay out of the ocean for 72 hours after rain. We tested whether a forecast could do better.")

    c1, c2 = st.columns(2)
    with c1.container(border=True):
        st.metric("This app's forecast", pct(test["sign_plus_rain.model_caught"][0]))
        st.caption("of unsafe days caught")
    with c2.container(border=True):
        st.metric("Signs + rain advisory", pct(test["sign_plus_rain.caught"][0]))
        st.caption("of unsafe days caught")
    st.markdown(
        f"Both warn on the same number of days ({test['sign_plus_rain.share'][0] * 100:.1f}% of tested days), so "
        f"the forecast isn't winning by just warning more. Difference: **{gain(test['sign_plus_rain.gain'])}**. "
        f"These are {fm['tested_beach_days']:,} tested beach-days at {fm['beaches']} beaches from {start:%B %Y} "
        f"to {end:%B %Y} that the model never saw while learning.")

    # Bar chart: each of today's rules vs. the model at the same number of warnings.
    names = {"sign": "Signs", "sign_plus_rain": "Signs + rain advisory",
             "sign3": "Signs (3 tests)", "sign3_plus_rain": "Signs (3 tests) + rain"}
    rows = []
    for key, label in names.items():
        rows.append({"rule": label, "who": "Today's rule", "caught": test[f"{key}.caught"][0] * 100})
        rows.append({"rule": label, "who": "This app's forecast", "caught": test[f"{key}.model_caught"][0] * 100})
    bars_df = pd.DataFrame(rows)
    with st.container(border=True):
        st.markdown("**Unsafe days caught, with the same number of warnings**")
        st.html(chips([(ORANGE, "Today's rule"), (BLUE, "This app's forecast")]))
        who_scale = alt.Scale(domain=["Today's rule", "This app's forecast"], range=[ORANGE, BLUE])
        base = alt.Chart(bars_df).encode(
            y=alt.Y("rule:N", sort=list(names.values()), title=None, axis=alt.Axis(labelLimit=200)),
            yOffset=alt.YOffset("who:N", sort=who_scale.domain),
        )
        bars = base.mark_bar(cornerRadiusEnd=4, height=14).encode(
            x=alt.X("caught:Q", title="% of unsafe days caught", scale=alt.Scale(domain=[0, 100])),
            color=alt.Color("who:N", scale=who_scale, legend=None),
            tooltip=[alt.Tooltip("rule:N", title="Compared with"), alt.Tooltip("who:N", title=" "),
                     alt.Tooltip("caught:Q", title="% caught", format=".0f")],
        )
        labels = base.mark_text(align="left", dx=4, color=NAVY, fontSize=12).encode(
            x="caught:Q", text=alt.Text("caught:Q", format=".0f"))
        st.altair_chart((bars + labels).properties(height=300), width="stretch")
        st.caption("\"3 tests\" = signs posted when any of the three state ocean tests fails (enterococcus, "
                   f"fecal coliform, total coliform), which is what counties post on. Against signs alone: "
                   f"{gain(test['sign.gain'])}.")

    st.markdown("#### What the forecast looks at")
    st.markdown(
        "- **The beach's track record**: how often its past tests failed. This matters most.\n"
        "- **The last test result**, counted only once it would really be public (2 days after sampling).\n"
        "- **Rain** in the day, 3 days, week and month before, and how many days since it last rained.\n"
        "- **Tides**: beaches fail about twice as often during big spring tides (Boehm & Weisberg, 2005).\n"
        "- The **season** (wet or dry) and **where** the beach is.\n\n"
        f"It learned from {fm['train_samples']:,} beach tests from 2010 to 2022 (a machine-learning method called "
        f"gradient boosting, from scikit-learn), then was graded only on {start.year} to {end.year}.")

    caution = f"{fm['threshold'] * 100:.0f}%"
    st.markdown("#### What the colors mean")
    st.html(chips([(LEVELS[k]["hex"], f"{LEVELS[k]['word']}: {t}") for k, t in
                   [("low", f"under {caution}"), ("caution", f"{caution} to 50%"), ("high", "50% or more")]]))
    st.markdown(
        "The percent is the chance that a water test would come back over the safe limit (104 enterococcus "
        f"bacteria per 100 mL). **Caution** starts at {caution}, the level that gives the same number of warnings "
        "as today's signs plus rain advisory. Setting that level using only 2022 data gives almost the same results.")

    # Say "better" only if the whole 95% range is above zero; otherwise "about as good as".
    d3 = day3["sign_plus_rain.gain"]
    d3_verdict = ("The whole range is above zero, so it does **better** than today's signs."
                  if d3[1] > 0 else
                  "The range includes zero, so the 3-day forecast is **about as good as** today's signs, not better.")
    st.markdown("#### How sure are we?")
    st.markdown(
        f"- **Ranking:** a score of {test['auc'][0]:.2f} for telling unsafe days from safe ones (1.0 is perfect, "
        "0.5 is a coin flip).\n"
        f"- **Live forecast (today):** the live app uses forecast rain, not measured rain. Estimated: "
        f"{pct(live['sign_plus_rain.model_caught'][0])} vs {pct(live['sign_plus_rain.caught'][0])} of unsafe days "
        f"caught: {gain(live['sign_plus_rain.gain'])}.\n"
        f"- **3 days ahead:** {pct(day3['sign_plus_rain.model_caught'][0])} vs "
        f"{pct(day3['sign_plus_rain.caught'][0])} for signs and rain advisory posted on the day itself: "
        f"{gain(d3)}. {d3_verdict}\n"
        "- The 95% ranges come from resampling whole beaches 1,000 times.")

    st.markdown("#### Limits")
    st.markdown(
        "- Results count days a beach was actually tested. Beaches are often retested after a bad result, so "
        "unsafe days show up more often than on the calendar.\n"
        "- The signs were rebuilt from the test data, not from records of real postings.\n"
        "- Real rain forecasts get less accurate further ahead, so later days are less certain.\n"
        "- This is a student project, not an official health advisory. Always follow posted signs.")

    st.caption("Data: California State Water Boards (Safe to Swim), Open-Meteo (rain), NOAA (tides), "
               "CIWQS (sewage spills), CAL FIRE (burn areas).")


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

st.title("🌊 Beach Water Forecast")
st.caption("Will the ocean be safe to swim at LA-area beaches? A forecast for today and the next 3 days.")

check_model_ids()
tabs = st.tabs(["Forecast", "Causes", "After the Fires", "How It Works"])
with tabs[0]:
    forecast_tab()
with tabs[1]:
    causes_tab()
with tabs[2]:
    fires_tab()
with tabs[3]:
    how_tab()

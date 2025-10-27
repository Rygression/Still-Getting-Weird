# app.py
import os
import io
import re
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

# ---------------------------
# Config
# ---------------------------
st.set_page_config(page_title="DFS Friends League Tracker", layout="wide")

MAX_WEEKS = 18
ADAPTIVE_RATIO = 0.55  # N = round(ADAPTIVE_RATIO * weeks_played_so_far)

# >>>>>>>>>>>>>>>>>>>>>
# PLACEHOLDER DATA LOCATION — replace with your real path(s)
DATA_DIR = "/Users/ryankamp/FanDuel Tracker"   # e.g., "/Users/you/Dropbox/DFSLeague"
DATA_FILE = "league_scores.csv"              # your canonical file name
# <<<<<<<<<<<<<<<<<<<<<

st.title("🏈 DFS Friends League Tracker — Cumulative vs Adaptive Top-N (55%)")

with st.expander("How this works", expanded=True):
    st.markdown(f"""
- **Data source:** The app reads from `DATA_DIR/DATA_FILE` (see constants at the top).  
- **Supported formats:**
  - **Wide** (your sheet): `Name, Username, Week 1 … Week {MAX_WEEKS}` (extra columns are ignored)
  - **Long:** `week, player, score` (+ optional `username`)
- **Modes:**
  1) **Cumulative Total** — sum of all scores through each week.
  2) **Adaptive Top-N (55%)** — at week *w*, use **N = round({ADAPTIVE_RATIO} × w)**; each player's total is the sum of their best **N** scores among their first *w* weeks.
- **Charts:** 
  - Total score over time (per selected mode)
  - Change in league rank by week (Δ vs previous week)
""")

# ---------------------------
# Sidebar Controls
# ---------------------------
with st.sidebar:
    st.header("Settings")
    mode = st.radio(
        "Scoring mode",
        options=["Cumulative Total", "Adaptive Top-N (55%)"],
        index=1  # default to your adaptive request
    )
    name_field_preference = st.selectbox("Label players by", ["Name", "Username"], index=0)
    highlight_players = st.multiselect("Highlight players (optional)", [], default=[])
    label_latest = st.toggle("Label latest points on line chart", value=True)

    st.divider()
    st.header("Filters")
    week_filter = st.multiselect("Weeks to include", list(range(1, MAX_WEEKS+1)),
                                 default=list(range(1, MAX_WEEKS+1)))

# ---------------------------
# Data Loading (fixed path)
# ---------------------------
WEEK_COL_RE = re.compile(r"^Week\s*(\d+)$", re.IGNORECASE)

def is_wide_schema(df: pd.DataFrame) -> bool:
    return ("Name" in df.columns or "Username" in df.columns) and any(WEEK_COL_RE.match(c) for c in df.columns)

def melt_wide_to_long(df_wide: pd.DataFrame) -> pd.DataFrame:
    week_cols = []
    for c in df_wide.columns:
        m = WEEK_COL_RE.match(str(c).strip())
        if m:
            w = int(m.group(1))
            if 1 <= w <= MAX_WEEKS:
                week_cols.append(c)

    id_cols = [c for c in ["Name", "Username"] if c in df_wide.columns]
    keep = id_cols + week_cols
    long_df = df_wide[keep].melt(id_vars=id_cols, value_vars=week_cols,
                                 var_name="week_col", value_name="score")
    long_df["week"] = long_df["week_col"].str.extract(WEEK_COL_RE).astype(float).astype("Int64")
    long_df.drop(columns=["week_col"], inplace=True)
    long_df["score"] = pd.to_numeric(long_df["score"], errors="coerce")

    # Preferred display label
    if "Name" in long_df.columns:
        long_df["player"] = long_df["Name"].astype(str).str.strip()
    else:
        long_df["player"] = long_df["Username"].astype(str).str.strip()

    if "Username" in long_df.columns:
        long_df["username"] = long_df["Username"].astype(str).str.strip()

    long_df = long_df[["week", "player", "score"] + (["username"] if "username" in long_df.columns else [])]
    long_df = long_df.dropna(subset=["week"]).query("1 <= week <= @MAX_WEEKS")
    return long_df

def normalize_input(df: pd.DataFrame) -> pd.DataFrame:
    cols = {c.lower().strip(): c for c in df.columns}
    if is_wide_schema(df):
        return melt_wide_to_long(df)
    if all(k in cols for k in ["week", "player", "score"]):
        out = df.rename(columns={cols["week"]: "week", cols["player"]: "player", cols["score"]: "score"})
        if "username" in cols:
            out = out.rename(columns={cols["username"]: "username"})
        out["week"] = pd.to_numeric(out["week"], errors="coerce").astype("Int64")
        out["score"] = pd.to_numeric(out["score"], errors="coerce")
        out = out.dropna(subset=["week"]).query("1 <= week <= @MAX_WEEKS")
        return out[["week", "player", "score"] + (["username"] if "username" in out.columns else [])]
    raise ValueError("CSV not recognized. Provide a wide file (Name/Username + Week 1..18) or long (week, player, score).")

@st.cache_data
def load_data_from_path(data_dir: str, data_file: str) -> pd.DataFrame:
    path = os.path.join(data_dir, data_file)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found at: {path}")
    # Let pandas sniff commas/tabs
    raw = pd.read_csv(path, sep=None, engine="python")
    return normalize_input(raw)

try:
    long_df = load_data_from_path(DATA_DIR, DATA_FILE)
except Exception as e:
    st.error(f"Problem loading data: {e}")
    st.stop()

# Apply week filter (optional)
if week_filter:
    long_df = long_df[long_df["week"].isin(week_filter)]

# If the user prefers Username for labels and we loaded wide schema originally,
# rebuild label map to swap Name -> Username on the fly.
if name_field_preference == "Username":
    # Attempt to rebuild mapping from the same source
    try:
        raw_src = pd.read_csv(os.path.join(DATA_DIR, DATA_FILE), sep=None, engine="python")
        if is_wide_schema(raw_src) and "Name" in raw_src.columns and "Username" in raw_src.columns:
            mapping = dict(zip(raw_src["Name"].astype(str).str.strip(),
                               raw_src["Username"].astype(str).str.strip()))
            long_df["player"] = long_df["player"].map(lambda p: mapping.get(p, p))
    except Exception:
        pass

# ---------------------------
# Core Computation (modes)
# ---------------------------
@st.cache_data
def compute_time_series(df: pd.DataFrame, mode_name: str, ratio: float):
    """
    Returns weekly per-player totals & ranks according to the chosen mode.
    - 'Cumulative Total': sum of all scores through week w
    - 'Adaptive Top-N (55%)': at week w, N = round(ratio * w); sum of best N scores among first w weeks
    """
    d = df.dropna(subset=["player", "week"]).copy()
    d["player"] = d["player"].astype(str)
    weeks = sorted(d["week"].unique().tolist())
    players = sorted(d["player"].unique().tolist())

    recs = []
    for w in weeks:
        upto = d[d["week"] <= w]
        for p in players:
            s = upto.loc[upto["player"] == p, "score"].dropna().values  # 0.0 is valid
            if mode_name.startswith("Cumulative"):
                total = float(np.sum(s)) if s.size else 0.0
                counted = int(np.count_nonzero(~np.isnan(s)))
            else:
                # Adaptive N = round(ratio * weeks_played_so_far)
                wp = int(np.count_nonzero(~np.isnan(s)))
                N = int(round(ratio * wp)) if wp > 0 else 0
                N = min(N, wp)
                total = float(np.sort(s)[-N:].sum()) if N > 0 else 0.0
                counted = N
            recs.append({
                "week": int(w),
                "player": p,
                "counted_weeks_so_far": counted,
                "mode_total": round(total, 3),
                "weeks_played_so_far": int(np.count_nonzero(~np.isnan(s))),
                "adaptive_N": int(round(ratio * np.count_nonzero(~np.isnan(s)))) if not mode_name.startswith("Cumulative") else None
            })

    weekly = pd.DataFrame(recs)
    if weekly.empty:
        return weekly

    # Rank each week on the total for the chosen mode
    weekly["rank"] = weekly.groupby("week")["mode_total"].rank(method="min", ascending=False).astype(int)
    weekly = weekly.sort_values(["player", "week"])
    weekly["prev_rank"] = weekly.groupby("player")["rank"].shift(1)
    weekly["rank_delta"] = weekly["prev_rank"] - weekly["rank"]
    return weekly

weekly = compute_time_series(long_df, mode, ADAPTIVE_RATIO)

# ---------------------------
# UI & Visuals
# ---------------------------
if weekly.empty:
    st.warning("No data to compute yet. Check your source file or filters.")
    st.stop()

latest_week = int(weekly["week"].max())

# Standings
standings = (
    weekly[weekly["week"] == latest_week]
    .sort_values(["mode_total", "player"], ascending=[False, True])
    .assign(place=lambda d: range(1, len(d) + 1))
    [["place", "player", "counted_weeks_so_far", "mode_total", "rank", "rank_delta"]]
    .rename(columns={"mode_total": "total_to_date"})
)

left, mid, right = st.columns([1,1,1])
with left:  st.subheader(f"📊 Standings after Week {latest_week}")
with mid:   st.metric("Mode", mode)
with right:
    if mode.startswith("Adaptive"):
        st.metric("Rule", f"N = round({ADAPTIVE_RATIO:.2f} × weeks)")

# Expose players for highlight control
all_players = sorted(weekly["player"].unique().tolist())
if not highlight_players:
    # refresh sidebar options only once data is ready
    st.sidebar.multiselect("Highlight players (optional)", options=all_players, default=[], key="hl_refresh")

# Fancy table (with progress visualization if adaptive)
try:
    col_cfg = {
        "place": st.column_config.NumberColumn("Place", format="%d", width="small"),
        "player": st.column_config.TextColumn("Player"),
        "total_to_date": st.column_config.NumberColumn("Total (to-date)", format="%.2f"),
        "rank": st.column_config.NumberColumn("Rank", format="%d", width="small"),
        "rank_delta": st.column_config.NumberColumn("Δ Rank vs Prev", width="small"),
    }
    if mode.startswith("Adaptive"):
        col_cfg["counted_weeks_so_far"] = st.column_config.NumberColumn("Counted (N)", format="%d")
    else:
        col_cfg["counted_weeks_so_far"] = st.column_config.NumberColumn("Weeks Played", format="%d")

    st.dataframe(standings, use_container_width=True, column_config=col_cfg)
except Exception:
    st.dataframe(standings, use_container_width=True)

# Downloads
cA, cB = st.columns([1,1])
with cA:
    st.download_button(
        "Download Standings CSV",
        standings.to_csv(index=False).encode("utf-8"),
        file_name=f"standings_week_{latest_week}.csv",
        mime="text/csv",
    )
with cB:
    st.download_button(
        "Download Weekly Time Series CSV",
        weekly.to_csv(index=False).encode("utf-8"),
        file_name="weekly_timeseries.csv",
        mime="text/csv",
    )

st.markdown("---")

# Line chart: Total over time (mode-dependent)
st.markdown("### 📈 Total Score Over Time")
chart_df = weekly.copy()

line = alt.Chart(chart_df).mark_line().encode(
    x=alt.X("week:O", title="Week"),
    y=alt.Y("mode_total:Q", title="Total (to-date)", scale=alt.Scale(zero=True)),
    color=alt.Color("player:N", legend=alt.Legend(title="Player", orient="bottom")),
    tooltip=[
        alt.Tooltip("player:N"),
        alt.Tooltip("week:O"),
        alt.Tooltip("mode_total:Q", title="Total", format=".2f"),
        alt.Tooltip("counted_weeks_so_far:Q", title=("Counted (N)" if mode.startswith("Adaptive") else "Weeks Played")),
        alt.Tooltip("adaptive_N:Q", title="Adaptive N", format="d", condition=alt.ConditionNamePredicate("mode", True))
    ],
)

if highlight_players:
    line = line.encode(
        opacity=alt.condition(alt.FieldOneOfPredicate(field="player", oneOf=highlight_players), alt.value(1.0), alt.value(0.25))
    )

chart = line.properties(height=420).interactive(bind_y=False)

if label_latest:
    last_pts = chart_df.sort_values("week").groupby("player").tail(1)
    text = alt.Chart(last_pts).mark_text(dx=5, dy=0, align="left").encode(
        x="week:O", y="mode_total:Q", text=alt.Text("player:N")
    )
    chart = chart + text

st.altair_chart(chart, use_container_width=True)

# Rank change chart
st.markdown("### 📉 Change in League Rank by Week (Δ vs previous week)")
rank_df = weekly.copy()
rank_df["rank_change"] = rank_df["rank_delta"].fillna(0)

rank_chart = alt.Chart(rank_df).mark_line(point=True).encode(
    x=alt.X("week:O", title="Week"),
    y=alt.Y("rank_change:Q", title="Rank Change (+ = improved)"),
    color=alt.Color("player:N", legend=alt.Legend(title="Player", orient="bottom")),
    tooltip=[
        alt.Tooltip("player:N"),
        alt.Tooltip("week:O"),
        alt.Tooltip("rank:Q", title="Rank"),
        alt.Tooltip("rank_delta:Q", title="Δ Rank vs Prev"),
    ],
)

if highlight_players:
    rank_chart = rank_chart.encode(
        opacity=alt.condition(alt.FieldOneOfPredicate(field="player", oneOf=highlight_players), alt.value(1.0), alt.value(0.25))
    )

st.altair_chart(rank_chart.properties(height=360), use_container_width=True)

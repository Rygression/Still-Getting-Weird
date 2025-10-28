# app.py
import os
import re
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

# ---------------------------
# App Config
# ---------------------------
st.set_page_config(page_title="DFS Friends League Tracker", layout="wide")
st.title("🏈 Still Getting Weird Weekly Tracker")

MAX_WEEKS = 18
ADAPTIVE_RATIO = 0.55  # for Pace: K = round(ADAPTIVE_RATIO * weeks_count)
WEEK_COL_RE = re.compile(r"^Week\s*(\d+)$", re.IGNORECASE)

# ---------------------------
# FIXED DATA PATH
# ---------------------------
DATA_PATH = "/Users/ryankamp/FanDuel_Tracker/league_scores.csv"

# ---------------------------
# Data Loading (Username-only, whitelist)
# ---------------------------
def is_wide_schema(df: pd.DataFrame) -> bool:
    return ("Username" in df.columns) and any(WEEK_COL_RE.match(str(c).strip()) for c in df.columns)

def melt_wide_to_long(df_wide: pd.DataFrame, allowed_usernames: set) -> pd.DataFrame:
    # Collect week columns
    week_cols = []
    for c in df_wide.columns:
        m = WEEK_COL_RE.match(str(c).strip())
        if m:
            w = int(m.group(1))
            if 1 <= w <= MAX_WEEKS:
                week_cols.append(c)
    if not week_cols:
        raise ValueError("No 'Week N' columns found.")

    # Keep only Username + week columns; ignore Name/Email/etc
    if "Username" not in df_wide.columns:
        raise ValueError("Expected 'Username' column in the CSV.")
    keep = ["Username"] + week_cols
    melted = df_wide[keep].melt(
        id_vars=["Username"],
        value_vars=week_cols,
        var_name="week_col",
        value_name="score"
    )

    # Extract numeric week
    melted["week"] = melted["week_col"].str.extract(WEEK_COL_RE).astype(float).astype("Int64")
    melted.drop(columns=["week_col"], inplace=True)

    # Normalize fields
    melted["player"] = melted["Username"].astype(str).str.strip()
    melted["score"] = pd.to_numeric(melted["score"], errors="coerce")

    # Strict whitelist
    melted = melted[melted["player"].isin(allowed_usernames)]
    melted = melted.dropna(subset=["week"]).query("1 <= week <= @MAX_WEEKS")

    return melted[["week", "player", "score"]]

def load_long_from_wide(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found at: {path}")
    raw = pd.read_csv(path, sep=None, engine="python")

    if not is_wide_schema(raw):
        raise ValueError("CSV is not in the expected wide format with 'Username' and 'Week N' columns.")

    # Authoritative whitelist of usernames
    allowed_usernames = set(raw["Username"].astype(str).str.strip().dropna().unique().tolist())
    return melt_wide_to_long(raw, allowed_usernames)

# Load data
try:
    long_df = load_long_from_wide(DATA_PATH)
except Exception as e:
    st.error(f"Problem loading data: {e}")
    st.stop()

# ---------------------------
# Sidebar Controls (after data load)
# ---------------------------
all_players = sorted(long_df["player"].unique().tolist())
weeks_present = sorted(long_df["week"].dropna().unique().tolist())

with st.sidebar:
    st.header("Settings")
    mode = st.radio("Scoring mode", options=["Total", "Pace"], index=1)
    highlight_players = st.multiselect("Highlight players (optional)", options=all_players, default=[])
    label_latest = st.toggle("Label latest points on line chart", value=False)  # default OFF

    st.divider()
    st.header("Filters")
    week_filter = st.multiselect(
        "Weeks to include",
        options=list(range(1, MAX_WEEKS + 1)),
        default=weeks_present or list(range(1, MAX_WEEKS + 1))
    )

# Apply filters
if week_filter:
    long_df = long_df[long_df["week"].isin(week_filter)]

# ---------------------------
# Core Computation (per-mode totals & ranks)
# ---------------------------
@st.cache_data
def compute_time_series(df: pd.DataFrame, mode_name: str, ratio: float):
    """
    Returns weekly per-player totals & ranks per chosen mode.
    - 'Total': sum of all scores through week w
    - 'Pace': at week w, N = round(ratio * weeks_played_so_far); sum best N scores among first w weeks
    """
    d = df.dropna(subset=["player", "week"]).copy()
    d["player"] = d["player"].astype(str)

    weeks = sorted(d["week"].unique().tolist())
    players = sorted(d["player"].unique().tolist())

    rows = []
    for w in weeks:
        upto = d[d["week"] <= w]
        for p in players:
            s = upto.loc[upto["player"] == p, "score"].dropna().values  # 0.0 is valid
            weeks_played = int(np.count_nonzero(~np.isnan(s)))

            if mode_name == "Total":
                total = float(np.sum(s)) if s.size else 0.0
                counted = weeks_played
                adaptive_N = None
            else:  # Pace
                N = int(round(ratio * weeks_played)) if weeks_played > 0 else 0
                N = min(N, weeks_played)
                total = float(np.sort(s)[-N:].sum()) if N > 0 else 0.0
                counted = N
                adaptive_N = N

            rows.append({
                "week": int(w),
                "player": p,
                "mode_total": round(total, 3),
                "counted_weeks_so_far": counted,
                "weeks_played_so_far": weeks_played,
                "adaptive_N": adaptive_N
            })

    weekly = pd.DataFrame(rows)
    if weekly.empty:
        return weekly

    # Week-by-week ranks
    weekly["rank"] = weekly.groupby("week")["mode_total"].rank(method="min", ascending=False).astype(int)
    weekly = weekly.sort_values(["player", "week"])
    weekly["prev_rank"] = weekly.groupby("player")["rank"].shift(1)
    weekly["rank_delta"] = weekly["prev_rank"] - weekly["rank"]  # for reference
    return weekly

weekly = compute_time_series(long_df, mode, ADAPTIVE_RATIO)
if weekly.empty:
    st.warning("No data to compute yet. Check your source file or filters.")
    st.stop()

# ---------------------------
# Weeks in view + latest/prev week
# ---------------------------
weeks_in_view = sorted(weekly["week"].unique().tolist())
weeks_count = len(weeks_in_view)
latest_week = int(weeks_in_view[-1])
prev_week = int(weeks_in_view[-2]) if len(weeks_in_view) >= 2 else None

# ---------------------------
# Winnings helpers
# ---------------------------
def compute_weekly_winnings(df_long: pd.DataFrame) -> pd.DataFrame:
    """
    Weekly $50 prize to the single highest raw score each week.
    Assumes no ties; if equal max occurs, first idxmax wins.
    """
    win_amounts = {p: 0.0 for p in df_long["player"].unique()}
    for w, sub in df_long.groupby("week"):
        sub = sub[["player", "score"]].dropna()
        if sub.empty:
            continue
        winner = sub.loc[sub["score"].idxmax(), "player"]
        win_amounts[winner] += 50.0
    return pd.DataFrame({"player": list(win_amounts.keys()),
                         "weekly_winnings": list(win_amounts.values())})

SEASON_PRIZES = {1: 450.0, 2: 270.0, 3: 180.0}

def compute_onpace_payouts(df_long: pd.DataFrame, mode_name: str, ratio: float) -> pd.DataFrame:
    """
    Season 'on-pace' winnings, no tie handling:
      - Total: K = 10
      - Pace:  K = round(0.55 * weeks_in_view)
    Ranking based on sum of best K weekly scores per player among weeks in view.
    """
    weeks_in_view = sorted(df_long["week"].unique().tolist())
    weeks_count = len(weeks_in_view)

    if mode_name == "Total":
        K = 10
    else:
        K = int(round(ratio * weeks_count))
    K = max(0, K)

    # Aggregate best-K per player
    rows = []
    for p, sub in df_long.groupby("player"):
        s = sub["score"].dropna().values
        wp = int(np.count_nonzero(~np.isnan(s)))
        k_use = min(K, wp)
        total_k = float(np.sort(s)[-k_use:].sum()) if k_use > 0 else 0.0
        rows.append({"player": p, "bestK_total": total_k})

    season_df = pd.DataFrame(rows).sort_values(["bestK_total", "player"], ascending=[False, True])
    season_df["rank_pos"] = range(1, len(season_df) + 1)
    season_df["season_winnings"] = season_df["rank_pos"].map(SEASON_PRIZES).fillna(0.0)
    return season_df[["player", "season_winnings"]]

# ---------------------------
# Winnings: compute + KPIs + stacked bar
# ---------------------------
weekly_cash = compute_weekly_winnings(long_df)
season_cash = compute_onpace_payouts(long_df, mode, ADAPTIVE_RATIO)

winnings = (
    pd.merge(weekly_cash, season_cash, on="player", how="outer")
    .fillna(0.0)
)
winnings["total_winnings"] = winnings["weekly_winnings"] + winnings["season_winnings"]

# KPI 1: Highest single-week score + who/when
if not long_df.dropna(subset=["score"]).empty:
    idx = long_df["score"].idxmax()
    top_score = float(long_df.loc[idx, "score"])
    top_player = str(long_df.loc[idx, "player"])
    top_week = int(long_df.loc[idx, "week"])
else:
    top_score, top_player, top_week = 0.0, "-", 0

# KPI 2: Number of players who have broken even on weekly prizes ($100 buy-in)
breakeven_count = int((winnings["weekly_winnings"] >= 100.0).sum())

k1, k2 = st.columns([1, 1])
with k1:
    st.metric(label="Highest Single-Week Score", value=f"{top_score:.2f}", help=f"{top_player} in Week {top_week}")
with k2:
    st.metric(label="Players To Break Even", value=breakeven_count)

# Stacked bar chart (Weekly = green, Season = blue)
st.markdown("### 💵 Cash Winnings (to date) — Weekly (green) + Season On-Pace (blue)")

bar_df = (
    winnings.melt(id_vars=["player"], value_vars=["weekly_winnings", "season_winnings"],
                  var_name="component", value_name="amount")
    .replace({"weekly_winnings": "Weekly", "season_winnings": "Season"})
)

bar = (
    alt.Chart(bar_df)
    .mark_bar()
    .encode(
        x=alt.X("player:N", title="Player", sort=sorted(winnings["player"].tolist())),
        y=alt.Y("amount:Q", title="Amount ($)", stack="zero"),
        color=alt.Color(
            "component:N",
            scale=alt.Scale(domain=["Weekly", "Season"], range=["#2ecc71", "#3498db"]),
            legend=alt.Legend(orient="bottom", direction="horizontal", columns=2, title=None),
        ),
        tooltip=[
            alt.Tooltip("player:N"),
            alt.Tooltip("component:N"),
            alt.Tooltip("amount:Q", title="Amount ($)", format="$.2f"),
        ],
    )
    .properties(height=360)
)

st.altair_chart(bar.interactive(), use_container_width=True)

st.markdown("---")

# ---------------------------
# Compute explicit Δ from latest vs previous week (for standings)
# ---------------------------
latest_ranks = weekly.loc[weekly["week"] == latest_week, ["player", "rank"]].rename(columns={"rank": "curr_rank"})
if prev_week is not None:
    prev_ranks = weekly.loc[weekly["week"] == prev_week, ["player", "rank"]].rename(columns={"rank": "prev_rank"})
else:
    prev_ranks = pd.DataFrame({"player": latest_ranks["player"], "prev_rank": np.nan})

rank_merge = latest_ranks.merge(prev_ranks, on="player", how="left")
rank_merge["delta"] = rank_merge["prev_rank"] - rank_merge["curr_rank"]  # positive = improved

# ---------------------------
# Standings + Labels ABOVE the standings table (full view, no index)
# ---------------------------
standings_base = (
    weekly[weekly["week"] == latest_week]
    .sort_values(["mode_total", "player"], ascending=[False, True])
    .assign(place=lambda d: range(1, len(d) + 1))
    [["place", "player", "mode_total"]]
)

standings_joined = standings_base.merge(rank_merge[["player", "delta"]], on="player", how="left")

hdr1, hdr2 = st.columns([1, 1])
with hdr1:
    st.metric("Mode", mode)
with hdr2:
    weeks_value = weeks_count if mode == "Total" else int(round(ADAPTIVE_RATIO * weeks_count))
    st.metric("Weeks", weeks_value)

st.subheader(f"📊 Standings after Week {latest_week}")

standings_disp = (
    standings_joined
    .rename(columns={
        "place": "Rank",
        "player": "Player",
        "mode_total": "Total (to-date)",
        "delta": "Δ Rank vs Previous"
    })
    [["Rank", "Player", "Total (to-date)", "Δ Rank vs Previous"]]
)

def _style_delta(col: pd.Series):
    max_abs = np.nanmax(np.abs(col.values)) if len(col) else 0
    styles = []
    for v in col.values:
        if pd.isna(v) or max_abs == 0:
            styles.append("")
            continue
        alpha = 0.15 + 0.75 * (min(abs(v) / max_abs, 1.0))
        if v > 0:
            styles.append(f"background-color: rgba(0, 170, 0, {alpha}); color: #000;")
        elif v < 0:
            styles.append(f"background-color: rgba(220, 0, 0, {alpha}); color: #000;")
        else:
            styles.append("background-color: rgba(0,0,0,0.04); color: #000;")
    return styles

def _fmt_delta(v):
    return "" if pd.isna(v) else f"{int(v):+d}"

row_height = 34
header_pad = 64
table_height = header_pad + max(10, len(standings_disp)) * row_height

styled = (
    standings_disp.style
    .format({"Total (to-date)": "{:.2f}", "Δ Rank vs Previous": _fmt_delta})
    .apply(_style_delta, subset=["Δ Rank vs Previous"])
)

st.dataframe(styled, use_container_width=True, hide_index=True, height=min(table_height, 1400))

st.markdown("---")

# ---------------------------
# Chart: Total Score Over Time (with wrapped legend)
# ---------------------------
st.markdown("### 📈 Total Score Over Time")

chart_df = weekly.copy()

line = (
    alt.Chart(chart_df)
    .mark_line(point=True)
    .encode(
        x=alt.X("week:O", title="Week"),
        y=alt.Y("mode_total:Q", title="Total (to-date)", scale=alt.Scale(zero=True)),
        color=alt.Color(
            "player:N",
            legend=alt.Legend(
                title="Player",
                orient="bottom",
                direction="horizontal",
                columns=5,
                labelFontSize=11,
                titleFontSize=12,
                symbolSize=100,
            ),
        ),
        tooltip=[
            alt.Tooltip("player:N"),
            alt.Tooltip("week:O"),
            alt.Tooltip("mode_total:Q", title="Total", format=".2f"),
            alt.Tooltip(
                "counted_weeks_so_far:Q",
                title=("Counted (N)" if mode == "Pace" else "Weeks Played"),
            ),
            alt.Tooltip("adaptive_N:Q", title="Adaptive N", format="d"),
        ],
    )
    .properties(height=420)
)

if highlight_players:
    line = line.encode(
        opacity=alt.condition(
            alt.FieldOneOfPredicate(field="player", oneOf=highlight_players),
            alt.value(1.0),
            alt.value(0.25),
        )
    )

if label_latest:
    last_pts = chart_df.sort_values("week").groupby("player").tail(1)
    text = (
        alt.Chart(last_pts)
        .mark_text(dx=5, dy=0, align="left")
        .encode(x="week:O", y="mode_total:Q", text=alt.Text("player:N"))
    )
    line = line + text

st.altair_chart(line.interactive(bind_y=False), use_container_width=True)

# ---------------------------
# Chart: Rank by Week (1 = top) with wrapped legend, smooth line
# ---------------------------
st.markdown("### 📉 League Rank by Week (1 = best)")

rank_df = weekly.copy()

rank_chart = (
    alt.Chart(rank_df)
    .mark_line(point=True, interpolate="monotone")  # smooth line + points
    .encode(
        x=alt.X("week:O", title="Week"),
        y=alt.Y(
            "rank:Q",
            title="Rank",
            scale=alt.Scale(domain=[len(rank_df["player"].unique()), 1]),  # invert y-axis
        ),
        color=alt.Color(
            "player:N",
            legend=alt.Legend(
                title="Player",
                orient="bottom",
                direction="horizontal",
                columns=5,
                labelFontSize=11,
                titleFontSize=12,
                symbolSize=100,
            ),
        ),
        tooltip=[
            alt.Tooltip("player:N"),
            alt.Tooltip("week:O"),
            alt.Tooltip("rank:Q", title="Rank"),
        ],
    )
    .properties(height=360)
)

if highlight_players:
    rank_chart = rank_chart.encode(
        opacity=alt.condition(
            alt.FieldOneOfPredicate(field="player", oneOf=highlight_players),
            alt.value(1.0),
            alt.value(0.25),
        )
    )

st.altair_chart(rank_chart.interactive(bind_y=False), use_container_width=True)

# Fanduel_app.py

import re
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

# ---------------------------
# App Config
# ---------------------------
st.set_page_config(
    page_title="DFS Friends League Tracker",
    layout="wide"
)

st.title("🏈 Still Getting Weird Weekly Tracker")

MAX_WEEKS = 18
PACE_RATIO = 0.55
WEEK_COL_RE = re.compile(r"^Week\s*(\d+)$", re.IGNORECASE)

GITHUB_RAW_CSV = (
    "https://raw.githubusercontent.com/Rygression/"
    "Still-Getting-Weird/refs/heads/main/league_scores.csv"
)

# ---------------------------
# Data Loading
# ---------------------------
@st.cache_data
def load_raw_csv():
    return pd.read_csv(GITHUB_RAW_CSV)

raw = load_raw_csv()

# ---------------------------
# Transform wide → long
# ---------------------------
week_cols = [c for c in raw.columns if WEEK_COL_RE.match(str(c))]
if not week_cols:
    st.error("No Week columns found in CSV.")
    st.stop()

long_df = (
    raw
    .melt(
        id_vars=["Username"],
        value_vars=week_cols,
        var_name="week_col",
        value_name="score"
    )
)

long_df["week"] = (
    long_df["week_col"]
    .str.extract(WEEK_COL_RE)
    .astype(int)
)

long_df["player"] = long_df["Username"].astype(str).str.strip()
long_df["score"] = pd.to_numeric(long_df["score"], errors="coerce")

long_df = long_df.dropna(subset=["player", "week", "score"])
long_df = long_df.query("1 <= week <= @MAX_WEEKS")

# ---------------------------
# Sidebar Controls
# ---------------------------
all_players = sorted(long_df["player"].unique())

with st.sidebar:
    st.header("Settings")
    mode = st.radio("Scoring Mode", ["Total", "Pace"], index=1)

    highlight_players = st.multiselect(
        "Highlight players",
        options=all_players
    )

    label_latest = st.toggle(
        "Label latest points on line chart",
        value=False
    )

# ---------------------------
# Core Computation
# ---------------------------
@st.cache_data
def compute_weekly_totals(df, mode):
    rows = []

    for week in sorted(df["week"].unique()):
        up_to = df[df["week"] <= week]

        for player, sub in up_to.groupby("player"):
            scores = sub["score"].values
            weeks_played = len(scores)

            if mode == "Total":
                K = 10
            else:
                K = int(round(PACE_RATIO * weeks_played))

            K = min(K, weeks_played)
            total = np.sort(scores)[-K:].sum() if K > 0 else 0.0

            rows.append({
                "week": week,
                "player": player,
                "total": total,
                "weeks_used": K
            })

    out = pd.DataFrame(rows)

    out["rank"] = (
        out.groupby("week")["total"]
        .rank(ascending=False, method="min")
        .astype(int)
    )

    out = out.sort_values(["player", "week"])
    out["prev_rank"] = out.groupby("player")["rank"].shift(1)
    out["rank_delta"] = out["prev_rank"] - out["rank"]

    return out

weekly = compute_weekly_totals(long_df, mode)

latest_week = weekly["week"].max()

# ---------------------------
# KPIs
# ---------------------------
idx = long_df["score"].idxmax()
top_score = long_df.loc[idx, "score"]
top_player = long_df.loc[idx, "player"]
top_week = long_df.loc[idx, "week"]

weekly_wins = (
    long_df
    .loc[long_df.groupby("week")["score"].idxmax()]
    .groupby("player")
    .size()
    .mul(50)
)

breakeven_count = int((weekly_wins >= 100).sum())

k1, k2 = st.columns(2)
k1.metric(
    "Highest Single-Week Score",
    f"{top_score:.2f}",
    help=f"{top_player} — Week {top_week}"
)
k2.metric(
    "Players Who Have Broken Even",
    breakeven_count
)

# ---------------------------
# Line Chart: Total / Pace
# ---------------------------
st.markdown("### 📈 Player Totals Over Time")

line = (
    alt.Chart(weekly)
    .mark_line(point=True)
    .encode(
        x=alt.X("week:O", title="Week"),
        y=alt.Y("total:Q", title="Total"),
        color=alt.Color(
            "player:N",
            legend=alt.Legend(
                orient="bottom",
                direction="horizontal",
                columns=5
            )
        ),
        tooltip=[
            "player",
            "week",
            alt.Tooltip("total:Q", format=".2f"),
            alt.Tooltip("weeks_used:Q", title="Weeks Counted")
        ]
    )
    .properties(height=420)
)

if highlight_players:
    line = line.encode(
        opacity=alt.condition(
            alt.FieldOneOfPredicate("player", highlight_players),
            alt.value(1),
            alt.val

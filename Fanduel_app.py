import streamlit as st
import pandas as pd
import altair as alt
import math
import re

# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(
    page_title="FanDuel League Tracker",
    layout="wide"
)

# -----------------------------
# Load data (WIDE -> LONG)
# -----------------------------
DATA_URL = "https://raw.githubusercontent.com/Rygression/Still-Getting-Weird/refs/heads/main/league_scores.csv"

@st.cache_data
def load_data():
    wide = pd.read_csv(DATA_URL)
    wide.columns = wide.columns.str.strip()

    # Rename fixed columns
    wide = wide.rename(columns={
        "Name": "player",
        "Username": "username"
    })

    # Identify week columns
    week_cols = [c for c in wide.columns if re.match(r"Week \d+", c)]

    # Melt to long format
    df = wide.melt(
        id_vars=["player", "username"],
        value_vars=week_cols,
        var_name="week",
        value_name="score"
    )

    # Clean week + score
    df["week"] = df["week"].str.replace("Week ", "", regex=False).astype(int)
    df["score"] = pd.to_numeric(df["score"], errors="coerce")


    return df

df = load_data()

weeks_played = df["week"].nunique()

# -----------------------------
# Mode selector
# -----------------------------
mode = st.radio("Scoring Mode", ["Total", "Pace"], horizontal=True)

if mode == "Total":
    K = 10
else:
    K = max(1, round(weeks_played * 0.55))

# -----------------------------
# Core scoring logic
# -----------------------------
def calc_total(x, k):
    return x["score"].sort_values(ascending=False).head(k).sum()

totals = (
    df.groupby("player", as_index=False)
      .apply(lambda x: calc_total(x, K))
      .reset_index(drop=True)
      .rename(columns={0: "total_score"})
)

totals["Rank"] = totals["total_score"].rank(
    ascending=False,
    method="first"
).astype(int)

totals = totals.sort_values("Rank")

# -----------------------------
# KPI callouts
# -----------------------------
col1, col2 = st.columns(2)

best_week = df.loc[df["score"].idxmax()]
col1.metric(
    "Highest Single Week Score",
    f"{best_week['score']:.2f}",
    best_week["player"]
)

weekly_winners = (
    df.loc[df.groupby("week")["score"].idxmax()]
      .groupby("player")
      .size()
      .mul(50)
)

break_even = (weekly_winners >= 100).sum()
col2.metric("Players Broken Even", break_even)

# -----------------------------
# Cash winnings (stacked bars)
# -----------------------------
season_payouts = {1: 450, 2: 270, 3: 180}

cash = totals.copy()
cash["weekly_cash"] = cash["player"].map(weekly_winners).fillna(0)
cash["season_cash"] = cash["Rank"].map(season_payouts).fillna(0)

cash_long = cash.melt(
    id_vars="player",
    value_vars=["weekly_cash", "season_cash"],
    var_name="type",
    value_name="amount"
)

cash_long["type"] = cash_long["type"].map({
    "weekly_cash": "Weekly Winnings",
    "season_cash": "On-Pace Season Winnings"
})

st.subheader("Cash Winnings (On-Pace)")

bar_chart = (
    alt.Chart(cash_long)
    .mark_bar()
    .encode(
        x=alt.X("player:N", sort="-y", title="Player"),
        y=alt.Y("amount:Q", title="Dollars"),
        color=alt.Color("type:N", legend=alt.Legend(orient="bottom"))
    )
    .properties(height=400)
)

st.altair_chart(bar_chart, use_container_width=True)

# -----------------------------
# Standings table
# -----------------------------
st.markdown(f"### Standings — **{mode}** ({K} weeks)")

table = totals[["Rank", "player", "total_score"]] \
    .rename(columns={
        "player": "Player",
        "total_score": "Total Score"
    })

st.dataframe(
    table.style.format({"Total Score": "{:.2f}"}),
    use_container_width=True,
    hide_index=True
)

# -----------------------------
# Rank by week chart
# -----------------------------
weekly_rank = (
    df.groupby(["week", "player"])["score"]
      .sum()
      .reset_index()
)

weekly_rank["rank"] = weekly_rank.groupby("week")["score"] \
    .rank(ascending=False, method="first")

rank_chart = (
    alt.Chart(weekly_rank)
    .mark_line(interpolate="monotone", point=True)
    .encode(
        x=alt.X("week:O", title="Week"),
        y=alt.Y("rank:Q", scale=alt.Scale(reverse=True), title="Rank"),
        color=alt.Color("player:N", legend=alt.Legend(orient="bottom"))
    )
    .properties(height=400)
)

st.subheader("Rank by Week")
st.altair_chart(rank_chart, use_container_width=True)

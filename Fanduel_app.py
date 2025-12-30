import streamlit as st
import pandas as pd
import altair as alt
import re
import math

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="FanDuel League Tracker",
    layout="wide"
)

# -----------------------------
# Data loading
# -----------------------------
DATA_URL = "https://raw.githubusercontent.com/Rygression/Still-Getting-Weird/refs/heads/main/league_scores.csv"

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL)
    df.columns = df.columns.str.strip()

    df = df.rename(columns={
        "Name": "player",
        "Username": "username"
    })

    week_cols = [c for c in df.columns if re.match(r"Week \d+", c)]

    df_long = df.melt(
        id_vars=["player", "username"],
        value_vars=week_cols,
        var_name="week",
        value_name="score"
    )

    df_long["week"] = (
        df_long["week"]
        .str.replace("Week ", "", regex=False)
        .astype(int)
    )

    df_long["score"] = pd.to_numeric(df_long["score"], errors="coerce")
    df_long = df_long.dropna(subset=["score"])
    df_long = df_long[df_long["score"] > 0]

    return df_long.sort_values(["week", "player"])

df = load_data()

weeks_played = df["week"].nunique()

# -----------------------------
# Controls
# -----------------------------
mode = st.radio(
    "Mode",
    ["Total", "Pace"],
    horizontal=True
)

if mode == "Total":
    K = 10
    weeks_label = "10"
else:
    K = max(1, round(weeks_played * 0.55))
    weeks_label = str(K)

# -----------------------------
# Core calculations
# -----------------------------
def calc_player_total(player_df, k):
    scores = player_df["score"].sort_values(ascending=False)
    return scores.head(k).sum()

totals = (
    df.groupby("player", as_index=False)
      .apply(lambda x: calc_player_total(x, K))
      .reset_index(drop=True)
      .rename(columns={0: "total_score"})
)

totals["Rank"] = totals["total_score"].rank(
    ascending=False,
    method="first"
).astype(int)

totals = totals.sort_values("Rank")

# -----------------------------
# Previous week rank (for delta)
# -----------------------------
latest_week = df["week"].max()
prev_week = latest_week - 1

latest_df = df[df["week"] <= latest_week]
prev_df = df[df["week"] <= prev_week]

latest_totals = (
    latest_df.groupby("player", as_index=False)
    .apply(lambda x: calc_player_total(x, K))
    .reset_index(drop=True)
    .rename(columns={0: "score"})
)

prev_totals = (
    prev_df.groupby("player", as_index=False)
    .apply(lambda x: calc_player_total(x, K))
    .reset_index(drop=True)
    .rename(columns={0: "score"})
)

latest_totals["rank_now"] = latest_totals["score"].rank(
    ascending=False,
    method="first"
)

prev_totals["rank_prev"] = prev_totals["score"].rank(
    ascending=False,
    method="first"
)

rank_delta = latest_totals.merge(
    prev_totals[["player", "rank_prev"]],
    on="player",
    how="left"
)

rank_delta["∆ Rank vs Previous"] = (
    rank_delta["rank_prev"] - rank_delta["rank_now"]
)

totals = totals.merge(
    rank_delta[["player", "∆ Rank vs Previous"]],
    on="player",
    how="left"
)

totals["∆ Rank vs Previous"] = totals["∆ Rank vs Previous"].fillna(0).astype(int)

# -----------------------------
# KPI callouts
# -----------------------------
col1, col2 = st.columns(2)

max_week = df.loc[df["score"].idxmax()]
col1.metric(
    "Highest Single-Week Score",
    f"{max_week['score']:.2f}",
    max_week["player"]
)

weekly_wins = (
    df.loc[df.groupby("week")["score"].idxmax()]
      .groupby("player")
      .size()
      .mul(50)
)

break_even_count = (weekly_wins >= 100).sum()
col2.metric("Players Broken Even", break_even_count)

# -----------------------------
# Cash winnings (for chart)
# -----------------------------
weekly_cash = weekly_wins.reset_index()
weekly_cash.columns = ["player", "weekly_cash"]

season_payouts = {1: 450, 2: 270, 3: 180}

totals["season_cash"] = totals["Rank"].map(season_payouts).fillna(0)

cash = totals.merge(weekly_cash, on="player", how="left").fillna(0)

cash_long = cash.melt(
    id_vars=["player"],
    value_vars=["weekly_cash", "season_cash"],
    var_name="type",
    value_name="amount"
)

cash_long["type"] = cash_long["type"].map({
    "weekly_cash": "Weekly Winnings",
    "season_cash": "On-Pace Season Winnings"
})

# -----------------------------
# Stacked bar chart
# -----------------------------
st.subheader("Cash Winnings (On-Pace)")

cash_chart = (
    alt.Chart(cash_long)
    .mark_bar()
    .encode(
        x=alt.X("player:N", sort="-y", title="Player"),
        y=alt.Y("amount:Q", title="Dollars"),
        color=alt.Color(
            "type:N",
            legend=alt.Legend(orient="bottom")
        )
    )
    .properties(height=400)
)

st.altair_chart(cash_chart, use_container_width=True)

# -----------------------------
# Standings table
# -----------------------------
st.markdown(f"### Standings — **Mode:** {mode} | **Weeks:** {weeks_label}")

styled = totals[[
    "Rank",
    "player",
    "total_score",
    "∆ Rank vs Previous"
]].rename(columns={
    "player": "Player",
    "total_score": "Total"
})

styled = styled.style.format({"Total": "{:.2f}"}) \
    .background_gradient(
        subset=["∆ Rank vs Previous"],
        cmap="RdYlGn"
    )

st.dataframe(
    styled,
    use_container_width=True,
    hide_index=True
)

# -----------------------------
# Rank by week chart
# -----------------------------
rank_by_week = (
    df.groupby(["week", "player"])["score"]
      .sum()
      .reset_index()
)

rank_by_week["rank"] = rank_by_week.groupby("week")["score"] \
    .rank(ascending=False, method="first")

rank_chart = (
    alt.Chart(rank_by_week)
    .mark_line(interpolate="monotone", point=True)
    .encode(
        x=alt.X("week:O", title="Week"),
        y=alt.Y(
            "rank:Q",
            scale=alt.Scale(reverse=True),
            title="Rank"
        ),
        color=alt.Color(
            "player:N",
            legend=alt.Legend(orient="bottom")
        )
    )
    .properties(height=400)
)

st.subheader("Rank by Week")
st.altair_chart(rank_chart, use_container_width=True)

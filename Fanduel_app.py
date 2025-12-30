import streamlit as st
import pandas as pd
import altair as alt
from io import StringIO
import requests

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="FanDuel Friends League",
    layout="wide"
)

# -----------------------------
# Data loading (GitHub CSV)
# -----------------------------
@st.cache_data
def load_data():
    github_raw_url = "https://raw.githubusercontent.com/Rygression/Still-Getting-Weird/refs/heads/main/league_scores.csv"
    

    response = requests.get(github_raw_url)
    response.raise_for_status()

    df = pd.read_csv(StringIO(response.text))
    return df


df_raw = load_data()

# -----------------------------
# Clean + standardize
# -----------------------------
df = (
    df_raw
    .rename(columns={
        "Player": "player",
        "Week": "week",
        "Score": "score"
    })
)

df = df.dropna(subset=["player", "week", "score"])
df["week"] = df["week"].astype(int)
df["score"] = df["score"].astype(float)

players = sorted(df["player"].unique())
weeks_played = df["week"].nunique()

# -----------------------------
# Mode toggle
# -----------------------------
mode = st.radio(
    "Mode",
    ["Total", "Pace"],
    horizontal=True
)

# -----------------------------
# Core computation
# -----------------------------
def compute_totals(scores_df, mode):
    if mode == "Total":
        K = 10
    else:
        K = int(round(weeks_played * 0.55))

    def top_k_sum(group):
        return (
            group.sort_values("score", ascending=False)
                 .head(K)["score"]
                 .sum()
        )

    totals = (
        scores_df
        .groupby("player", group_keys=False)
        .apply(top_k_sum)
        .reset_index(name="total_score")
    )

    return totals, K


player_totals, K = compute_totals(df, mode)

# -----------------------------
# Weekly winner winnings
# -----------------------------
weekly_winners = (
    df.loc[df.groupby("week")["score"].idxmax()]
    .groupby("player")
    .size()
    .reset_index(name="weeks_won")
)

weekly_winners["weekly_winnings"] = weekly_winners["weeks_won"] * 50

# -----------------------------
# Standings
# -----------------------------
standings = (
    player_totals
    .merge(weekly_winners, on="player", how="left")
    .fillna({"weekly_winnings": 0, "weeks_won": 0})
    .sort_values("total_score", ascending=False)
    .reset_index(drop=True)
)

standings["Rank"] = standings.index + 1

# Previous week ranks
def previous_week_ranks(scores_df, mode):
    prev_weeks = sorted(scores_df["week"].unique())[:-1]
    if not prev_weeks:
        return pd.DataFrame(columns=["player", "prev_rank"])

    prev_df = scores_df[scores_df["week"].isin(prev_weeks)]
    prev_totals, _ = compute_totals(prev_df, mode)

    prev_totals = (
        prev_totals
        .sort_values("total_score", ascending=False)
        .reset_index(drop=True)
    )
    prev_totals["prev_rank"] = prev_totals.index + 1

    return prev_totals[["player", "prev_rank"]]


prev_ranks = previous_week_ranks(df, mode)

standings = standings.merge(prev_ranks, on="player", how="left")

standings["∆ Rank vs Previous"] = (
    standings["prev_rank"] - standings["Rank"]
)

standings = standings.drop(columns=["prev_rank"])

# -----------------------------
# KPIs
# -----------------------------
max_score_row = df.loc[df["score"].idxmax()]
highest_score = int(max_score_row["score"])
highest_score_player = max_score_row["player"]

break_even_players = standings[
    standings["weekly_winnings"] >= 100
].shape[0]

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "Highest Single-Week Score",
        f"{highest_score}",
        highest_score_player
    )

with col2:
    st.metric(
        "Players Who Broke Even",
        break_even_players
    )

# -----------------------------
# Stacked winnings chart
# -----------------------------
payouts = {1: 450, 2: 270, 3: 180}

standings["season_payout"] = standings["Rank"].map(payouts).fillna(0)

winnings_chart_df = standings.melt(
    id_vars=["player"],
    value_vars=["weekly_winnings", "season_payout"],
    var_name="type",
    value_name="amount"
)

winnings_chart = (
    alt.Chart(winnings_chart_df)
    .mark_bar()
    .encode(
        x=alt.X("player:N", sort="-y", title=None),
        y=alt.Y("amount:Q", title="Winnings ($)"),
        color=alt.Color(
            "type:N",
            scale=alt.Scale(
                domain=["weekly_winnings", "season_payout"],
                range=["#2ecc71", "#3498db"]
            ),
            title=None
        ),
        tooltip=["player", "type", "amount"]
    )
    .properties(height=300)
)

st.altair_chart(winnings_chart, use_container_width=True)

# -----------------------------
# Labels above standings
# -----------------------------
label_col1, label_col2 = st.columns(2)

with label_col1:
    st.markdown(f"**Mode:** {mode}")

with label_col2:
    st.markdown(f"**Weeks:** {K}")

# -----------------------------
# Standings table
# -----------------------------
styled = standings[[
    "Rank",
    "player",
    "total_score",
    "∆ Rank vs Previous",
    "weekly_winnings"
]].rename(columns={
    "player": "Player",
    "total_score": "Total",
    "weekly_winnings": "Weekly Winnings ($)"
})

styled = styled.style.background_gradient(
    subset=["∆ Rank vs Previous"],
    cmap="RdYlGn"
)

st.dataframe(
    styled,
    hide_index=True,
    use_container_width=True
)

# -----------------------------
# Line chart: totals over time
# -----------------------------
def cumulative_scores(scores_df, mode):
    weeks = sorted(scores_df["week"].unique())
    rows = []

    for w in weeks:
        sub_df = scores_df[scores_df["week"] <= w]
        totals, _ = compute_totals(sub_df, mode)
        totals["week"] = w
        rows.append(totals)

    return pd.concat(rows)


cumulative_df = cumulative_scores(df, mode)

line_chart = (
    alt.Chart(cumulative_df)
    .mark_line(point=True, interpolate="monotone")
    .encode(
        x=alt.X("week:O", title="Week"),
        y=alt.Y("total_score:Q", title="Total"),
        color=alt.Color(
            "player:N",
            legend=alt.Legend(
                orient="bottom",
                columns=4
            )
        ),
        tooltip=["player", "week", "total_score"]
    )
    .properties(height=400)
)

st.altair_chart(line_chart, use_container_width=True)

# -----------------------------
# Rank by week chart
# -----------------------------
rank_rows = []

for w in sorted(df["week"].unique()):
    sub_df = df[df["week"] <= w]
    totals, _ = compute_totals(sub_df, mode)
    totals = totals.sort_values("total_score", ascending=False)
    totals["rank"] = range(1, len(totals) + 1)
    totals["week"] = w
    rank_rows.append(totals)

rank_df = pd.concat(rank_rows)

rank_chart = (
    alt.Chart(rank_df)
    .mark_line(point=True, interpolate="monotone")
    .encode(
        x=alt.X("week:O", title="Week"),
        y=alt.Y(
            "rank:Q",
            scale=alt.Scale(reverse=True),
            title="Rank"
        ),
        color=alt.Color(
            "player:N",
            legend=alt.Legend(
                orient="bottom",
                columns=4
            )
        ),
        tooltip=["player", "week", "rank"]
    )
    .properties(height=400)
)

st.altair_chart(rank_chart, use_container_width=True)

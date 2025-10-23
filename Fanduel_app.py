# app.py
import io
import re
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

# ---------------------------
# App Config
# ---------------------------
st.set_page_config(page_title="DFS Friends League Tracker", layout="wide")

MAX_WEEKS = 18
COUNTED_WEEKS = 10

st.title("🏈 DFS Friends League Tracker (FanDuel) — Top 10 of 18")

with st.expander("How this works", expanded=True):
    st.markdown(f"""
**League rule:** Only the **best {COUNTED_WEEKS}** weekly scores (out of {MAX_WEEKS}) count toward the season total.

**Upload Options**
1) **Your sheet (wide):** `Name, Username, Week 1, Week 2, ... Week 18, Last Rank, New Rank, Change, Average, Top 4`  
2) **Long format:** `week, player, score` (+ optional `username`)

**What we compute each week _w_:**
- `topk_total` = sum of each player's best **min(w,{COUNTED_WEEKS})** scores so far
- `on_pace_total` (optional) = scales current counted average up to {COUNTED_WEEKS} if you have fewer than {COUNTED_WEEKS} valid weeks
- `rank` and **Δ rank** vs. previous week

**Charts:**  
- Total score over time (Top-10 rule applied)  
- Change in league rank by week (Δ vs previous week)

Blank cells are ignored; **0** is treated as a valid score (did play, scored zero).
""")

# ---------------------------
# Sidebar
# ---------------------------
with st.sidebar:
    st.header("Data")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    st.caption("Your wide sheet or a long `week,player,score` CSV.")

    st.divider()
    st.header("Display")
    use_projection = st.toggle("Show 'On-pace' projection (<10 weeks)", value=True)
    name_field = st.selectbox("Use this for player labels", options=["Name", "Username"], index=0)
    highlight_latest_labels = st.toggle("Label latest points on charts", value=True)

    st.divider()
    st.header("Filters")
    selected_weeks = st.multiselect("Weeks to include", list(range(1, MAX_WEEKS+1)), default=list(range(1, MAX_WEEKS+1)))

# ---------------------------
# Helpers
# ---------------------------
WEEK_COL_RE = re.compile(r"^Week\s*(\d+)$", re.IGNORECASE)

def is_wide_schema(df: pd.DataFrame) -> bool:
    return ("Name" in df.columns or "Username" in df.columns) and any(WEEK_COL_RE.match(c) for c in df.columns)

def melt_wide_to_long(df_wide: pd.DataFrame) -> pd.DataFrame:
    # Find week columns
    week_cols = []
    for c in df_wide.columns:
        m = WEEK_COL_RE.match(c.strip())
        if m:
            week_num = int(m.group(1))
            if 1 <= week_num <= MAX_WEEKS:
                week_cols.append(c)

    if not week_cols:
        raise ValueError("Could not find any 'Week N' columns (e.g., 'Week 1').")

    id_cols = [c for c in ["Name", "Username"] if c in df_wide.columns]
    wide = df_wide.copy()

    # Keep only relevant columns (drop things like Last Rank, New Rank, Change, Average, Top 4, etc.)
    keep_cols = id_cols + week_cols
    missing_ids = [c for c in ["Name", "Username"] if c not in keep_cols]
    long_df = wide[keep_cols].melt(id_vars=id_cols, value_vars=week_cols, var_name="week_col", value_name="score")

    # Extract numeric week from "Week N"
    long_df["week"] = (
        long_df["week_col"].str.extract(WEEK_COL_RE).astype(float).astype("Int64")
    )
    long_df.drop(columns=["week_col"], inplace=True)

    # Coerce score; blank -> NaN, numeric stays numeric; "0" stays 0
    long_df["score"] = pd.to_numeric(long_df["score"], errors="coerce")

    # Build a player label that is stable & human-friendly
    # Prefer Name if present; otherwise Username
    if "Name" in long_df.columns:
        long_df["player"] = long_df["Name"].astype(str).str.strip()
    else:
        long_df["player"] = long_df["Username"].astype(str).str.strip()

    # Keep username if present
    if "Username" in long_df.columns:
        long_df["username"] = long_df["Username"].astype(str).str.strip()

    # Order & clean
    long_df = long_df[["week", "player", "score"] + (["username"] if "username" in long_df.columns else [])]
    long_df = long_df.dropna(subset=["week"]).query("week >= 1 and week <= @MAX_WEEKS")

    return long_df

def normalize_input(df: pd.DataFrame) -> pd.DataFrame:
    """
    Accept either:
      - long: week, player, score (+ optional username)
      - wide: Name/Username + Week 1..18
    Return long: [week, player, score, username?]
    """
    cols = {c.lower().strip(): c for c in df.columns}
    if is_wide_schema(df):
        return melt_wide_to_long(df)
    elif all(k in cols for k in ["week", "player", "score"]):
        out = df.rename(columns={cols["week"]: "week", cols["player"]: "player", cols["score"]: "score"})
        if "username" in cols:
            out = out.rename(columns={cols["username"]: "username"})
        out["week"] = pd.to_numeric(out["week"], errors="coerce").astype("Int64")
        out["score"] = pd.to_numeric(out["score"], errors="coerce")
        out = out.dropna(subset=["week"]).query("1 <= week <= @MAX_WEEKS")
        return out[["week", "player", "score"] + (["username"] if "username" in out.columns else [])]
    else:
        raise ValueError("CSV not recognized. Provide your wide sheet or a long CSV with columns: week, player, score.")

@st.cache_data
def compute_time_series(input_df: pd.DataFrame, counted_weeks: int, use_projection_flag: bool):
    """
    For each week w in data:
      - Sum of top min(w, counted_weeks) scores to date per player
      - Optional on-pace projection to counted_weeks if counted < counted_weeks
      - Rank per week & Δ rank vs previous week
    """
    df = input_df.copy()
    df = df.dropna(subset=["player", "week"])
    df["player"] = df["player"].astype(str)

    weeks_present = sorted(df["week"].dropna().unique().tolist())
    players = sorted(df["player"].dropna().unique().tolist())

    recs = []
    for w in weeks_present:
        df_w = df[df["week"] <= w]
        for p in players:
            s = df_w.loc[df_w["player"] == p, "score"].dropna().values
            counted = int(min(len(s), counted_weeks))
            topk_sum = float(np.sort(s)[-counted:].sum()) if counted > 0 else 0.0

            if use_projection_flag and 0 < counted < counted_weeks:
                on_pace = topk_sum * counted_weeks / counted
            else:
                on_pace = topk_sum

            recs.append({
                "week": w,
                "player": p,
                "counted_weeks_so_far": counted,
                "topk_total": round(topk_sum, 3),
                "on_pace_total": round(on_pace, 3),
            })

    weekly = pd.DataFrame(recs)
    if weekly.empty:
        return weekly

    metric = "on_pace_total" if use_projection_flag else "topk_total"
    weekly["rank"] = weekly.groupby("week")[metric].rank(method="min", ascending=False).astype(int)
    weekly = weekly.sort_values(["player", "week"])
    weekly["prev_rank"] = weekly.groupby("player")["rank"].shift(1)
    weekly["rank_delta"] = weekly["prev_rank"] - weekly["rank"]  # + = improved vs last week
    return weekly

# ---------------------------
# Load Data
# ---------------------------
if uploaded is None:
    st.info("Upload your CSV to begin. You can also download a template below.")
    long_df = pd.DataFrame(columns=["week", "player", "score"])
else:
    try:
        raw = pd.read_csv(uploaded, sep=None, engine="python")  # auto-detect commas/tabs
        long_df = normalize_input(raw)
    except Exception as e:
        st.error(f"Could not read/parse CSV: {e}")
        st.stop()

# Apply week filter (optional)
if not long_df.empty and selected_weeks:
    long_df = long_df[long_df["week"].isin(selected_weeks)]

# Compute
weekly = compute_time_series(long_df, COUNTED_WEEKS, use_projection)

if long_df.empty or weekly.empty:
    st.warning("No data to compute yet. Add scores or check your filters.")
else:
    latest_week = int(weekly["week"].max())
    metric = "on_pace_total" if use_projection else "topk_total"

    # Build standings for latest week
    standings = (
        weekly[weekly["week"] == latest_week]
        .sort_values([metric, "player"], ascending=[False, True])
        .assign(place=lambda d: range(1, len(d) + 1))
        [["place", "player", "counted_weeks_so_far", metric, "rank", "rank_delta"]]
        .rename(columns={metric: "season_total_to_date"})
    )

    # Progress toward 10 counted weeks
    with st.container():
        c1, c2, c3 = st.columns([1, 1, 1])
        with c1:
            st.subheader(f"📊 Standings after Week {latest_week}")
        with c2:
            st.metric("Counting Rule", f"Top {COUNTED_WEEKS} of {MAX_WEEKS}")
        with c3:
            st.metric("Projection", "On" if use_projection else "Off")

    # Fancy dataframe with progress bar (Streamlit column config)
    try:
        st.dataframe(
            standings,
            use_container_width=True,
            column_config={
                "place": st.column_config.NumberColumn("Place", format="%d", width="small"),
                "player": st.column_config.TextColumn("Player"),
                "counted_weeks_so_far": st.column_config.ProgressColumn(
                    "Counted Weeks",
                    help=f"Number of weeks currently counting toward Top {COUNTED_WEEKS}",
                    min_value=0, max_value=COUNTED_WEEKS
                ),
                "season_total_to_date": st.column_config.NumberColumn("Season Total (to-date)", format="%.2f"),
                "rank": st.column_config.NumberColumn("Rank", format="%d", width="small"),
                "rank_delta": st.column_config.NumberColumn("Δ Rank vs Prev", help="+ = improved", width="small"),
            }
        )
    except Exception:
        st.dataframe(standings, use_container_width=True)

    # Downloads
    colA, colB = st.columns([1, 1])
    with colA:
        st.download_button(
            "Download Standings CSV",
            standings.to_csv(index=False).encode("utf-8"),
            file_name=f"standings_week_{latest_week}.csv",
            mime="text/csv",
        )
    with colB:
        st.download_button(
            "Download Weekly Time Series CSV",
            weekly.to_csv(index=False).encode("utf-8"),
            file_name="weekly_timeseries.csv",
            mime="text/csv",
        )

    st.markdown("---")

    # Label selection
    all_players = sorted(weekly["player"].unique())
    sel_players = st.multiselect("Highlight players (optional)", options=all_players, default=[])

    # If user prefers Username for labels (wide CSV), allow quick relabel
    if uploaded is not None and is_wide_schema(pd.read_csv(uploaded, sep=None, engine="python")) and name_field == "Username":
        # Rebuild label map from the original wide file
        wide_src = pd.read_csv(uploaded, sep=None, engine="python")
        label_map = {}
        for _, r in wide_src.iterrows():
            name = str(r.get("Name", "")).strip()
            uname = str(r.get("Username", "")).strip()
            if name and uname:
                label_map[name] = uname
        weekly["player"] = weekly["player"].map(lambda p: label_map.get(p, p))
        standings["player"] = standings["player"].map(lambda p: label_map.get(p, p))

    # ---------------------------
    # Chart 1: Total Score Over Time
    # ---------------------------
    st.markdown("### 📈 Total Score Over Time (Top-10 rule applied)")

    metric = "on_pace_total" if use_projection else "topk_total"
    chart_df = weekly.copy()

    base = alt.Chart(chart_df).mark_line().encode(
        x=alt.X("week:O", title="Week"),
        y=alt.Y(f"{metric}:Q", title="Season Total (to-date)", scale=alt.Scale(zero=True)),
        color=alt.Color("player:N", legend=alt.Legend(title="Player", orient="bottom")),
        tooltip=[
            alt.Tooltip("player:N"),
            alt.Tooltip("week:O"),
            alt.Tooltip("counted_weeks_so_far:Q", title="Counted Weeks So Far"),
            alt.Tooltip(f"{metric}:Q", title="Total (to-date)", format=".2f"),
        ],
    )

    if sel_players:
        base = base.encode(
            opacity=alt.condition(alt.FieldOneOfPredicate(field="player", oneOf=sel_players), alt.value(1.0), alt.value(0.25))
        )

    chart = base.properties(height=420).interactive(bind_y=False)

    if highlight_latest_labels:
        last_points = chart_df.sort_values("week").groupby("player").tail(1)
        text = alt.Chart(last_points).mark_text(dx=5, dy=0, align="left").encode(
            x="week:O",
            y=f"{metric}:Q",
            text=alt.Text("player:N")
        )
        chart = chart + text

    st.altair_chart(chart, use_container_width=True)

    # ---------------------------
    # Chart 2: Change in League Rank by Week
    # ---------------------------
    st.markdown("### 📉 Change in League Rank by Week (Δ vs previous week)")

    rank_df = weekly.copy()
    rank_df["rank_change"] = rank_df["rank_delta"].fillna(0)

    base_rank = alt.Chart(rank_df).mark_line(point=True).encode(
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

    if sel_players:
        base_rank = base_rank.encode(
            opacity=alt.condition(alt.FieldOneOfPredicate(field="player", oneOf=sel_players), alt.value(1.0), alt.value(0.25))
        )

    st.altair_chart(base_rank.properties(height=380), use_container_width=True)

    # Optional table
    with st.expander("Table: Weekly Totals / Ranks", expanded=False):
        st.dataframe(
            weekly.sort_values(["week", "rank"])
                  .rename(columns={"topk_total": "top10_total_to_date", "on_pace_total": "on_pace_total_to_date"}),
            use_container_width=True
        )

# ---------------------------
# Template Downloads (Wide + Long)
# ---------------------------
st.markdown("---")
st.subheader("📥 Templates")

# Wide template (your schema)
wide_cols = ["Name", "Username"] + [f"Week {i}" for i in range(1, MAX_WEEKS+1)] + ["Last Rank", "New Rank", "Change", "Average", "Top 4"]
wide_templ = pd.DataFrame(columns=wide_cols)
buf_wide = io.StringIO(); wide_templ.to_csv(buf_wide, index=False)
st.download_button("Download Wide Template (like your sheet)", buf_wide.getvalue().encode("utf-8"),
                   file_name="dfs_league_wide_template.csv", mime="text/csv")

# Long template
long_templ = pd.DataFrame(columns=["week", "player", "score"])
buf_long = io.StringIO(); long_templ.to_csv(buf_long, index=False)
st.download_button("Download Long Template (week,player,score)", buf_long.getvalue().encode("utf-8"),
                   file_name="dfs_league_long_template.csv", mime="text/csv")

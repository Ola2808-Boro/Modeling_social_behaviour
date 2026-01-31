import json
import os
from collections import Counter
from functools import reduce

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from utils import (
    ACTIVITIES_DATA_PATH,
    BODY_DATA_PATH,
    MAPPING_FILE_PATH,
    PREPROCESSED_DATA_PATH,
    read_data,
)

options = ["source"]
periods = ["year_month_day"]


def subreddit_activity():
    """
    Aggregate subreddit activity counts over time and labels, and save results to CSV files.

    The function iterates over predefined options (source/target/entity) and time periods
    (e.g. year, month, day). For each subreddit identifier defined in the mapping file,
    it filters the preprocessed dataset, assigns a time period, aggregates activity counts
    by period and label, and appends the results to a CSV file.

    Output files are saved under:
        ACTIVITIES_DATA_PATH/<option>/<period>/activities.csv

    Side effects:
        - Creates directories if they do not exist.
        - Appends or creates CSV files with aggregated activity data.
    """
    mapping_data = read_data(path=MAPPING_FILE_PATH)
    preprocessed_data = read_data(path=PREPROCESSED_DATA_PATH, sep=",")
    subreddit_idxs = list(mapping_data.values())
    for option in options:
        for period in periods:
            path = f"{ACTIVITIES_DATA_PATH}/{option}/{period}/activities.csv"
            os.makedirs(f"{ACTIVITIES_DATA_PATH}/{option}/{period}", exist_ok=True)
            for idx in subreddit_idxs:
                if option == "source":
                    selected_subreddit = preprocessed_data[
                        preprocessed_data["Source"] == idx
                    ]
                elif option == "target":
                    selected_subreddit = preprocessed_data[
                        preprocessed_data["Target"] == idx
                    ]
                else:
                    selected_subreddit = preprocessed_data[
                        (preprocessed_data["Source"] == idx)
                        | (preprocessed_data["Target"] == idx)
                    ]
                selected_subreddit["Timestamp"] = pd.to_datetime(
                    selected_subreddit["Timestamp"], errors="coerce"
                )
                if period == "year":
                    selected_subreddit["period"] = selected_subreddit[
                        "Timestamp"
                    ].dt.year.astype(str)
                elif period == "year_month":
                    selected_subreddit["period"] = (
                        selected_subreddit["Timestamp"].dt.to_period("M").astype(str)
                    )
                elif period == "year_month_day":
                    selected_subreddit["period"] = (
                        selected_subreddit["Timestamp"].dt.to_period("D").astype(str)
                    )
                else:
                    raise ValueError("Unknown period: " + period)

                grouped = (
                    selected_subreddit.groupby(["period", "Label"])
                    .size()
                    .reset_index(name="count")
                )
                if option == "source":
                    grouped["source"] = idx
                    out_df = grouped[["source", "period", "Label", "count"]]
                elif option == "target":
                    grouped["target"] = idx
                    out_df = grouped[["target", "period", "Label", "count"]]
                else:
                    grouped["entity"] = idx
                    out_df = grouped[["entity", "period", "Label", "count"]]

                header = not os.path.exists(path)
                out_df.to_csv(path, mode="a", header=header, index=False)


def plot_N_the_most_least_active_subreddit(N: int):
    """
    Identify and visualize the N most and N least active subreddits.

    The function aggregates total activity counts per subreddit, selects the top-N and
    bottom-N entities, resolves subreddit names using the mapping file, and generates
    bar plots saved as PNG files.

    Additionally, a summary CSV file with total activity counts is created.

    Parameters
    ----------
    N : int
        Number of most and least active subreddits to visualize.

    Output files:
        - <N>_most_active.png
        - <N>_least_active.png
        - activities_summary.csv

    All files are saved under:
        ACTIVITIES_DATA_PATH/<option>/<period>/
    """
    mapping_data = read_data(path=MAPPING_FILE_PATH)
    if isinstance(mapping_data, dict):
        sample_vals = list(mapping_data.values())
        sample_keys = list(mapping_data.keys())
        if all(isinstance(v, (int, str)) for v in sample_vals) and any(
            not str(k).isdigit() for k in sample_keys
        ):
            idx_to_name = {str(v): str(k) for k, v in mapping_data.items()}
        else:

            idx_to_name = {str(k): str(v) for k, v in mapping_data.items()}
    else:
        try:
            name_col = mapping_data.columns[0]
            idx_col = mapping_data.columns[1]
            idx_to_name = dict(
                zip(
                    mapping_data[idx_col].astype(str),
                    mapping_data[name_col].astype(str),
                )
            )
        except Exception:
            idx_to_name = {str(v): str(k) for k, v in mapping_data.items()}

    for option in options:
        for period in periods:
            path = f"{ACTIVITIES_DATA_PATH}/{option}/{period}/activities.csv"
            out_path = (
                f"{ACTIVITIES_DATA_PATH}/{option}/{period}/activities_summary.csv"
            )
            os.makedirs(f"{ACTIVITIES_DATA_PATH}/{option}/{period}", exist_ok=True)

            subreddits = read_data(path=path, sep=",")
            grouped = (
                subreddits.groupby(by=list(subreddits.columns)[0])["count"]
                .sum()
                .reset_index(name="total_count")
            )
            agg_sorted = grouped.sort_values("total_count", ascending=False)
            selected_N_the_most_active = agg_sorted.head(N).copy()
            selected_N_the_least_active = agg_sorted.tail(N).copy()

            id_col = list(selected_N_the_most_active.columns)[0]
            ids_most = selected_N_the_most_active[id_col].astype(str).tolist()
            ids_least = selected_N_the_least_active[id_col].astype(str).tolist()

            names_most = [idx_to_name.get(i, f"UNK_{i}") for i in ids_most]
            names_least = [idx_to_name.get(i, f"UNK_{i}") for i in ids_least]

            def make_unique_labels(names, ids):
                seen = {}
                labels = []
                for name, idv in zip(names, ids):
                    key = name
                    count = seen.get(key, 0) + 1
                    seen[key] = count
                    if count > 1:
                        label = f"{name} ({idv})"
                    else:
                        label = name
                    labels.append(label)
                return labels

            labels_most = make_unique_labels(names_most, ids_most)
            labels_least = make_unique_labels(names_least, ids_least)

            selected_N_the_most_active = selected_N_the_most_active.reset_index(
                drop=True
            )
            selected_N_the_most_active["id_str"] = ids_most
            selected_N_the_most_active["name"] = names_most
            selected_N_the_most_active["subreddits"] = labels_most

            selected_N_the_least_active = selected_N_the_least_active.reset_index(
                drop=True
            )
            selected_N_the_least_active["id_str"] = ids_least
            selected_N_the_least_active["name"] = names_least
            selected_N_the_least_active["subreddits"] = labels_least

            print(
                f"{option}/{period} - most: rows={len(selected_N_the_most_active)} unique_names={selected_N_the_most_active['name'].nunique()}"
            )
            print(
                f"mapped_values_the_most_active: {selected_N_the_most_active['name'].tolist()}"
            )

            category_order_most = selected_N_the_most_active["subreddits"].tolist()
            fig1 = px.bar(
                selected_N_the_most_active,
                x="subreddits",
                y="total_count",
                title=f"The {N} most active subreddits",
                category_orders={"subreddits": category_order_most},
            )
            fig1.update_layout(xaxis_tickangle=-45, height=600, margin=dict(b=200))
            fig1.update_traces(texttemplate="%{y:,}", textposition="outside")
            fig1.write_image(f"{path.replace('activities.csv',f'{N}_most_active.png')}")

            category_order_least = selected_N_the_least_active["subreddits"].tolist()
            fig2 = px.bar(
                selected_N_the_least_active,
                x="subreddits",
                y="total_count",
                title=f"The {N} least active subreddits",
                category_orders={"subreddits": category_order_least},
            )
            fig2.update_layout(xaxis_tickangle=-45, height=600, margin=dict(b=200))
            fig2.update_traces(texttemplate="%{y:,}", textposition="outside")
            fig2.write_image(
                f"{path.replace('activities.csv',f'{N}_least_active.png')}"
            )

            header = not os.path.exists(out_path)
            agg_sorted.to_csv(out_path, mode="a", header=header, index=False)


def the_most_active_day(N: int):
    """
    Identify and visualize the N most active days in the dataset.

    The function aggregates total activity counts per day, selects the top-N days,
    analyzes subreddit overlap across peak activity days, and produces a bar chart.

    Parameters
    ----------
    N : int
        Number of peak activity days to analyze.

    Output:
        - <N>_most_activite_days.png
        - Console output with subreddit overlap statistics
    """
    out_path = (
        f"{ACTIVITIES_DATA_PATH}/{options[0]}/{periods[0]}/activities_per_day.csv"
    )
    df = read_data(
        path=f"{ACTIVITIES_DATA_PATH}/{options[0]}/{periods[0]}/activities.csv",
        sep=",",
    )
    original_df = df.copy()
    original_df["period_dt"] = pd.to_datetime(df["period"], errors="coerce")
    original_df["period_str"] = (
        original_df["period_dt"]
        .dt.strftime("%d.%m.%Y")
        .fillna(original_df["period"].astype(str))
    )

    df = df.groupby("period")["count"].sum().reset_index(name="total_count")
    df["period_dt"] = pd.to_datetime(df["period"], errors="coerce")
    selected = df.sort_values("total_count", ascending=False).head(N).copy()
    selected["period_str"] = (
        selected["period_dt"]
        .dt.strftime("%d.%m.%Y")
        .fillna(selected["period"].astype(str))
    )

    category_order = list(selected["period_str"])
    top_subreddits_per_day = {}
    print(f"original_df: {original_df.columns}")
    for day in category_order:
        day_df = original_df[original_df["period_str"] == day]
        top_subs = (
            day_df.groupby("source")["count"]
            .sum()
            .sort_values(ascending=False)
            .index.tolist()
        )
        top_subreddits_per_day[day] = set(top_subs)

    common_subreddits = reduce(set.intersection, top_subreddits_per_day.values())

    union_subreddits = reduce(set.union, top_subreddits_per_day.values())
    counter = Counter()

    for subs in top_subreddits_per_day.values():
        counter.update(subs)

    most_common_subs = counter.most_common(10)
    print("=== PEAK DAY COMMUNITY ANALYSIS ===")
    print(f"Number of peak days analyzed: {N}")

    print(f"Subreddits present in ALL peak days ({len(common_subreddits)}):")
    for s in common_subreddits:
        print(f" - {s}")

    print("Most frequently appearing subreddits across peak days:")
    for sub, freq in most_common_subs:
        print(f" - {sub}: {freq}/{10} days")

    fig = px.bar(
        selected,
        x="period_str",
        y="total_count",
        title=f"The {N} most active days",
        labels={"period_str": "Day", "total_count": "Total activity count"},
        category_orders={"period_str": category_order},
    )

    fig.update_layout(
        height=600,
        bargap=0.2,
        margin=dict(t=60, b=160),
    )
    fig.update_traces(texttemplate="%{y:,}", textposition="outside")
    fig.write_image(
        f"{out_path.replace('activities_per_day.csv',f'{N}_most_activite_days.png')}"
    )


def analyze_activities_trend():
    """
    Analyze temporal trends in the number of active nodes (unique subreddits).

    The function computes the daily number of unique nodes appearing as source
    or target, and visualizes the trend over time using a line plot.

    Output:
        - nodes_over_time.png saved to ACTIVITIES_DATA_PATH
    """
    df = read_data(path=f"{PREPROCESSED_DATA_PATH}", sep=",")
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    df["period"] = df["Timestamp"].dt.to_period("D").astype(str)
    tmp = df[["period", "Source", "Target"]].copy()
    tmp = tmp.melt(
        id_vars=["period"], value_vars=["Source", "Target"], value_name="entity"
    )
    tmp = tmp.dropna(subset=["entity"])

    grouped_df = tmp.groupby("period", as_index=False).agg(
        nodes=("entity", lambda s: list(pd.Series(s).dropna().unique())),
        nodes_count=("entity", lambda s: int(pd.Series(s).nunique())),
    )
    mean_val = grouped_df["nodes_count"].mean()
    idx_min = grouped_df["nodes_count"].idxmin()
    idx_max = grouped_df["nodes_count"].idxmax()
    min_row = grouped_df.loc[idx_min]
    max_row = grouped_df.loc[idx_max]
    print(grouped_df["nodes_count"])
    fig = go.Figure(
        [
            go.Scatter(
                x=grouped_df["period"],
                y=grouped_df["nodes_count"].values,
                mode="lines",
                name="Nodes count",
                marker=dict(size=6),
                line=dict(width=2),
                hovertemplate="%{x|%d.%m.%Y}<br>Nodes: %{y}<extra></extra>",
            )
        ]
    )
    fig.add_trace(
        go.Scatter(
            x=grouped_df["period"],
            y=[mean_val] * len(grouped_df),
            mode="lines",
            name="Mean",
            line=dict(color="orange", width=2, dash="dash"),
            hoverinfo="skip",
            showlegend=True,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[min_row["period"]],
            y=[min_row["nodes_count"]],
            mode="markers",
            marker=dict(color="red", size=10, symbol="triangle-down"),
            name="Min",
            hovertemplate="%{x|%d.%m.%Y}<br>Min: %{y}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[max_row["period"]],
            y=[max_row["nodes_count"]],
            mode="markers",
            marker=dict(color="green", size=10, symbol="triangle-up"),
            name="Max",
            hovertemplate="%{x|%d.%m.%Y}<br>Max: %{y}<extra></extra>",
        )
    )
    fig.add_annotation(
        x=min_row["period"],
        y=min_row["nodes_count"],
        text=f"Min: {min_row['nodes_count']}<br>{min_row['period']}",
        showarrow=True,
        arrowhead=2,
        ax=0,
        ay=30,
        font=dict(color="red", size=11),
        bgcolor="rgba(255,255,255,0.8)",
    )

    fig.add_annotation(
        x=max_row["period"],
        y=max_row["nodes_count"],
        text=f"Max: {max_row['nodes_count']}<br>{max_row['period']}",
        showarrow=True,
        arrowhead=2,
        ax=0,
        ay=-40,
        font=dict(color="green", size=11),
        bgcolor="rgba(255,255,255,0.8)",
    )

    fig.add_annotation(
        x=grouped_df["period"].iloc[-1],
        y=mean_val,
        xanchor="left",
        text=f"Mean = {mean_val:.2f}",
        showarrow=False,
        font=dict(color="orange", size=11),
        bgcolor="rgba(255,255,255,0.8)",
    )
    fig.update_layout(
        title=dict(text="Change in number of nodes over time", x=0.5),
        xaxis=dict(
            title="Date",
            type="date",
            tickformat="%d.%m.%Y",
            tickangle=45,
            tickmode="auto",
            nticks=20,
            range=[
                grouped_df["period"].min(),
                grouped_df["period"].max(),
            ],
        ),
        yaxis=dict(title="Number of new nodes"),
    )
    fig.write_image(f"{ACTIVITIES_DATA_PATH}/nodes_over_time.png")


def analyze_edges():
    """
    Analyze temporal distribution of positive and negative subreddit hyperlinks.

    The function aggregates edge labels per month, computes descriptive statistics
    for positive and negative edges, visualizes their temporal distribution, and
    saves statistics to JSON files.

    Output files:
        - edges_distribution.png
        - positive.json
        - negative.json

    All files are saved under:
        ACTIVITIES_DATA_PATH/polarization/
    """
    out_dir = f"{ACTIVITIES_DATA_PATH}/polarization"
    df = read_data(path=f"{PREPROCESSED_DATA_PATH}", sep=",")
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    df["period"] = df["Timestamp"].dt.to_period("M").astype(str)
    print(df["Label"].value_counts())
    grouped_df = (df.sort_values(by="period").groupby(by="period", as_index=False))[
        "Label"
    ].value_counts()
    print(grouped_df)
    positive_df = grouped_df[grouped_df["Label"] == 1]
    negative_df = grouped_df[grouped_df["Label"] == -1]
    cases = {"positive": positive_df, "negative": negative_df}
    stats = {}
    for name, df in cases.items():
        vals = df["count"].values
        stats[name] = {
            "mean": float(vals.mean()),
            "std": float(vals.std(ddof=0)),
            "min": int(vals.min()),
            "max": int(vals.max()),
        }
    fig = px.bar(
        data_frame=grouped_df,
        x="period",
        y="count",
        color="Label",
        title=f"Temporal distribution of positive and negative subreddit hyperlinks",
        labels={"1": "positive", "-1": "negative"},
    )
    fig.update_layout(
        xaxis=dict(tickformat="%Y-%m", tickangle=45),
        legend=dict(title="Label"),
        height=600,
        margin=dict(t=80, b=160),
    )
    fig.write_image(f"{out_dir}/edges_distribution.png")

    with open(f"{out_dir}/positive.json", "w") as f:
        json.dump(stats["positive"], f, indent=4)

    with open(f"{out_dir}/negative.json", "w") as f:
        json.dump(stats["negative"], f, indent=4)

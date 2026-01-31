import json
import os
from collections import defaultdict

import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from utils import (
    ACTIVITIES_DATA_PATH,
    MAPPING_FILE_PATH,
    NET_PATH,
    NET_POSITVE_PATH,
    PREPROCESSED_DATA_PATH,
    get_all_keys_by_value,
    read_data,
)


def build_network_and_analyze(top_N: int = 20):
    """
    Build a directed weighted network of subreddit interactions and perform
    degree-based centrality analysis.

    The function aggregates multiple interactions between the same subreddit
    pairs, separately tracks positive and negative edge weights, computes
    in/out degree statistics, and generates visualizations and summary files.

    Parameters
    ----------
    top_N : int, optional
        Number of top nodes to include in rankings and plots (default: 20).
    """
    mapping_data = read_data(path=MAPPING_FILE_PATH)
    preprocessed_data = read_data(path=PREPROCESSED_DATA_PATH, sep=",")

    if isinstance(mapping_data, dict):
        keys = list(mapping_data.keys())[:10]
        vals = list(mapping_data.values())[:10]
        if any(not str(k).isdigit() for k in keys) and all(
            str(v).isdigit() for v in vals
        ):
            idx_to_name = {str(v): str(k) for k, v in mapping_data.items()}
        else:
            idx_to_name = {str(k): str(v) for k, v in mapping_data.items()}
    else:
        cols = list(mapping_data.columns)
        if len(cols) >= 2:
            name_col, idx_col = cols[0], cols[1]
            idx_to_name = dict(
                zip(
                    mapping_data[idx_col].astype(str),
                    mapping_data[name_col].astype(str),
                )
            )
        else:

            idx_to_name = {str(v): str(k) for k, v in mapping_data.items()}

    mdg = nx.MultiDiGraph()
    mdg.add_nodes_from([str(i) for i in idx_to_name.keys()])
    if (
        "Source" not in preprocessed_data.columns
        or "Target" not in preprocessed_data.columns
    ):
        raise ValueError("preprocessed_data must contain 'Source' and 'Target' columns")

    for _, row in preprocessed_data.iterrows():
        src = row["Source"]
        tgt = row["Target"]
        label = row.get("Label", None)
        if pd.isna(src) or pd.isna(tgt):
            continue
        s = str(src)
        t = str(tgt)
        mdg.add_edge(s, t, label=label, weight=1)

    G = nx.DiGraph()
    G.add_nodes_from(mdg.nodes())

    for u, v, data in mdg.edges(data=True):
        lbl = data.get("label", None)

        if not G.has_edge(u, v):
            G.add_edge(u, v, weight=0, pos_weight=0, neg_weight=0)

        G[u][v]["weight"] += data.get("weight", 1)

        if lbl == 1 or str(lbl) == "1":
            G[u][v]["pos_weight"] = G[u][v].get("pos_weight", 0) + 1
        elif lbl == -1 or str(lbl) == "-1":
            G[u][v]["neg_weight"] = G[u][v].get("neg_weight", 0) + 1
        else:

            pass

    out_deg = dict(G.out_degree())
    weighted_out_deg = dict(G.out_degree(weight="weight"))
    in_deg = dict(G.in_degree())
    weighted_in_deg = dict(G.in_degree(weight="weight"))

    pos_out = defaultdict(float)
    neg_out = defaultdict(float)
    pos_in = defaultdict(float)
    neg_in = defaultdict(float)

    for u, v, data in G.edges(data=True):
        pw = data.get("pos_weight", 0) or 0
        nw = data.get("neg_weight", 0) or 0
        pos_out[u] += pw
        neg_out[u] += nw
        pos_in[v] += pw
        neg_in[v] += nw

    all_nodes = sorted(set(G.nodes()) | set(idx_to_name.keys()))

    rows = []
    for node in all_nodes:
        rows.append(
            {
                "id": node,
                "name": idx_to_name.get(node, f"UNK_{node}"),
                "out_degree": int(out_deg.get(node, 0)),
                "weighted_out_degree": float(weighted_out_deg.get(node, 0.0)),
                "in_degree": int(in_deg.get(node, 0)),
                "weighted_in_degree": float(weighted_in_deg.get(node, 0.0)),
                "pos_out_weight": float(pos_out.get(node, 0.0)),
                "neg_out_weight": float(neg_out.get(node, 0.0)),
                "pos_in_weight": float(pos_in.get(node, 0.0)),
                "neg_in_weight": float(neg_in.get(node, 0.0)),
            }
        )

    df = pd.DataFrame(rows)

    df["role_score"] = df["weighted_out_degree"] - df["weighted_in_degree"]

    df["out_pos_ratio"] = df.apply(
        lambda r: (
            (r["pos_out_weight"] / (r["pos_out_weight"] + r["neg_out_weight"]))
            if (r["pos_out_weight"] + r["neg_out_weight"]) > 0
            else np.nan
        ),
        axis=1,
    )
    df["in_pos_ratio"] = df.apply(
        lambda r: (
            (r["pos_in_weight"] / (r["pos_in_weight"] + r["neg_in_weight"]))
            if (r["pos_in_weight"] + r["neg_in_weight"]) > 0
            else np.nan
        ),
        axis=1,
    )

    out_dir = "data/network_centrality"
    os.makedirs(out_dir, exist_ok=True)

    df_sorted_out = df.sort_values("weighted_out_degree", ascending=False)
    df_sorted_in = df.sort_values("weighted_in_degree", ascending=False)

    df.to_csv(os.path.join(out_dir, "degree_measures_all_nodes.csv"), index=False)
    df_sorted_out.to_csv(
        os.path.join(out_dir, f"top_{top_N}_by_weighted_out.csv"), index=False
    )
    df_sorted_in.to_csv(
        os.path.join(out_dir, f"top_{top_N}_by_weighted_in.csv"), index=False
    )

    def save_json_sorted(dataframe, col, fname):
        arr = []
        for _, r in dataframe.head(500).iterrows():
            arr.append((r["id"], r["name"], float(r[col])))
        with open(os.path.join(out_dir, fname), "w") as f:
            json.dump(arr, f, indent=2)

    save_json_sorted(df_sorted_out, "weighted_out_degree", "weighted_out_degree.json")
    save_json_sorted(df_sorted_in, "weighted_in_degree", "weighted_in_degree.json")
    save_json_sorted(
        df.sort_values("role_score", ascending=False), "role_score", "role_score.json"
    )

    top_out = df_sorted_out.head(top_N).copy()
    fig = px.bar(
        top_out,
        x="name",
        y="weighted_out_degree",
        title=f"Top {top_N} subreddits by weighted out-degree (activity)",
        text="weighted_out_degree",
    )
    fig.update_layout(xaxis_tickangle=-45, height=600, margin=dict(b=200))
    fig.write_image(os.path.join(out_dir, f"top_{top_N}_weighted_out.png"))

    top_in = df_sorted_in.head(top_N).copy()
    fig = px.bar(
        top_in,
        x="name",
        y="weighted_in_degree",
        title=f"Top {top_N} subreddits by weighted in-degree (influence)",
        text="weighted_in_degree",
    )
    fig.update_layout(xaxis_tickangle=-45, height=600, margin=dict(b=200))
    fig.write_image(os.path.join(out_dir, f"top_{top_N}_weighted_in.png"))

    fig = px.scatter(
        df,
        x="weighted_out_degree",
        y="weighted_in_degree",
        hover_name="name",
        title="Activity (weighted out) vs Influence (weighted in)",
        log_x=False,
        log_y=False,
    )
    fig.add_shape(
        type="line",
        x0=0,
        y0=0,
        x1=max(df["weighted_out_degree"].max(), 1),
        y1=max(df["weighted_out_degree"].max(), 1),
        line=dict(dash="dash"),
    )
    fig.update_layout(height=700)
    fig.write_image(os.path.join(out_dir, "activity_vs_influence_scatter.png"))

    fig = px.histogram(
        df,
        x="role_score",
        nbins=50,
        title="Distribution of role_score (activity - influence)",
    )
    fig.update_layout(height=500)
    fig.write_image(os.path.join(out_dir, "role_score_histogram.png"))

    top_out_posneg = top_out.copy()
    top_out_posneg = top_out_posneg.assign(
        pos=top_out_posneg["pos_out_weight"], neg=top_out_posneg["neg_out_weight"]
    )
    melted = top_out_posneg.melt(
        id_vars=["name"],
        value_vars=["pos", "neg"],
        var_name="sentiment",
        value_name="count",
    )
    fig = px.bar(
        melted,
        x="name",
        y="count",
        color="sentiment",
        title=f"Top {top_N} outgoing pos/neg weights",
    )
    fig.update_layout(xaxis_tickangle=-45, height=600, margin=dict(b=200))
    fig.write_image(os.path.join(out_dir, f"top_{top_N}_pos_neg_outgoing.png"))

    try:
        from scipy.stats import spearmanr

        rho, p = spearmanr(df["weighted_out_degree"], df["weighted_in_degree"])
    except Exception:
        rho, p = None, None

    print("\nTop by weighted_out_degree (activity):")
    print(
        df_sorted_out[["id", "name", "weighted_out_degree"]]
        .head(top_N)
        .to_string(index=False)
    )
    print("\nTop by weighted_in_degree (influence):")
    print(
        df_sorted_in[["id", "name", "weighted_in_degree"]]
        .head(top_N)
        .to_string(index=False)
    )
    if rho is not None:
        print(
            f"\nSpearman rank correlation weighted_out vs weighted_in: rho={rho:.3f}, p-value={p:.3e}"
        )
    else:
        print("\nSpearman correlation not available (scipy missing).")

    top_act = df_sorted_out.head(5)["name"].tolist()
    top_inf = df_sorted_in.head(5)["name"].tolist()

    analysis = (
        "Centrality analysis: top active subreddits (weighted out-degree): "
        + ", ".join(top_act)
        + ". Top influential (weighted in-degree): "
        + ", ".join(top_inf)
        + ".\n"
    )

    with open(
        os.path.join(out_dir, "analysis_summary_en.txt"), "w", encoding="utf-8"
    ) as f:
        f.write(analysis)

    nx.write_graphml(G, os.path.join(out_dir, "net.graphml"))


def build_network_positive_edges():
    """
    Construct a network containing only positive interactions and export it
    in both directed and undirected GraphML formats.
    """
    mapping_data = read_data(path=MAPPING_FILE_PATH)
    preprocessed_data = read_data(path=PREPROCESSED_DATA_PATH, sep=",")

    df_pos = preprocessed_data[preprocessed_data["Label"] == 1]
    G = nx.DiGraph()
    G.add_nodes_from(mapping_data.values())

    for _, row in df_pos.iterrows():
        src, tgt = row["Source"], row["Target"]
        if G.has_edge(src, tgt):
            G[src][tgt]["weight"] += 1
        else:
            G.add_edge(src, tgt, weight=1)

    G_undirected = G.to_undirected(reciprocal=False)
    nx.write_graphml_lxml(G_undirected, "net_undirected_positive_edges.graphml")
    nx.write_graphml_lxml(G, "net_directed_positive_edges.graphml")


def analyze_degree_centrality():
    """
    Analyze degree centrality of nodes in the network.

    The function computes:
    - raw degree (number of connections)
    - normalized degree centrality

    Results are stored as JSON files sorted in descending order.
    """
    mapping_data = read_data(path=MAPPING_FILE_PATH)
    node_names = list(mapping_data.keys())
    net = read_net()
    data = {}
    degree = nx.degree_centrality(net)
    for i in range(len(node_names)):
        print(list(net.degree([str(i)]))[0])
        _, connections = list(net.degree([str(i)]))[0]
        data.update({node_names[i]: connections})
    data = dict(sorted(data.items(), key=lambda item: item[1], reverse=True))
    os.makedirs("data/network_centrality/", exist_ok=True)
    with open("data/network_centrality/normalized_degree_centrality.json", "w") as f:
        json.dump(degree, f, indent=1)
    with open("data/network_centrality/degree_centrality.json", "w") as f:
        json.dump(data, f, indent=1)


def analyze_closeness_centrality():
    """
    Compute and store closeness centrality for all nodes.
    """
    net = read_net()
    closeness = nx.closeness_centrality(net)
    sorted_closeness = dict(
        sorted(closeness.items(), key=lambda item: item[1], reverse=True)
    )
    os.makedirs("data/network_centrality/", exist_ok=True)
    with open("data/network_centrality/normalized_closeness_centrality.json", "w") as f:
        json.dump(sorted_closeness, f, indent=1)


def analyze_eigenvector_centrality():
    """
    Compute and store eigenvector centrality for the network.
    """
    net = read_net()
    eigenvector = nx.eigenvector_centrality(net)
    sorted_eigenvector = dict(
        sorted(eigenvector.items(), key=lambda item: item[1], reverse=True)
    )
    os.makedirs("data/network_centrality/", exist_ok=True)
    with open(
        "data/network_centrality/normalized_eigenvector_centrality.json", "w"
    ) as f:
        json.dump(sorted_eigenvector, f, indent=1)


def analyze_betweenness_centrality():
    """
    Compute and store betweenness centrality for the network.
    """
    mapping_data = read_data(path=MAPPING_FILE_PATH)
    net = read_net()
    betweenness = nx.betweenness_centrality(net)
    sorted_betweenness = dict(
        sorted(betweenness.items(), key=lambda item: item[1], reverse=True)
    )
    os.makedirs("data/network_centrality/", exist_ok=True)
    with open(
        "data/network_centrality/normalized_betweenness_centrality.json", "w"
    ) as f:
        json.dump(sorted_betweenness, f, indent=1)


def read_net(type: str = "normal"):
    if type == "positive":
        net = nx.read_graphml(NET_POSITVE_PATH)
    else:
        net = nx.read_graphml(NET_PATH)
    return net


def in_degree_analyze(top_n: int = 20):
    """
    Analyze concentration of outgoing activity among top-N most influential nodes.

    The function examines how much of the total outgoing activity
    (out-degree and weighted out-degree) is generated by nodes that rank
    highest according to weighted in-degree (influence).

    This allows assessing whether influence in the network is concentrated
    among a small number of highly active subreddits.

    Parameters
    ----------
    top_n : int, optional
        Number of top nodes considered in the analysis (default: 20).

    Outputs
    -------
    - JSON file with summary statistics
    """
    df = read_data(
        path=r"data\network_centrality\top_20_by_weighted_in.csv",
        sep=",",
    )
    all_out_degree_std = df["out_degree"].std()
    all_out_degree = df["out_degree"].sum()
    all_weighted_out_degree_std = df["weighted_out_degree"].std()
    all_weighted_out_degree = df["weighted_out_degree"].sum()
    top_20_out_degree = df["out_degree"].head(top_n).sum()
    top_20_weighted_out_degree = df["weighted_out_degree"].head(top_n).sum()
    total_out = len(df.index)
    print("Top20 weighted_out share (%) =", 100 * top_20_out_degree / all_out_degree)
    print(
        "Top20 out share (%) =",
        100 * top_20_weighted_out_degree / all_weighted_out_degree,
    )
    print("Weighted_out std =", all_weighted_out_degree_std)
    print("Out std =", all_out_degree_std)


def analyze_metrics(file_path: str, metric: str, N: int):
    """
    Visualize and summarize a centrality metric.

    Parameters
    ----------
    file_path : str or Path
        Path to the JSON file containing centrality values.
    metric : str
        Name of the metric (used for labeling).
    top_n : int
        Number of top nodes to visualize.
    """
    with open(file_path, "r") as f:
        data = json.load(f)
    subreddits = data.keys()
    values = data.values()
    df = pd.DataFrame(data={"subreddits": subreddits, f"{metric}": values})
    selected = df.head(N).sort_values(by=[metric])
    fig = px.bar(
        selected,
        x="subreddits",
        y=f"{metric}",
        title=f"Top {N} nodes - {metric}",
    )

    fig.update_layout(
        height=600,
        bargap=0.2,
        margin=dict(t=60, b=160),
    )
    fig.write_image(f"{file_path.replace('json',f'png')}")
    stats = {
        "mean": float(df[metric].mean()),
        "std": float(df[metric].std(ddof=0)),
        "min": float(df[metric].min()),
        "max": float(df[metric].max()),
    }
    with open(
        f"{file_path.replace('.json',f'_stats.json')}",
        "w",
    ) as f:
        json.dump(stats, f, indent=4)

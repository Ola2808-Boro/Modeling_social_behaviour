import json
from collections import Counter

import networkx as nx
import pandas as pd
import plotly.graph_objects as go
from network import read_net
from utils import MAPPING_FILE_PATH, read_data


def detect_louvain_communities():
    """
    Detect communities in the positive-interaction network using
    the Louvain modularity optimization algorithm.

    The function:
    - identifies communities,
    - saves their compositions,
    - analyzes the distribution of community sizes,
    - generates a histogram of community size frequencies.

    Parameters
    ----------
    seed : int
        Random seed for Louvain algorithm reproducibility.

    Outputs
    -------
    - JSON with community compositions
    - PNG histogram of community size distribution
    """
    net = read_net(type="positive")
    mapping_data = read_data(path=MAPPING_FILE_PATH)
    communities = nx.community.louvain_communities(net, seed=123)
    id_to_name = {v: k for k, v in mapping_data.items()}
    communities_dict = {}
    counts = []
    for idx, community in enumerate(communities):
        size_od_community = len(community)
        data = {
            "count": size_od_community,
            "names": [id_to_name[int(node_id)] for node_id in community],
        }
        counts.append(len(community))
        communities_dict[idx] = data
    with open("data/community/louvain_communities.json", "w") as f:
        json.dump(communities_dict, f, indent=2)
    df = pd.DataFrame(list(Counter(counts).items()), columns=["value", "count"])
    df = df.sort_values("value").reset_index(drop=True)
    df["xstr"] = df["value"].astype(str)
    max_idx = int(df["count"].idxmax())
    colors = ["crimson" if i == max_idx else "steelblue" for i in df.index]
    fig = go.Figure(
        data=go.Bar(
            x=df["xstr"],
            y=df["count"],
            marker_color=colors,
            text=df["count"],
            textposition="outside",
            textfont=dict(size=14, family="Arial", color="black"),
        )
    )
    fig.update_traces(cliponaxis=False)
    fig.update_layout(
        title="Distribution of community sizes - Louvain algorithm",
        xaxis_title="size of community",
        yaxis_title="count",
        xaxis=dict(type="category", tickangle=-45),
    )

    fig.add_annotation(
        x=df.loc[max_idx, "xstr"],
        y=df.loc[max_idx, "count"],
        text=f"MAX = {df.loc[max_idx,'count']}",
        showarrow=True,
        arrowhead=2,
        ax=10,
        ay=-30,
        font=dict(size=14, color="black"),
        bgcolor="rgba(255,255,255,0.8)",
    )
    y_max = df["count"].max()
    fig.update_yaxes(range=[0, y_max * 1.15])
    fig.write_image(f"data/community/louvain_communities_histogram.png")

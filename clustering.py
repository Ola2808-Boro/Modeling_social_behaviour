import json
from collections import Counter

import pandas as pd
import plotly.graph_objects as go
from sklearn.cluster import HDBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from utils import read_data


def clustering_HDBSCAN():
    """
    Perform HDBSCAN clustering on feature vectors stored in the PROPERTIES column.

    Steps:
    - parse feature vectors
    - standardize features
    - reduce dimensionality using PCA
    - apply HDBSCAN clustering

    Outputs:
    - clustering_data.csv
    """
    data = read_data(
        path=r"data\all_data.csv",
        sep=",",
    )
    text = data["PROPERTIES"]
    converted_data = []
    hdb = HDBSCAN(copy=True, min_cluster_size=20)
    for properties in text:
        values = [float(x) for x in properties.split(",")]
        converted_data.append(values)
    data["converted_text"] = converted_data
    data.to_csv(r"data\clustering\clustering_converted.csv")
    X_scaled = StandardScaler().fit_transform(converted_data)
    X_pca = PCA(n_components=20, random_state=42).fit_transform(X_scaled)
    hdb.fit(X_pca)
    labels = hdb.labels_
    data["cluster_HDBSCAN_labels"] = labels
    data.to_csv(
        r"C:\Users\olkab\Desktop\Modeling of social behaviour\Modeling_social_behaviour\data\clustering\clustering_data.csv"
    )
    print("Clusters:", set(labels))
    print("Noise count (-1):", (labels == -1).sum())


def analyze_cluster(top_k: int):
    """
    Analyze cluster size distribution and noise produced by HDBSCAN.
    """
    data = pd.read_csv(
        r"C:\Users\olkab\Desktop\Modeling of social behaviour\Modeling_social_behaviour\data\clustering\clustering_data.csv"
    )
    noises = data[data["cluster_HDBSCAN_labels"] == -1]
    data = data[data["cluster_HDBSCAN_labels"] != -1]
    df = pd.DataFrame(
        list(Counter(data["cluster_HDBSCAN_labels"].values).items()),
        columns=["value", "count"],
    )
    df = df.sort_values("value", ascending=False).reset_index(drop=True)
    df_others = df.sort_values("value", ascending=True).head(
        len(list(df.index)) - top_k
    )
    df = df.head(top_k)
    df.loc[-1] = ["others", df_others["count"].sum()]
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
        title="Distribution of clusters - HDBSCAN algorithm",
        xaxis_title="id of cluster",
        yaxis_title="number of members in the cluster",
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
        font=dict(size=12, color="black"),
        bgcolor="rgba(255,255,255,0.8)",
    )
    y_max = df["count"].max()
    fig.update_yaxes(range=[0, y_max * 1.15])
    fig.write_image(f"data/clustering/HDBSCAN_histogram.png")
    with open("data/clustering/HDBSCAN_noises.txt", "w") as f:
        f.write(f"Number of nosies: {len(list(noises.index))}")


def compare_cluster_community_result():
    """
    Compare largest HDBSCAN cluster with largest Louvain community
    and analyze overlap between noise points.
    """
    data = pd.read_csv(r"data\clustering\clustering_data.csv")
    noises_cluster = data[data["cluster_HDBSCAN_labels"] == -1]["SOURCE_SUBREDDIT"]
    noises_community = []
    community_the_largest_idx = 0
    community_max = 0
    with open("data/community/louvain_communities.json", "r") as f:
        community_data = json.load(f)
    for idx, community in community_data.items():
        if community_max < community["count"]:
            community_max = community["count"]
            community_the_largest_idx = idx
        if community["count"] == 1:
            for name in community["names"]:
                noises_community.append(name)
    noises_cluster_set = set(noises_cluster)
    noises_community_set = set(noises_community)
    a = noises_cluster_set.intersection(noises_community_set)
    b = noises_community_set.intersection(noises_cluster_set)
    data = data[data["cluster_HDBSCAN_labels"] != -1]
    df = pd.DataFrame(
        list(Counter(data["cluster_HDBSCAN_labels"].values).items()),
        columns=["value", "count"],
    )
    cluster_the_largest_idx = df.sort_values("count", ascending=False)["value"].values[
        0
    ]
    cluster_the_largest = data[
        data["cluster_HDBSCAN_labels"] == cluster_the_largest_idx
    ]["SOURCE_SUBREDDIT"]
    community_the_largest = community_data[community_the_largest_idx]["names"]
    the_largest_cluster_set = set(cluster_the_largest.values)
    the_largest_community_set = set(community_the_largest)
    x = the_largest_cluster_set.intersection(the_largest_community_set)
    y = the_largest_community_set.intersection(the_largest_cluster_set)
    with open("data/clustering/general_knowledge.txt", "w") as f:
        f.write(
            f"the_largest_community_set: {len(the_largest_community_set)}\nthe_largest_cluster_set: {len(cluster_the_largest.values)}\nthe_largest_cluster_set in the_largest_community_set: {len(y)}\nthe_largest_community_set in the_largest_cluster_set : {len(x)}\nnoises_cluster: {len(list(noises_cluster.index))}\nnoises_community:{len(noises_community)}\nnoises_community in  noises_cluster: {len(a)}\nnoises_cluster in noises_community: {len(b)}"
        )

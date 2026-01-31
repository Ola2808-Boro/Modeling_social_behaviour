import json

import networkx as nx
import numpy as np
import pandas as pd
from pyvis.network import Network
from utils import BODY_DATA_PATH, PREPROCESSED_DATA_PATH, TITLE_DATA_PATH, read_data


def generate_network(sample_size: int = 1000):
    """
    Generate an interactive network visualization from the preprocessed dataset.

    The function loads the preprocessed CSV file, extracts source–target pairs,
    builds a graph using PyVis, and assigns edge weights based on the link label.

    Parameters
    ----------
    sample_size : int, optional
        Number of rows to use from the dataset (default: 1000).

    Notes
    -----
    The resulting network object can be exported to HTML using:
        net.write_html("network.html")
    """
    data = pd.read_csv(PREPROCESSED_DATA_PATH, header=0).head(sample_size).to_numpy()
    net = Network(
        height="750px",
        width="100%",
        bgcolor="#222222",
        font_color="white",
        select_menu=True,
    )
    source = np.unique(data[:, 0])
    target = np.unique(data[:, 1])
    unique_nodes = np.unique(np.concatenate((source, target), axis=None))
    for node in unique_nodes:
        net.add_node(node, label=node)
    for row in data:
        net.add_edge(row[0], row[1], value=row[-1])
    net.write_html("nx.html")


def prepare_dataset_for_analysis(vocabulary: dict, data: pd.DataFrame) -> pd.DataFrame:
    """
    Map subreddit names to integer identifiers and create a preprocessed dataset.

    Parameters
    ----------
    vocabulary : dict
        Mapping from subreddit names to integer IDs.
    data : pandas.DataFrame
        Raw dataset containing SOURCE_SUBREDDIT, TARGET_SUBREDDIT,
        TIMESTAMP and LINK_SENTIMENT columns.
    out_path : Path or str, optional
        Output path for the preprocessed CSV file.

    Returns
    -------
    pandas.DataFrame
        Preprocessed dataset with mapped identifiers.
    """
    mapped_data = pd.DataFrame(
        {
            "Source": data["SOURCE_SUBREDDIT"].map(vocabulary),
            "Target": data["TARGET_SUBREDDIT"].map(vocabulary),
            "Timestamp": data["TIMESTAMP"],
            "Label": data["LINK_SENTIMENT"],
        }
    )
    mapped_data.to_csv(f"preprocessed_data.csv", index=False)
    return mapped_data


def mapping_data(path: str):
    """
    Create a mapping from subreddit names to integer identifiers.

    The function extracts all unique source and target subreddits from the raw
    dataset, assigns a unique integer ID to each, saves the mapping to a JSON
    file, and generates a preprocessed CSV dataset.

    Parameters
    ----------
    path : Path or str
        Path to the raw input CSV file.
    """
    data = pd.read_csv(path, sep=",", header=0)
    print(data.columns)
    source = set(data["SOURCE_SUBREDDIT"].values)
    target = set(data["TARGET_SUBREDDIT"].values)
    unique_source_target = set(source.union(target))
    vocabulary = {word: idx for idx, word in enumerate(unique_source_target)}
    prepare_dataset_for_analysis(vocabulary=vocabulary, data=data)
    with open(f"mapping.json", "w") as f:
        json.dump(vocabulary, f, indent=1)


def union_data():
    """
    Merge title-based and body-based Reddit hyperlink datasets.

    The function loads two TSV files, concatenates them row-wise,
    and saves the unified dataset to a CSV file.

    Parameters
    ----------
    path_title : Path or str
        Path to the title-based hyperlinks dataset.
    path_body : Path or str
        Path to the body-based hyperlinks dataset.
    out_path : Path or str
        Output path for the merged CSV file.
    """
    df_title = read_data(path=TITLE_DATA_PATH, sep="\t")
    df_body = read_data(path=BODY_DATA_PATH, sep="\t")
    concat_df = pd.concat([df_title, df_body])
    concat_df.to_csv("all_data.csv")

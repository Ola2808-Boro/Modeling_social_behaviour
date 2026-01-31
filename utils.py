import json
import logging
from pathlib import Path

import pandas as pd

ROOT_PATH = Path(__file__).resolve().parent / "data"
MAPPING_FILE_PATH = ROOT_PATH / "mapping.json"
PREPROCESSED_DATA_PATH = ROOT_PATH / "preprocessed_data.csv"
BODY_DATA_PATH = ROOT_PATH / "soc-redditHyperlinks-body.tsv"
TITLE_DATA_PATH = ROOT_PATH / "soc-redditHyperlinks-title.tsv"
ACTIVITIES_DATA_PATH = ROOT_PATH / "activities"
NET_PATH = ROOT_PATH / "net.graphml"
NET_POSITIVE_PATH = ROOT_PATH / "net_undirected_positive_edges.graphml"

logging.basicConfig(
    filename="app.log",
    encoding="utf-8",
    filemode="a",
    format="{asctime} - {levelname} - {message}",
    style="{",
    datefmt="%Y-%m-%d %H:%M",
)


def get_all_keys_by_value(mapping: dict, value: int) -> list:
    """
    Retrieve all keys from a dictionary that correspond to a given value.

    Parameters
    ----------
    mapping : dict
        Dictionary mapping keys to values.
    value : int
        Value to search for.

    Returns
    -------
    list
        List of keys whose associated value equals the given value.
    """
    return [k for k, v in mapping.items() if v == value]


def read_data(path: str, **kwargs) -> pd.DataFrame | dict:
    """
    Read data from a file and return it as a pandas DataFrame or a dictionary.

    Supported formats:
        - CSV (.csv)
        - TSV (.tsv)
        - JSON (.json)

    Parameters
    ----------
    path : Path or str
        Path to the input file.
    **kwargs
        Additional keyword arguments passed to pandas.read_csv
        (e.g. sep=',', sep='\\t').

    Returns
    -------
    pandas.DataFrame or dict
        Loaded dataset.

    Raises
    ------
    ValueError
        If the file format is unsupported.
    """
    if ".csv" in path or ".tsv" in path:
        data = pd.read_csv(path, sep=kwargs["sep"], header=0)
    elif ".json" in path:
        with open(path, "r") as f:
            data = json.load(f)
    else:
        logging.error(f"Unsupported file format: {path}")
    return data

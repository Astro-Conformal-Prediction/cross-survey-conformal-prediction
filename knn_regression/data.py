"""Data loading and preprocessing for cross-survey embedding alignment."""

import numpy as np
from datasets import load_dataset


def data_loader(dataset, subset):
    """Load a survey pair and return (train, calibration, test) splits.

    Standardizes column names, drops missing/NaN/zero-norm rows, and
    L2-normalizes both source and target embeddings.
    """
    ds = load_dataset(dataset, subset)

    # standardize column names
    cols = ds[next(iter(ds))].column_names
    ds = ds.rename_columns(dict(zip(cols[1:], ["source", "target"])))

    # convert to numeric arrays
    ds = ds.map(
        lambda x: {"source": np.array(x["source"]), "target": np.array(x["target"])}
    )

    # drop missing, NaN, and zero-norm rows
    ds = ds.filter(lambda x: all(v is not None for v in x.values()))
    ds = ds.filter(
        lambda x: not np.isnan(x["source"]).any() and not np.isnan(x["target"]).any()
    )
    ds = ds.filter(
        lambda x: np.linalg.norm(x["source"]) > 0 and np.linalg.norm(x["target"]) > 0
    )

    # L2 normalize
    ds = ds.map(
        lambda x: {
            "source": x["source"] / np.linalg.norm(x["source"]),
            "target": x["target"] / np.linalg.norm(x["target"]),
        }
    )

    return ds["train"], ds["calibration"], ds["test"]


def extract_columns(ds):
    """Return (source, target) as (n, d) float arrays."""
    return np.array(ds["source"]), np.array(ds["target"])

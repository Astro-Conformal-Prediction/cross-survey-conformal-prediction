"""
dataloader.py
Loads cross-survey embedding data from HuggingFace, matching the pattern
used in ridge-regression/cross-survey-translation/src/dataloader.py.

The HF datasets are parquet-backed with columns like:
    object_id, {config}_hsc, {config}_jwst, split, ...

load_data returns a dict with keys: "train", "calibration", "test"
(and "full" when import_splits=False, though for Exp 4 we always use splits).
"""

from __future__ import annotations

import numpy as np
from datasets import load_dataset


def load_data(
    repo_id: str,
    config_name: str,
    source_embed: str,
    target_embed: str,
    normalize: bool = True,
    import_splits: bool = True,
) -> dict:
    """
    Load embeddings from a HuggingFace dataset repo.

    Parameters
    ----------
    repo_id       : HF repo, e.g. "astro-conformal-project/split_jwst_hsc_embeddings"
    config_name   : HF config / subset name, e.g. "astropt_850m"
    source_embed  : column name for source embeddings, e.g. "astropt_850m_hsc"
    target_embed  : column name for target embeddings, e.g. "astropt_850m_jwst"
    normalize     : L2-normalise each embedding vector
    import_splits : if True, load train/calibration/test splits from the dataset;
                    if False, return the full dataset under key "full"

    Returns
    -------
    dict with keys "train", "calibration", "test"  (DataFrames)
    or            "full"                            (DataFrame, if import_splits=False)
    """
    ds = load_dataset(repo_id, config_name, trust_remote_code=True)

    def _to_df(split_name: str):
        split = ds[split_name]
        cols = [c for c in [source_embed, target_embed, "object_id"]
                if c in split.column_names]
        df = split.select_columns(cols).to_pandas()

        if normalize:
            for col in [source_embed, target_embed]:
                if col in df.columns:
                    vecs = np.stack(df[col].values)
                    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
                    norms = np.maximum(norms, 1e-8)
                    df[col] = list(vecs / norms)

        return df

    if import_splits:
        return {
            "train":       _to_df("train"),
            "calibration": _to_df("calibration"),
            "test":        _to_df("test"),
        }
    else:
        # Concatenate all splits into one DataFrame
        import pandas as pd
        frames = [_to_df(s) for s in ds.keys()]
        return {"full": pd.concat(frames, ignore_index=True)}


def df_to_arrays(
    df,
    source_col: str,
    target_col: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract stacked numpy arrays from a DataFrame."""
    X = np.stack([np.asarray(e) for e in df[source_col]])
    Y = np.stack([np.asarray(e) for e in df[target_col]])
    return X.astype(np.float32), Y.astype(np.float32)

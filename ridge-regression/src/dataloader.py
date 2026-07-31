"""
dataloader.py

This file handles dataset loading and preprocessing: 

    1. Load a specific Hugging Face cross-matched embedding dataset/config 
    2. Convert HF split to dataframe format
    3. L2-normalize embeddings 
    4. Merge the splits if k-fold split on runtime. 
"""

import numpy as np
import pandas as pd
from datasets import load_dataset

default_dataset = "astro-conformal-project/split_sdss_hsc_embeddings"

def l2_normalize(x):
    arr = np.asarray(x)
    norm = np.linalg.norm(arr)  

    if norm < 1e-8:
        return arr.tolist()
    return (arr / norm).astype(np.float32).tolist()

def load_embeds(
    repo_id: str = default_dataset,
    config_name: str = None,
    source_embed: str = None,
    target_embed: str = None):
    """
    Load a Hugging Face dataset config and convert all available splits to pandas DF
    """
    dataset = load_dataset(repo_id, config_name)
    dfs = {split: dataset[split].to_pandas() for split in dataset.keys()}

    for split_name, df in dfs.items():
        missing_cols = [
            col for col in [source_embed, target_embed]
            if col not in df.columns
        ]
        if missing_cols:
            raise ValueError(
                f"Missing columns in split '{split_name}': {missing_cols}\n"
                f"Available columns: {list(df.columns)}"
            )
    return dfs # dicts of Dataframes


def preprocess_splits(
    dfs,
    source_embed: str,
    target_embed: str,
    normalize: bool = True):

    for split_name, df in dfs.items():
        if normalize:
            df[source_embed] = df[source_embed].apply(l2_normalize)
            df[target_embed] = df[target_embed].apply(l2_normalize)
        else:
            df[source_embed] = df[source_embed].apply(
                lambda x: np.asarray(x, dtype=np.float32).tolist()
            )
            df[target_embed] = df[target_embed].apply(
                lambda x: np.asarray(x, dtype=np.float32).tolist()
            )
        dfs[split_name] = df.reset_index(drop=True)
    return dfs 


# merging the existing splits for K-fold on runtime: 
def merge_splits(dfs):
    split_names = set(dfs.keys())
    has_train = "train" in split_names
    has_test = "test" in split_names
    has_calibration = "calibration" in split_names

    if has_train and has_test and has_calibration:
        merged_train_df = pd.concat(
            [
                dfs["train"],
                dfs["test"],
                dfs["calibration"]
            ],
            axis=0,
            ignore_index=True
        )

        return {
            "full": merged_train_df.reset_index(drop=True)
        }

    return dfs


def load_data(
    repo_id: str,
    config_name: str,
    source_embed: str,
    target_embed: str,
    normalize: bool = True,
    import_splits: bool = True):

    dfs = load_embeds(
        repo_id=repo_id,
        config_name=config_name,
        source_embed=source_embed,
        target_embed=target_embed)

    dfs = preprocess_splits(
        dfs=dfs,
        source_embed=source_embed,
        target_embed=target_embed,
        normalize=normalize)

    if not import_splits:
        dfs = merge_splits(dfs)
        return dfs
    else:
        return dfs



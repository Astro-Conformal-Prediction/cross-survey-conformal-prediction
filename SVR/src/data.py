
import os

import numpy as np
from datasets import load_dataset
from sklearn.preprocessing import normalize

from src.constants import SURVEY_CONFIGS


def load_embedding_pair(survey, model_name, split, hf_token=None):
    
    if survey not in SURVEY_CONFIGS:
        raise ValueError(f"Unknown survey '{survey}'. Choose from {list(SURVEY_CONFIGS)}")

    cfg = SURVEY_CONFIGS[survey]
    token = hf_token or os.environ.get("HF_TOKEN")
    if cfg["requires_token"] and not token:
        raise RuntimeError(
            f"Survey '{survey}' needs a Hugging Face token. Set the HF_TOKEN "
            "environment variable or pass it as hf_token to load_embedding_pair()"
        )

    if cfg["streaming"]:
        dataset = load_dataset(
            cfg["hf_repo"], model_name, split=split,
            streaming=True, token=token,
        )
        columns = list(dataset.features.keys())
        source_col = columns[cfg["source_col"]]
        target_col = columns[cfg["target_col"]]
        

        X_list, Y_list = [], []
        for example in dataset:
            X_list.append(example[source_col])
            Y_list.append(example[target_col])
            
        X_stack = np.stack(X_list)
        Y_stack = np.stack(Y_list)
    else:
        dataset = load_dataset(cfg["hf_repo"], model_name, split=split, token=token)
        data = dataset.to_pandas()
        source_col = data.columns[cfg["source_col"]]
        target_col = data.columns[cfg["target_col"]]
        X_stack = np.stack(data[source_col].to_list())
        Y_stack = np.stack(data[target_col].to_list())
       

    X = normalize(X_stack, norm="l2", axis=1)
    Y = normalize(Y_stack, norm="l2", axis=1)
    return X, Y, source_col, target_col

def load_embedding_pair_with_objects(survey, model_name, split, hf_token=None):
    
    if survey not in SURVEY_CONFIGS:
        raise ValueError(f"Unknown survey '{survey}'. Choose from {list(SURVEY_CONFIGS)}")

    cfg = SURVEY_CONFIGS[survey]
    token = hf_token or os.environ.get("HF_TOKEN")
    if cfg["requires_token"] and not token:
        raise RuntimeError(
            f"Survey '{survey}' needs a Hugging Face token. Set the HF_TOKEN "
            "environment variable or pass it as hf_token to load_embedding_pair()"
        )

    else:
        dataset = load_dataset(
            cfg["hf_repo"], model_name, split=split,
            streaming=True, token=token,
        )
        columns = list(dataset.features.keys())
        source_col = columns[cfg["source_col"]]
        target_col = columns[cfg["target_col"]]
        id_col = columns[0]
        

        X_list, Y_list, id = [], [], []
        for example in dataset:
            X_list.append(example[source_col])
            Y_list.append(example[target_col])
            id.append(example[id_col])

        X_stack = np.stack(X_list)
        Y_stack = np.stack(Y_list)

    X = normalize(X_stack, norm="l2", axis=1)
    Y = normalize(Y_stack, norm="l2", axis=1)
    ids = np.array(id)
    return X, Y, ids, source_col, target_col


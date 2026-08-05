"""
local_support.py
Step 9 – local-support features for the difficulty model.
Step 10 – kNN-local calibration baseline.

These are kept separate so the basic adaptive CP pipeline (Steps 1–8) can be
run without touching this file.
"""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

from adaptive_conformal import conformal_quantile


# ---------------------------------------------------------------------------
# Step 9 – Support features
# ---------------------------------------------------------------------------

def build_support_features(
    X_query: np.ndarray,
    X_train: np.ndarray,
    n_neighbors: int = 20,
    pca_components: int | None = 64,
) -> np.ndarray:
    """
    Compute local-support features for each query object.

    Features
    --------
    - distance to nearest training neighbour (1-NN)
    - mean distance to 5 nearest neighbours
    - mean distance to 20 nearest neighbours

    If pca_components is given, PCA is applied to X_train/X_query first
    (useful when the source embedding dimension is very large).

    Parameters
    ----------
    X_query    : (n_query, d) embeddings for which we want support info
    X_train    : (n_train, d) translator-training embeddings
    n_neighbors: how many neighbours to retrieve (>= 20)
    pca_components: reduce to this many dims before kNN; None = no PCA

    Returns
    -------
    features : (n_query, 3) array  [dist_1nn, mean_5nn, mean_20nn]
    """
    if pca_components is not None and pca_components < X_train.shape[1]:
        pca = PCA(n_components=pca_components, random_state=42)
        X_train = pca.fit_transform(X_train)
        X_query = pca.transform(X_query)

    k = max(n_neighbors, 20)
    nn = NearestNeighbors(n_neighbors=k, metric="euclidean", n_jobs=-1)
    nn.fit(X_train)
    distances, _ = nn.kneighbors(X_query)  # (n_query, k)

    dist_1nn = distances[:, 0]
    mean_5nn = distances[:, :5].mean(axis=1)
    mean_20nn = distances[:, :20].mean(axis=1)

    return np.column_stack([dist_1nn, mean_5nn, mean_20nn])


def augment_with_support(
    X_embeddings: np.ndarray,
    support_features: np.ndarray,
    pca_components: int | None = 64,
) -> np.ndarray:
    """
    Concatenate (optionally PCA-reduced) source embeddings with
    local-support features for the difficulty model.

    Parameters
    ----------
    X_embeddings     : (n, d) raw source embeddings
    support_features : (n, 3) output of build_support_features
    pca_components   : if given, embed the source embeddings first

    Returns
    -------
    X_aug : (n, pca_components + 3) or (n, d + 3)
    """
    if pca_components is not None and pca_components < X_embeddings.shape[1]:
        pca = PCA(n_components=pca_components, random_state=42)
        X_reduced = pca.fit_transform(X_embeddings)
    else:
        X_reduced = X_embeddings

    return np.column_stack([X_reduced, support_features])


# ---------------------------------------------------------------------------
# Step 10 – kNN-local calibration baseline
# ---------------------------------------------------------------------------

def local_conformal_radii(
    X_test: np.ndarray,
    X_cal: np.ndarray,
    r_cal: np.ndarray,
    alpha: float,
    k: int = 50,
    pca_components: int | None = 64,
) -> np.ndarray:
    """
    kNN-local calibration baseline (empirical, no formal guarantee).

    For each test object, find its k nearest calibration neighbours in
    source-embedding space and compute the conformal quantile from those
    local residuals only.

    Parameters
    ----------
    X_test         : (n_test, d) test source embeddings
    X_cal          : (n_cal, d)  calibration source embeddings
    r_cal          : (n_cal,)    calibration residuals
    alpha          : miscoverage rate
    k              : neighbourhood size
    pca_components : optional PCA before kNN

    Returns
    -------
    local_radii : (n_test,) per-object radii

    Notes
    -----
    This is a *heuristic baseline*, not a formally valid localized conformal
    predictor.  Formal localized CP requires a careful weighting scheme.
    Label this in results as 'kNN-local calibration baseline'.
    """
    if pca_components is not None and pca_components < X_cal.shape[1]:
        pca = PCA(n_components=pca_components, random_state=42)
        X_cal_r = pca.fit_transform(X_cal)
        X_test_r = pca.transform(X_test)
    else:
        X_cal_r = X_cal
        X_test_r = X_test

    nn = NearestNeighbors(n_neighbors=k, metric="euclidean", n_jobs=-1)
    nn.fit(X_cal_r)
    _, idx = nn.kneighbors(X_test_r)  # (n_test, k)

    local_radii = np.empty(len(X_test), dtype=float)
    for i, neighbours in enumerate(idx):
        local_scores = r_cal[neighbours]
        local_radii[i] = conformal_quantile(local_scores, alpha)

    return local_radii

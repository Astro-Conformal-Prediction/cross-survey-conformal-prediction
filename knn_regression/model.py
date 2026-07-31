"""k-NN translator with cosine-distance hyperparameter tuning."""

import numpy as np
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor

from data import extract_columns


def mean_cosine_distance(Y_true, Y_pred):
    """Mean cosine distance between true and predicted embeddings."""
    Y_true = Y_true / np.linalg.norm(Y_true, axis=1, keepdims=True)
    Y_pred = Y_pred / np.linalg.norm(Y_pred, axis=1, keepdims=True)
    return (1 - (Y_true * Y_pred).sum(axis=1)).mean()


def mean_cosine_scorer(scorer):
    """Wrap a distance function as a sklearn scorer (lower is better)."""
    return make_scorer(scorer, greater_is_better=False)


def grid_search_knn(train_ds, param_grid, cv=5, n_jobs=-1):
    """Grid search over k on the training split. Returns the fitted GridSearchCV."""
    scoring_function = mean_cosine_scorer(mean_cosine_distance)

    grid = GridSearchCV(
        KNeighborsRegressor(weights="distance", metric="cosine"),
        param_grid,
        cv=cv,
        scoring=scoring_function,
        n_jobs=n_jobs,
    )

    X_train, Y_train = extract_columns(train_ds)
    grid.fit(X_train, Y_train)

    return grid

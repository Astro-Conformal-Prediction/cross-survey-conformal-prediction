"""
difficulty_model.py
Learns to predict translator residual magnitude from source embeddings.

Targets: log(r_i + eps)   →  exp(prediction) = sigma_hat(x_i)

Modelling in log-space keeps sigma_hat positive and handles the heavy right
tail of residual distributions.  This matches the residual-normalised score
construction described in MAPIE's documentation.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import RegressorMixin
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor


# ---------------------------------------------------------------------------
# Difficulty model wrapper
# ---------------------------------------------------------------------------

class ResidualDifficultyModel:
    """
    Predicts translator residual scale from source embedding.

    Usage
    -----
    dm = ResidualDifficultyModel()
    dm.fit(X_diff_train, r_diff)           # r_diff are raw (positive) residuals
    sigma_hat = dm.predict(X_test)         # returns exp(log-prediction)
    """

    def __init__(
        self,
        estimator: RegressorMixin | None = None,
        eps: float = 1e-6,
    ) -> None:
        self.eps = eps
        self.estimator: RegressorMixin = estimator or HistGradientBoostingRegressor(
            max_depth=4,
            learning_rate=0.05,
            max_iter=200,
            random_state=42,
        )

    def fit(
        self,
        x: np.ndarray,
        residuals: np.ndarray,
    ) -> "ResidualDifficultyModel":
        """
        Fit the difficulty model.

        Parameters
        ----------
        x         : (n, d_source) source embeddings
        residuals : (n,) raw (positive) residuals from the translator
        """
        log_residuals = np.log(np.maximum(residuals, self.eps))
        self.estimator.fit(x, log_residuals)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Return predicted residual scale (sigma_hat), always >= eps.

        Parameters
        ----------
        x : (n, d_source) source embeddings

        Returns
        -------
        sigma_hat : (n,) positive floats
        """
        log_scale = self.estimator.predict(x)
        scale = np.exp(log_scale)
        return np.maximum(scale, self.eps)


# ---------------------------------------------------------------------------
# Convenience constructors for the three difficulty models in the spec
# ---------------------------------------------------------------------------

def make_difficulty_model(kind: str = "hgbt", **kwargs) -> ResidualDifficultyModel:
    """
    Parameters
    ----------
    kind : one of "ridge", "hgbt", "knn"
    """
    if kind == "ridge":
        est = Ridge(alpha=kwargs.get("alpha", 1.0))
    elif kind == "hgbt":
        est = HistGradientBoostingRegressor(
            max_depth=kwargs.get("max_depth", 4),
            learning_rate=kwargs.get("learning_rate", 0.05),
            max_iter=kwargs.get("max_iter", 200),
            random_state=42,
        )
    elif kind == "knn":
        est = KNeighborsRegressor(
            n_neighbors=kwargs.get("n_neighbors", 10),
            metric=kwargs.get("metric", "euclidean"),
        )
    else:
        raise ValueError(f"Unknown difficulty model kind: {kind!r}")
    return ResidualDifficultyModel(estimator=est)

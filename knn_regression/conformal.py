"""Split conformal prediction: nonconformity scores, quantile, coverage."""

import numpy as np

from data import extract_columns


def compute_residuals(ds, model):
    """Return the (n,) array of L2 nonconformity scores.

    Predictions are L2-normalized so both sides lie on the unit sphere.
    """
    source, target = extract_columns(ds)
    target_pred = model.predict(source)
    target_pred = target_pred / np.linalg.norm(target_pred, axis=1, keepdims=True)
    return np.linalg.norm(target_pred - target, axis=1)


def compute_conformal_quantile(scores, coverage=0.90):
    """The ceil((n+1) * coverage)-th smallest calibration score."""
    n = len(scores)
    level = np.ceil((n + 1) * coverage) / n
    if level > 1:
        return np.inf  # too few calibration points for this coverage
    return np.quantile(scores, level, method="inverted_cdf")


def compute_empirical_coverage(scores, quantile):
    """Fraction of test objects whose score falls inside the conformal region."""
    return (scores <= quantile).mean()


def compute_coverage_error(empirical_coverage, target_coverage):
    return abs(empirical_coverage - target_coverage)

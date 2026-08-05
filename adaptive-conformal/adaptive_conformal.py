"""
adaptive_conformal.py
Core utilities for adaptive (residual-normalised) conformal prediction.

Reuses the same residual definition as Experiment 3 / conformal-prediction-analysis:
    r_i = ||y_i - y_hat_i||_2 / sqrt(d)
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Residuals
# ---------------------------------------------------------------------------

def vector_residual(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    normalize_dim: bool = True,
) -> np.ndarray:
    """
    Compute per-object L2 residuals, optionally normalised by sqrt(d).

    Parameters
    ----------
    y_true : (n, d) array
    y_pred : (n, d) array
    normalize_dim : bool
        If True, divide by sqrt(d) to make residuals dimension-independent.

    Returns
    -------
    residuals : (n,) array of non-negative floats
    """
    diff = y_true - y_pred
    residual = np.linalg.norm(diff, axis=1)
    if normalize_dim:
        residual = residual / np.sqrt(y_true.shape[1])
    return residual


# ---------------------------------------------------------------------------
# Conformal quantile (finite-sample corrected)
# ---------------------------------------------------------------------------

def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """
    Finite-sample–corrected conformal quantile.

    Uses the ceil((n+1)(1-alpha))/n level, clamped at 1.0.

    Parameters
    ----------
    scores : (n,) calibration scores
    alpha  : miscoverage rate in (0, 1)

    Returns
    -------
    q : float
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0,1), got {alpha}")

    scores = np.asarray(scores, dtype=float)
    if len(scores) == 0:
        raise ValueError("calibration scores are empty")

    n = len(scores)
    level = np.ceil((n + 1) * (1 - alpha)) / n
    level = min(level, 1.0)
    return float(np.quantile(scores, level, method="higher"))


# ---------------------------------------------------------------------------
# Adaptive (normalised) scores and radii
# ---------------------------------------------------------------------------

def adaptive_scores(
    residuals: np.ndarray,
    predicted_scale: np.ndarray,
    eps: float = 1e-6,
) -> np.ndarray:
    """
    Normalised conformity scores:  a_i = r_i / sigma_hat(x_i).

    Parameters
    ----------
    residuals       : (n,) raw residuals on calibration set
    predicted_scale : (n,) difficulty estimates (sigma_hat)

    Returns
    -------
    scores : (n,) normalised scores
    """
    scale = np.maximum(predicted_scale, eps)
    return residuals / scale


def adaptive_radii(
    quantile: float,
    predicted_scale: np.ndarray,
    eps: float = 1e-6,
) -> np.ndarray:
    """
    Per-object prediction radii:  R_i = q * sigma_hat(x_i).

    Parameters
    ----------
    quantile        : adaptive conformal quantile q
    predicted_scale : (n,) difficulty estimates on the test set

    Returns
    -------
    radii : (n,) positive floats
    """
    scale = np.maximum(predicted_scale, eps)
    return quantile * scale


# ---------------------------------------------------------------------------
# Coverage helpers
# ---------------------------------------------------------------------------

def compute_coverage(
    residuals: np.ndarray,
    radii: np.ndarray,
) -> dict[str, float | np.ndarray]:
    """
    Return coverage statistics.

    Parameters
    ----------
    residuals : (n,) actual residuals
    radii     : (n,) per-object radii

    Returns
    -------
    dict with keys: covered (bool array), empirical_coverage (float)
    """
    covered = residuals <= radii
    return {
        "covered": covered,
        "empirical_coverage": float(covered.mean()),
    }


def coverage_by_quartile(
    residuals: np.ndarray,
    predicted_difficulty: np.ndarray,
    global_radii: np.ndarray,
    adaptive_radii_arr: np.ndarray,
) -> list[dict]:
    """
    Split objects into difficulty quartiles; report coverage and mean radius
    for global CP and adaptive CP in each group.

    Returns
    -------
    List of 4 dicts, one per quartile.
    """
    quantiles = np.quantile(predicted_difficulty, [0.25, 0.50, 0.75])
    bins = np.digitize(predicted_difficulty, quantiles)  # 0,1,2,3 → Q1-Q4

    rows = []
    for q in range(4):
        mask = bins == q
        if mask.sum() == 0:
            continue
        r_q = residuals[mask]
        gr = global_radii[mask]
        ar = adaptive_radii_arr[mask]
        rows.append(
            {
                "quartile": f"Q{q + 1}",
                "n": int(mask.sum()),
                "global_coverage": float((r_q <= gr).mean()),
                "adaptive_coverage": float((r_q <= ar).mean()),
                "global_mean_radius": float(gr.mean()),
                "adaptive_mean_radius": float(ar.mean()),
            }
        )
    return rows

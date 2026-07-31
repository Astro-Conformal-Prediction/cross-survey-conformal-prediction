"""Full pipeline: fit each backbone once, evaluate at every coverage level."""

import pandas as pd

from conformal import (
    compute_conformal_quantile,
    compute_coverage_error,
    compute_empirical_coverage,
    compute_residuals,
)
from data import data_loader
from model import grid_search_knn

MODELS = [
    "astropt_15m", "astropt_850m", "astropt_95m",
    "convnext_base", "convnext_large", "convnext_nano", "convnext_tiny",
    "dino_base", "dino_giant", "dino_large", "dino_small",
    "ijepa_giant", "ijepa_huge",
    "vit_base", "vit_huge", "vit_large",
]

COVERAGES = [0.8, 0.9, 0.95]

PARAM_GRID = {"n_neighbors": [5, 7, 10, 15, 20, 30, 40, 60, 80, 100, 130]}


def fit_and_score(dataset, subset, param_grid=None):
    """Everything independent of the coverage level. Runs once per backbone."""
    train_ds, calibration_ds, test_ds = data_loader(dataset, subset)

    model = grid_search_knn(train_ds, param_grid or PARAM_GRID)

    calibration_scores = compute_residuals(calibration_ds, model)
    test_scores = compute_residuals(test_ds, model)

    return model, calibration_scores, test_scores


def conformal_region_pipeline(calibration_scores, test_scores, coverage):
    """Build the conformal region at one coverage level and evaluate it."""
    conformal_quantile = compute_conformal_quantile(
        calibration_scores, coverage=coverage
    )
    empirical_coverage = compute_empirical_coverage(test_scores, conformal_quantile)
    coverage_error = compute_coverage_error(empirical_coverage, coverage)

    return conformal_quantile, empirical_coverage, coverage_error


def full_pipeline(dataset, models=MODELS, coverages=COVERAGES, skip_errors=True):
    """Run every backbone at every coverage level. Returns a tidy DataFrame."""
    rows = []

    for subset in models:
        try:
            model, calibration_scores, test_scores = fit_and_score(dataset, subset)
        except Exception as e:
            if not skip_errors:
                raise
            print(f"[skip] {subset}: {type(e).__name__}: {e}")
            continue

        for coverage in coverages:
            quantile, empirical, error = conformal_region_pipeline(
                calibration_scores, test_scores, coverage
            )
            rows.append({
                "model": subset,
                "target_coverage": coverage,
                "best_k": model.best_params_["n_neighbors"],
                "radius": quantile,
                "empirical_coverage": empirical,
                "coverage_error": error,
                "n_calib": len(calibration_scores),
                "n_test": len(test_scores),
            })

    return pd.DataFrame(rows)

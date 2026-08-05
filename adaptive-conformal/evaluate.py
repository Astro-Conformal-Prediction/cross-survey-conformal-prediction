"""
evaluate.py
Build per-object and per-run metrics DataFrames for Experiment 4.

Designed to be called from run_adaptive_cp.py after every (backbone, survey_pair,
target_coverage) combination.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from adaptive_conformal import coverage_by_quartile


# ---------------------------------------------------------------------------
# Per-object rows
# ---------------------------------------------------------------------------

def per_object_rows(
    *,
    row_ids: np.ndarray,
    survey_pair: str,
    backbone: str,
    target_coverage: float,
    true_residual: np.ndarray,
    predicted_difficulty: np.ndarray,
    global_radius: np.ndarray,
    adaptive_radius: np.ndarray,
    nearest_neighbor_distance: np.ndarray | None = None,
    local_radius: np.ndarray | None = None,
    split_seed: int = 42,
) -> pd.DataFrame:
    """
    Build one row per test object.
    """
    n = len(row_ids)
    global_covered = true_residual <= global_radius
    adaptive_covered = true_residual <= adaptive_radius

    data: dict[str, Any] = {
        "row_id": row_ids,
        "survey_pair": survey_pair,
        "backbone": backbone,
        "target_coverage": target_coverage,
        "true_residual": true_residual,
        "predicted_difficulty": predicted_difficulty,
        "global_radius": global_radius,
        "adaptive_radius": adaptive_radius,
        "global_covered": global_covered,
        "adaptive_covered": adaptive_covered,
        "split_seed": np.full(n, split_seed),
    }

    if nearest_neighbor_distance is not None:
        data["nearest_neighbor_distance"] = nearest_neighbor_distance
    if local_radius is not None:
        data["local_radius"] = local_radius
        data["local_covered"] = true_residual <= local_radius

    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# Per-run summary row
# ---------------------------------------------------------------------------

def run_summary(
    *,
    survey_pair: str,
    backbone: str,
    difficulty_model_kind: str,
    target_coverage: float,
    true_residual: np.ndarray,
    predicted_difficulty: np.ndarray,
    global_radius: np.ndarray,
    adaptive_radius: np.ndarray,
    local_radius: np.ndarray | None = None,
) -> dict[str, Any]:
    """
    Compute summary statistics for one (backbone, survey_pair, target_coverage) run.
    """
    global_covered = true_residual <= global_radius
    adaptive_covered = true_residual <= adaptive_radius

    # Spearman correlation: predicted difficulty vs actual residual
    spear_r, spear_p = spearmanr(predicted_difficulty, true_residual)

    # Quartile coverage
    quartile_rows = coverage_by_quartile(
        true_residual,
        predicted_difficulty,
        global_radius,
        adaptive_radius,
    )
    q_cov = {f"global_coverage_q{i+1}": r["global_coverage"] for i, r in enumerate(quartile_rows)}
    q_cov.update({f"adaptive_coverage_q{i+1}": r["adaptive_coverage"] for i, r in enumerate(quartile_rows)})
    q_mean_r = {f"global_mean_radius_q{i+1}": r["global_mean_radius"] for i, r in enumerate(quartile_rows)}
    q_mean_r.update({f"adaptive_mean_radius_q{i+1}": r["adaptive_mean_radius"] for i, r in enumerate(quartile_rows)})

    row: dict[str, Any] = {
        "survey_pair": survey_pair,
        "backbone": backbone,
        "difficulty_model": difficulty_model_kind,
        "target_coverage": target_coverage,
        # Global CP
        "global_empirical_coverage": float(global_covered.mean()),
        "global_mean_radius": float(global_radius.mean()),
        "global_median_radius": float(np.median(global_radius)),
        "global_p90_radius": float(np.percentile(global_radius, 90)),
        "global_radius_std": float(global_radius.std()),
        # Adaptive CP
        "adaptive_empirical_coverage": float(adaptive_covered.mean()),
        "adaptive_mean_radius": float(adaptive_radius.mean()),
        "adaptive_median_radius": float(np.median(adaptive_radius)),
        "adaptive_p90_radius": float(np.percentile(adaptive_radius, 90)),
        "adaptive_radius_std": float(adaptive_radius.std()),
        # Diagnostics
        "spearman_difficulty_residual": float(spear_r),
        "spearman_p_value": float(spear_p),
        "radius_reduction_pct": float(
            100.0 * (global_radius.mean() - adaptive_radius.mean()) / global_radius.mean()
        ),
    }

    if local_radius is not None:
        local_covered = true_residual <= local_radius
        row.update(
            {
                "local_empirical_coverage": float(local_covered.mean()),
                "local_mean_radius": float(local_radius.mean()),
                "local_median_radius": float(np.median(local_radius)),
            }
        )

    row.update(q_cov)
    row.update(q_mean_r)
    return row


# ---------------------------------------------------------------------------
# Pretty-print comparison table (Step 7)
# ---------------------------------------------------------------------------

def print_comparison_table(summary: dict[str, Any]) -> None:
    """
    Print the 2-row comparison table from Step 7 of the spec.
    """
    tc = summary["target_coverage"]
    print(
        f"\n{'Method':<30} {'Target':>8} {'Empirical':>10} {'Mean R':>10} "
        f"{'Median R':>10} {'P90 R':>10}"
    )
    print("-" * 80)
    print(
        f"{'Global split CP':<30} {tc:>8.0%} "
        f"{summary['global_empirical_coverage']:>10.4f} "
        f"{summary['global_mean_radius']:>10.4f} "
        f"{summary['global_median_radius']:>10.4f} "
        f"{summary['global_p90_radius']:>10.4f}"
    )
    print(
        f"{'Adaptive residual CP':<30} {tc:>8.0%} "
        f"{summary['adaptive_empirical_coverage']:>10.4f} "
        f"{summary['adaptive_mean_radius']:>10.4f} "
        f"{summary['adaptive_median_radius']:>10.4f} "
        f"{summary['adaptive_p90_radius']:>10.4f}"
    )
    if "local_empirical_coverage" in summary:
        print(
            f"{'kNN-local baseline':<30} {tc:>8.0%} "
            f"{summary['local_empirical_coverage']:>10.4f} "
            f"{summary['local_mean_radius']:>10.4f} "
            f"{summary['local_median_radius']:>10.4f} "
            f"{'—':>10}"
        )
    print()
    print(
        f"Spearman(difficulty, residual) = {summary['spearman_difficulty_residual']:.4f} "
        f"(p={summary['spearman_p_value']:.3e})"
    )


def print_quartile_table(summary: dict[str, Any]) -> None:
    """
    Print the quartile-level coverage table from Step 8.
    """
    print(
        f"\n{'Quartile':<12} {'Global cov':>12} {'Adaptive cov':>14} "
        f"{'Global R':>10} {'Adaptive R':>12}"
    )
    print("-" * 62)
    for i in range(1, 5):
        q = f"Q{i}"
        print(
            f"{q:<12} "
            f"{summary.get(f'global_coverage_q{i}', float('nan')):>12.4f} "
            f"{summary.get(f'adaptive_coverage_q{i}', float('nan')):>14.4f} "
            f"{summary.get(f'global_mean_radius_q{i}', float('nan')):>10.4f} "
            f"{summary.get(f'adaptive_mean_radius_q{i}', float('nan')):>12.4f}"
        )
    print()

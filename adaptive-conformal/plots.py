"""
plots.py
All five required plots for Experiment 4.

Plot 1 – Histogram of adaptive radii with global radius marked
Plot 2 – Predicted difficulty vs actual residual (scatter + Spearman)
Plot 3 – Radius by difficulty quartile (boxplots)
Plot 4 – Coverage by difficulty quartile (grouped bar)
Plot 5 – Galaxy example thumbnails (placeholder; requires image data)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ALPHA = 0.4


def _save(fig: plt.Figure, path: str | Path, dpi: int = 150) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {path}")


# ---------------------------------------------------------------------------
# Plot 1: histogram of adaptive radii with global radius marked
# ---------------------------------------------------------------------------

def plot_radius_distribution(
    adaptive_radius: np.ndarray,
    global_radius: float,
    *,
    survey_pair: str,
    backbone: str,
    target_coverage: float,
    out_dir: str | Path = "results",
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(adaptive_radius, bins=60, color="steelblue", alpha=0.75, label="Adaptive radius")
    ax.axvline(global_radius, color="crimson", lw=2, ls="--", label=f"Global radius = {global_radius:.4f}")
    ax.set_xlabel("Prediction radius")
    ax.set_ylabel("Count")
    ax.set_title(
        f"Adaptive vs Global radius  |  {survey_pair}  |  {backbone}\n"
        f"target coverage = {target_coverage:.0%}"
    )
    ax.legend()
    fname = (
        f"plot1_radius_hist_{survey_pair}_{backbone}_{int(target_coverage*100)}.png"
    )
    _save(fig, Path(out_dir) / fname)


# ---------------------------------------------------------------------------
# Plot 2: predicted difficulty vs actual residual
# ---------------------------------------------------------------------------

def plot_difficulty_vs_residual(
    predicted_difficulty: np.ndarray,
    true_residual: np.ndarray,
    spearman_r: float,
    *,
    survey_pair: str,
    backbone: str,
    target_coverage: float,
    out_dir: str | Path = "results",
) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(
        predicted_difficulty,
        true_residual,
        s=5,
        alpha=_ALPHA,
        color="darkorange",
        rasterized=True,
    )
    ax.set_xlabel(r"Predicted difficulty $\hat{\sigma}(x)$")
    ax.set_ylabel("Actual residual $r$")
    ax.set_title(
        f"Difficulty vs residual  |  {survey_pair}  |  {backbone}\n"
        f"Spearman ρ = {spearman_r:.3f}"
    )
    fname = (
        f"plot2_difficulty_residual_{survey_pair}_{backbone}_{int(target_coverage*100)}.png"
    )
    _save(fig, Path(out_dir) / fname)


# ---------------------------------------------------------------------------
# Plot 3: radius by difficulty quartile (boxplots)
# ---------------------------------------------------------------------------

def plot_radius_by_quartile(
    predicted_difficulty: np.ndarray,
    adaptive_radius: np.ndarray,
    global_radius: np.ndarray,
    *,
    survey_pair: str,
    backbone: str,
    target_coverage: float,
    out_dir: str | Path = "results",
) -> None:
    quantiles_vals = np.quantile(predicted_difficulty, [0.25, 0.50, 0.75])
    bins = np.digitize(predicted_difficulty, quantiles_vals)

    groups_adaptive = [adaptive_radius[bins == q] for q in range(4)]
    groups_global = [global_radius[bins == q] for q in range(4)]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)
    labels = ["Q1\n(easiest)", "Q2", "Q3", "Q4\n(hardest)"]

    for ax, groups, title in zip(
        axes,
        [groups_adaptive, groups_global],
        ["Adaptive radius", "Global radius"],
    ):
        ax.boxplot(
            groups,
            labels=labels,
            patch_artist=True,
            boxprops=dict(facecolor="steelblue", alpha=0.6),
            medianprops=dict(color="black", lw=2),
            flierprops=dict(marker=".", alpha=0.3, ms=3),
        )
        ax.set_xlabel("Difficulty quartile")
        ax.set_ylabel("Radius")
        ax.set_title(f"{title}  |  {survey_pair}  |  {backbone}")

    fig.suptitle(f"Target coverage = {target_coverage:.0%}", y=1.01)
    fname = (
        f"plot3_radius_quartile_{survey_pair}_{backbone}_{int(target_coverage*100)}.png"
    )
    _save(fig, Path(out_dir) / fname)


# ---------------------------------------------------------------------------
# Plot 4: coverage by difficulty quartile (grouped bar)
# ---------------------------------------------------------------------------

def plot_coverage_by_quartile(
    predicted_difficulty: np.ndarray,
    true_residual: np.ndarray,
    global_radius: np.ndarray,
    adaptive_radius: np.ndarray,
    *,
    target_coverage: float,
    survey_pair: str,
    backbone: str,
    out_dir: str | Path = "results",
) -> None:
    quantiles_vals = np.quantile(predicted_difficulty, [0.25, 0.50, 0.75])
    bins = np.digitize(predicted_difficulty, quantiles_vals)

    global_cov, adaptive_cov = [], []
    for q in range(4):
        mask = bins == q
        if mask.sum() == 0:
            global_cov.append(float("nan"))
            adaptive_cov.append(float("nan"))
        else:
            global_cov.append((true_residual[mask] <= global_radius[mask]).mean())
            adaptive_cov.append((true_residual[mask] <= adaptive_radius[mask]).mean())

    x = np.arange(4)
    width = 0.35
    labels = ["Q1 (easiest)", "Q2", "Q3", "Q4 (hardest)"]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, global_cov, width, label="Global CP", color="steelblue", alpha=0.8)
    ax.bar(x + width / 2, adaptive_cov, width, label="Adaptive CP", color="darkorange", alpha=0.8)
    ax.axhline(target_coverage, color="black", ls="--", lw=1.5, label=f"Target {target_coverage:.0%}")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Empirical coverage")
    ax.set_title(
        f"Coverage by difficulty quartile  |  {survey_pair}  |  {backbone}\n"
        f"target coverage = {target_coverage:.0%}"
    )
    ax.legend()
    fname = (
        f"plot4_coverage_quartile_{survey_pair}_{backbone}_{int(target_coverage*100)}.png"
    )
    _save(fig, Path(out_dir) / fname)


# ---------------------------------------------------------------------------
# Plot 5: galaxy examples by radius category
# ---------------------------------------------------------------------------

def plot_galaxy_examples(
    true_residual: np.ndarray,
    adaptive_radius: np.ndarray,
    predicted_difficulty: np.ndarray,
    row_ids: np.ndarray,
    *,
    survey_pair: str,
    backbone: str,
    target_coverage: float,
    out_dir: str | Path = "results",
    image_fetcher=None,   # callable(row_id) -> (H,W,3) array or None
) -> None:
    """
    Show five representative objects:
        - low radius (easy, covered)
        - medium radius
        - high radius (hard, covered)
        - high radius and uncovered

    If image_fetcher is None, prints object metadata only.
    """
    covered = true_residual <= adaptive_radius
    r_pct = np.percentile(adaptive_radius, [10, 50, 90])

    categories = {
        "Low radius\n(covered)": (adaptive_radius < r_pct[0]) & covered,
        "Medium radius\n(covered)": (
            (adaptive_radius >= r_pct[0]) & (adaptive_radius < r_pct[1]) & covered
        ),
        "High radius\n(covered)": (adaptive_radius >= r_pct[2]) & covered,
        "High radius\n(uncovered)": (adaptive_radius >= r_pct[2]) & ~covered,
    }

    if image_fetcher is None:
        print("\nGalaxy examples (no image fetcher provided):")
        for cat, mask in categories.items():
            idx = np.where(mask)[0]
            if len(idx) == 0:
                print(f"  {cat}: no objects found")
                continue
            i = idx[0]
            print(
                f"  {cat}: row_id={row_ids[i]}  "
                f"r={true_residual[i]:.4f}  "
                f"R={adaptive_radius[i]:.4f}  "
                f"difficulty={predicted_difficulty[i]:.4f}"
            )
        return

    n_cats = len(categories)
    fig, axes = plt.subplots(1, n_cats, figsize=(4 * n_cats, 4))
    if n_cats == 1:
        axes = [axes]

    for ax, (cat, mask) in zip(axes, categories.items()):
        idx = np.where(mask)[0]
        if len(idx) == 0:
            ax.set_title(f"{cat}\n(none found)")
            ax.axis("off")
            continue
        i = idx[0]
        img = image_fetcher(row_ids[i])
        if img is not None:
            ax.imshow(img)
        else:
            ax.text(0.5, 0.5, "no image", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(
            f"{cat}\nr={true_residual[i]:.3f}  R={adaptive_radius[i]:.3f}",
            fontsize=9,
        )
        ax.axis("off")

    fig.suptitle(
        f"Galaxy examples  |  {survey_pair}  |  {backbone}\n"
        f"target coverage = {target_coverage:.0%}",
        y=1.02,
    )
    fname = (
        f"plot5_galaxy_examples_{survey_pair}_{backbone}_{int(target_coverage*100)}.png"
    )
    _save(fig, Path(out_dir) / fname)


# ---------------------------------------------------------------------------
# All plots for one run
# ---------------------------------------------------------------------------

def make_all_plots(
    *,
    true_residual: np.ndarray,
    predicted_difficulty: np.ndarray,
    global_radius: np.ndarray,
    adaptive_radius: np.ndarray,
    spearman_r: float,
    row_ids: np.ndarray,
    survey_pair: str,
    backbone: str,
    target_coverage: float,
    out_dir: str | Path = "results",
    image_fetcher=None,
) -> None:
    """Convenience wrapper: produce all 5 plots in one call."""
    g_scalar = float(global_radius[0])  # global radius is constant

    plot_radius_distribution(
        adaptive_radius, g_scalar,
        survey_pair=survey_pair, backbone=backbone,
        target_coverage=target_coverage, out_dir=out_dir,
    )
    plot_difficulty_vs_residual(
        predicted_difficulty, true_residual, spearman_r,
        survey_pair=survey_pair, backbone=backbone,
        target_coverage=target_coverage, out_dir=out_dir,
    )
    plot_radius_by_quartile(
        predicted_difficulty, adaptive_radius, global_radius,
        survey_pair=survey_pair, backbone=backbone,
        target_coverage=target_coverage, out_dir=out_dir,
    )
    plot_coverage_by_quartile(
        predicted_difficulty, true_residual, global_radius, adaptive_radius,
        target_coverage=target_coverage, survey_pair=survey_pair, backbone=backbone,
        out_dir=out_dir,
    )
    plot_galaxy_examples(
        true_residual, adaptive_radius, predicted_difficulty, row_ids,
        survey_pair=survey_pair, backbone=backbone,
        target_coverage=target_coverage, out_dir=out_dir,
        image_fetcher=image_fetcher,
    )

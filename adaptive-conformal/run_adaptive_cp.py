"""
run_adaptive_cp.py
Experiment 4: Object-Adaptive Conformal Alignment

Entry point.  Reads a YAML config, loads embeddings, runs the full
adaptive CP pipeline (Steps 1–10), saves per-object CSVs and summary CSVs,
and produces all plots.

Usage
-----
python run_adaptive_cp.py --config configs/hsc_jwst.yaml
python run_adaptive_cp.py --config configs/legacy_hsc.yaml --coverage 90
python run_adaptive_cp.py --config configs/hsc_jwst.yaml --difficulty_model hgbt ridge knn
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

# Local imports
from adaptive_conformal import (
    adaptive_radii as make_adaptive_radii,
    adaptive_scores,
    conformal_quantile,
    vector_residual,
)
from difficulty_model import make_difficulty_model
from evaluate import (
    per_object_rows,
    print_comparison_table,
    print_quartile_table,
    run_summary,
)
from local_support import build_support_features, augment_with_support, local_conformal_radii
from plots import make_all_plots


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def load_embeddings(cfg: dict, split_key: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load source and target embeddings for a given split.

    Expects cfg to contain:
        source_embeddings_<split_key>: path to .npy file
        target_embeddings_<split_key>: path to .npy file

    OR a single file for the whole dataset (cfg.source_embeddings / target_embeddings)
    with row indices in cfg.<split_key>_indices.
    """
    src_key = f"source_embeddings_{split_key}"
    tgt_key = f"target_embeddings_{split_key}"

    if src_key in cfg:
        X = np.load(cfg[src_key])
        Y = np.load(cfg[tgt_key])
    else:
        # Fall back: load full arrays, slice by index file
        X_all = np.load(cfg["source_embeddings"])
        Y_all = np.load(cfg["target_embeddings"])
        idx = np.load(cfg[f"{split_key}_indices"])
        X, Y = X_all[idx], Y_all[idx]

    return X.astype(np.float32), Y.astype(np.float32)


def load_row_ids(cfg: dict, split_key: str) -> np.ndarray:
    key = f"row_ids_{split_key}"
    if key in cfg:
        return np.load(cfg[key])
    n = np.load(cfg[f"source_embeddings_{split_key}"]).shape[0]
    return np.arange(n)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_one_backbone(
    backbone: str,
    cfg: dict,
    target_coverages: list[float],
    difficulty_model_kinds: list[str],
    split_seed: int,
    out_dir: Path,
    run_local_cp: bool = True,
    add_support_features: bool = True,
    local_k_values: list[int] | None = None,
) -> tuple[list[pd.DataFrame], list[dict]]:
    """
    Run the full pipeline for one backbone.
    Returns (list of per-object DataFrames, list of summary dicts).
    """
    survey_pair = cfg["survey_pair"]
    print(f"\n{'='*60}")
    print(f"  backbone   : {backbone}")
    print(f"  survey pair: {survey_pair}")
    print(f"{'='*60}")

    # ------------------------------------------------------------------
    # 1. Load pre-split data
    # ------------------------------------------------------------------
    print("Loading embeddings …")
    X_train, Y_train = load_embeddings(cfg, "train")
    X_cal, Y_cal = load_embeddings(cfg, "cal")
    X_test, Y_test = load_embeddings(cfg, "test")
    row_ids_test = load_row_ids(cfg, "test")

    print(f"  train: {X_train.shape}  cal: {X_cal.shape}  test: {X_test.shape}")

    # ------------------------------------------------------------------
    # 2. Internal train split (translator vs difficulty)
    # ------------------------------------------------------------------
    diff_frac = cfg.get("difficulty_train_fraction", 0.20)
    X_fit, X_diff, Y_fit, Y_diff = train_test_split(
        X_train, Y_train, test_size=diff_frac, random_state=split_seed
    )
    print(f"  translator training : {X_fit.shape[0]}  difficulty training : {X_diff.shape[0]}")

    # ------------------------------------------------------------------
    # 3. Fit Ridge translator (reused from Experiments 2 & 3)
    # ------------------------------------------------------------------
    alpha_ridge = cfg.get("ridge_alpha", 1.0)
    print(f"Fitting Ridge translator (alpha={alpha_ridge}) …")
    translator = Ridge(alpha=alpha_ridge, fit_intercept=True)
    translator.fit(X_fit, Y_fit)

    # ------------------------------------------------------------------
    # 4. Difficulty-training residuals
    # ------------------------------------------------------------------
    Y_diff_pred = translator.predict(X_diff)
    r_diff = vector_residual(Y_diff, Y_diff_pred)

    # ------------------------------------------------------------------
    # 5. Optionally augment with support features (Step 9)
    # ------------------------------------------------------------------
    pca_dim = cfg.get("pca_dim", 64)

    if add_support_features:
        print("Computing local-support features …")
        # Build support features relative to translator-training set
        sup_diff = build_support_features(X_diff, X_fit, pca_components=pca_dim)
        sup_cal = build_support_features(X_cal, X_fit, pca_components=pca_dim)
        sup_test = build_support_features(X_test, X_fit, pca_components=pca_dim)

        X_diff_feat = augment_with_support(X_diff, sup_diff, pca_components=pca_dim)
        X_cal_feat = augment_with_support(X_cal, sup_cal, pca_components=pca_dim)
        X_test_feat = augment_with_support(X_test, sup_test, pca_components=pca_dim)
    else:
        X_diff_feat = X_diff
        X_cal_feat = X_cal
        X_test_feat = X_test

    # 1-NN distances for logging
    from sklearn.neighbors import NearestNeighbors
    nn_1 = NearestNeighbors(n_neighbors=1, metric="euclidean", n_jobs=-1).fit(X_fit)
    nn_dist_test, _ = nn_1.kneighbors(X_test)
    nn_dist_test = nn_dist_test[:, 0]

    # ------------------------------------------------------------------
    # 6. Calibration residuals (shared across difficulty models)
    # ------------------------------------------------------------------
    Y_cal_pred = translator.predict(X_cal)
    r_cal = vector_residual(Y_cal, Y_cal_pred)

    # ------------------------------------------------------------------
    # 7. Test residuals (shared)
    # ------------------------------------------------------------------
    Y_test_pred = translator.predict(X_test)
    r_test = vector_residual(Y_test, Y_test_pred)

    all_obj_dfs: list[pd.DataFrame] = []
    all_summaries: list[dict] = []

    # ------------------------------------------------------------------
    # Loop over difficulty model types
    # ------------------------------------------------------------------
    for dm_kind in difficulty_model_kinds:
        print(f"\n  difficulty model: {dm_kind}")
        dm = make_difficulty_model(dm_kind)
        dm.fit(X_diff_feat, r_diff)

        # Predict difficulty
        sigma_cal = dm.predict(X_cal_feat)
        sigma_test = dm.predict(X_test_feat)

        # Normalised calibration scores
        a_cal = adaptive_scores(r_cal, sigma_cal)

        # ------------------------------------------------------------------
        # Loop over target coverages
        # ------------------------------------------------------------------
        for tc in target_coverages:
            alpha = 1.0 - tc
            print(f"    coverage = {tc:.0%}  (alpha = {alpha:.2f})")

            # Quantiles
            q_global = conformal_quantile(r_cal, alpha)
            q_adaptive = conformal_quantile(a_cal, alpha)

            # Radii
            global_radius = np.full(len(X_test), q_global)
            adaptive_radius = make_adaptive_radii(q_adaptive, sigma_test)

            # kNN-local baseline (Step 10)
            local_radius = None
            if run_local_cp:
                k_local = (local_k_values or [50])[0]
                local_radius = local_conformal_radii(
                    X_test, X_cal, r_cal, alpha, k=k_local, pca_components=pca_dim
                )

            # Summary
            from scipy.stats import spearmanr
            spear_r, _ = spearmanr(sigma_test, r_test)

            summary = run_summary(
                survey_pair=survey_pair,
                backbone=backbone,
                difficulty_model_kind=dm_kind,
                target_coverage=tc,
                true_residual=r_test,
                predicted_difficulty=sigma_test,
                global_radius=global_radius,
                adaptive_radius=adaptive_radius,
                local_radius=local_radius,
            )
            all_summaries.append(summary)

            print_comparison_table(summary)
            print_quartile_table(summary)

            # Per-object rows
            obj_df = per_object_rows(
                row_ids=row_ids_test,
                survey_pair=survey_pair,
                backbone=backbone,
                target_coverage=tc,
                true_residual=r_test,
                predicted_difficulty=sigma_test,
                global_radius=global_radius,
                adaptive_radius=adaptive_radius,
                nearest_neighbor_distance=nn_dist_test,
                local_radius=local_radius,
                split_seed=split_seed,
            )
            all_obj_dfs.append(obj_df)

            # Plots
            plot_dir = out_dir / f"{survey_pair}_{backbone}_{dm_kind}"
            make_all_plots(
                true_residual=r_test,
                predicted_difficulty=sigma_test,
                global_radius=global_radius,
                adaptive_radius=adaptive_radius,
                spearman_r=spear_r,
                row_ids=row_ids_test,
                survey_pair=survey_pair,
                backbone=f"{backbone}_{dm_kind}",
                target_coverage=tc,
                out_dir=plot_dir,
            )

    return all_obj_dfs, all_summaries


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Experiment 4: Object-Adaptive Conformal Alignment"
    )
    p.add_argument("--config", required=True, help="Path to YAML config file")
    p.add_argument(
        "--coverage",
        nargs="+",
        type=float,
        default=[80, 90, 95],
        help="Target coverage percentages (default: 80 90 95)",
    )
    p.add_argument(
        "--difficulty_model",
        nargs="+",
        default=["hgbt"],
        choices=["ridge", "hgbt", "knn"],
        help="Difficulty model types to run (default: hgbt)",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no_local_cp", action="store_true", help="Skip kNN-local baseline")
    p.add_argument("--no_support_features", action="store_true", help="Skip Step 9 support features")
    p.add_argument(
        "--local_k",
        nargs="+",
        type=int,
        default=[50],
        help="k values for kNN-local baseline (first value used by default)",
    )
    p.add_argument("--out_dir", default=None, help="Override output directory")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        print(f"ERROR: config not found: {cfg_path}", file=sys.stderr)
        sys.exit(1)

    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    target_coverages = [c / 100.0 if c > 1.0 else c for c in args.coverage]
    backbones: list[str] = cfg.get("backbones", ["astropt_850m"])

    out_dir = Path(args.out_dir) if args.out_dir else Path("results") / cfg["survey_pair"]
    out_dir.mkdir(parents=True, exist_ok=True)

    all_obj_dfs: list[pd.DataFrame] = []
    all_summaries: list[dict] = []

    for backbone in backbones:
        # Allow per-backbone embedding paths if provided in config
        bcfg = {**cfg, **cfg.get("backbone_overrides", {}).get(backbone, {})}

        obj_dfs, summaries = run_one_backbone(
            backbone=backbone,
            cfg=bcfg,
            target_coverages=target_coverages,
            difficulty_model_kinds=args.difficulty_model,
            split_seed=args.seed,
            out_dir=out_dir,
            run_local_cp=not args.no_local_cp,
            add_support_features=not args.no_support_features,
            local_k_values=args.local_k,
        )
        all_obj_dfs.extend(obj_dfs)
        all_summaries.extend(summaries)

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    if all_obj_dfs:
        obj_path = out_dir / "per_object_results.csv"
        pd.concat(all_obj_dfs, ignore_index=True).to_csv(obj_path, index=False)
        print(f"\nPer-object results saved → {obj_path}")

    if all_summaries:
        summary_path = out_dir / "run_summary.csv"
        pd.DataFrame(all_summaries).to_csv(summary_path, index=False)
        print(f"Run summary saved       → {summary_path}")


if __name__ == "__main__":
    main()

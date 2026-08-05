"""
run_adaptive_cp.py
Experiment 4: Object-Adaptive Conformal Alignment

Mirrors the data-loading pattern from:
  ridge-regression/cross-survey-translation/train_kfold_ridge.py
  ridge-regression/conformal-prediction-analysis/conformal_prediction.py

Usage
-----
# Recommended: start with AstroPT-850M on HSC->JWST at 90% only
python run_adaptive_cp.py --dataset jwst_hsc --config astropt_850m --coverage 90

# Full run
python run_adaptive_cp.py --dataset jwst_hsc --coverage 80 90 95 --difficulty_model hgbt

# All datasets, all backbones
python run_adaptive_cp.py --dataset all --config all --coverage 80 90 95
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split
from scipy.stats import spearmanr

from dataloader import load_data, df_to_arrays
from adaptive_conformal import (
    adaptive_radii as make_adaptive_radii,
    adaptive_scores,
    conformal_quantile,
    vector_residual,
)
from difficulty_model import make_difficulty_model
from evaluate import per_object_rows, print_comparison_table, print_quartile_table, run_summary
from local_support import build_support_features, augment_with_support, local_conformal_radii
from plots import make_all_plots


# ---------------------------------------------------------------------------
# Dataset registry  (mirrors conformal_prediction.py exactly)
# ---------------------------------------------------------------------------

DATASETS = {
    "jwst_hsc": {
        "repo_id": "astro-conformal-project/split_jwst_hsc_embeddings",
        "source_embed": "{config}_hsc",
        "target_embed": "{config}_jwst",
        # 60/20/20 split — difficulty fraction inside train
        "difficulty_train_fraction": 0.30,
    },
    "sdss_hsc": {
        "repo_id": "astro-conformal-project/split_sdss_hsc_embeddings",
        "source_embed": "specformer_base_sdss",   # source is fixed for sdss
        "target_embed": "{config}_hsc",
        "difficulty_train_fraction": 0.30,
    },
    "legacysurvey_hsc": {
        "repo_id": "astro-conformal-project/split_legacysurvey_hsc_embeddings",
        "source_embed": "{config}_legacysurvey",
        "target_embed": "{config}_hsc",
        # 70/15/15 split
        "difficulty_train_fraction": 0.20,
    },
}

# All backbones from the existing scripts
ALL_CONFIGS = [
    "astropt_850m",   # ← start here per spec
    "astropt_95m",
    "astropt_15m",
    "vit_huge",
    "vit_large",
    "vit_base",
    "dino_small",
    "dino_base",
    "dino_large",
    "dino_giant",
    "convnext_nano",
    "convnext_tiny",
    "convnext_base",
    "convnext_large",
    "ijepa_huge",
    "ijepa_giant",
]

ALPHAS_RIDGE = np.logspace(-4, 4, 25)


# ---------------------------------------------------------------------------
# Seed
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


# ---------------------------------------------------------------------------
# Core pipeline for one (dataset, config) pair
# ---------------------------------------------------------------------------

def run_one(
    dataset_name: str,
    config: str,
    target_coverages: list[float],
    difficulty_model_kinds: list[str],
    seed: int,
    out_dir: Path,
    run_local_cp: bool,
    add_support_features: bool,
    local_k: int,
    pca_dim: int,
) -> tuple[list[pd.DataFrame], list[dict]]:

    spec = DATASETS[dataset_name]
    source_embed = spec["source_embed"].format(config=config)
    target_embed = spec["target_embed"].format(config=config)
    diff_frac = spec["difficulty_train_fraction"]

    print(f"\n{'='*60}")
    print(f"  dataset : {dataset_name}")
    print(f"  config  : {config}")
    print(f"  {source_embed} -> {target_embed}")
    print(f"{'='*60}")

    # ------------------------------------------------------------------
    # 1. Load data from HuggingFace (static splits)
    # ------------------------------------------------------------------
    print("Loading from HuggingFace …")
    data = load_data(
        repo_id=spec["repo_id"],
        config_name=config,
        source_embed=source_embed,
        target_embed=target_embed,
        normalize=True,
        import_splits=True,
    )

    train_df = data["train"]
    cal_df   = data["calibration"]
    test_df  = data["test"]
    print(f"  train={len(train_df)}  cal={len(cal_df)}  test={len(test_df)}")

    X_train, Y_train = df_to_arrays(train_df, source_embed, target_embed)
    X_cal,   Y_cal   = df_to_arrays(cal_df,   source_embed, target_embed)
    X_test,  Y_test  = df_to_arrays(test_df,  source_embed, target_embed)

    # Row IDs for uncovered-object tracking
    id_col = "object_id"
    row_ids = (
        test_df[id_col].values if id_col in test_df.columns
        else np.arange(len(test_df))
    )

    # ------------------------------------------------------------------
    # 2. Internal train split: translator vs difficulty (Step 2)
    # ------------------------------------------------------------------
    X_fit, X_diff, Y_fit, Y_diff = train_test_split(
        X_train, Y_train, test_size=diff_frac, random_state=seed
    )
    print(f"  translator={len(X_fit)}  difficulty={len(X_diff)}")

    # ------------------------------------------------------------------
    # 3. Fit Ridge translator (Step 2 — same as Experiments 2 & 3)
    # ------------------------------------------------------------------
    print("Fitting RidgeCV translator …")
    translator = RidgeCV(alphas=ALPHAS_RIDGE, cv=5, scoring="neg_mean_squared_error")
    translator.fit(X_fit, Y_fit)
    print(f"  chosen alpha: {translator.alpha_}")

    # ------------------------------------------------------------------
    # 4. Difficulty-training residuals (Step 2)
    # ------------------------------------------------------------------
    Y_diff_pred = translator.predict(X_diff)
    r_diff = vector_residual(Y_diff, Y_diff_pred)

    # ------------------------------------------------------------------
    # 5. Local-support features (Step 9, optional)
    # ------------------------------------------------------------------
    if add_support_features:
        print("Computing local-support features …")
        sup_diff = build_support_features(X_diff, X_fit, pca_components=pca_dim)
        sup_cal  = build_support_features(X_cal,  X_fit, pca_components=pca_dim)
        sup_test = build_support_features(X_test, X_fit, pca_components=pca_dim)

        X_diff_feat = augment_with_support(X_diff, sup_diff, pca_components=pca_dim)
        X_cal_feat  = augment_with_support(X_cal,  sup_cal,  pca_components=pca_dim)
        X_test_feat = augment_with_support(X_test, sup_test, pca_components=pca_dim)
    else:
        X_diff_feat = X_diff
        X_cal_feat  = X_cal
        X_test_feat = X_test

    # 1-NN distance for logging
    from sklearn.neighbors import NearestNeighbors
    nn1 = NearestNeighbors(n_neighbors=1, metric="euclidean", n_jobs=-1).fit(X_fit)
    nn_dist_test = nn1.kneighbors(X_test)[0][:, 0]

    # ------------------------------------------------------------------
    # 6. Calibration + test residuals (shared across difficulty models)
    # ------------------------------------------------------------------
    Y_cal_pred  = translator.predict(X_cal)
    r_cal       = vector_residual(Y_cal,  Y_cal_pred)

    Y_test_pred = translator.predict(X_test)
    r_test      = vector_residual(Y_test, Y_test_pred)

    all_obj_dfs:  list[pd.DataFrame] = []
    all_summaries: list[dict]        = []

    # ------------------------------------------------------------------
    # Loop: difficulty model types
    # ------------------------------------------------------------------
    for dm_kind in difficulty_model_kinds:
        print(f"\n  difficulty model: {dm_kind}")

        # Step 3: Train difficulty model
        dm = make_difficulty_model(dm_kind)
        dm.fit(X_diff_feat, r_diff)

        sigma_cal  = dm.predict(X_cal_feat)
        sigma_test = dm.predict(X_test_feat)

        # Step 4: Normalised calibration scores
        a_cal = adaptive_scores(r_cal, sigma_cal)

        # ----------------------------------------------------------------
        # Loop: target coverages
        # ----------------------------------------------------------------
        for tc in target_coverages:
            alpha = 1.0 - tc
            print(f"    coverage={tc:.0%}  alpha={alpha:.2f}")

            # Step 5: Quantiles
            q_global   = conformal_quantile(r_cal, alpha)
            q_adaptive = conformal_quantile(a_cal, alpha)

            # Step 6: Radii
            global_radius   = np.full(len(X_test), q_global)
            adaptive_radius = make_adaptive_radii(q_adaptive, sigma_test)

            # Step 10: kNN-local baseline
            local_radius = None
            if run_local_cp:
                local_radius = local_conformal_radii(
                    X_test, X_cal, r_cal, alpha,
                    k=local_k, pca_components=pca_dim,
                )

            spear_r, _ = spearmanr(sigma_test, r_test)

            summary = run_summary(
                survey_pair=dataset_name,
                backbone=config,
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

            obj_df = per_object_rows(
                row_ids=row_ids,
                survey_pair=dataset_name,
                backbone=config,
                target_coverage=tc,
                true_residual=r_test,
                predicted_difficulty=sigma_test,
                global_radius=global_radius,
                adaptive_radius=adaptive_radius,
                nearest_neighbor_distance=nn_dist_test,
                local_radius=local_radius,
                split_seed=seed,
            )
            all_obj_dfs.append(obj_df)

            # Plots
            plot_dir = out_dir / f"{dataset_name}_{config}_{dm_kind}"
            make_all_plots(
                true_residual=r_test,
                predicted_difficulty=sigma_test,
                global_radius=global_radius,
                adaptive_radius=adaptive_radius,
                spearman_r=spear_r,
                row_ids=row_ids,
                survey_pair=dataset_name,
                backbone=f"{config}_{dm_kind}",
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
    p.add_argument(
        "--dataset",
        nargs="+",
        default=["jwst_hsc"],
        help="Dataset(s) to run. Choices: jwst_hsc sdss_hsc legacysurvey_hsc all",
    )
    p.add_argument(
        "--config",
        nargs="+",
        default=["astropt_850m"],
        help="Backbone config(s) to run. Use 'all' for every backbone.",
    )
    p.add_argument(
        "--coverage",
        nargs="+",
        type=float,
        default=[90],
        help="Target coverage %% values (default: 90)",
    )
    p.add_argument(
        "--difficulty_model",
        nargs="+",
        default=["hgbt"],
        choices=["ridge", "hgbt", "knn"],
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no_local_cp", action="store_true")
    p.add_argument("--no_support_features", action="store_true")
    p.add_argument("--local_k", type=int, default=50)
    p.add_argument("--pca_dim", type=int, default=64)
    p.add_argument("--out_dir", default="results")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    # Resolve dataset list
    datasets = list(DATASETS.keys()) if "all" in args.dataset else args.dataset
    for d in datasets:
        if d not in DATASETS:
            print(f"ERROR: unknown dataset {d!r}. Choose from {list(DATASETS)}", file=sys.stderr)
            sys.exit(1)

    # Resolve config list
    configs = ALL_CONFIGS if "all" in args.config else args.config

    target_coverages = [c / 100.0 if c > 1.0 else c for c in args.coverage]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_obj_dfs:  list[pd.DataFrame] = []
    all_summaries: list[dict]        = []

    for dataset_name in datasets:
        for config in configs:
            try:
                obj_dfs, summaries = run_one(
                    dataset_name=dataset_name,
                    config=config,
                    target_coverages=target_coverages,
                    difficulty_model_kinds=args.difficulty_model,
                    seed=args.seed,
                    out_dir=out_dir,
                    run_local_cp=not args.no_local_cp,
                    add_support_features=not args.no_support_features,
                    local_k=args.local_k,
                    pca_dim=args.pca_dim,
                )
                all_obj_dfs.extend(obj_dfs)
                all_summaries.extend(summaries)

            except Exception as e:
                print(f"\nFAILED: {dataset_name} | {config} — {e}")
                continue

    # Save
    if all_obj_dfs:
        p = out_dir / "per_object_results.csv"
        pd.concat(all_obj_dfs, ignore_index=True).to_csv(p, index=False)
        print(f"\nPer-object results → {p}")

    if all_summaries:
        p = out_dir / "run_summary.csv"
        pd.DataFrame(all_summaries).to_csv(p, index=False)
        print(f"Run summary        → {p}")


if __name__ == "__main__":
    main()

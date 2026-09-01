"""
(KNN) Global Conformal Prediction : Shuffled Conformal prediction over the test + calibration set, Uncovered Objects Overlap
"""

import argparse
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
from tqdm.auto import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent / "ridge-regression"

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))


def load_module_from_path(module_name, path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {module_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ridge_dataloader = load_module_from_path(
    "ridge_experiment_dataloader", SRC_DIR / "src" / "dataloader.py"
)
ridge_residuals = load_module_from_path(
    "ridge_experiment_residuals", SRC_DIR / "src" / "residuals.py"
)

from model import grid_search_knn 

load_data = ridge_dataloader.load_data
normalized_l2 = ridge_residuals.normalized_l2


CONFIGS = [
    "astropt_15m",
    "astropt_850m",
    "astropt_95m",
    "convnext_base",
    "convnext_large",
    "convnext_nano",
    "convnext_tiny",
    "dino_base",
    "dino_giant",
    "dino_large",
    "dino_small",
    "ijepa_giant",
    "ijepa_huge",
    "vit_base",
    "vit_huge",
    "vit_large",
]

DATASETS = {
    "jwst_hsc": {
        "repo_id": "astro-conformal-project/split_jwst_hsc_embeddings",
        "source_embed": "{config}_hsc",
        "target_embed": "{config}_jwst",
    },
    "sdss_hsc": {
        "repo_id": "astro-conformal-project/split_sdss_hsc_embeddings",
        "source_embed": "specformer_base_sdss",
        "target_embed": "{config}_hsc",
    },
    "legacysurvey_hsc": {
        "repo_id": "astro-conformal-project/split_legacysurvey_hsc_embeddings",
        "source_embed": "{config}_legacysurvey",
        "target_embed": "{config}_hsc",
    },
}

DEFAULT_NEIGHBORS = [5, 7, 10, 15, 20, 30, 40, 60, 80, 100, 130]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Seed-shuffled split-conformal analysis with a tuned k-NN translator"
    )
    parser.add_argument(
        "--thresholds",
        type=int,
        nargs="+",
        default=[80, 90, 95],
        help="target coverage levels in percent",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="explicit seeds for the calibration/test reshuffle (overrides --nseeds)",
    )
    parser.add_argument(
        "--nseeds",
        type=int,
        default=3,
        help="number of seeds for the calibration/test reshuffle",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DATASETS),
        choices=list(DATASETS),
        help="which survey pairs to run",
    )
    parser.add_argument(
        "--configs", nargs="+", default=CONFIGS, help="which backbone configs to run"
    )
    parser.add_argument(
        "--out-dir",
        default=str(SCRIPT_DIR),
        help="directory to write thresh_XX/ subfolders into",
    )
    parser.add_argument(
        "--neighbors",
        type=int,
        nargs="+",
        default=DEFAULT_NEIGHBORS,
        help="n_neighbors values for the k-NN grid search",
    )
    parser.add_argument(
        "--knn-cv", type=int, default=5, help="folds used to tune n_neighbors"
    )
    parser.add_argument(
        "--n-jobs", type=int, default=-1, help="parallel jobs used by GridSearchCV"
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="disable L2 normalization of embeddings and predictions",
    )
    return parser.parse_args()


def frame_to_arrays(frame, source_embed, target_embed):
    source = [np.asarray(value) for value in frame[source_embed]]
    target = [np.asarray(value) for value in frame[target_embed]]
    return np.stack(source), np.stack(target)


def format_mean_std(values):
    return f"{np.mean(values):.6f} ± {np.std(values):.6f}"


def write_results(thresholds, threshold_dirs, result_rows, uncovered_rows):
    for threshold in thresholds:
        results_df = pd.DataFrame(
            result_rows[threshold],
            columns=[
                "Model",
                "Survey pair",
                "Coverage error",
                "Empirical coverage",
                "Mean Radius",
            ],
        )
        results_df.to_csv(
            os.path.join(threshold_dirs[threshold], "conformal_results.csv"),
            index=False,
        )

        uncovered_df = pd.DataFrame(
            uncovered_rows[threshold],
            columns=[
                "Model",
                "Survey pair",
                "object_id",
                "Residual",
                "Conformal radius",
                "Difference (radius - residual)",
                "Empirical coverage",
                "test_split_seed",
                "seeds_uncovered",
            ],
        )
        uncovered_df.to_csv(
            os.path.join(threshold_dirs[threshold], "uncovered_objects.csv"),
            index=False,
        )


def run_experiment(args):
    thresholds = args.thresholds
    seeds = args.seeds if args.seeds is not None else list(range(args.nseeds))
    datasets = {name: DATASETS[name] for name in args.datasets}

    threshold_dirs = {}
    result_rows = {threshold: [] for threshold in thresholds}
    uncovered_rows = {threshold: [] for threshold in thresholds}

    for threshold in thresholds:
        threshold_dir = os.path.join(args.out_dir, f"thresh_{threshold}")
        os.makedirs(threshold_dir, exist_ok=True)
        threshold_dirs[threshold] = threshold_dir

    for dataset_name, dataset_spec in datasets.items():
        for config in tqdm(args.configs, desc=dataset_name):
            source_embed = dataset_spec["source_embed"].format(config=config)
            target_embed = dataset_spec["target_embed"].format(config=config)
            print(
                f"\n=== {dataset_name} | {config} | "
                f"{source_embed} -> {target_embed} ==="
            )

            try:
                data = load_data(
                    repo_id=dataset_spec["repo_id"],
                    config_name=config,
                    source_embed=source_embed,
                    target_embed=target_embed,
                    normalize=not args.no_normalize,
                    import_splits=True,
                )
                train_df = data["train"]
                calib_df = data["calibration"]
                test_df = data["test"]

                print(f"Train size: {len(train_df)}")
                print(f"Calibration size: {len(calib_df)}")
                print(f"Test size: {len(test_df)}")

                X_train, Y_train = frame_to_arrays(
                    train_df, source_embed, target_embed
                )

                pool_df = pd.concat([calib_df, test_df], axis=0, ignore_index=True)
                n_calib = len(calib_df)
                n_pool = len(pool_df)
                pool_ids = (
                    pool_df["object_id"].to_numpy()
                    if "object_id" in pool_df.columns
                    else pool_df.index.to_numpy()
                )
                X_pool, Y_pool = frame_to_arrays(pool_df, source_embed, target_embed)

               
                model = grid_search_knn(
                    {"source": X_train, "target": Y_train},
                    {"n_neighbors": args.neighbors},
                    cv=args.knn_cv,
                    n_jobs=args.n_jobs,
                )
                print("Chosen n_neighbors:", model.best_params_["n_neighbors"])

                Y_pool_pred = model.predict(X_pool)
                if not args.no_normalize:
                    Y_pool_pred = normalize(Y_pool_pred, norm="l2", axis=1)
                pool_residuals = normalized_l2(
                    translated=Y_pool_pred, actual=Y_pool
                )

                seed_coverage_errors = {threshold: [] for threshold in thresholds}
                seed_coverages = {threshold: [] for threshold in thresholds}
                seed_radii = {threshold: [] for threshold in thresholds}
                seed_uncovered = {threshold: {} for threshold in thresholds}
                seed_tested = {}

                for seed in seeds:
                    rng = np.random.default_rng(seed)
                    permutation = rng.permutation(n_pool)
                    calib_idx = permutation[:n_calib]
                    test_idx = permutation[n_calib:]

                    calib_residuals = pool_residuals[calib_idx]
                    test_residuals = pool_residuals[test_idx]
                    test_ids = pool_ids[test_idx]

                    for object_id in test_ids:
                        seed_tested.setdefault(object_id, []).append(seed)

                    for threshold in thresholds:
                        target_coverage = threshold / 100.0
                        alpha = 1.0 - target_coverage
                        n = len(calib_residuals)
                        q_level = np.ceil((n + 1) * (1 - alpha)) / n
                        radius = (
                            np.inf
                            if q_level > 1
                            else float(
                                np.quantile(
                                    calib_residuals, q_level, method="higher"
                                )
                            )
                        )

                        uncovered_mask = test_residuals > radius
                        empirical_coverage = float(np.mean(~uncovered_mask))
                        coverage_error = abs(empirical_coverage - target_coverage)

                        seed_coverage_errors[threshold].append(coverage_error)
                        seed_coverages[threshold].append(empirical_coverage)
                        seed_radii[threshold].append(radius)

                        uncovered_ids = test_ids[uncovered_mask]
                        uncovered_residuals = test_residuals[uncovered_mask]
                        for object_id, residual in zip(
                            uncovered_ids, uncovered_residuals
                        ):
                            entry = seed_uncovered[threshold].setdefault(
                                object_id,
                                {
                                    "residuals": [],
                                    "radii": [],
                                    "differences": [],
                                    "seeds": [],
                                },
                            )
                            entry["residuals"].append(float(residual))
                            entry["radii"].append(radius)
                            entry["differences"].append(radius - float(residual))
                            entry["seeds"].append(seed)

                        print(
                            f"[seed {seed} | thresh {threshold}] "
                            f"radius: {radius:.6f} | "
                            f"coverage: {empirical_coverage:.4f} | "
                            f"error: {coverage_error:.4f} | "
                            f"uncovered: {len(uncovered_ids)}"
                        )

                for threshold in thresholds:
                    coverage_str = format_mean_std(seed_coverages[threshold])
                    result_rows[threshold].append(
                        {
                            "Model": config,
                            "Survey pair": dataset_name,
                            "Coverage error": format_mean_std(
                                seed_coverage_errors[threshold]
                            ),
                            "Empirical coverage": coverage_str,
                            "Mean Radius": format_mean_std(seed_radii[threshold]),
                        }
                    )

                    for object_id, entry in seed_uncovered[threshold].items():
                        uncovered_rows[threshold].append(
                            {
                                "Model": config,
                                "Survey pair": dataset_name,
                                "object_id": object_id,
                                "Residual": format_mean_std(entry["residuals"]),
                                "Conformal radius": format_mean_std(entry["radii"]),
                                "Difference (radius - residual)": format_mean_std(
                                    entry["differences"]
                                ),
                                "Empirical coverage": coverage_str,
                                "test_split_seed": ";".join(
                                    str(value)
                                    for value in sorted(seed_tested[object_id])
                                ),
                                "seeds_uncovered": ";".join(
                                    str(value) for value in sorted(entry["seeds"])
                                ),
                            }
                        )

                    print(
                        f"[thresh {threshold}] mean radius: "
                        f"{np.mean(seed_radii[threshold]):.6f} | "
                        f"mean coverage: {np.mean(seed_coverages[threshold]):.4f} | "
                        "uncovered objects across seeds: "
                        f"{len(seed_uncovered[threshold])}"
                    )

            except Exception as exc:
                print(f"\nFAILED CONFIG: {dataset_name} | {config} | {exc}")
                continue

            write_results(
                thresholds,
                threshold_dirs,
                result_rows,
                uncovered_rows,
            )

    print("\nSaved results under:", args.out_dir)


if __name__ == "__main__":
    run_experiment(parse_args())

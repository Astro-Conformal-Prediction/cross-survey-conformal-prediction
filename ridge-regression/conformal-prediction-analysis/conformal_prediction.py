import os
import sys
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from src.dataloader import load_data
from sklearn.linear_model import RidgeCV
from src.residuals import normalized_l2


configs = ['astropt_15m', 'astropt_850m', 'astropt_95m', 'convnext_base', 'convnext_large', 'convnext_nano', 'convnext_tiny', 'dino_base', 
           'dino_giant', 'dino_large', 'dino_small', 'ijepa_giant', 'ijepa_huge', 'vit_base', 'vit_huge', 'vit_large']

datasets = {

        "jwst_hsc": {
        "repo_id": "astro-conformal-project/split_jwst_hsc_embeddings",
        "source_embed": "{config}_hsc",
        "target_embed": "{config}_jwst",
        "import_splits": True,
    },
    "sdss_hsc": {
        "repo_id": "astro-conformal-project/split_sdss_hsc_embeddings",
        "source_embed": "specformer_base_sdss",
        "target_embed": "{config}_hsc",
        "import_splits": True,
    },

    "legacysurvey_hsc": {
        "repo_id": "astro-conformal-project/split_legacysurvey_hsc_embeddings",
        "source_embed": "{config}_legacysurvey",
        "target_embed": "{config}_hsc",
        "import_splits": True,
    },
}

EXP_DIR = os.path.dirname(os.path.abspath(__file__))

alphas = np.logspace(-4, 4, 25) # hyperparam grid

thresholds = [80, 90, 95]

objectid_col = "object_id"

thresh_dirs = {}
result_rows = {t: [] for t in thresholds}
uncovered_rows = {t: [] for t in thresholds}

for t in thresholds:
    thresh_dir = os.path.join(EXP_DIR, f"thresh_{t}")
    os.makedirs(thresh_dir, exist_ok=True)
    thresh_dirs[t] = thresh_dir


for dataset_name, dataset_spec in datasets.items():
    for config in tqdm(configs, desc=dataset_name):

        source_embed = dataset_spec["source_embed"].format(config=config)
        target_embed = dataset_spec["target_embed"].format(config=config)

        print(f"\n=== {dataset_name} | {config} | {source_embed} -> {target_embed} ===")

        try:
            data = load_data(
                repo_id=dataset_spec["repo_id"],
                config_name=config,
                source_embed=source_embed,
                target_embed=target_embed,
                normalize=True,
                import_splits=True)

            # importing and using the static splits  for conformal analysis : 
            train_df = data["train"]
            calib_df = data["calibration"]
            test_df = data["test"]

            print(f"Train size: {len(train_df)}")
            print(f"Calibration size: {len(calib_df)}")
            print(f"Test size: {len(test_df)}")

            X_train  = []
            Y_train = []

            for e1, e2 in zip(train_df[source_embed], train_df[target_embed]):
                e1, e2 = np.asarray(e1), np.asarray(e2)
                X_train.append(e1)
                Y_train.append(e2)
            X_train = np.stack(X_train)
            Y_train = np.stack(Y_train)

            X_calib  = []
            Y_calib = []

            for e1, e2 in zip(calib_df[source_embed], calib_df[target_embed]):
                e1, e2 = np.asarray(e1), np.asarray(e2)
                X_calib.append(e1)
                Y_calib.append(e2)
            X_calib = np.stack(X_calib)
            Y_calib = np.stack(Y_calib)

            X_test  = []
            Y_test = []

            for e1, e2 in zip(test_df[source_embed], test_df[target_embed]):
                e1, e2 = np.asarray(e1), np.asarray(e2)
                X_test.append(e1)
                Y_test.append(e2)
            X_test = np.stack(X_test)
            Y_test = np.stack(Y_test)

            # RidgeCV tuning in train split (k-fold only tunes alpha here) :
            model = RidgeCV(alphas=alphas, cv=5, scoring="neg_mean_squared_error")

            model.fit(X_train, Y_train)

            print("Chosen alpha:", model.alpha_)

            Y_calib_pred = model.predict(X_calib)
            calib_residuals = normalized_l2(translated=Y_calib_pred, actual=Y_calib)

            Y_test_pred = model.predict(X_test)
            test_residuals = normalized_l2(translated=Y_test_pred, actual=Y_test)

            for threshold in thresholds:
                target_coverage = threshold / 100.0
                alpha  = 1.0 - target_coverage
                n = len(calib_residuals)
                q_level = np.ceil((n+1) * (1-alpha)) / n
                radius = float(np.quantile(calib_residuals, q_level, method = "higher"))

                uncovered_mask = test_residuals > radius
                covered = int(np.sum(~uncovered_mask))
                empirical_coverage = covered / len(test_residuals)
                coverage_error = abs(empirical_coverage - target_coverage)

                result_rows[threshold].append(
                    {
                        "Model": config,
                        "Survey pair": dataset_name,
                        "Coverage error": coverage_error,
                        "Empirical coverage": empirical_coverage,
                        "Mean Radius": radius,
                    }
                )

                if objectid_col in test_df.columns:
                    uncovered_ids = test_df.loc[uncovered_mask, objectid_col].tolist()
                else:
                    uncovered_ids = test_df.index[uncovered_mask].tolist()

                for object_id in uncovered_ids:
                    uncovered_rows[threshold].append(
                        {
                            "Model": config,
                            "Survey pair": dataset_name,
                            "object_id": object_id,
                        }
                    )

                print(f"[thresh {threshold}] radius: {radius:.6f} | "
                      f"coverage: {empirical_coverage:.4f} | "
                      f"error: {coverage_error:.4f} | "
                      f"uncovered: {len(uncovered_ids)}")

        except Exception:
            print(f"\nFAILED CONFIG: {dataset_name} | {config}")
            continue

        for threshold in thresholds:
            results_df = pd.DataFrame(
                result_rows[threshold],
                columns=["Model", "Survey pair", "Coverage error", "Empirical coverage", "Mean Radius"],
            )
            results_df.to_csv(
                os.path.join(thresh_dirs[threshold], "conformal_results.csv"),
                index=False
            )

            uncovered_df = pd.DataFrame(
                uncovered_rows[threshold],
                columns=["Model", "Survey pair", "object_id"],
            )
            uncovered_df.to_csv(
                os.path.join(thresh_dirs[threshold], "uncovered_objects.csv"),
                index=True
            )

print("\nSaved results under:", EXP_DIR)

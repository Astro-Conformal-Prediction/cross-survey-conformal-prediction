import os
import sys
import joblib
import numpy as np
import pandas as pd
from tqdm.auto import tqdm



current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
    
from src.dataloader import load_data
from sklearn.linear_model import RidgeCV
from src.residuals import residual_summary, normalized_l2, pooled_residuals


configs = ['astropt_15m', 'astropt_850m', 'astropt_95m', 'convnext_base', 'convnext_large', 'convnext_nano', 'convnext_tiny', 'dino_base', 'dino_giant', 'dino_large', 'dino_small', 'ijepa_giant', 'ijepa_huge', 'vit_base', 'vit_huge', 'vit_large'] 

datasets = {
    
        "jwst_hsc": {
        "repo_id": "astro-conformal-project/split_jwst_hsc_embeddings",
        "source_embed": "{config}_hsc",
        "target_embed": "{config}_jwst",
        "import_splits": False,
    },
    "sdss_hsc": {
        "repo_id": "astro-conformal-project/split_sdss_hsc_embeddings",
        "source_embed": "specformer_base_sdss",  
        "target_embed": "{config}_hsc", 
        "import_splits": False,
    },
    
    # we will use the existing split for the legacy survey dataset for  70-15-15 split 
    "legacysurvey_hsc": {
        "repo_id": "astro-conformal-project/split_legacysurvey_hsc_embeddings",
        "source_embed": "{config}_legacysurvey",
        "target_embed": "{config}_hsc",
        "import_splits": True,
    },
}

EXP_DIR = os.path.dirname(os.path.abspath(__file__))

model_dir = os.path.join(EXP_DIR, "saved_models")
os.makedirs(model_dir, exist_ok=True)
 
results_path = os.path.join(EXP_DIR, "results.csv")


alphas = np.logspace(-4, 4, 25) # hyperparam grid 

# for 60-20-20 rotating split :
k = 5

rows = []

for dataset_name, dataset_spec in datasets.items():
    for config in tqdm(configs, desc=dataset_name):

        source_embed = dataset_spec["source_embed"].format(config=config)
        target_embed = dataset_spec["target_embed"].format(config=config)
        import_splits = dataset_spec["import_splits"]

        print(f"\n=== {dataset_name} | {config} | {source_embed} -> {target_embed} ===")

        data = load_data(
            repo_id=dataset_spec["repo_id"],
            config_name=config,
            source_embed=source_embed,
            target_embed=target_embed,
            normalize=True,
            import_splits=import_splits)

        residuals = []

        model_paths = []
        folds = []

        fold_splits = []

        if import_splits: # for legacy 70-15-15 split (k fold only for parameter tuning)
            fold_splits.append(
                ("imported", "imported", data["train"], data["calibration"], data["test"])
            )
        else:
            full_df = data["full"]
            n = full_df.shape[0]

            chunks = [
                full_df.iloc[idx].reset_index(drop=True)
                for idx in np.array_split(np.arange(n), k)
            ]

            for fold in range(k):
                test_idx = fold
                calib_idx = (fold + 1) % k

                train_chunks = []
                test_df = None
                calib_df = None

                for i in range(k):
                    chunk = chunks[i]

                    if i == test_idx:
                        test_df = chunk
                    elif i == calib_idx:
                        calib_df = chunk
                    else:
                        train_chunks.append(chunk)

                train_df = pd.concat(train_chunks, ignore_index=True)

                fold_splits.append((fold, calib_idx, train_df, calib_df, test_df))

        for fold, calib_idx, train_df, calib_df, test_df in fold_splits:
            print(f"Training on Fold {fold}: \n ")

            X_train  = []
            Y_train = []

            for e1, e2 in zip(train_df[source_embed], train_df[target_embed]):
                e1, e2 = np.asarray(e1), np.asarray(e2)
                X_train.append(e1)
                Y_train.append(e2)
            X_train = np.stack(X_train)
            Y_train = np.stack(Y_train)

            X_test  = []
            Y_test = []

            for e1, e2 in zip(test_df[source_embed], test_df[target_embed]):
                e1, e2 = np.asarray(e1), np.asarray(e2)
                X_test.append(e1)
                Y_test.append(e2)
            X_test = np.stack(X_test)
            Y_test = np.stack(Y_test)

            # RidgeCV tuning in train split : 
            model = RidgeCV(alphas=alphas,cv=5, scoring="neg_mean_squared_error")

            model.fit(X_train, Y_train)
            Y_test_pred = model.predict(X_test)

            fold_metrics = residual_summary( translated=Y_test_pred, actual=Y_test)

            fold_metrics["dataset"] = dataset_name
            fold_metrics["config"] = config
            fold_metrics["source_col"] = source_embed
            fold_metrics["target_col"] = target_embed
            fold_metrics["fold"] = fold
            fold_metrics["chosen_alpha"] = float(model.alpha_)
            fold_metrics["train_size"] = len(train_df)
            fold_metrics["calibration_size"] = len(calib_df)
            fold_metrics["test_size"] = len(test_df)

            folds.append(fold_metrics)
            test_fold = fold

            fold_l2 = normalized_l2(translated=Y_test_pred,actual=Y_test)
            residuals.extend(fold_l2.tolist())

            # Save trained model for this fold rotation
            model_path = os.path.join(
                model_dir,
                f"ridgecv_{dataset_name}_{config}_testfold_{test_fold}.joblib",
            )

            joblib.dump(
                {
                    "model": model,
                    "dataset_name": dataset_name,
                    "repo_id": dataset_spec["repo_id"],
                    "config_name": config,
                    "source_col": source_embed,
                    "target_col": target_embed,
                    "test_fold": test_fold,
                    "calibration_fold": calib_idx,
                    "k": k,

                    "chosen_alpha": float(model.alpha_),
                },
                model_path,
            )

            model_paths.append(model_path)

            print("Chosen alpha:", model.alpha_)
            print("Fold metrics:", fold_metrics)
            print("Saved model:", model_path)

        # pooled over every test fold of this dataset/config
        pooled_metrics = pooled_residuals(residuals)
        pooled_metrics["dataset"] = dataset_name
        pooled_metrics["config"] = config
        pooled_metrics["source_col"] = source_embed
        pooled_metrics["target_col"] = target_embed
        pooled_metrics["fold"] = "pooled"
        folds.append(pooled_metrics)

        rows.extend(folds)

        # written after every config so a crashed run keeps its partial results
        results_df = pd.DataFrame(rows)
        id_cols = ["dataset", "config", "source_col", "target_col", "fold"]
        results_df = results_df[
            id_cols + [c for c in results_df.columns if c not in id_cols]
        ]
        for size_col in ["train_size", "calibration_size", "test_size"]:
            results_df[size_col] = results_df[size_col].astype("Int64")
        results_df.to_csv(results_path, index=False)

print("\nSaved results:", results_path)

"""
Experiment 4: multi-model basket conformal prediction (Self-Contained Version).

Compares best/median/worst single-model baselines against Basket A/B/C on
one survey pair, writing every 4.6 deliverable as CSVs.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import load_dataset
from scipy.linalg import orthogonal_procrustes
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import normalize

try:
    import cuml.accel
    cuml.accel.install()
except ImportError:
    pass


# =============================================================================
# CONSTANTS 
# =============================================================================

FOUNDATION_MODELS = [
    "astropt_15m", "astropt_850m", "astropt_95m",
    "convnext_base", "convnext_large", "convnext_nano", "convnext_tiny",
    "dino_base", "dino_giant", "dino_large", "dino_small",
    "ijepa_giant", "ijepa_huge",
    "vit_base", "vit_huge", "vit_large",
]

SURVEY_CONFIGS = {
    "legacy": {
        "hf_repo": "astro-conformal-project/split_legacysurvey_hsc_embeddings",
        "source_col": 2,
        "target_col": 1,
        "streaming": True,
        "requires_token": True,
    },
    "sdss": {
        "hf_repo": "astro-conformal-project/split_sdss_hsc_embeddings",
        "source_col": 1,
        "target_col": 2, 
        "streaming": True,
        "requires_token": True,
    },
    "jwst": {
        "hf_repo": "astro-conformal-project/split_jwst_hsc_embeddings",
        "source_col": 1,
        "target_col": 2,
        "streaming": True,
        "requires_token": True,
    },
}

MODEL_FAMILIES = {
    "astropt": ["astropt_15m", "astropt_850m", "astropt_95m"],
    "convnext": ["convnext_base", "convnext_large", "convnext_nano", "convnext_tiny"],
    "dino": ["dino_base", "dino_giant", "dino_large", "dino_small"],
    "ijepa": ["ijepa_giant", "ijepa_huge"],
    "vit": ["vit_base", "vit_huge", "vit_large"],
}

MATCHED_ALPHAS = (0.20, 0.10, 0.05)
RIDGE_ALPHAS = np.logspace(-4, 4, 25)
DEFAULT_SEEDS = (42, 44, 46)


# =============================================================================
# DATA LOADING 
# =============================================================================

def load_embedding_pair_with_objects(survey, model_name, split, hf_token=None):
    if survey not in SURVEY_CONFIGS:
        raise ValueError(f"Unknown survey '{survey}'. Choose from {list(SURVEY_CONFIGS)}")

    cfg = SURVEY_CONFIGS[survey]
    token = hf_token or os.environ.get("HF_TOKEN")
    if cfg["requires_token"] and not token:
        raise RuntimeError(
            f"Survey '{survey}' needs a Hugging Face token. Set the HF_TOKEN "
            "environment variable or pass it as hf_token to load_embedding_pair_with_objects()"
        )
    else:
        dataset = load_dataset(
            cfg["hf_repo"], model_name, split=split,
            streaming=True, token=token,
        )
        columns = list(dataset.features.keys())
        source_col = columns[cfg["source_col"]]
        target_col = columns[cfg["target_col"]]
        id_col = columns[0]
        
        X_list, Y_list, id = [], [], []
        for example in dataset:
            X_list.append(example[source_col])
            Y_list.append(example[target_col])
            id.append(example[id_col])

        X_stack = np.stack(X_list)
        Y_stack = np.stack(Y_list)

    X = normalize(X_stack, norm="l2", axis=1)
    Y = normalize(Y_stack, norm="l2", axis=1)
    ids = np.array(id)
    return X, Y, ids, source_col, target_col


# =============================================================================
# METRICS 
# =============================================================================

def normalized_l2(translated, actual, eps=1e-8):
    translated = np.asarray(translated)
    actual = np.asarray(actual)
    num = np.linalg.norm(actual - translated, axis=1)
    denom = np.linalg.norm(actual, axis=1)
    return num / (denom + eps)

def conformal_metrics_multi_alpha(X_train, Y_train, X_cal, Y_cal, X_test, Y_test, alphas=(0.20, 0.10, 0.05)):
    model = RidgeCV(alphas=RIDGE_ALPHAS, cv=5, scoring="neg_mean_squared_error")
    model.fit(X_train, Y_train)

    # Raw predictions matching Source 2 (no prediction L2-normalization)
    cal_preds = model.predict(X_cal)
    cal_residuals = normalized_l2(translated=cal_preds, actual=Y_cal)

    test_preds = model.predict(X_test)
    test_residuals = normalized_l2(translated=test_preds, actual=Y_test)

    n = len(cal_residuals)
    per_alpha = {}
    for alpha in alphas:
        q_level = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
        q_threshold = np.quantile(cal_residuals, q_level, method="higher")
        is_covered = test_residuals <= q_threshold
        coverage = np.mean(is_covered)
        per_alpha[alpha] = {
            "target_coverage": 1 - alpha,
            "empirical_coverage": float(coverage),
            "mean_radius": float(q_threshold),
            "coverage_error": float(abs(coverage - (1 - alpha))),
            "is_covered": is_covered,
        }

    return {"cal_residuals": cal_residuals, "test_residuals": test_residuals, "per_alpha": per_alpha}


# =============================================================================
# BASKETS 
# =============================================================================

def _load_flat(survey, model_names, split, hf_token=None):
    X_list, Y_list = [], []
    ids_out, source_col, target_col = None, None, None
    for model_name in model_names:
        X, Y, ids, src, tgt = load_embedding_pair_with_objects(survey, model_name, split, hf_token)
        X_list.append(X)
        Y_list.append(Y)
        source_col, target_col = src, tgt
        if ids_out is None:
            ids_out = ids
    return np.concatenate(X_list, axis=1), np.concatenate(Y_list, axis=1), ids_out, source_col, target_col

def _load_by_family(survey, split, hf_token=None, families=None):
    families = families or MODEL_FAMILIES
    family_X, family_Y = {}, {}
    ids_out, source_col, target_col = None, None, None
    for family, members in families.items():
        X_f, Y_f, ids, src, tgt = _load_flat(survey, members, split, hf_token)
        family_X[family] = X_f
        family_Y[family] = Y_f
        source_col, target_col = src, tgt
        if ids_out is None:
            ids_out = ids
    return family_X, family_Y, ids_out, source_col, target_col

def basket_a(survey, model_names, split, state=None, n_components=128, random_seed=42, hf_token=None):
    is_train = (split == "train")
    if not is_train and state is None:
        raise ValueError("state (returned from split='train') must be provided for non-train splits.")

    X_concat, Y_concat, ids, source_col, target_col = _load_flat(survey, model_names, split, hf_token)

    if is_train:
        mean_x = np.mean(X_concat, axis=0)
        mean_y = np.mean(Y_concat, axis=0)
        pca_x = PCA(n_components=n_components, random_state=random_seed)
        pca_y = PCA(n_components=n_components, random_state=random_seed)
        X_pca = pca_x.fit_transform(X_concat - mean_x)
        Y_pca = pca_y.fit_transform(Y_concat - mean_y)
        state = {"pca_x": pca_x, "pca_y": pca_y, "mean_x": mean_x, "mean_y": mean_y}
        return X_pca, Y_pca, ids, source_col, target_col, state

    X_pca = state["pca_x"].transform(X_concat - state["mean_x"])
    Y_pca = state["pca_y"].transform(Y_concat - state["mean_y"])
    return X_pca, Y_pca, ids, source_col, target_col

def basket_b(survey, split, state=None, families=None, n_components=32, random_seed=42, hf_token=None):
    is_train = (split == "train")
    if not is_train and state is None:
        raise ValueError("state (returned from split='train') must be provided for non-train splits.")

    families = families or MODEL_FAMILIES
    family_X, family_Y, ids, source_col, target_col = _load_by_family(survey, split, hf_token, families)

    fitted = {} if is_train else state
    proj_x, proj_y = [], []
    for family in family_X:
        X_f, Y_f = family_X[family], family_Y[family]
        if is_train:
            mean_x, mean_y = np.mean(X_f, axis=0), np.mean(Y_f, axis=0)
            pca_x = PCA(n_components=n_components, random_state=random_seed)
            pca_y = PCA(n_components=n_components, random_state=random_seed)
            X_pca = pca_x.fit_transform(X_f - mean_x)
            Y_pca = pca_y.fit_transform(Y_f - mean_y)
            fitted[family] = {"pca_x": pca_x, "pca_y": pca_y, "mean_x": mean_x, "mean_y": mean_y}
        else:
            if family not in state:
                raise ValueError(f"No fitted state for family '{family}'")
            f = state[family]
            X_pca = f["pca_x"].transform(X_f - f["mean_x"])
            Y_pca = f["pca_y"].transform(Y_f - f["mean_y"])
        proj_x.append(X_pca)
        proj_y.append(Y_pca)

    X_final = np.concatenate(proj_x, axis=1)
    Y_final = np.concatenate(proj_y, axis=1)
    if is_train:
        return X_final, Y_final, ids, source_col, target_col, fitted
    return X_final, Y_final, ids, source_col, target_col

def basket_c(survey, split, state=None, families=None, n_components=32, random_seed=42, reference_model="astropt_850m", hf_token=None):
    is_train = (split == "train")
    if not is_train and state is None:
        raise ValueError("state (returned from split='train') must be provided for non-train splits.")

    families = families or MODEL_FAMILIES

    # 1. Load data by family
    family_X, family_Y, ids_out, source_col, target_col = _load_by_family(survey, split, hf_token, families)
    
    # 2. Load the single reference model to act as the geometric target
    ref_X, ref_Y, _, _, _ = load_embedding_pair_with_objects(survey, reference_model, split, hf_token)

    fitted = {} if is_train else state
    proj_x, proj_y = {}, {}

    # Step A: Fit PCA on the reference model
    if is_train:
        mean_ref_x, mean_ref_y = np.mean(ref_X, axis=0), np.mean(ref_Y, axis=0)
        pca_ref_x = PCA(n_components=n_components, random_state=random_seed)
        pca_ref_y = PCA(n_components=n_components, random_state=random_seed)
        target_x = pca_ref_x.fit_transform(ref_X - mean_ref_x)
        target_y = pca_ref_y.fit_transform(ref_Y - mean_ref_y)
        fitted["reference"] = {"pca_x": pca_ref_x, "pca_y": pca_ref_y, "mean_x": mean_ref_x, "mean_y": mean_ref_y}
    else:
        f = state["reference"]
        target_x = f["pca_x"].transform(ref_X - f["mean_x"])
        target_y = f["pca_y"].transform(ref_Y - f["mean_y"])

    # Step B: Fit PCA on each Family 
    for family in family_X:
        X_f, Y_f = family_X[family], family_Y[family]
        if is_train:
            mean_x, mean_y = np.mean(X_f, axis=0), np.mean(Y_f, axis=0)
            pca_x = PCA(n_components=n_components, random_state=random_seed)
            pca_y = PCA(n_components=n_components, random_state=random_seed)
            X_pca = pca_x.fit_transform(X_f - mean_x)
            Y_pca = pca_y.fit_transform(Y_f - mean_y)
            fitted[family] = {"pca_x": pca_x, "pca_y": pca_y, "mean_x": mean_x, "mean_y": mean_y}
        else:
            f = state[family]
            X_pca = f["pca_x"].transform(X_f - f["mean_x"])
            Y_pca = f["pca_y"].transform(Y_f - f["mean_y"])
        proj_x[family] = X_pca
        proj_y[family] = Y_pca

    # Step C: Procrustes align each family to the reference model geometry
    aligned_x, aligned_y = [], []
    for family in family_X:
        if is_train:
            R_x, _ = orthogonal_procrustes(proj_x[family], target_x)
            R_y, _ = orthogonal_procrustes(proj_y[family], target_y)
            fitted[family]["R_x"] = R_x
            fitted[family]["R_y"] = R_y
        else:
            R_x, R_y = state[family]["R_x"], state[family]["R_y"]

        aligned_x.append(proj_x[family] @ R_x)
        aligned_y.append(proj_y[family] @ R_y)

    # Step D: Concatenate the aligned family representations
    X_final = np.concatenate(aligned_x, axis=1)
    Y_final = np.concatenate(aligned_y, axis=1)
    if is_train:
        return X_final, Y_final, ids_out, source_col, target_col, fitted
    return X_final, Y_final, ids_out, source_col, target_col


# =============================================================================
# COMPARISON 
# =============================================================================

def evaluate_single(survey, model_name, alphas=MATCHED_ALPHAS, hf_token=None):
    X_train, Y_train, _, _, _ = load_embedding_pair_with_objects(survey, model_name, "train", hf_token)
    X_cal, Y_cal, _, _, _ = load_embedding_pair_with_objects(survey, model_name, "calibration", hf_token)
    X_test, Y_test, ids_test, _, _ = load_embedding_pair_with_objects(survey, model_name, "test", hf_token)
    result = conformal_metrics_multi_alpha(X_train, Y_train, X_cal, Y_cal, X_test, Y_test, alphas)
    result["ids_test"] = ids_test
    return result

def evaluate_basket_a(survey, alphas=MATCHED_ALPHAS, random_seed=42, hf_token=None, model_names=None, n_components=128):
    model_names = model_names or FOUNDATION_MODELS
    X_train, Y_train, _, _, _, state = basket_a(
        survey, model_names, "train", random_seed=random_seed, hf_token=hf_token, n_components=n_components)
    X_cal, Y_cal, _, _, _ = basket_a(
        survey, model_names, "calibration", state=state, random_seed=random_seed, hf_token=hf_token)
    X_test, Y_test, ids_test, _, _ = basket_a(
        survey, model_names, "test", state=state, random_seed=random_seed, hf_token=hf_token)
    result = conformal_metrics_multi_alpha(X_train, Y_train, X_cal, Y_cal, X_test, Y_test, alphas)
    result["ids_test"] = ids_test
    return result

def evaluate_basket_b(survey, alphas=MATCHED_ALPHAS, random_seed=42, hf_token=None, families=None, n_components=32):
    X_train, Y_train, _, _, _, state = basket_b(
        survey, "train", random_seed=random_seed, hf_token=hf_token, families=families, n_components=n_components)
    X_cal, Y_cal, _, _, _ = basket_b(
        survey, "calibration", state=state, random_seed=random_seed, hf_token=hf_token, families=families)
    X_test, Y_test, ids_test, _, _ = basket_b(
        survey, "test", state=state, random_seed=random_seed, hf_token=hf_token, families=families)
    result = conformal_metrics_multi_alpha(X_train, Y_train, X_cal, Y_cal, X_test, Y_test, alphas)
    result["ids_test"] = ids_test
    return result

def evaluate_basket_c(survey, alphas=MATCHED_ALPHAS, random_seed=42, reference_model="astropt_850m", hf_token=None, families=None, n_components=32):
    X_train, Y_train, _, _, _, state = basket_c(
        survey, "train", random_seed=random_seed, reference_model=reference_model,
        hf_token=hf_token, families=families, n_components=n_components)
    X_cal, Y_cal, _, _, _ = basket_c(
        survey, "calibration", state=state, random_seed=random_seed, reference_model=reference_model,
        hf_token=hf_token, families=families)
    X_test, Y_test, ids_test, _, _ = basket_c(
        survey, "test", state=state, random_seed=random_seed, reference_model=reference_model,
        hf_token=hf_token, families=families)
    result = conformal_metrics_multi_alpha(X_train, Y_train, X_cal, Y_cal, X_test, Y_test, alphas)
    result["ids_test"] = ids_test
    return result

def standardized_radius(mean_radius, cal_residuals):
    scale = np.median(cal_residuals)
    return float(mean_radius / scale) if scale else float("nan")

def rescue_and_failure_rates(single_is_covered, basket_is_covered, single_ids, basket_ids):
    single_map = dict(zip(single_ids, single_is_covered))
    basket_map = dict(zip(basket_ids, basket_is_covered))
    common_ids = [i for i in single_ids if i in basket_map]

    missed_by_single = [i for i in common_ids if not single_map[i]]
    rescued = [i for i in missed_by_single if basket_map[i]]

    covered_by_single = [i for i in common_ids if single_map[i]]
    basket_failed = [i for i in covered_by_single if not basket_map[i]]

    rescue_rate = len(rescued) / len(missed_by_single) if missed_by_single else float("nan")
    failure_rate = len(basket_failed) / len(covered_by_single) if covered_by_single else float("nan")

    return {
        "n_common_objects": len(common_ids),
        "n_missed_by_single": len(missed_by_single),
        "n_rescued_by_basket": len(rescued),
        "rescue_rate": rescue_rate,
        "n_covered_by_single": len(covered_by_single),
        "n_basket_induced_failures": len(basket_failed),
        "basket_induced_failure_rate": failure_rate,
    }

def seed_stability_table(survey, seeds=DEFAULT_SEEDS, alpha=0.10, reference_model="astropt_850m", hf_token=None, model_names=None, n_components_a=128, n_components_bc=32):
    rows = []
    for seed in seeds:
        res_a = evaluate_basket_a(survey, alphas=(alpha,), random_seed=seed,
                                   hf_token=hf_token, model_names=model_names, n_components=n_components_a)
        res_b = evaluate_basket_b(survey, alphas=(alpha,), random_seed=seed,
                                   hf_token=hf_token, n_components=n_components_bc)
        res_c = evaluate_basket_c(survey, alphas=(alpha,), random_seed=seed,
                                   reference_model=reference_model, hf_token=hf_token,
                                   n_components=n_components_bc)
        for name, res in (("Basket_A", res_a), ("Basket_B", res_b), ("Basket_C", res_c)):
            pa = res["per_alpha"][alpha]
            rows.append({
                "Basket": name, "seed": seed,
                "empirical_coverage": pa["empirical_coverage"],
                "mean_radius": pa["mean_radius"],
                "standardized_radius": standardized_radius(pa["mean_radius"], res["cal_residuals"]),
            })

    raw_df = pd.DataFrame(rows)
    summary_df = raw_df.groupby("Basket").agg(
        coverage_mean=("empirical_coverage", "mean"),
        coverage_std=("empirical_coverage", "std"),
        std_radius_mean=("standardized_radius", "mean"),
        std_radius_std=("standardized_radius", "std"),
    ).reset_index()
    return raw_df, summary_df

def consensus_hard_objects(single_results, basket_results, alpha=0.10, min_single_miss_fraction=0.5):
    basket_uncovered_sets = []
    for res in basket_results.values():
        pa = res["per_alpha"][alpha]
        ids = res["ids_test"]
        basket_uncovered_sets.append(set(ids[~pa["is_covered"]]))
    uncovered_by_all_baskets = set.intersection(*basket_uncovered_sets) if basket_uncovered_sets else set()

    n_models = len(single_results)
    miss_count = {}
    for res in single_results.values():
        pa = res["per_alpha"][alpha]
        ids = res["ids_test"]
        for oid in ids[~pa["is_covered"]]:
            miss_count[oid] = miss_count.get(oid, 0) + 1

    rows = []
    for oid in uncovered_by_all_baskets:
        n_missed = miss_count.get(oid, 0)
        frac = n_missed / n_models if n_models else 0.0
        if frac >= min_single_miss_fraction:
            rows.append({
                "object_id": oid,
                "n_single_models_missed": n_missed,
                "n_single_models_total": n_models,
                "fraction_single_models_missed": frac,
            })

    if not rows:
        return pd.DataFrame(columns=[
            "object_id", "n_single_models_missed", "n_single_models_total", "fraction_single_models_missed"
        ])
    return pd.DataFrame(rows).sort_values("fraction_single_models_missed", ascending=False).reset_index(drop=True)


# =============================================================================
# MAIN ORCHESTRATION (Experiment 4)
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--survey", required=True, choices=sorted(SURVEY_CONFIGS))
    parser.add_argument(
        "--models", default=",".join(FOUNDATION_MODELS),
        help="comma-separated foundation models to run as singles AND as Basket A's "
             "member list (default: all 16)",
    )
    parser.add_argument("--ranking-alpha", type=float, default=0.10,
                         help="alpha used to rank best/median/worst single, and for "
                              "rescue rate / consensus-hard-objects / seed-stability (default 0.10)")
    parser.add_argument("--reference-model", default="astropt_850m", choices=FOUNDATION_MODELS,
                         help="Basket C's Procrustes reference model")
    parser.add_argument("--seeds", default=",".join(str(s) for s in DEFAULT_SEEDS),
                         help="comma-separated seeds for the seed-stability table")
    parser.add_argument("--min-single-miss-fraction", type=float, default=0.5,
                         help="fraction of single models that must miss an object, on top of "
                              "being missed by every basket, to count as consensus-hard")
    parser.add_argument("--out-dir", required=True, help="directory to write all result CSVs into")
    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    all_alphas = MATCHED_ALPHAS

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- single-model baselines ----
    single_results = {}
    for i, m in enumerate(models):
        print(f"[{i + 1}/{len(models)}] single: {m}", flush=True)
        single_results[m] = evaluate_single(args.survey, m, alphas=all_alphas)

    ranking = sorted(
        (
            (m, standardized_radius(r["per_alpha"][args.ranking_alpha]["mean_radius"], r["cal_residuals"]))
            for m, r in single_results.items()
        ),
        key=lambda t: t[1],
    )
    best_name, best_std_r = ranking[0]
    worst_name, _ = ranking[-1]
    median_name, _ = ranking[len(ranking) // 2]
    print(f"Best single: {best_name} | Median: {median_name} | Worst: {worst_name}", flush=True)

    # ---- baskets ----
    print("Evaluating Basket A", flush=True)
    basket_results = {"Basket_A": evaluate_basket_a(args.survey, alphas=all_alphas, model_names=models)}
    print("Evaluating Basket B", flush=True)
    basket_results["Basket_B"] = evaluate_basket_b(args.survey, alphas=all_alphas)
    print("Evaluating Basket C", flush=True)
    basket_results["Basket_C"] = evaluate_basket_c(
        args.survey, alphas=all_alphas, reference_model=args.reference_model)

    # ---- deliverable 1: summary tables (one per alpha) ----
    for alpha in MATCHED_ALPHAS:
        # Get the standardized radius of the "best" single model at THIS specific alpha
        # to ensure the relative efficiency calculation is mathematically accurate for the tier.
        best_single_res = single_results[best_name]
        best_pa = best_single_res["per_alpha"][alpha]
        current_best_std_r = standardized_radius(best_pa["mean_radius"], best_single_res["cal_residuals"])

        def make_row(label, res):
            pa = res["per_alpha"][alpha]
            std_r = standardized_radius(pa["mean_radius"], res["cal_residuals"])
            
            row = {
                "Model": label,
                "Survey": args.survey,
                "target_coverage_%": int(round((1 - alpha) * 100)),
                "empirical_coverage": pa["empirical_coverage"],
                "mean_radius": pa["mean_radius"],
                "standardized_radius": std_r,
                "coverage_error": pa["coverage_error"],
                "relative_efficiency": std_r / current_best_std_r if current_best_std_r else float("nan")
            }
            return row

        summary_rows = [
            make_row(f"Best single ({best_name})", single_results[best_name]),
            make_row(f"Median single ({median_name})", single_results[median_name]),
            make_row(f"Worst single ({worst_name})", single_results[worst_name]),
        ]
        for name, res in basket_results.items():
            summary_rows.append(make_row(name, res))
            
        summary_df = pd.DataFrame(summary_rows)
        
        # Format the alpha string for the filename (e.g., "0.2" -> "0_20")
        alpha_str = f"{alpha:.2f}".replace(".", "_")
        csv_path = out_dir / f"{args.survey}_basket_summary_alpha_{alpha_str}.csv"
        
        summary_df.to_csv(csv_path, index=False)
        print(f"\n=== Summary (Alpha = {alpha}) ===")
        print(summary_df.to_string(index=False))

    # ---- deliverable 2: coverage-standardized-radius curves ----
    curve_rows = []
    named_results = {f"single:{m}": r for m, r in single_results.items()}
    named_results.update(basket_results)
    for name, res in named_results.items():
        for a in MATCHED_ALPHAS:
            pa = res["per_alpha"][a]
            curve_rows.append({
                "Model": name,
                "alpha": a,
                "target_coverage": pa["target_coverage"],
                "empirical_coverage": pa["empirical_coverage"],
                "standardized_radius": standardized_radius(pa["mean_radius"], res["cal_residuals"]),
            })
    pd.DataFrame(curve_rows).to_csv(out_dir / f"{args.survey}_coverage_radius_curve.csv", index=False)

    # ---- deliverable 3: seed-stability table (seeds 42/44/46) ----
    seed_raw_df, seed_summary_df = seed_stability_table(
        args.survey, seeds=seeds, alpha=args.ranking_alpha,
        reference_model=args.reference_model, model_names=models,
    )
    seed_raw_df.to_csv(out_dir / f"{args.survey}_seed_stability_raw.csv", index=False)
    seed_summary_df.to_csv(out_dir / f"{args.survey}_seed_stability_summary.csv", index=False)

    # ---- deliverable 4: per-object residual pairs, single vs Basket C ----
    scatter_rows = []
    ref_res = basket_results["Basket_C"]
    ref_ids = ref_res["ids_test"]
    ref_residual_map = dict(zip(ref_ids, ref_res["test_residuals"]))
    ref_scale = np.median(ref_res["cal_residuals"])
    for m, res in single_results.items():
        ids = res["ids_test"]
        single_scale = np.median(res["cal_residuals"])
        single_residual_map = dict(zip(ids, res["test_residuals"]))
        for oid in set(ids) & set(ref_ids):
            scatter_rows.append({
                "object_id": oid,
                "single_model": m,
                "single_normalized_residual": single_residual_map[oid] / single_scale if single_scale else np.nan,
                "basket_normalized_residual": ref_residual_map[oid] / ref_scale if ref_scale else np.nan,
            })
    pd.DataFrame(scatter_rows).to_csv(out_dir / f"{args.survey}_residual_scatter_pairs.csv", index=False)

    # ---- deliverable 5: consensus-hard objects ----
    hard_df = consensus_hard_objects(
        single_results, basket_results, alpha=args.ranking_alpha,
        min_single_miss_fraction=args.min_single_miss_fraction,
    )
    # 1. Define clear, human-readable column names
    hard_rename_mapping = {
        "object_id": "Object_ID",
        "n_single_models_total": "Total_Single_Models",
        "n_single_models_missed": "Single_Models_Missed",
        "fraction_single_models_missed": "Miss_Fraction"
    }
    
    # 2. Reorder columns logically (ID first, then totals, misses, and fraction)
    hard_column_order = [
        "Object_ID",
        "Total_Single_Models",
        "Single_Models_Missed",
        "Miss_Fraction"
    ]
    
    # 3. Rename, reorder, and save
    hard_df = hard_df.rename(columns=hard_rename_mapping)
    hard_df = hard_df[hard_column_order]
    hard_df.to_csv(out_dir / f"{args.survey}_consensus_hard_objects.csv", index=False)

    # ---- rescue rate / basket-induced failure rate ----
    rescue_rows = []
    for basket_name, basket_res in basket_results.items():
        pa_b = basket_res["per_alpha"][args.ranking_alpha]
        for single_name, single_res in single_results.items():
            pa_s = single_res["per_alpha"][args.ranking_alpha]
            r = rescue_and_failure_rates(
                pa_s["is_covered"], pa_b["is_covered"], single_res["ids_test"], basket_res["ids_test"])
            r["single_model"] = single_name
            r["basket"] = basket_name
            rescue_rows.append(r)
    # 1. Define clear, human-readable column names
    rename_mapping = {
        "single_model": "Single_Model",
        "basket": "Ensemble_Basket",
        "n_common_objects": "Total_Evaluated_Objects",
        "n_missed_by_single": "Single_Model_Misses",
        "n_rescued_by_basket": "Rescued_By_Basket",
        "rescue_rate": "Rescue_Rate",
        "n_covered_by_single": "Single_Model_Successes",
        "n_basket_induced_failures": "Basket_Induced_Failures", 
        "basket_induced_failure_rate": "Basket_induced_failure_rate"
    }
    
    # 2. Reorder columns logically
    column_order = [
        "Single_Model", 
        "Ensemble_Basket", 
        "Total_Evaluated_Objects",
        "Single_Model_Misses", 
        "Rescued_By_Basket", 
        "Rescue_Rate",
        "Single_Model_Successes", 
        "Basket_Induced_Failures", 
        "Basket_induced_failure_rate"
    ]
    
    # 3. Create the dataframe, rename, reorder, and save
    df_rescue = pd.DataFrame(rescue_rows)
    df_rescue = df_rescue.rename(columns=rename_mapping)
    df_rescue = df_rescue[column_order]
    df_rescue.to_csv(out_dir / f"{args.survey}_rescue_rates.csv", index=False)
    
    print(f"\nAll outputs written to {out_dir}/", flush=True)


if __name__ == "__main__":
    sys.exit(main())
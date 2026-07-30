
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.constants import FOUNDATION_MODELS, SURVEY_CONFIGS
from src.data import load_embedding_pair
from src.metrics import kfold_residual_metrics

try:
    import cuml.accel
    cuml.accel.install()
except ImportError:
    pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--survey", required=True, choices=sorted(SURVEY_CONFIGS),
        help="source survey to align onto HSC",
    )
    parser.add_argument(
        "--models", default=",".join(FOUNDATION_MODELS),
        help="comma-separated foundation model names to run (default: all 16)",
    )
    parser.add_argument(
        "--split", default=None,
        help="HF split to load. Defaults to 'train+test+calibration' for "
             "'legacy' and 'train' for 'sdss'/'jwst', matching the original "
             "notebooks/scripts.",
    )
    parser.add_argument("--out", required=True, help="output CSV path for results")
    parser.add_argument("--k", type=int, default=5, help="number of CV folds")
    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    default_split = "train+test+calibration" 
    split = args.split or default_split

    print(f"Running {len(models)} models on '{args.survey}' (split={split!r}): {models}", flush=True)

    rows = []
    for i, model_name in enumerate(models):
        print(f"[{i + 1}/{len(models)}] {model_name}", flush=True)
        X, Y, source_col, target_col = load_embedding_pair(args.survey, model_name, split)
        results = kfold_residual_metrics(X, Y, k=args.k)
        row = {
            "Model": model_name,
            "Survey_pair": f"{source_col}\u2192{target_col}",
            "Mean_residual": results["Mean_Residual_Norm"],
            "Median_residual": results["Median_Residual_Norm"],
            "90th_percentile_residual": results["90th_Percentile_Residual_Norm"],
        }
        print(row, flush=True)
        rows.append(row)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Saved {len(rows)} rows to {out_path}", flush=True)


if __name__ == "__main__":
    sys.exit(main())

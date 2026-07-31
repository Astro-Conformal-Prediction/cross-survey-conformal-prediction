"""Entry point: run split conformal prediction over all survey pairs.

Usage:
    python run_experiment.py
    python run_experiment.py --pairs legacy_hsc sdss_hsc --outdir results/
"""

import argparse
from pathlib import Path

from pipeline import COVERAGES, MODELS, full_pipeline

PAIRS = {
    "legacy_hsc": "astro-conformal-project/split_legacysurvey_hsc_embeddings",
    "jwst_hsc": "astro-conformal-project/split_jwst_hsc_embeddings",
    "sdss_hsc": "astro-conformal-project/split_sdss_hsc_embeddings",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", nargs="+", default=list(PAIRS), choices=list(PAIRS))
    parser.add_argument("--models", nargs="+", default=MODELS)
    parser.add_argument("--coverages", nargs="+", type=float, default=COVERAGES)
    parser.add_argument("--outdir", default="results")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    for pair in args.pairs:
        print(f"=== {pair} ===")
        df = full_pipeline(
            PAIRS[pair], models=args.models, coverages=args.coverages
        )
        df = df.sort_values(["target_coverage", "radius"]).reset_index(drop=True)

        path = outdir / f"{pair}.csv"
        df.to_csv(path, index=False)
        print(df.to_string(index=False))
        print(f"-> {path}\n")


if __name__ == "__main__":
    main()

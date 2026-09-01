"""
fetch_galaxy_images.py
Run this ON THE CLUSTER where HuggingFace is accessible.

Fetches galaxy cutout images for the 6 object categories and builds
Figure 5 (image panels) for the paper.

Usage:
    cd ~/cross-survey-conformal-prediction
    python adaptive-conformal/fetch_galaxy_images.py \
        --per_object /projects/bfir/ssourav/exp4_results/full/per_object_results.csv \
        --out_dir    /projects/bfir/ssourav/exp4_results/figures
"""

import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
warnings.filterwarnings("ignore")

# ── HF dataset registry ────────────────────────────────────────────────────
DATASET_CONFIGS = {
    "legacysurvey_hsc": {
        "repo_id":    "astro-conformal-project/split_legacysurvey_hsc_embeddings",
        "config":     "astropt_850m",
        "src_img_col": None,   # fill in after probing
        "tgt_img_col": None,
        "id_col":     "object_id",
    },
    "jwst_hsc": {
        "repo_id":    "astro-conformal-project/split_jwst_hsc_embeddings",
        "config":     "astropt_850m",
        "src_img_col": None,
        "tgt_img_col": None,
        "id_col":     "object_id",
    },
}

CATEGORY_LABELS = {
    "adaptive_rescue":       "Global MISSED → Adaptive COVERED\n(Adaptive rescue)",
    "silent_failure":        "Global COVERED → Adaptive MISSED\n(Silent failure)",
    "easy_control":          "Both covered  (easy control)",
    "hard_covered":          "Hard, correctly covered",
    "overestimated":         "High diff, low residual\n(Difficulty overestimated)",
    "underestimated":        "Low diff, high residual\n(Difficulty underestimated)",
}
CAT_COLORS = {
    "adaptive_rescue":  "#2ecc71",
    "silent_failure":   "#e74c3c",
    "easy_control":     "#3498db",
    "hard_covered":     "#9b59b6",
    "overestimated":    "#f39c12",
    "underestimated":   "#1abc9c",
}


def select_objects(obj, survey, backbone="astropt_850m", tc=0.9, n_per_cat=4):
    sub = obj[
        (obj.survey_pair == survey) &
        (obj.backbone == backbone) &
        (obj.target_coverage == tc)
    ].copy()

    q10_d = sub.predicted_difficulty.quantile(0.10)
    q90_d = sub.predicted_difficulty.quantile(0.90)
    q10_r = sub.true_residual.quantile(0.10)
    q90_r = sub.true_residual.quantile(0.90)

    cats = {
        "adaptive_rescue":
            sub[(~sub.global_covered) & (sub.adaptive_covered)].head(n_per_cat),
        "silent_failure":
            sub[(sub.global_covered) & (~sub.adaptive_covered)].head(n_per_cat),
        "easy_control":
            sub[(sub.global_covered) & (sub.adaptive_covered) &
                (sub.predicted_difficulty <= q10_d)].head(n_per_cat),
        "hard_covered":
            sub[(sub.global_covered) & (sub.adaptive_covered) &
                (sub.predicted_difficulty >= q90_d)].head(n_per_cat),
        "overestimated":
            sub[(sub.predicted_difficulty >= q90_d) &
                (sub.true_residual <= q10_r)].head(n_per_cat),
        "underestimated":
            sub[(sub.predicted_difficulty <= q10_d) &
                (sub.true_residual >= q90_r)].head(n_per_cat),
    }
    return cats, sub


def load_hf_splits(repo_id, config):
    """Load all splits and concatenate into one DataFrame."""
    from datasets import load_dataset
    ds = load_dataset(repo_id, config)
    frames = []
    for split_name in ds.keys():
        df = ds[split_name].to_pandas()
        df["_split"] = split_name
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def probe_image_columns(df):
    """Find columns that look like image data (PIL Image or bytes)."""
    from PIL import Image
    img_cols = []
    for col in df.columns:
        sample = df[col].iloc[0]
        if isinstance(sample, Image.Image):
            img_cols.append((col, "PIL"))
        elif isinstance(sample, bytes):
            img_cols.append((col, "bytes"))
        elif isinstance(sample, dict) and "bytes" in sample:
            img_cols.append((col, "dict_bytes"))
    return img_cols


def get_image(row_val):
    """Convert various HF image formats to numpy array."""
    from PIL import Image
    import io
    if isinstance(row_val, Image.Image):
        return np.array(row_val.convert("RGB"))
    if isinstance(row_val, bytes):
        return np.array(Image.open(io.BytesIO(row_val)).convert("RGB"))
    if isinstance(row_val, dict) and "bytes" in row_val:
        return np.array(Image.open(io.BytesIO(row_val["bytes"])).convert("RGB"))
    return None


def build_image_panel(cats, full_df, id_col, src_img_col, tgt_img_col,
                      survey, out_dir, n_per_cat=4):
    """Build one figure: rows = categories, cols = objects (src | tgt pairs)."""
    n_cats = len(cats)
    n_cols = n_per_cat * 2   # src + tgt per object

    fig, axes = plt.subplots(n_cats, n_cols,
                              figsize=(n_cols * 2.2, n_cats * 2.5))

    for row_i, (cat_key, cat_df) in enumerate(cats.items()):
        color = CAT_COLORS[cat_key]
        for obj_i, (_, obj_row) in enumerate(cat_df.iterrows()):
            if obj_i >= n_per_cat:
                break

            row_id = obj_row["row_id"]
            match  = full_df[full_df[id_col] == row_id]
            if match.empty:
                for k in range(2):
                    ax = axes[row_i][obj_i * 2 + k]
                    ax.set_facecolor("#eeeeee")
                    ax.text(0.5, 0.5, f"ID {row_id}\nnot found",
                            ha="center", va="center", transform=ax.transAxes,
                            fontsize=7)
                    ax.axis("off")
                continue

            hf_row = match.iloc[0]

            for k, (img_col, label) in enumerate(
                    [(src_img_col, "Source"), (tgt_img_col, "Target")]):
                ax = axes[row_i][obj_i * 2 + k]
                try:
                    img = get_image(hf_row[img_col]) if img_col else None
                except Exception:
                    img = None

                if img is not None:
                    ax.imshow(img)
                else:
                    ax.set_facecolor("#dddddd")
                    ax.text(0.5, 0.5, f"{label}\nno image",
                            ha="center", va="center",
                            transform=ax.transAxes, fontsize=7)

                ax.axis("off")
                if obj_i == 0 and k == 0:
                    ax.set_ylabel(CATEGORY_LABELS[cat_key],
                                  fontsize=8, color=color,
                                  fontweight="bold", labelpad=4)

                # annotate bottom
                ax.set_title(
                    f"r={obj_row.true_residual:.4f}\n"
                    f"σ̂={obj_row.predicted_difficulty:.4f}",
                    fontsize=6, pad=2,
                )

        # left spine colour per category
        for j in range(n_cols):
            for spine in axes[row_i][j].spines.values():
                spine.set_edgecolor(color)
                spine.set_linewidth(1.5)

    fig.suptitle(
        f"Figure 5 — Galaxy examples by object category\n{survey}  ·  AstroPT-850M  ·  90%",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = Path(out_dir) / f"fig5_galaxy_panels_{survey}.png"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  saved → {out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--per_object", required=True)
    p.add_argument("--out_dir",    required=True)
    p.add_argument("--survey",     default="legacysurvey_hsc")
    p.add_argument("--backbone",   default="astropt_850m")
    p.add_argument("--n_per_cat",  type=int, default=4)
    args = p.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    obj = pd.read_csv(args.per_object)
    cats, _ = select_objects(obj, args.survey, args.backbone,
                              n_per_cat=args.n_per_cat)

    print(f"Loading HF dataset for {args.survey} …")
    cfg  = DATASET_CONFIGS[args.survey]
    full_df = load_hf_splits(cfg["repo_id"], cfg["config"])
    print(f"  loaded {len(full_df)} rows, columns: {full_df.columns.tolist()[:12]}")

    img_cols = probe_image_columns(full_df)
    print(f"  image columns: {img_cols}")

    if not img_cols:
        print("  No image columns found — saving object ID list only")
        for cat_key, cat_df in cats.items():
            print(f"  {cat_key}: row_ids = {cat_df.row_id.tolist()}")
        return

    # pick first src/tgt image columns heuristically
    src_img = next((c for c,_ in img_cols if "legacy" in c.lower() or "src" in c.lower()
                    or "sdss" in c.lower() or "hsc" in c.lower()), img_cols[0][0])
    tgt_img = next((c for c,_ in img_cols if "jwst" in c.lower() or "tgt" in c.lower()
                    or c != src_img), img_cols[-1][0])

    print(f"  src_img_col={src_img}  tgt_img_col={tgt_img}")

    build_image_panel(cats, full_df,
                      id_col=cfg["id_col"],
                      src_img_col=src_img,
                      tgt_img_col=tgt_img,
                      survey=args.survey,
                      out_dir=args.out_dir,
                      n_per_cat=args.n_per_cat)


if __name__ == "__main__":
    main()

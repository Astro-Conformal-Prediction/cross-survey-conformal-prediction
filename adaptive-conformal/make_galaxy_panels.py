"""
make_galaxy_panels.py
Run on the cluster (gh-login01 or a GPU node with HF access).

Loads Smith42/legacysurvey_hsc_crossmatched and
astro-conformal-project/split_legacysurvey_hsc_embeddings,
joins on object_id, then builds Figure 5 (galaxy image panels)
for the six object categories.

Usage
-----
python make_galaxy_panels.py \
    --per_object /projects/bfir/ssourav/exp4_results/full/per_object_results.csv \
    --out_dir    /projects/bfir/ssourav/exp4_results/figures

Optional
--------
--survey      legacysurvey_hsc   (or jwst_hsc)
--backbone    astropt_850m
--n_per_cat   4
"""

import argparse
import io
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
warnings.filterwarnings("ignore")

# ── category config ────────────────────────────────────────────────────────
CATEGORY_LABELS = {
    "adaptive_rescue":  "Global MISSED → Adaptive COVERED\n(Adaptive rescue)",
    "silent_failure":   "Global COVERED → Adaptive MISSED\n(Silent failure)",
    "easy_control":     "Both covered — easy control\n(low predicted difficulty)",
    "hard_covered":     "Both covered — hard control\n(high predicted difficulty)",
    "overestimated":    "High predicted difficulty, low residual\n(difficulty overestimated)",
    "underestimated":   "Low predicted difficulty, high residual\n(difficulty underestimated)",
}
CAT_COLORS = {
    "adaptive_rescue": "#2ecc71",
    "silent_failure":  "#e74c3c",
    "easy_control":    "#3498db",
    "hard_covered":    "#9b59b6",
    "overestimated":   "#f39c12",
    "underestimated":  "#1abc9c",
}

# ── dataset registry ───────────────────────────────────────────────────────
CROSSMATCH = {
    "legacysurvey_hsc": {
        "repo_id":   "Smith42/legacysurvey_hsc_crossmatched",
        "id_col":    "legacysurvey_object_id",
        "src_label": "Legacy Survey (grz RGB)",
        "tgt_label": "HSC (RGB)",
    },
    "jwst_hsc": {
        "repo_id":   "Smith42/jwst_hsc_crossmatched",
        "id_col":    "hsc_object_id",
        "src_label": "HSC (RGB)",
        "tgt_label": "JWST (RGB)",
    },
}
EMBED_REPOS = {
    "legacysurvey_hsc": "astro-conformal-project/split_legacysurvey_hsc_embeddings",
    "jwst_hsc":         "astro-conformal-project/split_jwst_hsc_embeddings",
}


# ── helpers ────────────────────────────────────────────────────────────────

def pil_to_arr(img):
    import numpy as np
    from PIL import Image
    if img is None:
        return None
    if isinstance(img, Image.Image):
        return np.array(img.convert("RGB"))
    if isinstance(img, bytes):
        return np.array(Image.open(io.BytesIO(img)).convert("RGB"))
    if isinstance(img, dict) and "bytes" in img:
        return np.array(Image.open(io.BytesIO(img["bytes"])).convert("RGB"))
    return None


def flux_to_rgb(flux_dict, bands_pref=("hsc-i", "hsc-r", "hsc-g")):
    """
    Convert flux dict (from legacysurvey_image / hsc_image columns)
    to an RGB uint8 array using asinh stretch.
    flux_dict = {"band": [...], "flux": [[[...]]]}
    """
    if flux_dict is None:
        return None
    try:
        bands = flux_dict["band"]
        flux  = np.array(flux_dict["flux"])   # (n_bands, H, W)
        # pick three bands for RGB
        chosen = []
        for want in bands_pref:
            if want in bands:
                chosen.append(bands.index(want))
        # fallback: first three
        if len(chosen) < 3:
            chosen = list(range(min(3, len(bands))))
        if len(chosen) < 3:
            return None

        r = flux[chosen[0]]
        g = flux[chosen[1]]
        b = flux[chosen[2]]

        # asinh stretch per channel
        def stretch(x):
            x = np.arcsinh(x / (0.1 * np.percentile(np.abs(x) + 1e-6, 95) + 1e-6))
            lo, hi = np.percentile(x, [1, 99])
            x = np.clip((x - lo) / (hi - lo + 1e-6), 0, 1)
            return (x * 255).astype(np.uint8)

        rgb = np.stack([stretch(r), stretch(g), stretch(b)], axis=-1)
        return rgb
    except Exception:
        return None


def select_categories(obj, survey, backbone, tc=0.9, n_per_cat=4):
    sub = obj[
        (obj.survey_pair == survey) &
        (obj.backbone == backbone) &
        (obj.target_coverage == tc)
    ].copy()

    q10d = sub.predicted_difficulty.quantile(0.10)
    q90d = sub.predicted_difficulty.quantile(0.90)
    q10r = sub.true_residual.quantile(0.10)
    q90r = sub.true_residual.quantile(0.90)

    cats = {
        "adaptive_rescue":
            sub[(~sub.global_covered) & (sub.adaptive_covered)].head(n_per_cat),
        "silent_failure":
            sub[(sub.global_covered) & (~sub.adaptive_covered)].head(n_per_cat),
        "easy_control":
            sub[(sub.global_covered) & (sub.adaptive_covered) &
                (sub.predicted_difficulty <= q10d)].head(n_per_cat),
        "hard_covered":
            sub[(sub.global_covered) & (sub.adaptive_covered) &
                (sub.predicted_difficulty >= q90d)].head(n_per_cat),
        "overestimated":
            sub[(sub.predicted_difficulty >= q90d) &
                (sub.true_residual <= q10r)].head(n_per_cat),
        "underestimated":
            sub[(sub.predicted_difficulty <= q10d) &
                (sub.true_residual >= q90r)].head(n_per_cat),
    }
    return cats, sub


def load_crossmatch(repo_id, split="train"):
    from datasets import load_dataset
    print(f"  Loading crossmatch {repo_id} …")
    ds = load_dataset(repo_id, split=split)
    return ds.to_pandas()


def load_embed_with_ids(repo_id, config):
    """Load embedding splits and return a DataFrame with object_id + split info."""
    from datasets import load_dataset
    print(f"  Loading embeddings {repo_id} / {config} …")
    ds = load_dataset(repo_id, config)
    frames = []
    for sp in ds.keys():
        df = ds[sp].to_pandas()
        df["_split"] = sp
        df["_row_idx"] = range(len(df))
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def get_image_for_object(xmatch_df, id_col, obj_id):
    """
    Returns (src_img_arr, tgt_img_arr) for a given object ID.
    Tries `rgb` column first (PIL image), then flux arrays.
    """
    match = xmatch_df[xmatch_df[id_col].astype(str) == str(obj_id)]
    if match.empty:
        return None, None

    row = match.iloc[0]

    # ── source image ──────────────────────────────────────────────────────
    src_img = None
    if "rgb" in row.index and row["rgb"] is not None:
        src_img = pil_to_arr(row["rgb"])

    if src_img is None and "legacysurvey_image" in row.index:
        src_img = flux_to_rgb(row["legacysurvey_image"],
                               bands_pref=("des-i", "des-r", "des-g"))

    # ── target image ──────────────────────────────────────────────────────
    tgt_img = None
    if "hsc_image" in row.index:
        tgt_img = flux_to_rgb(row["hsc_image"],
                               bands_pref=("hsc-i", "hsc-r", "hsc-g"))
    elif "jwst_image" in row.index:
        tgt_img = flux_to_rgb(row["jwst_image"],
                               bands_pref=("f200w", "f150w", "f115w"))

    return src_img, tgt_img


# ── build figure ───────────────────────────────────────────────────────────

def build_panel(cats, xmatch_df, id_col, embed_df, embed_id_col,
                src_label, tgt_label, survey, out_dir, n_per_cat=4):

    n_cats = len(cats)
    n_cols = n_per_cat * 2   # src + tgt per object

    fig = plt.figure(figsize=(n_cols * 2.4, n_cats * 2.8))
    gs_outer = gridspec.GridSpec(n_cats, 1, figure=fig, hspace=0.55)

    for row_i, (cat_key, cat_df) in enumerate(cats.items()):
        color = CAT_COLORS[cat_key]
        gs_inner = gridspec.GridSpecFromSubplotSpec(
            1, n_cols, subplot_spec=gs_outer[row_i], wspace=0.05
        )
        axes = [fig.add_subplot(gs_inner[0, j]) for j in range(n_cols)]

        # category label on left
        axes[0].set_ylabel(CATEGORY_LABELS[cat_key],
                           fontsize=8, color=color, fontweight="bold",
                           labelpad=4)

        filled = 0
        for _, obj_row in cat_df.iterrows():
            if filled >= n_per_cat:
                break

            row_id  = obj_row["row_id"]

            # get object_id from embed_df by positional row_id
            # row_id is the index within the test split
            test_rows = embed_df[embed_df["_split"] == "test"].reset_index(drop=True)
            if row_id >= len(test_rows):
                obj_id = None
            else:
                obj_id = test_rows.iloc[int(row_id)][embed_id_col] \
                    if embed_id_col in test_rows.columns else None

            src_img, tgt_img = (None, None)
            if obj_id is not None:
                src_img, tgt_img = get_image_for_object(xmatch_df, id_col, obj_id)

            for k, (img, label) in enumerate([(src_img, src_label),
                                               (tgt_img, tgt_label)]):
                ax = axes[filled * 2 + k]
                ax.axis("off")
                for spine in ax.spines.values():
                    spine.set_edgecolor(color)
                    spine.set_linewidth(1.8)
                    spine.set_visible(True)

                if img is not None:
                    ax.imshow(img)
                else:
                    ax.set_facecolor("#e8e8e8")
                    ax.text(0.5, 0.5, f"{label}\n(ID: {obj_id})",
                            ha="center", va="center",
                            transform=ax.transAxes, fontsize=6.5, color="#555")

                if row_i == 0:
                    ax.set_title(label, fontsize=7, color="gray", pad=2)

                # metric annotation
                info = (f"r={obj_row.true_residual:.4f}\n"
                        f"σ̂={obj_row.predicted_difficulty:.4f}\n"
                        f"R={obj_row.adaptive_radius:.4f}")
                if k == 0:
                    ax.text(0.02, 0.02, info, transform=ax.transAxes,
                            fontsize=5.5, va="bottom", color="white",
                            bbox=dict(boxstyle="round,pad=0.15",
                                      fc="black", alpha=0.5))

            filled += 1

        # blank remaining slots
        for j in range(filled * 2, n_cols):
            axes[j].axis("off")

    fig.suptitle(
        f"Figure 5 — Galaxy image panels by object category\n"
        f"{survey}  ·  AstroPT-850M  ·  Ridge difficulty model  ·  90% coverage",
        fontsize=12, fontweight="bold",
    )
    out = Path(out_dir) / f"fig5_galaxy_panels_{survey}.png"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  Saved → {out}")


# ── main ───────────────────────────────────────────────────────────────────

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
    cats, _ = select_categories(obj, args.survey, args.backbone,
                                  n_per_cat=args.n_per_cat)

    print(f"\nSelected objects per category:")
    for k, v in cats.items():
        print(f"  {k}: {len(v)} objects")

    cfg = CROSSMATCH[args.survey]

    # Load crossmatch dataset (has images)
    xmatch_df = load_crossmatch(cfg["repo_id"])
    print(f"  Crossmatch columns: {[c for c in xmatch_df.columns if not c.endswith('image')][:15]}")

    # Load embedding dataset (has object_ids aligned with row_id)
    embed_repo = EMBED_REPOS[args.survey]
    embed_df   = load_embed_with_ids(embed_repo, args.backbone)
    print(f"  Embed columns: {[c for c in embed_df.columns if 'embed' not in c.lower()][:10]}")

    # Find the object_id column in embed dataset
    id_candidates = ["object_id", "legacysurvey_object_id", "hsc_object_id",
                     "source_id", "objid"]
    embed_id_col = next((c for c in id_candidates if c in embed_df.columns), None)
    print(f"  Embed ID column: {embed_id_col}")
    print(f"  Crossmatch ID column: {cfg['id_col']}")

    build_panel(
        cats=cats,
        xmatch_df=xmatch_df,
        id_col=cfg["id_col"],
        embed_df=embed_df,
        embed_id_col=embed_id_col,
        src_label=cfg["src_label"],
        tgt_label=cfg["tgt_label"],
        survey=args.survey,
        out_dir=args.out_dir,
        n_per_cat=args.n_per_cat,
    )
    print("\nDone.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
fig5_subfigures.py

Generate clean, manuscript-ready subfigure files for every category.

For each category, write FOUR independent panels:
  1) Legacy Survey — Flagged
  2) Legacy Survey — Matched control
  3) HSC           — Flagged
  4) HSC           — Matched control

Each panel contains only the image cutouts + small "Ex. 1", "Ex. 2", ... labels.
No category title, no Flagged/Control row labels, no survey title, no deltas
inside the image file. Those are supplied by Overleaf/AASTeX subcaptions.

This is intentionally minimalist: it lets LaTeX control the 2x2 layout and
keeps each scientific image large enough to inspect.

Outputs:
  /projects/bfir/ssourav/exp4_results/figures/subfigures/
      adaptive_rescue_legacy_flagged.pdf/png
      adaptive_rescue_legacy_control.pdf/png
      adaptive_rescue_hsc_flagged.pdf/png
      adaptive_rescue_hsc_control.pdf/png
      ...
      matching_deltas.csv

Important display choice:
  For each survey and flagged/control pair, both images use a genuinely shared
  asinh normalization (same scale AND same low/high display limits).

Run:
  python3 /projects/bfir/ssourav/fig5_subfigures.py
"""

import io
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
PAIRS_CSV = "/projects/bfir/ssourav/exp4_results/matched_pairs.csv"
PER_OBJ   = "/projects/bfir/ssourav/exp4_results/full/per_object_results.csv"
CACHE     = "/projects/bfir/ssourav/exp4_results/found_objects.pkl"
OUT_DIR   = Path("/projects/bfir/ssourav/exp4_results/figures/subfigures_v3")

# -----------------------------------------------------------------------------
# Categories
# -----------------------------------------------------------------------------
CATEGORIES = [
    "adaptive_rescue",
    "silent_failure",
    "hard_correct",
    "false_alarm",
    "adaptive_regression",
]

CATEGORY_LABELS = {
    "adaptive_rescue": "Adaptive rescue",
    "silent_failure": "Silent failure",
    "hard_correct": "Correctly anticipated hard",
    "false_alarm": "False alarm",
    "adaptive_regression": "Adaptive regression",
}

SRC_BW = ["des-i", "des-r", "des-g"]
TGT_BW = ["hsc-i", "hsc-r", "hsc-g"]

# -----------------------------------------------------------------------------
# Publication defaults
# -----------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 7.0,
    "axes.titlesize": 7.0,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.facecolor": "white",
    "savefig.edgecolor": "white",
})

# -----------------------------------------------------------------------------
# Load
# -----------------------------------------------------------------------------
print("Loading cache...", flush=True)

with open(CACHE, "rb") as f:
    cache = pickle.load(f)

found      = cache["found"]
row_to_oid = cache["row_to_oid"]

pairs_df = pd.read_csv(PAIRS_CSV)
obj      = pd.read_csv(PER_OBJ)

sub = obj[
    (obj.survey_pair == "legacysurvey_hsc") &
    (obj.backbone == "astropt_850m") &
    (obj.target_coverage == 0.9)
].set_index("row_id")

OUT_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# Pair/result helpers
# -----------------------------------------------------------------------------
def get_pairs(cat_key):
    """
    Preserve the pairing convention used by the original plotting script:
    pair flagged/control rows in their existing CSV order.
    """
    fr = pairs_df[
        (pairs_df.category == cat_key) &
        (pairs_df.type == "flagged")
    ]
    cr = pairs_df[
        (pairs_df.category == cat_key) &
        (pairs_df.type == "control")
    ]

    if len(fr) != len(cr):
        print(
            f"WARNING {cat_key}: {len(fr)} flagged rows, {len(cr)} controls; "
            f"using {min(len(fr), len(cr))} pairs.",
            flush=True,
        )

    return [
        (int(r.row_id), int(c.row_id))
        for (_, r), (_, c) in zip(fr.iterrows(), cr.iterrows())
    ]


def get_sub_record(row_id):
    """
    Return a single per-object result row.

    Some row_id values occur multiple times in per_object_results.csv.
    If duplicates exist, verify the plotted matching fields and use the first
    row. A warning is emitted only if duplicate values disagree.
    """
    if row_id not in sub.index:
        return None

    rec = sub.loc[row_id]

    if isinstance(rec, pd.DataFrame):
        for field in ("true_residual", "global_radius"):
            if field not in rec.columns:
                continue

            vals = pd.to_numeric(rec[field], errors="coerce").dropna().to_numpy()

            if (
                vals.size > 1 and
                not np.allclose(vals, vals[0], rtol=1e-7, atol=1e-10)
            ):
                print(
                    f"WARNING row_id={row_id}: duplicate {field} values "
                    f"{vals.tolist()}; using first row.",
                    flush=True,
                )

        rec = rec.iloc[0]

    return rec


# -----------------------------------------------------------------------------
# Image helpers
# -----------------------------------------------------------------------------
def pil_to_arr(img):
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


def extract_three_bands(fd, bands_wanted):
    if not fd:
        return None

    try:
        bands = list(fd["band"])
        flux = np.asarray(fd["flux"], dtype=np.float64)

        idx = [bands.index(b) for b in bands_wanted if b in bands]

        if len(idx) < 3:
            idx = list(range(min(3, len(bands))))

        if len(idx) < 3:
            return None

        cube = flux[idx[:3]]

        if cube.ndim != 3:
            return None

        return cube
    except Exception:
        return None


def finite_concat(a, b):
    vals = []

    for x in (a, b):
        if x is None:
            continue

        q = np.asarray(x, dtype=np.float64)
        q = q[np.isfinite(q)]

        if q.size:
            vals.append(q)

    if not vals:
        return np.array([0.0, 1.0], dtype=np.float64)

    return np.concatenate(vals)


def shared_asinh_rgb_pair(flag_fd, ctrl_fd, bands_wanted):
    """
    Build flagged/control RGB images with one shared normalization per channel.
    """
    f_cube = extract_three_bands(flag_fd, bands_wanted)
    c_cube = extract_three_bands(ctrl_fd, bands_wanted)

    if f_cube is None or c_cube is None:
        return None, None

    f_out = np.zeros_like(f_cube, dtype=np.float64)
    c_out = np.zeros_like(c_cube, dtype=np.float64)

    eps = 1e-10

    for k in range(3):
        fv = f_cube[k]
        cv = c_cube[k]

        joint = finite_concat(fv, cv)

        scale = 0.10 * np.nanpercentile(np.abs(joint) + eps, 95) + eps

        fs = np.arcsinh(fv / scale)
        cs = np.arcsinh(cv / scale)

        joint_s = finite_concat(fs, cs)

        lo, hi = np.nanpercentile(joint_s, [1.0, 99.0])

        if not np.isfinite(lo):
            lo = 0.0

        if not np.isfinite(hi) or hi <= lo:
            hi = lo + 1.0

        f_out[k] = np.clip((fs - lo) / (hi - lo), 0.0, 1.0)
        c_out[k] = np.clip((cs - lo) / (hi - lo), 0.0, 1.0)

    # Requested band order is [i, r, g], matching the original script's RGB.
    f_rgb = (np.moveaxis(f_out, 0, -1) * 255).astype(np.uint8)
    c_rgb = (np.moveaxis(c_out, 0, -1) * 255).astype(np.uint8)

    return f_rgb, c_rgb


def shared_rescale_rendered_pair(f_rgb, c_rgb):
    """
    Shared rescale for already-rendered RGB fallback images.
    """
    if f_rgb is None or c_rgb is None:
        return f_rgb, c_rgb

    f = np.asarray(f_rgb, dtype=np.float64)
    c = np.asarray(c_rgb, dtype=np.float64)

    if f.ndim != 3 or c.ndim != 3 or f.shape[-1] != 3 or c.shape[-1] != 3:
        return f_rgb, c_rgb

    fo = np.empty_like(f)
    co = np.empty_like(c)

    for k in range(3):
        joint = finite_concat(f[..., k], c[..., k])
        lo, hi = np.nanpercentile(joint, [1.0, 99.0])

        if not np.isfinite(hi) or hi <= lo:
            lo, hi = 0.0, 255.0

        fo[..., k] = np.clip((f[..., k] - lo) / (hi - lo), 0.0, 1.0)
        co[..., k] = np.clip((c[..., k] - lo) / (hi - lo), 0.0, 1.0)

    return (
        (fo * 255).astype(np.uint8),
        (co * 255).astype(np.uint8),
    )


def get_pair_imgs(f_row, c_row, bands):
    key = "legacysurvey_image" if bands == SRC_BW else "hsc_image"

    f_fd = f_row.get(key) if f_row is not None else None
    c_fd = c_row.get(key) if c_row is not None else None

    f_rgb, c_rgb = shared_asinh_rgb_pair(f_fd, c_fd, bands)

    if f_rgb is not None and c_rgb is not None:
        return f_rgb, c_rgb

    # Legacy fallback used by the original plotting code.
    if bands == SRC_BW:
        f_fb = pil_to_arr(f_row.get("rgb")) if f_row is not None else None
        c_fb = pil_to_arr(c_row.get("rgb")) if c_row is not None else None
        return shared_rescale_rendered_pair(f_fb, c_fb)

    return f_rgb, c_rgb


def get_meta(row):
    if row is None:
        return dict(psf=np.nan)

    p11 = float(row.get("i_sdssshape_psf_shape11", np.nan))
    p22 = float(row.get("i_sdssshape_psf_shape22", np.nan))

    if np.isfinite(p11 + p22) and (p11 + p22) > 0:
        psf = 2.355 * np.sqrt((p11 + p22) / 2)
    else:
        psf = np.nan

    return dict(psf=psf)


def draw_crosshair(ax, img_shape):
    """
    Small crosshair, deliberately much lighter than in the previous figure.
    """
    h, w = img_shape[:2]
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0

    size = min(h, w)
    gap  = 0.030 * size
    arm  = 0.050 * size

    segments = [
        ([cx - gap - arm, cx - gap], [cy, cy]),
        ([cx + gap, cx + gap + arm], [cy, cy]),
        ([cx, cx], [cy - gap - arm, cy - gap]),
        ([cx, cx], [cy + gap, cy + gap + arm]),
    ]

    for xs, ys in segments:
        # dark halo + thin white line = visible on both bright/dark backgrounds
        ax.plot(xs, ys, color="black", lw=0.9, alpha=0.55, zorder=5)
        ax.plot(xs, ys, color="white", lw=0.42, zorder=6)


def draw_tile(ax, img, mark_centre=True):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")

    if img is None:
        ax.set_facecolor("#EAEAEA")
        ax.text(
            0.5, 0.5, "n/a",
            ha="center", va="center",
            transform=ax.transAxes,
            fontsize=7,
            color="#555555",
        )
    else:
        ax.imshow(
            img,
            origin="upper",
            aspect="equal",
            interpolation="nearest",
            rasterized=True,
        )

        if mark_centre:
            draw_crosshair(ax, img.shape)

    # Neutral hairline border only.  No category color.
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor("#5F5F5F")
        spine.set_linewidth(0.55)


# -----------------------------------------------------------------------------
# Assemble data once
# -----------------------------------------------------------------------------
def collect_category_data(cat_key, n_examples=3):
    pairs = get_pairs(cat_key)[:n_examples]
    out = []

    for ex_i, (f_rid, c_rid) in enumerate(pairs, start=1):
        f_oid = row_to_oid.get(f_rid)
        c_oid = row_to_oid.get(c_rid)

        f_row = found.get(str(f_oid)) if f_oid is not None else None
        c_row = found.get(str(c_oid)) if c_oid is not None else None

        f_legacy, c_legacy = get_pair_imgs(f_row, c_row, SRC_BW)
        f_hsc,    c_hsc    = get_pair_imgs(f_row, c_row, TGT_BW)

        fm = get_meta(f_row)
        cm = get_meta(c_row)

        fr = get_sub_record(f_rid)
        cr = get_sub_record(c_rid)

        delta_r = np.nan
        delta_rg = np.nan
        delta_psf = np.nan

        if fr is not None and cr is not None:
            delta_r = float(fr["true_residual"]) - float(cr["true_residual"])
            delta_rg = float(fr["global_radius"]) - float(cr["global_radius"])

        if np.isfinite(fm["psf"]) and np.isfinite(cm["psf"]):
            delta_psf = fm["psf"] - cm["psf"]

        out.append({
            "example": ex_i,
            "f_rid": f_rid,
            "c_rid": c_rid,
            "f_legacy": f_legacy,
            "c_legacy": c_legacy,
            "f_hsc": f_hsc,
            "c_hsc": c_hsc,
            "delta_r": delta_r,
            "delta_rg": delta_rg,
            "delta_psf": delta_psf,
        })

    return out


# -----------------------------------------------------------------------------
# Individual manuscript tiles
# -----------------------------------------------------------------------------
def save_single_tile(img, outstem):
    """
    Save one image cutout with only a neutral hairline border and crosshair.
    All scientific labels are added in LaTeX.
    """
    fig, ax = plt.subplots(figsize=(1.15, 1.15))
    fig.subplots_adjust(left=0.002, right=0.998, bottom=0.002, top=0.998)

    draw_tile(ax, img, mark_centre=True)

    for ext in ("pdf", "png"):
        kwargs = dict(
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.004,
            facecolor="white",
        )
        if ext == "pdf":
            kwargs["metadata"] = {"Creator": "Matplotlib"}

        fig.savefig(f"{outstem}.{ext}", **kwargs)

    plt.close(fig)


# -----------------------------------------------------------------------------
# Matching-summary LaTeX
# -----------------------------------------------------------------------------
def write_matching_tex(cat_key, data):
    """
    Write a compact table in the scientifically natural orientation:

                   Pair 1   Pair 2   Pair 3
      Delta r       ...
      Delta R^g     ...
      Delta PSF     ...

    For two-pair categories, only Pair 1 and Pair 2 are emitted.
    """
    outpath = OUT_DIR / f"{cat_key}_matching_table.tex"

    n = len(data)
    headers = " & ".join([rf"\textbf{{Pair {i}}}" for i in range(1, n + 1)])

    def fmt(field, digits):
        vals = []
        for d in data:
            v = d[field]
            if np.isfinite(v):
                vals.append(f"{v:+.{digits}f}")
            else:
                vals.append(r"\mathrm{n/a}")
        return " & ".join(vals)

    dr = fmt("delta_r", 4)
    dg = fmt("delta_rg", 4)
    dp = fmt("delta_psf", 2)

    colspec = "l" + "c" * n

    tex = rf"""\begingroup
\centering
\footnotesize
\setlength{{\tabcolsep}}{{0.9em}}
\renewcommand{{\arraystretch}}{{1.08}}
\begin{{tabular}}{{@{{}}{colspec}@{{}}}}
\toprule
 & {headers} \\
\midrule
$\Delta r$ & {dr} \\
$\Delta R^{{\mathrm g}}$ & {dg} \\
$\Delta \mathrm{{PSF}}\;[^\prime\prime]$ & {dp} \\
\bottomrule
\end{{tabular}}
\endgroup
"""
    outpath.write_text(tex)
    print(f"  saved {outpath}", flush=True)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
delta_rows = []

for cat_key in CATEGORIES:
    print(f"Category: {cat_key}", flush=True)

    data = collect_category_data(cat_key, n_examples=3)

    # One file per pair × survey × role.
    # This is the most flexible format for the final NeurIPS grid.
    for d in data:
        p = d["example"]

        tile_specs = [
            ("legacy_flagged", d["f_legacy"]),
            ("hsc_flagged",    d["f_hsc"]),
            ("legacy_control", d["c_legacy"]),
            ("hsc_control",    d["c_hsc"]),
        ]

        for suffix, img in tile_specs:
            outstem = OUT_DIR / f"{cat_key}_pair{p}_{suffix}"
            save_single_tile(img, str(outstem))
            print(f"  saved {outstem}.pdf / .png", flush=True)

    write_matching_tex(cat_key, data)

    for d in data:
        delta_rows.append({
            "category": cat_key,
            "category_label": CATEGORY_LABELS[cat_key],
            "pair": d["example"],
            "flagged_row_id": d["f_rid"],
            "control_row_id": d["c_rid"],
            "delta_r": d["delta_r"],
            "delta_Rg": d["delta_rg"],
            "delta_PSF_arcsec": d["delta_psf"],
        })

pd.DataFrame(delta_rows).to_csv(
    OUT_DIR / "matching_deltas.csv",
    index=False,
)

print(f"Saved matching metadata -> {OUT_DIR / 'matching_deltas.csv'}", flush=True)
print("All done.", flush=True)

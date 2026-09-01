"""
fig5_v4.py — run on cluster (pickle cache, no streaming)

Layout per figure:
  (a) Legacy Survey — 4 examples across
      Row 1: Flagged    [img] [img] [img] [img]
      Row 2: Matched    [img] [img] [img] [img]
             control
      Δr / ΔR_g / ΔPSF below each column

  (b) HSC — same 4 examples

One figure per category triplet (main) or pair (appendix).
No title inside image area. No metadata boxes on images.
Labels: row labels left (8pt bold), panel letters (10pt bold),
        delta annotations below (7pt).
Borders: flagged = category colour, control = #888.
Identical stretch within each flagged/control pair (stated in caption).
Small crosshair on central source.
Saves both PDF (vector labels) and PNG (300dpi).
"""

import io, pickle, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrow
from scipy.ndimage import gaussian_filter
from pathlib import Path

warnings.filterwarnings("ignore")

PAIRS_CSV = "/projects/bfir/ssourav/exp4_results/matched_pairs.csv"
PER_OBJ   = "/projects/bfir/ssourav/exp4_results/full/per_object_results.csv"
CACHE     = "/projects/bfir/ssourav/exp4_results/found_objects.pkl"
OUT_DIR   = "/projects/bfir/ssourav/exp4_results/figures"

# ── load ───────────────────────────────────────────────────────────────────
print("Loading cache...", flush=True)
with open(CACHE, "rb") as f:
    cache = pickle.load(f)
found      = cache["found"]
row_to_oid = cache["row_to_oid"]

pairs_df = pd.read_csv(PAIRS_CSV)
obj      = pd.read_csv(PER_OBJ)
sub      = obj[(obj.survey_pair == "legacysurvey_hsc") &
               (obj.backbone == "astropt_850m") &
               (obj.target_coverage == 0.9)].set_index("row_id")

def get_pairs(cat_key):
    fr = pairs_df[(pairs_df.category == cat_key) & (pairs_df.type == "flagged")]
    cr = pairs_df[(pairs_df.category == cat_key) & (pairs_df.type == "control")]
    return [(int(r.row_id), int(c.row_id))
            for (_, r), (_, c) in zip(fr.iterrows(), cr.iterrows())]

# ── image helpers ──────────────────────────────────────────────────────────
def pil_to_arr(img):
    from PIL import Image
    if img is None: return None
    if isinstance(img, Image.Image): return np.array(img.convert("RGB"))
    if isinstance(img, bytes):
        return np.array(Image.open(io.BytesIO(img)).convert("RGB"))
    if isinstance(img, dict) and "bytes" in img:
        return np.array(Image.open(io.BytesIO(img["bytes"])).convert("RGB"))
    return None

def asinh_norm(arr, scale=None):
    if scale is None:
        scale = 0.1 * np.percentile(np.abs(arr) + 1e-6, 95) + 1e-6
    s = np.arcsinh(arr / scale)
    lo, hi = np.percentile(s, [1, 99])
    return np.clip((s - lo) / (hi - lo + 1e-6), 0, 1), scale

def make_rgb(fd, bw, scale=None):
    """Return (rgb uint8, scale_used) or (None, None)."""
    if not fd: return None, None
    try:
        bands = list(fd["band"])
        flux  = np.array(fd["flux"])
        idx   = [bands.index(b) for b in bw if b in bands]
        if len(idx) < 3: idx = list(range(min(3, len(bands))))
        if len(idx) < 3: return None, None
        ch0, sc = asinh_norm(flux[idx[0]], scale)
        ch1, _  = asinh_norm(flux[idx[1]], sc)
        ch2, _  = asinh_norm(flux[idx[2]], sc)
        return (np.stack([ch0, ch1, ch2], axis=-1) * 255).astype(np.uint8), sc
    except:
        return None, None

SRC_BW = ["des-i", "des-r", "des-g"]
TGT_BW = ["hsc-i", "hsc-r", "hsc-g"]

def get_pair_imgs(f_row, c_row, bands):
    """Same stretch for flagged and control."""
    f_rgb, sc = make_rgb(f_row.get(
        "legacysurvey_image" if bands == SRC_BW else "hsc_image"), bands)
    if f_rgb is None and bands == SRC_BW and f_row is not None:
        f_rgb = pil_to_arr(f_row.get("rgb"))
        sc = None
    c_rgb, _  = make_rgb(c_row.get(
        "legacysurvey_image" if bands == SRC_BW else "hsc_image") if c_row else None,
        bands, scale=sc)
    if c_rgb is None and bands == SRC_BW and c_row is not None:
        c_rgb = pil_to_arr(c_row.get("rgb"))
    return f_rgb, c_rgb

def get_meta(frow):
    if frow is None:
        return dict(mag=np.nan, psf=np.nan, r=np.nan, rg=np.nan, ra=np.nan)
    p11 = float(frow.get("i_sdssshape_psf_shape11", np.nan))
    p22 = float(frow.get("i_sdssshape_psf_shape22", np.nan))
    psf = 2.355 * np.sqrt((p11 + p22) / 2) if np.isfinite(p11 + p22) and (p11+p22)>0 else np.nan
    return dict(mag=float(frow.get("i_cmodel_mag", np.nan)), psf=psf)

def crosshair(ax, img_shape, col="white", lw=0.8, gap=0.06):
    """Four corner ticks around centre, no occlusion."""
    h, w = img_shape[:2]
    cx, cy = w / 2, h / 2
    arm = min(h, w) * 0.10
    g   = min(h, w) * gap
    for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
        ax.plot([cx + dx*g, cx + dx*(g+arm)],
                [cy + dy*g, cy + dy*(g+arm)],
                color=col, lw=lw, transform=ax.transData)

def show_img(ax, img, border_col, lw=1.8, mark_centre=True):
    ax.axis("off")
    if img is not None:
        ax.imshow(img, origin="upper", aspect="equal")
        if mark_centre:
            crosshair(ax, img.shape)
    else:
        ax.set_facecolor("#b0b0b0")
        ax.text(0.5, 0.5, "n/a", ha="center", va="center",
                transform=ax.transAxes, fontsize=7, color="#444")
    for sp in ax.spines.values():
        sp.set_edgecolor(border_col)
        sp.set_linewidth(lw)
        sp.set_visible(True)

# ══════════════════════════════════════════════════════════════════════════
# CORE BUILDER
# ══════════════════════════════════════════════════════════════════════════
CAT_META = {
    "adaptive_rescue":    ("#27ae60", "Adaptive rescue"),
    "silent_failure":     ("#c0392b", "Silent failure"),
    "hard_correct":       ("#8e44ad", "Correctly anticipated hard"),
    "false_alarm":        ("#d68910", "False alarm"),
    "adaptive_regression":("#ca6f1e", "Adaptive regression"),
}
PANEL_LETTERS = "abcdefgh"

def build_cat_figure(cat_key, pair_list, outpath, n_cols=4):
    """
    One figure per category. Two panels: (a) Legacy, (b) HSC.
    Each panel: Flagged row / Control row / delta row.
    n_cols pairs shown.
    """
    pairs = pair_list[:n_cols]
    col, cat_name = CAT_META[cat_key]
    CTRL_COL = "#777777"

    # Pre-fetch rows and images
    data = []
    for f_rid, c_rid in pairs:
        f_oid = row_to_oid.get(f_rid)
        c_oid = row_to_oid.get(c_rid)
        f_row = found.get(str(f_oid)) if f_oid else None
        c_row = found.get(str(c_oid)) if c_oid else None
        f_src, c_src = get_pair_imgs(f_row, c_row, SRC_BW)
        f_tgt, c_tgt = get_pair_imgs(f_row, c_row, TGT_BW)
        fm = get_meta(f_row); cm = get_meta(c_row)
        fr = sub.loc[[f_rid]].iloc[0] if f_rid in sub.index else None
        cr = sub.loc[[c_rid]].iloc[0] if c_rid in sub.index else None
        data.append(dict(
            f_src=f_src, c_src=c_src,
            f_tgt=f_tgt, c_tgt=c_tgt,
            fm=fm, cm=cm, fr=fr, cr=cr
        ))

    # Layout: 2 survey panels stacked, 3 image rows + delta row each
    TILE   = 1.80   # inches per tile
    GAP    = 0.06   # gutter between tiles
    ROW_LBL= 1.00   # left label column
    DELTA_H= 0.55   # delta annotation height
    PANEL_GAP = 0.35  # vertical gap between (a) and (b)

    panel_h = 2 * TILE + DELTA_H
    fig_w   = ROW_LBL + n_cols * TILE + (n_cols - 1) * GAP + 0.15
    fig_h   = 2 * panel_h + PANEL_GAP + 0.30   # 0.30 for panel letter rows

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white")

    for panel_i, (bands_label, img_key_f, img_key_c) in enumerate([
            ("(a) Legacy Survey", "f_src", "c_src"),
            ("(b) HSC",           "f_tgt", "c_tgt"),
    ]):
        # panel top in figure fraction
        ptop = 1.0 - panel_i * (panel_h + PANEL_GAP + 0.18) / fig_h - 0.10 / fig_h

        # Panel letter
        fig.text(ROW_LBL / fig_w, ptop + 0.015 / fig_h,
                 bands_label, fontsize=10, fontweight="bold",
                 color="#222", va="bottom", ha="left",
                 transform=fig.transFigure)

        # Row labels
        row_labels = ["Flagged", "Matched\ncontrol"]
        row_tops   = [ptop - 0.22/fig_h,
                      ptop - 0.22/fig_h - TILE/fig_h - GAP/fig_h]
        for rl, rt in zip(row_labels, row_tops):
            fig.text(0.005, rt + (TILE/fig_h)/2,
                     rl, fontsize=8.5, fontweight="bold",
                     color="#333", va="center", ha="left",
                     transform=fig.transFigure,
                     rotation=0)

        # Tiles
        for col_i, d in enumerate(data):
            for row_i, (img_key, border_col, lw) in enumerate([
                    (img_key_f, col,      2.2),
                    (img_key_c, CTRL_COL, 1.5),
            ]):
                img = d[img_key]
                ax_l = (ROW_LBL + col_i*(TILE+GAP)) / fig_w
                ax_b = (ptop - 0.22/fig_h
                        - row_i*(TILE+GAP)/fig_h
                        - TILE/fig_h)
                ax_w = TILE / fig_w
                ax_h = TILE / fig_h
                ax = fig.add_axes([ax_l, ax_b, ax_w, ax_h])
                show_img(ax, img, border_col, lw=lw, mark_centre=True)

                # Example number above first row first panel
                if row_i == 0:
                    ax.set_title(f"Ex. {col_i+1}", fontsize=7.5,
                                 color="#555", pad=2)

            # Delta annotations below each column
            fr = d["fr"]; cr = d["cr"]
            fm = d["fm"]; cm = d["cm"]

            if fr is not None and cr is not None:
                delta_r   = float(fr.true_residual)  - float(cr.true_residual)
                delta_rg  = float(fr.global_radius)  - float(cr.global_radius)
                delta_psf = fm["psf"] - cm["psf"] if (
                    np.isfinite(fm["psf"]) and np.isfinite(cm["psf"])) else np.nan
                lines = [
                    f"$\\Delta r$ = {delta_r:+.4f}",
                    f"$\\Delta R^{{\\rm g}}$ = {delta_rg:+.4f}",
                    f"$\\Delta$PSF = {delta_psf:+.2f}\"" if np.isfinite(delta_psf) else "",
                ]
                ann = "\n".join(l for l in lines if l)
                ax_l_ann = (ROW_LBL + col_i*(TILE+GAP)) / fig_w + (TILE/fig_w)/2
                ax_b_ann = (ptop - 0.22/fig_h
                            - 2*(TILE+GAP)/fig_h + GAP/fig_h
                            - DELTA_H/fig_h + 0.04/fig_h)
                fig.text(ax_l_ann, ax_b_ann, ann,
                         ha="center", va="top", fontsize=6.8,
                         color="#222", transform=fig.transFigure,
                         linespacing=1.5)

    # Thin horizontal rule between panels
    rule_y = 1.0 - (panel_h + 0.12) / fig_h - 0.10/fig_h
    fig.add_artist(plt.Line2D(
        [ROW_LBL/fig_w, 1.0-0.01], [rule_y, rule_y],
        transform=fig.transFigure, color="#ddd", lw=0.8))

    Path(outpath).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath.replace(".pdf",".png"),
                bbox_inches="tight", dpi=300, facecolor="white")
    fig.savefig(outpath,
                bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved → {outpath}", flush=True)


def build_multi_cat(cat_keys, outpath_stem, n_cols=4):
    """
    One combined figure for multiple categories, stacked vertically.
    (a) Legacy / (b) HSC within each category block.
    """
    CTRL_COL = "#777777"
    TILE     = 1.70
    GAP      = 0.055
    ROW_LBL  = 1.05
    DELTA_H  = 0.50
    CAT_GAP  = 0.45
    TOP_PAD  = 0.28

    n_cats  = len(cat_keys)
    # each category = 2 panels × (2 rows + delta) = one block
    cat_block_h = 2 * (2*TILE + GAP + DELTA_H) + 0.28 + 0.22   # panels + letters + gap between
    fig_w = ROW_LBL + n_cols*(TILE+GAP) - GAP + 0.15
    fig_h = n_cats * cat_block_h + (n_cats-1)*CAT_GAP + TOP_PAD

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white")

    cur_top = 1.0 - TOP_PAD/fig_h

    for cat_i, cat_key in enumerate(cat_keys):
        col, cat_name = CAT_META[cat_key]
        pair_list = get_pairs(cat_key)[:n_cols]

        # Category name strip
        fig.text(ROW_LBL/fig_w, cur_top - 0.02/fig_h,
                 cat_name, fontsize=10, fontweight="bold",
                 color=col, va="top", ha="left",
                 transform=fig.transFigure)

        # Pre-fetch
        data = []
        for f_rid, c_rid in pair_list:
            f_oid = row_to_oid.get(f_rid)
            c_oid = row_to_oid.get(c_rid)
            f_row = found.get(str(f_oid)) if f_oid else None
            c_row = found.get(str(c_oid)) if c_oid else None
            f_src, c_src = get_pair_imgs(f_row, c_row, SRC_BW)
            f_tgt, c_tgt = get_pair_imgs(f_row, c_row, TGT_BW)
            fm = get_meta(f_row); cm = get_meta(c_row)
            fr = sub.loc[[f_rid]].iloc[0] if f_rid in sub.index else None
            cr = sub.loc[[c_rid]].iloc[0] if c_rid in sub.index else None
            data.append(dict(f_src=f_src,c_src=c_src,
                             f_tgt=f_tgt,c_tgt=c_tgt,
                             fm=fm,cm=cm,fr=fr,cr=cr))

        panel_top = cur_top - 0.22/fig_h

        for panel_i, (panel_label, img_key_f, img_key_c) in enumerate([
                ("(a) Legacy Survey", "f_src","c_src"),
                ("(b) HSC",           "f_tgt","c_tgt"),
        ]):
            ptop = panel_top - panel_i * (2*TILE + GAP + DELTA_H + 0.22) / fig_h

            # Panel letter
            fig.text(ROW_LBL/fig_w, ptop + 0.01/fig_h,
                     panel_label, fontsize=9, fontweight="bold",
                     color="#333", va="bottom", ha="left",
                     transform=fig.transFigure)

            # Row labels
            for rl_i, rl in enumerate(["Flagged", "Matched\ncontrol"]):
                ry = ptop - 0.18/fig_h - rl_i*(TILE+GAP)/fig_h - (TILE/fig_h)/2
                fig.text(0.004, ry, rl,
                         fontsize=8, fontweight="bold", color="#333",
                         va="center", ha="left",
                         transform=fig.transFigure)

            for col_i, d in enumerate(data):
                for row_i, (img_key, border_col, lw) in enumerate([
                        (img_key_f, col, 2.0),
                        (img_key_c, CTRL_COL, 1.4),
                ]):
                    img = d[img_key]
                    ax_l = (ROW_LBL + col_i*(TILE+GAP)) / fig_w
                    ax_b = ptop - 0.18/fig_h \
                           - row_i*(TILE+GAP)/fig_h \
                           - TILE/fig_h
                    ax_w = TILE/fig_w; ax_h = TILE/fig_h
                    ax = fig.add_axes([ax_l, ax_b, ax_w, ax_h])
                    show_img(ax, img, border_col, lw=lw)
                    if row_i == 0 and panel_i == 0:
                        ax.set_title(f"Ex. {col_i+1}", fontsize=7,
                                     color="#666", pad=2)

                # Delta row
                fr = d["fr"]; cr = d["cr"]
                fm = d["fm"]; cm = d["cm"]
                if fr is not None and cr is not None:
                    dr   = float(fr.true_residual)  - float(cr.true_residual)
                    drg  = float(fr.global_radius)  - float(cr.global_radius)
                    dpsf = fm["psf"]-cm["psf"] if (
                        np.isfinite(fm["psf"]) and np.isfinite(cm["psf"])) else np.nan
                    lines = [f"$\\Delta r$={dr:+.4f}",
                             f"$\\Delta R^{{\\rm g}}$={drg:+.4f}"]
                    if np.isfinite(dpsf):
                        lines.append(f"$\\Delta$PSF={dpsf:+.2f}\"")
                    cx = (ROW_LBL + col_i*(TILE+GAP) + TILE/2) / fig_w
                    by = ptop - 0.18/fig_h \
                         - 2*(TILE+GAP)/fig_h + GAP/fig_h \
                         - DELTA_H/fig_h + 0.03/fig_h
                    fig.text(cx, by, "\n".join(lines),
                             ha="center", va="top", fontsize=6.5,
                             color="#333", transform=fig.transFigure,
                             linespacing=1.45)

        # Rule between categories
        if cat_i < n_cats - 1:
            rule_y = panel_top - (2*(2*TILE+GAP+DELTA_H+0.22)+0.15)/fig_h
            fig.add_artist(plt.Line2D(
                [ROW_LBL/fig_w, 1.0-0.01], [rule_y, rule_y],
                transform=fig.transFigure, color="#ccc", lw=0.7))

        cur_top -= (cat_block_h + CAT_GAP) / fig_h

    for ext in [".png", ".pdf"]:
        fig.savefig(outpath_stem + ext,
                    bbox_inches="tight", dpi=300, facecolor="white")
    plt.close()
    print(f"Saved → {outpath_stem}.png / .pdf", flush=True)


# ── run ────────────────────────────────────────────────────────────────────
OUT = OUT_DIR

# Main figure: 3 categories, 4 examples each
build_multi_cat(
    ["adaptive_rescue", "silent_failure", "hard_correct"],
    f"{OUT}/fig5_v4_main",
    n_cols=4,
)

# Appendix: 2 categories
build_multi_cat(
    ["false_alarm", "adaptive_regression"],
    f"{OUT}/fig5_v4_appendix",
    n_cols=4,
)

print("All done.", flush=True)

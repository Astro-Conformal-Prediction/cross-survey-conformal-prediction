"""
fig5_final_v2.py  —  run on cluster
Produces:
  fig5_main_tight.png       — 3 cats × 3 examples, tight layout, simplified annotations
  fig5_appendix_tight.png   — 2 cats × 3 examples
  table2_category_stats.png — quantitative comparison table
  fig6_psf_matched.png      — PSF-degraded comparison for top rescue/failure examples
"""
import pandas as pd, numpy as np, io, warnings, pickle
warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.ndimage import gaussian_filter
from pathlib import Path
from datasets import load_dataset

PAIRS_CSV = "/projects/bfir/ssourav/exp4_results/matched_pairs.csv"
PER_OBJ   = "/projects/bfir/ssourav/exp4_results/full/per_object_results.csv"
OUT_DIR   = "/projects/bfir/ssourav/exp4_results/figures"
CACHE     = "/projects/bfir/ssourav/exp4_results/found_objects.pkl"

# ── load data ──────────────────────────────────────────────────────────────
pairs_df = pd.read_csv(PAIRS_CSV)
obj = pd.read_csv(PER_OBJ)
sub = obj[(obj.survey_pair=="legacysurvey_hsc")&
          (obj.backbone=="astropt_850m")&
          (obj.target_coverage==0.9)].copy()
sub = sub.set_index("row_id")

pairs = []
for cat_key in ["adaptive_rescue","silent_failure","hard_correct",
                "false_alarm","adaptive_regression"]:
    f_rows = pairs_df[(pairs_df.category==cat_key)&(pairs_df.type=="flagged")]
    c_rows = pairs_df[(pairs_df.category==cat_key)&(pairs_df.type=="control")]
    for (_,fr),(_,cr) in zip(f_rows.iterrows(),c_rows.iterrows()):
        pairs.append((cat_key,int(fr.row_id),int(cr.row_id)))

# ── load cached objects ────────────────────────────────────────────────────
print("Loading cached objects...", flush=True)
with open(CACHE,"rb") as f:
    cache = pickle.load(f)
found      = cache["found"]
row_to_oid = cache["row_to_oid"]
print(f"  {len(found)} objects loaded.", flush=True)

# ── image helpers ──────────────────────────────────────────────────────────
def pil_to_arr(img):
    from PIL import Image
    if img is None: return None
    if isinstance(img, Image.Image): return np.array(img.convert("RGB"))
    if isinstance(img, bytes): return np.array(Image.open(io.BytesIO(img)).convert("RGB"))
    if isinstance(img, dict) and "bytes" in img:
        return np.array(Image.open(io.BytesIO(img["bytes"])).convert("RGB"))
    return None

def flux_to_channel(arr):
    """Asinh stretch a 2D flux array to [0,1]."""
    x = np.arcsinh(arr / (0.1*np.percentile(np.abs(arr)+1e-6,95)+1e-6))
    lo,hi = np.percentile(x,[1,99])
    return np.clip((x-lo)/(hi-lo+1e-6),0,1)

def flux_rgb(fd, bw):
    if not fd: return None
    try:
        bands=list(fd["band"]); flux=np.array(fd["flux"])
        idx=[bands.index(b) for b in bw if b in bands]
        if len(idx)<3: idx=list(range(min(3,len(bands))))
        if len(idx)<3: return None
        rgb = np.stack([flux_to_channel(flux[idx[i]]) for i in range(3)],axis=-1)
        return (rgb*255).astype(np.uint8)
    except: return None

def flux_channels(fd, bw):
    """Return individual stretched channels for PSF matching."""
    if not fd: return None, None
    try:
        bands=list(fd["band"]); flux=np.array(fd["flux"])
        idx=[bands.index(b) for b in bw if b in bands]
        if len(idx)<3: idx=list(range(min(3,len(bands))))
        if len(idx)<3: return None, None
        channels = [flux[i] for i in idx]
        return channels, idx
    except: return None, None

def get_imgs(frow):
    if frow is None: return None, None
    src = pil_to_arr(frow.get("rgb"))
    if src is None: src = flux_rgb(frow.get("legacysurvey_image"),["des-i","des-r","des-g"])
    tgt = flux_rgb(frow.get("hsc_image"),["hsc-i","hsc-r","hsc-g"])
    return src, tgt

def get_meta(frow):
    """Extract magnitude, HSC size, HSC PSF FWHM proxy, extendedness."""
    if frow is None:
        return {k: float("nan") for k in ["mag","size","psf_fwhm","extendedness","ebv"]}
    mag  = float(frow.get("i_cmodel_mag",float("nan")))
    size = float(frow.get("SHAPE_R",float("nan")))
    ebv  = float(frow.get("EBV",float("nan")))
    ext  = float(frow.get("i_extendedness_value",float("nan")))
    # HSC PSF FWHM proxy: sqrt(tr(PSF moments)) in pixels, ~0.168"/pix for HSC
    p11  = float(frow.get("i_sdssshape_psf_shape11",float("nan")))
    p22  = float(frow.get("i_sdssshape_psf_shape22",float("nan")))
    psf_px = np.sqrt(np.abs(p11)+np.abs(p22)) if np.isfinite(p11+p22) else float("nan")
    psf_arcsec = psf_px * 0.168 * 2.355  # moments->FWHM, pixels->arcsec
    return {"mag":mag,"size":size,"psf_fwhm":psf_arcsec,"extendedness":ext,"ebv":ebv}

def get_row(rid):
    r = sub.loc[[rid]].iloc[0] if rid in sub.index else None
    return r

ROW_META = {
    "adaptive_rescue":    ("#2ecc71","Adaptive rescue",
                           r"$r>R^{\rm g}$, $r\leq R^{\rm a}$"),
    "silent_failure":     ("#e74c3c","Silent failure",
                           r"Low $\hat\sigma$, high $r$"),
    "hard_correct":       ("#9b59b6","Correctly anticipated hard",
                           r"High $\hat\sigma$, high $r$, covered"),
    "false_alarm":        ("#f39c12","False alarm",
                           r"High $\hat\sigma$, low $r$"),
    "adaptive_regression":("#e67e22","Adaptive regression",
                           r"$r\leq R^{\rm g}$, $r>R^{\rm a}$"),
}

# ══════════════════════════════════════════════════════════════════════════
# FIGURE 5 MAIN — tight 3×3 layout
# Each category = one block of 3 rows (one per example)
# Columns: Legacy(flagged) | HSC(flagged) | Legacy(ctrl) | HSC(ctrl)
# ══════════════════════════════════════════════════════════════════════════
def build_tight(cat_keys, title, outpath, show_ids=False):
    cat_pairs = {k:[(f,c) for (ck,f,c) in pairs if ck==k] for k in cat_keys}

    # count total object rows
    total_rows = sum(len(v) for v in cat_pairs.values())
    n_cats     = len(cat_keys)

    fig_w = 4*2.5 + 2.0
    fig_h = total_rows * 2.6 + n_cats * 0.55
    fig = plt.figure(figsize=(fig_w, fig_h))

    # build flat list of (cat_key, f_rid, c_rid, is_first_in_cat)
    row_specs = []
    for k in cat_keys:
        for i,(f,c) in enumerate(cat_pairs[k]):
            row_specs.append((k,f,c,i==0))

    # heights: each image row = 1 unit; small separator between categories
    heights = []
    for i,(k,f,c,is_first) in enumerate(row_specs):
        heights.append(2.4)
        if i < len(row_specs)-1 and row_specs[i+1][0] != k:
            heights.append(0.3)  # separator

    outer_rows = []
    sep_mask = []
    i = 0
    for rs_i,(k,f,c,is_first) in enumerate(row_specs):
        outer_rows.append(rs_i + i)
        sep_mask.append(False)
        if rs_i < len(row_specs)-1 and row_specs[rs_i+1][0] != k:
            i += 1
            sep_mask.append(True)

    n_outer = len(row_specs) + sum(1 for rs_i in range(len(row_specs)-1)
                                    if row_specs[rs_i+1][0] != row_specs[rs_i][0])
    outer = gridspec.GridSpec(n_outer, 1, figure=fig,
                              height_ratios=[2.4 if not (i%2==1 and False) else 0.2
                                             for i in range(n_outer)],
                              hspace=0.65)

    # rebuild with correct heights
    hr = []
    flat = []
    for rs_i,(k,f,c,is_first) in enumerate(row_specs):
        hr.append(2.4); flat.append(("data",k,f,c,is_first))
        if rs_i<len(row_specs)-1 and row_specs[rs_i+1][0]!=k:
            hr.append(0.25); flat.append(("sep",None,None,None,None))
    outer2 = gridspec.GridSpec(len(hr),1,figure=fig,
                               height_ratios=hr, hspace=0.55)

    plot_ri = 0
    for item in flat:
        if item[0]=="sep":
            plot_ri+=1; continue
        _,cat_key,f_rid,c_rid,is_first = item
        col,main_lbl,sub_lbl = ROW_META[cat_key]

        inner = gridspec.GridSpecFromSubplotSpec(
            2,4,subplot_spec=outer2[plot_ri],
            wspace=0.035,hspace=0.04,height_ratios=[3.2,1])
        img_ax=[fig.add_subplot(inner[0,j]) for j in range(4)]
        ann_ax=[fig.add_subplot(inner[1,j]) for j in range(4)]
        for ax in ann_ax: ax.axis("off")

        if is_first:
            img_ax[0].annotate(
                f"{main_lbl}\n{sub_lbl}",
                xy=(-0.52,0.5),xycoords="axes fraction",
                fontsize=9,color=col,fontweight="bold",
                va="center",ha="right",annotation_clip=False)

        if plot_ri==0:
            for j,lbl in enumerate(["Legacy (flagged)","HSC (flagged)",
                                     "Legacy (control)","HSC (control)"]):
                img_ax[j].set_title(lbl,fontsize=8.5,color="gray",
                                    pad=3,fontweight="bold")

        f_oid=row_to_oid.get(f_rid); c_oid=row_to_oid.get(c_rid)
        f_row=found.get(str(f_oid)) if f_oid else None
        c_row=found.get(str(c_oid)) if c_oid else None
        f_src,f_tgt=get_imgs(f_row); c_src,c_tgt=get_imgs(c_row)
        f_meta=get_meta(f_row); c_meta=get_meta(c_row)
        f_orow=get_row(f_rid); c_orow=get_row(c_rid)

        for j,(img,bc) in enumerate(zip([f_src,f_tgt,c_src,c_tgt],
                                         [col,col,"#777","#777"])):
            ax=img_ax[j]; ax.axis("off")
            for sp in ax.spines.values():
                sp.set_edgecolor(bc); sp.set_linewidth(2.2); sp.set_visible(True)
            if img is not None: ax.imshow(img,origin="upper")
            else:
                ax.set_facecolor("#ccc")
                ax.text(0.5,0.5,"n/a",ha="center",va="center",
                        transform=ax.transAxes,fontsize=8)

        # simplified annotations — main paper: r, R_g, R_a, mag, size, PSF
        def fmt_ann(orow, meta, label, is_flagged):
            if orow is None: return ""
            r   = float(orow.true_residual)
            rg  = float(orow.global_radius)
            ra  = float(orow.adaptive_radius)
            mag = meta["mag"]; sz=meta["size"]; psf=meta["psf_fwhm"]
            id_str = f"  [{str(f_oid if is_flagged else c_oid).replace('b','').replace(chr(39),'')}]" \
                     if show_ids else ""
            return (f"{label}{id_str}\n"
                    f"$r$={r:.4f}   $R^{{\\rm g}}$={rg:.4f}   $R^{{\\rm a}}$={ra:.4f}\n"
                    f"$i$={mag:.1f}   sz={sz:.2f}\"   PSF={psf:.2f}\"")

        f_ann = fmt_ann(f_orow, f_meta, "Flagged", True)
        c_ann = fmt_ann(c_orow, c_meta, "Control", False)

        for ann_str, col_l, col_r, bc in [
                (f_ann, ann_ax[0], ann_ax[1], col),
                (c_ann, ann_ax[2], ann_ax[3], "#777")]:
            if not ann_str: continue
            cx=(col_l.get_position().x0+col_r.get_position().x1)/2
            cy= col_l.get_position().y0+0.002
            fig.text(cx,cy,ann_str,ha="center",va="bottom",fontsize=6.5,
                     color="#111",transform=fig.transFigure,
                     bbox=dict(boxstyle="round,pad=0.25",
                               fc="#fff8f8" if bc!=("#777") else "#f8f8f8",
                               ec=bc,alpha=0.93,lw=0.9))
        plot_ri += 1

    fig.suptitle(title,fontsize=11,fontweight="bold",y=1.003)
    Path(outpath).parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(outpath,bbox_inches="tight",dpi=150)
    plt.close()
    print(f"Saved → {outpath}", flush=True)

# ══════════════════════════════════════════════════════════════════════════
# TABLE 2 — quantitative category comparison
# ══════════════════════════════════════════════════════════════════════════
def build_stats_table(outpath):
    # Collect all objects in each category + easy_stable pool
    obj2 = pd.read_csv(PER_OBJ)
    sub2 = obj2[(obj2.survey_pair=="legacysurvey_hsc")&
                (obj2.backbone=="astropt_850m")&
                (obj2.target_coverage==0.9)].copy()
    q10d=sub2.predicted_difficulty.quantile(0.10)
    q90d=sub2.predicted_difficulty.quantile(0.90)
    q10r=sub2.true_residual.quantile(0.10)
    q90r=sub2.true_residual.quantile(0.90)
    q50r=sub2.true_residual.quantile(0.50)

    flagged_rids = set(pairs_df[pairs_df.type=="flagged"].row_id.tolist())
    ctrl_rids    = set(pairs_df[pairs_df.type=="control"].row_id.tolist())

    cat_masks = {
        "Adaptive\nrescue":    (~sub2.global_covered)&(sub2.adaptive_covered),
        "Silent\nfailure":     (sub2.predicted_difficulty<=q10d)&(sub2.true_residual>=q90r),
        "Hard\ncorrect":       (sub2.predicted_difficulty>=q90d)&(sub2.true_residual>=q50r)&sub2.adaptive_covered,
        "False\nalarm":        (sub2.predicted_difficulty>=q90d)&(sub2.true_residual<=q10r),
        "Adaptive\nregression":sub2.global_covered&(~sub2.adaptive_covered),
        "Easy\nstable":        (sub2.predicted_difficulty<=q10d)&(sub2.true_residual<=q10r)&sub2.global_covered,
    }

    # Get metadata from found objects for each rid
    all_rids_in_cats = set()
    for mask in cat_masks.values():
        all_rids_in_cats.update(sub2[mask].row_id.tolist())

    # Build metadata lookup from found objects
    embed_df = load_dataset("astro-conformal-project/split_legacysurvey_hsc_embeddings",
                            "astropt_850m", split="test").to_pandas()

    meta_rows = []
    for rid in all_rids_in_cats:
        if rid >= len(embed_df): continue
        oid = str(embed_df.iloc[rid]["legacysurvey_object_id"])
        frow = found.get(oid)
        if frow is None: continue
        m = get_meta(frow)
        m["row_id"] = rid
        m["nn_dist"] = float(sub2[sub2.row_id==rid].nearest_neighbor_distance.values[0]) \
                        if rid in sub2.row_id.values else float("nan")
        meta_rows.append(m)

    meta_df = pd.DataFrame(meta_rows).set_index("row_id")
    sub2_idx = sub2.set_index("row_id")

    stats = {}
    for cat_label, mask in cat_masks.items():
        rids = sub2[mask].row_id.tolist()
        m = meta_df.loc[[r for r in rids if r in meta_df.index]]
        s = sub2_idx.loc[[r for r in rids if r in sub2_idx.index]]
        stats[cat_label] = {
            "n": len(rids),
            "mag":        f"{m.mag.median():.1f}±{m.mag.std():.1f}",
            "size":       f"{m.size.median():.2f}±{m.size.std():.2f}\"",
            "HSC PSF":    f"{m.psf_fwhm.median():.2f}±{m.psf_fwhm.std():.2f}\"",
            "extendedness":f"{m.extendedness.median():.2f}",
            "nn_dist":    f"{m.nn_dist.median():.3f}±{m.nn_dist.std():.3f}",
            "true r":     f"{s.true_residual.median():.4f}",
            "pred σ̂":    f"{s.predicted_difficulty.median():.4f}",
        }

    cats_order = list(cat_masks.keys())
    props = ["n","mag","size","HSC PSF","extendedness","nn_dist","true r","pred σ̂"]
    prop_labels = ["N objects","i-mag (median±σ)","Size SHAPE_R","HSC PSF FWHM",
                   "Extendedness","NN distance","True residual r","Pred. difficulty σ̂"]

    fig, ax = plt.subplots(figsize=(16,5))
    ax.axis("off")
    cell_text = [[stats[c][p] for c in cats_order] for p in props]
    colors_header = ["#2ecc71","#e74c3c","#9b59b6","#f39c12","#e67e22","#3498db"]

    tbl = ax.table(
        cellText=cell_text,
        rowLabels=prop_labels,
        colLabels=[c.replace("\n"," ") for c in cats_order],
        loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1,1.7)

    for j,col in enumerate(colors_header):
        tbl[(0,j)].set_facecolor(col)
        tbl[(0,j)].set_text_props(color="white",fontweight="bold")
    for i in range(len(props)):
        tbl[(i+1,-1)].set_facecolor("#f0f0f0")
        tbl[(i+1,-1)].set_text_props(fontweight="bold")

    ax.set_title(
        "Table 2 — Quantitative category comparison\n"
        "Legacy→HSC · AstroPT-850M · Ridge · 90% coverage\n"
        "Median ± std across all objects in each category",
        fontsize=11,fontweight="bold",pad=10)
    fig.tight_layout()
    fig.savefig(outpath,bbox_inches="tight",dpi=150)
    plt.close()
    print(f"Saved → {outpath}", flush=True)

# ══════════════════════════════════════════════════════════════════════════
# FIGURE 6 — PSF-matching experiment
# For top 3 rescue + 3 silent-failure: show Legacy | HSC orig | HSC PSF-matched
# ══════════════════════════════════════════════════════════════════════════
def psf_degrade_rgb(hsc_rgb, hsc_psf_fwhm_arcsec, legacy_psf_fwhm_arcsec=1.2,
                    pixel_scale_hsc=0.168):
    """
    Convolve HSC RGB image with a Gaussian to match Legacy PSF.
    legacy_psf_fwhm_arcsec: typical Legacy Survey PSF ~1.0-1.5", use 1.2" as default.
    Returns degraded RGB uint8 array.
    """
    if hsc_rgb is None: return None
    if not np.isfinite(hsc_psf_fwhm_arcsec) or hsc_psf_fwhm_arcsec <= 0:
        hsc_psf_fwhm_arcsec = 0.6  # HSC median default

    # Extra blurring needed: sqrt(legacy^2 - hsc^2) in pixels
    target_fwhm_px = legacy_psf_fwhm_arcsec / pixel_scale_hsc
    hsc_fwhm_px    = hsc_psf_fwhm_arcsec    / pixel_scale_hsc
    extra_fwhm_px  = np.sqrt(max(target_fwhm_px**2 - hsc_fwhm_px**2, 0))
    extra_sigma_px = extra_fwhm_px / 2.355

    if extra_sigma_px < 0.1:
        return hsc_rgb  # already matched

    degraded = np.zeros_like(hsc_rgb, dtype=np.float32)
    for ch in range(3):
        degraded[:,:,ch] = gaussian_filter(
            hsc_rgb[:,:,ch].astype(np.float32), sigma=extra_sigma_px)
    return np.clip(degraded,0,255).astype(np.uint8)

def build_psf_panel(outpath):
    # Use top rescue + silent_failure pairs
    psf_cats = ["adaptive_rescue","silent_failure"]
    psf_pairs = [(k,f,c) for (k,f,c) in pairs if k in psf_cats][:6]  # top 3 each

    n_rows = len(psf_pairs)
    fig = plt.figure(figsize=(3*2.6+2.2, n_rows*2.8))
    outer = gridspec.GridSpec(n_rows,1,figure=fig,hspace=0.7)

    for ri,(cat_key,f_rid,c_rid) in enumerate(psf_pairs):
        col,main_lbl,_ = ROW_META[cat_key]
        inner = gridspec.GridSpecFromSubplotSpec(
            2,3,subplot_spec=outer[ri],wspace=0.04,hspace=0.05,height_ratios=[3,1])
        img_ax=[fig.add_subplot(inner[0,j]) for j in range(3)]
        ann_ax=[fig.add_subplot(inner[1,j]) for j in range(3)]
        for ax in ann_ax: ax.axis("off")

        img_ax[0].set_ylabel(f"{main_lbl}",fontsize=8.5,color=col,
                              fontweight="bold",labelpad=4)
        if ri==0:
            for j,lbl in enumerate(["Legacy (source)","HSC (original)",
                                     "HSC (PSF-matched to Legacy)"]):
                img_ax[j].set_title(lbl,fontsize=8,color="gray",pad=3,fontweight="bold")

        f_oid  = row_to_oid.get(f_rid)
        f_row  = found.get(str(f_oid)) if f_oid else None
        f_src, f_tgt = get_imgs(f_row)
        f_meta = get_meta(f_row)
        f_orow = get_row(f_rid)

        # PSF-degrade HSC to Legacy resolution
        f_tgt_deg = psf_degrade_rgb(f_tgt,
                                     hsc_psf_fwhm_arcsec=f_meta["psf_fwhm"],
                                     legacy_psf_fwhm_arcsec=1.2)

        imgs  = [f_src, f_tgt, f_tgt_deg]
        borders=[col,   col,    "#999"]
        for j,(img,bc) in enumerate(zip(imgs,borders)):
            ax=img_ax[j]; ax.axis("off")
            for sp in ax.spines.values():
                sp.set_edgecolor(bc); sp.set_linewidth(2.0); sp.set_visible(True)
            if img is not None: ax.imshow(img,origin="upper")
            else:
                ax.set_facecolor("#ccc")
                ax.text(0.5,0.5,"n/a",ha="center",va="center",
                        transform=ax.transAxes,fontsize=8)

        # annotation
        if f_orow is not None:
            r   = float(f_orow.true_residual)
            rg  = float(f_orow.global_radius)
            ra  = float(f_orow.adaptive_radius)
            mag = f_meta["mag"]; sz=f_meta["size"]; psf=f_meta["psf_fwhm"]
            annot = (f"$r$={r:.4f}  $R^{{\\rm g}}$={rg:.4f}  $R^{{\\rm a}}$={ra:.4f}\n"
                     f"$i$={mag:.1f}  sz={sz:.2f}\"  HSC PSF={psf:.2f}\"  Legacy PSF≈1.2\"")
            cx=(ann_ax[0].get_position().x0+ann_ax[2].get_position().x1)/2
            cy= ann_ax[0].get_position().y0+0.002
            fig.text(cx,cy,annot,ha="center",va="bottom",fontsize=6.8,
                     color="#111",transform=fig.transFigure,
                     bbox=dict(boxstyle="round,pad=0.25",fc="#f9f9f9",ec=col,alpha=0.93,lw=0.9))

    fig.suptitle(
        "Figure 6 — HSC images degraded to Legacy PSF\n"
        "Legacy→HSC · adaptive rescue and silent failure examples\n"
        "Column 3: HSC convolved with Gaussian kernel to match Legacy PSF FWHM ≈1.2\"",
        fontsize=11,fontweight="bold",y=1.005)
    Path(outpath).parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(outpath,bbox_inches="tight",dpi=150)
    plt.close()
    print(f"Saved → {outpath}", flush=True)

# ── run all ────────────────────────────────────────────────────────────────
OUT = OUT_DIR

build_tight(
    ["adaptive_rescue","silent_failure","hard_correct"],
    "Figure 5 — Flagged galaxies with matched easy-stable controls\n"
    "Legacy→HSC · AstroPT-850M · Ridge · 90% coverage · "
    "Coloured border = flagged · Grey border = matched control · "
    "Annotation: $r$, $R^{\\rm g}$, $R^{\\rm a}$, $i$-mag, size, HSC PSF FWHM",
    f"{OUT}/fig5_main_tight.png", show_ids=False)

build_tight(
    ["false_alarm","adaptive_regression"],
    "Figure 5 (appendix) — Additional categories with matched controls\n"
    "Legacy→HSC · AstroPT-850M · Ridge · 90% coverage",
    f"{OUT}/fig5_appendix_tight.png", show_ids=True)

build_stats_table(f"{OUT}/table2_category_stats.png")

build_psf_panel(f"{OUT}/fig6_psf_matched.png")

print("All done.", flush=True)

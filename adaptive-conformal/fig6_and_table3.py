"""
fig6_and_table3.py  —  run on cluster
Fixes:
  1. PSF FWHM formula: moments are in arcsec^2, so FWHM = 2.355*sqrt((p11+p22)/2)
  2. Figure 6 regenerated with correct blur kernel
  3. Table 3: bootstrap CIs for 3 pre-defined comparisons × 4 quantities
"""
import pandas as pd, numpy as np, io, warnings, pickle
warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.ndimage import gaussian_filter
from pathlib import Path
from datasets import load_dataset

PER_OBJ = "/projects/bfir/ssourav/exp4_results/full/per_object_results.csv"
CACHE   = "/projects/bfir/ssourav/exp4_results/found_objects.pkl"
PAIRS   = "/projects/bfir/ssourav/exp4_results/matched_pairs.csv"
OUT_DIR = "/projects/bfir/ssourav/exp4_results/figures"

# ── load ───────────────────────────────────────────────────────────────────
print("Loading cache...", flush=True)
with open(CACHE,"rb") as f:
    cache = pickle.load(f)
found = cache["found"]; row_to_oid = cache["row_to_oid"]

pairs_df = pd.read_csv(PAIRS)
obj = pd.read_csv(PER_OBJ)
sub = obj[(obj.survey_pair=="legacysurvey_hsc")&
          (obj.backbone=="astropt_850m")&
          (obj.target_coverage==0.9)].copy()
sub_idx = sub.set_index("row_id")

pairs = []
for cat_key in ["adaptive_rescue","silent_failure"]:
    f_rows = pairs_df[(pairs_df.category==cat_key)&(pairs_df.type=="flagged")]
    c_rows = pairs_df[(pairs_df.category==cat_key)&(pairs_df.type=="control")]
    for (_,fr),(_,cr) in zip(f_rows.iterrows(),c_rows.iterrows()):
        pairs.append((cat_key,int(fr.row_id),int(cr.row_id)))

# ── corrected PSF helper ───────────────────────────────────────────────────
def get_meta_corrected(frow):
    """
    PSF moments (i_sdssshape_psf_shape11/22) are in ARCSEC^2 in the HSC catalog.
    Correct formula: FWHM_arcsec = 2.355 * sqrt((p11+p22)/2)
    This gives ~0.6-0.8\" for typical HSC i-band seeing.
    """
    if frow is None:
        return {"mag":np.nan,"shape_r":np.nan,"psf_fwhm":np.nan,"extendedness":np.nan}
    mag     = float(frow.get("i_cmodel_mag",np.nan))
    shape_r = float(frow.get("SHAPE_R",np.nan))
    ext     = float(frow.get("i_extendedness_value",np.nan))
    p11     = float(frow.get("i_sdssshape_psf_shape11",np.nan))
    p22     = float(frow.get("i_sdssshape_psf_shape22",np.nan))
    # Correct: moments in arcsec^2
    if np.isfinite(p11) and np.isfinite(p22) and (p11+p22) > 0:
        psf_fwhm = 2.355 * np.sqrt((p11+p22)/2)
    else:
        psf_fwhm = 0.60  # HSC median fallback
    return {"mag":mag,"shape_r":shape_r,"psf_fwhm":psf_fwhm,"extendedness":ext}

# ── image helpers ──────────────────────────────────────────────────────────
def pil_to_arr(img):
    from PIL import Image
    if img is None: return None
    if isinstance(img, Image.Image): return np.array(img.convert("RGB"))
    if isinstance(img, bytes): return np.array(Image.open(io.BytesIO(img)).convert("RGB"))
    if isinstance(img, dict) and "bytes" in img:
        return np.array(Image.open(io.BytesIO(img["bytes"])).convert("RGB"))
    return None

def flux_rgb(fd, bw):
    if not fd: return None
    try:
        bands=list(fd["band"]); flux=np.array(fd["flux"])
        idx=[bands.index(b) for b in bw if b in bands]
        if len(idx)<3: idx=list(range(min(3,len(bands))))
        if len(idx)<3: return None
        def s(x):
            x=np.arcsinh(x/(0.1*np.percentile(np.abs(x)+1e-6,95)+1e-6))
            lo,hi=np.percentile(x,[1,99]); return np.clip((x-lo)/(hi-lo+1e-6),0,1)
        return (np.stack([s(flux[idx[i]]) for i in range(3)],axis=-1)*255).astype(np.uint8)
    except: return None

def get_imgs(frow):
    if frow is None: return None, None
    src = pil_to_arr(frow.get("rgb"))
    if src is None: src = flux_rgb(frow.get("legacysurvey_image"),["des-i","des-r","des-g"])
    tgt = flux_rgb(frow.get("hsc_image"),["hsc-i","hsc-r","hsc-g"])
    return src, tgt

def psf_degrade(hsc_rgb, hsc_fwhm_arcsec, legacy_fwhm_arcsec=1.2, pixel_scale=0.168):
    """
    Correct PSF degradation:
      sigma_kernel = sqrt(legacy_fwhm^2 - hsc_fwhm^2) / (2.355 * pixel_scale)
    """
    if hsc_rgb is None: return None
    hsc_fwhm   = max(hsc_fwhm_arcsec, 0.3)   # floor at 0.3"
    if legacy_fwhm_arcsec <= hsc_fwhm: return hsc_rgb  # already worse
    extra_fwhm = np.sqrt(legacy_fwhm_arcsec**2 - hsc_fwhm**2)
    sigma_px   = extra_fwhm / (2.355 * pixel_scale)
    deg = np.zeros_like(hsc_rgb, dtype=np.float32)
    for ch in range(3):
        deg[:,:,ch] = gaussian_filter(hsc_rgb[:,:,ch].astype(np.float32), sigma=sigma_px)
    return np.clip(deg,0,255).astype(np.uint8), sigma_px

ROW_COLORS = {"adaptive_rescue":"#2ecc71","silent_failure":"#e74c3c"}
ROW_NAMES  = {"adaptive_rescue":"Adaptive rescue","silent_failure":"Silent failure"}

# ══════════════════════════════════════════════════════════════════════════
# FIGURE 6 — corrected PSF matching
# ══════════════════════════════════════════════════════════════════════════
print("Building Figure 6 (corrected)...", flush=True)

n_rows = len(pairs)
fig = plt.figure(figsize=(3*2.8+2.4, n_rows*3.0))
outer = gridspec.GridSpec(n_rows,1,figure=fig,hspace=0.75)

for ri,(cat_key,f_rid,c_rid) in enumerate(pairs):
    col  = ROW_COLORS[cat_key]
    name = ROW_NAMES[cat_key]

    inner = gridspec.GridSpecFromSubplotSpec(
        2,3,subplot_spec=outer[ri],wspace=0.04,hspace=0.05,height_ratios=[3.2,1])
    img_ax=[fig.add_subplot(inner[0,j]) for j in range(3)]
    ann_ax=[fig.add_subplot(inner[1,j]) for j in range(3)]
    for ax in ann_ax: ax.axis("off")

    img_ax[0].annotate(name,xy=(-0.48,0.5),xycoords="axes fraction",
                       fontsize=9,color=col,fontweight="bold",
                       va="center",ha="right",annotation_clip=False)
    if ri==0:
        for j,lbl in enumerate(["Legacy grz\n(source survey)",
                                 "HSC riz\n(original)",
                                 "HSC riz\n(PSF-matched to Legacy)"]):
            img_ax[j].set_title(lbl,fontsize=8,color="gray",pad=3,fontweight="bold")

    f_oid = row_to_oid.get(f_rid)
    f_row = found.get(str(f_oid)) if f_oid else None
    f_src, f_tgt = get_imgs(f_row)
    f_meta = get_meta_corrected(f_row)
    f_orow = sub_idx.loc[[f_rid]].iloc[0] if f_rid in sub_idx.index else None

    # Corrected blur
    legacy_psf = 1.2  # arcsec, typical Legacy DR10
    result = psf_degrade(f_tgt, f_meta["psf_fwhm"], legacy_psf)
    if result is None or not isinstance(result, tuple):
        f_tgt_deg, sigma_px = f_tgt, 0.0
    else:
        f_tgt_deg, sigma_px = result

    for j,(img,bc) in enumerate(zip([f_src,f_tgt,f_tgt_deg],[col,col,"#999"])):
        ax=img_ax[j]; ax.axis("off")
        for sp in ax.spines.values():
            sp.set_edgecolor(bc); sp.set_linewidth(2.0); sp.set_visible(True)
        if img is not None: ax.imshow(img,origin="upper")
        else:
            ax.set_facecolor("#ccc")
            ax.text(0.5,0.5,"n/a",ha="center",va="center",
                    transform=ax.transAxes,fontsize=8)

    if f_orow is not None:
        r=float(f_orow.true_residual); rg=float(f_orow.global_radius)
        ra=float(f_orow.adaptive_radius)
        annot=(f"$r$={r:.4f}  $R^{{\\rm g}}$={rg:.4f}  $R^{{\\rm a}}$={ra:.4f}\n"
               f"$i$={f_meta['mag']:.1f}  size={f_meta['shape_r']:.2f}\"  "
               f"HSC PSF={f_meta['psf_fwhm']:.2f}\"  "
               f"Legacy PSF≈{legacy_psf}\"  blur σ={sigma_px:.1f}px")
        cx=(ann_ax[0].get_position().x0+ann_ax[2].get_position().x1)/2
        cy= ann_ax[0].get_position().y0+0.002
        fig.text(cx,cy,annot,ha="center",va="bottom",fontsize=6.8,color="#111",
                 transform=fig.transFigure,
                 bbox=dict(boxstyle="round,pad=0.25",fc="#f9f9f9",ec=col,alpha=0.93,lw=0.9))

fig.suptitle(
    "Figure 6 — PSF-matching experiment (corrected)\n"
    "HSC PSF FWHM from catalog moments (arcsec²): FWHM = 2.355×√((p₁₁+p₂₂)/2)\n"
    "Column 3: HSC convolved to match Legacy PSF ≈1.2\" — "
    "if col 3 ≈ col 1, morphological difference is resolution, not real structure",
    fontsize=10,fontweight="bold",y=1.005)
Path(OUT_DIR).mkdir(parents=True,exist_ok=True)
fig.savefig(f"{OUT_DIR}/fig6_psf_corrected.png",bbox_inches="tight",dpi=150)
plt.close()
print(f"Saved → {OUT_DIR}/fig6_psf_corrected.png", flush=True)

# ══════════════════════════════════════════════════════════════════════════
# TABLE 3 — bootstrap CIs for 3 comparisons × 4 quantities
# ══════════════════════════════════════════════════════════════════════════
print("Building Table 3 (bootstrap CIs)...", flush=True)

q10d = sub.predicted_difficulty.quantile(0.10)
q90d = sub.predicted_difficulty.quantile(0.90)
q10r = sub.true_residual.quantile(0.10)
q90r = sub.true_residual.quantile(0.90)
q50r = sub.true_residual.quantile(0.50)

groups_df = {
    "rescue":   sub[(~sub.global_covered)&(sub.adaptive_covered)],
    "silent":   sub[(sub.predicted_difficulty<=q10d)&(sub.true_residual>=q90r)],
    "hard":     sub[(sub.predicted_difficulty>=q90d)&(sub.true_residual>=q50r)&sub.adaptive_covered],
    "false_al": sub[(sub.predicted_difficulty>=q90d)&(sub.true_residual<=q10r)],
    "easy":     sub[(sub.predicted_difficulty<=q10d)&(sub.true_residual<=q10r)&sub.global_covered],
}

# Add magnitude and size from crossmatch
print("Loading embed split for metadata...", flush=True)
embed_df = load_dataset("astro-conformal-project/split_legacysurvey_hsc_embeddings",
                        "astropt_850m", split="test").to_pandas()

meta_lookup = {}
for rid in sub.row_id.tolist():
    if rid >= len(embed_df): continue
    oid = str(embed_df.iloc[rid]["legacysurvey_object_id"])
    frow = found.get(oid)
    if frow is None: continue
    m = get_meta_corrected(frow)
    meta_lookup[rid] = m

def get_col(group_df, col):
    """Get a column, fetching from crossmatch if needed."""
    if col in group_df.columns:
        return group_df[col].dropna().values
    # crossmatch column
    vals = [meta_lookup[r][col] for r in group_df.row_id if r in meta_lookup
            and np.isfinite(meta_lookup[r].get(col, np.nan))]
    return np.array(vals)

def bootstrap_ci(a, b, n_boot=2000, seed=42):
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        sa = rng.choice(a, size=len(a), replace=True)
        sb = rng.choice(b, size=len(b), replace=True)
        diffs.append(np.median(sa) - np.median(sb))
    d = np.array(diffs)
    return np.median(d), np.percentile(d,[2.5,97.5])

# 3 pre-defined comparisons
comparisons = [
    ("rescue","easy","Rescue vs Easy stable"),
    ("silent","easy","Silent failure vs Easy stable"),
    ("false_al","hard","False alarm vs Hard correct"),
]
quantities = [
    ("mag",           "i-magnitude",        ".2f"),
    ("shape_r",       "Size SHAPE_R (\")",   ".3f"),
    ("nearest_neighbor_distance","NN distance",".4f"),
    ("true_residual", "True residual r",     ".5f"),
]

rows = []
for g1,g2,comp_label in comparisons:
    for col,q_label,fmt in quantities:
        a = get_col(groups_df[g1], col)
        b = get_col(groups_df[g2], col)
        if len(a)==0 or len(b)==0:
            rows.append([comp_label, q_label,
                         f"n={len(a)}", "—", "—"])
            continue
        med_a = np.median(a); med_b = np.median(b)
        med_diff, (lo,hi) = bootstrap_ci(a,b)
        sig = "✓" if lo*hi > 0 else "○"  # CI excludes 0?
        rows.append([
            comp_label, q_label,
            f"{med_a:{fmt}} vs {med_b:{fmt}}",
            f"{med_diff:+{fmt}}",
            f"[{lo:+{fmt}}, {hi:+{fmt}}] {sig}",
        ])
        print(f"  {comp_label} | {q_label}: {med_diff:+{fmt}} [{lo:+{fmt}},{hi:+{fmt}}] {sig}", flush=True)

# Build table figure
fig, ax = plt.subplots(figsize=(17, 5.5))
ax.axis("off")

col_labels = ["Comparison","Quantity","Medians (flagged vs control)",
              "Median difference","95% bootstrap CI"]
cell_colors = []
comp_colors = {"Rescue vs Easy stable":    "#2ecc71",
               "Silent failure vs Easy stable":"#e74c3c",
               "False alarm vs Hard correct":  "#f39c12"}
prev_comp = None
for row in rows:
    c = comp_colors.get(row[0],"#ffffff")
    alpha_c = c + "33"  # light version
    sig = row[4].endswith("✓")
    cell_colors.append([
        c+"22","white","white",
        "#c8f7c5" if sig else "white",
        "#c8f7c5" if sig else "white",
    ])

tbl = ax.table(cellText=rows,colLabels=col_labels,
               cellColours=cell_colors,
               loc="center",cellLoc="center")
tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1,1.8)
for j in range(len(col_labels)):
    tbl[(0,j)].set_facecolor("#2c3e50")
    tbl[(0,j)].set_text_props(color="white",fontweight="bold")

# group dividers
prev = None
for i,row in enumerate(rows):
    if row[0] != prev:
        for j in range(len(col_labels)):
            tbl[(i+1,j)].set_text_props(fontweight="bold")
        prev = row[0]

ax.set_title(
    "Table 3 — Effect sizes with 95% bootstrap confidence intervals\n"
    "3 pre-defined comparisons · 2000 bootstrap resamples\n"
    "✓ = CI excludes zero (effect is reliably non-zero)  "
    "○ = CI includes zero (effect uncertain at this sample size)",
    fontsize=10,fontweight="bold",pad=12)
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/table3_bootstrap_ci.png",bbox_inches="tight",dpi=150)
plt.close()
print(f"Saved → {OUT_DIR}/table3_bootstrap_ci.png", flush=True)
print("All done.", flush=True)

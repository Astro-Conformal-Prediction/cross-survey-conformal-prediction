"""
All-in-one: stream → pickle → plot. Run as a single job.
"""
import pandas as pd, numpy as np, io, warnings, pickle
warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from datasets import load_dataset

PAIRS_CSV = "/projects/bfir/ssourav/exp4_results/matched_pairs.csv"
PER_OBJ   = "/projects/bfir/ssourav/exp4_results/full/per_object_results.csv"
OUT_DIR   = "/projects/bfir/ssourav/exp4_results/figures"
CACHE     = "/projects/bfir/ssourav/exp4_results/found_objects.pkl"

# ── 1. Load pairs and per-object data ─────────────────────────────────────
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

all_rids = set()
for _,f,c in pairs: all_rids.add(f); all_rids.add(c)

# ── 2. Get object IDs ──────────────────────────────────────────────────────
print("Loading embed split...", flush=True)
embed_df = load_dataset("astro-conformal-project/split_legacysurvey_hsc_embeddings",
                        "astropt_850m", split="test").to_pandas()
row_to_oid = {rid: str(embed_df.iloc[rid]["legacysurvey_object_id"])
              for rid in all_rids if rid < len(embed_df)}
target_ids = set(row_to_oid.values())
print(f"Need {len(target_ids)} objects", flush=True)

# ── 3. Stream or load from cache ───────────────────────────────────────────
if Path(CACHE).exists():
    print("Loading from cache...", flush=True)
    with open(CACHE,"rb") as f:
        cache = pickle.load(f)
    found = cache["found"]
    row_to_oid = cache["row_to_oid"]
    print(f"Loaded {len(found)} objects from cache.", flush=True)
else:
    print("Streaming crossmatch...", flush=True)
    found = {}
    for i, row in enumerate(load_dataset(
            "Smith42/legacysurvey_hsc_crossmatched",
            split="train", streaming=True)):
        oid = str(row.get("legacysurvey_object_id",""))
        if oid in target_ids:
            found[oid] = dict(row)
            print(f"  Found {len(found)}/{len(target_ids)}: {oid}", flush=True)
        if len(found) == len(target_ids): break
        if i % 100000 == 0 and i > 0:
            print(f"  Scanned {i:,} rows...", flush=True)
    print(f"Done. Found {len(found)}/{len(target_ids)}", flush=True)
    with open(CACHE,"wb") as f:
        pickle.dump({"found":found,"row_to_oid":row_to_oid}, f)
    print(f"Cached → {CACHE}", flush=True)

# ── 4. Image helpers ───────────────────────────────────────────────────────
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
        return (np.stack([s(flux[idx[0]]),s(flux[idx[1]]),s(flux[idx[2]])],
                         axis=-1)*255).astype(np.uint8)
    except: return None

def get_imgs(frow):
    if frow is None: return None, None
    src = pil_to_arr(frow.get("rgb"))
    if src is None: src = flux_rgb(frow.get("legacysurvey_image"),["des-i","des-r","des-g"])
    tgt = flux_rgb(frow.get("hsc_image"),["hsc-i","hsc-r","hsc-g"])
    return src, tgt

def get_meta(frow):
    if frow is None: return float("nan"), float("nan")
    return float(frow.get("i_cmodel_mag",float("nan"))), \
           float(frow.get("SHAPE_R",float("nan")))

# ── 5. Figure builder ──────────────────────────────────────────────────────
ROW_META = {
    "adaptive_rescue":    ("#2ecc71","Adaptive rescue",
                           r"$r>R^{\rm glob}$, $r\leq R^{\rm adapt}$"),
    "silent_failure":     ("#e74c3c","Silent failure",
                           r"Low $\hat\sigma$, high $r$"),
    "hard_correct":       ("#9b59b6","Correctly anticipated hard",
                           r"High $\hat\sigma$, high $r$, covered"),
    "false_alarm":        ("#f39c12","False alarm",
                           r"High $\hat\sigma$, low $r$"),
    "adaptive_regression":("#e67e22","Adaptive regression",
                           r"$r\leq R^{\rm glob}$, $r>R^{\rm adapt}$"),
}

def build(cat_keys, title, outpath):
    cat_pairs = {k:[(f,c) for (ck,f,c) in pairs if ck==k] for k in cat_keys}
    row_specs = []
    for k in cat_keys:
        for i,(f,c) in enumerate(cat_pairs[k]):
            row_specs.append((k,f,c,i==0))

    n_gs = len(row_specs)
    fig = plt.figure(figsize=(4*2.8+2.6, n_gs*3.6))
    outer = gridspec.GridSpec(n_gs, 1, figure=fig, hspace=0.88)

    for ri,(cat_key,f_rid,c_rid,is_first) in enumerate(row_specs):
        col,main_lbl,sub_lbl = ROW_META[cat_key]
        inner = gridspec.GridSpecFromSubplotSpec(
            2, 4, subplot_spec=outer[ri],
            wspace=0.04, hspace=0.06, height_ratios=[3.5,1])
        img_ax = [fig.add_subplot(inner[0,j]) for j in range(4)]
        ann_ax = [fig.add_subplot(inner[1,j]) for j in range(4)]
        for ax in ann_ax: ax.axis("off")

        if is_first:
            img_ax[0].annotate(
                f"{main_lbl}\n{sub_lbl}",
                xy=(-0.55,0.5), xycoords="axes fraction",
                fontsize=8.5, color=col, fontweight="bold",
                va="center", ha="right", annotation_clip=False)

        if ri == 0:
            for j,lbl in enumerate(["Legacy (flagged)","HSC (flagged)",
                                     "Legacy (control)","HSC (control)"]):
                img_ax[j].set_title(lbl, fontsize=8, color="gray",
                                    pad=3, fontweight="bold")

        f_oid = row_to_oid.get(f_rid)
        c_oid = row_to_oid.get(c_rid)
        f_row = found.get(str(f_oid)) if f_oid else None
        c_row = found.get(str(c_oid)) if c_oid else None
        f_src,f_tgt = get_imgs(f_row)
        c_src,c_tgt = get_imgs(c_row)
        f_mag,f_sz  = get_meta(f_row)
        c_mag,c_sz  = get_meta(c_row)

        for j,(img,bc) in enumerate(zip([f_src,f_tgt,c_src,c_tgt],
                                         [col,col,"#888","#888"])):
            ax = img_ax[j]; ax.axis("off")
            for sp in ax.spines.values():
                sp.set_edgecolor(bc); sp.set_linewidth(2.2); sp.set_visible(True)
            if img is not None: ax.imshow(img, origin="upper")
            else:
                ax.set_facecolor("#ccc")
                ax.text(0.5,0.5,"not found",ha="center",va="center",
                        transform=ax.transAxes,fontsize=7)

        # flagged annotation
        if f_rid in sub.index:
            r = sub.loc[f_rid]
            if hasattr(r,'iloc'): r = r.iloc[0]
            short_fid = str(f_oid).replace("b'","").replace("'","") if f_oid else "?"
            annot = (f"Flagged  {short_fid}\n"
                     f"$r$={float(r.true_residual):.4f}  "
                     f"$\\hat{{\\sigma}}$={float(r.predicted_difficulty):.4f}\n"
                     f"$R^{{\\rm g}}$={float(r.global_radius):.4f}  "
                     f"$R^{{\\rm a}}$={float(r.adaptive_radius):.4f}  "
                     f"$i$={f_mag:.1f}  sz={f_sz:.2f}\"")
            cx=(ann_ax[0].get_position().x0+ann_ax[1].get_position().x1)/2
            cy= ann_ax[0].get_position().y0+0.003
            fig.text(cx,cy,annot,ha="center",va="bottom",fontsize=6.3,
                     color="#111",transform=fig.transFigure,
                     bbox=dict(boxstyle="round,pad=0.28",fc="#fff8f8",
                               ec=col,alpha=0.92,lw=0.9))

        # control annotation
        if c_rid in sub.index:
            r = sub.loc[c_rid]
            if hasattr(r,'iloc'): r = r.iloc[0]
            short_cid = str(c_oid).replace("b'","").replace("'","") if c_oid else "?"
            annot = (f"Control  {short_cid}\n"
                     f"$r$={float(r.true_residual):.4f}  "
                     f"$\\hat{{\\sigma}}$={float(r.predicted_difficulty):.4f}\n"
                     f"$R^{{\\rm g}}$={float(r.global_radius):.4f}  "
                     f"$R^{{\\rm a}}$={float(r.adaptive_radius):.4f}  "
                     f"$i$={c_mag:.1f}  sz={c_sz:.2f}\"")
            cx=(ann_ax[2].get_position().x0+ann_ax[3].get_position().x1)/2
            cy= ann_ax[2].get_position().y0+0.003
            fig.text(cx,cy,annot,ha="center",va="bottom",fontsize=6.3,
                     color="#111",transform=fig.transFigure,
                     bbox=dict(boxstyle="round,pad=0.28",fc="#f8f8f8",
                               ec="#888",alpha=0.92,lw=0.9))

    fig.suptitle(title, fontsize=12, fontweight="bold", y=1.005)
    Path(outpath).parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(outpath, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Saved → {outpath}", flush=True)

build(["adaptive_rescue","silent_failure","hard_correct"],
      "Figure 5 — Flagged galaxies with matched easy-stable controls\n"
      "Legacy→HSC · AstroPT-850M · Ridge · 90% coverage\n"
      "Coloured border = flagged · Grey border = matched control",
      f"{OUT_DIR}/fig5_main.png")

build(["false_alarm","adaptive_regression"],
      "Figure 5 (appendix) — Additional categories with matched controls\n"
      "Legacy→HSC · AstroPT-850M · Ridge · 90% coverage",
      f"{OUT_DIR}/fig5_appendix.png")

print("All done.", flush=True)

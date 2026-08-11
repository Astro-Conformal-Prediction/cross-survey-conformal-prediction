"""
src/plotting.py

Plot functions for Experiment 4's deliverables (spec 4.6), matching the
style already set up in plots.ipynb (same rcParams, same family palette,
same "read a results CSV, write a PNG" function shape as
plot_failure_overlap_heatmap).
"""

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import glob



mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
    "font.size": 9,
    "axes.titlesize": 11,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.linewidth": 0.6,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

DEFAULT_FAMILY_COLORS = {
    "astropt": "#4d4d4d",
    "convnext": "#e41a1c",
    "dino": "#4daf4a",
    "ijepa": "#984ea3",
    "vit": "#ff7f00",
}

BASKET_COLORS = {"Basket_A": "#377eb8", "Basket_B": "#ff7f00", "Basket_C": "#e41a1c"}


SINGLE_MODEL_COLOR =  "#000000"  # hardcoded -- edit this line directly to change the shade
 
 
import glob
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

import glob
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

# Using your exact rcParams
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
    "font.size": 9,
    "axes.titlesize": 11,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.linewidth": 0.6,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

def plot_coverage_radius_curve(input_csv, output_path, figsize=(4.5, 2.5), title=None):
    """
    Publication-ready coverage vs. radius curve.
    Accepts a single file path, a list of file paths, or a wildcard string 
    (e.g., f"{RESULTS_DIR}/jwst_basket_summary_alpha_*.csv").
    """
    # 1. Resolve file paths whether it's a wildcard string, a list, or a single path
    if isinstance(input_csv, str):
        if "*" in input_csv:
            file_paths = glob.glob(input_csv)
            if not file_paths:
                raise ValueError(f"No files found matching pattern: {input_csv}")
        else:
            file_paths = [input_csv]
    else:
        file_paths = input_csv
        
    # Read and combine all resolved CSV files
    dfs = [pd.read_csv(csv) for csv in file_paths]
    df = pd.concat(dfs, ignore_index=True)
    
    # 2. Handle the percentage column format specific to these CSVs
    if 'target_coverage_%' in df.columns:
        df['target_coverage'] = df['target_coverage_%'] / 100.0
        
    fig, ax = plt.subplots(figsize=figsize)
    
    # 3. Custom color mapping with fallbacks for other models
    models = df["Model"].unique()
    default_colors = plt.cm.tab10(np.linspace(0, 1, len(models)))
    
    color_map = {}
    for i, model in enumerate(models):
        m_lower = model.lower()
        if "worst single" in m_lower or "worst" in m_lower:
            color_map[model] = "#e41a1c"   # Red for worst single
        elif "best single" in m_lower or "best" in m_lower:
            color_map[model] = "#4daf4a"   # Green for best single
        elif "basket c" in m_lower or "basket_c" in m_lower:
            color_map[model] = "#d95f02"   # Dark orange for Basket C
        else:
            color_map[model] = default_colors[i]
 
    for model, g in df.groupby("Model"):
        # Ensure data is sorted by coverage so lines draw left-to-right
        g = g.sort_values("target_coverage")
        
        # Clean up labels for the legend
        label = model.replace("_", " ")
            
        ax.plot(
            g["target_coverage"], g["standardized_radius"],
            color=color_map[model], 
            linewidth=1.5,
            alpha=0.9,
            label=label,
            zorder=3,
            marker='o',       
            markersize=4      
        )
 
    ax.set_xlabel(r"Target coverage ($1 - \alpha$)", fontsize=9)
    ax.set_ylabel("Standardized radius", fontsize=9)
    
    if title:
        ax.set_title(title, fontsize=11)
    
    # 4. Force ticks only at the exact coverages present (e.g., 0.80, 0.90, 0.95)
    unique_x_vals = sorted(df["target_coverage"].dropna().unique())
    ax.set_xticks(unique_x_vals)

    # Strip top and right borders
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    # Move the legend outside to the right so it doesn't overlap lines
    ax.legend(
        frameon=False, 
        fontsize=7, 
        loc="center left", 
        bbox_to_anchor=(1.05, 0.5), 
        borderpad=0, 
        borderaxespad=0.2
    )
    
    fig.tight_layout()
    
    # Check output extension to save appropriately
    file_format = output_path.split('.')[-1]
    fig.savefig(output_path, format=file_format, bbox_inches="tight")
    return fig



def plot_residual_scatter(input_csv, output_path, single_model=None, figsize=(4.5, 2.5)):
    """
    Publication-ready scatter plot with high-visibility points. 
    figsize=(4.5, 2.5) fits exactly in one NeurIPS column.
    """
    df = pd.read_csv(input_csv)
    if single_model is not None:
        df = df[df["single_model"] == single_model]

    fig, ax = plt.subplots(figsize=figsize)
    
    # Render crisp, prominent points
    ax.scatter(
        df["single_normalized_residual"], 
        df["basket_normalized_residual"],
        s=25,              # Large, visible points
        alpha=0.8,         # Solid presence
        color="#08519c",   # Dark blue for strong contrast
        edgecolors="none", 
        zorder=3,
        rasterized=True
    )
    
    upper = float(max(
        df["single_normalized_residual"].max() if len(df) else 1.0,
        df["basket_normalized_residual"].max() if len(df) else 1.0,
    )) * 1.05
    
    lims = [0, upper]
    ax.plot(lims, lims, color="#4d4d4d", linewidth=0.8, linestyle="--", zorder=1)
    
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    
    # Labels matching NeurIPS caption styling
    ax.set_xlabel("Single-model normalized residual", fontsize=9)
    ax.set_ylabel("Basket normalized residual", fontsize=9)
    
    # Clean spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_seed_stability(input_csv, output_path, figsize=(6, 4),
                         title="Stability across seeds (42, 44, 46)"):
    """input_csv: <survey>_seed_stability_summary.csv (columns: Basket,
    coverage_mean, coverage_std, std_radius_mean, std_radius_std)."""
    df = pd.read_csv(input_csv).sort_values("Basket")
    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(df))
    ax.bar(
        x, df["std_radius_mean"], yerr=df["std_radius_std"], capsize=4,
        color=[BASKET_COLORS.get(b, "#bbbbbb") for b in df["Basket"]],
    )
    ax.set_xticks(x)
    ax.set_xticklabels(df["Basket"])
    ax.set_ylabel("Standardized radius (mean \u00b1 std over seeds)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path)
    return fig

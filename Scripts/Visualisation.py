# =============================================================================
# visualisation.py
#
# PURPOSE:
#   Shared plotting utilities used across all pipeline scripts.
#   Centralises common figure-saving, colour-mapping, and layout helpers so
#   that every script produces consistent, publication-quality figures without
#   duplicating boilerplate code.
#
# WHAT IS IN HERE:
#   - save_figure()          : saves any matplotlib figure to disk (PNG/PDF/SVG)
#   - make_output_dir()      : creates a per-step output folder if it doesn't exist
#   - channel_colormap()     : returns a sequential matplotlib colormap for a
#                              single-channel intensity image
#   - percentile_stretch()   : clips a 2-D array to the 2nd–98th percentile for
#                              display (avoids hot-pixels dominating the scale)
#   - build_rgb_image()      : assembles a 3-component array into an H×W×3 uint8
#                              RGB image ready for plt.imshow()
#   - label_colormap()       : discrete colormap for cluster label images
#   - plot_channel_grid()    : small-multiple grid showing one map per element
#   - add_scalebar()         : adds a physical scale bar to an axes (μm)
#   - summary_stats_table()  : prints a formatted table of per-channel stats
#
# HOW TO USE:
#   from visualisation import save_figure, build_rgb_image, ...
#   (All other scripts already import config.py for parameters.)
# =============================================================================
 
import os
import math
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.ticker import MaxNLocator
from typing import Optional, List, Tuple
 
import config
 
 
# =============================================================================
# 1. OUTPUT DIRECTORY HELPER
# =============================================================================
 
def make_output_dir(base_dir: str, step_name: str) -> str:
    """
    Creates (if necessary) a sub-folder inside base_dir for a pipeline step.
 
    For example, make_output_dir("/results", "umap") creates
    "/results/umap/" and returns that path.
 
    Parameters
    ----------
    base_dir  : root output directory (created if it doesn't exist)
    step_name : name of the pipeline step — used as the sub-folder name
 
    Returns
    -------
    str : full path to the created sub-folder
    """
    # Build the full path: base_dir / step_name
    out_path = os.path.join(base_dir, step_name)
 
    # os.makedirs with exist_ok=True is safe to call even if the folder
    # already exists — it won't raise an error.
    os.makedirs(out_path, exist_ok=True)
 
    return out_path
 
 
# =============================================================================
# 2. FIGURE SAVING
# =============================================================================
 
def save_figure(
    fig: plt.Figure,
    out_dir: str,
    filename: str,
    fmt: Optional[str] = None,
    dpi: Optional[int] = None,
    tight: bool = True,
) -> str:
    """
    Saves a matplotlib figure to disk.
 
    Parameters
    ----------
    fig      : the matplotlib Figure object to save
    out_dir  : directory where the file will be written
    filename : base filename WITHOUT extension (e.g. "umap_rgb")
    fmt      : file format string, e.g. "png", "pdf", "svg"
               Defaults to config.FIGURE_FORMAT if not provided.
    dpi      : dots-per-inch resolution
               Defaults to config.FIGURE_DPI if not provided.
    tight    : if True, calls tight_layout() before saving to remove extra
               white space around the figure
 
    Returns
    -------
    str : full path of the saved file
    """
    fmt = fmt or config.FIGURE_FORMAT   # fall back to global config setting
    dpi = dpi or config.FIGURE_DPI      # fall back to global config setting
 
    # Ensure the output directory exists
    os.makedirs(out_dir, exist_ok=True)
 
    # Build the full file path
    full_path = os.path.join(out_dir, f"{filename}.{fmt}")
 
    # tight_layout() adjusts subplot parameters so that the figure fits the
    # canvas cleanly without clipped labels.
    if tight:
        try:
            fig.tight_layout()
        except Exception:
            # tight_layout can fail on complex layouts — skip silently
            pass
 
    fig.savefig(full_path, dpi=dpi, bbox_inches="tight")
    print(f"  Saved → {full_path}")
 
    return full_path
 
 
# =============================================================================
# 3. COLOUR MAP FOR SINGLE-CHANNEL INTENSITY IMAGES
# =============================================================================
 
def channel_colormap(channel_name: str = "") -> matplotlib.colors.Colormap:
    """
    Returns a suitable sequential matplotlib colormap for displaying a single
    elemental channel as a heatmap.
 
    Currently returns 'inferno' for all channels — warm, perceptually uniform,
    and prints well in greyscale. Could be extended to return channel-specific
    colourmaps (e.g. hot red for iron, cool blue for zinc) if desired.
 
    Parameters
    ----------
    channel_name : element name (unused now, reserved for future per-element logic)
 
    Returns
    -------
    matplotlib.colors.Colormap
    """
    # 'inferno' is a perceptually uniform sequential colormap — low values are
    # dark/purple, high values are bright yellow. Works well for intensity data.
    return plt.cm.inferno
 
 
# =============================================================================
# 4. PERCENTILE STRETCH FOR DISPLAY
# =============================================================================
 
def percentile_stretch(
    arr: np.ndarray,
    low_pct: float = 2.0,
    high_pct: float = 98.0,
) -> np.ndarray:
    """
    Clips a 2-D or 1-D array to the given percentile range and rescales to [0, 1].
 
    This is used ONLY for display — the actual data passed to UMAP/PCA is never
    clipped. A small number of very bright or very dark pixels can dominate a
    colour scale, making the rest of the image look flat. Percentile stretch
    removes those outliers from the colour scale so that the tissue structure
    is visible.
 
    Parameters
    ----------
    arr      : numpy array (any shape, float)
    low_pct  : lower percentile to clip (default 2 %)
    high_pct : upper percentile to clip (default 98 %)
 
    Returns
    -------
    np.ndarray : float32 array with values in [0, 1]
    """
    # Compute the clip bounds from the data itself
    vmin = np.percentile(arr, low_pct)
    vmax = np.percentile(arr, high_pct)
 
    # Avoid division by zero if the channel is completely flat
    if vmax == vmin:
        return np.zeros_like(arr, dtype=np.float32)
 
    # Clip and rescale to [0, 1]
    stretched = (arr - vmin) / (vmax - vmin)
    stretched = np.clip(stretched, 0.0, 1.0).astype(np.float32)
 
    return stretched
 
 
# =============================================================================
# 5. BUILD RGB IMAGE FROM 3-COMPONENT ARRAY
# =============================================================================
 
def build_rgb_image(
    components: np.ndarray,
    tissue_indices: np.ndarray,
    height: int,
    width: int,
    background: Tuple[int, int, int] = (0, 0, 0),
    stretch: bool = True,
) -> np.ndarray:
    """
    Assembles a 3-component embedding (e.g. UMAP or PCA) into an H×W×3 uint8
    RGB image that can be displayed with plt.imshow().
 
    Each of the 3 components is mapped to R, G, B respectively after
    optional percentile stretch.
 
    Parameters
    ----------
    components      : array of shape (n_tissue_pixels, 3) — the embedding
    tissue_indices  : flat pixel indices (into the H×W grid) for tissue pixels
                      (same as tissue_indices_final from preprocessing)
    height, width   : spatial dimensions of the original image
    background      : RGB tuple for non-tissue pixels (default black)
    stretch         : if True, apply percentile stretch to each component
                      (uses 2nd–98th percentile)
 
    Returns
    -------
    np.ndarray : uint8 array of shape (height, width, 3)
    """
    # Initialise a blank image filled with the background colour
    rgb = np.full((height * width, 3), background, dtype=np.float32)
 
    # For each of the 3 components, stretch to [0, 1] then place into the image
    for c in range(3):
        channel = components[:, c].astype(np.float32)
 
        if stretch:
            channel = percentile_stretch(channel)
        else:
            # Just rescale min–max to [0, 1] without clipping
            cmin, cmax = channel.min(), channel.max()
            if cmax > cmin:
                channel = (channel - cmin) / (cmax - cmin)
 
        # Write only to tissue pixel positions; background stays as initialised
        rgb[tissue_indices, c] = channel
 
    # Rescale to uint8 (0–255) and reshape to spatial dimensions
    rgb_img = (rgb * 255).clip(0, 255).astype(np.uint8).reshape(height, width, 3)
 
    return rgb_img
 
 
# =============================================================================
# 6. DISCRETE COLOURMAP FOR CLUSTER LABELS
# =============================================================================
 
def label_colormap(
    n_labels: int,
    include_noise: bool = False,
    noise_color: str = "lightgrey",
) -> Tuple[matplotlib.colors.ListedColormap, matplotlib.colors.BoundaryNorm]:
    """
    Creates a discrete colourmap for plotting cluster label images.
 
    Cluster labels are integers (0, 1, 2, ...). Noise pixels (label = -1 in
    HDBSCAN) can optionally be added as a fixed grey colour at the bottom of
    the colourmap.
 
    Parameters
    ----------
    n_labels      : number of distinct cluster labels (not counting noise)
    include_noise : if True, prepend a grey colour for the noise label (-1)
    noise_color   : colour string for noise pixels
 
    Returns
    -------
    cmap  : ListedColormap with one colour per label (+ optional noise colour)
    norm  : BoundaryNorm mapping label integers to colour slots
    """
    # Use matplotlib's tab20 for up to 20 clusters, then wrap around
    base_colors = plt.cm.tab20.colors  # tuple of 20 RGB tuples
 
    # Build a list of colours, repeating tab20 if n_labels > 20
    cluster_colors = [base_colors[i % len(base_colors)] for i in range(n_labels)]
 
    if include_noise:
        # Prepend the noise colour at position 0
        # Noise label (-1) will be mapped to the first slot
        all_colors = [mcolors.to_rgba(noise_color)] + [
            mcolors.to_rgba(c) for c in cluster_colors
        ]
        boundaries = list(range(-1, n_labels + 1))
    else:
        all_colors = [mcolors.to_rgba(c) for c in cluster_colors]
        boundaries = list(range(0, n_labels + 1))
 
    cmap = mcolors.ListedColormap(all_colors)
    norm = mcolors.BoundaryNorm(boundaries, cmap.N)
 
    return cmap, norm
 
 
# =============================================================================
# 7. SMALL-MULTIPLE CHANNEL GRID
# =============================================================================
 
def plot_channel_grid(
    df: pd.DataFrame,
    tissue_indices: np.ndarray,
    height: int,
    width: int,
    out_dir: str,
    filename: str = "channel_grid",
    title: str = "Channel overview",
    cmap: str = "inferno",
    ncols: int = 4,
) -> plt.Figure:
    """
    Creates a grid of small spatial maps — one per element in df.
 
    Each map shows the log1p-normalised intensity of one element across the
    tissue, using percentile stretch for the colour scale.
 
    Parameters
    ----------
    df              : pixel dataframe of shape (n_tissue_pixels, n_channels)
    tissue_indices  : flat pixel indices for tissue pixels
    height, width   : spatial dimensions
    out_dir         : where to save the figure
    filename        : base filename (no extension)
    title           : figure-level super-title
    cmap            : matplotlib colormap name for all subplots
    ncols           : number of columns in the grid
 
    Returns
    -------
    matplotlib.figure.Figure
    """
    channels = df.columns.tolist()
    n = len(channels)
 
    # Calculate number of rows needed for the grid
    nrows = math.ceil(n / ncols)
 
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 3))
 
    # Flatten axes array for easy iteration even if nrows=1
    axes_flat = np.array(axes).flatten()
 
    for idx, ch in enumerate(channels):
        ax = axes_flat[idx]
 
        # Build a blank background image
        img = np.zeros(height * width, dtype=np.float32)
 
        # Fill tissue pixels with percentile-stretched intensity
        vals = df[ch].values.astype(np.float32)
        img[tissue_indices] = percentile_stretch(vals)
 
        ax.imshow(img.reshape(height, width), cmap=cmap, vmin=0, vmax=1)
        ax.set_title(ch, fontsize=8)
        ax.axis("off")
 
    # Hide any unused subplot panels (if n is not a multiple of ncols)
    for idx in range(n, len(axes_flat)):
        axes_flat[idx].axis("off")
 
    fig.suptitle(title, fontsize=12, y=1.01)
 
    save_figure(fig, out_dir, filename)
 
    return fig
 
 
# =============================================================================
# 8. SCALE BAR
# =============================================================================
 
def add_scalebar(
    ax: plt.Axes,
    pixel_size_um: float,
    bar_um: float = 500.0,
    loc: str = "lower right",
    color: str = "white",
    fontsize: int = 8,
) -> None:
    """
    Adds a physical scale bar to a matplotlib Axes.
 
    Parameters
    ----------
    ax            : the axes to annotate
    pixel_size_um : physical size of one pixel in micrometres
    bar_um        : desired scale bar length in micrometres (default 500 μm)
    loc           : corner location — "lower right", "lower left",
                    "upper right", "upper left"
    color         : colour of the scale bar and label
    fontsize      : font size for the label
    """
    # Convert bar length from micrometres to pixels
    bar_px = bar_um / pixel_size_um
 
    # Get current axis limits to position the bar
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    x_range = xlim[1] - xlim[0]
    y_range = ylim[1] - ylim[0]
 
    margin_x = 0.05 * x_range   # 5 % margin from edge
    margin_y = 0.08 * y_range
 
    # Determine anchor position based on loc
    if "right" in loc:
        x_end = xlim[1] - margin_x
        x_start = x_end - bar_px
    else:
        x_start = xlim[0] + margin_x
        x_end = x_start + bar_px
 
    if "lower" in loc:
        y_pos = ylim[0] + margin_y   # note: imshow has y-axis inverted
    else:
        y_pos = ylim[1] - margin_y
 
    # Draw the scale bar as a thick horizontal line
    ax.plot([x_start, x_end], [y_pos, y_pos], color=color, linewidth=3,
            solid_capstyle="butt")
 
    # Label in micrometres
    label = f"{bar_um:.0f} μm" if bar_um >= 1 else f"{bar_um * 1000:.0f} nm"
    ax.text(
        (x_start + x_end) / 2, y_pos - 0.02 * y_range,
        label,
        ha="center", va="top",
        color=color, fontsize=fontsize,
    )
 
 
# =============================================================================
# 9. SUMMARY STATISTICS TABLE
# =============================================================================
 
def summary_stats_table(df: pd.DataFrame, title: str = "Channel statistics") -> None:
    """
    Prints a formatted table of per-channel descriptive statistics to stdout.
 
    Shows: min, 1st percentile, median, mean, 99th percentile, max, and the
    fraction of pixels that are exactly zero (useful for checking sparsity
    after log1p normalisation).
 
    Parameters
    ----------
    df    : pixel dataframe (n_tissue_pixels × n_channels)
    title : heading printed above the table
    """
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")
 
    # Build the stats DataFrame
    stats = pd.DataFrame({
        "min":    df.min(),
        "p01":    df.quantile(0.01),
        "median": df.median(),
        "mean":   df.mean(),
        "p99":    df.quantile(0.99),
        "max":    df.max(),
        "zero%":  (df == 0).mean() * 100,  # percentage of zero-valued pixels
    })
 
    # Round for readability
    print(stats.round(4).to_string())
    print(f"{'=' * 60}\n")
 
 
# =============================================================================
# 10. LEGEND PATCH HELPER
# =============================================================================
 
def make_legend_patches(
    labels: List[str],
    colors: List,
    title: str = "",
) -> List[mpatches.Patch]:
    """
    Creates a list of matplotlib Patch objects for use in a legend.
 
    Handy when the legend entries correspond to named clusters or channels
    rather than line/scatter artists.
 
    Parameters
    ----------
    labels : list of label strings
    colors : list of colour specs (strings, RGB tuples, etc.) — same length as labels
    title  : optional legend title (not added here, pass to ax.legend())
 
    Returns
    -------
    list of matplotlib.patches.Patch
    """
    patches = [
        mpatches.Patch(color=c, label=l)
        for c, l in zip(colors, labels)
    ]
    return patches
 
 
# =============================================================================
# 11. GLOBAL STYLE SETTINGS
# =============================================================================
 
def set_publication_style() -> None:
    """
    Applies global matplotlib rcParams for publication-quality figures.
 
    Call this once at the top of any script that produces final figures.
    Settings include larger fonts, thicker lines, and no top/right spines.
    """
    plt.rcParams.update({
        # Font sizes — larger than defaults for readability in papers/thesis
        "font.size":        11,
        "axes.titlesize":   12,
        "axes.labelsize":   11,
        "xtick.labelsize":  9,
        "ytick.labelsize":  9,
        "legend.fontsize":  9,
        "figure.titlesize": 13,
 
        # Line and marker weights
        "lines.linewidth":  1.5,
        "axes.linewidth":   0.8,
 
        # Remove top and right spines for a cleaner look
        "axes.spines.top":   False,
        "axes.spines.right": False,
 
        # White background (not grey)
        "axes.facecolor":   "white",
        "figure.facecolor": "white",
 
        # Save with tight bounding box by default
        "savefig.bbox":     "tight",
    })
# =============================================================================
# umap_reduction.py
#
# Purpose:
#   Runs UMAP on the log1p-transformed pixel data and produces all UMAP
#   visualisations. 
#
# UMAP (Uniform Manifold Approximation and Projection) reduces the high-
# dimensional elemental data (one dimension per element) down to 3 dimensions
# that preserve the local structure of the data. Pixels with similar elemental
# profiles end up close together in UMAP space. 
#  
# UMAP of three components because  
#  3D UMAP allows the three axes to be mapped directly to RGB colour channels,
#   producing a spatial image where colour encodes elemental similarity.
#   Structurally similar pixels appear the same colour in the tissue image.
#
# Order of steps:
#   1. Run UMAP (GPU via cuML if available, CPU fallback via umap-learn)
#   2. RGB spatial map, static matplotlib version
#   3. RGB spatial map, interactive Plotly version with per-pixel hover info
#   4. 3D scatter plot coloured by UMAP RGB (the "legend" for the spatial map)
#   5. 2D scatter per channel (auto log/linear from scale_suggestions)
#   6. All-channels grid, one 2D UMAP scatter per element
#   7. Dominant element map, each pixel coloured by its strongest element
# =============================================================================

import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import plotly.express as px
import plotly.graph_objects as go

import config


# =============================================================================
# 1. Run UMAP
# =============================================================================

def run_umap(df_normalised: pd.DataFrame):
    """
    Runs UMAP on the log1p-transformed pixel dataframe.
    Uses cuML GPU UMAP when config.USE_GPU = True, otherwise umap-learn (CPU).

    Settings (from config.py):
        UMAP_N_COMPONENTS = 3    : 3D embedding for RGB spatial map
        UMAP_N_NEIGHBORS  = 30   : how many neighbouring pixels to consider
                                   when learning the local structure
        UMAP_MIN_DIST     = 0.0  : allows clusters to pack tightly together
        UMAP_METRIC       = cosine: measures similarity by angle rather than
                                   absolute distance, better for intensity data
        UMAP_RANDOM_STATE = 42   : ensures reproducible results

    Parameters:
        df_normalised : log1p-transformed pixel dataframe, shape (n_pixels, n_channels)

    Returns:
        X_umap : numpy array of UMAP coordinates, shape (n_pixels, 3)
    """
    if config.USE_GPU:
        try:
            from cuml.manifold import UMAP as cuUMAP
            import cupy as cp
            print("Running UMAP on GPU (cuML).")
            reducer = cuUMAP(
                n_components=config.UMAP_N_COMPONENTS,
                n_neighbors=config.UMAP_N_NEIGHBORS,
                min_dist=config.UMAP_MIN_DIST,
                metric=config.UMAP_METRIC,
                random_state=config.UMAP_RANDOM_STATE,
                verbose=True,
            )
            X_gpu = cp.asarray(df_normalised.values.astype("float32"))
            X_umap = reducer.fit_transform(X_gpu)
            X_umap = cp.asnumpy(X_umap)
            print(f"\nUMAP complete (GPU). Output shape: {X_umap.shape}")
            return X_umap, None
        except Exception as e:
            print(f"  GPU UMAP failed ({e}). Falling back to CPU.")

    import umap
    print("Running UMAP on CPU (this may take several minutes for large datasets).")
    reducer = umap.UMAP(
        n_components=config.UMAP_N_COMPONENTS,
        n_neighbors=config.UMAP_N_NEIGHBORS,
        min_dist=config.UMAP_MIN_DIST,
        metric=config.UMAP_METRIC,
        random_state=config.UMAP_RANDOM_STATE,
        verbose=True,
    )
    X_umap = reducer.fit_transform(df_normalised.values)
    print(f"\nUMAP complete. Output shape: {X_umap.shape}")
    return X_umap, None


# =============================================================================
# 2. RGB spatial map
# =============================================================================

def _draw_scale_bar(ax, pixel_size_um, width_px, height_px, color='white'):
    """
    Draws a scale bar in the bottom-left corner of a spatial image axes.
    Automatically picks a round bar length (~15% of image width).
    """
    image_width_um = width_px * pixel_size_um
    raw_um    = image_width_um * 0.15
    magnitude = 10 ** np.floor(np.log10(raw_um))
    nice_um   = magnitude
    for factor in [1, 2, 5, 10]:
        if factor * magnitude >= raw_um * 0.5:
            nice_um = factor * magnitude
            break
    bar_px = nice_um / pixel_size_um
    x0 = width_px  * 0.04
    y0 = height_px * 0.94
    ax.plot([x0, x0 + bar_px], [y0, y0], color=color, lw=5,
            solid_capstyle='butt', transform=ax.transData)
    label = f'{int(nice_um)} µm' if nice_um >= 1 else f'{nice_um:.1f} µm'
    ax.text(x0 + bar_px / 2, y0 - height_px * 0.01, label,
            color=color, ha='center', va='top',
            fontsize=11, fontweight='bold', transform=ax.transData)


def plot_umap_rgb(X_umap: np.ndarray, tissue_indices_final: np.ndarray,
                   height: int, width: int,
                   pixel_size_um: float = None,
                   output_dir: str = None, show_plot: bool = False):
    """
    Creates a spatial RGB image where each tissue pixel's colour encodes its
    position in UMAP space:
        Red channel   = UMAP dimension 1 (normalised to 0–1)
        Green channel = UMAP dimension 2 (normalised to 0–1)
        Blue channel  = UMAP dimension 3 (normalised to 0–1)

    Also adds:
      - Inset UMAP scatter (UMAP1 vs UMAP2, coloured by the same RGB) in the
        bottom-right corner, acts as a colour legend for the spatial map.
      - Scale bar (bottom-left) if pixel_size_um is provided.

    Parameters:
        X_umap               : UMAP coordinates, shape (n_pixels, 3)
        tissue_indices_final : flat pixel indices of tissue pixels
        height, width        : original image dimensions
        pixel_size_um        : µm per pixel for scale bar (None = skip scale bar)
        output_dir           : if provided, saves the figure here
        show_plot            : if True, displays the figure

    Returns:
        rgb_image  : (H, W, 3) float array of the RGB image
        umap_norm  : (n_pixels, 3) normalised UMAP coordinates (used by other plots)
    """
    from skimage.exposure import equalize_hist
    umap_norm = np.zeros_like(X_umap, dtype=float)
    for c in range(3):
        col  = X_umap[:, c]
        vmin = np.percentile(col, 2)
        vmax = np.percentile(col, 98)
        clipped = np.clip((col - vmin) / (vmax - vmin), 0, 1)
        umap_norm[:, c] = equalize_hist(clipped)   # spreads values uniformly → vivid RGB

    rgb_image = np.ones((height, width, 3), dtype=float)
    ys, xs    = np.unravel_index(tissue_indices_final, (height, width))
    rgb_image[ys, xs, :] = umap_norm

    fig, ax = plt.subplots(figsize=(12, 7), facecolor='white')
    ax.imshow(rgb_image, origin='upper')
    ax.set_title("UMAP RGB Spatial Map: R=UMAP1  G=UMAP2  B=UMAP3", fontsize=14)
    ax.axis('off')

    # --- Scale bar ---
    if pixel_size_um is not None:
        _draw_scale_bar(ax, pixel_size_um, width, height)

    plt.tight_layout()

    if output_dir:
        plt.savefig(os.path.join(output_dir, "umap_rgb_map.png"),
                    dpi=config.FIGURE_DPI, bbox_inches='tight')
    if show_plot:
        plt.show()
    plt.close()

    return rgb_image, umap_norm


# =============================================================================
# 3. 3D Scatter interactive 
# =============================================================================

def plot_umap_3d_scatter(X_umap: np.ndarray, umap_norm: np.ndarray,
                          output_dir: str = None):
    """
    3D interactive scatter plot of the UMAP embedding, where each point is
    coloured by its own UMAP-derived RGB colour.

    This plot acts as the colour legend for the RGB spatial map, it shows
    what position in UMAP space corresponds to what colour in the tissue image.

    For large datasets, subsamples to 60,000 pixels for performance.

    Parameters:
        X_umap     : UMAP coordinates, shape (n_pixels, 3)
        umap_norm  : normalised UMAP coordinates [0,1], from plot_umap_rgb()
        output_dir : if provided, saves as interactive HTML
    """
    # Subsample for interactive performance
    rng = np.random.default_rng(42)
    n_plot = min(60_000, len(X_umap))
    idx = rng.choice(len(X_umap), size=n_plot, replace=False)

    # Each point's colour is its own UMAP RGB value
    colours_rgb = umap_norm[idx]
    colours_str = [f'rgb({int(r*255)},{int(g*255)},{int(b*255)})'
                   for r, g, b in colours_rgb]

    fig = go.Figure(data=go.Scatter3d(
        x=X_umap[idx, 0], y=X_umap[idx, 1], z=X_umap[idx, 2],
        mode='markers',
        marker=dict(size=2, color=colours_str, opacity=0.8),
        hovertemplate='UMAP1: %{x:.2f}<br>UMAP2: %{y:.2f}<br>UMAP3: %{z:.2f}<extra></extra>',
    ))

    fig.update_layout(
        title='UMAP 3D Space: coloured by RGB encoding'
              '<br><sub>This is the colour legend for the RGB spatial map</sub>',
        scene=dict(
            xaxis=dict(title='UMAP 1 (R', backgroundcolor='black'),
            yaxis=dict(title='UMAP 2 (G', backgroundcolor='black'),
            zaxis=dict(title='UMAP 3 (B', backgroundcolor='black'),
        ),
        width=900, height=800,
        margin=dict(l=0, r=0, t=80, b=0),
    )

    if output_dir:
        fig.write_html(os.path.join(output_dir, "umap_3d_scatter.html"))


# =============================================================================
# 4. 3D Scatter static
# =============================================================================

def plot_umap_3d_scatter_png(X_umap: np.ndarray, umap_norm: np.ndarray,
                              output_dir: str = None, show_plot: bool = False):
    """
    Static matplotlib version of the 3D UMAP scatter plot, saved as PNG.
    Points are coloured by their UMAP-derived RGB value, matching the spatial map.
    Subsamples to 50,000 points for performance.
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    rng    = np.random.default_rng(42)
    n_plot = min(50_000, len(X_umap))
    idx    = rng.choice(len(X_umap), size=n_plot, replace=False)

    colours = umap_norm[idx]  # already [0,1] RGB

    fig = plt.figure(figsize=(10, 8), facecolor='black')
    ax  = fig.add_subplot(111, projection='3d', facecolor='black')

    ax.scatter(
        X_umap[idx, 0], X_umap[idx, 1], X_umap[idx, 2],
        c=colours, s=1.5, alpha=0.7, rasterized=True
    )

    ax.set_xlabel('UMAP 1', color='white', labelpad=8)
    ax.set_ylabel('UMAP 2', color='white', labelpad=8)
    ax.set_zlabel('UMAP 3', color='white', labelpad=8)
    ax.set_title('UMAP 3D Embedding\n(coloured by RGB encoding)',
                 color='white', fontsize=13, pad=12)

    for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
        pane.fill = False
        pane.set_edgecolor('grey')

    ax.tick_params(colors='white', labelsize=8)
    ax.xaxis.line.set_color('grey')
    ax.yaxis.line.set_color('grey')
    ax.zaxis.line.set_color('grey')

    plt.tight_layout()

    if output_dir:
        plt.savefig(os.path.join(output_dir, "umap_3d_scatter.png"),
                    dpi=config.FIGURE_DPI, bbox_inches='tight',
                    facecolor='black')
    if show_plot:
        plt.show()
    plt.close()


# =============================================================================
# 5. 2D scatter static
# =============================================================================

def plot_umap_2d_scatter_png(X_umap: np.ndarray, umap_norm: np.ndarray,
                              output_dir: str = None, show_plot: bool = False):
    """
    Static matplotlib 2D scatter plot of UMAP1 vs UMAP2, coloured by the
    same RGB encoding as the spatial map. Subsamples to 80,000 points.
    """
    rng    = np.random.default_rng(42)
    n_plot = min(80_000, len(X_umap))
    idx    = rng.choice(len(X_umap), size=n_plot, replace=False)

    colours = umap_norm[idx]

    fig, ax = plt.subplots(figsize=(9, 7), facecolor='black')
    ax.set_facecolor('black')

    ax.scatter(
        X_umap[idx, 0], X_umap[idx, 1],
        c=colours, s=1.0, alpha=0.6, rasterized=True
    )

    ax.set_xlabel('UMAP 1', color='white', fontsize=12)
    ax.set_ylabel('UMAP 2', color='white', fontsize=12)
    ax.set_title(
        f'UMAP 2D Embedding  '
        f'(n_neighbours={config.UMAP_N_NEIGHBORS}, '
        f'min_dist={config.UMAP_MIN_DIST})',
        color='white', fontsize=13
    )
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_edgecolor('grey')

    plt.tight_layout()

    if output_dir:
        plt.savefig(os.path.join(output_dir, "umap_2d_scatter.png"),
                    dpi=config.FIGURE_DPI, bbox_inches='tight',
                    facecolor='black')
    if show_plot:
        plt.show()
    plt.close()


# =============================================================================
# 6. 2D UMAO scatter per channel
# =============================================================================

def plot_umap_by_channel(X_umap: np.ndarray, df: pd.DataFrame,
                          channel_name: str, scale_suggestions: dict,
                          output_dir: str = None):
    """
    2D UMAP scatter plot coloured by the intensity of a single element.

    Automatically shows a second log-scale version if the channel was flagged
    as LOG or POSSIBLY LOG by suggest_scale() in preprocessing.py.

    This is useful for identifying which region of UMAP space corresponds to
    high concentrations of a particular element (e.g. where is Fe enriched?).

    Parameters:
        X_umap           : UMAP coordinates, shape (n_pixels, 3)
        df               : pixel dataframe with raw element intensities
        channel_name     : name of the element to colour by (e.g. 'Fe', 'Zn')
        scale_suggestions: dict from preprocessing.suggest_scale()
        output_dir       : if provided, saves figures here
    """
    suggestion = scale_suggestions.get(channel_name, 'LINEAR')

    color_linear = df[channel_name].reset_index(drop=True)
    color_log    = np.log1p(color_linear)

    # Always show linear version
    fig_linear = px.scatter(
        x=X_umap[:, 0], y=X_umap[:, 1],
        color=color_linear,
        color_continuous_scale='hot',
        title=f'UMAP 2D {channel_name} (Linear)',
        labels={'x': 'UMAP 1', 'y': 'UMAP 2'},
    )
    fig_linear.update_traces(marker=dict(size=5, opacity=0.6))
    fig_linear.update_layout(coloraxis_colorbar=dict(title=channel_name),
                              width=1000, height=800)
    if output_dir:
        fig_linear.write_html(os.path.join(output_dir, f"umap_2d_{channel_name}_linear.html"))

    # Only show log version if flagged
    if suggestion in ['LOG', 'POSSIBLY LOG']:
        fig_log = px.scatter(
            x=X_umap[:, 0], y=X_umap[:, 1],
            color=color_log,
            color_continuous_scale='hot',
            title=f'UMAP 2D {channel_name} Log Scale (auto-flagged: {suggestion})',
            labels={'x': 'UMAP 1', 'y': 'UMAP 2'},
        )
        fig_log.update_traces(marker=dict(size=5, opacity=0.6))
        fig_log.update_layout(coloraxis_colorbar=dict(title=f'{channel_name} (log)'),
                               width=1000, height=800)
        if output_dir:
            fig_log.write_html(os.path.join(output_dir, f"umap_2d_{channel_name}_log.html"))


# =============================================================================
# 7. All channels grid
# =============================================================================

def plot_umap_all_channels(X_umap: np.ndarray, df: pd.DataFrame,
                            channel_names_filtered: list, scale_suggestions: dict,
                            output_dir: str = None, show_plot: bool = False):
    """
    Plots a grid of 2D UMAP scatter plots, one panel per element.

    Each panel shows the UMAP embedding coloured by that element's intensity,
    automatically applying log scale for channels flagged by suggest_scale().

    This gives a complete overview of how each element distributes across
    the UMAP embedding in a single figure.

    Parameters:
        X_umap                : UMAP coordinates, shape (n_pixels, 3)
        df                    : pixel dataframe with raw element intensities
        channel_names_filtered: list of all channel name strings
        scale_suggestions     : dict from preprocessing.suggest_scale()
        output_dir            : if provided, saves the figure here
        show_plot             : if True, displays the figure
    """
    n    = len(channel_names_filtered)
    cols = 4
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
    axes = axes.flatten()

    for i, channel_name in enumerate(channel_names_filtered):
        color_vals = df.iloc[:, i].reset_index(drop=True)

        # Apply log if flagged
        suggestion = scale_suggestions.get(channel_name, 'LINEAR')
        if suggestion in ['LOG', 'POSSIBLY LOG']:
            color_vals  = np.log1p(color_vals)
            scale_label = '(log)'
        else:
            scale_label = ''

        sc = axes[i].scatter(
            X_umap[:, 0], X_umap[:, 1],
            c=color_vals, cmap='hot', s=0.3, rasterized=True
        )
        plt.colorbar(sc, ax=axes[i], shrink=0.7)
        axes[i].set_title(f'{channel_name} {scale_label}', fontsize=12)
        axes[i].axis('off')

    # Hide any unused subplot panels
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.suptitle('UMAP: All Elements', fontsize=18, y=1.01)
    plt.tight_layout()

    if output_dir:
        plt.savefig(os.path.join(output_dir, "umap_all_channels.png"),
                    dpi=config.FIGURE_DPI, bbox_inches='tight')
    if show_plot:
        plt.show()
    plt.close()


# =============================================================================
# 8. Dominant element map 
# =============================================================================

def plot_umap_dominant_element(X_umap: np.ndarray, df: pd.DataFrame,
                                output_dir: str = None, show_plot: bool = False):
    """
    Colours each pixel in the UMAP scatter by which element is dominant at that pixel.

    For each pixel, the element with the highest normalised intensity is treated
    as the 'dominant' element and determines the pixel's colour.

    Normalisation uses 99th percentile clipping so that outlier hotspot pixels
    don't overwhelm all others in a single element.

    This plot is useful for quickly identifying whether any element strongly
    separates into a distinct region of UMAP space, which would suggest that
    element defines a tissue compartment.

    Parameters:
        X_umap     : UMAP coordinates, shape (n_pixels, 3)
        df         : pixel dataframe with raw element intensities
        output_dir : if provided, saves the figure here
        show_plot  : if True, displays the figure
    """
    X = df.values.astype(float)

    # Normalise each column by its 99th percentile then clip to [0,1]
    # This prevents a single very-high-intensity channel from always winning
    X_norm = np.clip(X / (np.percentile(X, 99, axis=0) + 1e-10), 0, 1)

    # For each pixel, find which column (element) has the highest normalised value
    dominant = X_norm.argmax(axis=1)

    # Assign a distinct colour to each element
    n = df.shape[1]
    labels = list(df.columns)
    palette = (plt.cm.tab10(np.linspace(0, 1, n)) if n <= 10
               else plt.cm.tab20(np.linspace(0, 1, n)))
    pixel_colours = palette[dominant]

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.scatter(X_umap[:, 0], X_umap[:, 1],
               c=pixel_colours, s=0.3, alpha=0.7)

    # Legend: one patch per element
    handles = [mpatches.Patch(color=palette[i], label=labels[i]) for i in range(n)]
    ax.legend(handles=handles, title='Dominant element',
              bbox_to_anchor=(1.02, 1), loc='upper left', frameon=False)

    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    ax.set_title('UMAP: Dominant element per pixel', fontsize=14)
    plt.tight_layout()

    if output_dir:
        plt.savefig(os.path.join(output_dir, "umap_dominant_element.png"),
                    dpi=config.FIGURE_DPI, bbox_inches='tight')
    if show_plot:
        plt.show()
    plt.close()


# =============================================================================
# 9. Element key and RGB map  + per element thumbnails
# =============================================================================

def plot_element_key(
    rgb_image: np.ndarray,
    df: pd.DataFrame,
    tissue_indices_final: np.ndarray,
    height: int,
    width: int,
    output_dir: str = None,
    show_plot: bool = False,
):
    """
    Produces a combined figure for biological interpretation of the RGB map:

    LEFT  : the UMAP RGB spatial map (the main output)
    RIGHT : a grid of small per-element intensity maps, one per channel

    Each thumbnail shows where that element is enriched in the tissue.
    By comparing the RGB map on the left with the thumbnails on the right,
    you can identify which colour region corresponds to which element.

    For example, if the teal region in the RGB map lights up brightly on
    the P thumbnail, you know that region is phosphorus-rich (likely brain
    or mineralising tissue).

    Parameters:
        rgb_image            : (H, W, 3) float array from plot_umap_rgb()
        df                   : pixel dataframe with log1p-normalised intensities
        tissue_indices_final : flat pixel indices of tissue pixels
        height, width        : spatial dimensions of the original image
        output_dir           : if provided, saves the figure here
        show_plot            : if True, displays the figure
    """
    channels = df.columns.tolist()
    n = len(channels)

    # Layout: 1 large RGB map on the left, grid of thumbnails on the right
    # We arrange thumbnails in 2 columns
    thumb_cols = 2
    thumb_rows = math.ceil(n / thumb_cols)

    # Figure width: RGB map (6 units) + thumbnail grid (4 units)
    fig = plt.figure(figsize=(10, max(6, thumb_rows * 2.2)))

    # Use gridspec for flexible layout
    import matplotlib.gridspec as gridspec
    gs = gridspec.GridSpec(
        thumb_rows, thumb_cols + 2,   # +2 columns for the RGB map
        figure=fig,
        wspace=0.05,
        hspace=0.4,
    )

    # Left panel: RGB spatial map 
    ax_rgb = fig.add_subplot(gs[:, :2])   # spans all rows, first 2 columns
    ax_rgb.imshow(rgb_image, origin='upper')
    ax_rgb.set_title("UMAP RGB Map", fontsize=11, fontweight='bold', pad=8)
    ax_rgb.axis('off')

    # Add a small label explaining the colour encoding
    ax_rgb.text(
        0.5, -0.02,
        "Colour = position in UMAP space\nSame colour = similar elemental profile",
        transform=ax_rgb.transAxes,
        ha='center', va='top', fontsize=7, color='grey',
    )

    # Right panel: per-element thumbnails 
    for idx, ch in enumerate(channels):
        row = idx // thumb_cols
        col = idx % thumb_cols + 2   # offset by 2 for the RGB map columns

        ax = fig.add_subplot(gs[row, col])

        # Build a blank (white background) image for this element
        img_flat = np.ones(height * width, dtype=np.float32)

        # Percentile stretch the element intensities for display
        vals = df[ch].values.astype(np.float32)
        vmin = np.percentile(vals, 2)
        vmax = np.percentile(vals, 98)
        if vmax > vmin:
            stretched = np.clip((vals - vmin) / (vmax - vmin), 0, 1)
        else:
            stretched = np.zeros_like(vals)

        img_flat[tissue_indices_final] = stretched
        ax.imshow(img_flat.reshape(height, width), cmap='inferno',
                  vmin=0, vmax=1, origin='upper')
        ax.set_title(ch, fontsize=8, fontweight='bold')
        ax.axis('off')

    # Hide any unused thumbnail panels
    for idx in range(n, thumb_rows * thumb_cols):
        row = idx // thumb_cols
        col = idx % thumb_cols + 2
        fig.add_subplot(gs[row, col]).axis('off')

    fig.suptitle(
        "Element Key: RGB map vs per-element spatial distribution",
        fontsize=12, fontweight='bold', y=1.01,
    )

    if output_dir:
        plt.savefig(
            os.path.join(output_dir, "umap_element_key.png"),
            dpi=config.FIGURE_DPI, bbox_inches='tight',
        )
        print(f"  Saved → {os.path.join(output_dir, 'umap_element_key.png')}")
    if show_plot:
        plt.show()
    plt.close()

# =============================================================================
# 10. Density and hexbin plot 
# =============================================================================

def plot_umap_density(X_umap: np.ndarray,
                      output_dir: str = None, show_plot: bool = False):
    """
    Two-panel density visualisation of the UMAP embedding (UMAP dim 1 vs dim 2).

    LEFT  — Hexbin 2D histogram: tiles the embedding with hexagonal bins and
             colours each bin by how many pixels fall inside it.  For large
             datasets (100 k+ pixels) individual scatter points overlap and
             give a false impression of uniformity, hexbin fixes that.

    RIGHT — KDE contour overlay: a Gaussian kernel density estimate plotted as
             filled contours over a faint subsampled scatter.  The contour lines
             reveal the topology of the embedding, dense cluster cores appear
             as bright peaks, sparse 'bridges' between clusters are visible as
             saddle-points in the density surface.

    Parameters:
        X_umap     : UMAP coordinates, shape (n_pixels, 3)
        output_dir : if provided, saves as umap_density.png
        show_plot  : if True, displays the figure
    """
    from scipy.stats import gaussian_kde

    x = X_umap[:, 0]
    y = X_umap[:, 1]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor='white')

    # Left: hexbin 
    ax = axes[0]
    hb = ax.hexbin(x, y, gridsize=60, cmap='inferno', mincnt=1, linewidths=0.2)
    cb = plt.colorbar(hb, ax=ax, shrink=0.8)
    cb.set_label('Pixel count', fontsize=10)
    ax.set_xlabel('UMAP 1', fontsize=11)
    ax.set_ylabel('UMAP 2', fontsize=11)
    ax.set_title('Hexbin density', fontsize=13, fontweight='bold')
    ax.set_facecolor('#0d0d0d')

    # Right: KDE contour
    ax2 = axes[1]

    rng = np.random.default_rng(42)

    # Background scatter (faint, subsampled)
    n_scatter = min(40_000, len(x))
    idx_sc = rng.choice(len(x), n_scatter, replace=False)
    ax2.scatter(x[idx_sc], y[idx_sc], s=0.3, c='#aaaaaa', alpha=0.3, rasterized=True)

    # KDE subsample (gaussian_kde scales quadratically)
    n_kde = min(15_000, len(x))
    idx_kde = rng.choice(len(x), n_kde, replace=False)
    xk, yk = x[idx_kde], y[idx_kde]

    kde = gaussian_kde(np.vstack([xk, yk]), bw_method='scott')
    pad_x = (x.max() - x.min()) * 0.05
    pad_y = (y.max() - y.min()) * 0.05
    gx = np.linspace(x.min() - pad_x, x.max() + pad_x, 200)
    gy = np.linspace(y.min() - pad_y, y.max() + pad_y, 200)
    GX, GY = np.meshgrid(gx, gy)
    Z = kde(np.vstack([GX.ravel(), GY.ravel()])).reshape(GX.shape)

    ax2.contourf(GX, GY, Z, levels=12, cmap='plasma', alpha=0.55)
    ax2.contour(GX, GY, Z, levels=12, colors='white', linewidths=0.5, alpha=0.6)
    ax2.set_xlabel('UMAP 1', fontsize=11)
    ax2.set_ylabel('UMAP 2', fontsize=11)
    ax2.set_title('KDE density contour', fontsize=13, fontweight='bold')
    ax2.set_facecolor('#111111')

    plt.suptitle('UMAP Embedding Density\n'
                 'Left: hexbin pixel count   Right: Gaussian KDE contours',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()

    if output_dir:
        path = os.path.join(output_dir, 'umap_density.png')
        plt.savefig(path, dpi=config.FIGURE_DPI, bbox_inches='tight')
        print(f"  Saved UMAP density → {path}")
    if show_plot:
        plt.show()
    plt.close()


# =============================================================================
# 11. Element ranking by UMAP correlation
# =============================================================================

def plot_umap_element_ranking(X_umap: np.ndarray, df: pd.DataFrame,
                               channel_names_filtered: list,
                               output_dir: str = None, show_plot: bool = False):
    """
    Ranks elements by how strongly they drive the UMAP embedding using
    Spearman rank correlation between each element and each UMAP dimension.

    For each element, the maximum |rho| across all three UMAP dims is used as
    an 'importance score'.  Bars are coloured by whichever UMAP dimension that
    element dominates (red = UMAP1, green = UMAP2, blue = UMAP3).

    A side panel shows the full 3-dim rho profile as a static heatmap, and
    the interactive version (umap_element_correlation.html) is also saved.

    Subsamples to 50,000 pixels for speed.

    Parameters:
        X_umap                 : UMAP coordinates (n_pixels, 3)
        df                     : log1p-normalised pixel dataframe
        channel_names_filtered : list of element name strings
        output_dir             : if provided, saves umap_element_ranking.png
                                 and umap_element_correlation.html
        show_plot              : if True, displays the figure
    """
    from scipy.stats import spearmanr

    n_ch        = len(channel_names_filtered)
    dim_labs    = ['UMAP1 (R)', 'UMAP2 (G)', 'UMAP3 (B)']
    dim_colours = ['#e74c3c', '#27ae60', '#2980b9']

    # Subsample for speed
    MAX_PIX = 50_000
    n_pix   = len(df)
    if n_pix > MAX_PIX:
        rng    = np.random.default_rng(42)
        idx    = rng.choice(n_pix, MAX_PIX, replace=False)
        X_sub  = X_umap[idx]
        df_sub = df.iloc[idx]
    else:
        X_sub  = X_umap
        df_sub = df

    print(f"  Computing Spearman rho for element ranking on {len(df_sub):,} pixels...")
    rho_matrix = np.zeros((n_ch, 3))
    for i, ch in enumerate(channel_names_filtered):
        for j in range(3):
            rho, _ = spearmanr(df_sub[ch].values, X_sub[:, j])
            rho_matrix[i, j] = rho

    max_abs_rho = np.abs(rho_matrix).max(axis=1)
    dom_dim     = np.abs(rho_matrix).argmax(axis=1)

    # Sort ascending so highest importance is at the top of horizontal bar
    order        = np.argsort(max_abs_rho)
    rho_sorted   = rho_matrix[order]
    names_sorted = [channel_names_filtered[i] for i in order]
    dom_sorted   = dom_dim[order]
    max_sorted   = max_abs_rho[order]
    bar_colours  = [dim_colours[d] for d in dom_sorted]

    fig, axes = plt.subplots(1, 2, figsize=(14, max(6, n_ch * 0.33)),
                              facecolor='white')

    # Left: ranked bar chart
    ax = axes[0]
    ax.barh(range(n_ch), max_sorted, color=bar_colours,
            edgecolor='white', linewidth=0.4, height=0.75)
    ax.set_yticks(range(n_ch))
    ax.set_yticklabels(names_sorted, fontsize=8)
    ax.set_xlabel('Max |Spearman rho| across UMAP dims', fontsize=10)
    ax.set_title('Element importance for UMAP embedding', fontsize=12, fontweight='bold')
    ax.set_xlim(0, 1)
    ax.axvline(0.3, color='grey', lw=0.8, ls='--', alpha=0.6)
    ax.axvline(0.6, color='grey', lw=0.8, ls=':', alpha=0.6)
    ax.text(0.31, n_ch * 0.02, 'rho=0.3', fontsize=7, color='grey', va='bottom')
    ax.text(0.61, n_ch * 0.02, 'rho=0.6', fontsize=7, color='grey', va='bottom')
    ax.set_facecolor('#f9f9f9')

    from matplotlib.patches import Patch
    legend_handles = [Patch(color=c, label=l)
                      for c, l in zip(dim_colours, dim_labs)]
    ax.legend(handles=legend_handles, title='Dominant UMAP dim',
              fontsize=8, loc='lower right', frameon=True)

    # Right: full rho heatmap (static) 
    ax2 = axes[1]
    im = ax2.imshow(rho_sorted, aspect='auto', cmap='RdBu_r',
                    vmin=-1, vmax=1, interpolation='nearest')
    ax2.set_xticks([0, 1, 2])
    ax2.set_xticklabels(dim_labs, fontsize=9)
    ax2.set_yticks(range(n_ch))
    ax2.set_yticklabels(names_sorted, fontsize=8)
    ax2.set_title('Spearman rho per UMAP dim', fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=ax2, shrink=0.6, label='Spearman rho')

    plt.suptitle('Element–UMAP Correlation Analysis\n'
                 'Bars sorted by max |rho|; colour = dominant UMAP dimension',
                 fontsize=13, y=1.01)
    plt.tight_layout()

    if output_dir:
        path = os.path.join(output_dir, 'umap_element_ranking.png')
        plt.savefig(path, dpi=config.FIGURE_DPI, bbox_inches='tight')
        print(f"  Saved element ranking → {path}")
    if show_plot:
        plt.show()
    plt.close()

    # Also generate the interactive heatmap HTML
    plot_umap_element_correlation(X_umap, df, channel_names_filtered,
                                  output_dir=output_dir, show_plot=False)


# =============================================================================
# 12. Moran's I spatial autocorrelation
# =============================================================================

def _morans_i_queen(values: np.ndarray, mask: np.ndarray) -> float:
    """
    Moran's I on a 2D raster using queen (8-connected) contiguity weights.

    Uses scipy.ndimage.convolve to vectorise the neighbourhood summation 
    no loops over pixels so it runs in seconds even for megapixel images.

    Parameters:
        values : 2D float array (H, W); non-tissue pixels are ignored
        mask   : boolean (H, W), True = tissue pixel

    Returns:
        Moran's I (float); E[I] = -1/(N-1) under spatial randomness
    """
    from scipy.ndimage import convolve

    queen   = np.array([[1, 1, 1],
                        [1, 0, 1],
                        [1, 1, 1]], dtype=np.float64)

    mask_f  = mask.astype(np.float64)
    n_neigh = convolve(mask_f, queen, mode='constant', cval=0.0)

    z_filled = np.where(mask, values, 0.0)
    z_mean   = z_filled[mask].mean()
    z_dev    = np.where(mask, values - z_mean, 0.0)

    W        = n_neigh[mask].sum()
    N        = int(mask.sum())

    neigh_dev_sum = convolve(z_dev, queen, mode='constant', cval=0.0)
    numerator     = (z_dev[mask] * neigh_dev_sum[mask]).sum()
    denominator   = (z_dev[mask] ** 2).sum()

    if W == 0 or denominator == 0:
        return 0.0
    return float((N / W) * (numerator / denominator))


def plot_umap_morans_i(X_umap: np.ndarray,
                       tissue_indices_final: np.ndarray,
                       height: int, width: int,
                       output_dir: str = None, show_plot: bool = False):
    """
    Computes and visualises Moran's I for each of the three UMAP dimensions
    when the UMAP values are mapped back to their tissue pixel positions.

    Interpretation:
        I >> 0  → strong positive spatial autocorrelation: nearby tissue pixels
                   have similar UMAP values, meaning the embedding captures real
                   tissue organisation (clusters = biological compartments).
        I ≈  0  → spatial randomness: the UMAP values are scattered with no
                   spatial pattern the embedding is not capturing tissue structure.
        I << 0  → unlikely in practice for continuous tissue sections.

    High Moran's I for all three dims is the key validation that UMAP-based
    dimensionality reduction is biologically meaningful for this MSI dataset.

    Reference:
        Alexandrov et al., Anal. Chem. 2019, spatial autocorrelation as a
        quality metric for evaluating MSI dimensionality reduction.

    Parameters:
        X_umap               : UMAP coordinates (n_pixels, 3)
        tissue_indices_final : flat pixel indices of tissue pixels
        height, width        : image spatial dimensions
        output_dir           : if provided, saves umap_morans_i.png
        show_plot            : if True, displays the figure
    """
    from matplotlib.gridspec import GridSpecFromSubplotSpec

    print("  Computing Moran's I for UMAP dimensions (using queen contiguity)...")

    dim_labels  = ["UMAP 1\n(Red)", "UMAP 2\n(Green)", "UMAP 3\n(Blue)"]
    dim_colours = ['#e74c3c', '#27ae60', '#2980b9']
    cmaps_list  = ['Reds', 'Greens', 'Blues']

    ys, xs = np.unravel_index(tissue_indices_final, (height, width))
    mask   = np.zeros((height, width), dtype=bool)
    mask[ys, xs] = True

    morans = []
    grids  = []
    for dim in range(3):
        grid = np.zeros((height, width), dtype=np.float64)
        grid[ys, xs] = X_umap[:, dim]
        grids.append(grid)
        I = _morans_i_queen(grid, mask)
        morans.append(I)
        print(f"    UMAP{dim + 1}: Moran's I = {I:.4f}")

    # Figure 
    fig = plt.figure(figsize=(14, 5), facecolor='white')
    gs  = fig.add_gridspec(1, 2, width_ratios=[1, 2], wspace=0.3)

    # Left: bar chart
    ax_bar = fig.add_subplot(gs[0])
    ax_bar.bar(range(3), morans, color=dim_colours,
               edgecolor='white', linewidth=0.5, width=0.55)
    ax_bar.set_xticks(range(3))
    ax_bar.set_xticklabels(dim_labels, fontsize=10)
    ax_bar.set_ylabel("Moran's I", fontsize=11)
    ymax = max(morans) * 1.18 if max(morans) > 0 else 0.1
    ymin = min(min(morans) - 0.05, -0.05)
    ax_bar.set_ylim(ymin, ymax)
    ax_bar.axhline(0, color='black', lw=0.8)
    expected = -1.0 / (mask.sum() - 1)
    ax_bar.axhline(expected, color='red', lw=0.8, ls='--', alpha=0.6)
    ax_bar.text(2.5, expected + 0.005, 'E[I] under\nrandomness',
                ha='right', va='bottom', fontsize=7, color='red', alpha=0.8)
    ax_bar.set_title("Moran's I per UMAP Dimension", fontsize=12, fontweight='bold')
    ax_bar.set_facecolor('#f9f9f9')
    for i, v in enumerate(morans):
        ax_bar.text(i, v + ymax * 0.02, f'{v:.3f}',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Right: 3 spatial maps
    sub_gs = GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[1], wspace=0.05)
    grid_nan = np.where(mask, 0.0, np.nan)
    for dim in range(3):
        ax_map = fig.add_subplot(sub_gs[dim])
        disp   = np.where(mask, grids[dim], np.nan)
        ax_map.imshow(disp, cmap=cmaps_list[dim], origin='upper',
                      interpolation='nearest')
        ax_map.set_title(f'UMAP {dim + 1}\nI = {morans[dim]:.3f}',
                         fontsize=9, fontweight='bold', color=dim_colours[dim])
        ax_map.axis('off')

    plt.suptitle(
        "Spatial Autocorrelation (Moran's I) of UMAP Dimensions\n"
        "I → 1: UMAP captures coherent tissue structure   I → 0: random spatial arrangement",
        fontsize=12, fontweight='bold', y=1.02
    )

    if output_dir:
        path = os.path.join(output_dir, 'umap_morans_i.png')
        plt.savefig(path, dpi=config.FIGURE_DPI, bbox_inches='tight')
        print(f"  Saved Moran's I → {path}")
    if show_plot:
        plt.show()
    plt.close()




# =============================================================================
# 13. KDE Cluster contours on UMAP space
# =============================================================================

def plot_umap_kde_cluster(X_umap: np.ndarray,
                           km_labels: np.ndarray,
                           hdb_labels: np.ndarray,
                           km_log_fc=None,
                           hdb_log_fc=None,
                           channel_names=None,
                           km_cluster_ids=None,
                           hdb_cluster_ids=None,
                           output_dir: str = None,
                           show_plot: bool = False):
    """
    KDE contour cluster visualisation on UMAP space (dim1 vs dim2).

    Two panels,  K-means and HDBSCAN. Each cluster is shown as a subsampled
    scatter with KDE density contours (more appropriate than ellipses for the
    non-linear UMAP manifold). All positively enriched channels (log2FC > 0)
    are listed beside each cluster centroid in descending order.

    Parameters:
        X_umap         : UMAP coordinates (n_pixels, 3)
        km_labels      : K-means label array
        hdb_labels     : HDBSCAN label array (-1 = noise)
        km_log_fc      : (n_clusters, n_channels) log2FC array for K-means
        hdb_log_fc     : same for HDBSCAN
        channel_names  : list of channel name strings
        km_cluster_ids : list of K-means cluster IDs (aligned with km_log_fc rows)
        hdb_cluster_ids: list of HDBSCAN cluster IDs
        output_dir     : if provided, saves umap_kde_cluster.png
        show_plot      : if True, displays the figure
    """
    from scipy.stats import gaussian_kde as _gaussian_kde
    import matplotlib.colors as _mcolors
    import matplotlib.gridspec as _gridspec

    rng   = np.random.default_rng(42)
    n_sub = min(50_000, len(X_umap))
    sub   = rng.choice(len(X_umap), n_sub, replace=False)
    ux    = X_umap[sub, 0]
    uy    = X_umap[sub, 1]

    # Shared grid for KDE evaluation
    x_min, x_max = ux.min() - 0.2, ux.max() + 0.2
    y_min, y_max = uy.min() - 0.2, uy.max() + 0.2
    grid_pts     = 150
    gx           = np.linspace(x_min, x_max, grid_pts)
    gy           = np.linspace(y_min, y_max, grid_pts)
    XX, YY       = np.meshgrid(gx, gy)
    grid_flat    = np.vstack([XX.ravel(), YY.ravel()])

    # Collect per-method, per-cluster enriched channel data for the legend table
    # Structure: {method_title: [(cid, col, [(ch, fc), ...]), ...]}
    enriched_data = {}

    titles        = ['K-Means', 'HDBSCAN']
    method_log_fc = [km_log_fc,      hdb_log_fc]
    method_cids   = [km_cluster_ids, hdb_cluster_ids]

    # figure layout: 2 scatter rows + 1 table row 
    # Table height scales with max number of enriched channels across all clusters
    # Table height scales with channel count (58 channels needs more space)
    fig = plt.figure(figsize=(22, 18), facecolor='white')
    gs  = _gridspec.GridSpec(2, 2, figure=fig,
                              height_ratios=[5, 6], hspace=0.3, wspace=0.12)
    scatter_axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
    table_axes   = [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]

    for ax, labels, title, log_fc_arr, cid_order in zip(
            scatter_axes, [km_labels, hdb_labels], titles,
            method_log_fc, method_cids):

        lsub       = labels[sub]
        unique_cls = sorted(c for c in np.unique(lsub) if c != -1)
        n_clusters = len(unique_cls)
        palette    = config.get_cluster_colours(n_clusters)
        enriched_data[title] = []

        # Grey noise scatter (HDBSCAN -1)
        noise_mask = lsub == -1
        if noise_mask.any():
            ax.scatter(ux[noise_mask], uy[noise_mask],
                       s=1, alpha=0.10, color='#cccccc', rasterized=True, zorder=1)

        for i, cid in enumerate(unique_cls):
            mask   = lsub == cid
            col    = palette[i % len(palette)]
            cx, cy = ux[mask], uy[mask]

            # Light scatter behind contours
            ax.scatter(cx, cy, s=1, alpha=0.15, color=col,
                       rasterized=True, zorder=2)

            # KDE contours, requires ≥ 20 points
            if mask.sum() >= 20:
                try:
                    kde   = _gaussian_kde(np.vstack([cx, cy]), bw_method='scott')
                    Z     = kde(grid_flat).reshape(XX.shape)
                    Z_max = Z.max()
                    if Z_max > 0:
                        Z /= Z_max
                    levels = [0.20, 0.50, 0.80]
                    rgba   = _mcolors.to_rgba(col)
                    ax.contourf(XX, YY, Z, levels=[levels[0], 1.0],
                                colors=[(*rgba[:3], 0.08)], zorder=3)
                    ax.contourf(XX, YY, Z, levels=[levels[1], 1.0],
                                colors=[(*rgba[:3], 0.15)], zorder=4)
                    ax.contourf(XX, YY, Z, levels=[levels[2], 1.0],
                                colors=[(*rgba[:3], 0.22)], zorder=5)
                    ax.contour(XX, YY, Z, levels=levels,
                               colors=[col], linewidths=[0.8, 1.2, 1.6],
                               alpha=0.85, zorder=6)
                except Exception:
                    pass

            # Cluster ID badge at centroid (no inline text annotations)
            mx, my = np.mean(cx), np.mean(cy)
            ax.text(mx, my, str(cid), ha='center', va='center',
                    fontsize=10, fontweight='bold', color='white',
                    bbox=dict(boxstyle='round,pad=0.25', fc=col, ec='none', alpha=0.92),
                    zorder=8)

            # Collect ALL channels for legend table (sorted by log2FC descending)
            if log_fc_arr is not None and channel_names is not None and cid_order is not None:
                if cid in cid_order:
                    row_idx    = list(cid_order).index(cid)
                    fc_row     = log_fc_arr[row_idx]
                    sorted_idx = np.argsort(fc_row)[::-1]
                    all_channels = [(channel_names[j], fc_row[j]) for j in sorted_idx]
                    enriched_data[title].append((cid, col, all_channels))

        ax.set_xlabel('UMAP 1', fontsize=11)
        ax.set_ylabel('UMAP 2', fontsize=11)
        ax.set_title(f'{title} Clusters: UMAP Space (KDE contours)\n'
                     f'{n_clusters} clusters | {n_sub:,} pixels | '
                     f'contours = 20/50/80 % density',
                     fontsize=11, fontweight='bold')
        ax.set_facecolor('#f8f8f8')
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

    # Enriched channel legend tables (one per method) 
    for tax, title in zip(table_axes, titles):
        tax.axis('off')
        clusters_info = enriched_data.get(title, [])
        if not clusters_info:
            tax.text(0.5, 0.5, 'No enrichment data available',
                     ha='center', va='center', fontsize=9, color='grey',
                     transform=tax.transAxes)
            continue

        n_cols   = len(clusters_info)
        col_w    = 1.0 / n_cols
        tax.set_title(f'{title} All Channels per Cluster (log₂FC, sorted descending)',
                      fontsize=10, fontweight='bold', pad=4)

        for col_i, (cid, col, channels) in enumerate(clusters_info):
            x_left = col_i * col_w
            # Cluster header badge
            tax.text(x_left + col_w * 0.5, 0.98,
                     f'  Cluster {cid}  ',
                     ha='center', va='top', fontsize=9, fontweight='bold',
                     color='white', transform=tax.transAxes,
                     bbox=dict(boxstyle='round,pad=0.3', fc=col, ec='none', alpha=0.9))
            # Channel rows, all channels, line height computed from count
            n_ch     = len(channels)
            line_h   = min(0.055, 0.92 / max(n_ch, 1))
            y_cursor = 0.92 - line_h
            rgba_col = _mcolors.to_rgba(col)
            for ch, fc in channels:
                # Positive FC = enriched (tinted), negative = depleted (plain)
                txt_color = '#1a6e1a' if fc > 0 else '#8b0000' if fc < -0.5 else '#555555'
                bg_alpha  = 0.10 if fc > 0 else 0.04
                tax.text(x_left + col_w * 0.05, y_cursor,
                         f'{ch}  {fc:+.2f}',
                         ha='left', va='top', fontsize=6.5,
                         color=txt_color, transform=tax.transAxes,
                         bbox=dict(boxstyle='square,pad=0.1',
                                   fc=(*rgba_col[:3], bg_alpha),
                                   ec='none'))
                y_cursor -= line_h

        # Subtle dividers between clusters
        for col_i in range(1, n_cols):
            tax.plot([col_i * col_w, col_i * col_w], [0, 1],
                     color='#dddddd', linewidth=0.8,
                     transform=tax.transAxes, clip_on=False)

    plt.suptitle('UMAP KDE Cluster Contour + Enriched Channels',
                 fontsize=13, fontweight='bold', y=1.01)

    if output_dir:
        path = os.path.join(output_dir, 'umap_kde_cluster.png')
        fig.savefig(path, dpi=config.FIGURE_DPI, bbox_inches='tight')
        print(f'  Saved → {path}')
        # Also save under the per-method names the app expects
        for _alias in ('umap_kde_cluster_kmeans.png', 'umap_kde_cluster_hdbscan.png'):
            fig.savefig(os.path.join(output_dir, _alias),
                        dpi=config.FIGURE_DPI, bbox_inches='tight')
    if show_plot:
        plt.show()
    plt.close()



# =============================================================================
# 14. Cluster-coloured density map in UMAP space
# =============================================================================

def plot_umap_cluster_density(X_umap: np.ndarray,
                               km_labels: np.ndarray,
                               hdb_labels: np.ndarray,
                               output_dir: str = None,
                               show_plot: bool = False):
    """
    Cluster-coloured density map in UMAP space.

    Two rows (K-Means / HDBSCAN), two columns each:
      Left  — hexbin grid where each cell is coloured by its dominant cluster
      Right — per-cluster KDE density overlaid as filled contours (no scatter)

    This combines the density information from the embedding density plot with
    the cluster identity from the cluster overlay, giving a cleaner view of
    how tightly each cluster is packed and where it sits in UMAP space.

    Parameters:
        X_umap     : UMAP coordinates (n_pixels, ≥ 2)
        km_labels  : K-means label array
        hdb_labels : HDBSCAN label array (-1 = noise)
        output_dir : if provided, saves umap_cluster_density.png
        show_plot  : if True, displays the figure
    """
    from scipy.stats import gaussian_kde as _gaussian_kde
    import matplotlib.colors as _mcolors
    from matplotlib.patches import Patch as _Patch

    rng   = np.random.default_rng(42)
    n_sub = min(100_000, len(X_umap))
    sub   = rng.choice(len(X_umap), n_sub, replace=False)
    ux    = X_umap[sub, 0]
    uy    = X_umap[sub, 1]

    x_min, x_max = ux.min() - 0.2, ux.max() + 0.2
    y_min, y_max = uy.min() - 0.2, uy.max() + 0.2

    # KDE grid
    grid_pts  = 150
    gx        = np.linspace(x_min, x_max, grid_pts)
    gy        = np.linspace(y_min, y_max, grid_pts)
    XX, YY    = np.meshgrid(gx, gy)
    grid_flat = np.vstack([XX.ravel(), YY.ravel()])

    fig, axes = plt.subplots(2, 2, figsize=(22, 14), facecolor='white')
    row_titles = ['K-Means', 'HDBSCAN']

    for row, (labels_full, method_name) in enumerate(
            zip([km_labels, hdb_labels], row_titles)):

        labels_sub = labels_full[sub]
        unique_cls = sorted(c for c in np.unique(labels_sub) if c != -1)
        n_clusters = len(unique_cls)
        palette    = config.get_cluster_colours(n_clusters)

        # Left: dominant-cluster hexbin
        ax_hex = axes[row, 0]

        # Build a 2D grid; for each cell record vote counts per cluster
        bins      = 80
        xedges    = np.linspace(x_min, x_max, bins + 1)
        yedges    = np.linspace(y_min, y_max, bins + 1)
        dominant  = np.full((bins, bins), -1, dtype=int)
        max_count = np.zeros((bins, bins), dtype=int)

        for ci, cid in enumerate(unique_cls):
            mask = labels_sub == cid
            if not mask.any():
                continue
            H, _, _ = np.histogram2d(ux[mask], uy[mask],
                                     bins=[xedges, yedges])
            H = H.astype(int)
            update = H > max_count
            dominant[update] = ci
            max_count[update] = H[update]

        # Build RGBA image from dominant-cluster grid
        # Transpose because histogram2d is (x_bins, y_bins) but imshow needs (y, x)
        rgb_grid = np.ones((bins, bins, 4))   # default white, fully transparent
        for ci, col in enumerate(palette[:n_clusters]):
            rgba       = np.array(_mcolors.to_rgba(col))
            cell_mask  = dominant == ci
            # Alpha proportional to count density (more pixels = more opaque)
            alpha_vals = np.clip(max_count / (max_count.max() + 1e-9), 0, 1)
            rgb_grid[cell_mask, :3] = rgba[:3]
            rgb_grid[cell_mask,  3] = 0.4 + 0.6 * alpha_vals[cell_mask]

        # imshow: origin='lower', extent matches xedges/yedges
        ax_hex.imshow(
            rgb_grid.transpose(1, 0, 2),   # swap x/y for imshow row=y col=x
            origin='lower',
            extent=[x_min, x_max, y_min, y_max],
            aspect='auto',
            interpolation='nearest',
            zorder=2,
        )
        ax_hex.set_facecolor('#1a1a1a')

        # Legend patches
        handles = [_Patch(facecolor=palette[i], label=f'C{cid}')
                   for i, cid in enumerate(unique_cls)]
        ax_hex.legend(handles=handles, loc='upper left',
                      fontsize=8, framealpha=0.7, markerscale=1.2)
        ax_hex.set_xlabel('UMAP 1', fontsize=11)
        ax_hex.set_ylabel('UMAP 2', fontsize=11)
        ax_hex.set_title(f'{method_name} Dominant Cluster Hexbin\n'
                         f'Each cell coloured by majority-vote cluster | '
                         f'Opacity ∝ pixel count',
                         fontsize=11, fontweight='bold')
        ax_hex.set_xlim(x_min, x_max)
        ax_hex.set_ylim(y_min, y_max)

        # Right: per-cluster KDE density (filled contours, no scatter)
        ax_kde = axes[row, 1]
        ax_kde.set_facecolor('#f8f8f8')

        for ci, cid in enumerate(unique_cls):
            mask   = labels_sub == cid
            col    = palette[ci % len(palette)]
            cx, cy = ux[mask], uy[mask]

            if mask.sum() < 20:
                continue
            try:
                kde   = _gaussian_kde(np.vstack([cx, cy]), bw_method='scott')
                Z     = kde(grid_flat).reshape(XX.shape)
                Z_max = Z.max()
                if Z_max > 0:
                    Z /= Z_max
                rgba   = _mcolors.to_rgba(col)
                levels = [0.10, 0.30, 0.55, 0.80]
                alphas = [0.10, 0.18, 0.26, 0.35]
                for lvl, alp in zip(levels, alphas):
                    ax_kde.contourf(XX, YY, Z, levels=[lvl, 1.0],
                                    colors=[(*rgba[:3], alp)], zorder=2)
                ax_kde.contour(XX, YY, Z, levels=levels,
                               colors=[col], linewidths=0.9, alpha=0.9, zorder=3)
                # Cluster label at density peak
                peak_flat = np.argmax(Z)
                peak_y, peak_x = np.unravel_index(peak_flat, Z.shape)
                ax_kde.text(gx[peak_x], gy[peak_y], str(cid),
                            ha='center', va='center', fontsize=10,
                            fontweight='bold', color='white',
                            bbox=dict(boxstyle='round,pad=0.25', fc=col,
                                      ec='none', alpha=0.92),
                            zorder=5)
            except Exception:
                pass

        ax_kde.set_xlabel('UMAP 1', fontsize=11)
        ax_kde.set_ylabel('UMAP 2', fontsize=11)
        ax_kde.set_title(f'{method_name} Per-Cluster KDE Density\n'
                         f'Filled contours at 10/30/55/80 % density | '
                         f'No scatter: density only',
                         fontsize=11, fontweight='bold')
        ax_kde.set_xlim(x_min, x_max)
        ax_kde.set_ylim(y_min, y_max)

    plt.suptitle('Cluster-Coloured Density Maps: UMAP Space',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()

    if output_dir:
        path = os.path.join(output_dir, 'umap_cluster_density.png')
        fig.savefig(path, dpi=config.FIGURE_DPI, bbox_inches='tight')
        print(f'  Saved → {path}')
    if show_plot:
        plt.show()
    plt.close()


# =============================================================================
# 15. Element cluster assignment on UMAP (interactive HTML)
# =============================================================================

def plot_umap_element_cluster_assignment(X_umap: np.ndarray,
                                          df_normalised,
                                          channel_names: list,
                                          km_labels: np.ndarray,
                                          output_dir: str = None,
                                          show_plot: bool = False):
    """
    Interactive HTML: pick any element from a dropdown and see the UMAP
    coloured by that element's log1p intensity, along with its dominant
    cluster assignment (the cluster where log2FC is highest).

    All elements are shown, no threshold is applied to the assignment.
    Assignment rule:
        log2FC[k, j] = (mean_log1p[k, j] - tissue_mean_log1p[j]) / log(2)
        Element j → cluster k* = argmax_k log2FC[k, j]

    Saves as an interactive HTML file.

    Parameters:
        X_umap        : UMAP coordinates (n_pixels, ≥ 2)
        df_normalised : DataFrame of log1p-normalised intensities
        channel_names : list of channel name strings
        km_labels     : K-means label array (used for cluster assignment)
        output_dir    : if provided, saves umap_element_cluster_assignment.html
        show_plot     : if True, calls fig.show()
    """
    import plotly.graph_objects as _go

    # Subsample for rendering speed
    MAX_PIX = 100_000
    rng     = np.random.default_rng(42)
    n_pix   = len(X_umap)
    if n_pix > MAX_PIX:
        idx        = rng.choice(n_pix, MAX_PIX, replace=False)
        X_sub      = X_umap[idx]
        labels_sub = km_labels[idx]
        vals_sub   = df_normalised[channel_names].values[idx].astype(np.float32)
    else:
        X_sub      = X_umap
        labels_sub = km_labels
        vals_sub   = df_normalised[channel_names].values.astype(np.float32)

    # log2FC per cluster per element (full dataset) 
    log1p_full  = df_normalised[channel_names].values.astype(np.float64)
    unique_cls  = sorted(c for c in np.unique(km_labels) if c != -1)
    n_clusters  = len(unique_cls)
    valid_mask  = km_labels >= 0
    tissue_mean = log1p_full[valid_mask].mean(axis=0)
    cluster_means = np.zeros((n_clusters, len(channel_names)))
    for i, cid in enumerate(unique_cls):
        cluster_means[i] = log1p_full[km_labels == cid].mean(axis=0)
    log2fc = (cluster_means - tissue_mean[np.newaxis, :]) / np.log(2)

    # Assign each element to its best cluster (no threshold)
    best_cluster_idx = np.argmax(log2fc, axis=0)   # (n_channels,)
    element_best_cid = [unique_cls[i] for i in best_cluster_idx]
    element_best_fc  = [log2fc[i, j] for j, i in enumerate(best_cluster_idx)]

    palette = config.get_cluster_colours(n_clusters)

    # Build Plotly figure with dropdown 
    init_vals = vals_sub[:, 0].tolist()
    init_cid  = element_best_cid[0]
    init_fc   = element_best_fc[0]

    def _make_title(ch, cid, fc):
        return (f'<b>{ch}</b> Assigned to Cluster {cid}  '
                f'(log₂FC = {fc:+.2f})<br>'
                f'<sup>Colour = log1p intensity  |  '
                f'Pick an element from the dropdown</sup>')

    fig = _go.Figure()
    fig.add_trace(_go.Scattergl(
        x=X_sub[:, 0].tolist(),
        y=X_sub[:, 1].tolist(),
        mode='markers',
        marker=dict(
            size=2,
            color=init_vals,
            colorscale='Hot',
            showscale=True,
            colorbar=dict(title='log1p intensity', thickness=14, len=0.75),
            opacity=0.85,
        ),
        hovertemplate='UMAP1: %{x:.2f}<br>UMAP2: %{y:.2f}<extra></extra>',
        name='',
    ))

    # Dropdown buttons, one per element
    buttons = []
    for j, ch in enumerate(channel_names):
        cid = element_best_cid[j]
        fc  = element_best_fc[j]
        buttons.append(dict(
            label=ch,
            method='update',
            args=[
                {'marker': {
                    'size': 2,
                    'color': [vals_sub[:, j].tolist()],
                    'colorscale': 'Hot',
                    'showscale': True,
                    'colorbar': {'title': 'log1p intensity', 'thickness': 14, 'len': 0.75},
                    'opacity': 0.85,
                }},
                {'title': {'text': _make_title(ch, cid, fc), 'font': {'size': 15}}},
            ],
        ))

    fig.update_layout(
        updatemenus=[dict(
            buttons=buttons,
            direction='down',
            showactive=True,
            x=0.0,
            xanchor='left',
            y=1.18,
            yanchor='top',
            bgcolor='white',
            bordercolor='#cccccc',
            font=dict(size=12),
            pad=dict(r=10, t=10),
        )],
        title=dict(
            text=_make_title(channel_names[0], init_cid, init_fc),
            font=dict(size=15),
        ),
        xaxis=dict(title='UMAP 1', showgrid=False, zeroline=False),
        yaxis=dict(title='UMAP 2', showgrid=False, zeroline=False, scaleanchor='x'),
        height=750,
        width=950,
        margin=dict(l=60, r=60, t=140, b=60),
        paper_bgcolor='white',
        plot_bgcolor='#f4f4f4',
    )

    if output_dir:
        path = os.path.join(output_dir, 'umap_element_cluster_assignment.html')
        fig.write_html(path, config={'responsive': True})
        print(f'  Saved interactive element cluster assignment → {path}')
    if show_plot:
        fig.show()


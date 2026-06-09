# =============================================================================
# umap_reduction.py
#
# PURPOSE:
#   Runs UMAP on the log1p-normalised pixel data and produces all UMAP
#   visualisations. This is the primary dimensionality reduction tool used
#   by the facility.
#
# UMAP (Uniform Manifold Approximation and Projection) reduces the high-
# dimensional elemental data (one dimension per element) down to 3 dimensions
# that preserve the local structure of the data. Pixels with similar elemental
# profiles end up close together in UMAP space.
#
# WHY 3 COMPONENTS?
#   3D UMAP allows the three axes to be mapped directly to RGB colour channels,
#   producing a spatial image where colour encodes elemental similarity.
#   Structurally similar pixels appear the same colour in the tissue image.
#
# ORDER OF STEPS:
#   1. Run UMAP (GPU via cuML if available, CPU fallback via umap-learn)
#   2. RGB spatial map — static matplotlib version
#   3. RGB spatial map — interactive Plotly version with per-pixel hover info
#   4. 3D scatter plot coloured by UMAP RGB (the "legend" for the spatial map)
#   5. 2D scatter per channel (auto log/linear from scale_suggestions)
#   6. All-channels grid — one 2D UMAP scatter per element
#   7. Dominant element map — each pixel coloured by its strongest element
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
# 1. RUN UMAP
# =============================================================================

def run_umap(df_normalised: pd.DataFrame):
    """
    Runs UMAP on the log1p-normalised pixel dataframe.

    Attempts to use GPU-accelerated cuML first (much faster on large datasets).
    Falls back to the standard CPU umap-learn library if cuML is not installed.

    Input is the log1p-normalised dataframe from preprocessing.apply_log1p().
    UMAP runs directly on the element intensities — no PCA step in between.

    Settings (from config.py):
        UMAP_N_COMPONENTS = 3    : 3D embedding for RGB spatial map
        UMAP_N_NEIGHBORS  = 30   : how many neighbouring pixels to consider
                                   when learning the local structure
        UMAP_MIN_DIST     = 0.0  : allows clusters to pack tightly together
        UMAP_METRIC       = cosine: measures similarity by angle rather than
                                   absolute distance — better for intensity data
        UMAP_RANDOM_STATE = 42   : ensures reproducible results

    Parameters:
        df_normalised : log1p-normalised pixel dataframe, shape (n_pixels, n_channels)

    Returns:
        X_umap : numpy array of UMAP coordinates, shape (n_pixels, 3)
    """
    # Try GPU (cuML) first — dramatically faster for large datasets
    if config.USE_GPU:
        try:
            from cuml.manifold import UMAP as cuUMAP
            print("cuML found — running UMAP on GPU.")
            reducer = cuUMAP(
                n_components=config.UMAP_N_COMPONENTS,
                n_neighbors=config.UMAP_N_NEIGHBORS,
                min_dist=config.UMAP_MIN_DIST,
                metric=config.UMAP_METRIC,
                random_state=config.UMAP_RANDOM_STATE,
                verbose=True,
            )
            X_umap = reducer.fit_transform(df_normalised.values.astype(np.float32))
            X_umap = np.array(X_umap)  # convert from cuDF to numpy if needed
        except ImportError:
            print("cuML not found — falling back to CPU umap-learn.")
            config.USE_GPU = False  # avoid retrying on subsequent calls

    if not config.USE_GPU:
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
    return X_umap


# =============================================================================
# 2. RGB SPATIAL MAP (static)
# =============================================================================

def plot_umap_rgb(X_umap: np.ndarray, tissue_indices_final: np.ndarray,
                   height: int, width: int,
                   output_dir: str = None, show_plot: bool = True):
    """
    Creates a spatial RGB image where each tissue pixel's colour encodes its
    position in UMAP space:
        Red channel   = UMAP dimension 1 (normalised to 0–1)
        Green channel = UMAP dimension 2 (normalised to 0–1)
        Blue channel  = UMAP dimension 3 (normalised to 0–1)

    Pixels with similar elemental profiles appear the same colour.
    Background pixels (outside the tissue mask) are shown as white.

    The 3D scatter plot (plot_umap_3d_scatter) acts as the colour legend for
    this image — it shows what colour corresponds to what position in UMAP space.

    Parameters:
        X_umap               : UMAP coordinates, shape (n_pixels, 3)
        tissue_indices_final : flat pixel indices of tissue pixels
        height, width        : original image dimensions
        output_dir           : if provided, saves the figure here
        show_plot            : if True, displays the figure

    Returns:
        rgb_image  : (H, W, 3) float array of the RGB image
        umap_norm  : (n_pixels, 3) normalised UMAP coordinates (used by other plots)
    """
    # Percentile stretch each UMAP axis to [0, 1] for RGB mapping.
    # We clip to the 2nd–98th percentile before rescaling so that a small
    # number of outlier pixels don't drag the colour scale, leaving the
    # majority of tissue pixels crammed into a narrow colour range.
    # This gives a full rainbow spread across the tissue rather than one
    # dominant colour.
    umap_norm = np.zeros_like(X_umap, dtype=float)
    for c in range(3):
        col = X_umap[:, c]
        vmin = np.percentile(col, 2)
        vmax = np.percentile(col, 98)
        stretched = (col - vmin) / (vmax - vmin)
        umap_norm[:, c] = np.clip(stretched, 0, 1)

    # Build (H, W, 3) image — start with white background
    rgb_image = np.ones((height, width, 3), dtype=float)

    # Place tissue pixel colours at their correct (y, x) positions
    ys, xs = np.unravel_index(tissue_indices_final, (height, width))
    rgb_image[ys, xs, :] = umap_norm

    plt.figure(figsize=(12, 7))
    plt.imshow(rgb_image, origin='upper')
    plt.title("UMAP RGB Spatial Map  —  R=UMAP1  G=UMAP2  B=UMAP3", fontsize=14)
    plt.axis('off')
    plt.tight_layout()

    if output_dir:
        plt.savefig(os.path.join(output_dir, "umap_rgb_map.png"),
                    dpi=config.FIGURE_DPI, bbox_inches='tight')
    if show_plot:
        plt.show()
    plt.close()

    return rgb_image, umap_norm


# =============================================================================
# 3. RGB SPATIAL MAP (interactive Plotly with hover)
# =============================================================================

def plot_umap_rgb_interactive(rgb_image: np.ndarray, X_umap: np.ndarray,
                               tissue_indices_final: np.ndarray,
                               df: pd.DataFrame, channel_names_filtered: list,
                               height: int, width: int,
                               output_dir: str = None):
    """
    Interactive version of the UMAP RGB spatial map using Plotly.

    Hovering over any tissue pixel shows:
        - The pixel's (x, y) coordinates
        - Its UMAP1, UMAP2, UMAP3 values
        - The raw intensity of every element at that pixel

    This is the key interactive tool for exploring the biology — you can
    hover over a bright region in the tissue and immediately see which elements
    are high there.

    Parameters:
        rgb_image            : (H, W, 3) RGB image from plot_umap_rgb()
        X_umap               : UMAP coordinates, shape (n_pixels, 3)
        tissue_indices_final : flat pixel indices of tissue pixels
        df                   : original pixel dataframe (raw intensities for hover)
        channel_names_filtered: list of channel name strings
        height, width        : original image dimensions
        output_dir           : if provided, saves as interactive HTML file
    """
    n_channels = len(channel_names_filtered)

    # Build a (H, W, 3 + n_channels) array to store hover data at each pixel
    # 3 = UMAP coordinates, n_channels = element intensities
    customdata = np.full((height, width, 3 + n_channels), np.nan, dtype=np.float32)

    ys, xs = np.unravel_index(tissue_indices_final, (height, width))
    customdata[ys, xs, 0:3] = X_umap.astype(np.float32)              # UMAP1, 2, 3
    customdata[ys, xs, 3:]  = df[channel_names_filtered].values.astype(np.float32)  # element intensities

    # Build the hover tooltip template
    hover_lines = [
        'Pixel: (x=%{x}, y=%{y})',
        '──────────────',
        'UMAP1: %{customdata[0]:.2f}',
        'UMAP2: %{customdata[1]:.2f}',
        'UMAP3: %{customdata[2]:.2f}',
        '──────────────',
    ] + [
        f'{name}: %{{customdata[{i+3}]:.0f}}'
        for i, name in enumerate(channel_names_filtered)
    ]
    hovertemplate = '<br>'.join(hover_lines) + '<extra></extra>'

    # go.Image expects uint8 (0–255) values
    fig = go.Figure(data=go.Image(
        z=(rgb_image * 255).astype(np.uint8),
        customdata=customdata,
        hovertemplate=hovertemplate,
    ))

    fig.update_layout(
        title='UMAP RGB Spatial Map — interactive hover',
        width=1400, height=780,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False, scaleanchor='x'),
        margin=dict(l=10, r=10, t=60, b=10),
        hoverlabel=dict(font_size=11, font_family='monospace'),
    )

    if output_dir:
        fig.write_html(os.path.join(output_dir, "umap_rgb_interactive.html"))

    fig.show()


# =============================================================================
# 4. 3D SCATTER — UMAP SPACE (coloured by UMAP RGB)
# =============================================================================

def plot_umap_3d_scatter(X_umap: np.ndarray, umap_norm: np.ndarray,
                          output_dir: str = None):
    """
    3D interactive scatter plot of the UMAP embedding, where each point is
    coloured by its own UMAP-derived RGB colour.

    This plot acts as the colour legend for the RGB spatial map — it shows
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
        title='UMAP 3D Space — coloured by RGB encoding'
              '<br><sub>This is the colour legend for the RGB spatial map</sub>',
        scene=dict(
            xaxis=dict(title='UMAP 1 → R', backgroundcolor='black'),
            yaxis=dict(title='UMAP 2 → G', backgroundcolor='black'),
            zaxis=dict(title='UMAP 3 → B', backgroundcolor='black'),
        ),
        width=900, height=800,
        margin=dict(l=0, r=0, t=80, b=0),
    )

    if output_dir:
        fig.write_html(os.path.join(output_dir, "umap_3d_scatter.html"))

    fig.show()


# =============================================================================
# 5. 2D UMAP SCATTER — PER CHANNEL
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
        title=f'UMAP 2D — {channel_name} (Linear)',
        labels={'x': 'UMAP 1', 'y': 'UMAP 2'},
    )
    fig_linear.update_traces(marker=dict(size=5, opacity=0.6))
    fig_linear.update_layout(coloraxis_colorbar=dict(title=channel_name),
                              width=1000, height=800)
    if output_dir:
        fig_linear.write_html(os.path.join(output_dir, f"umap_2d_{channel_name}_linear.html"))
    fig_linear.show()

    # Only show log version if flagged
    if suggestion in ['LOG', 'POSSIBLY LOG']:
        fig_log = px.scatter(
            x=X_umap[:, 0], y=X_umap[:, 1],
            color=color_log,
            color_continuous_scale='hot',
            title=f'UMAP 2D — {channel_name} Log Scale (auto-flagged: {suggestion})',
            labels={'x': 'UMAP 1', 'y': 'UMAP 2'},
        )
        fig_log.update_traces(marker=dict(size=5, opacity=0.6))
        fig_log.update_layout(coloraxis_colorbar=dict(title=f'{channel_name} (log)'),
                               width=1000, height=800)
        if output_dir:
            fig_log.write_html(os.path.join(output_dir, f"umap_2d_{channel_name}_log.html"))
        fig_log.show()


# =============================================================================
# 6. ALL-CHANNELS GRID
# =============================================================================

def plot_umap_all_channels(X_umap: np.ndarray, df: pd.DataFrame,
                            channel_names_filtered: list, scale_suggestions: dict,
                            output_dir: str = None, show_plot: bool = True):
    """
    Plots a grid of 2D UMAP scatter plots — one panel per element.

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

    plt.suptitle('UMAP — All Elements', fontsize=18, y=1.01)
    plt.tight_layout()

    if output_dir:
        plt.savefig(os.path.join(output_dir, "umap_all_channels.png"),
                    dpi=config.FIGURE_DPI, bbox_inches='tight')
    if show_plot:
        plt.show()
    plt.close()


# =============================================================================
# 7. DOMINANT ELEMENT MAP
# =============================================================================

def plot_umap_dominant_element(X_umap: np.ndarray, df: pd.DataFrame,
                                output_dir: str = None, show_plot: bool = True):
    """
    Colours each pixel in the UMAP scatter by which element is dominant at that pixel.

    For each pixel, the element with the highest normalised intensity is treated
    as the 'dominant' element and determines the pixel's colour.

    Normalisation uses 99th percentile clipping so that outlier hotspot pixels
    don't overwhelm all others in a single element.

    This plot is useful for quickly identifying whether any element strongly
    separates into a distinct region of UMAP space — which would suggest that
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

    # Legend — one patch per element
    handles = [mpatches.Patch(color=palette[i], label=labels[i]) for i in range(n)]
    ax.legend(handles=handles, title='Dominant element',
              bbox_to_anchor=(1.02, 1), loc='upper left', frameon=False)

    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    ax.set_title('UMAP — Dominant element per pixel', fontsize=14)
    plt.tight_layout()

    if output_dir:
        plt.savefig(os.path.join(output_dir, "umap_dominant_element.png"),
                    dpi=config.FIGURE_DPI, bbox_inches='tight')
    if show_plot:
        plt.show()
    plt.close()


# =============================================================================
# 8. ELEMENT KEY — RGB MAP + PER-ELEMENT SPATIAL THUMBNAILS
# =============================================================================

def plot_element_key(
    rgb_image: np.ndarray,
    df: pd.DataFrame,
    tissue_indices_final: np.ndarray,
    height: int,
    width: int,
    output_dir: str = None,
    show_plot: bool = True,
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

    # --- Left panel: RGB spatial map ---
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

    # --- Right panel: per-element thumbnails ---
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
        "Element Key — RGB map vs per-element spatial distribution",
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

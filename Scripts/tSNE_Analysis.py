# =============================================================================
# tSNE_Analysis.py
#
# PURPOSE:
#   Runs t-SNE on the log1p-normalised pixel data and produces visualisations.
#   t-SNE is used for thesis/paper discussion and comparison with UMAP.
#   It is NOT used as the primary facility tool (too slow for large datasets).
#
# WHY SUBSAMPLE?
#   t-SNE does not scale well to large datasets. The full embryo dataset has
#   ~627,000 tissue pixels — running t-SNE on all of them is impractical.
#   We randomly subsample to 50,000 pixels (configurable in config.py).
#   A backup/restore mechanism ensures the full dataset is preserved so
#   UMAP and clustering can still use all pixels if needed.
#
# WHY 3 COMPONENTS?
#   Like UMAP, 3D t-SNE allows the three axes to be mapped to RGB channels
#   for a spatial tissue map where colour encodes structural similarity.
#
# ORDER OF STEPS:
#   1. Subsample pixels (with backup/restore of full dataset)
#   2. Run t-SNE
#   3. RGB spatial map (t-SNE1→R, t-SNE2→G, t-SNE3→B)
#   4. 2D scatter per channel (auto log/linear)
#   5. All-channels grid
#   6. Dominant element map
#   7. Spatial cluster map side-by-side with reference channel
# =============================================================================
 
import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import plotly.express as px
from sklearn.manifold import TSNE
 
import config
 
 
# =============================================================================
# 1. SUBSAMPLE
# =============================================================================
 
def subsample(df: pd.DataFrame, tissue_indices_final: np.ndarray):
    """
    Randomly subsamples the pixel dataframe to a manageable size for t-SNE.
 
    t-SNE has O(n²) complexity — running it on 600,000+ pixels is too slow.
    We subsample to config.TSNE_MAX_PIXELS (default 50,000) before running.
 
    IMPORTANT — backup/restore mechanism:
        The full df and tissue_indices_final are backed up before subsampling.
        This function returns both the subsampled versions (for t-SNE) and
        the full backups (so UMAP/clustering can still use all pixels).
 
        If you call this function again on an already-subsampled df, it will
        detect that and skip re-subsampling to avoid double-reduction.
 
    Parameters:
        df                   : log1p-normalised pixel dataframe (full dataset)
        tissue_indices_final : flat pixel indices aligned to df
 
    Returns:
        df_sub               : subsampled dataframe for t-SNE
        tissue_indices_sub   : tissue indices aligned to the subsampled df
        df_full              : full dataset backup (pass to restore_full() later)
        tissue_indices_full  : full tissue indices backup
    """
    n_total   = len(df)
    n_samples = config.TSNE_MAX_PIXELS
 
    # Back up the full dataset before subsampling
    df_full             = df.copy()
    tissue_indices_full = tissue_indices_final.copy()
 
    if n_samples >= n_total:
        print(f"n_samples ({n_samples:,}) >= total pixels ({n_total:,}). Keeping full dataset.")
        return df.copy(), tissue_indices_final.copy(), df_full, tissue_indices_full
 
    # Random subsample — sorted to keep spatial order tidy
    rng           = np.random.default_rng(config.TSNE_RANDOM_STATE)
    subsample_idx = rng.choice(n_total, size=n_samples, replace=False)
    subsample_idx.sort()
 
    # Subsample df and tissue_indices in lockstep
    # They MUST stay aligned so spatial remapping works correctly later
    df_sub               = df.iloc[subsample_idx].reset_index(drop=True)
    tissue_indices_sub   = tissue_indices_full[subsample_idx]
 
    print(f"Subsampled for t-SNE:")
    print(f"  Kept:    {len(df_sub):,} pixels ({100 * len(df_sub) / n_total:.1f}%)")
    print(f"  Dropped: {n_total - len(df_sub):,} pixels")
 
    return df_sub, tissue_indices_sub, df_full, tissue_indices_full
 
 
# =============================================================================
# 2. RUN t-SNE
# =============================================================================
 
def run_tsne(df_sub: pd.DataFrame):
    """
    Runs t-SNE on the subsampled log1p-normalised pixel dataframe.
 
    t-SNE (t-distributed Stochastic Neighbour Embedding) is a non-linear
    dimensionality reduction method that places similar pixels close together
    in the embedding. Unlike UMAP, it does not preserve global structure —
    only local neighbourhoods are reliable.
 
    Settings (from config.py):
        TSNE_N_COMPONENTS = 3          : 3D for RGB spatial map
        TSNE_PERPLEXITY   = 30         : balances local vs global structure
        TSNE_MAX_ITER     = 3000       : more iterations = more stable embedding
        learning_rate     = n/12       : heuristic based on dataset size
        TSNE_INIT         = 'pca'      : PCA initialisation is more stable than random
        TSNE_METHOD       = 'barnes_hut': approximate algorithm, much faster than exact
        TSNE_METRIC       = 'cosine'   : angle-based similarity, better for intensity data
        TSNE_RANDOM_STATE = 42         : reproducible results
 
    Parameters:
        df_sub : subsampled log1p-normalised pixel dataframe from subsample()
 
    Returns:
        X_tsne : numpy array of t-SNE coordinates, shape (n_subsampled_pixels, 3)
    """
    # Learning rate heuristic: n_samples / early_exaggeration
    # Scales the learning rate with dataset size for more stable convergence
    learning_rate = len(df_sub) / 12
 
    print(f"Running t-SNE on {len(df_sub):,} pixels...")
    print(f"  learning_rate = {learning_rate:.0f}  (n={len(df_sub):,} / 12)")
 
    X_tsne = TSNE(
        n_components    = config.TSNE_N_COMPONENTS,
        perplexity      = config.TSNE_PERPLEXITY,
        early_exaggeration = 12,
        max_iter        = config.TSNE_MAX_ITER,
        learning_rate   = learning_rate,
        init            = config.TSNE_INIT,
        method          = config.TSNE_METHOD,
        metric          = config.TSNE_METRIC,
        n_jobs          = -1,            # use all CPU cores
        random_state    = config.TSNE_RANDOM_STATE,
        verbose         = 1,
    ).fit_transform(df_sub.values)
 
    print(f"\nt-SNE complete. Output shape: {X_tsne.shape}")
    return X_tsne
 
 
# =============================================================================
# 3. RGB SPATIAL MAP
# =============================================================================
 
def plot_tsne_rgb(X_tsne: np.ndarray, tissue_indices_sub: np.ndarray,
                   height: int, width: int,
                   output_dir: str = None, show_plot: bool = True):
    """
    Creates a spatial RGB image where each tissue pixel's colour encodes its
    position in t-SNE space:
        Red channel   = t-SNE dimension 1 (normalised to 0–1)
        Green channel = t-SNE dimension 2 (normalised to 0–1)
        Blue channel  = t-SNE dimension 3 (normalised to 0–1)
 
    Only the subsampled pixels are coloured — the rest of the tissue
    background remains white. This is expected behaviour since t-SNE
    was only run on a subset of pixels.
 
    Parameters:
        X_tsne              : t-SNE coordinates, shape (n_subsampled, 3)
        tissue_indices_sub  : flat pixel indices for the subsampled pixels
        height, width       : original image dimensions
        output_dir          : if provided, saves the figure here
        show_plot           : if True, displays the figure
 
    Returns:
        rgb_image  : (H, W, 3) float RGB image
        tsne_norm  : (n_subsampled, 3) normalised t-SNE coordinates
    """
    # Min-max normalise each t-SNE axis to [0, 1] for RGB mapping
    tsne_min  = X_tsne.min(axis=0)
    tsne_max  = X_tsne.max(axis=0)
    tsne_norm = (X_tsne - tsne_min) / (tsne_max - tsne_min)
 
    # Build (H, W, 3) image — white background
    rgb_image = np.ones((height, width, 3), dtype=float)
 
    # Place subsampled tissue pixel colours at their (y, x) positions
    ys, xs = np.unravel_index(tissue_indices_sub, (height, width))
    rgb_image[ys, xs, :] = tsne_norm
 
    plt.figure(figsize=(12, 7))
    plt.imshow(rgb_image, origin='upper')
    plt.title("t-SNE RGB Spatial Map  —  R=tSNE1  G=tSNE2  B=tSNE3\n"
              f"(subsampled: {len(X_tsne):,} pixels)", fontsize=13)
    plt.axis('off')
    plt.tight_layout()
 
    if output_dir:
        plt.savefig(os.path.join(output_dir, "tsne_rgb_map.png"),
                    dpi=config.FIGURE_DPI, bbox_inches='tight')
    if show_plot:
        plt.show()
    plt.close()
 
    return rgb_image, tsne_norm
 
 
# =============================================================================
# 4. 2D SCATTER — PER CHANNEL
# =============================================================================
 
def plot_tsne_by_channel(X_tsne: np.ndarray, df_sub: pd.DataFrame,
                          channel_name: str, scale_suggestions: dict,
                          output_dir: str = None):
    """
    2D t-SNE scatter plot coloured by the intensity of a single element.
 
    Automatically shows a second log-scale version if the channel was flagged
    as LOG or POSSIBLY LOG by preprocessing.suggest_scale().
 
    Parameters:
        X_tsne           : t-SNE coordinates, shape (n_subsampled, 3)
        df_sub           : subsampled pixel dataframe
        channel_name     : name of the element to colour by (e.g. 'Fe', 'Zn')
        scale_suggestions: dict from preprocessing.suggest_scale()
        output_dir       : if provided, saves figures here
    """
    suggestion    = scale_suggestions.get(channel_name, 'LINEAR')
    color_linear  = df_sub[channel_name].reset_index(drop=True)
    color_log     = np.log1p(color_linear)
 
    # Always show linear version
    fig_linear = px.scatter(
        x=X_tsne[:, 0], y=X_tsne[:, 1],
        color=color_linear,
        color_continuous_scale='hot',
        title=f't-SNE 2D — {channel_name} (Linear)',
        labels={'x': 't-SNE 1', 'y': 't-SNE 2'},
    )
    fig_linear.update_traces(marker=dict(size=5, opacity=0.6))
    fig_linear.update_layout(coloraxis_colorbar=dict(title=channel_name),
                              width=1000, height=800)
    if output_dir:
        fig_linear.write_html(os.path.join(output_dir, f"tsne_2d_{channel_name}_linear.html"))
    fig_linear.show()
 
    # Only show log version if flagged
    if suggestion in ['LOG', 'POSSIBLY LOG']:
        fig_log = px.scatter(
            x=X_tsne[:, 0], y=X_tsne[:, 1],
            color=color_log,
            color_continuous_scale='hot',
            title=f't-SNE 2D — {channel_name} Log Scale (auto-flagged: {suggestion})',
            labels={'x': 't-SNE 1', 'y': 't-SNE 2'},
        )
        fig_log.update_traces(marker=dict(size=5, opacity=0.6))
        fig_log.update_layout(coloraxis_colorbar=dict(title=f'{channel_name} (log)'),
                               width=1000, height=800)
        if output_dir:
            fig_log.write_html(os.path.join(output_dir, f"tsne_2d_{channel_name}_log.html"))
        fig_log.show()
 
 
# =============================================================================
# 5. ALL-CHANNELS GRID
# =============================================================================
 
def plot_tsne_all_channels(X_tsne: np.ndarray, df_sub: pd.DataFrame,
                            channel_names_filtered: list, scale_suggestions: dict,
                            output_dir: str = None, show_plot: bool = True):
    """
    Grid of 2D t-SNE scatter plots — one panel per element.
    Automatically applies log scale for flagged channels.
 
    Parameters:
        X_tsne                : t-SNE coordinates, shape (n_subsampled, 3)
        df_sub                : subsampled pixel dataframe
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
        color_vals = df_sub.iloc[:, i].reset_index(drop=True)
 
        suggestion = scale_suggestions.get(channel_name, 'LINEAR')
        if suggestion in ['LOG', 'POSSIBLY LOG']:
            color_vals  = np.log1p(color_vals)
            scale_label = '(log)'
        else:
            scale_label = ''
 
        sc = axes[i].scatter(
            X_tsne[:, 0], X_tsne[:, 1],
            c=color_vals, cmap='hot', s=0.5, rasterized=True
        )
        plt.colorbar(sc, ax=axes[i], shrink=0.7)
        axes[i].set_title(f'{channel_name} {scale_label}', fontsize=12)
        axes[i].axis('off')
 
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
 
    plt.suptitle('t-SNE — All Elements', fontsize=18, y=1.01)
    plt.tight_layout()
 
    if output_dir:
        plt.savefig(os.path.join(output_dir, "tsne_all_channels.png"),
                    dpi=config.FIGURE_DPI, bbox_inches='tight')
    if show_plot:
        plt.show()
    plt.close()
 
 
# =============================================================================
# 6. DOMINANT ELEMENT MAP
# =============================================================================
 
def plot_tsne_dominant_element(X_tsne: np.ndarray, df_sub: pd.DataFrame,
                                output_dir: str = None, show_plot: bool = True):
    """
    Colours each pixel in the t-SNE scatter by which element is dominant.
    Identical logic to the UMAP dominant element map for direct comparison.
 
    Parameters:
        X_tsne     : t-SNE coordinates, shape (n_subsampled, 3)
        df_sub     : subsampled pixel dataframe
        output_dir : if provided, saves the figure here
        show_plot  : if True, displays the figure
    """
    X      = df_sub.values.astype(float)
    X_norm = np.clip(X / (np.percentile(X, 99, axis=0) + 1e-10), 0, 1)
    dominant = X_norm.argmax(axis=1)
 
    n      = df_sub.shape[1]
    labels = list(df_sub.columns)
    palette = (plt.cm.tab10(np.linspace(0, 1, n)) if n <= 10
               else plt.cm.tab20(np.linspace(0, 1, n)))
    pixel_colours = palette[dominant]
 
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.scatter(X_tsne[:, 0], X_tsne[:, 1],
               c=pixel_colours, s=0.3, alpha=0.7)
 
    handles = [mpatches.Patch(color=palette[i], label=labels[i]) for i in range(n)]
    ax.legend(handles=handles, title='Dominant element',
              bbox_to_anchor=(1.02, 1), loc='upper left', frameon=False)
 
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.set_title('t-SNE — Dominant element per pixel', fontsize=14)
    plt.tight_layout()
 
    if output_dir:
        plt.savefig(os.path.join(output_dir, "tsne_dominant_element.png"),
                    dpi=config.FIGURE_DPI, bbox_inches='tight')
    if show_plot:
        plt.show()
    plt.close()
 
 
# =============================================================================
# 7. SPATIAL CLUSTER MAP (side-by-side with reference channel)
# =============================================================================
 
def plot_tsne_spatial_clusters(cluster_labels: np.ndarray,
                                tissue_indices_sub: np.ndarray,
                                img_filtered: np.ndarray,
                                channel_names_filtered: list,
                                threshold_lookup: dict,
                                height: int, width: int,
                                output_dir: str = None, show_plot: bool = True):
    """
    Maps t-SNE cluster labels back to their original spatial positions and plots
    them side-by-side with each reference element channel.
 
    This is the spatial validation step — it shows whether the clusters found
    in t-SNE space correspond to meaningful tissue structures in the original image.
 
    Parameters:
        cluster_labels        : cluster label per subsampled pixel (from clustering.py)
        tissue_indices_sub    : flat pixel indices for the subsampled pixels
        img_filtered          : filtered image array (n_channels, H, W)
        channel_names_filtered: list of channel name strings
        threshold_lookup      : dict of {channel_name: (min, max)} from preprocessing
        height, width         : original image dimensions
        output_dir            : if provided, saves figures here
        show_plot             : if True, displays the figures
    """
    n_clusters = len(np.unique(cluster_labels))
 
    # Build the spatial cluster map — -1 = background (masked)
    tsne_cluster_map = np.full((height, width), fill_value=-1, dtype=int)
    tsne_cluster_map.flat[tissue_indices_sub] = cluster_labels.astype(int)
 
    print(f"Mapping {n_clusters} t-SNE clusters to {height}x{width} spatial grid.")
 
    # Colour scheme — consistent with UMAP cluster maps
    cluster_colours = ['steelblue', 'red', 'limegreen', 'orange', 'purple', 'cyan']
    cmap_colours    = ['black'] + cluster_colours[:n_clusters]
    cmap            = ListedColormap(cmap_colours)
 
    legend_elements = [Patch(facecolor='black', label='Background (Masked)')]
    for i in range(n_clusters):
        legend_elements.append(Patch(facecolor=cluster_colours[i], label=f'Cluster {i}'))
 
    # Side-by-side: cluster map + each reference channel
    for i, channel_name in enumerate(channel_names_filtered):
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
 
        # Left: t-SNE cluster map
        axes[0].imshow(tsne_cluster_map, cmap=cmap,
                       vmin=-1, vmax=n_clusters - 1, origin='upper')
        axes[0].set_title('t-SNE K-Means Cluster Map', fontsize=14)
        axes[0].axis('off')
        axes[0].legend(handles=legend_elements, loc='upper right',
                       bbox_to_anchor=(1.3, 1))
 
        # Right: raw element channel for reference
        vmin_ch, vmax_ch = threshold_lookup.get(
            channel_name, (0, np.percentile(img_filtered[i], 99))
        )
        axes[1].imshow(img_filtered[i], cmap='inferno',
                       vmin=vmin_ch, vmax=vmax_ch, origin='upper')
        axes[1].set_title(f'Reference: {channel_name}', fontsize=14)
        axes[1].axis('off')
 
        plt.suptitle(f'Spatial Validation: t-SNE Clusters vs {channel_name}',
                     fontsize=16, y=1.05)
        plt.tight_layout()
 
        if output_dir:
            fname = f"tsne_spatial_vs_{channel_name}.png"
            plt.savefig(os.path.join(output_dir, fname),
                        dpi=config.FIGURE_DPI, bbox_inches='tight')
        if show_plot:
            plt.show()
        plt.close()
# =============================================================================
# Clustering.py
#
# PURPOSE:
#   Applies K-means and HDBSCAN clustering to the UMAP embedding and maps
#   the cluster labels back to their original spatial positions in the tissue.
#
# Note — PARAMETERS NOT FINALISED:
#   The HDBSCAN and K-means parameters in config.py are placeholders.
#   The correct workflow is:
#     1. Run run_hdbscan_sweep() first to test multiple parameter combinations
#     2. Look at the sweep results table and pick the best settings
#     3. Update HDBSCAN_MIN_CLUSTER_SIZE and HDBSCAN_MIN_SAMPLES in config.py
#     4. Run run_hdbscan() with those finalised parameters
#   Do the same for K-means using the elbow plot from run_kmeans_elbow().
#
# TWO CLUSTERING METHODS:
#   K-means  — forces every pixel into a cluster. Simple and fast.
#              Good for comparison and thesis discussion.
#   HDBSCAN  — density-based, identifies natural clusters and labels
#              low-density pixels as 'noise'. More biologically principled.
#              This is the primary clustering method for the pipeline.
#
# ORDER OF STEPS:
#   1. K-means elbow method (KneeLocator auto-detects optimal k)
#   2. Apply K-means with chosen k
#   3. K-means spatial map (side-by-side with reference channels)
#   4. Per-channel K-means tiering (sorted by intensity per element)
#   5. HDBSCAN parameter sweep (find best min_cluster_size + min_samples)
#   6. Apply HDBSCAN with chosen parameters
#   7. HDBSCAN spatial map (static matplotlib)
#   8. HDBSCAN spatial map (interactive Plotly with hover)
#   9. K-means vs HDBSCAN comparison plot
#  10. ARI/AMI agreement metrics
# =============================================================================
 
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
import hdbscan
 
import config
 
 
# =============================================================================
# 1. K-MEANS ELBOW METHOD
# =============================================================================
 
def run_kmeans_elbow(X_umap: np.ndarray, output_dir: str = None, show_plot: bool = True):
    """
    Tests K-means for k=1 to config.KMEANS_MAX_K and plots the elbow curve.
    Uses KneeLocator to automatically suggest the optimal k.
 
    WCSS (Within-Cluster Sum of Squares) measures how tight the clusters are.
    Lower WCSS = pixels within each cluster are more similar to each other.
    The 'elbow' is the point where adding more clusters stops giving meaningful
    improvement — that's your optimal k.
 
    Parameters:
        X_umap     : UMAP coordinates, shape (n_pixels, 3)
        output_dir : if provided, saves the figure here
        show_plot  : if True, displays the figure
 
    Returns:
        optimal_k  : automatically detected elbow point (or None if not found)
        wcss       : list of WCSS values for k=1 to max_k
    """
    from kneed import KneeLocator
 
    max_k = config.KMEANS_MAX_K
    wcss  = []
 
    print(f"Testing K-means for k=1 to {max_k}...")
    for k in range(1, max_k + 1):
        km = KMeans(n_clusters=k, init='k-means++', max_iter=300,
                    n_init='auto', random_state=config.KMEANS_RANDOM_STATE)
        km.fit(X_umap)
        wcss.append(km.inertia_)
        print(f"  k={k}: WCSS = {km.inertia_:.2f}")
 
    # Auto-detect the elbow point
    kl        = KneeLocator(range(1, max_k + 1), wcss, curve='convex', direction='decreasing')
    optimal_k = kl.elbow
    print(f"\nKneeLocator suggests k = {optimal_k}")
 
    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, max_k + 1), wcss, marker='o', linestyle='--', color='blue')
    if optimal_k is not None:
        plt.axvline(optimal_k, color='red', linestyle=':',
                    label=f'Elbow (k = {optimal_k})')
        plt.legend()
    plt.title("Elbow Method (KneeLocator) — UMAP 3D", fontsize=18)
    plt.xlabel("Number of Clusters (k)", fontsize=13)
    plt.ylabel("WCSS", fontsize=13)
    plt.xticks(range(1, max_k + 1))
    plt.grid(True, alpha=0.5)
    plt.tight_layout()
 
    if output_dir:
        plt.savefig(os.path.join(output_dir, "kmeans_elbow.png"),
                    dpi=config.FIGURE_DPI, bbox_inches='tight')
    if show_plot:
        plt.show()
    plt.close()
 
    return optimal_k, wcss
 
 
# =============================================================================
# 2. APPLY K-MEANS
# =============================================================================
 
def run_kmeans(X_umap: np.ndarray, df: pd.DataFrame, chosen_k: int = None):
    """
    Applies K-means clustering to the UMAP coordinates.
 
    If chosen_k is not provided, uses the optimal k from the elbow method.
    Falls back to 4 if neither is available.
 
    K-means runs on X_umap (the 3D UMAP coordinates), not on the raw pixel data.
    This means clusters are defined by position in UMAP space — pixels that are
    close together in UMAP space get the same cluster label.
 
    Parameters:
        X_umap    : UMAP coordinates, shape (n_pixels, 3)
        df        : pixel dataframe (cluster labels will be added as a column)
        chosen_k  : number of clusters to use (override elbow if needed)
 
    Returns:
        cluster_labels : integer array of cluster IDs, shape (n_pixels,)
        df             : updated dataframe with 'cluster' column added
    """
    k = chosen_k if chosen_k is not None else config.KMEANS_MAX_K
 
    print(f"Applying K-means with k={k} on UMAP coordinates...")
    km = KMeans(n_clusters=k, init='k-means++', max_iter=300,
                n_init='auto', random_state=config.KMEANS_RANDOM_STATE)
    cluster_labels = km.fit_predict(X_umap)
 
    # Add cluster labels to the dataframe as a string column
    # (string so plotting libraries treat them as discrete categories)
    df['cluster'] = cluster_labels.astype(str)
 
    print(f"K-means complete using k={k}.")
    print("\n--- Cluster Distribution ---")
    counts = df['cluster'].value_counts().sort_index()
    for cid, count in counts.items():
        pct = (count / len(df)) * 100
        print(f"  Cluster {cid}: {count:,} pixels ({pct:.1f}%)")
 
    return cluster_labels, df
 
 
# =============================================================================
# 3. K-MEANS SPATIAL MAP
# =============================================================================
 
def plot_kmeans_spatial(cluster_labels: np.ndarray, tissue_indices_final: np.ndarray,
                         img_filtered: np.ndarray, channel_names_filtered: list,
                         threshold_lookup: dict, height: int, width: int,
                         output_dir: str = None, show_plot: bool = True):
    """
    Maps K-means cluster labels back to their spatial positions in the tissue image
    and plots them side-by-side with each reference element channel.
 
    This validates whether the UMAP-derived clusters correspond to real tissue
    structures visible in the elemental maps.
 
    -1 in the cluster map = background (pixels outside the tissue mask)
 
    Parameters:
        cluster_labels        : K-means labels, shape (n_pixels,)
        tissue_indices_final  : flat pixel indices aligned to cluster_labels
        img_filtered          : filtered image array (n_channels, H, W)
        channel_names_filtered: list of channel name strings
        threshold_lookup      : dict of {channel_name: (min, max)} from preprocessing
        height, width         : original image dimensions
        output_dir            : if provided, saves figures here
        show_plot             : if True, displays the figures
    """
    n_clusters = len(np.unique(cluster_labels))
 
    # Build the 2D cluster map — fill with -1 (background)
    cluster_map = np.full((height, width), fill_value=-1, dtype=int)
    cluster_map.flat[tissue_indices_final] = cluster_labels.astype(int)
 
    print(f"Mapping {n_clusters} K-means clusters to {height}x{width} spatial grid.")
 
    # Colour scheme
    cluster_colours = ['steelblue', 'red', 'limegreen', 'orange', 'purple', 'cyan',
                       '#DD8452', '#8172B2', '#CCB974', '#64B5CD']
    cmap_colours = ['white'] + cluster_colours[:n_clusters]
    cmap         = ListedColormap(cmap_colours)
 
    legend_elements = [Patch(facecolor='white', edgecolor='black', label='Background (Masked)')]
    for i in range(n_clusters):
        legend_elements.append(Patch(facecolor=cluster_colours[i], label=f'Cluster {i}'))
 
    # Side-by-side: cluster map + each reference channel
    for i, channel_name in enumerate(channel_names_filtered):
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
 
        axes[0].imshow(cluster_map, cmap=cmap, vmin=-1, vmax=n_clusters - 1, origin='upper')
        axes[0].set_title('K-Means Cluster Map', fontsize=14)
        axes[0].axis('off')
        axes[0].legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.3, 1))
 
        vmin_ch, vmax_ch = threshold_lookup.get(
            channel_name, (0, np.percentile(img_filtered[i], 99))
        )
        axes[1].imshow(img_filtered[i], cmap='inferno',
                       vmin=vmin_ch, vmax=vmax_ch, origin='upper')
        axes[1].set_title(f'Reference: {channel_name}', fontsize=14)
        axes[1].axis('off')
 
        plt.suptitle(f'Spatial Validation: K-Means Clusters vs {channel_name}',
                     fontsize=16, y=1.02)
        plt.tight_layout()
 
        if output_dir:
            plt.savefig(os.path.join(output_dir, f"kmeans_spatial_vs_{channel_name}.png"),
                        dpi=config.FIGURE_DPI, bbox_inches='tight')
        if show_plot:
            plt.show()
        plt.close()
 
 
# =============================================================================
# 4. PER-CHANNEL K-MEANS TIERING
# =============================================================================
 
def plot_kmeans_per_channel(df: pd.DataFrame, tissue_indices_final: np.ndarray,
                             channel_names_filtered: list, img_filtered: np.ndarray,
                             threshold_lookup: dict, height: int, width: int,
                             n_tiers: int = 3,
                             output_dir: str = None, show_plot: bool = True):
    """
    Runs K-means independently on each element channel (1D clustering).
 
    For each element, pixels are split into n_tiers intensity bands:
    Cluster 0 = lowest intensity, Cluster 1 = medium, Cluster 2 = highest.
    The clusters are always sorted by mean intensity so the labelling is consistent.
 
    This is useful for identifying spatial intensity gradients per element
    and comparing them with the multi-channel UMAP-based clusters.
 
    Parameters:
        df                    : pixel dataframe
        tissue_indices_final  : flat pixel indices
        channel_names_filtered: list of channel name strings
        img_filtered          : filtered image array (n_channels, H, W)
        threshold_lookup      : dict of {channel_name: (min, max)} from preprocessing
        height, width         : original image dimensions
        n_tiers               : number of intensity bands (default 3)
        output_dir            : if provided, saves figures here
        show_plot             : if True, displays the figures
    """
    tier_colours = ['steelblue', 'red', 'limegreen', 'orange', 'purple', 'cyan']
    cmap_colours = ['white'] + tier_colours[:n_tiers]
    cmap_tiers   = ListedColormap(cmap_colours)
 
    legend_elements = [Patch(facecolor='white', edgecolor='black', label='Background (Masked)')]
    for k in range(n_tiers):
        legend_elements.append(Patch(facecolor=tier_colours[k], label=f'Cluster {k}'))
 
    for i, channel_name in enumerate(channel_names_filtered):
 
        # Run K-means on this channel's intensities only (1D)
        channel_values = df[channel_name].values.reshape(-1, 1)
        km_tier        = KMeans(n_clusters=n_tiers, init='k-means++', max_iter=300,
                                n_init='auto', random_state=config.KMEANS_RANDOM_STATE)
        raw_labels     = km_tier.fit_predict(channel_values)
 
        # Sort clusters by mean intensity so Cluster 0 is always the lowest
        cluster_means  = [channel_values[raw_labels == k].mean() for k in range(n_tiers)]
        sorted_indices = np.argsort(cluster_means)
        mapping        = {old: new for new, old in enumerate(sorted_indices)}
        sorted_labels  = np.array([mapping[label] for label in raw_labels])
 
        # Spatial map
        tier_map = np.full((height, width), fill_value=-1, dtype=int)
        tier_map.flat[tissue_indices_final] = sorted_labels
 
        # Side-by-side plot
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
 
        axes[0].imshow(tier_map, cmap=cmap_tiers, vmin=-1, vmax=n_tiers - 1, origin='upper')
        axes[0].set_title(f'{channel_name}: K-means Intensity Tiers', fontsize=14)
        axes[0].axis('off')
        axes[0].legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.25, 1))
 
        vmin_ch, vmax_ch = threshold_lookup.get(
            channel_name, (0, np.percentile(img_filtered[i], 99))
        )
        axes[1].imshow(img_filtered[i], cmap='inferno',
                       vmin=vmin_ch, vmax=vmax_ch, origin='upper')
        axes[1].set_title(f'Reference: {channel_name}', fontsize=14)
        axes[1].axis('off')
 
        plt.suptitle(f'{channel_name} — Intensity Tiering vs Raw Map', fontsize=16, y=1.02)
        plt.tight_layout()
 
        if output_dir:
            plt.savefig(os.path.join(output_dir, f"kmeans_tiers_{channel_name}.png"),
                        dpi=config.FIGURE_DPI, bbox_inches='tight')
        if show_plot:
            plt.show()
        plt.close()
 
 
# =============================================================================
# 5. HDBSCAN PARAMETER SWEEP
# =============================================================================
 
def run_hdbscan_sweep(X_umap: np.ndarray):
    """
    Tests multiple HDBSCAN parameter combinations and returns a summary table.
 
    Run this BEFORE run_hdbscan() to find the best parameter settings.
    Look for rows where:
        biggest_cluster_pct < 60%  : balanced clustering (no one cluster dominates)
        noise_pct < 5%             : most pixels are confidently assigned
        n_clusters between 3–15    : biologically interpretable number of regions
 
    Uses config.HDBSCAN_SWEEP_MIN_CLUSTER_SIZES and config.HDBSCAN_SWEEP_MIN_SAMPLES
    as the ranges to test. Update those lists in config.py to widen/narrow the search.
 
    Parameters:
        X_umap : UMAP coordinates to cluster, shape (n_pixels, 3)
 
    Returns:
        sweep_df : DataFrame sorted by biggest_cluster_pct (ascending)
                   — best balanced rows appear at the top
    """
    results = []
 
    sweep_mcs = config.HDBSCAN_SWEEP_MIN_CLUSTER_SIZES
    sweep_ms  = config.HDBSCAN_SWEEP_MIN_SAMPLES
 
    print(f"Running HDBSCAN sweep: {len(sweep_mcs) * len(sweep_ms)} combinations...")
 
    for mcs in sweep_mcs:
        for ms in sweep_ms:
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=mcs,
                min_samples=ms,
                cluster_selection_method=config.HDBSCAN_CLUSTER_SELECTION,
                core_dist_n_jobs=-1,
            )
            labels    = clusterer.fit_predict(X_umap)
            n_clusters = labels.max() + 1
            n_noise    = (labels == -1).sum()
 
            if n_clusters > 0:
                sizes       = pd.Series(labels[labels >= 0]).value_counts()
                biggest_pct = sizes.iloc[0] / sizes.sum() * 100
            else:
                biggest_pct = 100
 
            results.append({
                'min_cluster_size'  : mcs,
                'min_samples'       : ms,
                'n_clusters'        : n_clusters,
                'noise_pct'         : round(100 * n_noise / len(labels), 2),
                'biggest_cluster_pct': round(biggest_pct, 2),
            })
 
    sweep_df = pd.DataFrame(results).sort_values('biggest_cluster_pct')
    print("\nSweep results (sorted by biggest cluster %):")
    print(sweep_df.to_string(index=False))
    print("\n→ Update HDBSCAN_MIN_CLUSTER_SIZE and HDBSCAN_MIN_SAMPLES in config.py")
    print("  with your chosen values, then run run_hdbscan().")
 
    return sweep_df
 
 
# =============================================================================
# 6. APPLY HDBSCAN
# =============================================================================
 
def run_hdbscan(X_umap: np.ndarray):
    """
    Applies HDBSCAN clustering to the UMAP coordinates using parameters from config.py.
 
    HDBSCAN differs from K-means in two important ways:
      1. It finds clusters automatically based on density — you don't pick k
      2. Low-density pixels are labelled as noise (-1) instead of forced into a cluster
 
    The 'leaf' cluster selection method splits density peaks aggressively,
    producing more, finer-grained clusters than the 'eom' method.
 
    NOTE: Run run_hdbscan_sweep() first to find appropriate parameter values,
    then update config.py before calling this function.
 
    Parameters:
        X_umap : UMAP coordinates, shape (n_pixels, 3)
 
    Returns:
        hdb_labels : integer array of cluster IDs, -1 = noise, shape (n_pixels,)
        sizes      : Series of cluster sizes (excluding noise)
    """
    print(f"Running HDBSCAN (min_cluster_size={config.HDBSCAN_MIN_CLUSTER_SIZE}, "
          f"min_samples={config.HDBSCAN_MIN_SAMPLES}, "
          f"method='{config.HDBSCAN_CLUSTER_SELECTION}')...")
 
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=config.HDBSCAN_MIN_CLUSTER_SIZE,
        min_samples=config.HDBSCAN_MIN_SAMPLES,
        cluster_selection_method=config.HDBSCAN_CLUSTER_SELECTION,
        core_dist_n_jobs=-1,
    )
    hdb_labels = clusterer.fit_predict(X_umap)
 
    n_hdb  = hdb_labels.max() + 1
    n_noise = (hdb_labels == -1).sum()
    sizes   = pd.Series(hdb_labels[hdb_labels >= 0]).value_counts().sort_index()
 
    print(f"\nHDBSCAN complete.")
    print(f"  Clusters found: {n_hdb}")
    print(f"  Noise pixels:   {n_noise:,} ({100 * n_noise / len(hdb_labels):.2f}%)")
    print("\n  Cluster sizes:")
    for cid, count in sizes.items():
        pct = count / sizes.sum() * 100
        print(f"    Cluster {cid}: {count:>7,} pixels ({pct:.2f}%)")
 
    return hdb_labels, sizes
 
 
# =============================================================================
# 7. HDBSCAN SPATIAL MAP (static)
# =============================================================================
 
def plot_hdbscan_spatial(hdb_labels: np.ndarray, sizes: pd.Series,
                          tissue_indices_final: np.ndarray,
                          height: int, width: int,
                          output_dir: str = None, show_plot: bool = True):
    """
    Maps HDBSCAN cluster labels back to spatial positions and plots the result.
 
    Colour scheme:
        White  = background (outside tissue mask)
        Grey   = HDBSCAN noise pixels (-1)
        Light grey = the dominant (largest) cluster — bulk tissue
        Colours = all other clusters (biologically distinct regions)
 
    The dominant cluster is shown in light grey because it typically represents
    the bulk tissue background — the interesting biology is in the smaller clusters.
 
    Parameters:
        hdb_labels           : HDBSCAN labels, -1 = noise, shape (n_pixels,)
        sizes                : cluster size Series from run_hdbscan()
        tissue_indices_final : flat pixel indices aligned to hdb_labels
        height, width        : original image dimensions
        output_dir           : if provided, saves figures here
        show_plot            : if True, displays the figure
 
    Returns:
        hdb_map    : 2D spatial cluster map array (H, W)
        hdb_colours: list of colours used (needed by comparison plot and interactive plot)
        cmap_hdb   : ListedColormap used (needed by comparison plot)
    """
    n_hdb = hdb_labels.max() + 1
 
    # Generate distinct colours — scales with number of clusters
    if n_hdb <= 10:
        palette = plt.cm.tab10(np.linspace(0, 1, n_hdb))
    elif n_hdb <= 20:
        palette = plt.cm.tab20(np.linspace(0, 1, n_hdb))
    else:
        palette = plt.cm.gist_ncar(np.linspace(0, 1, n_hdb))
 
    # Desaturate the biggest cluster to light grey (it's usually bulk tissue)
    biggest_cluster = sizes.idxmax()
    hdb_colours     = [tuple(c) for c in palette]
    hdb_colours[biggest_cluster] = (0.91, 0.91, 0.91, 1.0)
 
    # Colour order: white (background -2), grey (noise -1), then cluster colours
    cmap_colours = ['white', '#BDBDBD'] + hdb_colours
    cmap_hdb     = ListedColormap(cmap_colours)
 
    # Build spatial map: -2 = background, -1 = noise, 0..n-1 = clusters
    hdb_map = np.full((height, width), fill_value=-2, dtype=int)
    hdb_map.flat[tissue_indices_final] = hdb_labels
 
    n_noise = (hdb_labels == -1).sum()
 
    # Legend
    legend_elements = [
        Patch(facecolor='white', edgecolor='black', label='Background (Masked)'),
        Patch(facecolor='#BDBDBD', label='HDBSCAN Noise'),
    ]
    for k in range(n_hdb):
        label = f'Cluster {k} (bulk tissue)' if k == biggest_cluster else f'Cluster {k}'
        legend_elements.append(Patch(facecolor=hdb_colours[k], label=label))
 
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.imshow(hdb_map, cmap=cmap_hdb, vmin=-2, vmax=n_hdb - 1,
              origin='upper', interpolation='nearest')
    ax.set_title(
        f"HDBSCAN Cluster Map — '{config.HDBSCAN_CLUSTER_SELECTION}' selection "
        f"(k={n_hdb}, {n_noise:,} noise pixels)", fontsize=14
    )
    ax.axis('off')
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.35, 1),
              frameon=True, framealpha=0.9, edgecolor='lightgrey', fontsize=9)
    plt.tight_layout()
 
    if output_dir:
        plt.savefig(os.path.join(output_dir, "hdbscan_spatial.png"),
                    dpi=config.FIGURE_DPI, bbox_inches='tight')
    if show_plot:
        plt.show()
    plt.close()
 
    return hdb_map, hdb_colours, cmap_hdb
 
 
# =============================================================================
# 8. HDBSCAN INTERACTIVE MAP (Plotly)
# =============================================================================
 
def plot_hdbscan_interactive(hdb_map: np.ndarray, hdb_colours: list,
                              hdb_labels: np.ndarray, sizes: pd.Series,
                              output_dir: str = None):
    """
    Interactive Plotly heatmap of the HDBSCAN cluster map.
    Hovering over a pixel shows its cluster name.
 
    Parameters:
        hdb_map    : 2D spatial map from plot_hdbscan_spatial()
        hdb_colours: colour list from plot_hdbscan_spatial()
        hdb_labels : HDBSCAN labels from run_hdbscan()
        sizes      : cluster size Series from run_hdbscan()
        output_dir : if provided, saves as interactive HTML
    """
    n_hdb           = hdb_labels.max() + 1
    n_noise         = (hdb_labels == -1).sum()
    biggest_cluster = sizes.idxmax()
 
    def rgba_to_hex(rgba):
        r, g, b = [int(c * 255) for c in rgba[:3]]
        return f'#{r:02x}{g:02x}{b:02x}'
 
    # Build stepped discrete colorscale for Plotly
    hex_colours = ['#FFFFFF', '#BDBDBD'] + [rgba_to_hex(c) for c in hdb_colours]
    n_steps     = n_hdb + 2
    boundaries  = np.linspace(0, 1, n_steps + 1)
    colorscale  = []
    for i, c in enumerate(hex_colours):
        colorscale.append([boundaries[i], c])
        colorscale.append([boundaries[i + 1], c])
 
    def cluster_name(k):
        if k == -2: return 'Background'
        if k == -1: return 'Noise'
        if k == biggest_cluster: return f'Cluster {k} (bulk tissue)'
        return f'Cluster {k}'
 
    hover_text = np.vectorize(cluster_name)(hdb_map)
 
    fig = go.Figure(data=go.Heatmap(
        z=hdb_map,
        customdata=hover_text,
        colorscale=colorscale,
        zmin=-2, zmax=n_hdb - 1,
        showscale=False,
        hovertemplate='x: %{x}<br>y: %{y}<br>%{customdata}<extra></extra>',
    ))
 
    fig.update_layout(
        title=dict(
            text=f"HDBSCAN Cluster Map — interactive  "
                 f"(k={n_hdb}, {n_noise:,} noise pixels)",
            x=0.5, xanchor='center',
        ),
        width=1300, height=750,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False, scaleanchor='x', autorange='reversed'),
        plot_bgcolor='white',
        margin=dict(l=20, r=20, t=60, b=20),
    )
 
    if output_dir:
        fig.write_html(os.path.join(output_dir, "hdbscan_interactive.html"))
 
    fig.show()
 
 
# =============================================================================
# 9. K-MEANS VS HDBSCAN COMPARISON
# =============================================================================
 
def plot_kmeans_vs_hdbscan(cluster_labels: np.ndarray, hdb_labels: np.ndarray,
                            hdb_map: np.ndarray, cmap_hdb: ListedColormap,
                            tissue_indices_final: np.ndarray,
                            height: int, width: int,
                            output_dir: str = None, show_plot: bool = True):
    """
    Side-by-side spatial comparison of K-means and HDBSCAN cluster maps.
 
    This is a key thesis figure showing the difference between the two methods:
        K-means  — forces every pixel into a cluster, no noise label
        HDBSCAN  — identifies density peaks, labels low-density pixels as noise
 
    Parameters:
        cluster_labels       : K-means labels from run_kmeans()
        hdb_labels           : HDBSCAN labels from run_hdbscan()
        hdb_map              : HDBSCAN spatial map from plot_hdbscan_spatial()
        cmap_hdb             : HDBSCAN colormap from plot_hdbscan_spatial()
        tissue_indices_final : flat pixel indices
        height, width        : original image dimensions
        output_dir           : if provided, saves the figure here
        show_plot            : if True, displays the figure
    """
    n_km  = len(np.unique(cluster_labels))
    n_hdb = hdb_labels.max() + 1
    n_noise = (hdb_labels == -1).sum()
 
    # K-means spatial map
    kmeans_colours = ['#4C72B0', '#DD8452', '#55A868', '#C44E52',
                      '#8172B2', '#CCB974', '#64B5CD', '#937860']
    km_cmap = ListedColormap(['white'] + kmeans_colours[:n_km])
    km_map  = np.full((height, width), fill_value=-1, dtype=int)
    km_map.flat[tissue_indices_final] = cluster_labels.astype(int)
 
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
 
    axes[0].imshow(km_map, cmap=km_cmap, vmin=-1, vmax=n_km - 1, origin='upper')
    axes[0].set_title(f'K-means (k={n_km})\nForces every pixel into a cluster',
                      fontsize=13)
    axes[0].axis('off')
 
    axes[1].imshow(hdb_map, cmap=cmap_hdb, vmin=-2, vmax=n_hdb - 1, origin='upper')
    axes[1].set_title(f"HDBSCAN (k={n_hdb})\n"
                      f"Density-based — {n_noise:,} pixels labelled noise",
                      fontsize=13)
    axes[1].axis('off')
 
    fig.suptitle('K-means vs HDBSCAN — same UMAP embedding, different approaches',
                 fontsize=15, y=1.02)
    plt.tight_layout()
 
    if output_dir:
        plt.savefig(os.path.join(output_dir, "kmeans_vs_hdbscan.png"),
                    dpi=config.FIGURE_DPI, bbox_inches='tight')
    if show_plot:
        plt.show()
    plt.close()
 
 
# =============================================================================
# 10. ARI / AMI AGREEMENT METRICS
# =============================================================================
 
def compute_agreement(cluster_labels: np.ndarray, hdb_labels: np.ndarray):
    """
    Computes Adjusted Rand Index (ARI) and Adjusted Mutual Information (AMI)
    to measure how much K-means and HDBSCAN agree on the same data.
 
    HDBSCAN noise pixels (-1) are excluded from the comparison since K-means
    has no concept of noise — comparing them would be unfair.
 
    Interpretation:
        1.0  = identical partitioning (both methods agree completely)
        0.5+ = strong agreement — both methods see the same biological structure
        0.2  = weak agreement — methods disagree on cluster boundaries
        0.0  = random — no agreement at all
 
    A high ARI/AMI means the clusters are robust and not method-dependent.
    A low ARI/AMI means the two methods are capturing different structure,
    which is worth discussing in the thesis.
 
    Parameters:
        cluster_labels : K-means labels from run_kmeans()
        hdb_labels     : HDBSCAN labels from run_hdbscan()
    """
    # Only compare pixels that HDBSCAN didn't label as noise
    mask = hdb_labels != -1
    ari  = adjusted_rand_score(cluster_labels[mask], hdb_labels[mask])
    ami  = adjusted_mutual_info_score(cluster_labels[mask], hdb_labels[mask])
 
    print("K-means vs HDBSCAN Agreement (excluding HDBSCAN noise pixels)")
    print("=" * 60)
    print(f"  Adjusted Rand Index (ARI):         {ari:.3f}")
    print(f"  Adjusted Mutual Information (AMI): {ami:.3f}")
    print()
    print("  Interpretation:")
    print("    1.0  = identical partitioning")
    print("    0.5+ = strong agreement — both methods see the same biology")
    print("    0.2  = weak agreement — methods disagree on boundaries")
    print("    0.0  = random / no agreement")
 
    return ari, ami
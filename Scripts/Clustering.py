# =============================================================================
# Clustering.py
#
# Purpose:
#   Applies K-means clustering to the UMAP embedding and maps
#   the cluster labels back to their original spatial positions in the tissue.
#
# Order of steps:
#   1.  K-means elbow method (KneeLocator + silhouette score)
#   2.  Apply K-means with chosen k
#   3.  K-means spatial map (single clean figure)
#   4.  Per-cluster intensity heatmap (K-means) one figure, all channels
#   5.  Per-cluster elemental profiles (static bar chart)
# =============================================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import plotly.graph_objects as go
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
# ── GPU detection ─────────────────────────────────────────────────────────────
try:
    import cuml
    import config as _cfg
    _GPU = getattr(_cfg, 'USE_GPU', True)
    if _GPU:
        cuml.set_global_output_type('numpy')
        from cuml.cluster import KMeans
    else:
        from sklearn.cluster import KMeans
except ImportError:
    from sklearn.cluster import KMeans
    _GPU = False

from sklearn.metrics import silhouette_score

import config


# =============================================================================
# 1. K-means elbow
# =============================================================================

def run_kmeans_elbow(X_umap: np.ndarray, output_dir: str = None, show_plot: bool = True):
    """
    Tests K-means for k=1 to config.KMEANS_MAX_K and plots the elbow curve
    alongside the silhouette score, two independent criteria for choosing k.

    WCSS (Within-Cluster Sum of Squares): lower = tighter clusters.
    Silhouette score: ranges -1 to 1; higher = better-separated clusters.
    The optimal k balances a low WCSS elbow with a high silhouette peak.

    Silhouette is computed on a 10,000-pixel subsample for speed.

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
    wcss        = []
    silhouettes = []

    # Subsample for silhouette (expensive on large datasets)
    sil_n    = min(10_000, len(X_umap))
    rng      = np.random.default_rng(config.KMEANS_RANDOM_STATE)
    sil_idx  = rng.choice(len(X_umap), size=sil_n, replace=False)
    X_sil    = X_umap[sil_idx]

    print(f"Testing K-means for k=1 to {max_k}  "
          f"({'GPU' if _GPU else 'CPU'}, silhouette on {sil_n:,}-pixel subsample)...")

    for k in range(1, max_k + 1):
        km = KMeans(n_clusters=k, init='k-means++', max_iter=300,
                    n_init='auto', random_state=config.KMEANS_RANDOM_STATE)
        km.fit(X_umap)
        wcss.append(float(km.inertia_))

        if k >= 2:
            sil = silhouette_score(X_sil, np.array(km.predict(X_sil), dtype=np.int32))
            silhouettes.append(sil)
            print(f"  k={k}: WCSS = {km.inertia_:.2f}   silhouette = {sil:.4f}")
        else:
            silhouettes.append(np.nan)
            print(f"  k={k}: WCSS = {km.inertia_:.2f}   silhouette = N/A")

    # Auto-detect the elbow point from WCSS
    kl        = KneeLocator(range(1, max_k + 1), wcss, curve='convex', direction='decreasing')
    optimal_k = kl.elbow

    # Best silhouette k (k >= 2 only)
    valid_sil = [(k + 1, s) for k, s in enumerate(silhouettes) if not np.isnan(s)]
    best_sil_k = max(valid_sil, key=lambda x: x[1])[0] if valid_sil else None

    print(f"\nKneeLocator (WCSS elbow) suggests k = {optimal_k}")
    print(f"Silhouette peak suggests          k = {best_sil_k}")

    # Plot: dual-axis 
    fig, ax1 = plt.subplots(figsize=(11, 5))

    colour_wcss = '#2166AC'
    colour_sil  = '#D6604D'

    ax1.plot(range(1, max_k + 1), wcss, marker='o', linestyle='--',
             color=colour_wcss, label='WCSS')
    ax1.set_xlabel('Number of Clusters (k)', fontsize=13)
    ax1.set_ylabel('WCSS', fontsize=13, color=colour_wcss)
    ax1.tick_params(axis='y', labelcolor=colour_wcss)
    ax1.set_xticks(range(1, max_k + 1))
    ax1.grid(True, alpha=0.3)

    if optimal_k is not None:
        ax1.axvline(optimal_k, color=colour_wcss, linestyle=':',
                    alpha=0.7, label=f'WCSS elbow (k={optimal_k})')

    ax2 = ax1.twinx()
    sil_ks = [k for k in range(2, max_k + 1)]
    sil_vals = [s for s in silhouettes[1:] if not np.isnan(s)]
    ax2.plot(sil_ks, sil_vals, marker='s', linestyle='-',
             color=colour_sil, label='Silhouette')
    ax2.set_ylabel('Silhouette Score', fontsize=13, color=colour_sil)
    ax2.tick_params(axis='y', labelcolor=colour_sil)

    if best_sil_k is not None:
        ax2.axvline(best_sil_k, color=colour_sil, linestyle=':',
                    alpha=0.7, label=f'Silhouette peak (k={best_sil_k})')

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=10)

    plt.title("K-means Elbow + Silhouette Score: UMAP 3D", fontsize=15)
    plt.tight_layout()

    if output_dir:
        plt.savefig(os.path.join(output_dir, "kmeans_elbow.png"),
                    dpi=config.FIGURE_DPI, bbox_inches='tight')
    if show_plot:
        plt.show()
    plt.close()

    return optimal_k, wcss


# =============================================================================
# 2. Applying K-means
# =============================================================================

def run_kmeans(X_umap: np.ndarray, df: pd.DataFrame, chosen_k: int = None):
    """
    Applies K-means clustering to the UMAP coordinates.

    If chosen_k is not provided, falls back to config.KMEANS_MAX_K.
    K-means runs on X_umap (the 3D UMAP coordinates), not on the raw pixel data.

    Parameters:
        X_umap    : UMAP coordinates, shape (n_pixels, 3)
        df        : pixel dataframe (cluster labels will be added as a column)
        chosen_k  : number of clusters to use

    Returns:
        cluster_labels : integer array of cluster IDs, shape (n_pixels,)
        df             : updated dataframe with 'cluster' column added
    """
    k = chosen_k if chosen_k is not None else config.KMEANS_MAX_K

    print(f"Applying K-means with k={k} on UMAP coordinates ({'GPU' if _GPU else 'CPU'})...")
    km = KMeans(n_clusters=k, init='k-means++', max_iter=300,
                n_init='auto', random_state=config.KMEANS_RANDOM_STATE)
    cluster_labels = np.array(km.fit_predict(X_umap), dtype=np.int32)

    df['cluster'] = cluster_labels.astype(str)

    print(f"K-means complete using k={k}.")
    print("\n--- Cluster Distribution ---")
    counts = df['cluster'].value_counts().sort_index()
    for cid, count in counts.items():
        pct = (count / len(df)) * 100
        print(f"  Cluster {cid}: {count:,} pixels ({pct:.1f}%)")

    return cluster_labels, df


# =============================================================================
# 3.K-means spatial map
# =============================================================================

def plot_kmeans_spatial(cluster_labels: np.ndarray, tissue_indices_final: np.ndarray,
                         img_filtered: np.ndarray, channel_names_filtered: list,
                         threshold_lookup: dict, height: int, width: int,
                         output_dir: str = None, show_plot: bool = True):
    """
    Maps K-means cluster labels back to their spatial positions and saves
    one clean figure.  No per-channel loop,  use plot_cluster_intensity_heatmap
    for the biological interpretation of each cluster.

    Parameters:
        cluster_labels        : K-means labels, shape (n_pixels,)
        tissue_indices_final  : flat pixel indices aligned to cluster_labels
        img_filtered          : filtered image array (n_channels, H, W)  [unused here]
        channel_names_filtered: list of channel name strings              [unused here]
        threshold_lookup      : dict of {channel_name: (min, max)}        [unused here]
        height, width         : original image dimensions
        output_dir            : if provided, saves the figure here
        show_plot             : if True, displays the figure
    """
    n_clusters = len(np.unique(cluster_labels))

    cluster_map = np.full((height, width), fill_value=-1, dtype=int)
    cluster_map.flat[tissue_indices_final] = cluster_labels.astype(int)

    print(f"Mapping {n_clusters} K-means clusters to {height}x{width} spatial grid.")

    # Colour palette central palette from config so every figure matches
    cluster_colours = config.get_cluster_colours(n_clusters)
    cmap = ListedColormap(['white'] + cluster_colours)

    legend_elements = [Patch(facecolor='white', edgecolor='grey', label='Background')]
    for i in range(n_clusters):
        legend_elements.append(Patch(facecolor=cluster_colours[i], label=f'Cluster {i}'))

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(cluster_map, cmap=cmap, vmin=-1, vmax=n_clusters - 1,
              origin='upper', interpolation='nearest')
    ax.set_title(f'K-Means Cluster Map  (k={n_clusters})', fontsize=14)
    ax.axis('off')
    ax.legend(handles=legend_elements, loc='upper right',
              bbox_to_anchor=(1.22, 1), frameon=True, framealpha=0.9, fontsize=10)
    plt.tight_layout()

    if output_dir:
        plt.savefig(os.path.join(output_dir, "kmeans_spatial.png"),
                    dpi=config.FIGURE_DPI, bbox_inches='tight')
    if show_plot:
        plt.show()
    plt.close()


# =============================================================================
# HDBSCAN  
# =============================================================================

def run_hdbscan(X_umap: np.ndarray, output_dir: str = None):
    """
    Applies HDBSCAN to the UMAP-reduced coordinates.

    Following McInnes et al. (UMAP authors), HDBSCAN is run on the UMAP
    embedding rather than the raw data. This resolves the curse of
    dimensionality that causes HDBSCAN to fail on high-dimensional inputs
    (in testing on raw data: 73% noise, 39 micro-clusters). Running on
    UMAP coordinates increases cluster density and dramatically reduces noise.

    Parameters:
        X_umap     : UMAP coordinates, shape (n_pixels, n_components)
        output_dir : if provided, saves hdbscan_labels.npy here

    Returns:
        labels : integer array cluster IDs, noise pixels = -1
    """
    min_cluster_size = config.HDBSCAN_MIN_CLUSTER_SIZE
    min_samples      = config.HDBSCAN_MIN_SAMPLES

    print(f"\nRunning HDBSCAN on UMAP coordinates ({'GPU' if _GPU else 'CPU'}) "
          f"(min_cluster_size={min_cluster_size}, min_samples={min_samples})...")

    X_f32 = X_umap.astype(np.float32)

    if _GPU:
        from cuml.cluster import hdbscan as _cuml_hdbscan
        clusterer = _cuml_hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            cluster_selection_method=config.HDBSCAN_CLUSTER_SELECTION,
        )
    else:
        import hdbscan as _hdbscan_cpu
        clusterer = _hdbscan_cpu.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            cluster_selection_method=config.HDBSCAN_CLUSTER_SELECTION,
            prediction_data=True,
        )

    labels        = np.array(clusterer.fit_predict(X_f32), dtype=np.int32)
    probabilities = np.array(clusterer.probabilities_, dtype=np.float32)

    # Re-label clusters by descending size (largest = 0) so label ordering is
    # consistent across runs. Noise pixels (-1) are left unchanged.
    core_ids = np.array(sorted(set(labels.tolist()) - {-1}))
    if len(core_ids) > 1:
        core_counts = np.array([(labels == cid).sum() for cid in core_ids])
        size_order  = core_ids[np.argsort(-core_counts)]   # largest first
        remap = {old: new for new, old in enumerate(size_order)}
        new_labels = labels.copy()
        for old, new in remap.items():
            new_labels[labels == old] = new
        labels = new_labels

    n_clusters  = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise     = int((labels == -1).sum())
    noise_pct   = n_noise / len(labels) * 100

    print(f"  Clusters found : {n_clusters}")
    print(f"  Noise pixels   : {n_noise:,}  ({noise_pct:.1f}%)")
    for cid in sorted(set(labels)):
        if cid == -1:
            continue
        n = int((labels == cid).sum())
        print(f"  Cluster {cid}: {n:,} pixels ({n/len(labels)*100:.1f}%)")

    if output_dir:
        np.save(os.path.join(output_dir, "hdbscan_labels.npy"), labels)
        np.save(os.path.join(output_dir, "hdbscan_probabilities.npy"), probabilities)
        _save_hdbscan_tree(clusterer, labels, output_dir)

    return labels, probabilities


def _save_hdbscan_tree(clusterer, labels, output_dir):
    """
    Saves the HDBSCAN condensed tree with selected clusters highlighted.

    CPU path  : uses clusterer.condensed_tree_.plot(select_clusters=True)
                 the canonical HDBSCAN diagnostic showing cluster stability
                  across density levels (λ). Selected clusters are coloured;
                  pruned/noise branches appear in grey.

    GPU path  : cuML does not expose condensed_tree_, but it does expose
                cluster_persistence_ (the per-cluster stability score that
                would normally be read off the condensed tree). Falls back
                to a bar chart of those scores.

    Output    : clustering/hdbscan_condensed_tree.png
    """
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    palette    = config.get_cluster_colours(n_clusters)

    # ── CPU path: full condensed tree ─────────────────────────────────────────
    # cuML raises TypeError (not AttributeError) when accessing condensed_tree_,
    # so we catch Exception broadly here.
    try:
        condensed = clusterer.condensed_tree_
        try:
            import seaborn as sns
            sel_palette = sns.color_palette('tab10', n_clusters)
        except ImportError:
            sel_palette = [tuple(int(h[i:i+2], 16)/255 for i in (1, 3, 5))
                           for h in palette[:n_clusters]]

        fig, ax = plt.subplots(
            figsize=(max(10, n_clusters * 1.5), 6), facecolor='white')
        condensed.plot(
            select_clusters=True,
            selection_palette=sel_palette,
            axis=ax,
        )
        ax.set_title(
            'HDBSCAN Condensed Tree: selected Clusters Highlighted\n'
            'Width ∝ cluster size  ·  y-axis = λ (1/distance, higher = denser)',
            fontsize=11, fontweight='bold', pad=10)
        ax.set_xlabel('Data Points', fontsize=10)
        ax.set_ylabel('λ  (1 / distance)', fontsize=10)
        ax.spines[['top', 'right']].set_visible(False)
        plt.tight_layout()
        path = os.path.join(output_dir, 'hdbscan_condensed_tree.png')
        fig.savefig(path, dpi=config.FIGURE_DPI, bbox_inches='tight')
        plt.close()
        print(f"  Saved HDBSCAN condensed tree → {path}")
        return
    except Exception:
        pass  # cuML raises TypeError/AttributeError fall through to persistence bar chart

    # ── GPU fallback: cluster persistence bar chart ────────────────────────────
    try:
        persistence = np.array(clusterer.cluster_persistence_, dtype=float)
        cluster_ids = np.arange(len(persistence))

        fig, ax = plt.subplots(
            figsize=(max(8, n_clusters * 1.0), 5), facecolor='white')
        bars = ax.bar(
            [f'C{i}' for i in cluster_ids],
            persistence,
            color=palette[:n_clusters],
            edgecolor='white', linewidth=0.5, width=0.6,
        )
        for bar, val in zip(bars, persistence):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + persistence.max() * 0.01,
                    f'{val:.3f}', ha='center', va='bottom',
                    fontsize=9, fontweight='bold')
        ax.set_xlabel('Cluster', fontsize=11)
        ax.set_ylabel('Persistence (excess of mass)', fontsize=11)
        ax.set_title(
            'HDBSCAN Cluster Stability\n'
            '(GPU run condensed tree unavailable; showing per-cluster persistence scores)',
            fontsize=11, fontweight='bold')
        ax.set_ylim(0, persistence.max() * 1.18)
        ax.spines[['top', 'right']].set_visible(False)
        ax.set_facecolor('#f9f9f9')
        plt.tight_layout()
        path = os.path.join(output_dir, 'hdbscan_condensed_tree.png')
        fig.savefig(path, dpi=config.FIGURE_DPI, bbox_inches='tight')
        plt.close()
        print(f"  Saved HDBSCAN cluster stability (GPU fallback) → {path}")
    except AttributeError:
        print("  HDBSCAN condensed tree/persistence not available skipping.")


def plot_hdbscan_spatial(hdbscan_labels: np.ndarray,
                          tissue_indices_final: np.ndarray,
                          height: int, width: int,
                          output_dir: str = None, show_plot: bool = False):
    """
    Maps HDBSCAN labels back to spatial tissue positions.
    Noise pixels (label == -1) are shown in grey.
    """
    cluster_ids = sorted(set(hdbscan_labels) - {-1})
    n_clusters  = len(cluster_ids)

    cluster_map = np.full((height, width), fill_value=-2, dtype=int)
    cluster_map.flat[tissue_indices_final] = hdbscan_labels

    cluster_cols  = config.get_cluster_colours(n_clusters)
    cmap          = ListedColormap(['white', '#aaaaaa'] + cluster_cols)

    legend_elements = [
        Patch(facecolor='white',   edgecolor='grey', label='Background'),
        Patch(facecolor='#aaaaaa', label='Noise'),
    ]
    for i, cid in enumerate(cluster_ids):
        legend_elements.append(Patch(facecolor=cluster_cols[i], label=f'Cluster {cid}'))

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(cluster_map, cmap=cmap, vmin=-2, vmax=n_clusters - 1,
              origin='upper', interpolation='nearest')
    n_noise   = int((hdbscan_labels == -1).sum())
    noise_pct = n_noise / len(hdbscan_labels) * 100
    ax.set_title(f'HDBSCAN Cluster Map  '
                 f'(k={n_clusters}, noise={noise_pct:.1f}%)', fontsize=14)
    ax.axis('off')
    ax.legend(handles=legend_elements, loc='upper right',
              bbox_to_anchor=(1.22, 1), frameon=True, framealpha=0.9, fontsize=10)
    plt.tight_layout()

    if output_dir:
        plt.savefig(os.path.join(output_dir, "hdbscan_spatial.png"),
                    dpi=config.FIGURE_DPI, bbox_inches='tight')
    if show_plot:
        plt.show()
    plt.close()


# =============================================================================
# FIGURE: PER-CLUSTER ELEMENTAL PROFILES (static bar chart)
# =============================================================================

def plot_cluster_profiles(cluster_labels: np.ndarray, df_normalised: pd.DataFrame,
                           channel_names_filtered: list,
                           output_dir: str = None, show_plot: bool = False,
                           label: str = 'kmeans'):
    """
    Static matplotlib figure: one subplot per cluster, each showing a
    horizontal bar chart of log2 fold-change (log2FC) vs whole-tissue mean.

    log2FC = (cluster_mean_log1p - tissue_mean_log1p) / log(2)
    Positive values = enriched in this cluster vs the rest of tissue.
    Negative values = depleted. Bars sorted by log2FC descending.

    Parameters:
        cluster_labels        : integer cluster labels (HDBSCAN noise = -1, excluded)
        df_normalised         : log1p-normalised pixel dataframe
        channel_names_filtered: list of element names
        output_dir            : if provided, saves as '<label>_cluster_profiles.png'
        show_plot             : if True, opens the figure window
        label                 : 'kmeans' or 'hdbscan' used in title and filename
    """
    # Exclude HDBSCAN noise pixels (label == -1)
    valid_mask   = cluster_labels >= 0
    labels_valid = cluster_labels[valid_mask]
    df_valid     = df_normalised[channel_names_filtered].values[valid_mask].astype(np.float64)

    cluster_ids  = np.unique(labels_valid)
    n_clusters   = len(cluster_ids)
    method_name  = 'K-Means' if label == 'kmeans' else 'HDBSCAN'
    tissue_mean  = df_valid.mean(axis=0)   # (n_channels,) whole-tissue mean

    # Compute log2FC per cluster
    log2fc = {}
    counts = {}
    for cid in cluster_ids:
        m = labels_valid == cid
        cluster_mean  = df_valid[m].mean(axis=0)
        log2fc[cid]   = (cluster_mean - tissue_mean) / np.log(2)
        counts[cid]   = int(m.sum())

    # Layout: up to 4 columns
    ncols = min(4, n_clusters)
    nrows = int(np.ceil(n_clusters / ncols))
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(ncols * 4.5, nrows * max(3.5, len(channel_names_filtered) * 0.22)),
        facecolor='white', squeeze=False,
    )

    palette = config.get_cluster_colours(n_clusters)

    for i, cid in enumerate(cluster_ids):
        ax   = axes[i // ncols][i % ncols]
        fc   = log2fc[cid]

        # Sort by log2FC descending
        order     = np.argsort(fc)[::-1]
        sorted_ch = [channel_names_filtered[j] for j in order]
        sorted_fc = fc[order]

        # Green for enriched, red for depleted
        bar_colours = ['#27ae60' if v >= 0 else '#e74c3c' for v in sorted_fc]

        ax.barh(sorted_ch, sorted_fc, color=bar_colours,
                alpha=0.85, edgecolor='white', linewidth=0.4)
        ax.axvline(0, color='grey', linewidth=0.8, alpha=0.6)
        ax.set_title(f'Cluster {cid}  (n = {counts[cid]:,})', fontsize=9,
                     fontweight='bold', color=palette[i])
        ax.set_xlabel('Log₂ fold-change vs tissue mean', fontsize=8)
        ax.tick_params(axis='y', labelsize=7)
        ax.tick_params(axis='x', labelsize=7)
        ax.spines[['top', 'right']].set_visible(False)
        ax.invert_yaxis()

    for j in range(n_clusters, nrows * ncols):
        axes[j // ncols][j % ncols].set_visible(False)

    fig.suptitle(
        f'{method_name} Log₂ Fold-Change per Cluster vs Tissue Mean\n'
        f'Green = enriched  |  Red = depleted  |  HDBSCAN noise excluded',
        fontsize=12, fontweight='bold', y=1.01,
    )
    plt.tight_layout()

    if output_dir:
        path = os.path.join(output_dir, f'{label}_cluster_profiles.png')
        fig.savefig(path, dpi=config.FIGURE_DPI, bbox_inches='tight')
        print(f"  Saved → {path}")
    if show_plot:
        plt.show()
    plt.close()


# =============================================================================
# PRE-COMPUTED SUMMARY DATA  
# =============================================================================

def save_cluster_summary(cluster_labels: np.ndarray, flat_log: np.ndarray,
                         channel_names: list, output_dir: str, label: str = 'kmeans'):
    """
    Pre-computes and saves all per-cluster statistics the app needs.

    Saves:
      {label}_mean_profiles.npy   — (n_clusters, n_channels) mean log1p per cluster
      {label}_tissue_mean.npy     — (n_channels,) whole-tissue mean log1p
      {label}_log_fc.npy          — (n_clusters, n_channels) log2 fold-change vs tissue
      {label}_counts.json         — {str(cluster_id): pixel_count}
      {label}_dominant.json       — {str(cluster_id): dominant_element_name}

    For HDBSCAN, noise pixels (label == -1) are excluded from tissue_mean.
    """
    import json
    valid_mask = cluster_labels >= 0
    cluster_ids = sorted(np.unique(cluster_labels[valid_mask]).tolist())
    n_clusters  = len(cluster_ids)

    tissue_mean   = flat_log[valid_mask].mean(axis=0)
    mean_profiles = np.zeros((n_clusters, flat_log.shape[1]), dtype=np.float64)
    counts        = {}
    dominant      = {}

    for i, cid in enumerate(cluster_ids):
        mask = cluster_labels == cid
        mean_profiles[i] = flat_log[mask].mean(axis=0)
        counts[str(cid)]   = int(mask.sum())
        dominant[str(cid)] = channel_names[int(np.argmax(mean_profiles[i]))]

    log_fc = (mean_profiles - tissue_mean[np.newaxis, :]) / np.log(2)

    np.save(os.path.join(output_dir, f'{label}_mean_profiles.npy'), mean_profiles)
    np.save(os.path.join(output_dir, f'{label}_tissue_mean.npy'),   tissue_mean)
    np.save(os.path.join(output_dir, f'{label}_log_fc.npy'),        log_fc)

    with open(os.path.join(output_dir, f'{label}_counts.json'), 'w') as f:
        json.dump(counts, f)
    with open(os.path.join(output_dir, f'{label}_dominant.json'), 'w') as f:
        json.dump(dominant, f)

    # Also save the cluster_ids list so the app knows the ordering
    with open(os.path.join(output_dir, f'{label}_cluster_ids.json'), 'w') as f:
        json.dump(cluster_ids, f)

    print(f"  Saved cluster summary data for {label} ({n_clusters} clusters)")
    return mean_profiles, tissue_mean, log_fc, counts, dominant, cluster_ids


def plot_per_cluster_charts(mean_profiles: np.ndarray, log_fc: np.ndarray,
                             cluster_ids: list, channel_names: list,
                             counts: dict, output_dir: str, label: str = 'kmeans',
                             top_n: int = 30):
    """
    Saves three PNG bar charts per cluster to output_dir:
      {label}_cluster{cid}_intensity.png  — mean log1p intensity (sorted desc)
      {label}_cluster{cid}_lfc.png        — log2 fold-change vs tissue (sorted desc)
      {label}_cluster{cid}_ioncount.png   — raw ion count (sorted desc)

    These are loaded directly by the app,  no computation at display time.
    """
    palette = config.get_cluster_colours(len(cluster_ids))

    for i, cid in enumerate(cluster_ids):
        cl_hex  = palette[i % len(palette)]
        profile = mean_profiles[i]
        fc      = log_fc[i]
        raw     = np.expm1(profile)
        n_pix   = counts.get(str(cid), counts.get(cid, 0))
        src_lbl = f"Cluster {cid}  (n = {n_pix:,} pixels)"

        # ── Mean intensity ────────────────────────────────────────────────
        order = np.argsort(profile)[::-1][:top_n]
        fig, ax = plt.subplots(figsize=(9, max(5, top_n * 0.22 + 1.5)), facecolor='white')
        ax.barh([channel_names[j] for j in order],
                [float(profile[j]) for j in order],
                color=cl_hex, edgecolor='none', height=0.75)
        ax.set_xlabel('Mean log1p intensity', fontsize=15)
        ax.set_title(f'{src_lbl}\nTop {top_n}  Mean log1p Intensity',
                     fontsize=15, fontweight='bold', color=cl_hex)
        ax.invert_yaxis()
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(axis='both', labelsize=13)
        plt.tight_layout()
        path = os.path.join(output_dir, f'{label}_cluster{cid}_intensity.png')
        fig.savefig(path, dpi=config.FIGURE_DPI, bbox_inches='tight')
        plt.close(fig)

        # ── Log2 fold-change ──────────────────────────────────────────────
        fc_order = np.argsort(fc)[::-1][:top_n]
        fc_vals  = [float(fc[j]) for j in fc_order]
        fc_cols  = ['#27ae60' if v >= 0 else '#e74c3c' for v in fc_vals]
        fig, ax = plt.subplots(figsize=(9, max(5, top_n * 0.22 + 1.5)), facecolor='white')
        ax.barh([channel_names[j] for j in fc_order], fc_vals,
                color=fc_cols, edgecolor='none', height=0.75)
        ax.axvline(0, color='grey', linewidth=0.8, alpha=0.6)
        ax.set_xlabel('Log₂ fold-change vs whole tissue', fontsize=15)
        ax.set_title(f'{src_lbl}\nTop {top_n} Log₂ Fold-Change vs Tissue',
                     fontsize=15, fontweight='bold', color=cl_hex)
        ax.invert_yaxis()
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(axis='both', labelsize=13)
        plt.tight_layout()
        path = os.path.join(output_dir, f'{label}_cluster{cid}_lfc.png')
        fig.savefig(path, dpi=config.FIGURE_DPI, bbox_inches='tight')
        plt.close(fig)

        # ── Ion count ─────────────────────────────────────────────────────
        ic_order = np.argsort(raw)[::-1][:top_n]
        ic_vals  = [float(raw[j]) for j in ic_order]
        ic_cols  = ['#27ae60' if fc[j] >= 0 else '#e74c3c' for j in ic_order]
        fig, ax = plt.subplots(figsize=(9, max(5, top_n * 0.22 + 1.5)), facecolor='white')
        ax.barh([channel_names[j] for j in ic_order], ic_vals,
                color=ic_cols, edgecolor='none', height=0.75)
        ax.set_xscale('log')
        ax.set_xlabel('Mean ion count (ions/pixel, log scale)', fontsize=15)
        ax.set_title(f'{src_lbl}\nTop {top_n} Ion Count (green = enriched vs tissue)',
                     fontsize=15, fontweight='bold', color=cl_hex)
        ax.invert_yaxis()
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(axis='both', labelsize=13)
        plt.tight_layout()
        path = os.path.join(output_dir, f'{label}_cluster{cid}_ioncount.png')
        fig.savefig(path, dpi=config.FIGURE_DPI, bbox_inches='tight')
        plt.close(fig)

    print(f"  Saved per-cluster charts for {label} ({len(cluster_ids)} clusters × 3 charts)")


# =============================================================================
# COSINE DISTANCE HEATMAP  
# =============================================================================

def plot_cosine_distance_heatmap(mean_profiles: np.ndarray, cluster_ids: list,
                                  output_dir: str = None, label: str = 'kmeans'):
    """
    Computes pairwise cosine distances between cluster mean spectra and
    visualises the result as an annotated heatmap.

    Each cluster is summarised by its mean log1p intensity vector across all
    channels (from save_cluster_summary). Cosine distance = 1 − cosine similarity,
    so 0 means the two clusters have identical spectral profiles and 1 means
    they are orthogonal (maximally dissimilar).

    Parameters:
        mean_profiles : (n_clusters, n_channels) mean log1p intensity per cluster
        cluster_ids   : ordered list of cluster IDs (matches rows of mean_profiles)
        output_dir    : if provided, saves as '{label}_cosine_distance.png'
        label         : 'kmeans' or 'hdbscan' used in title and filename
    """
    from scipy.spatial.distance import pdist, squareform

    dist_matrix = squareform(pdist(mean_profiles, metric='cosine'))
    n           = len(cluster_ids)
    tick_labels = [f'Cluster {cid}' for cid in cluster_ids]
    method_name = 'K-Means' if label == 'kmeans' else 'HDBSCAN'

    fig, ax = plt.subplots(figsize=(max(5, n + 1), max(4, n + 0.5)), facecolor='white')

    im = ax.imshow(dist_matrix, cmap='RdYlGn_r', vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, shrink=0.8,
                 label='Cosine distance  (0 = identical  ·  1 = orthogonal)')

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=10)
    ax.set_yticklabels(tick_labels, fontsize=10)

    for i in range(n):
        for j in range(n):
            val = dist_matrix[i, j]
            text_col = 'white' if val > 0.6 else 'black'
            ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                    fontsize=10, fontweight='bold', color=text_col)

    ax.set_title(
        f'{method_name} Pairwise Cosine Distance Between Cluster Mean Spectra\n'
        '0 = spectrally identical  ·  1 = spectrally orthogonal',
        fontsize=11, fontweight='bold', pad=12
    )
    ax.spines[['top', 'right', 'left', 'bottom']].set_visible(False)
    plt.tight_layout()

    if output_dir:
        path = os.path.join(output_dir, f'{label}_cosine_distance.png')
        fig.savefig(path, dpi=config.FIGURE_DPI, bbox_inches='tight')
        print(f"  Saved cosine distance heatmap → {path}")
    plt.close()


# =============================================================================
# HDBSCAN membership probability visualisation
# =============================================================================

def plot_hdbscan_membership_probability(
    probabilities: np.ndarray,
    hdb_labels: np.ndarray,
    tissue_indices_final: np.ndarray,
    height: int, width: int,
    output_dir: str = None, show_plot: bool = False,
):
    """
    Maps HDBSCAN membership probability back to tissue space and visualises it
    alongside per-cluster probability distributions.

    Each tissue pixel receives a probability in [0, 1] from HDBSCAN:
        1.0 → pixel sits in the dense core of its cluster (high confidence)
        0.0 → noise pixel, or pixel on the very edge of a cluster boundary
        0.0–1.0 → soft cluster boundary degree of certainty about assignment

    Layout:
        Left  — spatial map coloured by probability (green = high confidence)
        Right — violin/box plot showing probability distribution per cluster,
                so you can compare which clusters are tightly defined vs uncertain

    Saved as: hdbscan_membership_prob.png

    Parameters:
        probabilities        : per-pixel confidence array (n_pixels,) from hdbscan
        hdb_labels           : HDBSCAN cluster labels (n_pixels,); -1 = noise
        tissue_indices_final : flat pixel indices of tissue pixels
        height, width        : image spatial dimensions
        output_dir           : if provided, saves figure here
        show_plot            : if True, displays figure
    """
    import matplotlib.gridspec as _gridspec

    ys, xs = np.unravel_index(tissue_indices_final, (height, width))
    prob_grid = np.full((height, width), np.nan, dtype=np.float32)
    prob_grid[ys, xs] = probabilities.astype(np.float32)

    cluster_ids = sorted(c for c in np.unique(hdb_labels) if c != -1)
    n_clusters  = len(cluster_ids)
    palette     = config.get_cluster_colours(n_clusters)
    is_noise    = hdb_labels == -1

    fig = plt.figure(figsize=(16, 6), facecolor='white')
    gs  = _gridspec.GridSpec(1, 2, figure=fig, width_ratios=[2, 1.2], wspace=0.08)
    ax_map = fig.add_subplot(gs[0])
    ax_dist = fig.add_subplot(gs[1])

    # ── Left: spatial probability map ────────────────────────────────────────
    im = ax_map.imshow(prob_grid, cmap='RdYlGn', vmin=0, vmax=1,
                       origin='upper', interpolation='nearest')
    plt.colorbar(im, ax=ax_map, shrink=0.75, label='Membership probability',
                 orientation='vertical', pad=0.01)
    ax_map.set_title(
        'HDBSCAN Membership Probability: Tissue Map\n'
        'Green = high confidence (dense core)  |  Red = boundary / noise',
        fontsize=11, fontweight='bold'
    )
    ax_map.axis('off')

    # Right: per-cluster probability violin plot 
    data_per_cluster = []
    positions        = []
    colours          = []

    for i, cid in enumerate(cluster_ids):
        mask = hdb_labels == cid
        data_per_cluster.append(probabilities[mask])
        positions.append(i)
        colours.append(palette[i % len(palette)])

    # Add noise bar at the right
    if is_noise.any():
        data_per_cluster.append(probabilities[is_noise])
        positions.append(n_clusters)
        colours.append('#aaaaaa')
        x_labels = [f'C{c}' for c in cluster_ids] + ['Noise']
    else:
        x_labels = [f'C{c}' for c in cluster_ids]

    # Violin plot
    parts = ax_dist.violinplot(data_per_cluster, positions=positions,
                               showmedians=True, showextrema=True, widths=0.6)

    # Colour each violin
    for body, col in zip(parts['bodies'], colours):
        body.set_facecolor(col)
        body.set_alpha(0.75)
    parts['cmedians'].set_color('black')
    parts['cmedians'].set_linewidth(1.5)
    parts['cbars'].set_color('#555555')
    parts['cmins'].set_color('#555555')
    parts['cmaxes'].set_color('#555555')

    # Mean probability annotation per cluster
    for i, (cid_data, pos) in enumerate(zip(data_per_cluster, positions)):
        mean_p = float(np.mean(cid_data))
        ax_dist.text(pos, mean_p + 0.02, f'{mean_p:.2f}',
                     ha='center', va='bottom', fontsize=7.5, fontweight='bold',
                     color='#222222')

    ax_dist.set_xticks(positions)
    ax_dist.set_xticklabels(x_labels, fontsize=9)
    ax_dist.set_ylabel('Membership probability', fontsize=10)
    ax_dist.set_ylim(-0.05, 1.08)
    ax_dist.set_title('Per-Cluster Probability\nDistribution', fontsize=10, fontweight='bold')
    ax_dist.spines[['top', 'right']].set_visible(False)
    ax_dist.axhline(0.5, color='grey', lw=0.8, ls='--', alpha=0.5)

    # Summary stats annotation
    core_probs = probabilities[~is_noise]
    n_noise    = int(is_noise.sum())
    mean_core  = float(core_probs.mean()) if len(core_probs) > 0 else 0.0
    pct_high   = float((core_probs >= 0.8).sum() / len(core_probs) * 100) if len(core_probs) > 0 else 0.0
    ax_dist.text(0.02, 0.02,
                 f'Noise: {n_noise:,} px ({n_noise/len(hdb_labels)*100:.1f}%)\n'
                 f'Core mean prob: {mean_core:.3f}\n'
                 f'Core ≥ 0.8: {pct_high:.1f}%',
                 transform=ax_dist.transAxes, fontsize=8, va='bottom',
                 color='#444444',
                 bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#cccccc', alpha=0.8))

    plt.suptitle('HDBSCAN Cluster Membership Confidence', fontsize=13,
                 fontweight='bold', y=1.02)
    plt.tight_layout()

    if output_dir:
        path = os.path.join(output_dir, 'hdbscan_membership_prob.png')
        fig.savefig(path, dpi=config.FIGURE_DPI, bbox_inches='tight')
        print(f"  Saved membership probability → {path}")
    if show_plot:
        plt.show()
    plt.close()


# =============================================================================
# Full silhouette plot
# =============================================================================

def plot_silhouette_full(X_umap: np.ndarray, km_labels: np.ndarray,
                          output_dir: str = None, show_plot: bool = False,
                          label: str = 'kmeans'):
    """
    Full silhouette plot: each pixel's silhouette coefficient, grouped and
    sorted by cluster, displayed as a horizontal bar chart.

    The average silhouette score (shown in the elbow plot) collapses everything
    into one number. This plot shows the DISTRIBUTION within each cluster:
        - Wide positive bars,  tightly cohesive cluster, well-separated from neighbours
        - Bars crossing zero,  pixels that are closer to a neighbouring cluster
          than their own  potential misassignments or genuine tissue gradients

    For large datasets (>20,000 pixels) a random subsample is used so that the
    plot renders in seconds rather than minutes.

    Saved as: {label}_silhouette_full.png

    Parameters:
        X_umap     : UMAP coordinates used for clustering (n_pixels, 3)
        km_labels  : cluster labels (n_pixels,)
        output_dir : if provided, saves figure here
        show_plot  : if True, displays figure
        label      : 'kmeans', 'leiden', etc.  used for title and filename
    """
    from sklearn.metrics import silhouette_samples

    MAX_SIL = 20_000
    rng     = np.random.default_rng(42)
    if len(X_umap) > MAX_SIL:
        idx        = rng.choice(len(X_umap), MAX_SIL, replace=False)
        X_sub      = X_umap[idx]
        labels_sub = km_labels[idx]
        n_used     = MAX_SIL
    else:
        X_sub      = X_umap
        labels_sub = km_labels
        n_used     = len(X_umap)

    print(f"  Computing per-sample silhouette on {n_used:,} pixels...")
    sil_vals   = silhouette_samples(X_sub, labels_sub)
    cluster_ids = np.unique(labels_sub)
    n_clusters  = len(cluster_ids)
    palette     = config.get_cluster_colours(n_clusters)
    avg_sil     = float(sil_vals.mean())

    fig, ax = plt.subplots(figsize=(10, max(5, n_clusters * 1.6)), facecolor='white')

    y_lower = 10
    for i, cid in enumerate(cluster_ids):
        ith_sil   = np.sort(sil_vals[labels_sub == cid])
        size      = len(ith_sil)
        y_upper   = y_lower + size
        ax.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_sil,
                         facecolor=palette[i % len(palette)],
                         edgecolor='none', alpha=0.85)
        ax.text(-0.06, y_lower + 0.5 * size,
                f'C{cid}  (n={size:,})',
                ha='right', va='center', fontsize=8,
                color=palette[i % len(palette)], fontweight='bold')
        y_lower = y_upper + 10

    ax.axvline(avg_sil, color='red', linestyle='--', linewidth=1.5,
               label=f'Mean silhouette = {avg_sil:.3f}')
    ax.axvline(0, color='black', linewidth=0.8, alpha=0.5)
    ax.set_xlabel('Silhouette coefficient', fontsize=11)
    ax.set_yticks([])
    ax.set_xlim(-0.25, 1.0)
    ax.set_ylim(0, y_lower)
    ax.legend(fontsize=9, loc='lower right', frameon=True)
    ax.spines[['top', 'right', 'left']].set_visible(False)
    pretty = {'kmeans': 'K-Means', 'leiden': 'Leiden', 'hdbscan': 'HDBSCAN'}.get(label, label)
    ax.set_title(
        f'Silhouette Plot {pretty} (k={n_clusters})\n'
        f'{n_used:,} pixels subsampled | '
        'Width = cluster size | Positive = well-assigned | '
        'Crosses zero = potential misassignment',
        fontsize=11, fontweight='bold'
    )
    plt.tight_layout()

    if output_dir:
        path = os.path.join(output_dir, f'{label}_silhouette_full.png')
        fig.savefig(path, dpi=config.FIGURE_DPI, bbox_inches='tight')
        print(f"  Saved full silhouette plot → {path}")
    if show_plot:
        plt.show()
    plt.close()


# =============================================================================
# CLUSTER PROPORTIONS 
# =============================================================================

def plot_cluster_proportions(km_labels: np.ndarray, hdb_labels: np.ndarray,
                              output_dir: str = None, show_plot: bool = False):
    """
    Side-by-side bar charts showing what percentage of the total tissue area
    each cluster occupies  one panel for K-means, one for HDBSCAN.

    For HDBSCAN, noise pixels are shown as a separate 'Noise' bar so the
    reader can immediately see how much tissue was unclassified.

    This is typically the first quantitative result in an MSI clustering paper:
    it tells you whether the segmentation is balanced (clusters of similar size)
    or strongly asymmetric (one dominant tissue region, several small ones).

    Saved as: cluster_proportions.png

    Parameters:
        km_labels  : K-means labels (n_pixels,)
        hdb_labels : HDBSCAN labels (n_pixels,); -1 = noise
        output_dir : if provided, saves figure here
        show_plot  : if True, displays figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor='white')
    total_px   = len(km_labels)

    # ── K-means ───────────────────────────────────────────────────────────────
    km_ids     = np.unique(km_labels)
    km_palette = config.get_cluster_colours(len(km_ids))
    km_pcts    = [(km_labels == cid).sum() / total_px * 100 for cid in km_ids]
    km_labels_str = [f'C{cid}' for cid in km_ids]

    ax = axes[0]
    bars = ax.bar(km_labels_str, km_pcts, color=km_palette[:len(km_ids)],
                  edgecolor='white', linewidth=0.5, width=0.7)
    for bar, pct in zip(bars, km_pcts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f'{pct:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.set_xlabel('Cluster', fontsize=11)
    ax.set_ylabel('% of tissue pixels', fontsize=11)
    ax.set_title(f'K-Means Cluster Proportions\n({len(km_ids)} clusters, '
                 f'{total_px:,} tissue pixels)', fontsize=12, fontweight='bold')
    ax.set_ylim(0, max(km_pcts) * 1.18)
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_facecolor('#f9f9f9')

    # ── HDBSCAN ───────────────────────────────────────────────────────────────
    hdb_ids_all  = np.unique(hdb_labels)
    hdb_ids_core = [c for c in hdb_ids_all if c >= 0]
    hdb_palette  = config.get_cluster_colours(len(hdb_ids_core))

    all_ids    = hdb_ids_core + ([-1] if -1 in hdb_ids_all else [])
    all_pcts   = [(hdb_labels == cid).sum() / total_px * 100 for cid in all_ids]
    all_labels = [f'C{cid}' for cid in hdb_ids_core] + (['Noise'] if -1 in hdb_ids_all else [])
    all_cols   = hdb_palette[:len(hdb_ids_core)] + (['#aaaaaa'] if -1 in hdb_ids_all else [])

    ax2 = axes[1]
    bars2 = ax2.bar(all_labels, all_pcts, color=all_cols,
                    edgecolor='white', linewidth=0.5, width=0.7)
    for bar, pct in zip(bars2, all_pcts):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                 f'{pct:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax2.set_xlabel('Cluster', fontsize=11)
    ax2.set_ylabel('% of tissue pixels', fontsize=11)
    noise_pct = (hdb_labels == -1).sum() / total_px * 100
    ax2.set_title(f'HDBSCAN Cluster Proportions\n({len(hdb_ids_core)} clusters + '
                  f'{noise_pct:.1f}% noise, {total_px:,} tissue pixels)',
                  fontsize=12, fontweight='bold')
    ax2.set_ylim(0, max(all_pcts) * 1.18)
    ax2.spines[['top', 'right']].set_visible(False)
    ax2.set_facecolor('#f9f9f9')

    plt.suptitle(
        'Cluster Tissue Area Proportions K-Means vs HDBSCAN',
        fontsize=12, fontweight='bold', y=1.03
    )
    plt.tight_layout()

    if output_dir:
        path = os.path.join(output_dir, 'cluster_proportions.png')
        fig.savefig(path, dpi=config.FIGURE_DPI, bbox_inches='tight')
        print(f"  Saved cluster proportions → {path}")

    if show_plot:
        plt.show()
    plt.close()



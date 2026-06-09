# =============================================================================
# pca_analysis.py
#
# PURPOSE:
#   Runs PCA on the normalised pixel data and produces the outputs used
#   in the thesis and pipeline. PCA here is used for two things:
#     1. Understanding which elements drive variation in the tissue
#        (via loading plots — the primary scientific output)
#     2. Producing a spatial RGB map of the tissue coloured by PCA coordinates
#
# IMPORTANT — what PCA is NOT used for here:
#   - No scatter plots of pixels in PCA space
#   - No K-means clustering on PCA components
#   These were used during exploratory analysis but are not part of the
#   final pipeline. UMAP handles the clustering and visualisation.
#
# ORDER OF STEPS:
#   1. Fit PCA on the log1p-normalised pixel dataframe
#   2. Scree plot — how much variance does each component explain?
#   3. Cumulative variance plot — how many components do we need?
#   4. Covariance matrix — how do the elements relate to each other?
#   5. Loading matrix — which elements drive each principal component?
#   6. 2D loading plot — arrows showing element contributions to two PCs
#   7. PCA RGB spatial image — tissue coloured by PC1/PC2/PC3
#   8. Return X_pca for use as input to UMAP
# =============================================================================
 
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
import plotly.express as px
from sklearn.decomposition import PCA
 
import config
 
 
# =============================================================================
# 1. FIT PCA
# =============================================================================
 
def run_pca(df_normalised: pd.DataFrame):
    """
    Fits PCA on the log1p-normalised pixel dataframe.
 
    PCA (Principal Component Analysis) finds new axes (principal components)
    that capture the most variation in the data. PC1 captures the most variation,
    PC2 the second most, and so on.
 
    For metallomic imaging data, each principal component typically represents
    a combination of elements that co-vary across the tissue. The loading plots
    (later in this script) tell you which elements make up each component.
 
    No StandardScaler is applied here — log1p normalisation in preprocessing.py
    is sufficient to bring the channels onto a comparable scale.
 
    Parameters:
        df_normalised : log1p-normalised pixel dataframe from preprocessing.apply_log1p()
                        shape (n_tissue_pixels, n_channels)
 
    Returns:
        pca   : fitted sklearn PCA object (contains loadings, explained variance etc.)
        X_pca : numpy array of PCA coordinates, shape (n_tissue_pixels, n_components)
    """
    # Use all available components (one per channel)
    # This gives us the full picture — we can always decide how many to use later
    n_components = df_normalised.shape[1]
 
    pca = PCA(n_components=n_components)
 
    # fit_transform does two things at once:
    #   fit()      — learns the principal components from the data
    #   transform() — projects every pixel onto those components
    X_pca = pca.fit_transform(df_normalised.values)
 
    # Print how much variance each component explains
    print("Variance explained per component:")
    cumulative = 0
    for i, var in enumerate(pca.explained_variance_ratio_):
        cumulative += var * 100
        print(f"  PC{i+1}: {var*100:.1f}%  |  Cumulative: {cumulative:.1f}%")
 
    return pca, X_pca
 
 
# =============================================================================
# 2. SCREE PLOT
# =============================================================================
 
def plot_scree(pca: PCA, output_dir: str = None, show_plot: bool = True):
    """
    Plots the scree plot — the explained variance ratio for each principal component.
 
    The scree plot helps you identify the 'elbow' point: the number of components
    after which adding more components stops capturing meaningful variance.
    Components after the elbow are likely capturing noise rather than real biology.
 
    For example, if the elbow is at PC4, then PC1–PC4 capture the biologically
    meaningful variation and the rest can be ignored.
 
    Parameters:
        pca        : fitted PCA object from run_pca()
        output_dir : if provided, saves the figure here
        show_plot  : if True, displays the figure
    """
    n = len(pca.explained_variance_ratio_)
    x = range(1, n + 1)
 
    plt.figure(figsize=(10, 6))
 
    # Dots for each component
    plt.scatter(x, pca.explained_variance_ratio_,
                s=200, alpha=0.75, c='orange', edgecolor='k', label='Component variance')
 
    # Dashed line connecting the dots — helps see the elbow
    plt.plot(x, pca.explained_variance_ratio_,
             c='orange', linestyle='--', alpha=0.5)
 
    plt.grid(True)
    plt.title("Explained Variance Ratio — Scree Plot", fontsize=20)
    plt.xlabel("Principal Component", fontsize=14)
    plt.ylabel("Proportion of Variance Explained", fontsize=14)
    plt.xticks(x, fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
 
    if output_dir:
        plt.savefig(os.path.join(output_dir, "pca_scree.png"),
                    dpi=config.FIGURE_DPI, bbox_inches='tight')
    if show_plot:
        plt.show()
    plt.close()
 
 
# =============================================================================
# 3. CUMULATIVE VARIANCE PLOT
# =============================================================================
 
def plot_cumulative_variance(pca: PCA, output_dir: str = None):
    """
    Interactive Plotly plot of cumulative explained variance vs number of components.
 
    Use this alongside the scree plot to decide how many PCA components to feed
    into UMAP. A common threshold is 90–95% cumulative variance.
 
    For example, if 4 components reach 92%, those 4 components contain most of
    the meaningful elemental variation in the tissue.
 
    Parameters:
        pca        : fitted PCA object from run_pca()
        output_dir : if provided, saves as HTML (interactive Plotly can't save as PNG directly)
    """
    # Cumulative sum of explained variance, converted to percentage
    cumul = np.cumsum(pca.explained_variance_ratio_) * 100
    n = len(cumul)
 
    fig = px.area(
        x=range(1, n + 1),
        y=cumul,
        labels={"x": "Number of Principal Components",
                "y": "Cumulative Explained Variance (%)"},
        title="Cumulative Explained Variance by PCA Components"
    )
    fig.update_traces(mode='lines+markers')
    fig.update_layout(yaxis_range=[0, 105])
 
    if output_dir:
        fig.write_html(os.path.join(output_dir, "pca_cumulative_variance.html"))
 
    fig.show()
 
 
# =============================================================================
# 4. COVARIANCE MATRIX
# =============================================================================
 
def plot_covariance_matrix(df_normalised: pd.DataFrame, channel_names_filtered: list,
                            output_dir: str = None, show_plot: bool = True):
    """
    Visualises the covariance matrix as a heatmap.
 
    The covariance matrix shows how every pair of elements relates to each other
    across all tissue pixels. This is what PCA uses internally to find its
    principal components.
 
    How to read it:
        Red  (+1) = the two elements increase together in the same pixels
                    (e.g. Na and K both high in the same tissue region)
        Blue (-1) = when one element is high, the other tends to be low
        White (0) = no relationship between the two elements
        Diagonal  = always 1.0 (every element perfectly correlates with itself)
 
    This plot is useful for the thesis to show which elements are co-localised
    in the tissue before PCA has been run.
 
    Parameters:
        df_normalised          : log1p-normalised pixel dataframe
        channel_names_filtered : list of channel name strings
        output_dir             : if provided, saves the figure here
        show_plot              : if True, displays the figure
    """
    # Compute the covariance matrix from the normalised data
    cov_matrix = np.cov(df_normalised.values, rowvar=False)
    cov_df = pd.DataFrame(cov_matrix,
                           index=channel_names_filtered,
                           columns=channel_names_filtered)
 
    plt.figure(figsize=(12, 10))
    im = plt.imshow(cov_df, cmap='coolwarm', vmin=-1, vmax=1)
    plt.colorbar(im, label='Covariance')
 
    # Label axes with channel names
    plt.xticks(range(len(channel_names_filtered)), channel_names_filtered,
               rotation=45, ha='right', fontsize=10)
    plt.yticks(range(len(channel_names_filtered)), channel_names_filtered,
               rotation=45, ha='right', fontsize=10)
 
    # Write the actual number inside each cell
    for row in range(len(channel_names_filtered)):
        for col in range(len(channel_names_filtered)):
            plt.text(col, row, f"{cov_df.iloc[row, col]:.2f}",
                     ha='center', va='center', fontsize=7, color='black')
 
    plt.title("Covariance Matrix of Normalised Channel Intensities", fontsize=16)
    plt.tight_layout()
 
    if output_dir:
        plt.savefig(os.path.join(output_dir, "pca_covariance_matrix.png"),
                    dpi=config.FIGURE_DPI, bbox_inches='tight')
    if show_plot:
        plt.show()
    plt.close()
 
 
# =============================================================================
# 5. LOADING MATRIX (grid of all PC combinations)
# =============================================================================
 
def plot_loading_matrix(pca: PCA, channel_names_filtered: list,
                         num_components: int = 4,
                         output_dir: str = None, show_plot: bool = True):
    """
    Plots a grid of loading plots — one panel for every combination of PCs.
 
    This is the primary scientific output of PCA in this pipeline.
    Each panel shows which elements drive a particular pair of principal components.
 
    How to read each panel:
        - Each arrow represents one element
        - The direction of the arrow shows how that element contributes to
          the two PCs on the axes
        - The length of the arrow shows the strength of that contribution
        - Arrows pointing in similar directions = those elements are correlated
        - Arrows pointing in opposite directions = those elements are inversely related
        - Long arrows = strong influence, short arrows = weak influence
 
    For the thesis: the loading matrix for RCC data will show which elements
    (e.g. Fe, Zn, Cu) are driving the main axes of variation in tumour vs.
    healthy tissue.
 
    Parameters:
        pca                    : fitted PCA object from run_pca()
        channel_names_filtered : list of channel name strings
        num_components         : number of PCs to include in the grid (default 4)
        output_dir             : if provided, saves the figure here
        show_plot              : if True, displays the figure
    """
    loadings = pca.components_  # shape (n_components, n_channels)
 
    # Generate a distinct colour for each element
    colours = plt.cm.tab20(np.linspace(0, 1, len(channel_names_filtered)))
 
    fig, axes = plt.subplots(num_components, num_components,
                              figsize=(20, 20), sharex=True, sharey=True)
 
    for i in range(num_components):
        for j in range(num_components):
            ax = axes[i, j]
 
            # The diagonal (PC vs itself) is meaningless — hide it
            if i == j:
                ax.axis('off')
                continue
 
            xs = loadings[j]  # x-axis loadings (horizontal PC)
            ys = loadings[i]  # y-axis loadings (vertical PC)
 
            # Draw an arrow and dot for each element
            for k, name in enumerate(channel_names_filtered):
                colour = colours[k]
                # Arrow line from origin (0,0) to the loading point
                ax.plot([0, xs[k]], [0, ys[k]], color=colour,
                        alpha=0.8, zorder=2, linewidth=2)
                # Filled circle at the tip of the arrow
                ax.scatter(xs[k], ys[k], s=150, color=colour,
                           zorder=3, edgecolors='black')
 
            # Formatting
            ax.set_xlim(-1.1, 1.1)
            ax.set_ylim(-1.1, 1.1)
            ax.axhline(0, color='black', linestyle='--', alpha=0.5, zorder=1)
            ax.axvline(0, color='black', linestyle='--', alpha=0.5, zorder=1)
            ax.grid(True, alpha=0.3, zorder=0)
 
            # Axis labels only on the outer edges
            if i == num_components - 1:
                x_var = pca.explained_variance_ratio_[j] * 100
                ax.set_xlabel(f"PC{j+1}\n({x_var:.1f}%)", fontsize=14, weight='bold')
            if j == 0:
                y_var = pca.explained_variance_ratio_[i] * 100
                ax.set_ylabel(f"PC{i+1}\n({y_var:.1f}%)", fontsize=14, weight='bold')
 
    # Element legend on the right side
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label=name,
               markerfacecolor=colours[idx], markersize=14,
               markeredgecolor='black', markeredgewidth=1)
        for idx, name in enumerate(channel_names_filtered)
    ]
    fig.legend(handles=legend_elements,
               loc='center right',
               bbox_to_anchor=(0.98, 0.5),
               title="Element Key",
               fontsize=16,
               title_fontsize=18,
               frameon=True,
               fancybox=True,
               shadow=True,
               borderpad=1.2,
               labelspacing=1.0)
 
    plt.suptitle(f"PCA Loading Matrix (PC1 to PC{num_components})", fontsize=26, y=0.92)
    plt.tight_layout(rect=[0, 0, 0.85, 0.90])
 
    if output_dir:
        plt.savefig(os.path.join(output_dir, "pca_loading_matrix.png"),
                    dpi=config.FIGURE_DPI, bbox_inches='tight')
    if show_plot:
        plt.show()
    plt.close()
 
 
# =============================================================================
# 6. 2D LOADING PLOT (single PC pair)
# =============================================================================
 
def plot_loading_2d(pca: PCA, channel_names_filtered: list,
                    x_pc: int = 0, y_pc: int = 1,
                    output_dir: str = None, show_plot: bool = True):
    """
    Plots a single 2D loading plot for a chosen pair of principal components.
 
    This is a cleaner, publication-ready version of one panel from the loading matrix.
    Each element is shown as an arrow from the origin — the direction and length
    of the arrow shows how strongly that element contributes to each PC.
 
    Parameters:
        pca                    : fitted PCA object from run_pca()
        channel_names_filtered : list of channel name strings
        x_pc                   : index of the PC for the x-axis (0 = PC1, 1 = PC2 etc.)
        y_pc                   : index of the PC for the y-axis
        output_dir             : if provided, saves the figure here
        show_plot              : if True, displays the figure
    """
    loadings = pca.components_
    xs = loadings[x_pc]   # loading values for the x-axis PC
    ys = loadings[y_pc]   # loading values for the y-axis PC
 
    colours = plt.cm.tab20(np.linspace(0, 1, len(channel_names_filtered)))
 
    plt.figure(figsize=(10, 10))
 
    for i, name in enumerate(channel_names_filtered):
        colour = colours[i]
 
        # Scatter dot at the tip of the arrow
        plt.scatter(xs[i], ys[i], s=200, color=colour, label=name, zorder=3)
 
        # Arrow from origin to the loading position
        plt.arrow(0, 0, xs[i], ys[i],
                  color=colour,
                  head_width=0.02,
                  length_includes_head=True,
                  zorder=2)
 
    plt.grid(True, alpha=0.4)
    plt.xlim(-1.1, 1.1)
    plt.ylim(-1.1, 1.1)
    plt.axhline(0, color='black', linestyle='--', alpha=0.5)
    plt.axvline(0, color='black', linestyle='--', alpha=0.5)
 
    x_var = pca.explained_variance_ratio_[x_pc] * 100
    y_var = pca.explained_variance_ratio_[y_pc] * 100
    plt.xlabel(f"PC{x_pc+1} ({x_var:.1f}%)", fontsize=14)
    plt.ylabel(f"PC{y_pc+1} ({y_var:.1f}%)", fontsize=14)
    plt.title(f"PCA Loading Plot — PC{x_pc+1} vs PC{y_pc+1}", fontsize=18)
 
    # Legend outside the plot area
    plt.legend(bbox_to_anchor=(1.04, 0.5), loc="center left",
               borderaxespad=0, title="Element", fontsize=12, title_fontsize=13)
 
    plt.tight_layout()
 
    if output_dir:
        fname = f"pca_loading_PC{x_pc+1}_vs_PC{y_pc+1}.png"
        plt.savefig(os.path.join(output_dir, fname),
                    dpi=config.FIGURE_DPI, bbox_inches='tight')
    if show_plot:
        plt.show()
    plt.close()
 
 
# =============================================================================
# 7. PCA RGB SPATIAL IMAGE
# =============================================================================
 
def plot_pca_rgb(X_pca: np.ndarray, tissue_indices_final: np.ndarray,
                  height: int, width: int,
                  output_dir: str = None, show_plot: bool = True):
    """
    Creates a spatial RGB image where each tissue pixel's colour encodes its
    position in PCA space:
        Red channel   = PC1 score (normalised to 0–1)
        Green channel = PC2 score (normalised to 0–1)
        Blue channel  = PC3 score (normalised to 0–1)
 
    This gives a spatial overview of where different elemental patterns occur
    in the tissue. Pixels with similar colours have similar elemental profiles.
    Background pixels (not in the mask) are shown as white.
 
    The normalisation uses 2nd–98th percentile stretching (contrast stretching)
    to prevent a few extreme outlier pixels from dominating the colour range.
 
    Parameters:
        X_pca                : PCA coordinates, shape (n_tissue_pixels, n_components)
        tissue_indices_final : flat pixel indices of tissue pixels
        height, width        : original image dimensions
        output_dir           : if provided, saves the figure here
        show_plot            : if True, displays the figure
    """
    def _normalise_pc(values, lo_pct=2, hi_pct=98):
        """Stretches values to 0–1 range using percentile clipping."""
        lo, hi = np.percentile(values, lo_pct), np.percentile(values, hi_pct)
        if hi <= lo:
            return np.zeros_like(values)
        return np.clip((values - lo) / (hi - lo), 0, 1)
 
    # Normalise the first 3 PCs to 0-1 for RGB mapping
    pc1 = _normalise_pc(X_pca[:, 0])
    pc2 = _normalise_pc(X_pca[:, 1])
    pc3 = _normalise_pc(X_pca[:, 2])
 
    # Build the RGB image — start with white background (1, 1, 1)
    n_pixels = height * width
    rgb_flat = np.ones((n_pixels, 3), dtype=float)
 
    # Place the RGB values at the tissue pixel positions
    rgb_flat[tissue_indices_final, 0] = pc1  # Red   = PC1
    rgb_flat[tissue_indices_final, 1] = pc2  # Green = PC2
    rgb_flat[tissue_indices_final, 2] = pc3  # Blue  = PC3
 
    pca_rgb_image = rgb_flat.reshape(height, width, 3)
 
    plt.figure(figsize=(10, 10), facecolor='white')
    plt.imshow(pca_rgb_image, origin='upper')
    plt.title("PCA RGB Spatial Map  —  R=PC1  G=PC2  B=PC3", fontsize=14)
    plt.axis('off')
    plt.tight_layout()
 
    if output_dir:
        plt.savefig(os.path.join(output_dir, "pca_rgb_image.png"),
                    dpi=config.FIGURE_DPI, bbox_inches='tight')
    if show_plot:
        plt.show()
    plt.close()
 
    return pca_rgb_image
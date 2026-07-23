# =============================================================================
# config.py
# Central configuration for the metallomics dimensionality reduction pipeline.
# Change settings here, no need to touch the other scripts.
# =============================================================================

import os
import numpy as np


# =============================================================================
# PATHS
# =============================================================================

# Folder containing your OME-TIFF files
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "Data")

# Folder where all outputs (figures, CSVs) will be saved
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "Results")

# Path to your OME-TIFF 
TIFF_FILE = os.path.join(DATA_DIR, "IMC_whole_section.ome.tif")


# =============================================================================
# Channels
# =============================================================================

# Channels to drop before analysis.
# 0TIC (total ion count) must always be here, it's not a real element.
# Add any poor quality channels here too e.g. ["0TIC", "Mn", "K"]
CHANNELS_TO_DROP = ["0TIC", "89Y", "111Cd"]


# =============================================================================
# Tissue masking 
# =============================================================================
# Which channel to use to build the tissue mask.
# Index refers to position AFTER 0TIC is dropped (so 0 = first real element).
# 0=Na, 1=Mg, 2=P, 3=S, 4=K, 5=Ca, 6=Mn, 7=Fe, 8=Cu, 9=Zn, 10=Se
MASK_CHANNEL = 0

# Pixel intensity threshold: pixels below this value are treated as background.
# Increase if background noise is being included; decrease if tissue is being clipped.
MASK_THRESHOLD = 200000

# Minimum number of connected pixels to keep (removes small noise objects).
# e.g. 800 = any connected region smaller than 800 pixels is removed.
SMALL_OBJECT_REMOVAL = 50

# Whether to fill enclosed background holes inside the tissue mask.
# Set to True if blood vessels / internal holes should be included as tissue.
MASK_FILL_HOLES = True

# Fill holes in the tissue mask up to this area (in pixels).
# Only used if MASK_FILL_HOLES = True.
HOLE_AREA_THRESHOLD = 1000

# Number of binary erosion/dilation cycles to smooth mask edges.
# 0 = no smoothing. 2 is a good default for clean tissue edges.
BED_ITERATIONS = 1

# Set to 1 to display a 3-panel preview of the mask before continuing.
# Useful for checking that the mask looks correct before running the full pipeline.
SHOW_MASK_PREVIEW = 1

# Crop the image to a subregion before masking.
# Useful for quick testing on a smaller area without processing the full image.
# Set to None to use the full image (recommended for final runs).
# e.g. MASK_X_RANGE = [0, 1000] crops to the first 1000 columns
#      MASK_Y_RANGE = [0, 1000] crops to the first 1000 rows
MASK_X_RANGE = None
MASK_Y_RANGE = None


# =============================================================================
# Log1p Transfornation
# =============================================================================

# log1p = log(x + 1), applied element-wise.
# Standard for LA-ICP-MS ion count data (always >= 0).
# Safe for zeros: log(0 + 1) = 0.
NORMALISATION = 'log1p'


def inverse_normalisation(values: np.ndarray) -> np.ndarray:
    """
    Reverses log1p to recover approximate raw counts.
    Used in clustering visualisations for intensity-weighted centroids
    and log2FC calculations.
    """
    return np.expm1(values)


# =============================================================================
# PCA
# =============================================================================

# Number of PCA components to compute
# Set to None to use all available components (n_channels - 1)
N_PCA_COMPONENTS = None


# =============================================================================
# UMAP
# =============================================================================

# 3 components for RGB spatial map (UMAP V2 default)
UMAP_N_COMPONENTS = 3
UMAP_N_NEIGHBORS  = 30
UMAP_MIN_DIST     = 0.0
UMAP_METRIC       = "cosine"
UMAP_RANDOM_STATE = 42


# =============================================================================
# Clustering
# =============================================================================

# HDBSCAN
# Run on UMAP coordinates (not raw data), McInnes et al. recommendation.
# min_cluster_size scales with dataset size; 2000 suits ~300k tissue pixels.
# Increase for fewer, larger clusters; decrease for finer segmentation.
HDBSCAN_MIN_CLUSTER_SIZE  = 2000
HDBSCAN_MIN_SAMPLES       = 50
HDBSCAN_CLUSTER_SELECTION = "eom"   # 'eom' = fewer large clusters; 'leaf' = more fine-grained

# K-means 
KMEANS_MAX_K        = 10    # test k from 1 to this value for elbow plot
KMEANS_RANDOM_STATE = 42

# Force a specific K instead of using the elbow auto-detection.
# Set to None to use the elbow method (recommended first run).
# Set to an integer (e.g. 5, 6, 10) to override and use that K directly.
# Supervisor wanted this
KMEANS_FORCED_K     = None


# =============================================================================
# GPU
# =============================================================================

# Set to True to use GPU-accelerated PCA / UMAP / K-Means / HDBSCAN via cuML.
# Set to False (default) to use CPU implementations (sklearn, umap-learn, hdbscan).
# CPU mode is recommended for reproducibility, results match across machines.
# Note: when USE_GPU = True, UMAP connectivity plots are unavailable (cuML does
# not expose the graph_ attribute required by umap.plot.connectivity).
USE_GPU = False


# =============================================================================
# Figures 
# =============================================================================

FIGURE_DPI    = 300
FIGURE_FORMAT = "png"   # "png" or "pdf"

# Pixel size in micrometres (µm/pixel), used to draw scale bars on spatial maps.
# Check your LA-ICP-TOF-MS acquisition settings for the laser step size.
# Set to None to skip scale bars.
PIXEL_SIZE_UM = None   # e.g. 5.0 for a 5 µm step size


# =============================================================================
# Cluster colour palette
# =============================================================================
# Fixed colour list so every figure uses identical colours across all plots.
# Supports up to 20 clusters, extend if needed.

CLUSTER_COLOURS = [
    # Okabe-Ito palette (first 8), safe for deuteranopia, protanopia, tritanopia
    "#E69F00",  # 0  orange
    "#56B4E9",  # 1  sky blue
    "#009E73",  # 2  bluish green
    "#F0E442",  # 3  yellow
    "#0072B2",  # 4  blue
    "#D55E00",  # 5  vermillion
    "#CC79A7",  # 6  reddish purple
    "#000000",  # 7  black
    # Paul Tol's muted palette (extended set, still colourblind-friendly)
    "#332288",  # 8  indigo
    "#88CCEE",  # 9  cyan
    "#44AA99",  # 10 teal
    "#117733",  # 11 green
    "#999933",  # 12 olive
    "#DDCC77",  # 13 sand
    "#CC6677",  # 14 rose
    "#882255",  # 15 wine
    "#AA4499",  # 16 purple
    "#DDDDDD",  # 17 light grey
    "#771155",  # 18 dark purple
    "#48AFD0",  # 19 steel blue
]


def get_cluster_colours(n: int) -> list:
    """
    Returns a list of n hex colour strings for cluster labels 0 … n-1.
    Always the same colours in the same order so every figure matches.
    """
    if n > len(CLUSTER_COLOURS):
        import matplotlib.cm as _cm
        import matplotlib.colors as _mc
        cmap = _cm.get_cmap("tab20", n)
        return [_mc.to_hex(cmap(i)) for i in range(n)]
    return CLUSTER_COLOURS[:n]

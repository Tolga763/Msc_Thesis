# =============================================================================
# config.py
# This is the central configuration for the metallomics dimensionality reduction pipeline.
# The concept is that it acts as a control panel for all the scripts, you can adjust settings here so that they will be applied across the whole pipeline.
# This keeps all your parameters in one place and makes it easy to run multiple iterations with different settings by just changing this file.
# Each section is clearly labelled (e.g. "TISSUE MASKING", "PCA", "UMAP") with comments explaining what each parameter does and how to adjust it.
# =============================================================================
 
import os # Let's python work with file paths and folders without hardcoding absolute paths, making it more portable across different machines.
 
# =============================================================================
# PATHS
# =============================================================================
 
# This section defines all the file paths and directories used in the pipeline.
# Adjust these paths to point to your data and where you want outputs saved. The rest of the scripts will use these paths, so you only need to change them here.

# Folder containing your OME-TIFF files
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "Data")
 
# Folder where all outputs (figures, CSVs) will be saved
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "Outputs")
 
# Path to your metadata CSV (display ranges / threshold lookup)
METADATA_CSV = os.path.join(DATA_DIR, "meta_data.csv")
 
# Path to your OME-TIFF file
TIFF_FILE = os.path.join(DATA_DIR, "MURINE_EMBRYO.ome.tif")
 
# Root output folder, sub-folders per stage are created automatically
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "Results")
 
# =============================================================================
# CHANNELS
# =============================================================================
 
# This section defines which channels to include in the analysis and how to handle them.

# Channels to drop before analysis (total ion count — not a real element)
CHANNELS_TO_DROP = ["0TIC"]
 
# Channels to manually exclude due to poor quality (add names as needed)
# e.g. ['Mn', 'K'] — leave empty to keep all channels
CHANNELS_MANUAL_EXCLUDE = ['Mn', 'K']
 
# =============================================================================
# TISSUE MASKING
# =============================================================================

# This section controls how the tissue mask is generated to separate tissue from background.
 
# Which channel to use to build the tissue mask.
# Index refers to position AFTER 0TIC is dropped (so 0 = first real element).
# 0=Na, 1=Mg, 2=P, 3=S, 4=K, 5=Ca, 6=Mn, 7=Fe, 8=Cu, 9=Zn, 10=Se (for embryo dataset)
MASK_CHANNEL = 1
 
# Pixel intensity threshold — pixels below this value are treated as background.
# Increase if background noise is being included; decrease if tissue is being clipped.
MASK_THRESHOLD = 200
 
# Minimum number of connected pixels to keep (removes small noise objects).
# e.g. 800 = any connected region smaller than 800 pixels is removed.
SMALL_OBJECT_REMOVAL = 800

# Whether to fill enclosed background holes inside the tissue mask.
# Set to True if blood vessels / internal holes should be included as tissue.
MASK_FILL_HOLES = False

# Fill holes in the tissue mask up to this area (in pixels).
# Only used if MASK_FILL_HOLES = True.
HOLE_AREA_THRESHOLD = 10000

# Number of binary erosion/dilation cycles to smooth mask edges.
# 0 = no smoothing. 2 is a good default for clean tissue edges.
BED_ITERATIONS = 2

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
# NORMALISATION
# =============================================================================
 
# log1p applied to the data before PCA/UMAP (matches UMAP V2 pipeline)
# This is the only normalisation step — no z-scaling, no percentile clipping
USE_LOG1P = True
 
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
# 2 components for 2D visualisation (UMAP V1 default)
UMAP_N_COMPONENTS = 3
UMAP_N_NEIGHBORS  = 30
UMAP_MIN_DIST     = 0.0
UMAP_METRIC       = "cosine"
UMAP_RANDOM_STATE = 42
 
# =============================================================================
# tSNE
# =============================================================================
 
# tSNE is subsampled to keep it fast — set max number of pixels here
TSNE_MAX_PIXELS   = 50000
 
TSNE_N_COMPONENTS = 3       # set to 3 for RGB spatial visualisation
TSNE_PERPLEXITY   = 30
TSNE_MAX_ITER     = 1000
TSNE_METRIC       = "cosine"
TSNE_INIT         = "pca"
TSNE_METHOD       = "barnes_hut"
TSNE_RANDOM_STATE = 42
 
# =============================================================================
# CLUSTERING
# =============================================================================
 
# --- HDBSCAN ---
# 'leaf' splits density peaks more aggressively than 'eom'
HDBSCAN_MIN_CLUSTER_SIZE  = 200
HDBSCAN_MIN_SAMPLES       = 25
HDBSCAN_CLUSTER_SELECTION = "leaf"
 
# Parameter sweep ranges (used to find optimal HDBSCAN settings)
HDBSCAN_SWEEP_MIN_CLUSTER_SIZES = [50, 100, 200, 300, 500, 800]
HDBSCAN_SWEEP_MIN_SAMPLES       = [5, 25, 50]
 
# --- K-means (for comparison / thesis) ---
KMEANS_MAX_K        = 10    # test k from 1 to this value for elbow plot
KMEANS_RANDOM_STATE = 42
 
# =============================================================================
# GPU
# =============================================================================
 
# Set to True to attempt GPU-accelerated UMAP/HDBSCAN via cuML
# Falls back to CPU automatically if cuML is not installed
USE_GPU = True
 
# =============================================================================
# FIGURES
# =============================================================================
 
FIGURE_DPI    = 300 # High DPI for publication-quality figures
FIGURE_FORMAT = "png"   # "png" or "pdf"
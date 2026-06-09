# =============================================================================
# preprocessing.py
#
# PURPOSE:
#   This is the first script that runs in the pipeline.
#   It takes your raw OME-TIFF file and prepares the data for dimensionality
#   reduction (PCA, UMAP, tSNE). Nothing in this file does any DR, it just
#   cleans, masks, and organises the data so the other scripts can use it.
#
# ORDER OF STEPS:
#   1. Load the OME-TIFF image and extract channel names from its XML metadata
#   2. Drop unwanted channels (0TIC always, plus any manual exclusions already set from the config file)
#   3. Load the metadata CSV to get display ranges (min/max) per channel
#   4. Apply tissue masking to separate real tissue pixels from background
#   5. Build the pixel dataframe (df) and tissue_indices_final
#   6. Auto-suggest log vs linear display scale per channel for visualisation
#   7. Apply log1p normalisation ready for PCA/UMAP/tSNE
#   8. Visualise the element maps with custom colormaps (debating on whether to keep this)
# =============================================================================
 
import os       # File path manipulation without hardcoding absolute paths
import ast      # for safely parsing the metadata CSV string-encoded lists
import xml.etree.ElementTree as ET  # ET is created for reading the OME-XML metadata inside the TIFF
 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats                          # for skewness calculation in scale suggestion
from skimage.morphology import (
    remove_small_objects,   # removes tiny noise blobs from the mask
    remove_small_holes,     # fills enclosed background holes inside the tissue mask
    binary_erosion,         # shrinks mask edges inward (part of BED smoothing)
    binary_dilation,        # expands mask edges outward (part of BED smoothing)
    square,                 # defines the 3x3 structuring element used in BED
)
import tifffile  # reads OME-TIFF files produced by the LA-ICP-MS instrument

import config    # all settings and parameters live in config.py, we are importing it here
 
 
# =============================================================================
# 1. LOAD OME-TIFF
# =============================================================================
 
def load_image(file_path: str):
    """
    OME-TIFF is the file format produced by the LA-ICP-MS instrument.
    It stores a 3D array of shape (n_channels, height, width) where each
    channel is a 2D elemental map (e.g. one image for Fe, one for Zn, etc.)

    So loading an OME-TIFF file and extracts two things: 
    1.) The actual pixel data: a 3D array of shape (n_channels, height, width)
    2.) The channel names: a list of strings, one per channel

    The channel names are stored in XML metadata embedded inside the TIFF file.
    This function reads that XML and pulls out the names automatically, so you
    don't have to hardcode the names, making this method highly reproducible 
    and aplicable to any dataset.
 
    Returns:
        img_raw          : 3D numpy array, shape (n_channels, height, width)
        all_channel_names: list of channel name strings, one per channel
    """
    # Load the full image stack into memory as a numpy array
    img_raw = tifffile.imread(file_path)
 
    # Open the file again just to read the embedded OME-XML metadata string
    with tifffile.TiffFile(file_path) as tif:
        omexml_string = tif.ome_metadata
 
    if omexml_string:
        # Parse the XML and find all Channel elements
        # The namespace is required because OME-XML uses a specific XML schema
        namespaces = {'ome': 'http://www.openmicroscopy.org/Schemas/OME/2016-06'}
        root = ET.fromstring(omexml_string)
        channels = root.findall('.//ome:Channel', namespaces)
 
        # Extract the 'Name' attribute from each channel (e.g. 'Fe', 'Zn', '0TIC')
        # If a channel has no name, fall back to 'Channel_0', 'Channel_1', etc.
        all_channel_names = [c.attrib.get('Name', f"Channel_{i}") for i, c in enumerate(channels)]
    else:
        # No metadata found — generate placeholder names and warn the user
        print("Warning: No OME metadata found. Using placeholder channel names.")
        all_channel_names = [f"Channel_{i}" for i in range(img_raw.shape[0])]
 
    print(f"Original data shape: {img_raw.shape}")  # (n_channels, height, width)
    print(f"Data type: {img_raw.dtype}")
    print(f"Channels found: {all_channel_names}")
 
    return img_raw, all_channel_names
 
 
# =============================================================================
# 2. FILTER CHANNELS
# =============================================================================
 
def filter_channels(img_raw: np.ndarray, all_channel_names: list):
    """
    Removes unwanted channels from the image before analysis.
 
    Two types of channels are removed:
      - 0TIC (Total Ion Count): always dropped. This is a sum of all channels
        and does not represent a real element — including it would distort PCA/UMAP.
      - Manual exclusions: any channels listed in config.CHANNELS_MANUAL_EXCLUDE.
        Use this for channels with poor signal quality (e.g. very noisy Mn or Se).
 
    Also assigns a display colour to each remaining channel for visualisation.
 
    Returns:
        img                  : image after dropping 0TIC only (n_channels-1, H, W)
        img_filtered         : image after all exclusions (n_kept_channels, H, W)
        channel_names        : names after dropping 0TIC
        channel_names_filtered: names after all exclusions
        qupath_colours       : list of colour strings for each kept channel
    """
    # Step 1: Always remove channel at index 0 (0TIC — total ion count, not a real element)
    drop_indices = [i for i, 
                    name in enumerate(all_channel_names) 
                    if name in config.CHANNELS_TO_DROP]
    
    # The img line takes raw 3D image array (n_channels, H, W) and removes the channels at drop_indices
    # axis=0 means "remove the channel dimension", not the height or width. 
    # So if 0TIC was channel 0, the result is a new array with shape (n_channels-1, H, W) where the 0TIC channel has been removed.
    img = np.delete(img_raw, drop_indices, axis=0) 

    # Does the same thing but for the names list
    # Keeps only the names of channels whose index is NOT in drop_indices
    # So the names list stays in sync with the image array after dropping 0TIC
    # If you didn't do this, channel 0 in the image would no longer match channel 0 in the names list, causing confusion later on.
    channel_names = [name for i, name in enumerate(all_channel_names) if i not in drop_indices]
 
    # Step 2: Apply any manual exclusions defined in config.CHANNELS_MANUAL_EXCLUDE
    # e.g. if config has ['Mn', 'K'], those channels will be removed here
    manual_drop = config.CHANNELS_MANUAL_EXCLUDE
    good_mask = [name not in manual_drop for name in channel_names]
    channel_names_filtered = [name for name in channel_names if name not in manual_drop]
    img_filtered = img[good_mask]  # boolean index to keep only the good channels
 
    # Assign a unique display colour to each channel for visualisation
    # Uses a fixed list of colours, cycling if there are more channels than colours
    base_colours = ['lime', 'blue', 'yellow', 'cyan', 'magenta', 'orange',
                    'purple', 'pink', 'teal', 'lightgreen', 'gold', 'white']
    qupath_colours = [base_colours[i % len(base_colours)] for i in range(len(channel_names_filtered))]
 
    print(f"\nOriginal channels:                        {len(all_channel_names)}")
    print(f"After dropping 0TIC + manual exclusions:  {len(channel_names_filtered)}")
    print(f"Kept channels: {channel_names_filtered}")
    if manual_drop:
        print(f"Manually excluded: {manual_drop}")
 
    return img, img_filtered, channel_names, channel_names_filtered, qupath_colours
 
 
# =============================================================================
# 3. LOAD METADATA AND BUILD THRESHOLD LOOKUP
# =============================================================================
 
def load_metadata(metadata_path: str, channel_names_filtered: list):
    """
    Loads the metadata CSV exported from the LA-ICP-MS Balrog script 
    The metadata builds a lookup dictionary of (min, max) display ranges per channel.
 
    The metadata CSV contains three key rows:
      - 'processed isotopes': the channel names as recorded by the instrument
      - 'min thresholds': the recommended minimum display value per channel
      - 'max thresholds': the recommended maximum display value per channel
 
    Returns:
        threshold_lookup : dict of {channel_name: (min, max)}
        channel_ranges   : list of (min, max) tuples aligned to channel_names_filtered
    """
    meta_df = pd.read_csv(metadata_path, index_col=0)
 
    # The CSV stores these as string-encoded Python lists — ast.literal_eval converts them back
    all_min_thresholds    = ast.literal_eval(meta_df.loc['min thresholds', '0'])
    all_max_thresholds    = ast.literal_eval(meta_df.loc['max thresholds', '0'])
    all_channel_names_meta = ast.literal_eval(meta_df.loc['processed isotopes', '0'])
 
    # Build lookup dict mapping full isotope names to their display ranges
    # e.g. {'56Fe': (0, 80000), '23Na': (0, 5000), ...}
    # Full isotope notation is kept to avoid collisions if multiple isotopes
    # of the same element are present (e.g. 56Fe and 57Fe)
    threshold_lookup = {
        name: (all_min_thresholds[i], all_max_thresholds[i])
        for i, name in enumerate(all_channel_names_meta)
    }
 
    # Build a list of (min, max) ranges aligned to our filtered channel list
    # If a channel name can't be found in metadata, use (0, 1) as a safe fallback
    channel_ranges = []
    for name in channel_names_filtered:
        if name in threshold_lookup:
            channel_ranges.append(threshold_lookup[name])
        else:
            print(f"Warning: '{name}' not found in metadata — using (0, 1) as fallback.")
            channel_ranges.append((0, 1))
 
    print("\nThreshold lookup built:")
    for name, (mn, mx) in threshold_lookup.items():
        print(f"  {name}: min={mn}, max={int(mx)}")
 
    return threshold_lookup, channel_ranges
 

# =============================================================================
# 4. TISSUE MASKING
# =============================================================================

def percentile_95_excluding_zeros(image: np.ndarray) -> float:
    """
    Returns the 95th percentile brightness of non-zero, finite pixels.
    Used to set a sensible display maximum for the mask preview plot,
    so that a few very bright hotspot pixels don't wash out the image.
    Zero pixels are excluded because they represent true background/no signal.

    NOTE: This function only affects the display ranges of the preview plots.
    It does NOT affect the actual masking threshold or the pixel values used in PCA/UMAP/tSNE. Data stays same.
    The masking threshold is set separately in the config file and applied in the threshold_SOR_fill_BED function. 
    This function is purely for visualisation purposes to help you choose a good threshold value by showing a preview of the mask with a reasonable colour range.
    """
    valid = np.isfinite(image) & (image > 0)
    if not np.any(valid):
        raise ValueError("No non-zero finite pixels found in image.")
    return float(np.nanpercentile(image[valid], 95))
 
 
def threshold_SOR_fill_BED(
    image: np.ndarray,
    threshold: float,
    small_objects_removal: int = 0,
    fill_holes: bool = True,
    hole_area_threshold: int = int(1e4),
    BED_iterations: int = 0,
    connectivity: int = 2,
):
    """
    This is the core masking function. 
    It takes one elements 2D image and turns it into a binary mask (1 = tissue, 0 = background) through a series of 4 steps:
 
    Step 1 — THRESHOLD:
        Keep all pixels with intensity >= threshold.
        Pixels below the threshold are treated as background and set to 0.
 
    Step 2 — SMALL OBJECT REMOVAL (SOR):
        Remove tiny isolated clusters of pixels that are too small to be real tissue.
        'small_objects_removal' sets the minimum size (in pixels) to keep.
        e.g. 800 means any connected region smaller than 800 pixels is removed.
 
    Step 3 — FILL HOLES:
        Fill enclosed background regions inside the tissue mask.
        e.g. if a blood vessel shows as a dark hole inside the tissue, it gets filled in.
        'hole_area_threshold' sets the maximum hole size (pixels) to fill.
 
    Step 4 — BINARY EROSION + DILATION (BED):
        Smooths the mask edges by first shrinking (erosion) then expanding (dilation).
        This removes jagged edges and small protrusions from the mask boundary.
        Each iteration uses a 3x3 square structuring element.
 
    Returns a tuple of 8 items — an (image, mask) pair after each step:
        (thresh_image, thresh_mask, SOR_image, SOR_mask,
         fill_image, fill_mask, BED_image, BED_mask)
 
    Index [7] = BED_mask is the final mask used in the pipeline.
    """
    # Step 1: Threshold — pixels below threshold become 0
    thresh_mask  = image >= threshold
    thresh_image = image * thresh_mask
 
    # Step 2: Small object removal — remove connected regions smaller than min_size pixels
    if small_objects_removal > 0:
        SOR_mask = remove_small_objects(thresh_mask,
                                        min_size=small_objects_removal,
                                        connectivity=connectivity)
    else:
        SOR_mask = thresh_mask.copy()  # skip this step if value is 0
    SOR_image = image * SOR_mask
 
    # Step 3: Fill holes — fill enclosed background regions up to area_threshold pixels
    if fill_holes:
        fill_mask = remove_small_holes(SOR_mask,
                                       area_threshold=hole_area_threshold,
                                       connectivity=connectivity)
    else:
        fill_mask = SOR_mask.copy()  # skip this step if fill_holes is False
    fill_image = image * fill_mask
 
    # Step 4: BED — alternate between erosion and dilation to smooth mask edges
    if BED_iterations > 0:
        BED_mask = fill_mask.copy()
        selem = square(3)  # 3x3 square structuring element
        for _ in range(BED_iterations):
            BED_mask = binary_erosion(BED_mask, selem)   # shrink edges
            BED_mask = binary_dilation(BED_mask, selem)  # expand edges back
    else:
        BED_mask = fill_mask.copy()  # skip this step if BED_iterations is 0
    BED_image = image * BED_mask
 
    # Return all intermediate stages as uint8 (0 or 1 values)
    return (
        thresh_image, thresh_mask.astype(np.uint8),
        SOR_image,    SOR_mask.astype(np.uint8),
        fill_image,   fill_mask.astype(np.uint8),
        BED_image,    BED_mask.astype(np.uint8),
    )
 
 
def apply_mask(
    img: np.ndarray,
    img_filtered: np.ndarray,
    channel_names_filtered: list,
    channel_for_mask: int,
    mask_threshold: float,
    small_objects_removal: int  = None,
    fill_holes: bool            = False,
    BED_iterations: int         = None,
    show_preview: bool          = True,
    output_dir: str             = None,
):
    """
    This is where the mask actually gets applied to the full image and builds the pixel dataframe (df) and tissue_indices_final.
 
    This function:
      1. Runs threshold_SOR_fill_BED on a chosen channel to create the binary mask
      2. Optionally shows a preview of the mask (original / masked / binary)
      3. Flattens the 3D image into a 2D table of tissue pixels only
      4. Returns df (the pixel dataframe) and tissue_indices_final
 
    WHY tissue_indices_final?
      The image is 2D (height x width). When we flatten it to a 1D array of pixels,
      each pixel has a flat index (0 to height*width-1). tissue_indices_final stores
      which flat indices correspond to tissue pixels. Later scripts use this to map
      cluster labels back to their correct (x, y) position in the original image.
 
    Parameters:
        img                   : full image after dropping 0TIC (n_channels, H, W)
        img_filtered          : image after all channel exclusions (n_kept, H, W)
        channel_names_filtered: names of the kept channels
        channel_for_mask      : which channel index (in img) to use for masking
                                e.g. 1 = Mg (a good structural channel)
        mask_threshold        : pixel intensity threshold — pixels below this are background
        small_objects_removal : min size of connected regions to keep (overrides config if set)
        fill_holes            : whether to fill enclosed holes in the tissue mask
        BED_iterations        : number of erosion/dilation cycles (overrides config if set)
        show_preview          : if True, shows a 3-panel preview figure
        output_dir            : if provided, saves the preview figure as a PNG here
 
    Returns:
        df                   : DataFrame, shape (n_tissue_pixels, n_channels)
                               each row = one tissue pixel, each column = one element
        tissue_indices_final : 1D array of flat pixel indices for tissue pixels
        height, width        : dimensions of the original image
    """
    # Use config defaults unless override values were passed in
    sor = small_objects_removal if small_objects_removal is not None else config.SMALL_OBJECT_REMOVAL
    bed = BED_iterations        if BED_iterations        is not None else config.BED_ITERATIONS
    hat = config.HOLE_AREA_THRESHOLD
 
    # Select the channel used to generate the mask (before filtering, hence using img)
    mask_channel = img[channel_for_mask]
 
    # Optional crop — restricts masking and analysis to a subregion of the image.
    # Controlled by MASK_X_RANGE and MASK_Y_RANGE in config.py.
    # Set both to None (default) to use the full image.
    if config.MASK_X_RANGE is not None:
        mask_channel = mask_channel[:, config.MASK_X_RANGE[0]:config.MASK_X_RANGE[1]]
        img_filtered  = img_filtered[:, :, config.MASK_X_RANGE[0]:config.MASK_X_RANGE[1]]
    if config.MASK_Y_RANGE is not None:
        mask_channel = mask_channel[config.MASK_Y_RANGE[0]:config.MASK_Y_RANGE[1], :]
        img_filtered  = img_filtered[:, config.MASK_Y_RANGE[0]:config.MASK_Y_RANGE[1], :]
 
    # --- Optional preview ---
    if show_preview:
        # Run masking just to show what each step looks like
        preview = threshold_SOR_fill_BED(
            image=mask_channel,
            threshold=mask_threshold,
            small_objects_removal=sor,
            fill_holes=fill_holes,
            hole_area_threshold=hat,
            BED_iterations=bed,
        )
        # Use 95th percentile as vmax so hotspots don't wash out the preview
        vmax_preview = percentile_95_excluding_zeros(mask_channel)
 
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(12, 4))
        ax1.imshow(mask_channel, vmax=vmax_preview, cmap='inferno_r', interpolation='nearest')
        ax2.imshow(preview[6],   vmax=vmax_preview, cmap='inferno_r', interpolation='nearest')
        ax3.imshow(preview[7],                      cmap='gray',      interpolation='nearest')
        ax1.set_title("Original")
        ax2.set_title("Masked (fill+BED applied)")
        ax3.set_title("Binary mask")
        for ax in (ax1, ax2, ax3):
            ax.axis("off")
        mask_channel_name = channel_names_filtered[channel_for_mask] if channel_for_mask < len(channel_names_filtered) else f"channel {channel_for_mask}"
        fig.suptitle(
            f"Mask channel: {mask_channel_name}  |  Threshold={mask_threshold}  SOR={sor}  Fill={fill_holes}  BED={bed}",
            fontsize=10,
        )
        plt.tight_layout(rect=[0, 0, 1, 0.9])
        if output_dir:
            fig.savefig(os.path.join(output_dir, "mask_preview.png"),
                        dpi=config.FIGURE_DPI, bbox_inches='tight')
        plt.show()
 
    # --- Final mask (index [7] = BED_mask, the last and cleanest step) ---
    mask_data = threshold_SOR_fill_BED(
        image=mask_channel,
        threshold=mask_threshold,
        small_objects_removal=sor,
        fill_holes=fill_holes,
        hole_area_threshold=hat,
        BED_iterations=bed,
    )[7]
 
    # --- Build the pixel dataframe ---
    n_channels, height, width = img_filtered.shape
    n_pixels = height * width
 
    # Reshape from (n_channels, H, W) → (n_pixels, n_channels)
    # Each row is now one pixel, each column is one element — like a spreadsheet
    img_reshaped = img_filtered.reshape(n_channels, n_pixels).T
 
    # Find the flat indices of tissue pixels (where mask = 1)
    tissue_indices = np.where(mask_data.flatten() > 0)[0]
    tissue_indices_final = tissue_indices.copy()
 
    # Build the dataframe using only tissue pixels
    df = pd.DataFrame(img_reshaped[tissue_indices], columns=channel_names_filtered)
 
    print(f"\nTotal pixels:           {n_pixels:,}")
    print(f"Tissue pixels in mask:  {len(df):,}")
    print(f"Pixels excluded:        {n_pixels - len(df):,}")
 
    return df, tissue_indices_final, height, width
 
 
# =============================================================================
# 5. LOG vs LINEAR SCALE SUGGESTION
# =============================================================================
 
def suggest_scale(img_filtered: np.ndarray, channel_names_filtered: list,
                  output_dir: str = None, show_plot: bool = True):
    """
    Automatically suggests whether each channel should be displayed on a
    log or linear colour scale, based on three statistical metrics.
 
    WHY does this matter?
        LA-ICP-MS data often has extreme outlier pixels (e.g. a tiny hotspot of
        very high Fe concentration). On a linear scale, these hotspots dominate
        the colour range, making the rest of the tissue look flat and featureless.
        Log scaling compresses the bright end so you can see variation across the
        whole tissue.
 
    The three metrics used:
        Skew           — how lopsided the intensity distribution is.
                         High skew = most pixels are dim but a few are very bright.
        CV             — coefficient of variation (std / mean).
                         High CV = signal is highly variable relative to its average.
        P99/Median     — ratio of 99th percentile to the median.
                         High ratio = the brightest 1% of pixels are much brighter
                         than the typical pixel.
 
    Decision logic:
        LOG          = skew > 3  AND  P99/Median > 50  (both must be true)
        POSSIBLY LOG = skew > 2  OR   CV > 2           (only one needed)
        LINEAR       = none of the above
 
    For each channel, produces three plots:
        Left   — linear scale image
        Middle — log scale image
        Right  — intensity histogram with the metrics and suggestion shown
 
    Returns:
        scale_suggestions : dict of {channel_name: 'LOG' | 'POSSIBLY LOG' | 'LINEAR'}
        This dict is used downstream in UMAP/tSNE scatter plots to auto-apply
        log scaling when displaying per-channel intensity overlays.
    """
    scale_suggestions = {}
    n = len(channel_names_filtered)
 
    fig, axes = plt.subplots(nrows=n, ncols=3, figsize=(18, 4 * n))
    if n == 1:
        axes = [axes]  # ensure axes is always a 2D structure even with one channel
 
    for i in range(n):
        channel_data = img_filtered[i]
        flat_data    = channel_data.flatten()
        valid_data   = flat_data[flat_data > 0]  # exclude background zeros
 
        # Use 99.9th percentile as max for linear display to avoid hotspot washout
        robust_max = np.percentile(valid_data, 99.9) if len(valid_data) > 0 else 1
 
        # Left: linear scale image
        axes[i][0].imshow(channel_data, cmap='inferno', vmax=robust_max, origin='upper')
        axes[i][0].set_title(f"{channel_names_filtered[i]} — Linear")
        axes[i][0].axis('off')
 
        # Middle: log scale image
        # vmin=1 stops near-zero noise from dragging the colour scale down
        # vmax=95th percentile spreads colour more evenly across the tissue signal
        axes[i][1].imshow(channel_data, cmap='magma',
                          norm=colors.LogNorm(vmin=1, vmax=np.percentile(valid_data, 95)),
                          origin='upper')
        axes[i][1].set_title(f"{channel_names_filtered[i]} — Log")
        axes[i][1].axis('off')
 
        # Right: histogram + decision
        if len(valid_data) > 0:
            skew             = stats.skew(valid_data)
            cv               = np.std(valid_data) / np.mean(valid_data)
            percentile_ratio = np.percentile(valid_data, 99) / (np.median(valid_data) + 1e-10)
 
            if skew > 3 and percentile_ratio > 50:
                suggestion    = "LOG"
                suggest_color = 'red'
            elif skew > 2 or cv > 2:
                suggestion    = "POSSIBLY LOG"
                suggest_color = 'orange'
            else:
                suggestion    = "LINEAR"
                suggest_color = 'green'
 
            scale_suggestions[channel_names_filtered[i]] = suggestion
 
            axes[i][2].hist(valid_data, bins=100, color='gray', log=True)
            axes[i][2].set_xlabel("Pixel Brightness")
            axes[i][2].set_ylabel("Count (log scale)")
            axes[i][2].set_title(
                f"Skew: {skew:.1f}  |  CV: {cv:.1f}  |  P99/Median: {percentile_ratio:.0f}x",
                fontsize=8
            )
            # Show the suggestion text on the histogram
            axes[i][2].text(0.5, 0.9, f"Suggestion: {suggestion}",
                            transform=axes[i][2].transAxes,
                            color=suggest_color, weight='bold', ha='center')
 
    plt.tight_layout()
    if output_dir:
        fig.savefig(os.path.join(output_dir, "log_vs_linear.png"),
                    dpi=config.FIGURE_DPI, bbox_inches='tight')
    if show_plot:
        plt.show()
    plt.close()
 
    print("\nScale suggestions:")
    for name, sug in scale_suggestions.items():
        print(f"  {name}: {sug}")
 
    return scale_suggestions
 
 
# =============================================================================
# 6. LOG1P NORMALISATION
# =============================================================================
 
def apply_log1p(df: pd.DataFrame):
    """
    Applies log1p normalisation to the pixel dataframe before PCA/tSNE/UMAP.
 
    log1p(x) = log(x + 1)
 
    WHY log1p and not just log?
        LA-ICP-MS data contains genuine zero values (pixels with no signal for
        that element). log(0) is undefined (-infinity), which would break PCA/tSNE/UMAP.
        Adding 1 before taking the log ensures log(0+1) = 0, which is safe.
 
    WHY normalise at all?
        Elements vary enormously in absolute intensity. For example, Na and P
        might have values in the thousands while Se is in single digits. Without
        normalisation, PCA would be dominated by the highest-intensity channels
        regardless of their biological relevance. log1p compresses the dynamic
        range so all channels contribute more equally.
 
    The original df is NOT modified — a new normalised dataframe is returned.
    The raw counts are preserved in df and can always be accessed if needed.
 
    Returns:
        df_normalised : new DataFrame of the same shape as df, with log1p applied
    """
    if config.USE_LOG1P:
        # np.log1p applied element-wise across all pixel values
        df_normalised = np.log1p(df.values).astype(np.float32)
        df_normalised = pd.DataFrame(df_normalised, columns=df.columns)
        print("log1p normalisation applied.")
    else:
        df_normalised = df.copy()
        print("No normalisation applied (USE_LOG1P is False in config).")
 
    return df_normalised
 
 
# =============================================================================
# 7. VISUALISE CHANNELS (element maps with custom colourmaps)
# =============================================================================
 
def visualise_channels(img_filtered: np.ndarray, channel_names_filtered: list,
                        channel_ranges: list, qupath_colours: list,
                        output_dir: str = None, show_plot: bool = True):
    """
    Plots all filtered elemental channels side by side in a single row.
 
    Each channel is displayed with:
      - A custom black-to-colour colormap (black = no signal, colour = high signal)
      - Display range (vmin, vmax) taken from the metadata threshold lookup
        so the brightness is consistent with the acquisition software settings
 
    This gives you a quick visual overview of all elements in the tissue
    before any analysis is run.
 
    Parameters:
        img_filtered          : filtered image array (n_kept_channels, H, W)
        channel_names_filtered: names of the kept channels
        channel_ranges        : list of (min, max) display ranges from metadata
        qupath_colours        : list of colour strings, one per channel
        output_dir            : if provided, saves the figure as 'channel_maps.png'
        show_plot             : if True, displays the plot inline
    """
    n = len(channel_names_filtered)
    plt.figure(figsize=(4 * n, 4))
 
    for i in range(n):
        plt.subplot(1, n, i + 1)
        vmin, vmax = channel_ranges[i]
 
        # Create a custom colormap going from black (no signal) to the assigned colour
        cmap = LinearSegmentedColormap.from_list(
            f"black_to_{qupath_colours[i]}", ['black', qupath_colours[i]]
        )
 
        plt.imshow(img_filtered[i], cmap=cmap, vmin=vmin, vmax=vmax, origin='upper')
        plt.title(f"{channel_names_filtered[i]}\n{qupath_colours[i]}", fontsize=8)
        plt.axis('off')
 
    plt.tight_layout()
 
    if output_dir:
        plt.savefig(os.path.join(output_dir, "channel_maps.png"),
                    dpi=config.FIGURE_DPI, bbox_inches='tight')
    if show_plot:
        plt.show()
    plt.close()
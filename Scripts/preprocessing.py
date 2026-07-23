# =============================================================================
# preprocessing.py
#
# The purpose of this .py file:
#   This python file prepares raw LA-ICP-TOF-MS OME-TIFF data for downstream
#   dimensionality reduction. 
#   This file performs no analytical operations, it is only responsible  for 
#   loading the ome.tif file, masking, channel-filtering, and normalisation
#   This file is also responsible for generating two outputs, a tissue mask overlay figure
#   and a log1p transformation validation figure.
#
# Steps performed in this file:
#   1. Parse the ome.tif file and extract channel names from embedded XML metadata (found within the OME-TIFF)
#   2. Remove excluded channels already defined in config.py
#   3. Construct a binary tissue mask to isolate tissue pixels from background
#   4. Assemble the masked pixel dataframe (df) and record tissue_indices_final
#   5. Apply log1p transformation to normalise and prepare the pixel matrix for downstream dimensionality reduction and clustering
# =============================================================================

import os                             # Importing Python's built in OS utilities. os.path.join for building output file paths
import ast                            # For safely parsing the metadata CSV string-encoded lists
import xml.etree.ElementTree as ET    # Reads the  metadata embedded inside the OME-TIFF

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats               # for skewness calculation in scale suggestion
from skimage.morphology import (
    remove_small_objects,             # Removes tiny noise blobs from the mask
    remove_small_holes,               # Fills enclosed background holes inside the tissue mask
    binary_erosion,                   # Shrinks mask edges inward (part of BED smoothing)
    binary_dilation,                  # Expands mask edges outward (part of BED smoothing)
    square,                           # Defines the 3x3 structuring element used in BED
)
import tifffile                       # Reads OME-TIFF files produced by the in-house LMF Balrog python script (from LA-ICP-MS instrument)

import config                         # All settings and parameters live in config.py, in which we are importing it here


# =============================================================================
# 1. Loading the OME-TIFF 
# =============================================================================

# Defines a function called load_image that takes a file path as input and returns the raw image data and channel names.
def load_image(file_path: str):        
    """
    OME-TIFF is the file format produced by LMF'S in-house Balrog Python script.
    This script processes the data from LA-ICP-MS instrument, converting it into an ome.tif file format for use in this pipeline.
    The ome.tif file stores a 3D array of shape (n_channels, height, width) where each
    channel is a 2D elemental map (e.g. one image for Fe, one for Zn, etc.)

    Loading the ome.tif file extracts two things:
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
    # The result is a 3D array shaped as (n_channels, height, width)
    img_raw = tifffile.imread(file_path)

    # Open the file again just to read the embedded OME-XML metadata string
    # Then it stores it in omexml_string. This string contains all the metadata about the image, including channel names.
    with tifffile.TiffFile(file_path) as tif:
        omexml_string = tif.ome_metadata

    if omexml_string:
        # If the metadata string is not empty, we proceed to parse it
        # If it is empty, the file is invalide for this pipeline and we raise an error.
       
        # Parse the XML and find all Channel elements:
        # The namespace is required because OME-XML uses a specific XML schema
        # The root variable converts the XML text into a structured tree object that can be searched
        # The channels variable searches for all Channel elements within the parsed XML. 
        # Each channel corresponds to one elemental map in the image (e.g. Fe, Zn, 0TIC, etc.)
        namespaces = {'ome': 'http://www.openmicroscopy.org/Schemas/OME/2016-06'}
        root = ET.fromstring(omexml_string)
        channels = root.findall('.//ome:Channel', namespaces)

        # Extract the 'Name' attribute from each channel (e.g. 'Fe', 'Zn', '0TIC')
        # Then creates an empty list to store channel names 
        all_channel_names = []

        # This loops over each channel element and extracts the 'Name' attribute found in the OME-XML metadata.
        # If a channel is missing the 'Name' attribute, it raises an error.
        for i, c in enumerate(channels):
            if "Name" not in c.attrib:
                raise ValueError(
                    f"Channel {i} has no 'Name' attribute in OME-XML metadata. "
                    "Channel names are required for this pipeline. "
                    "Check that the in-house Balrog python script correctly exported the channel names."
                )
            # If the channel does have a name, append it to the list
            all_channel_names.append(c.attrib["Name"])

    # If the file has no OME-XML metadata, the pipeline cannot continue
    # Error warnings are put in place as shown below
    else:
        raise ValueError(
            "No OME-XML metadata found in the TIFF file. "
            "The pipeline requires a valid OME-TIFF with embedded channel names. "
            "Check that your file was exported correctly from the in-house Balrog Python Script."
        )

    print(f"Original data shape: {img_raw.shape}")  # Shows the shape of the pixel array (n_channels, height, width)
    print(f"Data type: {img_raw.dtype}")            # Shows the datatype of the pixel array 
    print(f"Channels found: {all_channel_names}")   # Lists the extracted elemental channel names 

    return img_raw, all_channel_names  


# =============================================================================
# 2. Filtering and removing unwanted channels (elements)
# =============================================================================

def filter_channels(img_raw: np.ndarray, all_channel_names: list):
    """
    Removes unwanted channels from the image before analysis.

    Two types of channels are removed:
      - 0TIC (Total Ion Count): always dropped. This is a sum of all channels
        and does not represent a real element. Including it would distort results.
      - Manual exclusions: any channels listed in config.CHANNELS_TO_DROP.
        Use this for channels with poor signal quality (e.g. very noisy 89Y and 111Cd).

    Returns:
        img_filtered          : image after dropping all listed channels (n_kept, H, W)
        channel_names_filtered: names of the kept channels
    """
    # Identify which channels should be removed based on their names
    # We compare each channel name against config.CHANNELS_TO_DROP
    # We look up by name rather than hardcoding index 0, because channel order may
    # vary between datasets.
    drop_indices = [i for i, name in enumerate(all_channel_names)
                    if name in config.CHANNELS_TO_DROP]

    # Remove the unwanted channels from the raw imag.
    # np.delete removes the channels at drop_indices along axis=0 (the channel axis)
    img_filtered = np.delete(img_raw, drop_indices, axis=0)

    # This builds a new list of channel names that excludes the dropped ones
    # This keeps the names aligned with the filtered image array 
    channel_names_filtered = [name for i, name in enumerate(all_channel_names)
                               if i not in drop_indices]

    # Printing the diagnostic information so the user can verify what was kept/dropped
    print(f"\nOriginal channels:       {len(all_channel_names)}")
    print(f"After dropping channels: {len(channel_names_filtered)}")
    print(f"Kept channels:           {channel_names_filtered}")
    print(f"Dropped channels:        {config.CHANNELS_TO_DROP}")
   
    # Return the filtered image and the updated list of channel names 
    return img_filtered, channel_names_filtered


# =============================================================================
# 3. Tissue Masking
# =============================================================================

def percentile_95_excluding_zeros(image: np.ndarray) -> float:
    """
    Returns the 95th percentile brightness of non-zero, finite pixels.
    This is used to set a sensible display maximum for the mask preview plot,
    so that a few very bright hotspot pixels don't wash out the image.
    Zero pixels are excluded because they represent true background/no signal.

    NOTE: This function only affects the display ranges of the preview plots.
    It does NOT affect the actual masking threshold or the pixel values used downstream. Data stays same.
    The masking threshold is set separately in the config file and applied in the threshold_SOR_fill_BED function.
    This function is purely for visualisation purposes to help you choose a good threshold value
    by showing a preview of the mask with a reasonable colour range.
    """
    
    # Identify valid pixels: finite values AND > 0 (exclude background)
    valid = np.isfinite(image) & (image > 0)

    # If no valid pixels exist, masking cannot proceed
    if not np.any(valid):
        raise ValueError("No non-zero finite pixels found in image.")
    
    # Compute the 95th percentile of the valid pixels for scaling the display 
    return float(np.nanpercentile(image[valid], 95))



# This defined the threshold_SOR_fill_BED function
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
    It takes one elements 2D image and turns it into a binary mask (1 = tissue, 0 = background)
    through a series of 4 steps:

    Step 1: Threshold
        Keep all pixels with intensity >= threshold.
        Pixels below the threshold are treated as background and set to 0.

    Step 2: Small Object Removal (SOR):
        Remove tiny isolated clusters of pixels that are too small to be real tissue.
        'small_objects_removal' sets the minimum size (in pixels) to keep.
        e.g. 800 means any connected region smaller than 800 pixels is removed.

    Step 3: Fill Holes
        Fill enclosed background regions inside the tissue mask.
        e.g. if a blood vessel shows as a dark hole inside the tissue, it gets filled in.
        'hole_area_threshold' sets the maximum hole size (pixels) to fill.

    Step 4: Binary Erosion + Dilation (BED):
        Smooths the mask edges by first shrinking (erosion) then expanding (dilation).
        This removes jagged edges and small protrusions from the mask boundary.
        Each iteration uses a 3x3 square structuring element.

    Returns a tuple of 8 items, an (image, mask) pair after each step:
        (thresh_image, thresh_mask, SOR_image, SOR_mask,
         fill_image, fill_mask, BED_image, BED_mask)

    Index [7] = BED_mask is the final mask used in the pipeline.
    """
    # Step 1: Threshold
    # Creates a binary mask where pixels below threshold become 0
    thresh_mask  = image >= threshold

    # Apply mask to the image (background becomes 0)
    thresh_image = image * thresh_mask


    # Step 2: Small object removal (SOR)
    # Removes tiny connected regions smaller than min_size pixels, or "small_objects_removal"
    if small_objects_removal > 0:
        SOR_mask = remove_small_objects(thresh_mask,
                                        min_size=small_objects_removal,
                                        connectivity=connectivity)
    # If SOR is disabled, just copy the threshold mask
    else:
        SOR_mask = thresh_mask.copy()  
    SOR_image = image * SOR_mask


    # Step 3: Fill holes
    # Fill enclosed background regions up to hole_area_threshold 
    if fill_holes:
        fill_mask = remove_small_holes(SOR_mask,
                                       area_threshold=hole_area_threshold,
                                       connectivity=connectivity)
    else:
        fill_mask = SOR_mask.copy()  # skip this step if fill_holes is False or disabled
    # Apply hole-filled mask to image
    fill_image = image * fill_mask


    # Step 4: BED 
    # Smooth mask edges using erosion followed by dilation 
    if BED_iterations > 0:
        BED_mask = fill_mask.copy()
        selem = square(3)  # 3x3 square structuring element
        
        # Perform erosion and dilation for each iteration 
        for _ in range(BED_iterations):
            BED_mask = binary_erosion(BED_mask, selem)   # shrink edges
            BED_mask = binary_dilation(BED_mask, selem)  # expand edges back
    else:
        BED_mask = fill_mask.copy()  # skip this step if BED_iterations is 0
    # Apply BED mask to image 
    BED_image = image * BED_mask

    # Return all intermediate stages as uint8 (0 or 1 values)
    return (
        thresh_image, thresh_mask.astype(np.uint8),
        SOR_image,    SOR_mask.astype(np.uint8),
        fill_image,   fill_mask.astype(np.uint8),
        BED_image,    BED_mask.astype(np.uint8),
    )



# This defines the apply_mask function
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
    This is where the mask actually gets applied to the full image and builds
    the pixel dataframe (df) and tissue_indices_final.

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
        mask_threshold        : pixel intensity threshold. Pixels below this are background
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

    # Optional crop: Restricts masking and analysis to a subregion of the image.
    # Controlled by MASK_X_RANGE and MASK_Y_RANGE in config.py.
    # Set both to None (default) to use the full image.
    if config.MASK_X_RANGE is not None:
        mask_channel = mask_channel[:, config.MASK_X_RANGE[0]:config.MASK_X_RANGE[1]]
        img_filtered  = img_filtered[:, :, config.MASK_X_RANGE[0]:config.MASK_X_RANGE[1]]
    if config.MASK_Y_RANGE is not None:
        mask_channel = mask_channel[config.MASK_Y_RANGE[0]:config.MASK_Y_RANGE[1], :]
        img_filtered  = img_filtered[:, config.MASK_Y_RANGE[0]:config.MASK_Y_RANGE[1], :]

    # Threshold diagnostic percentiles 
    # It flattens the masked channel and computes intensity percentiles for threshold tuning
    flat = mask_channel.flatten()
    flat_nonzero = flat[flat > 0]
    p50, p75, p90, p95, p99 = np.percentile(flat_nonzero, [50, 75, 90, 95, 99])

    # Prints the diagnostic statistics 
    print(f"\n  Mask channel intensity percentiles (non-zero pixels):")
    print(f"    50th: {p50:.1f}   75th: {p75:.1f}   90th: {p90:.1f}   95th: {p95:.1f}   99th: {p99:.1f}")
    print(f"    Current MASK_THRESHOLD: {mask_threshold}")
    print(f"    Tissue pixels above threshold: {(flat > mask_threshold).sum():,} / {flat.size:,} ({100*(flat > mask_threshold).mean():.1f}%)\n")

    # Masking preview
    # This can be turned on or off in the config.py file
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
        
        # Create a 3 panel preview figure
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(12, 4))
       
        # Panel 1: Original channel 
        ax1.imshow(mask_channel, vmax=vmax_preview, cmap='inferno_r', interpolation='nearest')
        
        # Panel 2: Masked image (after fill + BED)
        ax2.imshow(preview[6], vmax=vmax_preview, cmap='inferno_r', interpolation='nearest')
        
        # Panel 3: Finaly binary mask 
        ax3.imshow(preview[7], cmap='gray', interpolation='nearest')
        
        # Titles fpr ythe three panels 
        ax1.set_title("Original")
        ax2.set_title("Masked (fill+BED applied)")
        ax3.set_title("Binary mask")
        
        # Remove axes for a cleaner display
        for ax in (ax1, ax2, ax3):
            ax.axis("off")
        
        # Title showing masked parameters 
        mask_channel_name = channel_names_filtered[channel_for_mask] if channel_for_mask < len(channel_names_filtered) else f"channel {channel_for_mask}"
        fig.suptitle(
            f"Mask channel: {mask_channel_name}  |  Threshold={mask_threshold}  SOR={sor}  Fill={fill_holes}  BED={bed}",
            fontsize=10,
        )
        plt.tight_layout(rect=[0, 0, 1, 0.9])
        
        # Save preview if output directory provided 
        if output_dir:
            fig.savefig(os.path.join(output_dir, "mask_preview.png"),
                        dpi=config.FIGURE_DPI, bbox_inches='tight')
        plt.show()


    # Final mask (index [7] = BED_mask, the last and cleanest step)
    mask_data = threshold_SOR_fill_BED(
        image=mask_channel,
        threshold=mask_threshold,
        small_objects_removal=sor,
        fill_holes=fill_holes,
        hole_area_threshold=hat,
        BED_iterations=bed,
    )[7]


    # Building the pixel dataframe 
    # These extract dimensions
    n_channels, height, width = img_filtered.shape
    n_pixels = height * width

    # Reshape from (n_channels, H, W) to (n_pixels, n_channels)
    # Each row is now one pixel, each column is one element (like a spreadsheet)
    img_reshaped = img_filtered.reshape(n_channels, n_pixels).T

    # Find the flat indices of tissue pixels (where mask = 1)
    tissue_indices = np.where(mask_data.flatten() > 0)[0]
    tissue_indices_final = tissue_indices.copy()

    # Build the dataframe using only tissue pixels
    df = pd.DataFrame(img_reshaped[tissue_indices], columns=channel_names_filtered)
    
    # Print Summary
    print(f"\nTotal pixels:           {n_pixels:,}")
    print(f"Tissue pixels in mask:  {len(df):,}")
    print(f"Pixels excluded:        {n_pixels - len(df):,}")

    return df, tissue_indices_final, height, width


# =============================================================================
# 4. Log1p Transformation
# =============================================================================

def apply_normalisation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies log1p transformation [log(x + 1)] element-wise to the pixel
    dataframe before downstream dimensionality reduction
    
    log1p is a suitable technique used transform this is data
    This is because it compresses high dynamic range while preserving zeros.
      - Safe for zeros: log(0 + 1) = 0
      - Compresses the high dynamic range of ion counts
      - Brings extreme channels onto a comparable scale

    """
    transformed = np.log1p(df.values).astype(np.float32)
   
    print("Log1p transformation applied: log1p  [log(x + 1)]")
    return pd.DataFrame(transformed, columns=df.columns)


# Backward-compatible alias
apply_log1p = apply_normalisation


# =============================================================================
# Figure 1: Tissue mask overlay
# =============================================================================

def plot_mask_overlay(img_filtered: np.ndarray, tissue_indices_final: np.ndarray,
                      height: int, width: int,
                      channel_for_mask: int, channel_names_filtered: list,
                      output_dir: str = None):
    """
    Publication-quality 2-panel figure showing exactly which pixels were
    included in the analysis vs excluded as background.

    Panel 1 — Raw channel image (inferno colourmap, 99th percentile vmax).
    Panel 2 — Overlay: tissue pixels in colour, excluded pixels darkened to
               10% brightness. A cyan contour traces the tissue boundary.

    This is distinct from mask_preview.png (which is a quick diagnostic).
    This figure is designed for inclusion in the dissertation Results section
    to validate the tissue isolation step.
    """
    ch_img = img_filtered[channel_for_mask].astype(float)

    # Robust display max — exclude zeros and outlier hotspots
    nonzero = ch_img[ch_img > 0]
    vmax = float(np.percentile(nonzero, 99)) if len(nonzero) > 0 else float(ch_img.max())
    ch_norm = np.clip(ch_img / (vmax + 1e-10), 0, 1)

    # Reconstruct binary mask from tissue_indices_final
    mask = np.zeros(height * width, dtype=bool)
    mask[tissue_indices_final] = True
    mask = mask.reshape(height, width)

    # Build RGB overlay: tissue = inferno colour, background = 10% brightness
    tissue_rgb = plt.cm.inferno(ch_norm)[:, :, :3]          # (H, W, 3)
    overlay    = tissue_rgb * 0.10                            # dark background
    overlay[mask] = tissue_rgb[mask]                         # bright tissue

    ch_name   = channel_names_filtered[channel_for_mask]
    n_tissue  = int(tissue_indices_final.shape[0])
    n_total   = height * width
    pct       = 100.0 * n_tissue / n_total

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor='white')

    axes[0].imshow(ch_norm, cmap='inferno', origin='upper', vmin=0, vmax=1)
    axes[0].set_title(f"Raw — {ch_name} channel", fontsize=12, fontweight='bold')
    axes[0].axis('off')

    axes[1].imshow(overlay, origin='upper')
    axes[1].contour(mask, levels=[0.5], colors='cyan', linewidths=0.8, alpha=0.8)
    axes[1].set_title(
        f"Tissue mask overlay\n"
        f"{n_tissue:,} / {n_total:,} pixels retained ({pct:.1f}%)",
        fontsize=12, fontweight='bold',
    )
    axes[1].axis('off')

    fig.suptitle(
        f"Tissue Isolation — {ch_name} channel  |  cyan contour = mask boundary",
        fontsize=11, y=1.01,
    )
    plt.tight_layout()

    if output_dir:
        path = os.path.join(output_dir, "mask_overlay.png")
        fig.savefig(path, dpi=config.FIGURE_DPI, bbox_inches='tight')
        print(f"  Saved → {path}")
    plt.close()

# =============================================================================
# Figure 2: Log1p transformation Violin Plots
# =============================================================================

def plot_normalisation_comparison(df_raw: pd.DataFrame, df_normalised: pd.DataFrame,
                                   channel_names_filtered: list,
                                   output_dir: str = None):
    """
    Produces two separate figures:

    Figure 1 — normalisation_comparison_violins.png
        Side-by-side violin plots (a) raw counts and (b) log1p-normalised,
        labelled (a) and (b).

    Figure 2 — normalisation_comparison_tables.png
        Side-by-side summary tables (a) and (b) showing median and IQR
        per channel, with channels as rows to avoid label overlap.
    """
    # Pick up to 10 representative channels, evenly spaced
    n_show  = min(10, len(channel_names_filtered))
    indices = np.linspace(0, len(channel_names_filtered) - 1, n_show, dtype=int)
    selected = [channel_names_filtered[i] for i in indices]

    panels = [
        (df_raw,        "(a) Before normalisation (raw counts)", "Ion count"),
        (df_normalised, "(b) After log1p normalisation",         "log(x + 1)"),
    ]

    def fmt(v):
        return f'{v:.2e}' if abs(v) >= 1e4 or (abs(v) < 0.01 and v != 0) else f'{v:.2f}'

    # Figure 1: Violin plots 
    fig1, axes1 = plt.subplots(1, 2, figsize=(16, 6), facecolor='white')

    for ax, (df, title, ylabel) in zip(axes1, panels):
        data = [df[ch].values for ch in selected]

        vp = ax.violinplot(data, positions=range(n_show),
                           showmedians=True, showextrema=False, widths=0.7)
        for body in vp['bodies']:
            body.set_facecolor('#4c72b0')
            body.set_alpha(0.6)
        vp['cmedians'].set_color('white')
        vp['cmedians'].set_linewidth(2)

        ax.set_xticks(range(n_show))
        ax.set_xticklabels(selected, rotation=45, ha='right', fontsize=9)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.spines[['top', 'right']].set_visible(False)
        ax.yaxis.set_major_formatter(plt.matplotlib.ticker.FuncFormatter(
            lambda x, _: f'{int(x):,}' if x == int(x) else f'{x:,.1f}'
        ))

    fig1.suptitle(
        "Effect of log1p Normalisation on Element Intensity Distributions\n"
        "(representative channels shown)",
        fontsize=13, fontweight='bold',
    )
    plt.tight_layout()

    if output_dir:
        path1 = os.path.join(output_dir, "normalisation_comparison_violins.png")
        fig1.savefig(path1, dpi=config.FIGURE_DPI, bbox_inches='tight')
        print(f"  Saved → {path1}")
    plt.close()

    # Figure 2: Summary tables 
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 4), facecolor='white')

    for ax, (df, title, _) in zip(axes2, panels):
        medians = [float(np.median(df[ch].values))        for ch in selected]
        q25     = [float(np.percentile(df[ch].values, 25)) for ch in selected]
        q75     = [float(np.percentile(df[ch].values, 75)) for ch in selected]
        iqrs    = [q75[i] - q25[i]                         for i in range(n_show)]

        # Raw counts: full integer with commas. Normalised: 2 decimal places.
        is_raw = (df is df_raw)
        def fmt_cell(v):
            if is_raw:
                return f'{int(round(v)):,}'
            return f'{v:.2f}'

        # Channels as rows, statistics as columns, avoids label overlap
        cell_text  = [[fmt_cell(medians[i]), fmt_cell(iqrs[i])] for i in range(n_show)]
        row_labels = selected
        col_labels = ['Median', 'IQR']

        tbl = ax.table(
            cellText=cell_text,
            rowLabels=row_labels,
            colLabels=col_labels,
            cellLoc='center',
            loc='center',
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9)
        tbl.scale(1.6, 1.5)

        # Style
        for (r, c), cell in tbl.get_celld().items():
            cell.set_edgecolor('#cccccc')
            if r == 0 or c == -1:
                cell.set_facecolor('#dce8f8')
                cell.set_text_props(fontweight='bold')
            else:
                cell.set_facecolor('white')

        ax.axis('off')
        ax.set_title(title, fontsize=11, fontweight='bold', pad=12)

    fig2.suptitle(
        "Summary Statistics — Median and IQR per Channel\n"
        "Before (a) and After (b) log1p Normalisation",
        fontsize=12, fontweight='bold',
    )
    plt.tight_layout()

    if output_dir:
        path2 = os.path.join(output_dir, "normalisation_comparison_tables.png")
        fig2.savefig(path2, dpi=config.FIGURE_DPI, bbox_inches='tight')
        print(f"  Saved → {path2}")
    plt.close()
# =============================================================================
# main.py
#
# PURPOSE:
#   Orchestrates the entire dimensionality reduction pipeline from a single
#   entry point. Running this script will execute every stage in order:
#
#     1. Preprocessing  — load OME-TIFF, mask tissue, log1p normalise
#     2. PCA            — loading plots to understand elemental variation
#     3. UMAP           — 3D embedding and RGB spatial map (primary output)
#     4. tSNE           — comparison embedding (subsampled, thesis only)
#     5. Clustering     — K-means elbow + HDBSCAN sweep on UMAP coordinates
#
# HOW TO RUN:
#   From the terminal (inside the pipeline/Scripts folder or with it on PATH):
#
#     python main.py --input /path/to/image.ome.tiff --output /path/to/results
#
#   Optional flags:
#     --skip-pca       skip PCA step
#     --skip-tsne      skip tSNE step (saves time on large images)
#     --skip-cluster   skip clustering step
#     --hdbscan-sweep  run the full HDBSCAN parameter sweep instead of a single fit
#     --no-gpu         force CPU UMAP even if a GPU is available
#
# OUTPUT STRUCTURE:
#   results/
#   ├── preprocessing/   (channel maps, mask visualisation)
#   ├── pca/             (scree, loading matrix, PCA RGB map)
#   ├── umap/            (RGB spatial map, channel coloured plots, 3D scatter)
#   ├── tsne/            (RGB spatial map, channel coloured plots)
#   └── clustering/      (K-means elbow, HDBSCAN sweep, spatial label maps)
#
# NOTES:
#   - All parameters are controlled via config.py — do not hardcode values here.
#   - HDBSCAN and K-means parameters in config.py are PLACEHOLDERS until you
#     have run the sweep on your actual RCC dataset and chosen final values.
# =============================================================================

import os
import sys
import argparse
import time
import traceback
import numpy as np

# Import config first — all parameters live there
import config

# Import each pipeline stage
# (These imports are at the top so missing dependencies are caught immediately)
import preprocessing
import pca_analysis
import umap_reduction
import Clustering
import matplotlib.pyplot as plt


# =============================================================================
# SHARED UTILITIES  (previously in visualisation.py)
# =============================================================================

def make_output_dir(base_dir: str, step_name: str) -> str:
    """Creates a sub-folder inside base_dir for a pipeline step."""
    out_path = os.path.join(base_dir, step_name)
    os.makedirs(out_path, exist_ok=True)
    return out_path


def set_publication_style() -> None:
    """Applies global matplotlib rcParams for publication-quality figures."""
    plt.rcParams.update({
        "font.size":        11,
        "axes.titlesize":   12,
        "axes.labelsize":   11,
        "xtick.labelsize":  9,
        "ytick.labelsize":  9,
        "legend.fontsize":  9,
        "figure.titlesize": 13,
        "lines.linewidth":  1.5,
        "axes.linewidth":   0.8,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.facecolor":   "white",
        "figure.facecolor": "white",
        "savefig.bbox":     "tight",
    })


# =============================================================================
# ARGUMENT PARSER
# =============================================================================

def build_parser() -> argparse.ArgumentParser:
    """
    Defines the command-line interface for main.py.

    argparse turns command-line strings into a Python namespace object so that
    flags like --skip-pca become args.skip_pca = True inside the script.
    """
    parser = argparse.ArgumentParser(
        prog="main.py",
        description=(
            "Dimensionality reduction pipeline for LA-ICP-MS TOF spatial "
            "metallomic imaging (OME-TIFF input)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # --- Required arguments ---
    parser.add_argument(
        "--input", "-i",
        type=str,
        default=None,
        help="Path to the OME-TIFF file. Defaults to config.TIFF_FILE if not provided.",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Root output directory. Defaults to config.OUTPUT_DIR if not provided.",
    )

    # --- Optional pipeline control flags ---
    parser.add_argument(
        "--skip-pca",
        action="store_true",
        default=False,
        help="Skip the PCA step entirely.",
    )
    parser.add_argument(
        "--skip-umap",
        action="store_true",
        default=False,
        help="Skip the UMAP step.",
    )
    parser.add_argument(
        "--skip-cluster",
        action="store_true",
        default=False,
        help="Skip the clustering step.",
    )
    return parser


# =============================================================================
# STAGE RUNNER HELPER
# =============================================================================

def run_stage(stage_name: str, func, *args, **kwargs):
    """
    Wraps a pipeline stage call with timing and error handling.

    Prints a clear header/footer around each stage so the terminal output is
    easy to scan. If a stage raises an exception the error is printed and the
    pipeline continues (so one bad plot doesn't kill the whole run).

    Parameters
    ----------
    stage_name : human-readable name for this stage (printed in header)
    func       : callable to run
    *args      : positional arguments forwarded to func
    **kwargs   : keyword arguments forwarded to func

    Returns
    -------
    The return value of func, or None if func raised an exception.
    """
    sep = "=" * 60
    print(f"\n{sep}")
    print(f"  STAGE: {stage_name}")
    print(sep)

    t0 = time.time()
    result = None

    try:
        result = func(*args, **kwargs)
        elapsed = time.time() - t0
        print(f"  ✓ {stage_name} completed in {elapsed:.1f}s")
    except Exception as exc:
        elapsed = time.time() - t0
        print(f"\n  ✗ {stage_name} FAILED after {elapsed:.1f}s")
        print(f"  Error: {exc}")
        # Print the full traceback so the user can diagnose the problem
        traceback.print_exc()
        print(f"\n  Pipeline continuing — this stage will be skipped.\n")

    return result


# =============================================================================
# MAIN PIPELINE FUNCTION
# =============================================================================

def run_pipeline(args: argparse.Namespace) -> None:
    """
    Executes the full pipeline in order using the parsed CLI arguments.

    Parameters
    ----------
    args : namespace from argparse containing input, output, and flag values
    """
    # Apply publication-quality matplotlib style globally
    set_publication_style()

    # Fall back to config.py values if CLI arguments were not provided
    if args.input is None:
        args.input = config.TIFF_FILE
    if args.output is None:
        args.output = config.OUTPUT_DIR

    # Create the root output directory
    os.makedirs(args.output, exist_ok=True)
    print(f"\n  Output directory: {args.output}")

    # ------------------------------------------------------------------
    # STAGE 1: PREPROCESSING
    # ------------------------------------------------------------------
    # Load the OME-TIFF, build the tissue mask, apply log1p normalisation.
    # Returns: df_normalised, tissue_indices_final, height, width, img_filtered, channel_names_filtered
    # These are passed forward to every downstream stage.
    # ------------------------------------------------------------------

    preproc_dir = make_output_dir(args.output, "preprocessing")

    preproc_result = run_stage(
        "Preprocessing",
        _run_preprocessing,
        tiff_path=args.input,
        out_dir=preproc_dir,
    )

    if preproc_result is None:
        # Preprocessing is a hard dependency — cannot continue without it
        print("\n  FATAL: Preprocessing failed. Cannot continue.\n")
        sys.exit(1)

    df_normalised, tissue_indices_final, height, width, img_filtered, channel_names_filtered = preproc_result

    print(f"\n  Image dimensions : {height} × {width} px")
    print(f"  Tissue pixels    : {len(tissue_indices_final):,}")
    print(f"  Channels         : {df_normalised.shape[1]}")

    # ------------------------------------------------------------------
    # STAGE 2: PCA
    # ------------------------------------------------------------------
    # Fit PCA and produce loading plots.
    # Returns: X_pca (n_pixels × n_components), pca_object
    # Note: X_pca is available downstream but UMAP runs on df_normalised,
    # not on PCA-reduced data.
    # ------------------------------------------------------------------

    pca_dir = make_output_dir(args.output, "pca")
    X_pca = None

    if not args.skip_pca:
        pca_result = run_stage(
            "PCA Analysis",
            _run_pca,
            df_normalised=df_normalised,
            tissue_indices_final=tissue_indices_final,
            height=height,
            width=width,
            out_dir=pca_dir,
        )
        if pca_result is not None:
            X_pca, _ = pca_result
    else:
        print("\n  [skipped] PCA (--skip-pca flag set)")

    # ------------------------------------------------------------------
    # STAGE 3: UMAP
    # ------------------------------------------------------------------
    # Run 3D UMAP and produce RGB spatial map + channel scatter plots.
    # Returns: X_umap (n_pixels × 3), df_normalised (unchanged)
    # ------------------------------------------------------------------

    umap_dir    = make_output_dir(args.output, "umap")
    umap_cache  = os.path.join(umap_dir, "X_umap.npy")
    X_umap      = None
    umap_rgb    = None
    umap_mapper = None

    if not args.skip_umap:
        umap_run = run_stage(
            "UMAP",
            _run_umap,
            df_normalised=df_normalised,
            tissue_indices_final=tissue_indices_final,
            height=height,
            width=width,
            out_dir=umap_dir,
        )
        if umap_run is not None:
            X_umap, umap_rgb, umap_mapper = umap_run
            np.save(umap_cache, X_umap)
            print(f"\n  UMAP coordinates saved to: {umap_cache}")
    else:
        # Try to load cached coordinates from a previous run
        if os.path.exists(umap_cache):
            X_umap = np.load(umap_cache)
            print(f"\n  [skipped] UMAP — loaded cached coordinates from: {umap_cache}")
        else:
            print(f"\n  [skipped] UMAP — no cached coordinates found at: {umap_cache}")

    if X_umap is None:
        print("\n  WARNING: UMAP coordinates unavailable — clustering step will be skipped.\n")

    # ------------------------------------------------------------------
    # STAGE 4: CLUSTERING
    # ------------------------------------------------------------------
    # K-means elbow method + HDBSCAN on the UMAP coordinates.
    # Requires X_umap to have been computed successfully.
    # ------------------------------------------------------------------

    cluster_dir = make_output_dir(args.output, "clustering")

    if not args.skip_cluster:
        if X_umap is not None:
            run_stage(
                "Clustering",
                _run_clustering,
                X_umap=X_umap,
                df_normalised=df_normalised,
                img_filtered=img_filtered,
                channel_names_filtered=channel_names_filtered,
                tissue_indices_final=tissue_indices_final,
                height=height,
                width=width,
                out_dir=cluster_dir,
                umap_dir=umap_dir,
                pca_dir=pca_dir,
                umap_mapper=umap_mapper,
            )
        else:
            print("\n  [skipped] Clustering — UMAP did not complete successfully.")
    else:
        print("\n  [skipped] Clustering (--skip-cluster flag set)")

    # ------------------------------------------------------------------
    # DONE
    # ------------------------------------------------------------------

    print(f"\n{'=' * 60}")
    print(f"  Pipeline complete. All outputs saved to:")
    print(f"  {args.output}")
    print(f"{'=' * 60}\n")


# =============================================================================
# STAGE WRAPPER FUNCTIONS
# =============================================================================
#
# Each _run_* function below calls the relevant module functions in the correct
# order and passes outputs from one function to the next within that stage.
# Keeping them separate from run_pipeline() makes each stage easy to test or
# re-run independently.
#

def _run_preprocessing(tiff_path: str, out_dir: str):
    """
    Preprocessing stage:
      1. Load OME-TIFF — extracts the 3D image array and channel names from the file
      2. Filter channels — drops 0TIC and any manual exclusions from config
      3. Apply mask — runs threshold → SOR → fill → BED to isolate tissue pixels
      4. Apply log1p normalisation — log(x+1) on the tissue pixel dataframe

    Returns (df_normalised, tissue_indices_final, height, width,
             img_filtered, channel_names_filtered)
    """
    # --- Step 1: Load OME-TIFF ---
    # Returns: img_raw (n_channels, H, W), all_channel_names (list of strings)
    img_raw, all_channel_names = preprocessing.load_image(tiff_path)

    # --- Step 2: Filter channels ---
    # Returns: img_filtered, channel_names_filtered
    img_filtered, channel_names_filtered = \
        preprocessing.filter_channels(img_raw, all_channel_names)

    # --- Step 3: Apply mask ---
    # All mask parameters are read from config.py — change them there.
    # MASK_CHANNEL  : which element to use for building the mask (0=Na, 1=Mg, etc.)
    # MASK_THRESHOLD: pixel intensity below which a pixel is treated as background
    # SMALL_OBJECT_REMOVAL, MASK_FILL_HOLES, BED_ITERATIONS: mask cleaning steps
    # SHOW_MASK_PREVIEW: set to 1 in config to display the 3-panel mask preview

    df, tissue_indices_final, height, width = preprocessing.apply_mask(
        img=img_filtered,
        img_filtered=img_filtered,
        channel_names_filtered=channel_names_filtered,
        channel_for_mask=config.MASK_CHANNEL,
        mask_threshold=config.MASK_THRESHOLD,
        fill_holes=config.MASK_FILL_HOLES,
        show_preview=bool(config.SHOW_MASK_PREVIEW),
        output_dir=out_dir,
    )

    # --- Step 5: Normalisation ---
    # Check data for negative values and prompt the user to confirm or override
    # the normalisation method before applying it.
    # Updates config.NORMALISATION in place based on the user's selection.

    # Apply the chosen normalisation ('log1p', 'arcsinh', or 'none')
    df_normalised = preprocessing.apply_normalisation(df)

    # --- Normalisation summary statistics ---
    print("\n  Raw counts — summary statistics:")
    print(df.describe().T[['mean', '50%', 'std']].to_string())
    print("\n  log1p-normalised — summary statistics:")
    print(df_normalised.describe().T[['mean', '50%', 'std']].to_string())

    # --- Step 6: Cache image data for Streamlit app ---
    # Saves img_filtered and channel names so the app can render
    # per-element views (log / linear / histogram) interactively.
    import json
    np.save(os.path.join(out_dir, "img_filtered.npy"), img_filtered.astype(np.float32))
    with open(os.path.join(out_dir, "channel_names.json"), "w") as f:
        json.dump(channel_names_filtered, f)
    # Also save tissue_indices and image dims for interactive cluster map in app
    np.save(os.path.join(out_dir, "tissue_indices.npy"), tissue_indices_final)
    with open(os.path.join(out_dir, "image_dims.json"), "w") as f:
        json.dump({"height": height, "width": width}, f)
    print(f"\n  Cached img_filtered + channel names for Streamlit app.")

    # --- Step 8: Tissue mask overlay (dissertation figure) ---
    # Publication-quality 2-panel figure: raw channel + overlay showing which
    # pixels were retained as tissue vs excluded as background.
    preprocessing.plot_mask_overlay(
        img_filtered=img_filtered,
        tissue_indices_final=tissue_indices_final,
        height=height,
        width=width,
        channel_for_mask=config.MASK_CHANNEL,
        channel_names_filtered=channel_names_filtered,
        output_dir=out_dir,
    )

    # --- Step 9: Log1p normalisation validation (dissertation figure) ---
    # Violin plots comparing raw vs log1p intensity distributions across
    # a representative subset of channels. Justifies the normalisation choice.
    preprocessing.plot_normalisation_comparison(
        df_raw=df,
        df_normalised=df_normalised,
        channel_names_filtered=channel_names_filtered,
        output_dir=out_dir,
    )

    return df_normalised, tissue_indices_final, height, width, img_filtered, channel_names_filtered


def _run_pca(
    df_normalised, tissue_indices_final, height, width, out_dir
):
    """
    PCA stage:
      1. Fit PCA on log1p-normalised data
      2. Scree plot
      3. Cumulative variance plot
      4. Covariance matrix heatmap
      5. Loading matrix (PC × element heatmap)
      6. 2D loading plot (arrows)
      7. PCA RGB spatial map

    Returns (X_pca, pca_object)
    """
    # Fit PCA; returns (pca_object, X_pca)
    pca_obj, X_pca = pca_analysis.run_pca(df_normalised)
    ch_names = df_normalised.columns.tolist()

    # Save loadings + variance for interactive app plots (loading plot, biplot)
    np.save(os.path.join(out_dir, "pca_loadings.npy"), pca_obj.components_)         # (n_components, n_channels)
    np.save(os.path.join(out_dir, "pca_explained_variance.npy"), pca_obj.explained_variance_ratio_)
    np.save(os.path.join(out_dir, "X_pca.npy"), X_pca.astype(np.float32))
    import json
    with open(os.path.join(out_dir, "pca_channel_names.json"), "w") as f:
        json.dump(ch_names, f)
    print(f"  Saved PCA loadings + scores for interactive app.")

    # --- Variance / structure plots ---
    pca_analysis.plot_scree(pca_obj, output_dir=out_dir)
    pca_analysis.plot_cumulative_variance(pca_obj, output_dir=out_dir)
    pca_analysis.plot_covariance_matrix_interactive(df_normalised, ch_names, pca=pca_obj, output_dir=out_dir)
    pca_analysis.plot_correlation_matrix_interactive(df_normalised, ch_names, output_dir=out_dir)
    pca_analysis.plot_loadings(pca_obj, ch_names, output_dir=out_dir)
    pca_analysis.plot_variance_per_element(pca_obj, ch_names, output_dir=out_dir)

    return X_pca, pca_obj


def _run_umap(
    df_normalised, tissue_indices_final, height, width, out_dir
):
    """
    UMAP stage:
      1. Run 3D UMAP on log1p-normalised data (GPU if available)
      2. Static RGB spatial map
      3. Interactive Plotly RGB spatial map (hover with element values)
      4. 3D scatter coloured by RGB
      5. 2D scatter per channel
      6. All-channels grid
      7. Dominant element map

    Returns X_umap (n_pixels × 3)
    """
    X_umap, umap_mapper = umap_reduction.run_umap(df_normalised)

    # plot_umap_rgb returns (rgb_image, umap_norm) — we need both for downstream plots
    rgb_image, umap_norm = umap_reduction.plot_umap_rgb(
        X_umap, tissue_indices_final, height, width,
        pixel_size_um=config.PIXEL_SIZE_UM,
        output_dir=out_dir, show_plot=False
    )

    # 3D scatter — interactive HTML + static PNG
    umap_reduction.plot_umap_3d_scatter(X_umap, umap_norm, output_dir=out_dir)
    umap_reduction.plot_umap_3d_scatter_png(X_umap, umap_norm, output_dir=out_dir)

    # 2D scatter — static PNG (RGB coloured)
    umap_reduction.plot_umap_2d_scatter_png(X_umap, umap_norm, output_dir=out_dir)

    # Density plots — hexbin + KDE contour (2D, dim1 vs dim2)
    umap_reduction.plot_umap_density(X_umap, output_dir=out_dir, show_plot=False)

    # Moran's I — spatial autocorrelation of UMAP dims (validates embedding quality)
    umap_reduction.plot_umap_morans_i(
        X_umap, tissue_indices_final, height, width,
        output_dir=out_dir, show_plot=False
    )

    return X_umap, rgb_image, umap_mapper


def _run_clustering(
    X_umap,
    df_normalised,
    img_filtered,
    channel_names_filtered,
    tissue_indices_final,
    height,
    width,
    out_dir,
    umap_dir=None,
    pca_dir=None,
    umap_mapper=None,
):
    """
    Clustering stage:
      1. K-means elbow method (to determine optimal K)
      2. K-means fit using the chosen K
      3. K-means spatial map
      4. K-means per-cluster elemental profiles (log2FC bar chart)
      5. Statistical tests + volcano plots
      6. HDBSCAN on UMAP coordinates
      7. UMAP-linked cluster plots
    """
    # Always run elbow plot (for reference)
    k_optimal, inertias = Clustering.run_kmeans_elbow(X_umap, output_dir=out_dir, show_plot=False)
    print(f"\n  K-means elbow suggests K = {k_optimal}")

    # Use forced K from config if set, otherwise use elbow result
    if config.KMEANS_FORCED_K is not None:
        k_final = config.KMEANS_FORCED_K
        print(f"  KMEANS_FORCED_K={k_final} set in config — overriding elbow suggestion.")
    else:
        k_final = k_optimal
        print(f"  Using elbow K = {k_final}")

    # Fit K-means with chosen K
    km_labels, df_normalised = Clustering.run_kmeans(X_umap, df_normalised, chosen_k=k_final)

    # Save cluster labels and the exact normalised data used for clustering
    # so the app can load them with guaranteed pixel alignment
    np.save(os.path.join(out_dir, "km_labels.npy"), km_labels)
    np.save(os.path.join(out_dir, "df_normalised.npy"),
            df_normalised[channel_names_filtered].values.astype(np.float32))

    # Spatial map of K-means labels
    Clustering.plot_kmeans_spatial(
        km_labels, tissue_indices_final, img_filtered, channel_names_filtered,
        None, height, width, output_dir=out_dir, show_plot=False
    )

    # Per-cluster elemental profiles (K-means) — log2FC bar chart
    Clustering.plot_cluster_profiles(
        km_labels, df_normalised, channel_names_filtered,
        output_dir=out_dir, show_plot=False, label='kmeans'
    )

    # Pre-compute and save all per-cluster stats for the app
    flat_log = df_normalised[channel_names_filtered].values.astype(np.float64)
    km_mean_profiles, km_tissue_mean, km_log_fc, km_counts, km_dominant, km_cluster_ids = \
        Clustering.save_cluster_summary(km_labels, flat_log, channel_names_filtered,
                                        out_dir, label='kmeans')
    Clustering.plot_per_cluster_charts(km_mean_profiles, km_log_fc, km_cluster_ids,
                                       channel_names_filtered, km_counts,
                                       out_dir, label='kmeans')
    Clustering.plot_cosine_distance_heatmap(km_mean_profiles, km_cluster_ids,
                                            out_dir, label='kmeans')

    Clustering.plot_silhouette_full(X_umap, km_labels, output_dir=out_dir)

    # ── HDBSCAN on UMAP coordinates ──────────────────────────────────────────
    # run_hdbscan now returns (labels, probabilities) — probabilities used for
    # the membership confidence spatial map below.
    print("\n  Running HDBSCAN on UMAP coordinates for comparison...")
    hdb_labels, hdb_probs = Clustering.run_hdbscan(X_umap, output_dir=out_dir)
    Clustering.plot_hdbscan_spatial(
        hdb_labels, tissue_indices_final, height, width,
        output_dir=out_dir, show_plot=False
    )

    # HDBSCAN cluster profiles + pre-computed summary
    Clustering.plot_cluster_profiles(
        hdb_labels, df_normalised, channel_names_filtered,
        output_dir=out_dir, show_plot=False, label='hdbscan'
    )
    hdb_mean_profiles, hdb_tissue_mean, hdb_log_fc, hdb_counts, hdb_dominant, hdb_cluster_ids = \
        Clustering.save_cluster_summary(hdb_labels, flat_log, channel_names_filtered,
                                        out_dir, label='hdbscan')
    Clustering.plot_per_cluster_charts(hdb_mean_profiles, hdb_log_fc, hdb_cluster_ids,
                                       channel_names_filtered, hdb_counts,
                                       out_dir, label='hdbscan')
    Clustering.plot_cosine_distance_heatmap(hdb_mean_profiles, hdb_cluster_ids,
                                            out_dir, label='hdbscan')

    Clustering.plot_hdbscan_membership_probability(
        hdb_probs, hdb_labels, tissue_indices_final,
        height, width, output_dir=out_dir
    )

    # ── Cross-method comparison ───────────────────────────────────────────────
    Clustering.plot_cluster_proportions(km_labels, hdb_labels, output_dir=out_dir)

    # ── UMAP overlays ────────────────────────────────────────────────────────
    if umap_dir is not None:
        # KDE contour cluster plots — K-means and HDBSCAN in one figure
        umap_reduction.plot_umap_kde_cluster(
            X_umap, km_labels, hdb_labels,
            km_log_fc=km_log_fc, hdb_log_fc=hdb_log_fc,
            channel_names=channel_names_filtered,
            km_cluster_ids=list(km_cluster_ids), hdb_cluster_ids=list(hdb_cluster_ids),
            output_dir=umap_dir, show_plot=False
        )
        # Interactive element cluster assignment
        umap_reduction.plot_umap_element_cluster_assignment(
            X_umap, df_normalised, channel_names_filtered, km_labels,
            output_dir=umap_dir, show_plot=False
        )



# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    # Parse command-line arguments
    parser = build_parser()
    args = parser.parse_args()

    # Apply config defaults before validation
    if args.input is None:
        args.input = config.TIFF_FILE
    if args.output is None:
        args.output = config.OUTPUT_DIR

    # Validate that input file exists before starting
    if not os.path.isfile(args.input):
        print(f"\n  ERROR: Input file not found: {args.input}\n")
        sys.exit(1)

    # Run the pipeline
    run_pipeline(args)

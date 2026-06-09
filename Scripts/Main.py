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
import PCA_Analysis as pca_analysis
import UMAP_Analysis as umap_reduction
import tSNE_Analysis
import Clustering
import Visualisation as visualisation
 
 
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
        "--metadata", "-m",
        type=str,
        default=None,
        help="Path to the metadata CSV file. Defaults to config.METADATA_CSV if not provided.",
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
        "--skip-tsne",
        action="store_true",
        default=False,
        help="Skip the tSNE step (saves time on large images).",
    )
    parser.add_argument(
        "--skip-cluster",
        action="store_true",
        default=False,
        help="Skip the clustering step.",
    )
    parser.add_argument(
        "--hdbscan-sweep",
        action="store_true",
        default=False,
        help=(
            "Run the full HDBSCAN parameter sweep instead of a single fit. "
            "Use this to determine final HDBSCAN parameters before setting "
            "them in config.py."
        ),
    )
    parser.add_argument(
        "--no-gpu",
        action="store_true",
        default=False,
        help="Force CPU UMAP (umap-learn) even if a GPU is available.",
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
    visualisation.set_publication_style()
 
    # Fall back to config.py values if CLI arguments were not provided
    if args.input is None:
        args.input = config.TIFF_FILE
    if args.metadata is None:
        args.metadata = config.METADATA_CSV
    if args.output is None:
        args.output = config.OUTPUT_DIR
 
    # Override GPU setting if --no-gpu flag was passed
    if args.no_gpu:
        config.USE_GPU = False
        print("  [config] USE_GPU overridden to False via --no-gpu flag.")
 
    # Create the root output directory
    os.makedirs(args.output, exist_ok=True)
    print(f"\n  Output directory: {args.output}")
 
    # ------------------------------------------------------------------
    # STAGE 1: PREPROCESSING
    # ------------------------------------------------------------------
    # Load the OME-TIFF, build the tissue mask, apply log1p normalisation.
    # Returns: df_normalised, tissue_indices_final, height, width, scale_suggestions
    # These are passed forward to every downstream stage.
    # ------------------------------------------------------------------
 
    preproc_dir = visualisation.make_output_dir(args.output, "preprocessing")
 
    preproc_result = run_stage(
        "Preprocessing",
        _run_preprocessing,
        tiff_path=args.input,
        metadata_path=args.metadata,
        out_dir=preproc_dir,
    )
 
    if preproc_result is None:
        # Preprocessing is a hard dependency — cannot continue without it
        print("\n  FATAL: Preprocessing failed. Cannot continue.\n")
        sys.exit(1)
 
    df_normalised, tissue_indices_final, height, width, scale_suggestions = preproc_result
 
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
 
    pca_dir = visualisation.make_output_dir(args.output, "pca")
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
 
    umap_dir = visualisation.make_output_dir(args.output, "umap")
    umap_result = None
 
    if not args.skip_umap:
        umap_result = run_stage(
            "UMAP",
            _run_umap,
            df_normalised=df_normalised,
            tissue_indices_final=tissue_indices_final,
            height=height,
            width=width,
            scale_suggestions=scale_suggestions,
            out_dir=umap_dir,
        )
    else:
        print("\n  [skipped] UMAP (--skip-umap flag set)")
 
    if umap_result is None:
        print("\n  WARNING: UMAP failed — clustering step will be skipped.\n")
        X_umap = None
    else:
        X_umap = umap_result
 
    # ------------------------------------------------------------------
    # STAGE 4: tSNE
    # ------------------------------------------------------------------
    # Subsampled tSNE for thesis comparison. Optional.
    # ------------------------------------------------------------------
 
    tsne_dir = visualisation.make_output_dir(args.output, "tsne")
 
    if not args.skip_tsne:
        run_stage(
            "tSNE Analysis",
            _run_tsne,
            df_normalised=df_normalised,
            tissue_indices_final=tissue_indices_final,
            height=height,
            width=width,
            scale_suggestions=scale_suggestions,
            out_dir=tsne_dir,
        )
    else:
        print("\n  [skipped] tSNE (--skip-tsne flag set)")
 
    # ------------------------------------------------------------------
    # STAGE 5: CLUSTERING
    # ------------------------------------------------------------------
    # K-means elbow method + HDBSCAN on the UMAP coordinates.
    # Requires X_umap to have been computed successfully.
    # ------------------------------------------------------------------
 
    cluster_dir = visualisation.make_output_dir(args.output, "clustering")
 
    if not args.skip_cluster:
        if X_umap is not None:
            run_stage(
                "Clustering",
                _run_clustering,
                X_umap=X_umap,
                df_normalised=df_normalised,
                tissue_indices_final=tissue_indices_final,
                height=height,
                width=width,
                out_dir=cluster_dir,
                run_sweep=args.hdbscan_sweep,
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
 
def _run_preprocessing(tiff_path: str, metadata_path: str, out_dir: str):
    """
    Preprocessing stage:
      1. Load OME-TIFF — extracts the 3D image array and channel names from the file
      2. Filter channels — drops 0TIC and any manual exclusions from config
      3. Load metadata Excel — gets pixel size, display ranges, and log/linear per channel
      4. Apply mask — runs threshold → SOR → fill → BED to isolate tissue pixels
      5. Apply log1p normalisation — log(x+1) on the tissue pixel dataframe
      6. Visualise channels — saves element maps coloured by metadata display ranges
 
    Returns (df_normalised, tissue_indices_final, height, width, scale_suggestions)
    """
    # --- Step 1: Load OME-TIFF ---
    # Returns: img_raw (n_channels, H, W), all_channel_names (list of strings)
    img_raw, all_channel_names = preprocessing.load_image(tiff_path)
 
    # --- Step 2: Filter channels ---
    # Returns: img (after dropping TIC), img_filtered (after all exclusions),
    #          channel_names, channel_names_filtered, qupath_colours
    img, img_filtered, channel_names, channel_names_filtered, qupath_colours = \
        preprocessing.filter_channels(img_raw, all_channel_names)
 
    # --- Step 3: Load metadata CSV ---
    # Returns: threshold_lookup, channel_ranges
    threshold_lookup, channel_ranges = \
        preprocessing.load_metadata(metadata_path, channel_names_filtered)
 
    # --- Step 4: Apply mask ---
    # All mask parameters are read from config.py — change them there.
    # MASK_CHANNEL  : which element to use for building the mask (0=Na, 1=Mg, etc.)
    # MASK_THRESHOLD: pixel intensity below which a pixel is treated as background
    # SMALL_OBJECT_REMOVAL, MASK_FILL_HOLES, BED_ITERATIONS: mask cleaning steps
    # SHOW_MASK_PREVIEW: set to 1 in config to display the 3-panel mask preview
 
    df, tissue_indices_final, height, width = preprocessing.apply_mask(
        img=img,
        img_filtered=img_filtered,
        channel_names_filtered=channel_names_filtered,
        channel_for_mask=config.MASK_CHANNEL,
        mask_threshold=config.MASK_THRESHOLD,
        fill_holes=config.MASK_FILL_HOLES,
        show_preview=bool(config.SHOW_MASK_PREVIEW),
        output_dir=out_dir,
    )
 
    # --- Step 4b: Suggest log/linear scale per channel ---
    # Analyses the raw pixel distributions statistically to suggest display scales.
    # This is used downstream in UMAP/tSNE scatter plots.
    scale_suggestions = preprocessing.suggest_scale(
        img_filtered=img_filtered,
        channel_names_filtered=channel_names_filtered,
        output_dir=out_dir,
        show_plot=False,
    )
 
    # --- Step 5: Log1p normalisation ---
    # log(x + 1) applied element-wise — the only normalisation in this pipeline
    df_normalised = preprocessing.apply_log1p(df)
 
    # --- Step 6: Visualise channels ---
    # Shows all element maps coloured with custom colourmaps and metadata ranges
    preprocessing.visualise_channels(
        img_filtered=img_filtered,
        channel_names_filtered=channel_names_filtered,
        channel_ranges=channel_ranges,
        qupath_colours=qupath_colours,
        output_dir=out_dir,
        show_plot=False,   # don't open a window — just save to file
    )
 
    return df_normalised, tissue_indices_final, height, width, scale_suggestions
 
 
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
    # Fit PCA; returns (X_pca, pca_object, explained_variance_ratio)
    pca_obj, X_pca = pca_analysis.run_pca(df_normalised)
 
    pca_analysis.plot_scree(evr, out_dir=out_dir)
    pca_analysis.plot_cumulative_variance(evr, out_dir=out_dir)
    pca_analysis.plot_covariance_matrix(df_normalised, out_dir=out_dir)
    pca_analysis.plot_loading_matrix(pca_obj, df_normalised.columns.tolist(), out_dir=out_dir)
    pca_analysis.plot_loading_2d(pca_obj, df_normalised.columns.tolist(), out_dir=out_dir)
    pca_analysis.plot_pca_rgb(
        X_pca, tissue_indices_final, height, width, out_dir=out_dir
    )
 
    return X_pca, pca_obj
 
 
def _run_umap(
    df_normalised, tissue_indices_final, height, width, scale_suggestions, out_dir
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
    # Run UMAP — GPU (cuML) if config.USE_GPU=True and cuML is installed,
    # otherwise falls back to CPU umap-learn automatically.
    X_umap = umap_reduction.run_umap(df_normalised)
 
    # plot_umap_rgb returns (rgb_image, umap_norm) — we need both for downstream plots
    rgb_image, umap_norm = umap_reduction.plot_umap_rgb(
        X_umap, tissue_indices_final, height, width,
        output_dir=out_dir, show_plot=False
    )
 
    # Interactive map needs rgb_image and umap_norm from above, plus channel names
    umap_reduction.plot_umap_rgb_interactive(
        rgb_image, X_umap, tissue_indices_final, df_normalised,
        df_normalised.columns.tolist(), height, width,
        output_dir=out_dir
    )
 
    # 3D scatter needs umap_norm from plot_umap_rgb
    umap_reduction.plot_umap_3d_scatter(X_umap, umap_norm, output_dir=out_dir)
 
    # Per-channel scatter plots — pass the full scale_suggestions dict
    for ch in df_normalised.columns:
        umap_reduction.plot_umap_by_channel(
            X_umap, df_normalised, ch,
            scale_suggestions=scale_suggestions,
            output_dir=out_dir,
        )
 
    # All-channels grid — needs channel_names_filtered as a list
    umap_reduction.plot_umap_all_channels(
        X_umap, df_normalised, df_normalised.columns.tolist(),
        scale_suggestions, output_dir=out_dir, show_plot=False
    )
 
    # Dominant element map — no tissue_indices/height/width needed
    umap_reduction.plot_umap_dominant_element(
        X_umap, df_normalised, output_dir=out_dir, show_plot=False
    )
 
    # Element key — RGB map side by side with per-element spatial thumbnails
    # Use this to identify which colour in the RGB map corresponds to which element
    umap_reduction.plot_element_key(
        rgb_image, df_normalised, tissue_indices_final,
        height, width, output_dir=out_dir, show_plot=False
    )
 
    return X_umap
 
 
def _run_tsne(
    df_normalised, tissue_indices_final, height, width, scale_suggestions, out_dir
):
    """
    tSNE stage:
      1. Subsample to TSNE_MAX_PIXELS pixels
      2. Run tSNE on subsampled data
      3. RGB spatial map (subsampled pixels only)
      4. 2D scatter per channel
      5. All-channels grid
      6. Dominant element map
 
    The subsampled indices are used for the spatial map; the full tissue
    indices remain unchanged and are used for downstream stages (UMAP,
    clustering) which operate on all tissue pixels.
    """
    # Subsample — returns df_sub, tissue_indices_sub, df_full, tissue_indices_full
    df_sub, tissue_indices_sub, _, _ = tSNE_Analysis.subsample(
        df_normalised, tissue_indices_final
    )
 
    # Run tSNE on the subsampled data
    X_tsne = tSNE_Analysis.run_tsne(df_sub)
 
    tSNE_Analysis.plot_tsne_rgb(
        X_tsne, tissue_indices_sub, height, width, out_dir=out_dir
    )
    tSNE_Analysis.plot_tsne_all_channels(
        X_tsne, df_sub, scale_suggestions, out_dir=out_dir
    )
    tSNE_Analysis.plot_tsne_dominant_element(
        X_tsne, df_sub, tissue_indices_sub, height, width, out_dir=out_dir
    )
 
 
def _run_clustering(
    X_umap,
    df_normalised,
    tissue_indices_final,
    height,
    width,
    out_dir,
    run_sweep: bool = False,
):
    """
    Clustering stage:
      1. K-means elbow method (to determine optimal K)
      2. K-means fit using the chosen K
      3. K-means spatial map
      4. K-means per-channel intensity maps
      5. HDBSCAN parameter sweep (if --hdbscan-sweep flag set)
         OR single HDBSCAN fit using config parameters
      6. HDBSCAN spatial map
      7. Interactive HDBSCAN spatial map (Plotly)
      8. K-means vs HDBSCAN comparison figure
      9. ARI / AMI agreement metrics
 
    NOTE: HDBSCAN and K-means parameters in config.py are PLACEHOLDERS.
    Run with --hdbscan-sweep first on your dataset to choose final parameters,
    then update HDBSCAN_MIN_CLUSTER_SIZE and HDBSCAN_MIN_SAMPLES in config.py.
    """
    # --- K-means ---
    # Run elbow method across K=1..KMEANS_MAX_K and find the elbow
    k_optimal, inertias = Clustering.run_kmeans_elbow(X_umap, out_dir=out_dir)
    print(f"\n  K-means elbow suggests K = {k_optimal}")
 
    # Fit K-means with the optimal K
    km_labels = Clustering.run_kmeans(X_umap, k=k_optimal)
 
    # Spatial map of K-means labels
    Clustering.plot_kmeans_spatial(
        km_labels, tissue_indices_final, height, width, out_dir=out_dir
    )
 
    # Per-channel mean intensity per cluster
    Clustering.plot_kmeans_per_channel(
        km_labels, df_normalised, out_dir=out_dir
    )
 
    # --- HDBSCAN ---
    if run_sweep:
        # Full parameter sweep — produces a grid of figures showing how the
        # number of clusters and noise fraction change with parameters.
        # Use this output to choose final HDBSCAN_MIN_CLUSTER_SIZE and
        # HDBSCAN_MIN_SAMPLES values for config.py.
        print("\n  Running HDBSCAN parameter sweep (this may take several minutes)...")
        Clustering.run_hdbscan_sweep(X_umap, out_dir=out_dir)
        print(
            "\n  Sweep complete. Review the figures in the clustering/ folder,\n"
            "  choose your preferred parameters, and update config.py before\n"
            "  running the pipeline again without --hdbscan-sweep."
        )
        # Do not run a single HDBSCAN fit after a sweep — parameters not finalised
        return
 
    # Single HDBSCAN fit using config.py parameters
    hdb_labels = Clustering.run_hdbscan(X_umap)
 
    n_clusters = len(set(hdb_labels) - {-1})
    n_noise = (hdb_labels == -1).sum()
    print(f"\n  HDBSCAN: {n_clusters} clusters, {n_noise:,} noise pixels")
 
    # Static spatial map
    Clustering.plot_hdbscan_spatial(
        hdb_labels, tissue_indices_final, height, width, out_dir=out_dir
    )
 
    # Interactive Plotly spatial map (HTML file)
    Clustering.plot_hdbscan_interactive(
        hdb_labels, X_umap, tissue_indices_final, height, width,
        df_normalised, out_dir=out_dir
    )
 
    # Side-by-side K-means vs HDBSCAN
    Clustering.plot_kmeans_vs_hdbscan(
        km_labels, hdb_labels, tissue_indices_final, height, width, out_dir=out_dir
    )
 
    # ARI / AMI agreement metrics (excludes noise pixels)
    Clustering.compute_agreement(km_labels, hdb_labels)
 
 
# =============================================================================
# ENTRY POINT
# =============================================================================
 
if __name__ == "__main__":
    # Parse command-line arguments
    parser = build_parser()
    args = parser.parse_args()
 
    # Validate that input files exist before starting
    if not os.path.isfile(args.input):
        print(f"\n  ERROR: Input file not found: {args.input}\n")
        sys.exit(1)
    if not os.path.isfile(args.metadata):
        print(f"\n  ERROR: Metadata file not found: {args.metadata}\n")
        sys.exit(1)
 
    # Run the pipeline
    run_pipeline(args)
# Dimensionality Reduction Pipeline for Metallomic Imaging
### MSc Thesis: London Metallomic Facility

A Python pipeline for unsupervised dimensionality reduction and spatial clustering of LA-ICP-TOF-MS metallomic imaging data. Applied to a triple-negative breast cancer (TNBC) tissue section with 299,018 tissue pixels across 58 elemental/protein channels.

---

## Overview

LA-ICP-TOF-MS (Laser Ablation Inductively Coupled Plasma Time-of-Flight Mass Spectrometry) produces high-dimensional spatial metallomic images where each pixel carries a full mass spectrum. This pipeline processes those images through the following stages:

1. **Preprocessing**: tissue masking, channel filtering, log1p normalisation
2. **PCA**: elemental loading analysis, covariance/correlation matrices
3. **UMAP**: 3D embedding, RGB spatial map, Moran's I spatial validation
4. **Clustering**: K-means (elbow method) + HDBSCAN on UMAP coordinates, log₂ fold-change profiling
5. **Streamlit App**: interactive explorer for all pipeline outputs

---

## Repository Structure

```
Pipeline/
└── Scripts/
    ├── config.py           # All parameters (paths, thresholds, UMAP/clustering settings)
    ├── preprocessing.py    # Image loading, tissue masking, log1p normalisation
    ├── pca_analysis.py     # PCA fitting, scree plot, loading plots, covariance matrix
    ├── umap_reduction.py   # UMAP embedding, RGB spatial map, Moran's I
    ├── Clustering.py       # K-means elbow, HDBSCAN, log2FC profiling, cosine heatmap
    ├── main.py             # Pipeline orchestrator — runs all stages end to end
    ├── app.py              # Streamlit interactive explorer
    └── requirements.txt    # Python dependencies
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Tolga763/Msc_Thesis.git
cd Msc_Thesis/Pipeline
```

### 2. Create the conda environment

```bash
conda create -n metalomicDR python=3.10
conda activate metalomicDR
pip install -r Scripts/requirements.txt
```

---

## Running the Pipeline

Edit `Scripts/config.py` to set your input file path and output directory, then:

```bash
cd Scripts
python main.py
```

Or pass paths directly:

```bash
python main.py --input /path/to/image.ome.tiff --output /path/to/results
```

### Optional flags

| Flag | Effect |
|------|--------|
| `--skip-pca` | Skip PCA stage |
| `--skip-umap` | Skip UMAP (loads cached coordinates if available) |
| `--skip-cluster` | Skip clustering stage |

### Output structure

```
results/
├── preprocessing/    mask overlay, normalisation violin plots
├── pca/              scree plot, loading plots, covariance/correlation matrices
├── umap/             RGB spatial map, 3D scatter, density plot, Moran's I
└── clustering/       K-means + HDBSCAN spatial maps, per-cluster profiles,
                      silhouette plot, cosine distance heatmap, KDE overlays
```

---

## Running the Streamlit App

```bash
cd Scripts
streamlit run app.py
```

Then open `http://localhost:8501` in your browser. Point the app at a results folder using the sidebar path selector.

The app provides interactive views of all pipeline outputs including per-element UMAP colouring, lasso selection for custom cluster profiling, and interactive correlation/covariance matrices.

---

## Key Parameters (config.py)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `TIFF_FILE` | — | Path to input OME-TIFF |
| `OUTPUT_DIR` | — | Root output directory |
| `MASK_CHANNEL` | `0` (Na) | Channel used to build tissue mask |
| `MASK_THRESHOLD` | — | Intensity threshold for masking |
| `UMAP_N_NEIGHBORS` | `30` | UMAP neighbourhood size |
| `UMAP_MIN_DIST` | `0.0` | UMAP minimum distance |
| `KMEANS_FORCED_K` | `None` | Override elbow K (set to int to fix K) |
| `HDBSCAN_MIN_CLUSTER_SIZE` | `2000` | HDBSCAN minimum cluster size |

---

## Dataset

- **Tissue**: Triple-negative breast cancer (TNBC) FFPE section
- **Instrument**: LA-ICP-TOF-MS
- **Image size**: 929 × 919 px
- **Tissue pixels**: 299,018
- **Channels**: 58 (endogenous elements + metal-tagged antibodies)
- **Normalisation**: log1p (no z-scoring)

---

## Results Summary

| Method | Clusters | Notes |
|--------|----------|-------|
| K-means | 3 | Elbow at k=3; silhouette peak at k=2 |
| HDBSCAN | 2 | 149 noise pixels (0.05%) |
| UMAP Moran's I | — | UMAP1: 0.80, UMAP2: 0.78, UMAP3: 0.89 |
| PC1 variance | — | 38.7% explained |

---

## Author

Tolga Kiymak, MSc Thesis, King's College London  
London Metallomic Facility

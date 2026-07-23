# =============================================================================
# app.py  Metallomic Imaging Explorer (Streamlit)
#
# Layout: left sidebar = all controls, main panel = output.
# Preprocessing tab: element selector → renders log / linear / histogram live.
#
# Run with:   streamlit run app.py
# =============================================================================
import base64
import os
import json
import glob
import PIL
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import streamlit as st
import plotly.graph_objects as go
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
# =============================================================================
# PAGE CONFIG
# =============================================================================
st.set_page_config(
    page_title="London Metallomic Facility DR Explorer",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={},
)
st.markdown("""
    <style>
    [data-testid="stToolbar"] { display: none; }
    [data-testid="collapsedControl"] { display: none !important; }
    [data-testid="stSidebarCollapseButton"] { display: none !important; }
    section[data-testid="stSidebar"] {
        display: block !important;
        transform: none !important;
        min-width: 320px !important;
        width: 320px !important;
        visibility: visible !important;
    }
    section[data-testid="stSidebar"][aria-expanded="false"] {
        display: block !important;
        transform: none !important;
        min-width: 320px !important;
        width: 320px !important;
        visibility: visible !important;
    }
    </style>
""", unsafe_allow_html=True)
# =============================================================================
# CSS
# =============================================================================
st.markdown("""
<style>
    .title-bar {
        background: #2c3e50;
        color: white;
        padding: 14px 28px;
        border-radius: 8px;
        margin-bottom: 18px;
        font-size: 1.3rem;
        font-weight: 700;
    }
    /* Hide the hover toolbar that appears on st.image() */
    [data-testid="stImage"] ~ div button,
    [data-testid="stImageContainer"] button { display: none !important; }
</style>
""", unsafe_allow_html=True)
# =============================================================================
# HELPERS
# =============================================================================
@st.cache_data
def load_img_filtered(preproc_dir):
    path = os.path.join(preproc_dir, "img_filtered.npy")
    if os.path.exists(path):
        return np.load(path)
    return None

@st.cache_data
def load_spatial_data(preproc_dir):
    idx_path = os.path.join(preproc_dir, "tissue_indices.npy")
    dim_path = os.path.join(preproc_dir, "image_dims.json")
    if os.path.exists(idx_path) and os.path.exists(dim_path):
        indices = np.load(idx_path)
        with open(dim_path) as f:
            dims = json.load(f)
        return indices, int(dims["height"]), int(dims["width"])
    return None, None, None


def build_cluster_plotly(km_labels, tissue_indices, H, W, n_clusters,
                         selected_cluster=None, na_img=None, max_scatter=60_000):
    """
    go.Image for visual quality + invisible Scattergl overlay for lasso selection.
    Lasso captures point indices → customdata maps back to tissue_indices positions.
    """
    import sys as _sys, importlib as _il
    _sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    _cfg = _il.import_module("config")
    _pal = _cfg.get_cluster_colours(n_clusters)

    def _hex_to_rgb(h):
        h = h.lstrip('#')
        return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)

    rows = tissue_indices // W
    cols = tissue_indices % W

    # Na greyscale base layer (RGBA)
    canvas = np.zeros((H, W, 4), dtype=np.uint8)
    if na_img is not None:
        _na  = na_img.astype(float)
        _p99 = np.percentile(_na, 99)
        _na  = np.clip(_na / (_p99 + 1e-9), 0, 1)
        grey = (_na * 180).astype(np.uint8)
        canvas[:, :, 0] = grey
        canvas[:, :, 1] = grey
        canvas[:, :, 2] = grey
        canvas[:, :, 3] = 255

    # Paint cluster colours on top
    for cid in range(n_clusters):
        mask = km_labels == cid
        pr   = rows[mask]
        pc   = cols[mask]
        if selected_cluster is None or cid == int(selected_cluster):
            r, g, b = _hex_to_rgb(_pal[cid % len(_pal)])
            a = 220
        else:
            r, g, b = 30, 30, 30
            a = 200
        canvas[pr, pc] = [r, g, b, a]

    title_txt = (f"K-Means  k={n_clusters}  ·  Cluster {selected_cluster} highlighted"
                 if selected_cluster is not None
                 else f"K-Means  k={n_clusters}  ·  All clusters")

    fig = go.Figure(go.Image(z=canvas))

    # Invisible scatter overlay for lasso — subsampled for performance
    n_tissue = len(tissue_indices)
    if n_tissue > max_scatter:
        rng      = np.random.default_rng(42)
        sub_idx  = rng.choice(n_tissue, max_scatter, replace=False)
    else:
        sub_idx  = np.arange(n_tissue)

    fig.add_trace(go.Scattergl(
        x=cols[sub_idx].astype(float),
        y=rows[sub_idx].astype(float),
        mode='markers',
        marker=dict(size=4, opacity=0, color='rgba(0,0,0,0)'),
        customdata=sub_idx,   # position in tissue_indices / km_labels / flat_log
        showlegend=False,
        hoverinfo='skip',
    ))

    fig.update_layout(
        xaxis=dict(showticklabels=False, showgrid=False, zeroline=False,
                   scaleanchor=None),
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        plot_bgcolor="black",
        paper_bgcolor="black",
        margin=dict(l=0, r=0, t=24, b=0),
        height=420,
        title=dict(text=title_txt, font=dict(color="white", size=12), x=0.5),
        dragmode="lasso",
    )
    return fig


@st.cache_data
def load_channel_names(preproc_dir):
    path = os.path.join(preproc_dir, "channel_names.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None

def show_image(path, caption="", max_px=4000):
    if os.path.exists(path):
        img = Image.open(path)
        if max(img.size) > max_px:
            img.thumbnail((max_px, max_px), Image.LANCZOS)
        st.image(img, caption=caption, width="stretch")
    else:
        st.info(f"Output not found: `{os.path.basename(path)}`")

def embed_html(path, height=700):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            html = f.read()
        st.iframe(html, height=height)
    else:
        st.info(f"Interactive output not found: `{os.path.basename(path)}`")

OVERLAY_PALETTE = [
    (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.4, 1.0), (0.0, 1.0, 1.0),
    (1.0, 0.0, 1.0), (1.0, 1.0, 0.0), (1.0, 0.5, 0.0), (0.6, 0.0, 1.0),
    (0.0, 1.0, 0.5), (1.0, 0.0, 0.4),
]
PALETTE_NAMES = [
    "Red", "Green", "Blue", "Cyan", "Magenta",
    "Yellow", "Orange", "Purple", "Spring green", "Rose"
]

def apply_tissue_mask(img_filtered, tissue_indices, H, W):
    """
    Returns a copy of img_filtered (n_channels, H, W) with all background
    pixels set to zero, using the flat tissue_indices from preprocessing.
    """
    masked = np.zeros_like(img_filtered)
    flat_img = img_filtered.reshape(img_filtered.shape[0], -1)
    flat_masked = masked.reshape(masked.shape[0], -1)
    flat_masked[:, tissue_indices] = flat_img[:, tissue_indices]
    return masked

def render_multi_element_overlay(img_filtered, channel_names, selected_elements,
                                  tissue_indices=None, H=None, W=None):
    if tissue_indices is not None and H is not None and W is not None:
        img_filtered = apply_tissue_mask(img_filtered, tissue_indices, H, W)
    H = img_filtered.shape[1]
    W = img_filtered.shape[2]
    composite = np.zeros((H, W, 3), dtype=float)
    colour_map = {}
    for i, elem in enumerate(selected_elements):
        idx    = channel_names.index(elem)
        data   = img_filtered[idx].astype(float)
        # Use only tissue pixels for scaling
        if tissue_indices is not None:
            tissue_vals = data.flatten()[tissue_indices]
            vmax = float(np.percentile(tissue_vals, 99)) if len(tissue_vals) > 0 else float(data.max())
        else:
            nz = data[data > 0]
            vmax = float(np.percentile(nz, 99)) if len(nz) > 0 else float(data.max())
        norm   = np.clip(data / (vmax + 1e-10), 0, 1)
        colour = OVERLAY_PALETTE[i % len(OVERLAY_PALETTE)]
        colour_map[elem] = (colour, PALETTE_NAMES[i % len(PALETTE_NAMES)])
        for c in range(3):
            composite[:, :, c] += norm * colour[c]
    composite = np.clip(composite, 0, 1)
    return composite, colour_map


# =============================================================================
# SIDEBAR
# =============================================================================
with st.sidebar:
    st.markdown("### London Metallomic Facility DR Explorer")
    st.markdown("---")
    out = st.text_input("Results folder", value="../Results/tnbc_run1").rstrip("/")
    st.markdown("---")
    stage = st.selectbox(
        "Select analysis stage",
        ["Preprocessing", "PCA", "UMAP", "Clustering"],
    )
    st.markdown("---")
    preproc_dir = os.path.join(out, "preprocessing")
    pca_dir     = os.path.join(out, "pca")
    umap_dir    = os.path.join(out, "umap")
    cluster_dir = os.path.join(out, "clustering")

    if stage == "Preprocessing":
        preproc_view = st.selectbox(
            "Select view",
            ["Multi-element Overlay",
             "Tissue Mask", "Mask Overlay",
             "Normalisation Comparison"],
        )
        if preproc_view == "Multi-element Overlay":
            ch_names = load_channel_names(preproc_dir)
            if ch_names:
                st.markdown("**Select elements to overlay** (up to 10)")
                selected_elements = st.multiselect(
                    "Elements", ch_names, default=ch_names[:3], max_selections=10)
                st.markdown("---")
                st.markdown("**Colour assignments:**")
                for i, elem in enumerate(selected_elements):
                    r, g, b = OVERLAY_PALETTE[i % len(OVERLAY_PALETTE)]
                    hex_col = '#{:02x}{:02x}{:02x}'.format(int(r*255), int(g*255), int(b*255))
                    st.markdown(
                        f'<span style="color:{hex_col};">■</span> {elem} → '
                        f'{PALETTE_NAMES[i % len(PALETTE_NAMES)]}',
                        unsafe_allow_html=True)
            else:
                st.warning("channel_names.json not found — re-run pipeline.")
                selected_elements = []
    elif stage == "PCA":
        pca_view = st.selectbox(
            "Select view",
            ["Scree Plot", "Cumulative Variance",
             "Correlation Matrix", "Covariance Matrix",
             "Loading Plot",
             "Variance per Element"],
        )
    elif stage == "UMAP":
        umap_view = st.selectbox(
            "Select view",
            [
                "RGB Spatial Map",
                "2D Scatter (PNG)",
                "3D Scatter (PNG)",
                "3D Scatter",
                "Density / Hexbin",
                "KDE Cluster Contour (K-Means)",
                "KDE Cluster Contour (HDBSCAN)",
                "Element Cluster Assignment",
                "Element Intensity Explorer",
                "Moran's I",
            ],
        )
        if umap_view == "Element Intensity Explorer":
            _umap_ch_names = load_channel_names(preproc_dir)
            if _umap_ch_names:
                st.markdown("**Select element**")
                st.selectbox("Element", _umap_ch_names, key="umap_elem_sel")
    elif stage == "Clustering":
        km_view = st.selectbox(
            "Select view",
            [
                # ── K-means ──────────────────────────────
                "Elbow + Silhouette",
                "Silhouette Plot",
                "Spatial Map",
                "Cluster Explorer",
                "Cluster Profiles",
                "Statistical Tests",
                # ── HDBSCAN ──────────────────────────────
                "HDBSCAN Spatial Map",
                "HDBSCAN Cluster Explorer",
                "HDBSCAN Cluster Profiles",
                "HDBSCAN Membership Probability",
                "HDBSCAN Condensed Tree",
                # ── Comparison ───────────────────────────
                "Cluster Proportions",
            ],
        )
        if km_view == "Cluster Explorer":
            ch_names_cl    = load_channel_names(preproc_dir)
            km_labels_path = os.path.join(cluster_dir, "km_labels.npy")
            if os.path.exists(km_labels_path) and ch_names_cl:
                km_labels_loaded  = np.load(km_labels_path)
                n_clusters_loaded = len(np.unique(km_labels_loaded))
                if "km_cluster_sel" not in st.session_state:
                    st.session_state["km_cluster_sel"] = 0
                _radio_val = st.radio(
                    "Select cluster",
                    options=list(range(n_clusters_loaded)),
                    format_func=lambda x: f"Cluster {x}",
                    horizontal=True,
                    index=int(st.session_state["km_cluster_sel"]),
                )
                st.session_state["km_cluster_sel"] = int(_radio_val)
                selected_cluster = int(_radio_val)
                st.markdown("---")
                _max_ch = len(load_channel_names(preproc_dir) or []) or 60
                top_n = st.slider("Show top N elements", 5, _max_ch, min(20, _max_ch))
        if km_view == "HDBSCAN Cluster Explorer":
            ch_names_cl       = load_channel_names(preproc_dir)
            hdb_labels_path   = os.path.join(cluster_dir, "hdbscan_labels.npy")
            if os.path.exists(hdb_labels_path) and ch_names_cl:
                hdb_labels_loaded  = np.load(hdb_labels_path)
                _hdb_ids           = sorted([c for c in np.unique(hdb_labels_loaded) if c != -1])
                n_hdb_clusters     = len(_hdb_ids)
                if "hdb_cluster_sel" not in st.session_state:
                    st.session_state["hdb_cluster_sel"] = _hdb_ids[0] if _hdb_ids else 0
                _hdb_radio = st.radio(
                    "Select cluster",
                    options=_hdb_ids,
                    format_func=lambda x: f"Cluster {x}",
                    horizontal=True,
                    index=_hdb_ids.index(int(st.session_state["hdb_cluster_sel"]))
                    if int(st.session_state["hdb_cluster_sel"]) in _hdb_ids else 0,
                )
                st.session_state["hdb_cluster_sel"] = int(_hdb_radio)
                hdb_selected_cluster = int(_hdb_radio)
                st.markdown("---")
                _max_ch_hdb = len(load_channel_names(preproc_dir) or []) or 60
                top_n_hdb = st.slider("Show top N elements", 5, _max_ch_hdb, min(20, _max_ch_hdb), key="hdb_top_n")

# =============================================================================
# MAIN PANEL
# =============================================================================
st.markdown(f'<div class="title-bar"> Metallomic Imaging Explorer {stage}</div>',
            unsafe_allow_html=True)

if not os.path.isdir(out):
    st.warning(f"Results folder not found: `{out}`  \nUpdate the path in the sidebar.")
    st.stop()

# SHARED HELPERS (must be defined before the stage if/elif chain) 

@st.cache_data
def load_cluster_summary(cluster_dir, label):
    """Load pre-computed cluster summary saved by pipeline — no heavy computation."""
    import json
    base = os.path.join(cluster_dir, label)
    required = [f'{base}_mean_profiles.npy', f'{base}_tissue_mean.npy',
                f'{base}_log_fc.npy', f'{base}_counts.json',
                f'{base}_dominant.json', f'{base}_cluster_ids.json']
    if not all(os.path.exists(p) for p in required):
        return None
    mean_profiles = np.load(f'{base}_mean_profiles.npy')
    tissue_mean   = np.load(f'{base}_tissue_mean.npy')
    log_fc        = np.load(f'{base}_log_fc.npy')
    with open(f'{base}_counts.json') as f:
        counts = {int(k): v for k, v in json.load(f).items()}
    with open(f'{base}_dominant.json') as f:
        dominant = {int(k): v for k, v in json.load(f).items()}
    with open(f'{base}_cluster_ids.json') as f:
        cluster_ids = json.load(f)
    return mean_profiles, tissue_mean, log_fc, counts, dominant, cluster_ids


# PREPROCESSING 
if stage == "Preprocessing":
    if not os.path.isdir(preproc_dir):
        st.info("No preprocessing outputs found.")
    elif preproc_view == "Multi-element Overlay":
        st.markdown("### Multi-element Spatial Overlay")
        if not selected_elements:
            st.info("Select at least one element in the sidebar.")
        else:
            img = load_img_filtered(preproc_dir)
            ch  = load_channel_names(preproc_dir)
            if img is None or ch is None:
                st.warning("img_filtered.npy not found. Re-run the pipeline.")
            else:
                tissue_indices, H, W = load_spatial_data(preproc_dir)
                composite, colour_map = render_multi_element_overlay(
                    img, ch, selected_elements,
                    tissue_indices=tissue_indices, H=H, W=W)
                fig, ax = plt.subplots(figsize=(12, 8), facecolor='black')
                ax.imshow(composite, origin='upper'); ax.axis('off')
                import matplotlib.patches as mpatches
                legend_patches = [mpatches.Patch(color=colour, label=elem)
                                  for elem, (colour, _) in colour_map.items()]
                ax.legend(handles=legend_patches, loc='lower left', fontsize=9,
                          framealpha=0.6, facecolor='black', labelcolor='white', edgecolor='grey')
                ax.set_title(" + ".join(selected_elements), fontsize=11, color='white', pad=6)
                plt.tight_layout()
                st.pyplot(fig); plt.close(fig)
                st.caption("Additive colour composite. Each channel normalised to its 99th-percentile within tissue pixels only.")
    elif preproc_view == "Tissue Mask":
        st.markdown("### Tissue Mask Preview")
        show_image(os.path.join(preproc_dir, "mask_preview.png"))
    elif preproc_view == "Mask Overlay":
        st.markdown("### Tissue Mask Overlay")
        st.caption("Tissue pixels shown in colour; excluded background darkened to 10%.")
        show_image(os.path.join(preproc_dir, "mask_overlay.png"))
    elif preproc_view == "Normalisation Comparison":
        st.markdown("### log1p Normalisation Validation")
        st.markdown("#### Violin Plots")
        show_image(os.path.join(preproc_dir, "normalisation_comparison_violins.png"))
        st.markdown("#### Summary Statistics (Median & IQR)")
        show_image(os.path.join(preproc_dir, "normalisation_comparison_tables.png"))

# PCA 
elif stage == "PCA":
    if not os.path.isdir(pca_dir):
        st.info("No PCA outputs found.")
    elif pca_view == "Scree Plot":
        st.markdown("### Scree Plot")
        show_image(os.path.join(pca_dir, "pca_scree.png"))
    elif pca_view == "Cumulative Variance":
        st.markdown("### Cumulative Explained Variance")
        embed_html(os.path.join(pca_dir, "pca_cumulative_variance.html"), height=500)
    elif pca_view == "Correlation Matrix":
        st.markdown("### Pearson Correlation Matrix")
        _img_f_pca = load_img_filtered(preproc_dir)
        _ch_all    = load_channel_names(preproc_dir)
        if _img_f_pca is None or _ch_all is None:
            st.warning("img_filtered.npy not found. Re-run the pipeline.")
        else:
            _tissue_idx_pca, _, _ = load_spatial_data(preproc_dir)
            if _tissue_idx_pca is not None:
                _X_corr = np.log1p(np.stack(
                    [_img_f_pca[i].flatten()[_tissue_idx_pca] for i in range(len(_ch_all))], axis=1
                ).astype(float))
            else:
                _X_corr = np.log1p(np.stack(
                    [_img_f_pca[i].flatten() for i in range(len(_ch_all))], axis=1
                ).astype(float))
            _corr_mat = np.corrcoef(_X_corr.T)

            # Initialise pair state
            if 'corr_qa' not in st.session_state:
                st.session_state['corr_qa'] = _ch_all[0]
            if 'corr_qb' not in st.session_state:
                st.session_state['corr_qb'] = _ch_all[min(1, len(_ch_all)-1)]
            _pair_a = st.session_state['corr_qa'] if st.session_state['corr_qa'] in _ch_all else _ch_all[0]
            _pair_b = st.session_state['corr_qb'] if st.session_state['corr_qb'] in _ch_all else _ch_all[min(1, len(_ch_all)-1)]

            import plotly.graph_objects as _pgo_corr
            from scipy.stats import linregress as _linregress_corr

            _col_heat, _col_sc = st.columns([3, 2])

            with _col_heat:
                _fig_corr = _pgo_corr.Figure(_pgo_corr.Heatmap(
                    z=_corr_mat, x=_ch_all, y=_ch_all,
                    colorscale='RdBu_r', zmin=-1, zmax=1,
                    colorbar=dict(title="r"),
                    hovertemplate="<b>%{x}</b> vs <b>%{y}</b><br>r = %{z:.3f}<extra></extra>",
                ))
                _fig_corr.update_layout(
                    title="Pearson Correlation Matrix — click any cell for scatter →",
                    height=640, plot_bgcolor='white',
                    xaxis=dict(tickangle=-45, tickfont=dict(size=8), autorange=True,
                               range=[-0.5, len(_ch_all)-0.5]),
                    yaxis=dict(range=[len(_ch_all)-0.5, -0.5], tickangle=-45,
                               tickfont=dict(size=8)),
                    hoverlabel=dict(bgcolor='white',
                                   font=dict(size=13, family='Arial', color='black')),
                    margin=dict(l=100, r=10, t=50, b=100),
                    dragmode='select',
                    clickmode='select',
                )
                _corr_ev = st.plotly_chart(
                    _fig_corr, width="stretch",
                    key="corr_hm", on_select="rerun", selection_mode=["points"],
                    config={'modeBarButtons': [['toImage']], 'displaylogo': False,
                            'scrollZoom': False},
                )
                # Process heatmap click → update selected pair
                if (_corr_ev and hasattr(_corr_ev, 'selection')
                        and _corr_ev.selection and _corr_ev.selection.points):
                    _pt  = _corr_ev.selection.points[0]
                    _cx, _cy = str(_pt.get('x', '')), str(_pt.get('y', ''))
                    if (_cx in _ch_all and _cy in _ch_all and _cx != _cy and
                            (st.session_state.get('corr_qa') != _cx or
                             st.session_state.get('corr_qb') != _cy)):
                        st.session_state['corr_qa'] = _cx
                        st.session_state['corr_qb'] = _cy
                        st.rerun()

            with _col_sc:
                if _pair_a != _pair_b:
                    _xi  = _ch_all.index(_pair_a)
                    _yi  = _ch_all.index(_pair_b)
                    _r   = float(_corr_mat[_yi, _xi])
                    st.markdown(f"#### {_pair_a} vs {_pair_b}")
                    _mc1, _mc2 = st.columns(2)
                    _mc1.metric("Pearson r", f"{_r:.3f}")
                    _mc2.metric("R²", f"{(_r**2):.3f}")
                    _MAX_SC = 20_000
                    _n_px   = len(_X_corr)
                    if _n_px > _MAX_SC:
                        _sc_rng = np.random.default_rng(42)
                        _sc_idx = _sc_rng.choice(_n_px, _MAX_SC, replace=False)
                        _xs_sc  = _X_corr[_sc_idx, _xi]
                        _ys_sc  = _X_corr[_sc_idx, _yi]
                    else:
                        _xs_sc = _X_corr[:, _xi]
                        _ys_sc = _X_corr[:, _yi]
                    # Regression line
                    _sl, _ic, _, _, _ = _linregress_corr(_xs_sc, _ys_sc)
                    _xl = [float(_xs_sc.min()), float(_xs_sc.max())]
                    _yl = [_sl * _xl[0] + _ic, _sl * _xl[1] + _ic]
                    _fig_sc = _pgo_corr.Figure()
                    _fig_sc.add_trace(_pgo_corr.Scatter(
                        x=_xs_sc.tolist(), y=_ys_sc.tolist(),
                        mode='markers',
                        marker=dict(size=3, opacity=0.35, color='#3b82f6'),
                        name='Tissue pixels',
                        hovertemplate=f"{_pair_a}: %{{x:.2f}}<br>{_pair_b}: %{{y:.2f}}<extra></extra>",
                    ))
                    _fig_sc.add_trace(_pgo_corr.Scatter(
                        x=_xl, y=_yl, mode='lines',
                        line=dict(color='crimson', width=2),
                        name=f'Fit  y={_sl:.2f}x+{_ic:.2f}',
                    ))
                    _fig_sc.update_layout(
                        title=dict(text="Each dot = one tissue pixel (log1p intensity, up to 20k subsampled)",
                                   font=dict(size=11, color='#444')),
                        xaxis_title=f"{_pair_a}  (log1p)",
                        yaxis_title=f"{_pair_b}  (log1p)",
                        height=520, plot_bgcolor='white',
                        hoverlabel=dict(bgcolor='white',
                                        font=dict(size=13, family='Arial', color='black')),
                        margin=dict(l=60, r=10, t=50, b=60),
                        legend=dict(x=0.01, y=0.99, xanchor='left', yanchor='top',
                                    font=dict(size=10)),
                    )
                    st.plotly_chart(_fig_sc, width="stretch", key="corr_sc",
                                    config={'modeBarButtons': [['toImage']], 'displaylogo': False})
                else:
                    st.info("Click a cell in the heatmap, or select a pair below.")

            # Element pair query 
            st.markdown("---")
            st.caption("Select an element pair to highlight on the heatmap and view the scatter:")
            _qc1, _qc2 = st.columns(2)
            _qc1.selectbox("Element A", _ch_all, key="corr_qa")
            _qc2.selectbox("Element B", _ch_all, key="corr_qb")

    elif pca_view == "Covariance Matrix":
        st.markdown("### Covariance Matrix")
        _img_f_pca = load_img_filtered(preproc_dir)
        _ch_all    = load_channel_names(preproc_dir)
        if _img_f_pca is None or _ch_all is None:
            st.warning("img_filtered.npy not found. Re-run the pipeline.")
        else:
            _tissue_idx_pca, _, _ = load_spatial_data(preproc_dir)
            if _tissue_idx_pca is not None:
                _X_cov = np.log1p(np.stack(
                    [_img_f_pca[i].flatten()[_tissue_idx_pca] for i in range(len(_ch_all))], axis=1
                ).astype(float))
            else:
                _X_cov = np.log1p(np.stack(
                    [_img_f_pca[i].flatten() for i in range(len(_ch_all))], axis=1
                ).astype(float))
            _cov_mat     = np.cov(_X_cov.T)
            _abs_max_cov = float(np.percentile(np.abs(_cov_mat), 99))

            # Initialise pair state
            if 'cov_qa' not in st.session_state:
                st.session_state['cov_qa'] = _ch_all[0]
            if 'cov_qb' not in st.session_state:
                st.session_state['cov_qb'] = _ch_all[min(1, len(_ch_all)-1)]
            _cov_pa = st.session_state['cov_qa'] if st.session_state['cov_qa'] in _ch_all else _ch_all[0]
            _cov_pb = st.session_state['cov_qb'] if st.session_state['cov_qb'] in _ch_all else _ch_all[min(1, len(_ch_all)-1)]

            import plotly.graph_objects as _pgo_cov

            _col_cov_heat, _col_cov_val = st.columns([3, 2])

            with _col_cov_heat:
                _fig_cov = _pgo_cov.Figure(_pgo_cov.Heatmap(
                    z=_cov_mat, x=_ch_all, y=_ch_all,
                    colorscale='RdBu_r', zmin=-_abs_max_cov, zmax=_abs_max_cov,
                    colorbar=dict(title="Cov"),
                    hovertemplate="<b>%{x}</b> vs <b>%{y}</b><br>Cov = %{z:.4f}<extra></extra>",
                ))
                _fig_cov.update_layout(
                    title="Covariance Matrix — click any cell to view the value →",
                    height=640, plot_bgcolor='white',
                    xaxis=dict(tickangle=-45, tickfont=dict(size=8),
                               range=[-0.5, len(_ch_all)-0.5]),
                    yaxis=dict(range=[len(_ch_all)-0.5, -0.5], tickangle=-45,
                               tickfont=dict(size=8)),
                    hoverlabel=dict(bgcolor='white',
                                   font=dict(size=13, family='Arial', color='black')),
                    margin=dict(l=100, r=10, t=50, b=100),
                    dragmode='select',
                    clickmode='select',
                )
                _cov_ev = st.plotly_chart(
                    _fig_cov, width="stretch",
                    key="cov_hm", on_select="rerun", selection_mode=["points"],
                    config={'modeBarButtons': [['toImage']], 'displaylogo': False,
                            'scrollZoom': False},
                )
                # Process heatmap click → update selected pair
                if (_cov_ev and hasattr(_cov_ev, 'selection')
                        and _cov_ev.selection and _cov_ev.selection.points):
                    _pt  = _cov_ev.selection.points[0]
                    _cx, _cy = str(_pt.get('x', '')), str(_pt.get('y', ''))
                    if (_cx in _ch_all and _cy in _ch_all and _cx != _cy and
                            (st.session_state.get('cov_qa') != _cx or
                             st.session_state.get('cov_qb') != _cy)):
                        st.session_state['cov_qa'] = _cx
                        st.session_state['cov_qb'] = _cy
                        st.rerun()

            with _col_cov_val:
                if _cov_pa != _cov_pb:
                    _cxi = _ch_all.index(_cov_pa)
                    _cyi = _ch_all.index(_cov_pb)
                    _cov_val  = float(_cov_mat[_cyi, _cxi])
                    _corr_val = float(np.corrcoef(_X_cov.T)[_cyi, _cxi])
                    st.markdown(f"#### {_cov_pa} vs {_cov_pb}")
                    st.metric("Covariance", f"{_cov_val:.4f}")
                    st.metric("Pearson r (for reference)", f"{_corr_val:.3f}")
                    st.markdown(
                        "Covariance is the unnormalised version of the correlation. "
                        "A large positive value means the two elements tend to be "
                        "high together across tissue pixels; negative = inverse relationship."
                    )
                else:
                    st.info("Click a cell in the heatmap, or select a pair below.")

            # Element pair query 
            st.markdown("---")
            st.caption("Select an element pair to highlight on the heatmap:")
            _cqc1, _cqc2 = st.columns(2)
            _cqc1.selectbox("Element A", _ch_all, key="cov_qa")
            _cqc2.selectbox("Element B", _ch_all, key="cov_qb")
    elif pca_view == "Loading Plot":
        st.markdown("### 2D Loading Plot — Interactive")
        _ld_path  = os.path.join(pca_dir, "pca_loadings.npy")
        _ev_path  = os.path.join(pca_dir, "pca_explained_variance.npy")
        _cn_path  = os.path.join(pca_dir, "pca_channel_names.json")
        if not all(os.path.exists(p) for p in [_ld_path, _ev_path, _cn_path]):
            st.info("Interactive loading data not found — re-run the pipeline to generate it.")
            show_image(os.path.join(pca_dir, "pca_loading_PC1_vs_PC2.png"))
        else:
            _loadings = np.load(_ld_path)          # (n_components, n_channels)
            _ev       = np.load(_ev_path)
            with open(_cn_path) as _f:
                _cnames = json.load(_f)
            _n_pcs = _loadings.shape[0]
            _pc_opts = [f"PC{i+1} ({_ev[i]*100:.1f}%)" for i in range(min(_n_pcs, 20))]
            _c1, _c2, _c3 = st.columns([1, 1, 2])
            _pcx_sel = _c1.selectbox("X axis", _pc_opts, index=0, key="lp_pcx")
            _pcy_sel = _c2.selectbox("Y axis", _pc_opts, index=1, key="lp_pcy")
            _pcx_i   = int(_pcx_sel.split("PC")[1].split(" ")[0]) - 1
            _pcy_i   = int(_pcy_sel.split("PC")[1].split(" ")[0]) - 1
            _ch_sel  = _c3.multiselect("Channels to show", _cnames, default=_cnames, key="lp_ch")
            if not _ch_sel:
                st.warning("Select at least one channel.")
            else:
                _idx_sel = [_cnames.index(c) for c in _ch_sel]
                _lx = _loadings[_pcx_i, _idx_sel]
                _ly = _loadings[_pcy_i, _idx_sel]
                import plotly.graph_objects as _pgo
                _fig_lp = _pgo.Figure()
                for _ci, (_cx, _cy, _cn) in enumerate(zip(_lx, _ly, _ch_sel)):
                    _fig_lp.add_trace(_pgo.Scatter(
                        x=[0, _cx], y=[0, _cy], mode='lines+markers+text',
                        line=dict(width=2), marker=dict(size=[4, 10]),
                        text=["", _cn], textposition="top center",
                        textfont=dict(size=10, color='#111111'),
                        name=_cn, showlegend=False,
                        hovertemplate=f"<b>{_cn}</b><br>{_pcx_sel}: %{{x:.3f}}<br>{_pcy_sel}: %{{y:.3f}}<extra></extra>"
                    ))
                _fig_lp.add_hline(y=0, line_dash="dash", line_color="grey", line_width=0.8)
                _fig_lp.add_vline(x=0, line_dash="dash", line_color="grey", line_width=0.8)
                _fig_lp.update_layout(
                    xaxis_title=_pcx_sel, yaxis_title=_pcy_sel,
                    title="PCA Loading Plot", plot_bgcolor="white",
                    paper_bgcolor="white",
                    font=dict(color='#111111'),
                    height=650, margin=dict(l=60, r=40, t=50, b=60),
                    xaxis=dict(zeroline=False, showgrid=True, gridcolor="#eee",
                               tickfont=dict(color='#111111'), title_font=dict(color='#111111')),
                    yaxis=dict(zeroline=False, showgrid=True, gridcolor="#eee",
                               scaleanchor="x", scaleratio=1,
                               tickfont=dict(color='#111111'), title_font=dict(color='#111111')),
                    hoverlabel=dict(bgcolor='white',
                                    font=dict(size=13, family='Arial', color='black')),
                )
                st.plotly_chart(_fig_lp, width="stretch",
                                config={'displayModeBar': False})
    elif pca_view == "Variance per Element":
        st.markdown("### Element Variance Captured by PCA")
        st.caption("How much of each element's spatial variance is retained by the first few PCs. "
                   "Green ≥ 80%: well-represented. Red < 50%: poorly captured.")
        show_image(os.path.join(pca_dir, "pca_variance_per_element.png"))
# ── UMAP ──────────────────────────────────────────────────────────────────────
elif stage == "UMAP":
    if not os.path.isdir(umap_dir):
        st.info("No UMAP outputs found.")
    elif umap_view == "RGB Spatial Map":
        st.markdown("### UMAP RGB Spatial Map")
        show_image(os.path.join(umap_dir, "umap_rgb_map.png"))
    elif umap_view == "2D Scatter (PNG)":
        st.markdown("### UMAP 2D Scatter — coloured by RGB")
        st.caption("Static PNG. Colour encodes position in UMAP space (same as the RGB spatial map).")
        show_image(os.path.join(umap_dir, "umap_2d_scatter.png"))
    elif umap_view == "3D Scatter (PNG)":
        st.markdown("### UMAP 3D Scatter — coloured by RGB")
        st.caption("Static PNG version for export and thesis figures.")
        show_image(os.path.join(umap_dir, "umap_3d_scatter.png"))
    elif umap_view == "3D Scatter":
        st.markdown("### UMAP 3D Scatter — coloured by RGB")
        st.caption("Rotate to explore the 3D embedding. Colour = position in UMAP space (same as the RGB spatial map).")
        embed_html(os.path.join(umap_dir, "umap_3d_scatter.html"), height=800)
    elif umap_view == "Density / Hexbin":
        st.markdown("### UMAP Embedding Density")
        st.caption("Left: hexbin pixel count. Right: Gaussian KDE contours. Reveals dense cluster cores and sparse bridges between clusters.")
        show_image(os.path.join(umap_dir, "umap_density.png"))
    elif umap_view == "KDE Cluster Contour (K-Means)":
        st.markdown("### UMAP — KDE Cluster Contour (K-Means)")
        st.caption(
            "K-Means cluster boundaries estimated by Gaussian KDE. "
            "Contour lines = 20/50/80 % density levels. "
            "Enriched channels (log₂FC > 0) annotated beside each cluster centroid."
        )
        show_image(os.path.join(umap_dir, "umap_kde_cluster_kmeans.png"))
    elif umap_view == "KDE Cluster Contour (HDBSCAN)":
        st.markdown("### UMAP — KDE Cluster Contour (HDBSCAN)")
        st.caption(
            "HDBSCAN cluster boundaries estimated by Gaussian KDE. "
            "Contour lines = 20/50/80 % density levels. "
            "Enriched channels (log₂FC > 0) annotated beside each cluster centroid."
        )
        show_image(os.path.join(umap_dir, "umap_kde_cluster_hdbscan.png"))
    elif umap_view == "Element Cluster Assignment":
        st.markdown("### UMAP — Element Cluster Assignment")
        st.caption(
            "Pick an element from the dropdown to colour UMAP pixels by its log1p intensity. "
            "Each element is assigned to its dominant cluster (argmax log₂FC). "
            "Use the dropdown in the chart to switch elements."
        )
        embed_html(os.path.join(umap_dir, "umap_element_cluster_assignment.html"), height=700)
    elif umap_view == "Element Intensity Explorer":
        st.markdown("### UMAP — Element Intensity Explorer")
        st.caption(
            "UMAP 2D scatter coloured by the log₁p intensity of the selected element. "
            "Each dot = one tissue pixel (up to 50 k subsampled, fixed seed). "
            "Select element in the sidebar. Scroll to zoom, drag to pan."
        )
        _xu_path  = os.path.join(umap_dir, "X_umap.npy")
        _dn_path  = os.path.join(cluster_dir, "df_normalised.npy")
        _ch_names = load_channel_names(preproc_dir)
        if not os.path.exists(_xu_path):
            st.info("X_umap.npy not found — run the UMAP stage first.")
        elif not os.path.exists(_dn_path):
            st.info("df_normalised.npy not found — run the clustering stage first.")
        elif not _ch_names:
            st.info("channel_names.json not found — run preprocessing first.")
        else:
            @st.cache_data
            def _load_umap_elem_data(xu_path, dn_path):
                return np.load(xu_path), np.load(dn_path)
            _X_umap, _df_flat = _load_umap_elem_data(_xu_path, _dn_path)

            # Fixed subsample — recomputed once and cached implicitly via seed
            _N, _MAX = len(_X_umap), 50_000
            _rng = np.random.default_rng(42)
            _sub = _rng.choice(_N, min(_N, _MAX), replace=False)

            _elem = st.session_state.get("umap_elem_sel", _ch_names[0])
            if _elem not in _ch_names:
                _elem = _ch_names[0]
            _ch_idx = _ch_names.index(_elem)

            _x = _X_umap[_sub, 0]
            _y = _X_umap[_sub, 1]
            _c = _df_flat[_sub, _ch_idx]

            _elem_fig = go.Figure(go.Scattergl(
                x=_x, y=_y,
                mode='markers',
                marker=dict(
                    size=2,
                    color=_c,
                    colorscale='Hot',
                    colorbar=dict(
                        title=dict(text=f'{_elem}<br>(log₁p)', side='right'),
                        thickness=15, len=0.8,
                    ),
                    opacity=0.75,
                ),
                hovertemplate=(
                    f'<b>{_elem}</b> (log₁p): %{{marker.color:.3f}}<br>'
                    'UMAP 1: %{x:.2f}<br>UMAP 2: %{y:.2f}<extra></extra>'
                ),
                hoverlabel=dict(bgcolor='white',
                                font=dict(size=12, color='black', family='Arial')),
            ))
            _elem_fig.update_layout(
                title=dict(
                    text=(f'UMAP 2D  ·  {_elem}  (log₁p intensity)  '
                          f'·  {len(_sub):,} pixels'),
                    font=dict(size=13, color='white'),
                ),
                xaxis=dict(title='UMAP 1', gridcolor='#333',
                           zerolinecolor='#444', tickfont=dict(color='white')),
                yaxis=dict(title='UMAP 2', gridcolor='#333',
                           zerolinecolor='#444', tickfont=dict(color='white')),
                height=660,
                paper_bgcolor='#0f0f0f',
                plot_bgcolor='#0f0f0f',
                font=dict(color='white'),
                margin=dict(l=60, r=60, t=60, b=50),
            )
            st.plotly_chart(
                _elem_fig,
                width='stretch',
                config={
                    'modeBarButtons': [['toImage', 'zoom2d', 'pan2d', 'resetScale2d']],
                    'displaylogo': False,
                    'scrollZoom': True,
                },
            )
    elif umap_view == "Moran's I":
        st.markdown("### Spatial Autocorrelation: Moran's I")
        st.caption("Moran's I for each UMAP dimension mapped back to tissue space (queen contiguity). I → 1 confirms the embedding captures coherent tissue structure.")
        show_image(os.path.join(umap_dir, "umap_morans_i.png"))

# CLUSTERING 
elif stage == "Clustering":
    if not os.path.isdir(cluster_dir):
        st.info("No clustering outputs found.")
    elif km_view == "Cluster Explorer":
        st.markdown("### Cluster Explorer — Element Intensity Profiles")

        @st.cache_data
        def load_flat_log_for_lasso(cluster_dir, preproc_dir):
            """Only loaded when lasso is active — heavy, cached after first load."""
            km_path  = os.path.join(cluster_dir, "km_labels.npy")
            npy_path = os.path.join(cluster_dir, "df_normalised.npy")
            if not os.path.exists(km_path) or not os.path.exists(npy_path):
                return None, None
            km_labels = np.load(km_path)
            flat_log  = np.load(npy_path).astype(np.float64)
            n = min(len(km_labels), len(flat_log))
            return km_labels[:n], flat_log[:n]

        summary = load_cluster_summary(cluster_dir, 'kmeans')
        km_labels_path = os.path.join(cluster_dir, "km_labels.npy")
        ch_names_cl    = load_channel_names(preproc_dir)

        if summary is None:
            st.warning("Pre-computed cluster summary not found — re-run the pipeline.")
        elif ch_names_cl is None:
            st.warning("channel_names.json not found — re-run preprocessing.")
        elif not os.path.exists(km_labels_path):
            st.warning("km_labels.npy not found — re-run the pipeline.")
        else:
            mean_profiles_arr, tissue_mean, log_fc_arr, counts, dominant, cluster_ids_list = summary
            km_labels_data = np.load(km_labels_path)
            ch_names_data  = ch_names_cl
            n_clusters     = len(cluster_ids_list)
            flat_log_data  = None   # loaded on-demand only if lasso is used

            if km_labels_data is not None:
                selected_cluster = int(st.session_state.get("km_cluster_sel", 0))
                # Map cluster_id → row index in pre-computed arrays
                cid_to_idx = {cid: i for i, cid in enumerate(cluster_ids_list)}

                import sys as _sys, importlib as _il
                _sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
                _cfg = _il.import_module("config")
                _pal = _cfg.get_cluster_colours(n_clusters)

                # Show the config settings that were used for this run
                _fk  = _cfg.KMEANS_FORCED_K
                _fk_str = f"forced = {_fk}" if _fk is not None else f"auto elbow (max tested k = {_cfg.KMEANS_MAX_K})"
                st.caption(
                    f"K-Means config — **k = {n_clusters}**  ({_fk_str})  ·  "
                    f"random_state = {_cfg.KMEANS_RANDOM_STATE}"
                )

                col_map, col_profile = st.columns([1, 1])

                with col_map:
                    st.markdown("**K-Means Spatial Map** — lasso for ad-hoc region; sidebar to select cluster")
                    tissue_idx, _H, _W = load_spatial_data(preproc_dir)
                    lasso_positions = None
                    if tissue_idx is not None:
                        _na_bg = None
                        _img_f = load_img_filtered(preproc_dir)
                        if _img_f is not None:
                            _na_cands = [i for i, n in enumerate(ch_names_data)
                                         if "23Na" in n or n == "Na" or "23na" in n.lower()]
                            if _na_cands:
                                _na_bg = np.zeros((_H, _W), dtype=float)
                                _na_bg.flat[tissue_idx] = _img_f[_na_cands[0]].flatten()[tissue_idx]
                        fig_map = build_cluster_plotly(
                            km_labels_data, tissue_idx, _H, _W, n_clusters,
                            selected_cluster=selected_cluster, na_img=_na_bg
                        )
                        map_event = st.plotly_chart(
                            fig_map, width="stretch",
                            key="km_interactive_map",
                            on_select="rerun",
                            selection_mode=["lasso", "box"],
                        )
                        if (map_event and map_event.selection
                                and map_event.selection.points):
                            _cd = [p.get("customdata") for p in map_event.selection.points
                                   if p.get("customdata") is not None]
                            if _cd:
                                lasso_positions = np.array(_cd, dtype=int)
                    else:
                        show_image(os.path.join(cluster_dir, "kmeans_spatial.png"))

                    if lasso_positions is not None:
                        st.caption(f"🔲 Lasso: **{len(lasso_positions):,} pixels** — profiles computed on-the-fly. Clear to reset.")
                    else:
                        _cnt = counts.get(selected_cluster, '?')
                        _dom = dominant.get(selected_cluster, '?')
                        st.caption(f"Cluster {selected_cluster} — {_cnt:,} pixels — dominant: **{_dom}**")

                with col_profile:
                    cl_hex = _pal[cid_to_idx.get(selected_cluster, 0) % len(_pal)]

                    if lasso_positions is not None and len(lasso_positions) > 0:
                        # Lasso: load flat_log on-demand and compute for the selection only
                        _km_raw, _flat_raw = load_flat_log_for_lasso(cluster_dir, preproc_dir)
                        if _flat_raw is not None:
                            lasso_profile = _flat_raw[lasso_positions].mean(axis=0)
                            lasso_fc      = (lasso_profile - tissue_mean) / np.log(2)
                            lasso_raw     = np.expm1(lasso_profile)
                            src_lbl = f"Lasso ({len(lasso_positions):,} pixels)"
                            tab_int, tab_fc, tab_ic = st.tabs(["Mean Intensity", "Enrichment vs Tissue", "Ion Count"])
                            with tab_int:
                                _ord = np.argsort(lasso_profile)[::-1][:top_n]
                                fig_int = go.Figure(go.Bar(
                                    x=[float(lasso_profile[j]) for j in _ord],
                                    y=[ch_names_data[j] for j in _ord],
                                    orientation='h', marker_color=cl_hex,
                                    hovertemplate="<b>%{y}</b><br>Mean log1p: %{x:.4f}<extra></extra>",
                                ))
                                fig_int.update_layout(title=f"Top {top_n} — Mean Intensity  ({src_lbl})",
                                    xaxis_title="Mean log1p intensity",
                                    yaxis=dict(autorange='reversed', tickfont_size=16),
                                    height=max(400, top_n * 22 + 80), plot_bgcolor='white',
                                    margin=dict(l=10, r=10, t=60, b=50),
                                    title_font_size=18, font=dict(size=15),
                                    hoverlabel=dict(bgcolor='white', font_size=13, font_family='Arial'))
                                st.plotly_chart(fig_int, width="stretch", key="lasso_int")
                            with tab_fc:
                                _fo = np.argsort(lasso_fc)[::-1][:top_n]
                                _fv = [float(lasso_fc[j]) for j in _fo]
                                fig_fc = go.Figure(go.Bar(
                                    x=_fv, y=[ch_names_data[j] for j in _fo], orientation='h',
                                    marker_color=['#27ae60' if v >= 0 else '#e74c3c' for v in _fv],
                                    hovertemplate="<b>%{y}</b><br>Log₂FC: %{x:+.4f}<extra></extra>",
                                ))
                                fig_fc.add_vline(x=0, line_width=1, line_color="grey")
                                fig_fc.update_layout(title=f"Top {top_n} — Enrichment  ({src_lbl})",
                                    xaxis_title="Log₂ fold-change vs tissue",
                                    yaxis=dict(autorange='reversed', tickfont_size=16),
                                    height=max(400, top_n * 22 + 80), plot_bgcolor='white',
                                    margin=dict(l=10, r=10, t=60, b=50),
                                    title_font_size=18, font=dict(size=15),
                                    hoverlabel=dict(bgcolor='white', font_size=13, font_family='Arial'))
                                st.plotly_chart(fig_fc, width="stretch", key="lasso_fc")
                            with tab_ic:
                                _io = np.argsort(lasso_raw)[::-1][:top_n]
                                _iv = [float(lasso_raw[j]) for j in _io]
                                fig_ic = go.Figure(go.Bar(
                                    x=_iv, y=[ch_names_data[j] for j in _io], orientation='h',
                                    marker_color=['#27ae60' if lasso_fc[j] >= 0 else '#e74c3c' for j in _io],
                                    hovertemplate="<b>%{y}</b><br>Ion count: %{x:,.1f}<extra></extra>",
                                ))
                                fig_ic.update_layout(title=f"Top {top_n} — Ion Count  ({src_lbl})",
                                    xaxis=dict(title="Mean ion count (ions/pixel)", type="log", title_font_size=15, tickfont_size=14),
                                    yaxis=dict(autorange='reversed', tickfont_size=16),
                                    height=max(400, top_n * 22 + 80), plot_bgcolor='white',
                                    margin=dict(l=10, r=10, t=60, b=50),
                                    title_font_size=18, font=dict(size=15),
                                    hoverlabel=dict(bgcolor='white', font_size=13, font_family='Arial'))
                                st.plotly_chart(fig_ic, width="stretch", key="lasso_ic")
                    else:
                        # No lasso — build interactive charts from pre-computed summary
                        _cidx    = cid_to_idx.get(selected_cluster, 0)
                        _profile = mean_profiles_arr[_cidx]
                        _fc      = log_fc_arr[_cidx]
                        _raw     = np.expm1(_profile)
                        _n_ch    = len(ch_names_data)
                        src_lbl  = f"Cluster {selected_cluster} ({counts.get(selected_cluster, '?'):,} px)"
                        tab_int, tab_fc, tab_ic = st.tabs(["Mean Intensity", "Enrichment vs Tissue", "Ion Count"])
                        with tab_int:
                            _ord = np.argsort(_profile)[::-1][:top_n]
                            _fig_ci = go.Figure(go.Bar(
                                x=[float(_profile[j]) for j in _ord],
                                y=[ch_names_data[j] for j in _ord],
                                orientation='h', marker_color=cl_hex,
                                hovertemplate="<b>%{y}</b><br>Mean log1p: %{x:.4f}<extra></extra>",
                            ))
                            _fig_ci.update_layout(
                                title=f"Top {top_n} — Mean Intensity  ({src_lbl})",
                                xaxis_title="Mean log1p intensity",
                                yaxis=dict(autorange='reversed', tickfont_size=16),
                                height=max(400, top_n * 22 + 80), plot_bgcolor='white',
                                margin=dict(l=10, r=10, t=60, b=50),
                                title_font_size=18, font=dict(size=15),
                                hoverlabel=dict(bgcolor='white', font_size=13, font_family='Arial'))
                            st.plotly_chart(_fig_ci, width="stretch", key="cl_int")
                        with tab_fc:
                            _fo = np.argsort(_fc)[::-1][:top_n]
                            _fv = [float(_fc[j]) for j in _fo]
                            _fig_cf = go.Figure(go.Bar(
                                x=_fv, y=[ch_names_data[j] for j in _fo], orientation='h',
                                marker_color=['#27ae60' if v >= 0 else '#e74c3c' for v in _fv],
                                hovertemplate="<b>%{y}</b><br>Log₂FC: %{x:+.4f}<extra></extra>",
                            ))
                            _fig_cf.add_vline(x=0, line_width=1, line_color="grey")
                            _fig_cf.update_layout(
                                title=f"Top {top_n} — Enrichment  ({src_lbl})",
                                xaxis_title="Log₂ fold-change vs tissue",
                                yaxis=dict(autorange='reversed', tickfont_size=16),
                                height=max(400, top_n * 22 + 80), plot_bgcolor='white',
                                margin=dict(l=10, r=10, t=60, b=50),
                                title_font_size=18, font=dict(size=15),
                                hoverlabel=dict(bgcolor='white', font_size=13, font_family='Arial'))
                            st.plotly_chart(_fig_cf, width="stretch", key="cl_fc")
                        with tab_ic:
                            _io = np.argsort(_raw)[::-1][:top_n]
                            _iv = [float(_raw[j]) for j in _io]
                            _fig_cion = go.Figure(go.Bar(
                                x=_iv, y=[ch_names_data[j] for j in _io], orientation='h',
                                marker_color=['#27ae60' if _fc[j] >= 0 else '#e74c3c' for j in _io],
                                hovertemplate="<b>%{y}</b><br>Ion count: %{x:,.1f}<extra></extra>",
                            ))
                            _fig_cion.update_layout(
                                title=f"Top {top_n} — Ion Count  ({src_lbl})",
                                xaxis=dict(title="Mean ion count (ions/pixel)", type="log", title_font_size=15, tickfont_size=14),
                                yaxis=dict(autorange='reversed', tickfont_size=16),
                                height=max(400, top_n * 22 + 80), plot_bgcolor='white',
                                margin=dict(l=10, r=10, t=60, b=50),
                                title_font_size=18, font=dict(size=15),
                                hoverlabel=dict(bgcolor='white', font_size=13, font_family='Arial'))
                            st.plotly_chart(_fig_cion, width="stretch", key="cl_ic")

    elif km_view == "Elbow + Silhouette":
        st.markdown("### K-Means Elbow + Silhouette Score")
        import sys as _sys2, importlib as _il2
        _sys2.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        _cfg2 = _il2.import_module("config")
        _fk2  = _cfg2.KMEANS_FORCED_K
        st.caption(
            f"Tested k = 1 to {_cfg2.KMEANS_MAX_K}  ·  "
            f"Forced K: {'auto (elbow)' if _fk2 is None else _fk2}  ·  "
            f"random_state = {_cfg2.KMEANS_RANDOM_STATE}"
        )
        show_image(os.path.join(cluster_dir, "kmeans_elbow.png"))
    elif km_view == "Silhouette Plot":
        st.markdown("### Full Silhouette Plot — K-Means")
        st.caption("Each bar = one pixel's silhouette coefficient, grouped and sorted by cluster. "
                   "Wide positive bars = tight, well-separated clusters. Bars crossing zero = potential misassignments.")
        show_image(os.path.join(cluster_dir, "kmeans_silhouette_full.png"))
    elif km_view == "Spatial Map":
        st.markdown("### K-Means Spatial Map")
        import sys as _sys3, importlib as _il3
        _sys3.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        _cfg3 = _il3.import_module("config")
        _km_path3 = os.path.join(cluster_dir, "km_labels.npy")
        _k3 = len(np.unique(np.load(_km_path3))) if os.path.exists(_km_path3) else "?"
        st.caption(
            f"k = {_k3}  ·  "
            f"random_state = {_cfg3.KMEANS_RANDOM_STATE}  ·  "
            f"metric = UMAP coordinates"
        )
        show_image(os.path.join(cluster_dir, "kmeans_spatial.png"))
    elif km_view == "Cluster Profiles":
        st.markdown("### K-Means Cluster Elemental Profiles")
        show_image(os.path.join(cluster_dir, "kmeans_cluster_profiles.png"))
    elif km_view == "Statistical Tests":
        st.markdown("### Statistical Tests — Mann-Whitney U per Cluster")
        st.caption("Each element tested for significant enrichment/depletion vs rest of tissue. "
                   "Benjamini-Hochberg FDR correction applied per cluster (q < 0.05).")
        csv_path = os.path.join(cluster_dir, "stats_results.csv")
        if os.path.exists(csv_path):
            import pandas as _pd
            _df_stats = _pd.read_csv(csv_path)
            st.dataframe(_df_stats, width="stretch")
        else:
            st.info("stats_results.csv not found — re-run the pipeline.")
    elif km_view == "HDBSCAN Cluster Profiles":
        st.markdown("### HDBSCAN Cluster Elemental Profiles")
        show_image(os.path.join(cluster_dir, "hdbscan_cluster_profiles.png"))
    elif km_view == "HDBSCAN Membership Probability":
        st.markdown("### HDBSCAN Membership Confidence")
        st.caption("Left: spatial probability map (green = high confidence core, red = boundary/noise). "
                   "Right: violin plots per cluster showing probability distribution.")
        show_image(os.path.join(cluster_dir, "hdbscan_membership_prob.png"))
    elif km_view == "HDBSCAN Condensed Tree":
        st.markdown("### HDBSCAN Condensed Tree — Cluster Stability")
        _tree_path = os.path.join(cluster_dir, "hdbscan_condensed_tree.png")
        if os.path.exists(_tree_path):
            st.caption(
                "Each coloured branch is a selected cluster. Width ∝ number of pixels in the cluster. "
                "Height (λ = 1/distance) shows how long the cluster persisted as density increased — "
                "taller = more stable. Grey branches were pruned as noise or sub-threshold. "
                "On GPU runs, a bar chart of per-cluster persistence scores is shown instead."
            )
            show_image(_tree_path)
        else:
            st.info("hdbscan_condensed_tree.png not found — re-run the clustering stage.")
    elif km_view == "Cluster Proportions":
        st.markdown("### Cluster Tissue Area Proportions — K-Means vs HDBSCAN")
        st.caption("% of total tissue pixels assigned to each cluster. "
                   "HDBSCAN noise bar shows unclassified pixels.")
        show_image(os.path.join(cluster_dir, "cluster_proportions.png"))
    elif km_view == "HDBSCAN Spatial Map":
        st.markdown("### HDBSCAN Spatial Map")
        hdb_path = os.path.join(cluster_dir, "hdbscan_spatial.png")
        if os.path.exists(hdb_path):
            show_image(hdb_path)
            hdb_labels_path = os.path.join(cluster_dir, "hdbscan_labels.npy")
            if os.path.exists(hdb_labels_path):
                hdb_labels_loaded = np.load(hdb_labels_path)
                n_clusters_hdb = len(set(hdb_labels_loaded)) - (1 if -1 in hdb_labels_loaded else 0)
                n_noise   = int((hdb_labels_loaded == -1).sum())
                noise_pct = n_noise / len(hdb_labels_loaded) * 100
                col1, col2, col3 = st.columns(3)
                col1.metric("Clusters found", n_clusters_hdb)
                col2.metric("Noise pixels", f"{n_noise:,}")
                col3.metric("Noise %", f"{noise_pct:.1f}%")
        else:
            st.info("HDBSCAN has not been run yet — re-run the pipeline.")
    elif km_view == "HDBSCAN Cluster Explorer":
        st.markdown("### HDBSCAN Cluster Explorer — Element Intensity Profiles")

        hdb_summary = load_cluster_summary(cluster_dir, 'hdbscan')
        hdb_labels_path = os.path.join(cluster_dir, "hdbscan_labels.npy")

        if hdb_summary is None:
            st.warning("Pre-computed HDBSCAN summary not found — re-run the pipeline.")
        elif not os.path.exists(hdb_labels_path):
            st.warning("hdbscan_labels.npy not found — re-run the pipeline.")
        else:
            (hdb_mp_arr, hdb_tissue_mean, hdb_lfc_arr,
             hdb_counts, hdb_dominant, hdb_cluster_ids_list) = hdb_summary
            hdb_labels_data = np.load(hdb_labels_path)
            n_hdb           = len(hdb_cluster_ids_list)
            hdb_selected    = int(st.session_state.get("hdb_cluster_sel",
                                                        hdb_cluster_ids_list[0]))
            hdb_cid_to_idx  = {cid: i for i, cid in enumerate(hdb_cluster_ids_list)}

            n_noise   = int((hdb_labels_data == -1).sum())
            noise_pct = n_noise / len(hdb_labels_data) * 100

            import sys as _sys2, importlib as _il2
            _sys2.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            _cfg2  = _il2.import_module("config")
            _pal2  = _cfg2.get_cluster_colours(n_hdb)

            col_map_h, col_prof_h = st.columns([1, 1])

            with col_map_h:
                st.markdown("**HDBSCAN Spatial Map** — lasso for ad-hoc region; sidebar to select cluster")
                tissue_idx_h, _Hh, _Wh = load_spatial_data(preproc_dir)
                hdb_lasso_positions = None
                if tissue_idx_h is not None:
                    _na_bg_h = None
                    _img_fh  = load_img_filtered(preproc_dir)
                    _ch_hdb  = load_channel_names(preproc_dir)
                    if _img_fh is not None and _ch_hdb is not None:
                        _na_cands = [i for i, n in enumerate(_ch_hdb)
                                     if "23Na" in n or n == "Na" or "23na" in n.lower()]
                        if _na_cands:
                            _na_bg_h = np.zeros((_Hh, _Wh), dtype=float)
                            _na_bg_h.flat[tissue_idx_h] = _img_fh[_na_cands[0]].flatten()[tissue_idx_h]

                    hdb_remap = np.full(len(hdb_labels_data), -1, dtype=int)
                    for new_id, cid in enumerate(hdb_cluster_ids_list):
                        hdb_remap[hdb_labels_data == cid] = new_id

                    fig_hdb = build_cluster_plotly(
                        hdb_remap, tissue_idx_h, _Hh, _Wh, n_hdb,
                        selected_cluster=hdb_cid_to_idx.get(hdb_selected, 0),
                        na_img=_na_bg_h,
                    )
                    hdb_map_event = st.plotly_chart(
                        fig_hdb, width="stretch",
                        key="hdb_interactive_map",
                        on_select="rerun",
                        selection_mode=["lasso", "box"],
                    )
                    if (hdb_map_event and hdb_map_event.selection
                            and hdb_map_event.selection.points):
                        _cd = [p.get("customdata") for p in hdb_map_event.selection.points
                               if p.get("customdata") is not None]
                        if _cd:
                            hdb_lasso_positions = np.array(_cd, dtype=int)
                else:
                    show_image(os.path.join(cluster_dir, "hdbscan_spatial.png"))

                if hdb_lasso_positions is not None:
                    st.caption(f"🔲 Lasso: **{len(hdb_lasso_positions):,} pixels** — profiles computed on-the-fly. Clear to reset.")
                else:
                    _hcnt = hdb_counts.get(hdb_selected, '?')
                    _hdom = hdb_dominant.get(hdb_selected, '?')
                    st.caption(f"Cluster {hdb_selected} — {_hcnt:,} pixels — dominant: **{_hdom}**  |  "
                               f"Noise: {n_noise:,} px ({noise_pct:.1f}%)")

            with col_prof_h:
                cl_hex_h = _pal2[hdb_cid_to_idx.get(hdb_selected, 0) % len(_pal2)]

                if hdb_lasso_positions is not None and len(hdb_lasso_positions) > 0:
                    _hkm, _hfl = load_flat_log_for_lasso(cluster_dir, preproc_dir)
                    _hch = load_channel_names(preproc_dir) or []
                    if _hfl is not None:
                        hdb_lp   = _hfl[hdb_lasso_positions].mean(axis=0)
                        hdb_lfc  = (hdb_lp - hdb_tissue_mean) / np.log(2)
                        hdb_lraw = np.expm1(hdb_lp)
                        hdb_src  = f"Lasso ({len(hdb_lasso_positions):,} pixels)"
                        tab_int_h, tab_fc_h, tab_ic_h = st.tabs(
                            ["Mean Intensity", "Enrichment vs Tissue", "Ion Count"])
                        with tab_int_h:
                            _o = np.argsort(hdb_lp)[::-1][:top_n_hdb]
                            fig_h_int = go.Figure(go.Bar(
                                x=[float(hdb_lp[j]) for j in _o],
                                y=[_hch[j] for j in _o],
                                orientation='h', marker_color=cl_hex_h,
                                hovertemplate="<b>%{y}</b><br>Mean log1p: %{x:.4f}<extra></extra>",
                            ))
                            fig_h_int.update_layout(
                                title=f"Top {top_n_hdb} — Mean Intensity  ({hdb_src})",
                                xaxis_title="Mean log1p intensity",
                                yaxis=dict(autorange='reversed', tickfont_size=16),
                                height=max(400, top_n_hdb * 22 + 80),
                                plot_bgcolor='white', margin=dict(l=10, r=10, t=60, b=50),
                                title_font_size=18, font=dict(size=15),
                                hoverlabel=dict(bgcolor='white', font_size=13, font_family='Arial'))
                            st.plotly_chart(fig_h_int, width="stretch", key="hdb_lasso_int")
                        with tab_fc_h:
                            _fo = np.argsort(hdb_lfc)[::-1][:top_n_hdb]
                            _fv = [float(hdb_lfc[j]) for j in _fo]
                            fig_h_fc = go.Figure(go.Bar(
                                x=_fv, y=[_hch[j] for j in _fo], orientation='h',
                                marker_color=['#27ae60' if v >= 0 else '#e74c3c' for v in _fv],
                                hovertemplate="<b>%{y}</b><br>Log₂FC: %{x:+.4f}<extra></extra>",
                            ))
                            fig_h_fc.add_vline(x=0, line_width=1, line_color="grey")
                            fig_h_fc.update_layout(
                                title=f"Top {top_n_hdb} — Enrichment  ({hdb_src})",
                                xaxis_title="Log₂ fold-change vs tissue",
                                yaxis=dict(autorange='reversed', tickfont_size=16),
                                height=max(400, top_n_hdb * 22 + 80),
                                plot_bgcolor='white', margin=dict(l=10, r=10, t=60, b=50),
                                title_font_size=18, font=dict(size=15),
                                hoverlabel=dict(bgcolor='white', font_size=13, font_family='Arial'))
                            st.plotly_chart(fig_h_fc, width="stretch", key="hdb_lasso_fc")
                        with tab_ic_h:
                            _io = np.argsort(hdb_lraw)[::-1][:top_n_hdb]
                            _iv = [float(hdb_lraw[j]) for j in _io]
                            fig_h_ic = go.Figure(go.Bar(
                                x=_iv, y=[_hch[j] for j in _io], orientation='h',
                                marker_color=['#27ae60' if hdb_lfc[j] >= 0 else '#e74c3c' for j in _io],
                                hovertemplate="<b>%{y}</b><br>Ion count: %{x:,.1f}<extra></extra>",
                            ))
                            fig_h_ic.update_layout(
                                title=f"Top {top_n_hdb} — Ion Count  ({hdb_src})",
                                xaxis=dict(title="Mean ion count (ions/pixel)", type="log", title_font_size=15, tickfont_size=14),
                                yaxis=dict(autorange='reversed', tickfont_size=16),
                                height=max(400, top_n_hdb * 22 + 80),
                                plot_bgcolor='white', margin=dict(l=10, r=10, t=60, b=50),
                                title_font_size=18, font=dict(size=15),
                                hoverlabel=dict(bgcolor='white', font_size=13, font_family='Arial'))
                            st.plotly_chart(fig_h_ic, width="stretch", key="hdb_lasso_ic")
                else:
                    # Default: pre-saved PNGs — zero computation
                    tab_int_h, tab_fc_h, tab_ic_h = st.tabs(
                        ["Mean Intensity", "Enrichment vs Tissue", "Ion Count"])
                    with tab_int_h:
                        show_image(os.path.join(cluster_dir,
                            f"hdbscan_cluster{hdb_selected}_intensity.png"))
                    with tab_fc_h:
                        st.caption("Green = enriched vs whole tissue; red = depleted.")
                        show_image(os.path.join(cluster_dir,
                            f"hdbscan_cluster{hdb_selected}_lfc.png"))
                    with tab_ic_h:
                        st.caption("Raw ion count. Green = enriched; red = depleted.")
                        show_image(os.path.join(cluster_dir,
                            f"hdbscan_cluster{hdb_selected}_ioncount.png"))


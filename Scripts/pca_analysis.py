# =============================================================================
# pca_analysis.py
#
# Purpose of this .py file:
#   Runs PCA on the log1p transformed pixel data. 
#   PCA mainly used to understand which elements drive variation in the tissue
#   (via loading plots)
#
# Steps performed in this file:
#   1. Fit PCA on the log1p-normalised pixel dataframe
#   2. Scree plot: how much variance does each component explain?
#   3. Cumulative variance plot: how many components do we need?
#   4. Covariance matrix: how do the elements relate to each other?
#   5. 2D loading plot: arrows showing element contributions to two PCs
#   6. Return X_pca for use as input to UMAP
# =============================================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import plotly.express as px

# Figure 1: GPU detection 
try:
    import cuml
    import config as _cfg
    _GPU = getattr(_cfg, 'USE_GPU', True)
    if _GPU:
        cuml.set_global_output_type('numpy')
        from cuml.decomposition import PCA
    else:
        from sklearn.decomposition import PCA
except ImportError:
    from sklearn.decomposition import PCA
    _GPU = False

import config


# =============================================================================
# 1. Fit PCA
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

    No StandardScaler is applied here, log1p normalisation in preprocessing.py
    is sufficient to bring the channels onto a comparable scale.

    Parameters:
        df_normalised : log1p-normalised pixel dataframe from preprocessing.apply_log1p()
                        shape (n_tissue_pixels, n_channels)

    Returns:
        pca   : fitted sklearn PCA object (contains loadings, explained variance etc.)
        X_pca : numpy array of PCA coordinates, shape (n_tissue_pixels, n_components)
    """
    # Use all available components (one per channel)
    n_components = df_normalised.shape[1]

    print(f"Running PCA ({'GPU' if _GPU else 'CPU'}) on {df_normalised.shape[0]:,} pixels × {n_components} channels...")
    X = df_normalised.values.astype(np.float32)

    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)

    # Guarantee plain numpy output
    if hasattr(X_pca, 'to_numpy'):
        X_pca = X_pca.to_numpy()
    elif hasattr(X_pca, 'get'):
        X_pca = X_pca.get()
    X_pca = np.array(X_pca, dtype=np.float32)

    # Print how much variance each component explains
    print("Variance explained per component:")
    cumulative = 0
    for i, var in enumerate(pca.explained_variance_ratio_):
        cumulative += float(var) * 100
        print(f"  PC{i+1}: {float(var)*100:.1f}%  |  Cumulative: {cumulative:.1f}%")

    return pca, X_pca


# =============================================================================
# 2. Scree Plot
# =============================================================================

def plot_scree(pca: PCA, output_dir: str = None, show_plot: bool = False):
    """
    Plots the scree plot, which shows the explained variance ratio for each principal component.

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

    # Scale figure width with number of PCs so bars don't squish for large n
    fig_w = max(12, n * 0.35)
    plt.figure(figsize=(fig_w, 6))

    # Dots for each component
    dot_size = max(40, 200 - n * 2)  # shrink dots slightly for many PCs
    plt.scatter(x, pca.explained_variance_ratio_,
                s=dot_size, alpha=0.75, c='orange', edgecolor='k', label='Component variance')

    # Dashed line connecting the dots, helps see the elbow
    plt.plot(x, pca.explained_variance_ratio_,
             c='orange', linestyle='--', alpha=0.5)

    plt.grid(True)
    plt.title("Explained Variance Ratio: Scree Plot", fontsize=20)
    plt.xlabel("Principal Component", fontsize=14)
    plt.ylabel("Proportion of Variance Explained", fontsize=14)

    # For large PC counts, only label every 5th tick to avoid overlap
    if n <= 20:
        plt.xticks(list(x), fontsize=12)
    else:
        tick_positions = [i for i in x if i % 5 == 0 or i == 1]
        plt.xticks(tick_positions, fontsize=10)

    plt.yticks(fontsize=12)
    plt.tight_layout()

    if output_dir:
        plt.savefig(os.path.join(output_dir, "pca_scree.png"),
                    dpi=config.FIGURE_DPI, bbox_inches='tight')
    if show_plot:
        plt.show()
    plt.close()


# =============================================================================
# 3. Cumulative Variance Plot
# =============================================================================

def plot_cumulative_variance(pca: PCA, output_dir: str = None, show_plot: bool = False):
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

    if show_plot:
        fig.show()


# =============================================================================
# 4. Correlation heatmap 
# =============================================================================

def plot_correlation_matrix_interactive(df_normalised: pd.DataFrame,
                                         channel_names_filtered: list,
                                         output_dir: str = None):
    """
    Interactive Plotly version of the Pearson correlation heatmap.
    Hover over any cell to see: Element A, Element B, Pearson r, R².
    Click any cell to show a scatter plot of the two elements in the right panel.
    Saves as an HTML file.
    """
    import plotly.graph_objects as go
    import json

    channel_names = channel_names_filtered
    n_ch = len(channel_names)

    # Compute correlation matrix and R²
    corr_matrix = np.corrcoef(df_normalised.values, rowvar=False)
    r2_matrix   = corr_matrix ** 2

    # Build hover text matrix
    hover_text = []
    for i in range(n_ch):
        row_text = []
        for j in range(n_ch):
            r  = corr_matrix[i, j]
            r2 = r2_matrix[i, j]
            row_text.append(
                f"<b>{channel_names[i]}</b> vs <b>{channel_names[j]}</b><br>"
                f"Pearson r: {r:.3f}<br>"
                f"R²: {r2:.3f}<br>"
                f"<i>Click to view scatter →</i>"
            )
        hover_text.append(row_text)

    fig = go.Figure(go.Heatmap(
        z=corr_matrix,
        x=channel_names,
        y=channel_names,
        zmin=-1, zmax=1,
        colorscale='RdBu_r',
        hovertext=hover_text,
        hovertemplate='%{hovertext}<extra></extra>',
        colorbar=dict(title='Pearson r', thickness=15, len=0.8),
    ))

    fig.update_layout(
        title=dict(text='Pearson Correlation Matrix', font=dict(size=18)),
        width=1200, height=800,
        autosize=True,
        margin=dict(l=10, r=10, t=60, b=10),
        xaxis=dict(tickfont=dict(size=max(6, 11 - n_ch // 10)), tickangle=45),
        yaxis=dict(tickfont=dict(size=max(6, 11 - n_ch // 10)), autorange='reversed'),
        plot_bgcolor='white',
        paper_bgcolor='white',
    )

    if output_dir:
        # Subsample for scatter, max 20k points to keep HTML small
        MAX_SCATTER = 20000
        n_pixels = len(df_normalised)
        if n_pixels > MAX_SCATTER:
            idx = np.random.choice(n_pixels, MAX_SCATTER, replace=False)
            scatter_data = df_normalised.iloc[idx].values.tolist()
        else:
            scatter_data = df_normalised.values.tolist()

        post_script = f"""
(function() {{
    var channelNames = {json.dumps(channel_names)};
    var scatterData  = {json.dumps(scatter_data)};

    var mainDiv = document.querySelectorAll('.plotly-graph-div')[0];

    // Wrap in flex row
    var container = document.createElement('div');
    container.style.cssText = 'display:flex;flex-direction:row;width:100vw;height:100vh;overflow:hidden;';
    mainDiv.parentNode.insertBefore(container, mainDiv);
    container.appendChild(mainDiv);
    mainDiv.style.cssText = 'flex:1;min-width:0;height:100vh;';
    Plotly.relayout(mainDiv, {{autosize:true}});

    // Right panel
    var panel = document.createElement('div');
    panel.style.cssText = 'width:420px;min-width:420px;height:100vh;background:#111827;display:flex;flex-direction:column;border-left:1px solid #374151;font-family:monospace;overflow:hidden;';
    container.appendChild(panel);

    var header = document.createElement('div');
    header.id = 'corr-header';
    header.style.cssText = 'color:#9ca3af;font-size:12px;padding:10px 12px 8px;border-bottom:1px solid #374151;text-align:center;flex-shrink:0;';
    header.textContent = 'Click any cell to view scatter plot';
    panel.appendChild(header);

    var scatterDiv = document.createElement('div');
    scatterDiv.id = 'corr-scatter';
    scatterDiv.style.cssText = 'flex:1;min-height:0;';
    panel.appendChild(scatterDiv);

    var scatterLayout = {{
        paper_bgcolor:'#111827', plot_bgcolor:'#1f2937',
        font:{{color:'#d1d5db',family:'monospace',size:11}},
        margin:{{l:50,r:20,t:20,b:50}},
        xaxis:{{gridcolor:'#374151',linecolor:'#4b5563',zeroline:false}},
        yaxis:{{gridcolor:'#374151',linecolor:'#4b5563',zeroline:false}},
        showlegend:false
    }};

    Plotly.newPlot('corr-scatter',
        [{{type:'scatter',mode:'markers',x:[],y:[],
           marker:{{size:3,color:'#60a5fa',opacity:0.4}}}}],
        scatterLayout, {{responsive:true,displayModeBar:false}});

    mainDiv.on('plotly_click', function(data) {{
        var pt = data.points[0];
        if (pt === undefined) return;
        var xi = pt.x, yi = pt.y;
        var xi_idx = channelNames.indexOf(xi);
        var yi_idx = channelNames.indexOf(yi);
        if (xi_idx < 0 || yi_idx < 0) return;

        var xs = scatterData.map(function(row){{return row[xi_idx];}});
        var ys = scatterData.map(function(row){{return row[yi_idx];}});

        var r  = pt.z.toFixed(3);
        var r2 = (pt.z * pt.z).toFixed(3);

        document.getElementById('corr-header').innerHTML =
            '<b style="color:#f9fafb">' + xi + ' vs ' + yi + '</b><br>' +
            '<span style="color:#f87171">r = ' + r + '</span>' +
            '&nbsp;&nbsp;<span style="color:#4ade80">R² = ' + r2 + '</span>';

        // Colour points by intensity for visual richness
        var maxX = Math.max.apply(null, xs.map(Math.abs)) || 1;
        var colours = xs.map(function(v){{
            var t = Math.abs(v)/maxX;
            return 'rgb('+Math.round(96+t*159)+','+Math.round(165-t*100)+','+Math.round(250-t*200)+')';
        }});

        var updatedLayout = JSON.parse(JSON.stringify(scatterLayout));
        updatedLayout.xaxis.title = xi;
        updatedLayout.yaxis.title = yi;

        Plotly.react('corr-scatter',
            [{{type:'scatter',mode:'markers',
               x:xs, y:ys,
               marker:{{size:3,color:colours,opacity:0.5}},
               hovertemplate: xi+': %{{x:.2f}}<br>'+yi+': %{{y:.2f}}<extra></extra>'}}],
            updatedLayout);
    }});
}})();
"""
        path = os.path.join(output_dir, "pca_correlation_matrix_interactive.html")
        fig.write_html(path, post_script=post_script, config={'responsive': True})
        print(f"Saved interactive correlation matrix → {path}")


# =============================================================================
# 5 Covariance Matrix
# =============================================================================

def plot_covariance_matrix_interactive(df_normalised: pd.DataFrame,
                                        channel_names_filtered: list,
                                        pca: 'PCA' = None,
                                        output_dir: str = None):
    """
    Interactive Plotly version of the true covariance matrix.
    Hover over any cell to see covariance value.
    Right panel shows PC1 vs PC2 pixel scatter (if pca object provided).
    Saves as an HTML file.
    """
    import plotly.graph_objects as go
    import json

    channel_names = channel_names_filtered
    n_ch = len(channel_names)

    # Compute true covariance matrix
    cov_matrix = np.cov(df_normalised.values, rowvar=False)
    abs_max = float(np.percentile(np.abs(cov_matrix), 99))

    # Build hover text
    hover_text = []
    for i in range(n_ch):
        row_text = []
        for j in range(n_ch):
            row_text.append(
                f"<b>{channel_names[i]}</b> vs <b>{channel_names[j]}</b><br>"
                f"Covariance: {cov_matrix[i, j]:.4f}<br>"
                f"<i>Click to highlight row/col</i>"
            )
        hover_text.append(row_text)

    fig = go.Figure(go.Heatmap(
        z=cov_matrix.tolist(),
        x=channel_names,
        y=channel_names,
        zmin=-abs_max, zmax=abs_max,
        colorscale='RdBu_r',
        hovertext=hover_text,
        hovertemplate='%{hovertext}<extra></extra>',
        colorbar=dict(title='Covariance', thickness=15, len=0.8),
    ))

    fig.update_layout(
        title=dict(text='Covariance Matrix (log1p-normalised)', font=dict(size=18)),
        width=1200, height=800,
        autosize=True,
        margin=dict(l=10, r=10, t=60, b=10),
        xaxis=dict(tickfont=dict(size=max(6, 11 - n_ch // 10)), tickangle=45),
        yaxis=dict(tickfont=dict(size=max(6, 11 - n_ch // 10)), autorange='reversed'),
        plot_bgcolor='white',
        paper_bgcolor='white',
    )

    if output_dir:
        # Prepare PC1 vs PC2 scatter data if pca available
        if pca is not None:
            scores = pca.transform(df_normalised.values)
            MAX_SC = 20000
            if len(scores) > MAX_SC:
                idx = np.random.choice(len(scores), MAX_SC, replace=False)
                scores = scores[idx]
            pc1 = scores[:, 0].tolist()
            pc2 = scores[:, 1].tolist()
            pc1_var = float(pca.explained_variance_ratio_[0] * 100)
            pc2_var = float(pca.explained_variance_ratio_[1] * 100)
        else:
            pc1, pc2, pc1_var, pc2_var = [], [], 0.0, 0.0

        post_script = f"""
(function() {{
    var pc1Data  = {json.dumps(pc1)};
    var pc2Data  = {json.dumps(pc2)};
    var pc1Var   = {pc1_var:.1f};
    var pc2Var   = {pc2_var:.1f};

    var mainDiv = document.querySelectorAll('.plotly-graph-div')[0];

    var container = document.createElement('div');
    container.style.cssText = 'display:flex;flex-direction:row;width:100vw;height:100vh;overflow:hidden;';
    mainDiv.parentNode.insertBefore(container, mainDiv);
    container.appendChild(mainDiv);
    mainDiv.style.cssText = 'flex:1;min-width:0;height:100vh;';
    Plotly.relayout(mainDiv, {{autosize:true}});

    var panel = document.createElement('div');
    panel.style.cssText = 'width:420px;min-width:420px;height:100vh;background:#111827;display:flex;flex-direction:column;border-left:1px solid #374151;font-family:monospace;overflow:hidden;';
    container.appendChild(panel);

    var header = document.createElement('div');
    header.id = 'cov-header';
    header.style.cssText = 'color:#9ca3af;font-size:12px;padding:10px 12px 8px;border-bottom:1px solid #374151;text-align:center;flex-shrink:0;';
    header.textContent = 'PC1 vs PC2: pixel projections';
    panel.appendChild(header);

    var scatterDiv = document.createElement('div');
    scatterDiv.id = 'cov-scatter';
    scatterDiv.style.cssText = 'flex:1;min-height:0;';
    panel.appendChild(scatterDiv);

    var scatterLayout = {{
        paper_bgcolor:'#111827', plot_bgcolor:'#1f2937',
        font:{{color:'#d1d5db',family:'monospace',size:11}},
        margin:{{l:50,r:20,t:20,b:50}},
        xaxis:{{title:'PC1 (' + pc1Var.toFixed(1) + '% var)', gridcolor:'#374151',linecolor:'#4b5563',zeroline:true,zerolinecolor:'#6b7280'}},
        yaxis:{{title:'PC2 (' + pc2Var.toFixed(1) + '% var)', gridcolor:'#374151',linecolor:'#4b5563',zeroline:true,zerolinecolor:'#6b7280'}},
        showlegend:false
    }};

    Plotly.newPlot('cov-scatter',
        [{{type:'scatter',mode:'markers',x:pc1Data,y:pc2Data,
           marker:{{size:2,color:'#60a5fa',opacity:0.3}}}}],
        scatterLayout, {{responsive:true,displayModeBar:false}});

    mainDiv.on('plotly_click', function(data) {{
        var pt = data.points[0];
        if (pt === undefined) return;
        document.getElementById('cov-header').innerHTML =
            '<b style="color:#f9fafb">' + pt.y + ' vs ' + pt.x + '</b><br>' +
            '<span style="color:#f87171">Covariance: ' + pt.z.toFixed(4) + '</span>';
    }});
}})();
"""
        path = os.path.join(output_dir, "pca_covariance_matrix_interactive.html")
        fig.write_html(path, post_script=post_script, config={'responsive': True})
        print(f"Saved interactive covariance matrix → {path}")


# =============================================================================
# 6. Loading Plots
# =============================================================================

def plot_loadings(pca, channel_names: list, output_dir: str = None,
                  show_plot: bool = False, pc_pairs: tuple = ((1, 2), (2, 3))):
    """
    Loading plots showing each channel's contribution to selected pairs of
    principal components. Each channel is drawn as a labelled arrow from the
    origin to its (loading_PCx, loading_PCy) coordinate.

    Interpretation:
      - Channels far from the origin contribute strongly to those PCs.
      - Channels pointing in the same direction co-vary along those axes.
      - Channels pointing in opposite directions are inversely related.

    Parameters:
        pca           : fitted PCA object from run_pca()
        channel_names : list of channel name strings
        output_dir    : if provided, saves one PNG per PC pair
        show_plot     : if True, displays each figure
        pc_pairs      : tuple of (pc_x, pc_y) pairs to plot, 1-indexed
                        default: ((1, 2), (2, 3))
    """
    for (pc_x, pc_y) in pc_pairs:
        idx_x = pc_x - 1
        idx_y = pc_y - 1

        loadings_x = pca.components_[idx_x]
        loadings_y = pca.components_[idx_y]
        var_x = float(pca.explained_variance_ratio_[idx_x]) * 100
        var_y = float(pca.explained_variance_ratio_[idx_y]) * 100

        fig, ax = plt.subplots(figsize=(10, 8), facecolor='white')

        # Quadrant reference lines
        ax.axhline(0, color='grey', lw=0.8, ls='--', alpha=0.5)
        ax.axvline(0, color='grey', lw=0.8, ls='--', alpha=0.5)

        # Draw arrow and label for each channel
        for i, name in enumerate(channel_names):
            lx = float(loadings_x[i])
            ly = float(loadings_y[i])
            ax.annotate('', xy=(lx, ly), xytext=(0, 0),
                        arrowprops=dict(arrowstyle='->', color='#2c7bb6', lw=1.2))
            ax.text(lx * 1.08, ly * 1.08, name, fontsize=7,
                    ha='center', va='center', color='#333333')

        ax.set_xlabel(f'PC{pc_x} ({var_x:.1f}% variance)', fontsize=12)
        ax.set_ylabel(f'PC{pc_y} ({var_y:.1f}% variance)', fontsize=12)
        ax.set_title(f'PCA Loading Plot: PC{pc_x} vs PC{pc_y}',
                     fontsize=14, fontweight='bold')
        ax.spines[['top', 'right']].set_visible(False)
        plt.tight_layout()

        if output_dir:
            path = os.path.join(output_dir, f'pca_loadings_PC{pc_x}_PC{pc_y}.png')
            fig.savefig(path, dpi=config.FIGURE_DPI, bbox_inches='tight')
            print(f'  Saved → {path}')
        if show_plot:
            plt.show()
        plt.close()


# =============================================================================
# 7. Variance per Element Bar Chart
# =============================================================================

def plot_variance_per_element(pca: PCA, channel_names: list,
                               output_dir: str = None, show_plot: bool = False,
                               n_pcs: int = 5):
    """
    Bar chart showing what fraction of each element's variance is captured
    by the first n_pcs principal components.

    Uses the exact decomposition:
        var of element j explained by PC k  =  eigenvalue_k × loading_kj²
        total var of element j              =  sum over ALL PCs

    This does not require the original data, only the fitted PCA object.

    Green bars (≥ 80%)  = element well-represented by PCA; its spatial
                          pattern is captured in the reduced space.
    Amber bars (50–80%) = partially represented.
    Red bars   (< 50%)  = poorly captured; important spatial variation in
                          this element is lost after dimensionality reduction.

    Parameters:
        pca           : fitted sklearn PCA object
        channel_names : list of element name strings
        output_dir    : if provided, saves as 'pca_variance_per_element.png'
        show_plot     : if True, opens the figure
        n_pcs         : how many PCs to sum over (default 5)
    """
    n_pcs_use = min(n_pcs, len(pca.explained_variance_))
    n_ch      = len(channel_names)

    # Variance contributed by each PC to each element
    # pca.components_ shape: (n_components, n_features)
    # pca.explained_variance_ shape: (n_components,)
    total_var = np.zeros(n_ch)
    captured  = np.zeros(n_ch)

    for k in range(len(pca.explained_variance_)):
        contrib = pca.explained_variance_[k] * pca.components_[k] ** 2
        total_var += contrib
        if k < n_pcs_use:
            captured += contrib

    frac = captured / (total_var + 1e-10)

    order        = np.argsort(frac)[::-1]
    sorted_names = [channel_names[i] for i in order]
    sorted_frac  = frac[order]

    colours = ['#27ae60' if v >= 0.8 else ('#f39c12' if v >= 0.5 else '#e74c3c')
               for v in sorted_frac]

    fig_h = max(6, n_ch * 0.28)
    fig, ax = plt.subplots(figsize=(10, fig_h), facecolor='white')
    ax.barh(sorted_names, sorted_frac, color=colours, edgecolor='none', height=0.75)
    ax.axvline(0.5, color='grey',    lw=1.2, ls='--', alpha=0.7, label='50% threshold')
    ax.axvline(0.8, color='#2980b9', lw=1.2, ls='--', alpha=0.7, label='80% threshold')
    ax.set_xlabel(f'Fraction of variance captured by PC1–PC{n_pcs_use}', fontsize=12)
    ax.set_title(
        f'Element Variance Captured by First {n_pcs_use} PCs\n'
        'Green ≥ 80%: well-represented  |  Amber 50–80%: partial  |  Red < 50%: poorly captured',
        fontsize=11, fontweight='bold'
    )
    ax.set_xlim(0, 1.05)
    ax.invert_yaxis()
    ax.legend(fontsize=10, loc='lower right')
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(axis='y', labelsize=9)
    plt.tight_layout()

    if output_dir:
        path = os.path.join(output_dir, 'pca_variance_per_element.png')
        fig.savefig(path, dpi=config.FIGURE_DPI, bbox_inches='tight')
        print(f'  Saved → {path}')
    if show_plot:
        plt.show()
    plt.close()
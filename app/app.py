import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from fmannot.query.alphagenome import read_tracks_for_variant
from fmannot.plotting import plot_track
from fmannot.data_utils import load_gtf
from alphagenome.models import variant_scorers

import matplotlib as mpl
mpl.use("agg")

EFFECT_PREDICTIONS_PATH = Path("out/effect_predictions.feather")
TRACK_PREDICTIONS_PATH = Path("out/track_predictions.zarr")

@st.cache_data
def get_variant_list():
    """Loads the list of variants from the predictions file for dropdowns."""
    if not EFFECT_PREDICTIONS_PATH.exists():
        st.error(f"Data file not found: {EFFECT_PREDICTIONS_PATH}")
        return []
    df = pd.read_feather(EFFECT_PREDICTIONS_PATH)
    return sorted(df['variant'].unique().tolist())

@st.cache_data
def load_variant_effect_heatmap_data():
    """Performs the data processing for the heatmap."""
    best_scorers = [
        'ATAC', 'CONTACT_MAPS', 'DNASE', 'CHIP_TF', 'CHIP_HISTONE', 'CAGE',
        'PROCAP', 'RNA_SEQ', 'SPLICE_SITES', 'SPLICE_SITE_USAGE',
        'SPLICE_JUNCTIONS', 'POLYADENYLATION'
    ]
    scorers_to_keep = [str(variant_scorers.RECOMMENDED_VARIANT_SCORERS.get(s, None)) for s in best_scorers]

    df = (pd
        .read_feather(EFFECT_PREDICTIONS_PATH)
        .query('variant_scorer in @scorers_to_keep')
        .assign(abs_quantile_score = lambda x: x['quantile_score'].abs())
    )
    idx_max_abs = df.groupby(['variant', 'output_type'])['abs_quantile_score'].idxmax()
    df_wide_matrix = (
        df
        .loc[idx_max_abs]
        .assign(snp_gene = lambda x: x['variant'] + '\n(' + x['closest_gene'] + ')')
        .pivot(index='snp_gene', columns='output_type', values='quantile_score')
        .fillna(0)
    ).T
    return df_wide_matrix

@st.cache_data(show_spinner="Generating clustermap...")
def create_clustermap(df_wide):
    """Caches the Matplotlib Figure object itself."""
    g = sns.clustermap(
        df_wide,
        cmap=sns.diverging_palette(220, 20, as_cmap=True),
        vmin=-1.0, vmax=1.0, center=0.0,
        standard_scale=None, z_score=None,
        figsize=(18, 12),
        linewidths=0.5, linecolor='white',
        cbar_kws={'label': 'Quantile Score (Impact)'},
        xticklabels=True, yticklabels=True
    )
    g.fig.suptitle('Clustermap of Variant Impact by Feature', fontsize=18, y=1.03)
    g.fig.tight_layout(rect=[0, 0, 1, 0.96])
    return g.fig

@st.cache_data(show_spinner="Generating track plot...")
def create_track_plot(variant, _ref_track_data, _alt_track_data, _transcript_extractor):
    """Caches the Matplotlib Figure object for the track plot."""
    fig = plot_track(ref_track_data, alt_track_data, transcript_extractor, variant)
    return fig

@st.cache_resource(show_spinner="Loading GTF annotations...")
def get_gtf_data():
    """
    Loads and processes the GTF file using the library's smart loader.
    Cached so it only runs once per app session.
    """
    try:
        return load_gtf()
    
    except FileNotFoundError as e:
        st.error(f"Failed to load annotations: {e}")
        return None, None, None
    except Exception as e:
        st.error(f"An unexpected error occurred while loading GTF: {e}")
        return None, None, None

@st.cache_data(show_spinner="Loading Zarr track data...")
def cached_read_tracks(variant_id):
    """Caches the Zarr read for a specific variant."""
    return read_tracks_for_variant(
        variant_id, 
        store_path=str(TRACK_PREDICTIONS_PATH)
    )

# app info
st.set_page_config(page_title="FMAnnot Dashboard", layout="wide")
st.title("🧬 FMAnnot: Variant Annotation Dashboard")
st.markdown("Use the sidebar to select an analysis mode.")

# sidebar
st.sidebar.title("Controls")
analysis_mode = st.sidebar.selectbox(
    "Select Analysis",
    ["Variant Effects (Heatmap)", "Genome Tracks (by Variant)"]
)

# app body
if analysis_mode == "Variant Effects (Heatmap)":
    st.header("Variant Effect Clustermap", divider="rainbow")
    st.markdown("Clustering of variants (x-axis) and features (y-axis) by impact score.")
    
    try:
        df_wide = load_variant_effect_heatmap_data()
        fig = create_clustermap(df_wide)
        st.pyplot(fig, clear_figure=True)
        
    except FileNotFoundError:
        st.error(f"Error: Data file not found at {EFFECT_PREDICTIONS_PATH}.")
    except Exception as e:
        st.error(f"An error occurred: {e}")

else:
    st.header("Genome Track Inspection", divider="rainbow")
    
    variant_list = get_variant_list()
    if not variant_list:
        st.error("Could not load variant list. Please run the query script.")
        st.stop()
        
    selected_variant = st.sidebar.selectbox(
        "Select Variant",
        variant_list,
        index=variant_list.index('rs4844610') if 'rs4844610' in variant_list else 0,
        help="Select a variant to display its genomic tracks."
    )
    
    st.subheader(f"Displaying tracks for: {selected_variant}")
    
    try:
        gtf, gtf_transcripts, transcript_extractor = get_gtf_data()
        
        if gtf is None:
            st.stop()
        
        variant, ref_track_data, alt_track_data = cached_read_tracks(selected_variant)

        fig = create_track_plot(variant, ref_track_data, alt_track_data, transcript_extractor)
        st.pyplot(fig, clear_figure=True)

    except FileNotFoundError:
        st.error(f"Error: Data files not found. Ensure {TRACK_PREDICTIONS_PATH} exists.")
    except Exception as e:
        st.error(f"An error occurred: {e}")
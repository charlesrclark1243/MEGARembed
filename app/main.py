"""MEGARes Embedding Explorer — Streamlit entry point.

Run with:  uv run streamlit run app/main.py
"""

from __future__ import annotations

import faiss
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import umap

from mee.embed import MODELS, embed_batch, load_model
from mee.project import ONTOLOGY_LEVELS, load_reducer, project_query, scatter_frame
from mee.search import annotate_hits, load_index, prepare_query, search

st.set_page_config(page_title="MEGARes Embedding Explorer", layout="wide")

MAX_LEGEND_CATEGORIES: int = 12


# ---------------------------------------------------------------------------
# Cached loaders.
#
# The whole script reruns top-to-bottom on every interaction, so anything
# expensive has to be cached or you would reload the model on every keystroke.
# cache_resource is for live objects (models, indexes) that are not copied;
# cache_data is for values (DataFrames) that Streamlit may copy per caller.
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner="Loading ESM-2 model…")
def get_model(model_size: str, device: str = "cuda"):
    """Load the tokenizer and model once per model size."""

    return load_model(model_size, device=device)


@st.cache_resource(show_spinner="Loading FAISS index…")
def get_index(model_size: str) -> faiss.Index:
    """Load the FAISS index once per model size."""

    return load_index(model_size)


@st.cache_resource(show_spinner="Loading UMAP reducer…")
def get_reducer(model_size: str) -> umap.UMAP:
    """Load the fitted UMAP reducer once per model size."""

    return load_reducer(model_size)


@st.cache_data(show_spinner="Loading map…")
def get_scatter_frame(model_size: str) -> pd.DataFrame:
    """Load the coordinates + ontology table once per model size."""

    return scatter_frame(model_size)


def collapse_rare(labels: pd.Series, limit: int = MAX_LEGEND_CATEGORIES) -> pd.Series:
    """Keep the most common categories and bucket the rest, so the legend stays readable."""

    keep = labels.value_counts().head(limit).index

    return labels.where(labels.isin(keep), other="Other")


# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------

st.sidebar.header("Settings")
model_size: str = st.sidebar.selectbox("ESM-2 model", list(MODELS), index=1)
top_k: int = st.sidebar.slider("Neighbors to return", 5, 50, 10)
color_by: str = st.sidebar.selectbox("Color map by", ONTOLOGY_LEVELS, index=1)

st.title("MEGARes Embedding Explorer")
st.caption(
    "Exploratory research and education tool — **not** clinical or diagnostic. "
    "Cosine similarity over ESM-2 embeddings of translated MEGARes 4.0 genes."
)

# ---------------------------------------------------------------------------
# Query input.
#
# Results go into st.session_state because the script reruns on EVERY
# interaction: without this, changing the colour dropdown would re-embed the
# query and re-run the 2.4s UMAP transform.
# ---------------------------------------------------------------------------

sequence: str = st.text_area(
    "Query sequence",
    height=140,
    placeholder="Paste a protein sequence, or a nucleotide sequence (it will be translated).",
)

if st.button("Search", type="primary") and sequence.strip():
    try:
        protein, note = prepare_query(sequence)
    except ValueError as error:
        st.error(str(error))
        st.stop()

    tokenizer, model = get_model(model_size)

    with st.spinner("Embedding, searching and projecting…"):
        embedding: np.ndarray = embed_batch([protein], tokenizer, model)
        scores, indices = search(get_index(model_size), embedding, k=top_k)

        st.session_state["result"] = {
            "note": note,
            "model_size": model_size,
            "hits": annotate_hits(scores, indices, model_size),
            # search() works on its own copy, so this embedding is still raw —
            # which is what the reducer was fitted on
            "point": project_query(get_reducer(model_size), embedding),
        }

result: dict | None = st.session_state.get("result")
if result is not None and result["model_size"] != model_size:
    st.warning(f"Results below are from {result['model_size']}. Search again to use {model_size}.")

# ---------------------------------------------------------------------------
# Results table
# ---------------------------------------------------------------------------

if result is not None:
    st.info(f"Query {result['note']}")
    hits: pd.DataFrame = result["hits"]

    st.subheader(f"Top {len(hits)} neighbors")
    st.dataframe(hits, width="stretch", hide_index=True)

    if hits["requires_snp"].any():
        st.warning(
            "Rows flagged **requires_snp** confer resistance only via a specific "
            "variant, not by gene presence. A hit is not evidence of resistance."
        )

# ---------------------------------------------------------------------------
# UMAP scatter
# ---------------------------------------------------------------------------

st.subheader("Embedding map")
st.caption(
    "UMAP is for layout only — 2D distances are not metric. The neighbor table above, "
    "computed on the full embeddings, is the source of truth for similarity."
)

frame: pd.DataFrame = get_scatter_frame(model_size).copy()
frame["legend"] = collapse_rare(frame[color_by])

figure = px.scatter(
    frame,
    x="x",
    y="y",
    color="legend",
    render_mode="webgl",  # 10k+ SVG points would crawl
    hover_data={"meg_id": True, "x": False, "y": False, "legend": False, color_by: True},
    labels={"legend": color_by},
    height=650,
)
figure.update_traces(marker=dict(size=4, opacity=0.65))

if result is not None and result["model_size"] == model_size:
    point: np.ndarray = result["point"]
    figure.add_scatter(
        x=[point[0]],
        y=[point[1]],
        mode="markers",
        marker=dict(size=18, symbol="diamond", color="black", line=dict(width=2, color="white")),
        name="your query",
        hovertext="your query",
    )

figure.update_layout(legend=dict(itemsizing="constant"), margin=dict(l=0, r=0, t=10, b=0))

# on_select="rerun" makes Streamlit rerun the script when the user clicks or
# lassos points, handing the selection back as the return value.
event = st.plotly_chart(figure, key="umap", on_select="rerun", width="stretch")

selected: list = event.selection["points"] if event and event.selection else []
if selected:
    ids: list[str] = [p["customdata"][0] for p in selected if "customdata" in p]
    chosen: pd.DataFrame = frame[frame["meg_id"].isin(ids)]
    st.subheader(f"{len(chosen)} selected point(s)")
    st.dataframe(
        chosen[["meg_id", "aa_len", "requires_snp", *ONTOLOGY_LEVELS]],
        width="stretch",
        hide_index=True,
    )

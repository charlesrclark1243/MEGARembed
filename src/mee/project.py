from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import umap

from mee.common import OUT, index_map_path, umap_coords_path, umap_reducer_path

ONTOLOGY_LEVELS: list[str] = ["type", "class", "mechanism", "group"]


def load_coords(model_size: str) -> np.ndarray:
    """
    Load the precomputed 2D UMAP coordinates for a model size.

    Args:
        model_size (str): The ESM-2 model size, e.g. '650M'.

    Returns:
        np.ndarray: An (n, 2) array of coordinates, aligned to the index map.

    Raises:
        FileNotFoundError: If the projection has not been run yet.
    """

    path: Path = umap_coords_path(model_size)
    if not path.exists():
        raise FileNotFoundError(
            f"No UMAP coordinates at {path}. Generate them with: "
            f"uv run scripts/04_umap.py -s {model_size}"
        )

    return np.load(path)


def load_reducer(model_size: str) -> umap.UMAP:
    """
    Load the fitted UMAP reducer so new queries can be placed on the same map.

    Args:
        model_size (str): The ESM-2 model size, e.g. '650M'.

    Returns:
        umap.UMAP: The fitted reducer.

    Raises:
        FileNotFoundError: If the projection has not been run yet.
    """

    path: Path = umap_reducer_path(model_size)
    if not path.exists():
        raise FileNotFoundError(
            f"No UMAP reducer at {path}. Generate it with: "
            f"uv run scripts/04_umap.py -s {model_size}"
        )

    return joblib.load(path)


def scatter_frame(model_size: str) -> pd.DataFrame:
    """
    Build the table backing the scatter: one row per indexed protein, with
    coordinates and ontology side by side.

    Args:
        model_size (str): The ESM-2 model size, e.g. '650M'.

    Returns:
        pd.DataFrame: Columns x, y, meg_id, aa_len and the ontology levels.

    Raises:
        ValueError: If the coordinates and index map are out of sync.
    """

    coords: np.ndarray = load_coords(model_size)
    index_map: pd.DataFrame = pd.read_parquet(index_map_path(model_size))
    proteins: pd.DataFrame = pd.read_parquet(OUT / "megares_proteins.parquet")

    if len(coords) != len(index_map):
        raise ValueError(
            f"{len(coords)} coordinates but {len(index_map)} index-map rows; "
            "rerun 04_umap.py against the current embeddings."
        )

    frame: pd.DataFrame = index_map[["meg_id"]].copy()
    frame["x"] = coords[:, 0]
    frame["y"] = coords[:, 1]

    return frame.merge(
        proteins[["meg_id", "aa_len", "requires_snp", *ONTOLOGY_LEVELS]],
        on="meg_id",
        how="left",
        validate="one_to_one",
    )


def project_query(reducer: umap.UMAP, embedding: np.ndarray) -> np.ndarray:
    """
    Place a query embedding on the existing map.

    The reducer was fitted on raw (unnormalized) embeddings, so the query must be
    passed in the same form — the normalization that FAISS search does happens on
    its own copy and must not be applied here.

    Args:
        reducer (umap.UMAP): The fitted reducer.
        embedding (np.ndarray): A (1, dim) or (dim,) query embedding.

    Returns:
        np.ndarray: The (2,) coordinate for the query.
    """

    return reducer.transform(embedding.reshape(1, -1))[0]

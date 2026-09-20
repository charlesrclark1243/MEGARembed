from pathlib import Path

import numpy as np
import pandas as pd

ROOT: Path = Path(__file__).resolve().parents[2]

V4: Path = ROOT / "data" / "raw" / "megares_db_maintenance" / "database_files" / "v4"
FASTA: Path = V4 / "megares_database_v4.00.fasta"
ANNOT: Path = V4 / "megares_annotations_with_clusters_v4.00.csv"
OUT: Path = ROOT / "data" / "processed"

ARTIFACTS: Path = ROOT / "artifacts"

# Every artifact is keyed by model size so results stay comparable across models.
# These live here, at the bottom of the import stack, so no module that needs a
# path has to import a higher layer.


def embeddings_path(model_size: str) -> Path:
    """
    Path to the embeddings array for a model size.

    Args:
        model_size (str): The ESM-2 model size, e.g. '650M'.

    Returns:
        Path: Path to the embeddings .npy file.
    """

    return ARTIFACTS / f"embeddings_{model_size}.npy"


def index_path(model_size: str) -> Path:
    """
    Path to the FAISS index for a model size.

    Args:
        model_size (str): The ESM-2 model size, e.g. '650M'.

    Returns:
        Path: Path to the FAISS index file.
    """

    return ARTIFACTS / f"megares_{model_size}.faiss"


def index_map_path(model_size: str) -> Path:
    """
    Path to the row_idx -> meg_id map for a model size.

    Args:
        model_size (str): The ESM-2 model size, e.g. '650M'.

    Returns:
        Path: Path to the index map parquet file.
    """

    return ARTIFACTS / f"index_map_{model_size}.parquet"


def umap_coords_path(model_size: str) -> Path:
    """
    Path to the 2D UMAP coordinates for a model size.

    Args:
        model_size (str): The ESM-2 model size, e.g. '650M'.

    Returns:
        Path: Path to the coordinates .npy file.
    """

    return ARTIFACTS / f"umap_coords_{model_size}.npy"


def umap_reducer_path(model_size: str) -> Path:
    """
    Path to the fitted UMAP reducer for a model size.

    Args:
        model_size (str): The ESM-2 model size, e.g. '650M'.

    Returns:
        Path: Path to the reducer .joblib file.
    """

    return ARTIFACTS / f"umap_reducer_{model_size}.joblib"


def load_embeddings(model_size: str) -> np.ndarray:
    """
    Load protein embeddings and check they match the index map written by 02_embed.py.

    Args:
        model_size (str): The ESM-2 model size, e.g. '650M'.

    Returns:
        np.ndarray: A 2D float32 array of protein embeddings.

    Raises:
        FileNotFoundError: If the embeddings have not been generated.
        ValueError: If the embeddings and index map disagree on row count.
    """

    path: Path = embeddings_path(model_size)
    if not path.exists():
        raise FileNotFoundError(
            f"No embeddings at {path}. Generate them with: "
            f"uv run scripts/02_embed.py -s {model_size}"
        )

    embeddings: np.ndarray = np.load(path).astype(np.float32)
    rows: int = len(pd.read_parquet(index_map_path(model_size)))

    if len(embeddings) != rows:
        raise ValueError(
            f"{path.name} has {len(embeddings)} rows but its index map has {rows}; "
            "rerun 02_embed.py so they are written together."
        )

    return embeddings

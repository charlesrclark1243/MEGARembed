from __future__ import annotations

from pathlib import Path

import faiss
import numpy as np
import pandas as pd

from mee.common import ARTIFACTS, OUT
from mee.translate import classify

NUCLEOTIDE_ALPHABET: frozenset[str] = frozenset("ACGTUN")
NUCLEOTIDE_THRESHOLD: float = 0.9  # fraction of residues that must be ACGTUN

DISPLAY_COLUMNS: list[str] = [
    "meg_id",
    "type",
    "class",
    "mechanism",
    "group",
    "requires_snp",
]


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


def embeddings_path(model_size: str) -> Path:
    """
    Path to the embeddings array for a model size.

    Args:
        model_size (str): The ESM-2 model size, e.g. '650M'.

    Returns:
        Path: Path to the embeddings .npy file.
    """

    return ARTIFACTS / f"embeddings_{model_size}.npy"


def looks_like_nucleotide(sequence: str) -> bool:
    """
    Guess whether a sequence is DNA/RNA rather than protein.

    A, C, G, T, U and N are all valid amino acid letters, so ESM-2 will happily
    embed a nucleotide sequence and return a confident, meaningless answer. Any
    query has to be screened before it reaches the model.

    Args:
        sequence (str): The raw sequence to inspect.

    Returns:
        bool: True if the sequence is overwhelmingly nucleotide characters.
    """

    cleaned: str = "".join(sequence.split()).upper()
    if not cleaned:
        return False

    nucleotide_count: int = sum(char in NUCLEOTIDE_ALPHABET for char in cleaned)

    return nucleotide_count / len(cleaned) >= NUCLEOTIDE_THRESHOLD


def prepare_query(sequence: str) -> tuple[str, str]:
    """
    Normalize a user-supplied sequence to protein, translating nucleotide input.

    Args:
        sequence (str): The raw sequence, protein or nucleotide.

    Returns:
        tuple[str, str]: The protein sequence and a human-readable note about what was done.

    Raises:
        ValueError: If the sequence is empty or cannot be translated cleanly.
    """

    cleaned: str = "".join(sequence.split()).upper()
    if not cleaned:
        raise ValueError("Empty query sequence.")

    if not looks_like_nucleotide(cleaned):
        return cleaned, f"treated as protein ({len(cleaned)} aa)"

    result = classify(cleaned, mechanism="", group="")
    if result.status != "ok":
        raise ValueError(
            f"Query looks like nucleotide but did not translate cleanly "
            f"(status: {result.status}, internal stops: {result.internal_stops})."
        )

    note: str = (
        f"detected as nucleotide ({len(cleaned)} nt), "
        f"translated to {len(result.aa_seq)} aa"
    )
    if result.trimmed_bases:
        note += f", trimmed {result.trimmed_bases} trailing base(s)"

    return result.aa_seq, note


def load_index(model_size: str) -> faiss.Index:
    """
    Load the FAISS index for a model size.

    Args:
        model_size (str): The ESM-2 model size, e.g. '650M'.

    Returns:
        faiss.Index: The loaded index.

    Raises:
        FileNotFoundError: If the index has not been built yet.
    """

    path: Path = index_path(model_size)
    if not path.exists():
        raise FileNotFoundError(
            f"No index at {path}. Build it with: "
            f"uv run scripts/03_build_index.py -s {model_size}"
        )

    return faiss.read_index(str(path))


def load_index_map(model_size: str) -> pd.DataFrame:
    """
    Load the row_idx -> meg_id map for a model size.

    Args:
        model_size (str): The ESM-2 model size, e.g. '650M'.

    Returns:
        pd.DataFrame: The index map.

    Raises:
        FileNotFoundError: If the map has not been written yet.
    """

    path: Path = index_map_path(model_size)
    if not path.exists():
        raise FileNotFoundError(
            f"No index map at {path}. Build it with: "
            f"uv run scripts/02_embed.py -s {model_size}"
        )

    return pd.read_parquet(path)


def search(index: faiss.Index, query: np.ndarray, k: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """
    Search the index with a single query vector, returning true cosine scores.

    The index holds L2-normalized vectors, so the query must be normalized too:
    without it the ranking is still correct but the scores come back scaled by
    the query's norm and are not cosines.

    Args:
        index (faiss.Index): The FAISS index to search.
        query (np.ndarray): A (1, dim) or (dim,) query embedding.
        k (int): Number of neighbors to retrieve.

    Returns:
        tuple[np.ndarray, np.ndarray]: Cosine scores and row indices, with any
            unfilled (-1) slots removed from both.
    """

    vector: np.ndarray = np.ascontiguousarray(
        query.reshape(1, -1).astype(np.float32)
    )

    if vector.shape[1] != index.d:
        raise ValueError(
            f"Query has dimension {vector.shape[1]} but the index expects "
            f"{index.d} — the query and index were built with different models."
        )

    faiss.normalize_L2(vector)
    scores, indices = index.search(vector, k)

    # FAISS pads with -1 when it cannot fill k; drop those from BOTH arrays so
    # scores never drift out of alignment with their rows
    found: np.ndarray = indices[0] >= 0

    return scores[0][found], indices[0][found]


def annotate_hits(
    scores: np.ndarray, indices: np.ndarray, model_size: str
) -> pd.DataFrame:
    """
    Join FAISS row indices to their MEGARes accession and ontology.

    Args:
        scores (np.ndarray): Cosine scores from search().
        indices (np.ndarray): Row indices from search().
        model_size (str): The ESM-2 model size, used to pick the matching index map.

    Returns:
        pd.DataFrame: One row per hit with cosine score, accession and ontology.
    """

    index_map: pd.DataFrame = load_index_map(model_size)
    proteins: pd.DataFrame = pd.read_parquet(OUT / "megares_proteins.parquet")

    meg_ids: np.ndarray = index_map["meg_id"].to_numpy()[indices]

    hits: pd.DataFrame = pd.DataFrame({"cosine": scores, "meg_id": meg_ids})
    hits = hits.merge(
        proteins[DISPLAY_COLUMNS + ["aa_len"]],
        on="meg_id",
        how="left",
        validate="many_to_one",
    )

    if hits["class"].isna().any():
        raise ValueError(
            "Some hits did not resolve to an annotation — the index map and "
            "protein table are out of sync; rerun 02_embed.py."
        )

    return hits

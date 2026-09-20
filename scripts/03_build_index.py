from __future__ import annotations

import argparse

import faiss
import numpy as np

from mee.embed import MODELS
from mee.search import embeddings_path, index_map_path, index_path


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the FAISS index creation script.

    Returns:
        argparse.Namespace: The parsed command-line arguments.
    """

    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Build a FAISS cosine index from ESM-2 protein embeddings."
    )

    parser.add_argument(
        "--model-size",
        "-s",
        type=str,
        choices=MODELS.keys(),
        required=True,
        help="Model size whose embeddings to index. Determines every artifact path.",
    )

    return parser.parse_args()


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

    import pandas as pd

    path = embeddings_path(model_size)
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


def generate_index(embeddings: np.ndarray) -> faiss.Index:
    """
    Generate a FAISS index from protein embeddings.

    Normalizing in place turns inner product into cosine similarity. Queries must
    be normalized the same way, which mee.search.search() handles.

    Args:
        embeddings (np.ndarray): A 2D array of protein embeddings.

    Returns:
        faiss.Index: A FAISS index built from the protein embeddings.
    """

    faiss.normalize_L2(embeddings)  # in place; cosine == inner product after this
    index: faiss.Index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    return index


def main() -> None:
    """
    Build and write the FAISS index for one model size.
    """

    args: argparse.Namespace = parse_args()

    embeddings: np.ndarray = load_embeddings(args.model_size)
    print(f"Loaded {embeddings.shape} embeddings for {args.model_size}")

    index: faiss.Index = generate_index(embeddings)

    out = index_path(args.model_size)
    faiss.write_index(index, str(out))
    print(f"Wrote {out} ({index.ntotal} vectors, dim {index.d})")


if __name__ == "__main__":
    main()

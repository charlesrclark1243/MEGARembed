from __future__ import annotations

import argparse

import faiss
import numpy as np

from mee.common import index_path, load_embeddings
from mee.embed import MODELS


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

from __future__ import annotations

import argparse

import joblib
import numpy as np
import umap

from mee.common import load_embeddings, umap_coords_path, umap_reducer_path
from mee.embed import MODELS


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for UMAP dimensionality reduction.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """

    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Project protein embeddings to 2D with UMAP for the explorer map."
    )

    parser.add_argument(
        "--model-size",
        "-s",
        type=str,
        choices=MODELS.keys(),
        required=True,
        help="Model size whose embeddings to project. Determines every artifact path.",
    )
    parser.add_argument(
        "--num-neighbors",
        "-n",
        type=int,
        default=15,
        help="UMAP n_neighbors. Higher values favour global structure.",
    )
    parser.add_argument(
        "--min-dist",
        type=float,
        default=0.1,
        help="UMAP min_dist. Lower values pack clusters more tightly.",
    )
    parser.add_argument(
        "--metric",
        "-m",
        type=str,
        choices=["euclidean", "manhattan", "cosine"],
        default="cosine",
        help="Distance metric. Cosine matches how the FAISS index ranks neighbors.",
    )
    parser.add_argument(
        "--center",
        action="store_true",
        help="Subtract the mean vector first. Try this if the map is one undifferentiated blob.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed. Pinning it makes UMAP deterministic but single-threaded.",
    )

    return parser.parse_args()


def project(
    embeddings: np.ndarray,
    num_neighbors: int,
    min_dist: float,
    metric: str,
    seed: int,
) -> tuple[np.ndarray, umap.UMAP]:
    """
    Fit UMAP on the embeddings and return both the coordinates and the reducer.

    The reducer is kept so the app can place a user's query on this same frozen
    manifold via reducer.transform().

    Args:
        embeddings (np.ndarray): The input embeddings to reduce.
        num_neighbors (int): Number of neighbors for UMAP.
        min_dist (float): Minimum distance for UMAP.
        metric (str): Distance metric for UMAP.
        seed (int): Random seed for UMAP.

    Returns:
        tuple[np.ndarray, umap.UMAP]: The 2D coordinates and the fitted reducer.
    """

    reducer: umap.UMAP = umap.UMAP(
        n_neighbors=num_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=seed,
    )
    coords: np.ndarray = reducer.fit_transform(embeddings)

    return coords, reducer


def main() -> None:
    """
    Project one model's embeddings to 2D and write the coords and reducer.
    """

    args: argparse.Namespace = parse_args()

    embeddings: np.ndarray = load_embeddings(args.model_size)
    print(f"Loaded {embeddings.shape} embeddings for {args.model_size}")

    if args.center:
        embeddings = embeddings - embeddings.mean(axis=0, keepdims=True)
        print("Centered embeddings (subtracted the mean vector)")

    print(
        f"Fitting UMAP (n_neighbors={args.num_neighbors}, min_dist={args.min_dist}, "
        f"metric={args.metric}, seed={args.seed}) — single-threaded, this takes a few minutes"
    )
    coords, reducer = project(
        embeddings,
        num_neighbors=args.num_neighbors,
        min_dist=args.min_dist,
        metric=args.metric,
        seed=args.seed,
    )

    if len(coords) != len(embeddings):
        raise ValueError(
            f"UMAP returned {len(coords)} rows for {len(embeddings)} embeddings; "
            "coordinates must stay aligned with the index map."
        )

    coords_out = umap_coords_path(args.model_size)
    np.save(coords_out, coords.astype(np.float32))
    print(f"Wrote {coords_out} {coords.shape}")

    reducer_out = umap_reducer_path(args.model_size)
    joblib.dump(reducer, reducer_out)
    print(f"Wrote {reducer_out}")


if __name__ == "__main__":
    main()

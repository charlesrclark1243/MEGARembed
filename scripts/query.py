from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from mee.embed import MODELS, embed_batch, load_model
from mee.search import annotate_hits, load_index, prepare_query, search


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the query script.

    Returns:
        argparse.Namespace: The parsed command-line arguments.
    """

    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Find the MEGARes proteins nearest to a query sequence."
    )

    parser.add_argument(
        "--query",
        "-q",
        type=str,
        required=True,
        help="Sequence to search with. Protein, or nucleotide (translated automatically).",
    )
    parser.add_argument(
        "--model-size",
        "-s",
        type=str,
        choices=MODELS.keys(),
        required=True,
        help="Model size to embed with. Must match the model the index was built from.",
    )
    parser.add_argument(
        "--k", "-k", type=int, default=10, help="Number of neighbors to return."
    )
    parser.add_argument(
        "--device", "-d", type=str, default="cuda", help="Device to embed on."
    )

    return parser.parse_args()


def embed_query(sequence: str, model_size: str, device: str) -> np.ndarray:
    """
    Embed a single protein sequence using the specified ESM-2 model.

    Args:
        sequence (str): The protein sequence to embed.
        model_size (str): The size of the ESM-2 model to use.
        device (str): The device to embed on.

    Returns:
        np.ndarray: The (1, dim) embedding of the protein sequence.
    """

    tokenizer, model = load_model(model_size, device=device)

    return embed_batch([sequence], tokenizer, model, device=device)


def format_hits(hits: pd.DataFrame) -> str:
    """
    Render the hit table for the terminal, flagging SNP-confirmation entries.

    Args:
        hits (pd.DataFrame): Annotated hits from annotate_hits().

    Returns:
        str: A formatted table.
    """

    display: pd.DataFrame = hits.copy()
    display["cosine"] = display["cosine"].map(lambda value: f"{value:.4f}")
    display["meg_id"] = [
        f"{meg_id} *" if snp else meg_id
        for meg_id, snp in zip(display["meg_id"], display["requires_snp"])
    ]
    display = display.drop(columns=["requires_snp"])

    return display.to_string(index=False)


def main() -> None:
    """
    Embed a query sequence and report its nearest MEGARes neighbors.
    """

    args: argparse.Namespace = parse_args()

    sequence, note = prepare_query(args.query)
    print(f"Query: {note}")

    index = load_index(args.model_size)
    embedding: np.ndarray = embed_query(sequence, args.model_size, args.device)
    scores, indices = search(index, embedding, k=args.k)

    hits: pd.DataFrame = annotate_hits(scores, indices, args.model_size)

    print(f"\nTop {len(hits)} neighbors ({args.model_size}, cosine similarity):")
    print(format_hits(hits))

    if hits["requires_snp"].any():
        print(
            "\n* RequiresSNPConfirmation: resistance depends on a specific variant, "
            "not on gene presence. A hit here is not evidence of resistance."
        )


if __name__ == "__main__":
    main()

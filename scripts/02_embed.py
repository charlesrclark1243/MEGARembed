from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from mee.common import ARTIFACTS, OUT
from mee.embed import MODELS, embed_batch, load_model

DEFAULT_INPUT = OUT / "megares_proteins.parquet"
SPECIAL_TOKENS: int = 2  # BOS + EOS added by the ESM-2 tokenizer


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the embedding script.

    Returns:
        argparse.Namespace: The parsed command-line arguments.
    """

    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Embed protein sequences using ESM-2 models."
    )

    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default=str(DEFAULT_INPUT),
        help='Parquet file of protein sequences. Must have "meg_id" and "aa_seq" columns.',
    )
    parser.add_argument(
        "--model-size",
        "-s",
        type=str,
        choices=MODELS.keys(),
        required=True,
        help="Size of the ESM-2 model to use.",
    )
    parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=32,
        help="Maximum sequences per batch. Batches are also capped by --max-tokens.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=8192,
        help="Maximum padded tokens (sequences x longest in batch) per batch.",
    )
    parser.add_argument(
        "--max-len",
        type=int,
        default=1024,
        help="Model position limit. Longer sequences are truncated and logged.",
    )
    parser.add_argument(
        "--device", "-d", type=str, default="cuda", help="Device to embed on."
    )

    return parser.parse_args()


def load_proteins(path: str) -> pd.DataFrame:
    """
    Load the embeddable protein table and validate the columns the pipeline depends on.

    Args:
        path (str): Path to the parquet file.

    Returns:
        pd.DataFrame: The protein table, with row order preserved as the canonical row_idx.
    """

    df: pd.DataFrame = pd.read_parquet(path)

    missing: set[str] = {"meg_id", "aa_seq"} - set(df.columns)
    if missing:
        raise ValueError(f"Input is missing required column(s): {sorted(missing)}")

    # never drop rows here: row i of the output must stay row i of this table
    if df["aa_seq"].isna().any() or (df["aa_seq"].str.len() == 0).any():
        raise ValueError(
            "Input contains null/empty aa_seq values; rerun 01_parse_translate.py"
        )

    return df


def split_into_batches(
    lengths: list[int], batch_size: int, max_tokens: int, max_len: int
) -> list[list[int]]:
    """
    Group row indices into length-sorted batches, capped by both count and padded tokens.

    Sorting by length keeps padding waste low, and the token cap keeps the long tail
    from producing batches that will not fit in VRAM.

    Args:
        lengths (list[int]): Residue count per row.
        batch_size (int): Maximum sequences per batch.
        max_tokens (int): Maximum padded tokens per batch.
        max_len (int): Model position limit.

    Returns:
        list[list[int]]: Batches of row indices, each sorted short-to-long.
    """

    order: list[int] = sorted(range(len(lengths)), key=lambda i: lengths[i])

    batches: list[list[int]] = []
    current: list[int] = []
    longest: int = 0

    for idx in order:
        tokens: int = min(lengths[idx] + SPECIAL_TOKENS, max_len)
        candidate_longest: int = max(longest, tokens)

        if current and (
            len(current) + 1 > batch_size
            or (len(current) + 1) * candidate_longest > max_tokens
        ):
            batches.append(current)
            current, longest = [idx], tokens
        else:
            current.append(idx)
            longest = candidate_longest

    if current:
        batches.append(current)

    return batches


def embed_with_oom_retry(
    seqs: list[str],
    tokenizer: AutoTokenizer,
    model: AutoModel,
    device: str,
    max_len: int,
) -> np.ndarray:
    """
    Embed a batch, halving it and retrying if the GPU runs out of memory.

    Args:
        seqs (list[str]): Sequences in this batch.
        tokenizer (AutoTokenizer): The tokenizer for the ESM-2 model.
        model (AutoModel): The ESM-2 model.
        device (str): The device to embed on.
        max_len (int): Model position limit.

    Returns:
        np.ndarray: Embeddings for this batch, in the order given.
    """

    try:
        return embed_batch(
            seqs=seqs, tokenizer=tokenizer, model=model, device=device, max_len=max_len
        )
    except torch.cuda.OutOfMemoryError:
        if len(seqs) == 1:
            raise
        torch.cuda.empty_cache()
        mid: int = len(seqs) // 2
        tqdm.write(f"OOM on batch of {len(seqs)}; splitting into {mid} + {len(seqs) - mid}")
        first: np.ndarray = embed_with_oom_retry(
            seqs[:mid], tokenizer, model, device, max_len
        )
        second: np.ndarray = embed_with_oom_retry(
            seqs[mid:], tokenizer, model, device, max_len
        )
        return np.vstack([first, second])


def embed_sequences(
    df: pd.DataFrame,
    batches: list[list[int]],
    tokenizer: AutoTokenizer,
    model: AutoModel,
    device: str,
    max_len: int,
) -> np.ndarray:
    """
    Embed every batch and scatter the results back into input row order.

    Args:
        df (pd.DataFrame): The protein table.
        batches (list[list[int]]): Batches of row indices from split_into_batches.
        tokenizer (AutoTokenizer): The tokenizer for the ESM-2 model.
        model (AutoModel): The ESM-2 model.
        device (str): The device to embed on.
        max_len (int): Model position limit.

    Returns:
        np.ndarray: (n_rows, hidden_dim) float32 embeddings aligned to df's row order.
    """

    sequences: list[str] = df["aa_seq"].tolist()
    embeddings: np.ndarray | None = None

    for batch in tqdm(batches, desc="Embedding batches", unit="batch"):
        result: np.ndarray = embed_with_oom_retry(
            seqs=[sequences[i] for i in batch],
            tokenizer=tokenizer,
            model=model,
            device=device,
            max_len=max_len,
        )

        if embeddings is None:
            embeddings = np.zeros((len(sequences), result.shape[1]), dtype=np.float32)

        # scatter back: batches are length-sorted, the output must not be
        embeddings[batch] = result

    if embeddings is None:
        raise ValueError("No batches to embed.")

    return embeddings


def main() -> None:
    """
    Embed the translated MEGARes proteins and write the embeddings plus their index map.
    """

    args: argparse.Namespace = parse_args()
    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    df: pd.DataFrame = load_proteins(args.input)
    print(f"Loaded {len(df)} proteins from {args.input}")

    lengths: list[int] = df["aa_seq"].str.len().tolist()
    truncated: pd.Series = pd.Series(lengths) > (args.max_len - SPECIAL_TOKENS)
    if truncated.any():
        print(
            f"{truncated.sum()} sequences exceed {args.max_len - SPECIAL_TOKENS} residues "
            f"and will be truncated (longest: {max(lengths)})"
        )

    batches: list[list[int]] = split_into_batches(
        lengths, args.batch_size, args.max_tokens, args.max_len
    )
    print(f"Grouped into {len(batches)} length-sorted batches")

    tokenizer, model = load_model(args.model_size, device=args.device)

    embeddings: np.ndarray = embed_sequences(
        df=df,
        batches=batches,
        tokenizer=tokenizer,
        model=model,
        device=args.device,
        max_len=args.max_len,
    )

    emb_path = ARTIFACTS / f"embeddings_{args.model_size}.npy"
    np.save(emb_path, embeddings)
    print(f"Wrote {emb_path} {embeddings.shape} {embeddings.dtype}")

    # row_idx here is the contract: row i of the .npy is row i of this table
    index_map: pd.DataFrame = pd.DataFrame(
        {
            "row_idx": np.arange(len(df), dtype=np.int64),
            "meg_id": df["meg_id"].to_numpy(),
            "aa_len": lengths,
            "truncated": truncated.to_numpy(),
        }
    )
    map_path = ARTIFACTS / f"index_map_{args.model_size}.parquet"
    index_map.to_parquet(map_path, index=False)
    print(f"Wrote {map_path} ({len(index_map)} rows, {int(truncated.sum())} truncated)")


if __name__ == "__main__":
    main()

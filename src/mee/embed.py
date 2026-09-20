import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

MODELS: dict[str, str] = {
    "150M": "facebook/esm2_t30_150M_UR50D",
    "650M": "facebook/esm2_t33_650M_UR50D",
    "3B": "facebook/esm2_t36_3B_UR50D",
}


def load_model(model_id: str, device: str = "cuda") -> tuple[AutoTokenizer, AutoModel]:
    """
    Load the ESM-2 model and tokenizer based on the specified model ID.

    Args:
        model_id (str): The model ID to load. Must be one of '150M', '650M', '3B', or '15B'.
        device (str): The device to load the model onto. Default is 'cuda'.

    Returns:
        tuple[AutoTokenizer, AutoModel]: The loaded tokenizer and model.
    """

    if model_id not in MODELS:
        raise ValueError(
            f"Invalid model_id '{model_id}'. Must be one of {list(MODELS.keys())}."
        )

    tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(MODELS[model_id])
    model: AutoModel = (
        AutoModel.from_pretrained(MODELS[model_id], dtype=torch.float16)
        .to(device)
        .eval()
    )

    return tokenizer, model


@torch.inference_mode()
def embed_batch(
    seqs: list[str],
    tokenizer: AutoTokenizer,
    model: AutoModel,
    device: str = "cuda",
    max_len: int = 1024,
) -> np.ndarray:
    """
    Embed a batch of protein sequences using the specified ESM-2 model and tokenizer.

    Args:
        seqs (list[str]): A list of protein sequences to embed.
        tokenizer (AutoTokenizer): The tokenizer for the ESM-2 model.
        model (AutoModel): The ESM-2 model.
        device (str): The device to perform embedding on. Default is 'cuda'.
        max_len (int): The maximum length of sequences to embed. Sequences longer than this will be truncated.

    Returns:
        np.ndarray: A 2D array of embeddings, where each row corresponds to a sequence.
    """

    encodings: dict = tokenizer(
        seqs, padding=True, truncation=True, max_length=max_len, return_tensors="pt"
    ).to(device)
    outputs: torch.Tensor = model(**encodings).last_hidden_state

    mask: torch.Tensor = encodings["attention_mask"].clone().float()
    mask[:, 0] = 0.0  # zero out special tokens

    lengths: torch.Tensor = encodings["attention_mask"].sum(dim=1) - 1  # index of EOS
    mask[torch.arange(mask.size(0), device=mask.device), lengths] = 0.0
    mask = mask.unsqueeze(-1)

    summed: torch.Tensor = (outputs * mask).sum(dim=1)
    counts: torch.Tensor = mask.sum(dim=1).clamp(min=1)

    return (summed / counts).float().cpu().numpy()  # save as float32

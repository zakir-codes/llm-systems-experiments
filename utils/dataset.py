"""
Dataset utilities for training language models
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader

from utils.tokenizer_utils import load_tokenizer


class TextDataset(Dataset):
    """
    Dataset for autoregressive language modeling.

    Given a token sequence:
        [t0, t1, t2, t3...]

    It returns:
        x = [t0, t1, t2]
        y = [t1, t2, t3]
    """

    def __init__(self, tokens, block_size):

        self.tokens = torch.tensor(tokens, dtype=torch.long)
        self.block_size = block_size

    def __len__(self):
        return len(self.tokens) - self.block_size

    def __getitem__(self, idx):

        x = self.tokens[idx : idx + self.block_size]
        y = self.tokens[idx + 1 : idx + self.block_size + 1]

        return x, y


def load_or_create_tokens(file_path, tokenizer):
    """
    Tokenize dataset and cache tokens for faster future runs.
    """

    cache_path = file_path + ".tokens.pt"

    if os.path.exists(cache_path):

        tokens = torch.load(cache_path)

    else:

        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        tokens = tokenizer.encode(text)

        torch.save(tokens, cache_path)

    return tokens


def get_dataloader(
    file_path,
    batch_size,
    block_size,
    tokenizer_name,
    config,
):
    """
    Create DataLoader for training.
    """

    # Prepare tokenizer kwargs based on type and config
    tokenizer_kwargs = {}
    tokenizer_type = tokenizer_name

    if tokenizer_type in ("char", "word"):
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        tokenizer_kwargs["text"] = text
    elif tokenizer_type == "bpe":
        tok_cfg = config["tokenizer"]
        vocab_size = tok_cfg["vocab_size"]

        # Derive output_dir and model_prefix from tokenizer_path in config
        tokenizer_path = tok_cfg["tokenizer_path"]  # e.g. artifacts/tokenizers/bpe.model
        output_dir, filename = os.path.split(tokenizer_path)
        model_prefix, _ = os.path.splitext(filename)

        tokenizer_kwargs.update(
            dict(
                input_file=file_path,
                vocab_size=vocab_size,
                model_prefix=model_prefix,
                output_dir=output_dir,
            )
        )
    else:
        raise ValueError(f"Unknown tokenizer type: {tokenizer_type}")

    tokenizer = load_tokenizer(tokenizer_type=tokenizer_type, **tokenizer_kwargs)

    tokens = load_or_create_tokens(file_path, tokenizer)

    dataset = TextDataset(tokens, block_size)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )

    return loader
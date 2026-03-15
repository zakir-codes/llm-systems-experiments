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
):
    """
    Create DataLoader for training.
    """

    tokenizer = load_tokenizer(tokenizer_name)

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
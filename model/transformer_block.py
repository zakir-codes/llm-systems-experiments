"""Transformer Block Implementation"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.attention import MultiHead


class FeedForward(nn.Module):
    """Feed Forward Neural Network"""

    def __init__(self, n_embed, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Transformer block"""

    def __init__(self, n_head, n_embed, dropout, block_size, layer_idx=None):
        super().__init__()
        self.multi_head = MultiHead(n_head, n_embed, dropout, block_size, layer_idx)
        self.ff = FeedForward(n_embed, dropout)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x, kv_cache=None):
        x = x + self.multi_head(self.ln1(x), kv_cache=kv_cache)
        x = x + self.ff(self.ln2(x))
        return x

"""
Multi-Head Attention Implementation

Use PyTorch scaled dot-product attention (Flash Attention when available)
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHead(nn.Module):
    """Multiple heads of self-attention"""

    def __init__(self, n_head, n_embed, dropout, block_size, layer_idx=None):
        super().__init__()
        self.n_head = n_head
        self.head_size = n_embed // n_head
        self.layer_idx = layer_idx

        self.qkv_proj = nn.Linear(n_embed, 3 * n_embed, bias=False)

        self.out_proj = nn.Linear(n_embed, n_embed, bias=False)
        self.dropout = nn.Dropout(dropout)

        self.register_buffer(
            "mask",
            torch.tril(torch.ones(block_size, block_size)).view(
                1, 1, block_size, block_size
            ),
        )

    def forward(self, x, kv_cache=None):
        B, T, C = x.shape
        qkv = self.qkv_proj(x)
        Q, K, V = qkv.chunk(3, dim=-1)  # (3, B, T, n_head, head_size)

        Q = Q.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        K = K.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        V = V.view(B, T, self.n_head, self.head_size).transpose(1, 2)

        if kv_cache is not None:
            K, V = kv_cache.update(self.layer_idx, K, V)

        # Tq, Tk = Q.shape[-2], K.shape[-2]

        output = F.scaled_dot_product_attention(
            Q,
            K,
            V,
            # attn_mask=None if kv_cache is not None else self.mask[:, :, :Tq, :Tk],
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=kv_cache is None,
        )

        output = output.transpose(1, 2).contiguous().view(B, T, C)

        output = self.out_proj(output)
        output = self.dropout(output)

        return output

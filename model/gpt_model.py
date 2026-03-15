"""NanoGPT Language Model Implementation"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.transformer_block import Block
from systems.kv_cache import KVCache

class NanoGPTLanguageModel(nn.Module):
    """NanoGPT Language Model"""

    def __init__(self, block_size, vocab_size, n_layer, n_head, n_embed, dropout):
        super().__init__()
        self.token_embed_table = nn.Embedding(vocab_size, n_embed)
        self.pos_embed_table = nn.Embedding(block_size, n_embed)

        self.blocks = nn.ModuleList(
            [
                Block(n_head, n_embed, dropout, block_size, layer_idx=layer_idx)
                for layer_idx in range(n_layer)
            ]
        )
        self.ln = nn.LayerNorm(n_embed)
        self.lm = nn.Linear(n_embed, vocab_size)

    def forward(self, x, kv_cache=None, target=None):
        B, T = x.shape
        device = x.device
        pos_embed = self.pos_embed_table(torch.arange(T, device=device))
        output = self.token_embed_table(x) + pos_embed
        
        for block in self.blocks:
            output = block(output, kv_cache=kv_cache)
        output = self.ln(output)
        output = self.lm(output)

        if target is None:
            loss = None
        else:
            B, T, C = output.shape
            output = output.view(B * T, C)
            target = target.view(B * T)
            loss = F.cross_entropy(output, target)
        return output, loss  # logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):

        kv_cache = KVCache()
        logits, _ = self(idx, kv_cache=kv_cache)
        for _ in range(max_new_tokens):

            probs = torch.softmax(logits[:, -1, :], dim=-1)
            next_token = torch.multinomial(probs, 1)
            
            idx = torch.cat((idx, next_token), dim=1)
            
            logits, _ = self(next_token, kv_cache=kv_cache)

        return idx

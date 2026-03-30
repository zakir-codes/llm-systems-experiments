"""NanoGPT Language Model Implementation"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

from model.transformer_block import Block
from inference.kv_cache import KVCache


class NanoGPTLanguageModel(nn.Module):
    """NanoGPT Language Model"""

    def __init__(self, block_size, vocab_size, n_layer, n_head, n_embed, dropout):
        super().__init__()
        self.block_size = block_size
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
        self.apply(self._init_weights)

    def _init_weights(self, module):

        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x, kv_cache=None, target=None):
        B, T = x.shape
        device = x.device
        if kv_cache is not None:
            pos = kv_cache.cache_length()
            pos_embed = self.pos_embed_table(torch.arange(pos, pos + T, device=device))
        else:
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
    def generate(self, idx, max_new_tokens, kv_cache=None):
        """
        Generate new tokens using the model.
        
        Args:
            idx: Input token indices [batch_size, seq_len]
            max_new_tokens: Number of new tokens to generate
            kv_cache: KVCache instance for optimization (optional)
            
        Returns:
            Generated token indices [batch_size, seq_len + max_new_tokens]
        """
        if max_new_tokens <= 0:
            return idx
        
        try:
            for _ in range(max_new_tokens):
                # Determine context for forward pass
                if kv_cache is not None and not kv_cache.is_empty():
                    # Use KV cache: only need last token
                    idx_cond = idx[:, -1:]
                else:
                    # No cache: use full context (limited by block_size)
                    idx_cond = idx[:, -self.block_size:]
                
                # Forward pass
                logits, _ = self(idx_cond, kv_cache=kv_cache)
                
                # Sample next token
                probs = torch.softmax(logits[:, -1, :], dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Append to sequence
                idx = torch.cat((idx, next_token), dim=1)
                
        except Exception as e:
            logging.error(f"Error during generation: {e}")
            raise RuntimeError(f"Generation failed: {e}") from e
        
        return idx

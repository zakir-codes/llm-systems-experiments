"""NanoGPT Language Model Implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.transformer_block import Block

class NanoGPTLanguageModel(nn.Module):
    """NanoGPT Language Model"""
    def __init__(self, time, vocab_size, n_layer, n_head, n_embed, dropout):
        super().__init__()
        self.token_embed_table = nn.Embedding(vocab_size, n_embed)
        self.pos_embed_table = nn.Embedding(time, n_embed)

        self.sa = nn.Sequential(*[Block(n_head, n_embed, dropout,time) for _ in range(n_layer)])
        self.ln = nn.LayerNorm(n_embed)
        self.lm = nn.Linear(n_embed, vocab_size)

    def forward(self,x,target=None):
        B,T = x.shape
        device = x.device
        pos_embed = self.pos_embed_table( torch.arange(T, device=device))
        output = self.token_embed_table(x)+pos_embed
        
        output = self.sa(output)
        output = self.ln(output)
        output = self.lm(output)

        if target is None:
            loss = None
        else:
            B,T,C = output.shape
            output = output.view(B*T,C)
            target = target.view(B*T)
            loss = F.cross_entropy(output, target)
        return output, loss # logits, loss
    def generate(self):
        pass
    
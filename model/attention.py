"""
Multi-Head Attention Implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Head(nn.Module):
    """One head of causal self-attention"""
    def __init__(self,n_embed,head_size,dropout,block_size):
        super().__init__()

        self.query = nn.Linear(n_embed,head_size,bias=False)
        self.key = nn.Linear(n_embed,head_size, bias=False)
        self.value = nn.Linear(n_embed,head_size, bias=False)
        self.register_buffer("mask", torch.tril(torch.ones(block_size,block_size))) 
        self.dropout = nn.Dropout(dropout)
    
    def forward(self,x):
        B,T,C = x.shape

        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        
        output = Q @ K.transpose(-2,-1) * (Q.shape[-1]**-0.5) # attention score
        output = output.masked_fill(self.mask[:T,:T]==0, float("-inf"))
        output = torch.softmax(output, dim=-1)
        output = self.dropout(output)
        output = output @ V
        return output

class MultiHead(nn.Module):
    """Multiple heads of self-attention"""
    def __init__(self, n_head, n_embed, dropout, block_size):
        super().__init__()
        head_size = n_embed//n_head
        
        self.heads = nn.ModuleList([Head(n_embed,head_size, dropout, block_size) for _ in range(n_head)])
        self.project = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self,x):

        output = torch.cat([h(x) for h in self.heads], dim=-1)
        output = self.project(output)
        output = self.dropout(output)
        return output

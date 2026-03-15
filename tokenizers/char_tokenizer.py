
"""Tokenizer that splits text into characters."""
from tokenizers.base import BaseTokenizer


class CharTokenizer(BaseTokenizer):
    def __init__(self, text):
        chars = sorted(list(set(text)))
        self.stoi = {ch:i for i,ch in enumerate(chars)}
        self.itos = {i:ch for i,ch in enumerate(chars)}

    def encode(self, s):
        return [self.stoi[c] for c in s]

    def decode(self, l):
        return ''.join([self.itos[i] for i in l])

    @property
    def vocab_size(self):
        return len(self.stoi)
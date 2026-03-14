
"""Tokenizer that splits text into words."""
from tokenizers.base import BaseTokenizer


class WordTokenizer(BaseTokenizer):
    def __init__(self, text):
        words = text.split()
        vocab = sorted(set(words))

        self.stoi = {w:i for i,w in enumerate(vocab)}
        self.itos = {i:w for i,w in enumerate(vocab)}

    def encode(self, text):
        return [self.stoi[w] for w in text.split() if w in self.stoi]

    def decode(self, tokens):
        return " ".join([self.itos[i] for i in tokens])

    @property
    def vocab_size(self):
        return len(self.stoi)
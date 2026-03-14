"""Base class for tokenizers."""
class BaseTokenizer:
    def encode(self, text):
        raise NotImplementedError

    def decode(self, tokens):
        raise NotImplementedError

    @property
    def vocab_size(self):
        raise NotImplementedError
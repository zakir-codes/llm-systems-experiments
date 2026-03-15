""" This file defines the tokenizer, which is responsible for converting text to tokens and vice versa"""
from tokenizers import CharTokenizer, BPETokenizer, WordTokenizer

def load_tokenizer(tokenizer_type, **kwargs):
    if tokenizer_type == "char":
        return CharTokenizer(**kwargs)
    elif tokenizer_type == "bpe":
        return BPETokenizer(**kwargs)
    elif tokenizer_type == "word":
        return WordTokenizer(**kwargs)
    else:
        raise ValueError(f"Unknown tokenizer type: {tokenizer_type}")
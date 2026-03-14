"""Byte Pair Encoding (BPE) tokenizer implementation."""

import os

from tokenizers.base import BaseTokenizer
import sentencepiece as spm

class BPETokenizer(BaseTokenizer):
    def __init__(self, model_file=None, input_file=None, model_prefix='bpe_tokenizer', vocab_size=8000):
        self.sp = spm.SentencePieceProcessor()
        if model_file:
            self.sp.load(model_file)
        elif input_file:
            self._train(input_file, model_prefix, vocab_size)
        else:
            raise ValueError("Either model_file or input_file must be provided.")
        
    def _train(self, input_file, model_prefix, vocab_size=8000):
        output_dir = "checkpoints/tokenizer"
        os.makedirs(output_dir, exist_ok=True)

        model_prefix_path = os.path.join(output_dir, model_prefix)
        model_file = f"{model_prefix_path}.model"
        vocab_file = f"{model_prefix_path}.vocab"
        # if files already exist, avoid retraining
        if os.path.exists(model_file) and os.path.exists(vocab_file):
            # user might have trained this tokenizer earlier
            self.sp.load(model_file)
            return

        spm.SentencePieceTrainer.train(
                                    input=input_file,
                                    model_prefix=model_prefix_path,
                                    vocab_size=vocab_size,
                                    
                                    )
        # Load the newly trained model so encode/decode work immediately
        self.sp.load(model_file)
        

    def encode(self, text):
        return self.sp.encode(text)

    def decode(self, tokens):
        return self.sp.decode(tokens)

    @property
    def vocab_size(self):
        return self.sp.vocab_size()
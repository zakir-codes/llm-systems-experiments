"""KV Cache Implementation"""

import torch


class KVCache:
    """Key-Value Cache for Transformer Models"""

    def __init__(self, max_seq_len, n_head, head_dim, batch_size, device):
        self.key_cache = torch.zeros(
            batch_size, n_head, max_seq_len, head_dim, device=device
        )

        self.value_cache = torch.zeros(
            batch_size, n_head, max_seq_len, head_dim, device=device
        )

        self.pos = 0

    def update(self, layer_idx, key, value):
        T = key.size(2)

        self.key_cache[:, :, self.pos:self.pos + T] = key
        self.value_cache[:, :, self.pos:self.pos + T] = value

        self.pos += T

        return (
            self.key_cache[:, :, :self.pos],
            self.value_cache[:, :, :self.pos],
        )

    def cache_length(self):
        return self.pos

"""KV Cache Implementation"""

import torch

class KVCache:
    """Key-Value Cache for Transformer Models"""
    def __init__(self):
        self.cache = {}

    def store(self, layer_idx, key, value):
        """Store key and value for a given layer index"""
        self.cache[layer_idx] = (key, value)

    def update(self, layer_idx, key, value):
        """Update key and value for a given layer index"""
        if layer_idx in self.cache:
            existing_key, existing_value = self.cache[layer_idx]
            updated_key = torch.cat((existing_key, key), dim=1)
            updated_value = torch.cat((existing_value, value), dim=1)
            self.cache[layer_idx] = (updated_key, updated_value)
        else:
            self.store(layer_idx, key, value)
        return self.cache[layer_idx]

    def retrieve(self, layer_idx):
        """Retrieve key and value for a given layer index"""
        return self.cache.get(layer_idx, (None, None))
"""KV Cache Implementation for Transformer Models"""

import torch
import logging

logger = logging.getLogger(__name__)


class KVCache:
    """Key-Value Cache for Transformer Models with per-layer storage"""

    def __init__(self, max_seq_len, n_head, head_dim, n_layer, batch_size, device):
        """
        Initialize KV cache with per-layer storage.
        
        Args:
            max_seq_len: Maximum sequence length for cache
            n_head: Number of attention heads
            head_dim: Dimension of each attention head
            n_layer: Number of transformer layers
            batch_size: Batch size for cached tensors
            device: Device for cache storage
        """
        self.max_seq_len = max_seq_len
        self.n_head = n_head
        self.head_dim = head_dim
        self.n_layer = n_layer
        self.batch_size = batch_size
        self.device = device
        
        # Initialize per-layer cache storage
        self.key_caches = {}
        self.value_caches = {}
        self.pos = 0
        
        # Pre-allocate cache tensors for each layer
        for layer_idx in range(n_layer):
            self.key_caches[layer_idx] = torch.zeros(
                batch_size, n_head, max_seq_len, head_dim, device=device
            )
            self.value_caches[layer_idx] = torch.zeros(
                batch_size, n_head, max_seq_len, head_dim, device=device
            )
        
        logger.debug(f"Initialized KVCache: {n_layer} layers, max_seq_len={max_seq_len}")

    def update(self, layer_idx, key, value):
        """
        Update cache for specific layer and return cached values.
        
        Args:
            layer_idx: Layer index to update
            key: New key tensor [batch_size, n_head, seq_len, head_dim]
            value: New value tensor [batch_size, n_head, seq_len, head_dim]
            
        Returns:
            Tuple of (cached_keys, cached_values) for this layer
        """
        if layer_idx not in self.key_caches:
            raise ValueError(f"Invalid layer index: {layer_idx}")
        
        batch_size, n_head, seq_len, head_dim = key.shape
        
        # Validate tensor shapes
        if (n_head != self.n_head or head_dim != self.head_dim or 
            batch_size != self.batch_size):
            raise ValueError(
                f"Tensor shape mismatch: expected ({self.batch_size}, {self.n_head}, "
                f"*, {self.head_dim}), got {key.shape}"
            )
        
        # Check cache capacity
        if self.pos + seq_len > self.max_seq_len:
            logger.warning(f"Cache overflow: pos={self.pos}+{seq_len} > max_seq_len={self.max_seq_len}")
            # Reset cache if overflow
            self.pos = 0
        
        # Store new keys and values
        self.key_caches[layer_idx][:, :, self.pos:self.pos + seq_len] = key
        self.value_caches[layer_idx][:, :, self.pos:self.pos + seq_len] = value
        
        # Update position
        self.pos += seq_len
        
        # Return cached values for this layer
        return (
            self.key_caches[layer_idx][:, :, :self.pos],
            self.value_caches[layer_idx][:, :, :self.pos],
        )

    def get_cached_kv(self, layer_idx):
        """
        Get cached keys and values for specific layer.
        
        Args:
            layer_idx: Layer index
            
        Returns:
            Tuple of (cached_keys, cached_values) or (None, None) if not cached
        """
        if layer_idx not in self.key_caches or self.pos == 0:
            return None, None
        
        return (
            self.key_caches[layer_idx][:, :, :self.pos],
            self.value_caches[layer_idx][:, :, :self.pos],
        )

    def cache_length(self):
        """Return current cache length (number of cached tokens)"""
        return self.pos

    def reset(self):
        """Reset cache position to 0 (clear cached content)"""
        self.pos = 0
        logger.debug("KVCache reset")

    def is_empty(self):
        """Check if cache is empty"""
        return self.pos == 0

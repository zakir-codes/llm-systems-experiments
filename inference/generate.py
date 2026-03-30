"""Text Generation Utilities for NanoGPT

This module provides a clean interface for text generation that works
with NanoGPT model's generate method, supporting both KV cache
and non-KV cache inference modes.
"""

import torch
import logging
from typing import Optional, Union

from inference.kv_cache import KVCache

logger = logging.getLogger(__name__)


class GenerationConfig:
    """Configuration for text generation parameters"""
    
    def __init__(
        self,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        use_kv_cache: bool = True,
        block_size: int = 128,
        n_layer: int = 4,
        n_head: int = 4,
        n_embed: int = 256,
        batch_size: int = 1,
        device: str = "cpu"
    ):
        """
        Initialize generation configuration.
        
        Args:
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (1.0 = no scaling)
            top_k: Top-k sampling threshold (None = disabled)
            use_kv_cache: Whether to use KV cache optimization
            block_size: Model's block size for context window
            n_layer: Number of transformer layers
            n_head: Number of attention heads
            n_embed: Embedding dimension
            batch_size: Batch size for generation
            device: Device for computation
        """
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_k = top_k
        self.use_kv_cache = use_kv_cache
        self.block_size = block_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embed = n_embed
        self.batch_size = batch_size
        self.device = device
        
        self._validate_config()
    
    def _validate_config(self):
        """Validate configuration parameters"""
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
        if self.top_k is not None and self.top_k <= 0:
            raise ValueError("top_k must be positive if specified")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")


def create_kv_cache(config: GenerationConfig) -> Optional[KVCache]:
    """
    Create a properly initialized KV cache from configuration.
    
    Args:
        config: GenerationConfig instance
        
    Returns:
        KVCache instance or None if use_kv_cache is False
    """
    if not config.use_kv_cache:
        return None
    
    try:
        head_dim = config.n_embed // config.n_head
        return KVCache(
            max_seq_len=config.block_size,
            n_head=config.n_head,
            head_dim=head_dim,
            n_layer=config.n_layer,
            batch_size=config.batch_size,
            device=config.device
        )
    except Exception as e:
        logger.error(f"Failed to create KV cache: {e}")
        raise RuntimeError(f"KV cache creation failed: {e}") from e


def generate(
    model,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    use_kv_cache: bool = True,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
) -> torch.Tensor:
    """
    Generate text using NanoGPT model.
    
    This function provides a clean interface that handles KV cache
    initialization and delegates to model's generate method.
    
    Args:
        model: NanoGPTLanguageModel instance
        input_ids: Input token indices [batch_size, seq_len]
        max_new_tokens: Number of tokens to generate
        use_kv_cache: Whether to use KV cache optimization
        temperature: Sampling temperature (1.0 = no scaling)
        top_k: Top-k sampling threshold (None = disabled)
    
    Returns:
        Generated token indices [batch_size, input_len + max_new_tokens]
        
    Raises:
        RuntimeError: If generation fails
        ValueError: If parameters are invalid
    """
    # Validate inputs
    if not isinstance(input_ids, torch.Tensor):
        raise ValueError("input_ids must be a torch.Tensor")
    if input_ids.dim() != 2:
        raise ValueError("input_ids must be 2-dimensional [batch_size, seq_len]")
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    
    # Extract model parameters for cache creation
    try:
        config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            use_kv_cache=use_kv_cache,
            block_size=model.block_size,
            n_layer=len(model.blocks),
            n_head=model.blocks[0].n_head if model.blocks else 4,
            n_embed=model.n_embed,
            batch_size=input_ids.size(0),
            device=input_ids.device
        )
    except Exception as e:
        logger.error(f"Failed to create generation config: {e}")
        raise ValueError(f"Invalid generation parameters: {e}") from e
    
    # Create KV cache if needed
    kv_cache = create_kv_cache(config)
    
    if kv_cache is None and use_kv_cache:
        logger.warning("KV cache requested but not available, using non-cached generation")
    
    try:
        # Delegate to model's generate method
        generated_tokens = model.generate(
            idx=input_ids,
            max_new_tokens=max_new_tokens,
            kv_cache=kv_cache
        )
        
        logger.info(f"Generated {max_new_tokens} tokens successfully")
        return generated_tokens
        
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        raise RuntimeError(f"Text generation failed: {e}") from e


def generate_with_config(
    model,
    input_ids: torch.Tensor,
    config: GenerationConfig
) -> torch.Tensor:
    """
    Generate text using a GenerationConfig object.
    
    This provides an alternative interface for more complex generation scenarios.
    
    Args:
        model: NanoGPTLanguageModel instance
        input_ids: Input token indices [batch_size, seq_len]
        config: GenerationConfig instance
        
    Returns:
        Generated token indices [batch_size, input_len + max_new_tokens]
        
    Raises:
        RuntimeError: If generation fails
        ValueError: If config is invalid
    """
    if not isinstance(config, GenerationConfig):
        raise ValueError("config must be a GenerationConfig instance")
    
    # Override batch size and device based on input
    config.batch_size = input_ids.size(0)
    config.device = input_ids.device
    
    # Create KV cache
    kv_cache = create_kv_cache(config)
    
    try:
        generated_tokens = model.generate(
            idx=input_ids,
            max_new_tokens=config.max_new_tokens,
            kv_cache=kv_cache
        )
        
        logger.info(f"Generated {config.max_new_tokens} tokens with config")
        return generated_tokens
        
    except Exception as e:
        logger.error(f"Config-based generation failed: {e}")
        raise RuntimeError(f"Text generation failed: {e}") from e


def validate_generation_inputs(
    model,
    input_ids: torch.Tensor,
    max_new_tokens: int
) -> bool:
    """
    Validate inputs for generation functions.
    
    Args:
        model: NanoGPTLanguageModel instance
        input_ids: Input token indices
        max_new_tokens: Number of tokens to generate
        
    Returns:
        True if inputs are valid
        
    Raises:
        ValueError: If inputs are invalid
    """
    if not hasattr(model, 'generate'):
        raise ValueError("Model must have a generate method")
    
    if not isinstance(input_ids, torch.Tensor):
        raise ValueError("input_ids must be a torch.Tensor")
    
    if input_ids.dim() != 2:
        raise ValueError("input_ids must be 2-dimensional [batch_size, seq_len]")
    
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    
    return True
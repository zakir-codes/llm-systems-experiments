"""Factory for creating system optimization modules"""

import logging
from .mixed_precision import MixedPrecision
from .gradient_accumulation import GradientAccumulation

logger = logging.getLogger(__name__)


class SystemsFactory:
    """Factory for creating training optimization systems"""
    
    @staticmethod
    def create_mixed_precision(model, optimizer, device):
        """Create mixed precision system if device supports it"""
        try:
            if device == "cuda":
                return MixedPrecision(model, optimizer, device)
            else:
                logger.info(f"Mixed precision disabled: device {device} is not CUDA")
                return None
        except Exception as e:
            logger.error(f"Failed to create mixed precision system: {e}")
            return None
    
    @staticmethod
    def create_gradient_accumulation(model, optimizer, accumulation_steps):
        """Create gradient accumulation system if enabled"""
        try:
            if accumulation_steps > 1:
                return GradientAccumulation(model, optimizer, accumulation_steps)
            else:
                logger.info(f"Gradient accumulation disabled: steps={accumulation_steps}")
                return None
        except Exception as e:
            logger.error(f"Failed to create gradient accumulation system: {e}")
            return None
    
    @staticmethod
    def create_systems(model, optimizer, device, use_amp=False, accumulation_steps=1):
        """Create all requested systems with proper error handling"""
        systems = {}
        
        # Create mixed precision system
        if use_amp:
            systems["mixed_precision"] = SystemsFactory.create_mixed_precision(
                model, optimizer, device
            )
        
        # Create gradient accumulation system
        if accumulation_steps > 1:
            systems["gradient_accumulation"] = SystemsFactory.create_gradient_accumulation(
                model, optimizer, accumulation_steps
            )
        
        logger.info(f"Created systems: {list(systems.keys())}")
        return systems

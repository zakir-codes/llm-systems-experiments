"""Factory for creating system optimization modules"""

import logging
from .normal_training import NormalTraining
from .mixed_precision import MixedPrecision
from .gradient_accumulation import GradientAccumulation
from .combined_training import CombinedTraining

logger = logging.getLogger(__name__)


class SystemsFactory:
    """Factory for creating training optimization systems"""
    
    @staticmethod
    def create_training_system(system_name, model, optimizer, device, **kwargs):
        """Create training system based on config specification"""
        
        if system_name == "normal":
            return NormalTraining(
                model, optimizer, device, 
                grad_clip=kwargs.get("grad_clip", 1.0)
            )
        
        elif system_name == "mixed_precision":
            return MixedPrecision(model, optimizer, device)
        
        elif system_name == "gradient_accumulation":
            return GradientAccumulation(
                model, optimizer, 
                kwargs.get("accumulation_steps", 1)
            )
        
        elif system_name == "combined":
            return CombinedTraining(
                model, optimizer, device,
                accumulation_steps=kwargs.get("accumulation_steps", 4),
                grad_clip=kwargs.get("grad_clip", 1.0)
            )
        
        else:
            raise ValueError(f"Unknown training system: {system_name}")
    
    @staticmethod
    def create_from_config(config, model, optimizer, device):
        """Create training system from full config"""
        systems_config = config.get("systems", {})
        system_name = systems_config.get("training_system", "normal")
        
        # Extract system-specific parameters
        kwargs = {}
        
        if system_name in ["gradient_accumulation", "combined"]:
            grad_config = systems_config.get("gradient_accumulation", {})
            kwargs["accumulation_steps"] = grad_config.get("steps", 4)
        
        if system_name in ["normal", "combined"]:
            kwargs["grad_clip"] = systems_config.get("grad_clip", 1.0)
        
        logger.info(f"Creating training system: {system_name}")
        return SystemsFactory.create_training_system(
            system_name, model, optimizer, device, **kwargs
        )

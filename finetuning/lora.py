"""Production-Grade LoRA (Low-Rank Adaptation) Implementation

Paper-accurate implementation following standard LoRA formulation:
- LoRA paper: https://arxiv.org/abs/2106.09685
- Standard shapes: A (r, in_features), B (out_features, r)
- Forward: W + BA (not W + AB)
- PEFT-style design patterns with generic target modules
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import copy


class LoRALinear(nn.Module):
    """LoRA linear layer wrapper with paper-accurate formulation"""
    
    def __init__(self, original_layer: nn.Linear, r: int = 8, alpha: int = 16, dropout: float = 0.05):
        super().__init__()
        self.original_layer = original_layer
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        
        # Paper-accurate LoRA weight shapes
        # A: (r, in_features), B: (out_features, r)
        self.lora_A = nn.Parameter(torch.randn(r, original_layer.in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(original_layer.out_features, r))
        self.dropout = nn.Dropout(dropout)
        
        # Freeze original weights
        for param in self.original_layer.parameters():
            param.requires_grad = False
        
        # Store original layer properties
        self.in_features = original_layer.in_features
        self.out_features = original_layer.out_features
        self.merged = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass implementing W + BA formulation"""
        # Original forward: Wx
        result = self.original_layer(x)
        
        # LoRA forward: dropout(x) @ A.T @ B.T * scaling
        # This implements W + BA where BA is the low-rank update
        lora_result = (self.dropout(x) @ self.lora_A.T) @ self.lora_B.T * self.scaling
        
        return result + lora_result
    
    def merge_weights(self) -> None:
        """Merge LoRA weights into original layer permanently"""
        # Compute BA: (out_features, r) @ (r, in_features) = (out_features, in_features)
        lora_weight = self.lora_B @ self.lora_A * self.scaling
        
        # Update original weight: W ← W + BA (since linear uses W @ x)
        self.original_layer.weight.data += lora_weight
        
        # Remove LoRA parameters
        del self.lora_A, self.lora_B
        self.lora_A = None
        self.lora_B = None
        self.merged = True
    
    def unmerge_weights(self) -> None:
        """Unmerge LoRA weights (requires storing original weights)"""
        if not hasattr(self, 'merged') or not self.merged:
            raise RuntimeError("LoRA weights are not merged")
        
        # This would require storing the original weight delta
        # For now, raise an error as unmerging needs careful implementation
        raise NotImplementedError("Unmerging requires storing original weight deltas")


class LoRAConfig:
    """Configuration for LoRA adapters"""
    
    def __init__(self, r: int = 8, alpha: int = 16, dropout: float = 0.05, 
                 target_modules: list = None):
        self.r = r
        self.alpha = alpha
        self.dropout = dropout
        # Generic target modules for framework-agnostic compatibility
        self.target_modules = target_modules or [
            "qkv_proj", "out_proj", "ffn_linear1", "ffn_linear2"
        ]
    
    def to_dict(self) -> dict:
        return {
            "r": self.r,
            "alpha": self.alpha,
            "dropout": self.dropout,
            "target_modules": self.target_modules
        }
    
    @classmethod
    def from_dict(cls, config_dict: dict):
        return cls(**config_dict)


def should_apply_lora(name: str, module: nn.Module, target_modules: list) -> bool:
    """Generic module identification for LoRA application"""
    if not isinstance(module, nn.Linear):
        return False
    
    # Generic matching logic for different module types
    if "qkv_proj" in target_modules and name.endswith("qkv_proj"):
        return True
    if "out_proj" in target_modules and name.endswith("out_proj"):
        return True
    if "ffn_linear1" in target_modules and "ff" in name and "net.0" in name:
        return True
    if "ffn_linear2" in target_modules and "ff" in name and "net.2" in name:
        return True
    
    return False


def get_parent_module(model: nn.Module, name: str) -> nn.Module:
    """Get parent module from full module name"""
    parts = name.split('.')
    parent = model
    for part in parts[:-1]:  # All parts except the last one
        if part:
            parent = getattr(parent, part)
    return parent


def apply_lora_to_model(model: nn.Module, config: LoRAConfig) -> nn.Module:
    """Apply LoRA adapters to a model using generic target modules specification"""
    
    # Phase 1: Collect target modules to replace
    targets = []
    for name, module in model.named_modules():
        if should_apply_lora(name, module, config.target_modules):
            # Apply LoRA to this linear layer
            parent = get_parent_module(model, name)
            child_name = name.split('.')[-1]
            targets.append((parent, child_name, module))
    
    # Phase 2: Replace modules safely
    for parent, child_name, module in targets:
        setattr(parent, child_name, LoRALinear(
            module, config.r, config.alpha, config.dropout
        ))
    
    return model


def get_lora_parameters(model: nn.Module) -> list:
    """Get only LoRA trainable parameters"""
    lora_params = []
    
    for module in model.modules():
        if isinstance(module, LoRALinear):
            # Get all lora_ parameters from LoRA modules
            for name, param in module.named_parameters():
                if 'lora_' in name and param.requires_grad:
                    lora_params.append(param)
    
    return lora_params


def count_lora_parameters(model: nn.Module) -> int:
    """Count LoRA parameters"""
    return sum(p.numel() for p in get_lora_parameters(model))


def count_total_parameters(model: nn.Module) -> int:
    """Count total parameters in model"""
    return sum(p.numel() for p in model.parameters())


def save_lora_weights(model: nn.Module, path: str, config: LoRAConfig = None) -> None:
    """Save LoRA weights along with configuration"""
    
    # Extract LoRA weights
    lora_state_dict = {}
    
    for name, param in model.named_parameters():
        if 'lora_' in name and param.requires_grad:
            lora_state_dict[name] = param.data.cpu()
    
    # Save with configuration
    checkpoint = {
        "lora_weights": lora_state_dict,
        "config": config.to_dict() if config else None
    }
    
    torch.save(checkpoint, path)


def load_lora_weights(model: nn.Module, path: str) -> LoRAConfig:
    """Load LoRA weights into model and return configuration"""
    
    checkpoint = torch.load(path, map_location='cpu')
    
    if "lora_weights" not in checkpoint:
        raise ValueError("Invalid LoRA checkpoint format")
    
    lora_state_dict = checkpoint["lora_weights"]
    config_dict = checkpoint.get("config", {})
    config = LoRAConfig.from_dict(config_dict) if config_dict else LoRAConfig()
    
    # Validate that all loaded parameters are LoRA parameters
    for param_name in lora_state_dict.keys():
        if 'lora_' not in param_name:
            raise ValueError(f"Non-LoRA parameter found in LoRA weights: {param_name}")
    
    # Check that model has corresponding LoRA parameters
    model_state_dict = model.state_dict()
    missing_params = []
    
    for param_name in lora_state_dict.keys():
        if param_name not in model_state_dict:
            missing_params.append(param_name)
    
    if missing_params:
        raise ValueError(f"Missing LoRA parameters in model: {missing_params}")
    
    # Load weights
    model.load_state_dict(lora_state_dict, strict=False)
    
    return config


def merge_lora_weights(model: nn.Module) -> nn.Module:
    """Merge all LoRA weights into the base model permanently"""
    
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.merge_weights()
    
    return model


def print_lora_model_info(model: nn.Module) -> None:
    """Print detailed LoRA model information"""
    
    total_params = count_total_parameters(model)
    lora_params = count_lora_parameters(model)
    
    print(f"Total parameters: {total_params/1e6:.2f}M")
    print(f"LoRA trainable parameters: {lora_params/1e6:.2f}M")
    print(f"LoRA parameter ratio: {lora_params/total_params*100:.2f}%")
    
    # Count LoRA modules by type
    lora_modules = {}
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            parent_name = '.'.join(name.split('.')[:-1])
            layer_type = name.split('.')[-2] if '.' in name else name
            lora_modules[layer_type] = lora_modules.get(layer_type, 0) + 1
    
    print("\nLoRA modules by type:")
    for layer_type, count in lora_modules.items():
        print(f"  {layer_type}: {count}")

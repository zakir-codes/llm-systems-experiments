"""Full finetuning utilities for complete model parameter updates"""

import torch
import torch.nn as nn


def prepare_model_for_full_finetuning(model):
    """Prepare model for full finetuning by unfreezing all parameters"""
    
    # Unfreeze all parameters
    for param in model.parameters():
        param.requires_grad = True
    
    return model


def count_trainable_parameters(model):
    """Count trainable parameters in the model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_total_parameters(model):
    """Count total parameters in the model"""
    return sum(p.numel() for p in model.parameters())


def get_trainable_parameters(model):
    """Get all trainable parameters"""
    return [p for p in model.parameters() if p.requires_grad]


def save_full_model(model, path):
    """Save the complete model state dict"""
    torch.save(model.state_dict(), path)


def load_full_model(model, path, device=None):
    """Load complete model state dict"""
    if device is None:
        device = next(model.parameters()).device
    
    state_dict = torch.load(path, map_location=device)
    model.load_state_dict(state_dict)
    
    return model


def print_model_parameter_info(model):
    """Print detailed parameter information"""
    total_params = count_total_parameters(model)
    trainable_params = count_trainable_parameters(model)
    
    print(f"Total parameters: {total_params/1e6:.2f}M")
    print(f"Trainable parameters: {trainable_params/1e6:.2f}M")
    print(f"Trainable ratio: {trainable_params/total_params*100:.2f}%")
    
    # Print parameter counts by layer type
    layer_counts = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            layer_type = name.split('.')[0]
            if layer_type not in layer_counts:
                layer_counts[layer_type] = 0
            layer_counts[layer_type] += param.numel()
    
    print("\nParameter counts by component:")
    for layer_type, count in layer_counts.items():
        print(f"  {layer_type}: {count/1e6:.2f}M")

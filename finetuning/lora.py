"""LoRA (Low-Rank Adaptation) implementation for efficient finetuning"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """LoRA linear layer wrapper"""
    
    def __init__(self, original_layer, r=8, alpha=16, dropout=0.05):
        super().__init__()
        self.original_layer = original_layer
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        
        # LoRA weights
        self.lora_A = nn.Parameter(torch.randn(original_layer.in_features, r) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(r, original_layer.out_features))
        self.dropout = nn.Dropout(dropout)
        
        # Freeze original weights
        for param in self.original_layer.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        # Original forward
        result = self.original_layer(x)
        
        # LoRA forward: x @ A @ B * scaling
        lora_result = x @ self.lora_A @ self.lora_B * self.scaling
        lora_result = self.dropout(lora_result)
        
        return result + lora_result


class LoRAFeedForward(nn.Module):
    """LoRA adapted FeedForward layer"""
    
    def __init__(self, original_ff, r=8, alpha=16, dropout=0.05):
        super().__init__()
        self.original_ff = original_ff
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        
        # Get original layers
        self.linear1 = original_ff.net[0]  # n_embed -> 4*n_embed
        self.linear2 = original_ff.net[2]  # 4*n_embed -> n_embed
        
        # LoRA adapters
        self.lora_A1 = nn.Parameter(torch.randn(self.linear1.in_features, r) * 0.01)
        self.lora_B1 = nn.Parameter(torch.zeros(r, self.linear1.out_features))
        
        self.lora_A2 = nn.Parameter(torch.randn(self.linear2.in_features, r) * 0.01)
        self.lora_B2 = nn.Parameter(torch.zeros(r, self.linear2.out_features))
        
        self.dropout = nn.Dropout(dropout)
        
        # Freeze original weights
        for param in self.original_ff.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        # Original forward
        result = self.original_ff(x)
        
        # LoRA forward for first linear
        lora1 = x @ self.lora_A1 @ self.lora_B1 * self.scaling
        lora1 = F.relu(lora1)
        
        # LoRA forward for second linear
        lora2 = lora1 @ self.lora_A2 @ self.lora_B2 * self.scaling
        lora2 = self.dropout(lora2)
        
        return result + lora2


def apply_lora_to_model(model, r=8, alpha=16, dropout=0.05):
    """Apply LoRA adapters to a model"""
    
    # Replace linear layers in attention
    for block in model.blocks:
        # Apply LoRA to QKV projection in multi-head attention
        multi_head = block.multi_head
        if hasattr(multi_head, 'qkv_proj'):
            multi_head.qkv_proj = LoRALinear(multi_head.qkv_proj, r, alpha, dropout)
        
        # Apply LoRA to output projection in multi-head attention
        if hasattr(multi_head, 'out_proj'):
            multi_head.out_proj = LoRALinear(multi_head.out_proj, r, alpha, dropout)
        
        # Apply LoRA to feedforward
        block.ff = LoRAFeedForward(block.ff, r, alpha, dropout)
    
    # Apply LoRA to final language model head
    model.lm = LoRALinear(model.lm, r, alpha, dropout)
    
    return model


def get_lora_parameters(model):
    """Get only LoRA trainable parameters"""
    lora_params = []
    
    for name, param in model.named_parameters():
        if 'lora_' in name and param.requires_grad:
            lora_params.append(param)
    
    return lora_params


def count_lora_parameters(model):
    """Count LoRA parameters"""
    return sum(p.numel() for p in get_lora_parameters(model))


def count_total_parameters(model):
    """Count total parameters in model"""
    return sum(p.numel() for p in model.parameters())


def save_lora_weights(model, path):
    """Save only LoRA weights"""
    lora_state_dict = {}
    
    for name, param in model.named_parameters():
        if 'lora_' in name and param.requires_grad:
            lora_state_dict[name] = param.data
    
    torch.save(lora_state_dict, path)


def load_lora_weights(model, path):
    """Load LoRA weights into model"""
    lora_state_dict = torch.load(path)
    
    model.load_state_dict(lora_state_dict, strict=False)
    return model
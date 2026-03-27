"""Normal Training Implementation"""

import torch


class NormalTraining:
    """Normal training without AMP or gradient accumulation, supports gradient clipping"""
    
    def __init__(self, model, optimizer, device, grad_clip=1.0):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.grad_clip = grad_clip
    
    def train_step(self, x, target):
        """Perform a single training step with gradient clipping"""
        self.optimizer.zero_grad()
        logits, loss = self.model(x, target=target)
        loss.backward()
        
        # Gradient clipping for stability
        if self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        
        self.optimizer.step()
        return loss.item()

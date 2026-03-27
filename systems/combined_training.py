"""Combined Training Implementation (Mixed Precision + Gradient Accumulation)"""

import torch


class CombinedTraining:
    """Combined mixed precision and gradient accumulation training"""
    
    def __init__(self, model, optimizer, device, accumulation_steps=4, grad_clip=1.0):
        if device != "cuda":
            raise ValueError("Combined training requires CUDA device for mixed precision")
        
        if accumulation_steps < 1:
            raise ValueError("accumulation_steps must be >= 1")
        
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.accumulation_steps = accumulation_steps
        self.grad_clip = grad_clip
        self.step_count = 0
        self.scaler = torch.cuda.amp.GradScaler()
    
    def train_step(self, x, target):
        """Perform a single training step with mixed precision and gradient accumulation"""
        with torch.cuda.amp.autocast():
            logits, loss = self.model(x, target=target)
            # Scale loss for accumulation
            scaled_loss = loss / self.accumulation_steps
        
        self.scaler.scale(scaled_loss).backward()
        self.step_count += 1
        
        if self.step_count % self.accumulation_steps == 0:
            # Gradient clipping
            if self.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
            self.step_count = 0
        
        return loss.item()

""" Mixed Precision Implementation """

import torch

class MixedPrecision:
    """Mixed Precision Training Utility"""
    def __init__(self, model, optimizer, device):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.scaler = torch.cuda.amp.GradScaler()

    def train_step(self, x, target):
        """Perform a single training step with mixed precision"""
        self.optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            output, loss = self.model(x, target)
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        return output, loss
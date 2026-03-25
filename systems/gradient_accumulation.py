"""Gradient Accumulation Implementation"""

class GradientAccumulation:
    """Utility for Gradient Accumulation"""
    def __init__(self, model, optimizer, accumulation_steps):
        self.model = model
        self.optimizer = optimizer
        self.accumulation_steps = accumulation_steps
        self.step_count = 0

    def train_step(self, x, target):
        """Perform a single training step with gradient accumulation"""
        output, loss = self.model(x, target)
        loss = loss / self.accumulation_steps  # Scale loss for accumulation
        loss.backward()
        self.step_count += 1

        if self.step_count % self.accumulation_steps == 0:
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.step_count = 0

        return loss.item()
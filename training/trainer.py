"""
Trainer Implementation

Supports:
- Standard training
- Mixed Precision (AMP)
- Gradient Accumulation
"""

import torch


class Trainer:
    """
    Trainer class responsible for performing training steps.

    Features:
    - AMP training
    - Gradient accumulation
    - Gradient clipping
    """

    def __init__(
        self,
        model,
        optimizer,
        device,
        accumulation_steps=1,
        use_amp=False,
        grad_clip=1.0,
    ):
        self.model = model
        self.optimizer = optimizer
        self.device = device

        self.accumulation_steps = accumulation_steps
        self.use_amp = use_amp
        self.grad_clip = grad_clip

        self.step_count = 0

        # AMP scaler (only needed on CUDA)
        if self.use_amp and device == "cuda":
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None

        # Initialize gradients
        self.optimizer.zero_grad()

    def train_step(self, x, target):
        """
        Perform one training step.

        Returns:
            float: training loss
        """

        self.model.train()

        x = x.to(self.device)
        target = target.to(self.device)

        # ---------------- Forward Pass ----------------

        if self.use_amp and self.scaler is not None:

            with torch.cuda.amp.autocast():

                logits, loss = self.model(x, target=target)

                loss = loss / self.accumulation_steps

            self.scaler.scale(loss).backward()

        else:

            logits, loss = self.model(x, target=target)

            loss = loss / self.accumulation_steps

            loss.backward()

        # ---------------- Gradient Accumulation ----------------

        self.step_count += 1

        if self.step_count % self.accumulation_steps == 0:

            # Gradient clipping (important for stability)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.grad_clip
            )

            if self.use_amp and self.scaler is not None:

                self.scaler.step(self.optimizer)
                self.scaler.update()

            else:

                self.optimizer.step()

            self.optimizer.zero_grad()

        # Return original loss value
        return loss.item() * self.accumulation_steps
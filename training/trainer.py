"""
Trainer Implementation

Simplified trainer that delegates all training logic to a specific training system.
"""

import torch
import logging


class Trainer:
    """
    Simplified trainer class that delegates training steps to a specified system.
    
    Features:
    - Config-driven system selection
    - No fallback logic
    - Clean separation of concerns
    """

    def __init__(
        self,
        model,
        optimizer,
        device,
        training_system,
    ):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.training_system = training_system
        
        logging.info(f"Trainer initialized with system: {type(training_system).__name__}")

    def train_step(self, x, target):
        """
        Perform one training step by delegating to the training system.

        Returns:
            float: training loss
        """
        return self.training_system.train_step(x, target)
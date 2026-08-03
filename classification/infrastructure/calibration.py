import json
import os
from typing import Dict, Any
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from classification.domain.interfaces import IConfidenceCalibrator


class TemperatureScalingCalibrator(IConfidenceCalibrator):
    """Temperature scaling calibrator for neural network logits."""

    def __init__(self, temperature: float = 1.0) -> None:
        """Initializes the calibrator.

        Args:
            temperature: Temperature scaling parameter T. Must be positive.
        """
        self.temperature = max(temperature, 0.01)

    def calibrate(self, logits: np.ndarray) -> np.ndarray:
        """Calibrates raw logits (numpy array) to probabilities.

        Args:
            logits: Uncalibrated logits of shape (N, num_classes) or (num_classes,).

        Returns:
            Calibrated probabilities of the same shape.
        """
        logits_t = torch.from_numpy(logits)
        probs = self.calibrate_tensor(logits_t)
        return probs.numpy()

    def calibrate_tensor(self, logits: torch.Tensor) -> torch.Tensor:
        """Calibrates raw logits (PyTorch Tensor) to probabilities.

        Args:
            logits: Uncalibrated logits of shape (N, num_classes) or (num_classes,).

        Returns:
            Calibrated probabilities of the same shape.
        """
        # Ensure we do not divide by 0 or negative temperature
        temp = max(self.temperature, 0.01)
        scaled_logits = logits / temp
        return torch.softmax(scaled_logits, dim=-1)

    def get_metadata(self) -> Dict[str, Any]:
        """Exposes calibration parameters and metadata.

        Returns:
            A dictionary of calibration information.
        """
        return {
            "method": "Temperature Scaling",
            "temperature": self.temperature,
            "is_calibrated": True
        }

    def save(self, filepath: str) -> None:
        """Saves temperature parameter to a JSON file.

        Args:
            filepath: Destination file path.
        """
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        data = {
            "temperature": self.temperature,
            "method": "Temperature Scaling"
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)

    def load(self, filepath: str) -> None:
        """Loads temperature parameter from a JSON file.

        Args:
            filepath: Source file path.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Calibration configuration not found at: {filepath}")
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
            self.temperature = float(data.get("temperature", 1.0))

    def fit(self, logits: torch.Tensor, labels: torch.Tensor, lr: float = 0.01, max_iter: int = 100) -> float:
        """Optimizes the temperature parameter T on the validation set logits and labels.

        Args:
            logits: PyTorch tensor of shape (N, num_classes) containing uncalibrated logits.
            labels: PyTorch tensor of shape (N,) containing ground truth labels.
            lr: Learning rate for the L-BFGS optimizer.
            max_iter: Maximum number of iterations for optimization.

        Returns:
            The optimized temperature value.
        """
        # Optimize temperature T using L-BFGS
        temperature_param = nn.Parameter(torch.ones(1) * self.temperature)
        optimizer = optim.LBFGS([temperature_param], lr=lr, max_iter=max_iter)
        criterion = nn.CrossEntropyLoss()

        # Place on CPU for optimization safety and consistency
        logits = logits.cpu()
        labels = labels.cpu()

        def eval_loss():
            optimizer.zero_grad()
            # Enforce positive temperature
            temp = torch.clamp(temperature_param, min=0.01)
            loss = criterion(logits / temp, labels)
            loss.backward()
            return loss

        optimizer.step(eval_loss)

        self.temperature = float(torch.clamp(temperature_param, min=0.01).item())
        return self.temperature

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple
import numpy as np


class IMetricsCalculator(ABC):
    """Interface for calculating classification metrics."""

    @abstractmethod
    def calculate(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> Dict[str, float]:
        """Calculates metrics such as accuracy, precision, recall, and F1.

        Args:
            y_true: Ground truth labels.
            y_pred: Predicted labels.

        Returns:
            A dictionary mapping metric names to their values.
        """
        pass


class IModelAdapter(ABC):
    """Interface for interacting with the PyTorch model under Clean Architecture."""

    @abstractmethod
    def train_epoch(self, dataloader: Any) -> float:
        """Trains the model for one epoch.

        Args:
            dataloader: Training dataloader (e.g., PyTorch DataLoader).

        Returns:
            The average training loss for the epoch.
        """
        pass

    @abstractmethod
    def evaluate(self, dataloader: Any) -> Tuple[float, Dict[str, float]]:
        """Evaluates the model on a dataset.

        Args:
            dataloader: Validation/testing dataloader.

        Returns:
            A tuple containing:
                - The average validation loss.
                - A dictionary of performance metrics.
        """
        pass

    @abstractmethod
    def predict(self, image_tensor: Any) -> Tuple[int, float, Dict[str, float]]:
        """Infers the class for a single input tensor.

        Args:
            image_tensor: The preprocessed image tensor (e.g., PyTorch Tensor).

        Returns:
            A tuple containing:
                - The predicted class label index.
                - The confidence score.
                - A dictionary mapping class names to probabilities.
        """
        pass

    @abstractmethod
    def save(self, filepath: str) -> None:
        """Saves the model weights to the specified filepath.

        Args:
            filepath: The path to save the weights/checkpoint.
        """
        pass

    @abstractmethod
    def load(self, filepath: str) -> None:
        """Loads model weights from the specified filepath.

        Args:
            filepath: The path to the model weights file.
        """
        pass

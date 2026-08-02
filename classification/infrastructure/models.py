import os
from typing import Dict, Any, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.models as models
from classification.domain.entities import BrainTumorClass
from classification.domain.interfaces import IModelAdapter, IMetricsCalculator


class EfficientNetB0Model(nn.Module):
    """EfficientNet-B0 model wrapper for 4-class brain tumor classification."""

    def __init__(self, pretrained: bool = True, num_classes: int = 4) -> None:
        """Initializes the model.

        Args:
            pretrained: Whether to load ImageNet pre-trained weights.
            num_classes: Number of target categories.
        """
        super().__init__()
        
        # Load EfficientNet-B0 with pre-trained weights if requested
        if pretrained:
            try:
                # Modern torchvision API (>=0.13)
                weights = models.EfficientNet_B0_Weights.DEFAULT
                self.backbone = models.efficientnet_b0(weights=weights)
            except AttributeError:
                # Older torchvision API
                self.backbone = models.efficientnet_b0(pretrained=True)
        else:
            try:
                self.backbone = models.efficientnet_b0(weights=None)
            except TypeError:
                self.backbone = models.efficientnet_b0(pretrained=False)

        # Modify the classifier head for 4 classes
        # EfficientNet-B0 has model.classifier containing Dropout and Linear layers
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier[1] = nn.Linear(
            in_features=in_features, out_features=num_classes
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.

        Args:
            x: Input image tensor of shape (B, C, H, W).

        Returns:
            Raw logits tensor of shape (B, num_classes).
        """
        return self.backbone(x)


class PyTorchModelAdapter(IModelAdapter):
    """Adapter wrapping a PyTorch model and executing operations under the IModelAdapter interface."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optional[optim.Optimizer] = None,
        criterion: Optional[nn.Module] = None,
        metrics_calculator: Optional[IMetricsCalculator] = None,
        device: str = "cpu",
    ) -> None:
        """Initializes the adapter.

        Args:
            model: The PyTorch neural network model.
            optimizer: Optional PyTorch optimizer (required for training).
            criterion: Optional PyTorch loss function (required for training/evaluation).
            metrics_calculator: Optional calculator for computing evaluation metrics.
            device: Device to run computations on ('cpu' or 'cuda').
        """
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.metrics_calculator = metrics_calculator
        self.device = torch.device(device)

    def train_epoch(self, dataloader: DataLoader) -> float:
        if self.optimizer is None or self.criterion is None:
            raise ValueError(
                "Optimizer and criterion must be provided to run training."
            )

        self.model.train()
        total_loss = 0.0

        for inputs, targets in dataloader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * inputs.size(0)

        epoch_loss = total_loss / len(dataloader.dataset)
        return epoch_loss

    def evaluate(self, dataloader: DataLoader) -> Tuple[float, Dict[str, float]]:
        if self.criterion is None:
            raise ValueError("Criterion must be provided to run evaluation.")

        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_targets = []

        with torch.inference_mode():
            for inputs, targets in dataloader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                total_loss += loss.item() * inputs.size(0)

                # Get predictions
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

        avg_loss = total_loss / len(dataloader.dataset)

        metrics = {}
        if self.metrics_calculator is not None:
            metrics = self.metrics_calculator.calculate(
                np.array(all_targets), np.array(all_preds)
            )

        return avg_loss, metrics

    def predict(
        self, image_tensor: torch.Tensor
    ) -> Tuple[int, float, Dict[str, float]]:
        self.model.eval()

        # Ensure image has a batch dimension
        if len(image_tensor.shape) == 3:
            image_tensor = image_tensor.unsqueeze(0)

        image_tensor = image_tensor.to(self.device)

        device_type = self.device.type
        is_autocast_supported = device_type in ["cuda", "cpu"]

        with torch.inference_mode():
            if is_autocast_supported:
                dtype = torch.float16 if device_type == "cuda" else torch.bfloat16
                with torch.amp.autocast(device_type=device_type, dtype=dtype):
                    outputs = self.model(image_tensor)
            else:
                outputs = self.model(image_tensor)
            
            # Apply softmax to get probabilities
            probs = torch.softmax(outputs, dim=1).squeeze(0).cpu().numpy()

        predicted_idx = int(np.argmax(probs))
        confidence = float(probs[predicted_idx])

        # Map class probabilities dictionary
        probs_dict = {}
        for cls in BrainTumorClass:
            class_name = BrainTumorClass.get_name_by_value(cls.value)
            probs_dict[class_name] = float(probs[cls.value])

        return predicted_idx, confidence, probs_dict

    def save(self, filepath: str) -> None:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        torch.save(self.model.state_dict(), filepath)

    def load(self, filepath: str) -> None:
        if not os.path.exists(filepath):
            raise FileNotFoundError(
                f"Model checkpoint file not found at: {filepath}"
            )
        self.model.load_state_dict(
            torch.load(filepath, map_location=self.device)
        )

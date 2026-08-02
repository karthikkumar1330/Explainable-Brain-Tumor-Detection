from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class ClassificationConfig:
    """Hyperparameter and path configuration for the brain tumor classification pipeline."""

    # Dataset Paths
    train_dir: str = "datasets/classification/train"
    val_dir: str = "datasets/classification/valid"
    test_dir: str = "datasets/classification/test"

    # Preprocessing Configurations
    input_size: Tuple[int, int] = (224, 224)  # Standard size for EfficientNet-B0
    clahe: bool = True
    zscore: bool = True

    # Training Hyperparameters
    epochs: int = 15
    batch_size: int = 16
    lr: float = 1e-4
    weight_decay: float = 1e-4
    optimizer_name: str = "Adam"  # Support 'Adam', 'SGD'
    device: str = "cuda"  # Will fallback to 'cpu' dynamically if not available

    # Checkpoint Configuration
    checkpoint_dir: str = "models/classification"
    checkpoint_name: str = "efficientnet_b0_brain_tumor.pth"

    # Logging Configurations
    log_dir: str = "logs"
    log_filename: str = "classification.log"

    # Explainability Configurations
    explainability_dir: str = "outputs/explainability"

    def get_checkpoint_path(self) -> str:
        """Returns the full path to save/load checkpoints."""
        import os
        return os.path.join(self.checkpoint_dir, self.checkpoint_name)

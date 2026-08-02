import argparse
import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import albumentations as A

# Adjust python path if executed from root directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from classification.config import ClassificationConfig
from classification.infrastructure.logging import get_logger
from classification.infrastructure.dataset import BrainTumorClassificationDataset
from classification.infrastructure.models import EfficientNetB0Model, PyTorchModelAdapter
from classification.infrastructure.metrics import PyTorchMetricsCalculator
from classification.application.use_cases import EvaluateModelUseCase


def parse_args() -> argparse.Namespace:
    """Parses command line arguments for validation."""
    parser = argparse.ArgumentParser(
        description="Brain Tumor Classification Evaluation Script"
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=None,
        help="Path to evaluation dataset folder (e.g. datasets/classification/test)",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default=None,
        help="Path to the trained model checkpoint (.pth)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=None, help="Batch size for evaluation"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run evaluation on (cuda or cpu)",
    )
    return parser.parse_args()


def main() -> None:
    # 1. Initialize configuration and overrides
    config = ClassificationConfig()
    args = parse_args()

    # Default to test dataset if available, fallback to valid
    eval_dir = args.dataset_dir or config.test_dir
    if not os.path.exists(eval_dir):
        eval_dir = config.val_dir

    if args.batch_size:
        config.batch_size = args.batch_size
    if args.device:
        config.device = args.device

    checkpoint_path = args.checkpoint_path or config.get_checkpoint_path()

    # 2. Setup Logging
    logger = get_logger(
        name="brain_tumor_classification_val",
        log_dir=config.log_dir,
        log_filename=config.log_filename,
    )

    logger.info("Initializing Brain Tumor Classification Evaluation Script")
    logger.info(f"  Evaluation dataset directory: {eval_dir}")
    logger.info(f"  Checkpoint path: {checkpoint_path}")

    # Set device configuration dynamically
    device = config.device
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available. Falling back to CPU.")
        device = "cpu"
    logger.info(f"  Target Device: {device}")

    # 3. Setup transforms
    val_transform = A.Compose(
        [
            A.Resize(config.input_size[0], config.input_size[1]),
        ]
    )

    # 4. Load dataset & Dataloader
    try:
        dataset = BrainTumorClassificationDataset(
            base_dir=eval_dir,
            transform=val_transform,
            clahe=config.clahe,
            zscore=config.zscore,
        )
    except Exception as e:
        logger.error(f"Failed to load evaluation dataset: {e}")
        sys.exit(1)

    logger.info(f"Loaded {len(dataset)} evaluation samples.")
    if len(dataset) == 0:
        logger.error("Evaluation dataset is empty. Aborting evaluation.")
        sys.exit(1)

    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device == "cuda"),
    )

    # 5. Initialize model and wrapper adapter
    model = EfficientNetB0Model(pretrained=False, num_classes=4)
    criterion = nn.CrossEntropyLoss()
    metrics_calculator = PyTorchMetricsCalculator()

    model_adapter = PyTorchModelAdapter(
        model=model,
        criterion=criterion,
        metrics_calculator=metrics_calculator,
        device=device,
    )

    # Load weights
    try:
        logger.info(f"Loading checkpoint weights from {checkpoint_path}...")
        model_adapter.load(checkpoint_path)
    except Exception as e:
        logger.error(f"Failed to load model weights: {e}")
        sys.exit(1)

    # 6. Execute Use Case
    eval_use_case = EvaluateModelUseCase(
        model_adapter=model_adapter, logger=logger
    )
    val_loss, metrics = eval_use_case.execute(loader)

    logger.info("Evaluation runs completed successfully.")


if __name__ == "__main__":
    main()

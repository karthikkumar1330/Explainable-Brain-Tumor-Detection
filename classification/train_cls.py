import argparse
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import albumentations as A

# Adjust python path if executed from root directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from classification.config import ClassificationConfig
from classification.infrastructure.logging import get_logger
from classification.infrastructure.dataset import BrainTumorClassificationDataset
from classification.infrastructure.models import EfficientNetB0Model, PyTorchModelAdapter
from classification.infrastructure.metrics import PyTorchMetricsCalculator
from classification.application.use_cases import TrainModelUseCase


def parse_args() -> argparse.Namespace:
    """Parses command line arguments overriding config defaults."""
    parser = argparse.ArgumentParser(
        description="Brain Tumor Classification Training Pipeline"
    )
    parser.add_argument(
        "--train-dir",
        type=str,
        default=None,
        help="Path to training dataset folder",
    )
    parser.add_argument(
        "--val-dir",
        type=str,
        default=None,
        help="Path to validation dataset folder",
    )
    parser.add_argument(
        "--epochs", type=int, default=None, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size", type=int, default=None, help="Batch size for training"
    )
    parser.add_argument(
        "--lr", type=float, default=None, help="Learning rate"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run training on (cuda or cpu)",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default=None,
        help="Full path to save best checkpoint (.pth)",
    )
    return parser.parse_args()


def main() -> None:
    # 1. Initialize Default Configurations
    config = ClassificationConfig()

    # 2. Parse overrides
    args = parse_args()
    if args.train_dir:
        config.train_dir = args.train_dir
    if args.val_dir:
        config.val_dir = args.val_dir
    if args.epochs:
        config.epochs = args.epochs
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.lr:
        config.lr = args.lr
    if args.device:
        config.device = args.device
    if args.checkpoint_path:
        config.checkpoint_dir = os.path.dirname(args.checkpoint_path)
        config.checkpoint_name = os.path.basename(args.checkpoint_path)

    # 3. Setup Logging
    logger = get_logger(
        name="brain_tumor_classification_train",
        log_dir=config.log_dir,
        log_filename=config.log_filename,
    )

    logger.info("Initializing Brain Tumor Classification Pipeline Configuration")
    logger.info(f"  Train directory: {config.train_dir}")
    logger.info(f"  Val directory: {config.val_dir}")
    logger.info(f"  Batch size: {config.batch_size}")
    logger.info(f"  Epochs: {config.epochs}")
    logger.info(f"  Initial Learning rate: {config.lr}")

    # Set device configuration dynamically
    device = config.device
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available. Falling back to CPU.")
        device = "cpu"
    logger.info(f"  Target Device: {device}")

    # 4. Setup Albumentations Transforms
    train_transform = A.Compose(
        [
            A.Resize(config.input_size[0], config.input_size[1]),
            A.HorizontalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5
            ),
        ]
    )
    val_transform = A.Compose(
        [
            A.Resize(config.input_size[0], config.input_size[1]),
        ]
    )

    # 5. Initialize Datasets & DataLoaders
    try:
        train_dataset = BrainTumorClassificationDataset(
            base_dir=config.train_dir,
            transform=train_transform,
            clahe=config.clahe,
            zscore=config.zscore,
        )
        val_dataset = BrainTumorClassificationDataset(
            base_dir=config.val_dir,
            transform=val_transform,
            clahe=config.clahe,
            zscore=config.zscore,
        )
    except Exception as e:
        logger.error(f"Failed to load datasets: {e}")
        sys.exit(1)

    logger.info(f"Loaded {len(train_dataset)} training samples.")
    logger.info(f"Loaded {len(val_dataset)} validation samples.")

    if len(train_dataset) == 0 or len(val_dataset) == 0:
        logger.error("Dataset splits are empty. Aborting training.")
        sys.exit(1)

    # We set num_workers=0 to avoid multiprocessing issues in some Windows envs
    # during quick debugging, but allow customization or standard usage.
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device == "cuda"),
    )

    # 6. Initialize Model, Criterion, Optimizer
    logger.info("Instantiating EfficientNet-B0 Model Wrapper")
    model = EfficientNetB0Model(pretrained=True, num_classes=4)
    criterion = nn.CrossEntropyLoss()

    if config.optimizer_name == "Adam":
        optimizer = optim.Adam(
            model.parameters(), lr=config.lr, weight_decay=config.weight_decay
        )
    elif config.optimizer_name == "SGD":
        optimizer = optim.SGD(
            model.parameters(),
            lr=config.lr,
            momentum=0.9,
            weight_decay=config.weight_decay,
        )
    else:
        logger.error(f"Unsupported optimizer name: {config.optimizer_name}")
        sys.exit(1)

    metrics_calculator = PyTorchMetricsCalculator()

    # 7. Create Adapter and execute Use Case
    model_adapter = PyTorchModelAdapter(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        metrics_calculator=metrics_calculator,
        device=device,
    )

    train_use_case = TrainModelUseCase(
        model_adapter=model_adapter, logger=logger
    )

    checkpoint_filepath = config.get_checkpoint_path()
    logger.info(f"Target checkpoint path: {checkpoint_filepath}")

    history = train_use_case.execute(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config.epochs,
        checkpoint_path=checkpoint_filepath,
    )

    logger.info("Training pipeline runs completed successfully.")


if __name__ == "__main__":
    main()

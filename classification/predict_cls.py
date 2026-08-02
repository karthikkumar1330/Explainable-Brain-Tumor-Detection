import argparse
import os
import sys
import cv2
import numpy as np
import torch
import albumentations as A

# Adjust python path if executed from root directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from classification.config import ClassificationConfig
from classification.infrastructure.models import EfficientNetB0Model, PyTorchModelAdapter
from classification.application.use_cases import PredictUseCase, ExplainPredictionUseCase
from classification.domain.entities import PredictionResult
from classification.infrastructure.explainability import GradCAMService
from classification.infrastructure.visualization import save_explainability_outputs
from classification.infrastructure.logging import get_logger


def parse_args() -> argparse.Namespace:
    """Parses command line arguments for single-image prediction."""
    parser = argparse.ArgumentParser(
        description="Brain Tumor Classification Inference Script"
    )
    parser.add_argument(
        "--image-path",
        type=str,
        required=True,
        help="Path to the input brain MRI scan image file",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default=None,
        help="Path to the trained model checkpoint (.pth)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to run inference on (cuda or cpu)",
    )
    parser.add_argument(
        "--explain",
        action="store_true",
        help="Generate and save Grad-CAM explainability heatmaps",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Custom output directory to save explainability heatmaps (defaults to config explainability_dir)",
    )
    return parser.parse_args()


def preprocess_image(
    image_path: str, config: ClassificationConfig
) -> torch.Tensor:
    """Loads and preprocesses an image for classification.

    Follows the exact same preprocessing pipeline (CLAHE + Normalization)
    as the training dataset loader.

    Args:
        image_path: Path to the target image file.
        config: The configuration object containing size and normalizations.

    Returns:
        A torch.Tensor of shape (C, H, W) ready for inference.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise IOError(f"Could not load image at path: {image_path}")

    # Apply CLAHE contrast enhancement if enabled
    if config.clahe:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe_obj = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe_obj.apply(gray)
        img = cv2.merge([cl, cl, cl])

    # Resize to configuration size
    img = cv2.resize(img, (config.input_size[1], config.input_size[0]))

    # Preprocessing & Normalization
    img = img.astype(np.float32)
    if config.zscore:
        mean = img.mean()
        std = img.std()
        img = (img - mean) / (std + 1e-8)
    else:
        img = img / 255.0

    # Transpose to channel-first (C, H, W)
    img = img.transpose(2, 0, 1)

    # Convert to torch tensor
    return torch.from_numpy(img)


def main() -> None:
    # 1. Load Configurations and overrides
    config = ClassificationConfig()
    args = parse_args()

    checkpoint_path = args.checkpoint_path or config.get_checkpoint_path()
    device = args.device

    if device == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA is not available. Defaulting to CPU inference.")
        device = "cpu"

    # 2. Verify files exist
    if not os.path.exists(args.image_path):
        print(f"Error: Target image file not found at: {args.image_path}")
        sys.exit(1)

    if not os.path.exists(checkpoint_path):
        print(f"Error: Model checkpoint file not found at: {checkpoint_path}")
        print("Please train a model first or specify a valid --checkpoint-path.")
        sys.exit(1)

    # 3. Preprocess Image
    try:
        image_tensor = preprocess_image(args.image_path, config)
    except Exception as e:
        print(f"Error during image loading and preprocessing: {e}")
        sys.exit(1)

    # 4. Instantiate Model and Adapter
    model = EfficientNetB0Model(pretrained=False, num_classes=4)
    model_adapter = PyTorchModelAdapter(model=model, device=device)

    # Load weights
    try:
        model_adapter.load(checkpoint_path)
    except Exception as e:
        print(f"Error: Failed to load model weights from checkpoint: {e}")
        sys.exit(1)

    # 5. Run Prediction Use Case
    predict_use_case = PredictUseCase(model_adapter=model_adapter)
    result: PredictionResult = predict_use_case.execute(image_tensor)

    # 6. Display Output
    print("\n" + "=" * 50)
    print("BRAIN TUMOR CLASSIFICATION RESULT")
    print("=" * 50)
    print(f"Predicted Class    : {result.class_name}")
    print(f"Confidence Score   : {result.confidence_score:.4%}")
    print("-" * 50)
    print("Class Probabilities:")
    for cls_name, prob in result.probabilities.items():
        print(f"  - {cls_name:<16}: {prob:.4%}")
    print("=" * 50 + "\n")

    # 7. Generate Explainability
    if args.explain:
        logger = get_logger(
            name="predict_cls",
            log_dir=config.log_dir,
            log_filename=config.log_filename,
        )
        logger.info("Starting explainability pipeline...")
        try:
            # The target layer is the last convolutional block in the EfficientNet-B0 backbone.
            # EfficientNetB0Model wraps efficientnet_b0 where self.backbone.features is a Sequential.
            # features[8] is the last block containing the final Conv2d, BatchNorm2d, and SiLU activation.
            target_layer = model.backbone.features[8]

            explain_service = GradCAMService(
                model=model,
                target_layer=target_layer,
                device=torch.device(device),
            )

            explain_use_case = ExplainPredictionUseCase(
                predict_use_case=predict_use_case,
                explain_service=explain_service,
                logger=logger,
            )

            # Generate heatmap for the predicted class
            _, heatmap = explain_use_case.execute(image_tensor, target_class=result.label)

            # Load the original image (un-normalized) for visual overlay
            original_image = cv2.imread(args.image_path)
            if original_image is None:
                raise IOError(
                    f"Could not load original image for overlay at: {args.image_path}"
                )

            # Output paths setup
            output_dir = args.output_dir or config.explainability_dir
            base_filename = os.path.splitext(os.path.basename(args.image_path))[0]

            save_explainability_outputs(
                original_image=original_image,
                heatmap=heatmap,
                output_dir=output_dir,
                base_filename=base_filename,
                alpha=0.6,
                logger=logger,
            )
            print(f"Explainability outputs successfully saved to: {output_dir}")

        except Exception as e:
            logger.error(f"Error generating explainability heatmap: {e}")
            print(f"Error: Failed to generate explainability outputs: {e}")


if __name__ == "__main__":
    main()

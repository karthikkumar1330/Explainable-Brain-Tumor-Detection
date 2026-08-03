import os
import sys
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import albumentations as A
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(os.getcwd())

from classification.config import ClassificationConfig
from classification.infrastructure.dataset import BrainTumorClassificationDataset
from classification.infrastructure.models import EfficientNetB0Model, PyTorchModelAdapter
from classification.infrastructure.calibration import TemperatureScalingCalibrator


def compute_ece(probs: np.ndarray, labels: np.ndarray, num_bins: int = 10) -> float:
    """Computes the Expected Calibration Error (ECE) for classification."""
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels)

    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    ece = 0.0

    for i in range(num_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += prop_in_bin * np.abs(avg_confidence_in_bin - accuracy_in_bin)

    return ece


def plot_reliability_diagram(probs_before, probs_after, labels, save_path):
    """Plots and saves a reliability diagram comparing pre- and post-calibration."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    num_bins = 10
    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    bin_centers = 0.5 * (bin_boundaries[:-1] + bin_boundaries[1:])

    for ax, probs, title in zip(axes, [probs_before, probs_after], ["Before Calibration", "After Calibration"]):
        confidences = np.max(probs, axis=1)
        predictions = np.argmax(probs, axis=1)
        accuracies = (predictions == labels)

        bin_accuracies = []
        bin_confidences = []
        bin_counts = []

        for i in range(num_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)

            if np.sum(in_bin) > 0:
                bin_accuracies.append(np.mean(accuracies[in_bin]))
                bin_confidences.append(np.mean(confidences[in_bin]))
                bin_counts.append(np.sum(in_bin))
            else:
                bin_accuracies.append(0.0)
                bin_confidences.append(bin_centers[i])
                bin_counts.append(0)

        # Perfect calibration line
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect Calibration")
        # Accuracy bars
        ax.bar(bin_centers, bin_accuracies, width=1.0/num_bins, edgecolor="black", color="royalblue", alpha=0.7, label="Outputs")
        # Confidence line
        ax.plot(bin_centers, bin_confidences, color="darkorange", marker="o", label="Avg. Confidence")

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Confidence")
        ax.set_ylabel("Accuracy")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Calibrate Classifier Confidence Scores")
    parser.add_argument("--dataset-dir", type=str, default=None, help="Path to validation dataset")
    parser.add_argument("--checkpoint-path", type=str, default=None, help="Path to classification checkpoint")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run on (cpu or cuda)")
    args = parser.parse_args()

    config = ClassificationConfig()
    device = torch.device(args.device)

    # Resolve paths
    dataset_dir = args.dataset_dir or config.val_dir
    checkpoint_path = args.checkpoint_path or "models/classification/best_v2.pt"
    if not os.path.exists(checkpoint_path):
        checkpoint_path = config.get_checkpoint_path()

    print(f"=== Starting Confidence Calibration pipeline ===")
    print(f"Dataset directory : {dataset_dir}")
    print(f"Model Checkpoint  : {checkpoint_path}")
    print(f"Target Device     : {device}")

    # Load dataset
    val_transform = A.Compose([A.Resize(config.input_size[0], config.input_size[1])])
    dataset = BrainTumorClassificationDataset(
        base_dir=dataset_dir,
        transform=val_transform,
        clahe=config.clahe,
        zscore=config.zscore,
    )
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False)
    print(f"Loaded {len(dataset)} validation samples.")

    # Load Model
    model = EfficientNetB0Model(pretrained=False, num_classes=4)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model = model.to(device)
    model.eval()

    # Collect uncalibrated logits and true labels
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            logits = model(inputs)
            all_logits.append(logits.cpu())
            all_labels.append(targets)

    logits_tensor = torch.cat(all_logits, dim=0)
    labels_tensor = torch.cat(all_labels, dim=0)

    # Compute uncalibrated probabilities
    uncal_probs = torch.softmax(logits_tensor, dim=1).numpy()
    labels_np = labels_tensor.numpy()

    # ECE before calibration
    ece_before = compute_ece(uncal_probs, labels_np)
    print(f"\nExpected Calibration Error (ECE) before Calibration: {ece_before:.4%}")

    # Initialize and Fit Calibrator
    calibrator = TemperatureScalingCalibrator()
    print("Fitting TemperatureScalingCalibrator...")
    optimized_temp = calibrator.fit(logits_tensor, labels_tensor)
    print(f"Optimized Temperature scaling factor (T): {optimized_temp:.4f}")

    # Compute calibrated probabilities
    cal_probs = calibrator.calibrate(logits_tensor.numpy())

    # ECE after calibration
    ece_after = compute_ece(cal_probs, labels_np)
    print(f"Expected Calibration Error (ECE) after Calibration: {ece_after:.4%}")
    improvement = ece_before - ece_after
    print(f"ECE absolute improvement: {improvement:.4%}")

    # Save calibration parameters next to checkpoint
    base_no_ext, _ = os.path.splitext(checkpoint_path)
    cal_save_path = f"{base_no_ext}_calibration.json"
    calibrator.save(cal_save_path)
    print(f"Saved calibration parameters to: {cal_save_path}")

    # Plot reliability diagrams
    plot_path = "outputs/explainability/calibration_curve.png"
    plot_reliability_diagram(uncal_probs, cal_probs, labels_np, plot_path)
    print(f"Saved reliability diagrams plot to: {plot_path}")
    print("=== Calibration Complete ===")


if __name__ == "__main__":
    main()

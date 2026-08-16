import argparse
import os
import sys
import time
import json
import csv
import hashlib
import numpy as np
import cv2
import torch
import torch.nn as nn
from glob import glob
from typing import Dict, Any, Tuple, List, Optional
import albumentations as A

# Reconfigure stdout for UTF-8 compatibility in Windows terminals
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Ensure core imports work by adding root folder to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from classification.infrastructure.models import EfficientNetB0Model, PyTorchModelAdapter
from classification.infrastructure.dataset import BrainTumorClassificationDataset
from dataset import Dataset as SegDataset
import archs

# Scikit-learn metric dependencies
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_curve,
    auc,
    roc_auc_score
)

def set_seeds(seed: int = 42) -> None:
    """Enforces deterministic evaluations for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_file_sha256(filepath: str) -> str:
    """Computes SHA-256 checksum of model weights file."""
    if not os.path.exists(filepath):
        return "N/A"
    sha256 = hashlib.sha256()
    try:
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                sha256.update(chunk)
        return sha256.hexdigest()
    except Exception:
        return "ERROR"

def sanitize_path(path: str) -> str:
    """Simplifies directory traversal checks."""
    return os.path.abspath(path)

# =====================================================================
# Metric Calculations
# =====================================================================

def calculate_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_probs: np.ndarray) -> Dict[str, Any]:
    """Calculates Accuracy, Precision, Recall, Specificity, F1, OVR ROC-AUC, and Confusion Matrix."""
    num_classes = 4
    class_names = ["Glioma", "Meningioma", "Pituitary", "No Tumor"]

    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))

    per_class = {}
    for c in range(num_classes):
        tp = cm[c, c]
        fn = np.sum(cm[c, :]) - tp
        fp = np.sum(cm[:, c]) - tp
        tn = np.sum(cm) - tp - fp - fn

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0

        per_class[class_names[c]] = {
            "sensitivity": float(sensitivity),
            "specificity": float(specificity),
            "precision": float(precision),
            "f1_score": float(f1),
            "support": int(tp + fn)
        }

    accuracy = accuracy_score(y_true, y_pred)
    macro_sensitivity = np.mean([per_class[name]["sensitivity"] for name in class_names])
    macro_specificity = np.mean([per_class[name]["specificity"] for name in class_names])
    macro_precision = np.mean([per_class[name]["precision"] for name in class_names])
    macro_f1 = np.mean([per_class[name]["f1_score"] for name in class_names])

    total_support = len(y_true)
    weighted_sensitivity = sum(per_class[name]["sensitivity"] * per_class[name]["support"] for name in class_names) / (total_support + 1e-8)
    weighted_specificity = sum(per_class[name]["specificity"] * per_class[name]["support"] for name in class_names) / (total_support + 1e-8)
    weighted_precision = sum(per_class[name]["precision"] * per_class[name]["support"] for name in class_names) / (total_support + 1e-8)
    weighted_f1 = sum(per_class[name]["f1_score"] * per_class[name]["support"] for name in class_names) / (total_support + 1e-8)

    try:
        if len(np.unique(y_true)) > 1:
            roc_auc_macro = roc_auc_score(y_true, y_probs, multi_class='ovr', average='macro')
            roc_auc_weighted = roc_auc_score(y_true, y_probs, multi_class='ovr', average='weighted')
        else:
            roc_auc_macro = 0.0
            roc_auc_weighted = 0.0
    except Exception:
        roc_auc_macro = 0.0
        roc_auc_weighted = 0.0

    return {
        "accuracy": float(accuracy),
        "macro_sensitivity": float(macro_sensitivity),
        "macro_specificity": float(macro_specificity),
        "macro_precision": float(macro_precision),
        "macro_f1": float(macro_f1),
        "weighted_sensitivity": float(weighted_sensitivity),
        "weighted_specificity": float(weighted_specificity),
        "weighted_precision": float(weighted_precision),
        "weighted_f1": float(weighted_f1),
        "roc_auc_macro": float(roc_auc_macro),
        "roc_auc_weighted": float(roc_auc_weighted),
        "per_class": per_class,
        "confusion_matrix": cm.tolist()
    }

def save_plots(y_true: np.ndarray, y_probs: np.ndarray, cm: list, output_dir: str) -> Tuple[str, str]:
    """Generates and saves the ROC curve and Confusion Matrix figures."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)

    # 1. Confusion Matrix Plot
    cm_path = os.path.join(output_dir, "external_confusion_matrix.png")
    class_names = ["Glioma", "Meningioma", "Pituitary", "No Tumor"]
    cm_arr = np.array(cm)

    plt.figure(figsize=(6, 5))
    plt.imshow(cm_arr, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("External Dataset Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    thresh = cm_arr.max() / 2.
    for i in range(cm_arr.shape[0]):
        for j in range(cm_arr.shape[1]):
            plt.text(j, i, format(cm_arr[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm_arr[i, j] > thresh else "black")

    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(cm_path, dpi=150)
    plt.close()

    # 2. ROC Curves Plot
    roc_path = os.path.join(output_dir, "external_roc_curves.png")
    plt.figure(figsize=(8, 6))
    for c in range(4):
        y_true_c = (y_true == c).astype(int)
        y_prob_c = y_probs[:, c]

        if len(np.unique(y_true_c)) == 2:
            fpr, tpr, _ = roc_curve(y_true_c, y_prob_c)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f"{class_names[c]} (AUC = {roc_auc:.4f})")
        else:
            plt.plot([0, 1], [0, 1], linestyle='--', label=f"{class_names[c]} (No Support)")

    plt.plot([0, 1], [0, 1], 'k--', label="Random (AUC = 0.50)")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("One-vs-Rest Multiclass ROC Curves")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(roc_path, dpi=150)
    plt.close()

    return cm_path, roc_path

# =====================================================================
# Benchmarking Execution
# =====================================================================

def evaluate_classification(
    class_dir: str,
    checkpoint_path: str,
    device: str,
    output_dir: str
) -> Optional[Dict[str, Any]]:
    """Runs evaluation over the external classification dataset directory."""
    if not os.path.exists(class_dir):
        print(f"Warning: Classification folder '{class_dir}' does not exist. Skipping classification evaluation.")
        return None

    print(f"Evaluating classification model from checkpoint: {checkpoint_path}")

    val_transform = A.Compose([A.Resize(224, 224)])

    try:
        dataset = BrainTumorClassificationDataset(
            base_dir=class_dir,
            transform=val_transform,
            clahe=True,
            zscore=True
        )
    except Exception as e:
        print(f"Error loading classification dataset: {e}")
        return None

    if len(dataset) == 0:
        print("Warning: Classification dataset is empty. Skipping.")
        return None

    loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False)

    model = EfficientNetB0Model(pretrained=False, num_classes=4)
    model_adapter = PyTorchModelAdapter(model=model, device=device)

    try:
        model_adapter.load(checkpoint_path)
    except Exception as e:
        print(f"Error loading model weights: {e}")
        return None

    model_adapter.model.eval()

    all_preds = []
    all_targets = []
    all_probs = []
    sample_records = []

    class_names = ["Glioma", "Meningioma", "Pituitary", "No Tumor"]

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            inputs = inputs.to(device)
            # Forward pass
            outputs = model_adapter.model(inputs)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)
            targets_np = targets.numpy()

            all_preds.extend(preds)
            all_targets.extend(targets_np)
            all_probs.extend(probs)

            # Map samples
            for i in range(len(preds)):
                idx = batch_idx * 4 + i
                file_path = dataset.samples[idx][0]
                sample_records.append({
                    "sample_path": file_path,
                    "true_label": int(targets_np[i]),
                    "true_class": class_names[int(targets_np[i])],
                    "predicted_label": int(preds[i]),
                    "predicted_class": class_names[int(preds[i])],
                    "confidence": float(probs[i][preds[i]]),
                    "correct": int(preds[i] == targets_np[i]),
                    "probs": [float(p) for p in probs[i]]
                })

    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)
    y_probs = np.array(all_probs)

    metrics = calculate_classification_metrics(y_true, y_pred, y_probs)
    metrics["samples"] = sample_records

    # Save plots
    cm_plot, roc_plot = save_plots(y_true, y_probs, metrics["confusion_matrix"], output_dir)
    metrics["confusion_matrix_plot"] = cm_plot
    metrics["roc_plot"] = roc_plot
    metrics["total_samples"] = len(y_true)

    return metrics

def evaluate_segmentation(
    images_dir: str,
    masks_dir: str,
    checkpoint_path: str,
    config_path: str,
    device: str,
    custom_threshold: Optional[float] = None
) -> Optional[Dict[str, Any]]:
    """Runs evaluation over the external segmentation dataset directories."""
    if not os.path.exists(images_dir) or not os.path.exists(masks_dir):
        print(f"Warning: Image or mask folder does not exist. Skipping segmentation evaluation.")
        return None

    print(f"Evaluating segmentation model from checkpoint: {checkpoint_path}")

    # Read config
    if not os.path.exists(config_path):
        print(f"Error: Config path '{config_path}' does not exist.")
        return None

    with open(config_path, "r") as f:
        import yaml
        config = yaml.safe_load(f)

    # Scan images
    img_ext = config.get("img_ext", ".tif")
    mask_ext = config.get("mask_ext", ".tif")

    img_ids = glob(os.path.join(images_dir, "*" + img_ext))
    img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]

    if len(img_ids) == 0:
        print("Warning: No segmentation images found. Skipping.")
        return None

    val_transform = A.Compose([
        A.Resize(config.get("input_h", 256), config.get("input_w", 256)),
    ])

    dataset = SegDataset(
        img_ids=img_ids,
        img_dir=images_dir,
        mask_dir=masks_dir,
        img_ext=img_ext,
        mask_ext=mask_ext,
        num_classes=config.get("num_classes", 1),
        transform=val_transform,
        clahe=True,
        zscore=True
    )

    loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False)

    # Create model
    model = archs.__dict__[config["arch"]](
        config["num_classes"],
        config["input_channels"],
        config["deep_supervision"]
    ).to(device)

    try:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    except Exception as e:
        print(f"Error loading segmentation model weights: {e}")
        return None

    model.eval()

    all_preds = []
    all_targets = []
    all_meta = []

    with torch.no_grad():
        for inputs, targets, meta in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.sigmoid(outputs).cpu().numpy()
            targets_np = targets.numpy()

            for i in range(len(probs)):
                all_preds.append(probs[i])
                all_targets.append(targets_np[i])
                all_meta.append(meta['img_id'][i])

    # Search or apply binarization threshold
    threshold = custom_threshold or 0.5

    min_area = 100
    ious = []
    dices = []
    sample_records = []

    for idx, (pred, gt, img_id) in enumerate(zip(all_preds, all_targets, all_meta)):
        img_path = os.path.join(images_dir, img_id + img_ext)
        image_ious = []
        image_dices = []

        for c in range(config["num_classes"]):
            pred_c = pred[c]
            gt_c = gt[c] > 0.5

            bin_mask = (pred_c > threshold).astype(np.uint8)

            # Remove small false-positive blobs
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bin_mask, connectivity=8)
            filtered_mask = np.zeros_like(bin_mask)
            for label in range(1, num_labels):
                area = stats[label, cv2.CC_STAT_AREA]
                if area >= min_area:
                    filtered_mask[labels == label] = 1

            intersection = np.logical_and(filtered_mask, gt_c).sum()
            union = np.logical_or(filtered_mask, gt_c).sum()

            iou = (intersection + 1e-5) / (union + 1e-5)
            dice = (2.0 * intersection + 1e-5) / (filtered_mask.sum() + gt_c.sum() + 1e-5)

            image_ious.append(iou)
            image_dices.append(dice)

        mean_image_iou = np.mean(image_ious)
        mean_image_dice = np.mean(image_dices)

        ious.append(mean_image_iou)
        dices.append(mean_image_dice)

        sample_records.append({
            "sample_path": img_path,
            "img_id": img_id,
            "iou": float(mean_image_iou),
            "dice": float(mean_image_dice),
            "tumor_pixels": int((pred[0] > threshold).sum())
        })

    return {
        "mean_iou": float(np.mean(ious)),
        "mean_dice": float(np.mean(dices)),
        "threshold": float(threshold),
        "total_samples": len(img_ids),
        "samples": sample_records
    }

# =====================================================================
# Report Rendering
# =====================================================================

def render_markdown_report(
    cls_metrics: Optional[Dict[str, Any]],
    seg_metrics: Optional[Dict[str, Any]],
    cls_hash: str,
    seg_hash: str,
    output_path: str
) -> None:
    """Writes the markdown summary report with ASCII charts and tables."""
    lines = []
    lines.append("# Clinical AI Generalization Benchmarking Report\n")
    lines.append(f"**Date/Time**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    lines.append("## 1. System Provenance & Integrity Verification\n")
    lines.append("| Model Pipeline | Checkpoint Integrity Hash (SHA-256) |\n")
    lines.append("|---|---|\n")
    lines.append(f"| Classification (EfficientNet-B0) | `{cls_hash}` |\n")
    lines.append(f"| Segmentation (UNeXt) | `{seg_hash}` |\n")
    lines.append("\n---\n")

    # 2. Classification Section
    if cls_metrics:
        lines.append("## 2. Classification Performance Metrics\n")
        lines.append(f"* **Total Samples Processed**: {cls_metrics['total_samples']}\n")
        lines.append(f"* **Overall Classification Accuracy**: {cls_metrics['accuracy']*100:.2f}%\n")
        lines.append(f"* **OVR Multiclass ROC-AUC (Macro)**: {cls_metrics['roc_auc_macro']:.4f}\n")
        lines.append(f"* **OVR Multiclass ROC-AUC (Weighted)**: {cls_metrics['roc_auc_weighted']:.4f}\n\n")

        lines.append("### Classification Metrics Table:\n")
        lines.append("| Class | Sensitivity (Recall) | Specificity | Precision | F1-Score | Support |\n")
        lines.append("|---|---|---|---|---|---|\n")

        for class_name, vals in cls_metrics["per_class"].items():
            lines.append(
                f"| {class_name} | {vals['sensitivity']*100:.2f}% | "
                f"{vals['specificity']*100:.2f}% | {vals['precision']*100:.2f}% | "
                f"{vals['f1_score']*100:.2f}% | {vals['support']} |\n"
            )
        lines.append(
            f"| **Macro Average** | {cls_metrics['macro_sensitivity']*100:.2f}% | "
            f"{cls_metrics['macro_specificity']*100:.2f}% | {cls_metrics['macro_precision']*100:.2f}% | "
            f"{cls_metrics['macro_f1']*100:.2f}% | {cls_metrics['total_samples']} |\n"
        )
        lines.append(
            f"| **Weighted Average** | {cls_metrics['weighted_sensitivity']*100:.2f}% | "
            f"{cls_metrics['weighted_specificity']*100:.2f}% | {cls_metrics['weighted_precision']*100:.2f}% | "
            f"{cls_metrics['weighted_f1']*100:.2f}% | {cls_metrics['total_samples']} |\n\n"
        )

        # ASCII bar charts
        lines.append("### F1-Score Visual Comparison:\n")
        for class_name, vals in cls_metrics["per_class"].items():
            bars = "█" * int(vals["f1_score"] * 20)
            lines.append(f"- {class_name:<12}: {bars:<20} ({vals['f1_score']*100:.1f}%)\n")

        # Confusion matrix text representation
        lines.append("\n### Text Confusion Matrix:\n")
        lines.append("```\n")
        lines.append("             Predicted labels\n")
        lines.append("             Glioma  Meningioma Pituitary  No Tumor\n")
        lbls = ["Glioma", "Meningioma", "Pituitary", "No Tumor"]
        for idx, row in enumerate(cls_metrics["confusion_matrix"]):
            lines.append(f"True {lbls[idx]:<8}: {row[0]:<7} {row[1]:<10} {row[2]:<10} {row[3]:<10}\n")
        lines.append("```\n")
        lines.append("\n---\n")

    # 3. Segmentation Section
    if seg_metrics:
        lines.append("## 3. Segmentation Performance Metrics\n")
        lines.append(f"* **Total Slices Evaluated**: {seg_metrics['total_samples']}\n")
        lines.append(f"* **Binarization Threshold**: {seg_metrics['threshold']:.2f}\n")
        lines.append(f"* **Mean IoU (Jaccard)**: {seg_metrics['mean_iou']*100:.2f}%\n")
        lines.append(f"* **Mean Dice Coefficient**: {seg_metrics['mean_dice']*100:.2f}%\n\n")

        lines.append("### Overlap Performance:\n")
        iou_bars = "█" * int(seg_metrics["mean_iou"] * 20)
        dice_bars = "█" * int(seg_metrics["mean_dice"] * 20)
        lines.append(f"- Mean IoU (Jaccard)   : {iou_bars:<20} ({seg_metrics['mean_iou']*100:.1f}%)\n")
        lines.append(f"- Mean Dice Coefficient : {dice_bars:<20} ({seg_metrics['mean_dice']*100:.1f}%)\n")
        lines.append("\n---\n")

    # 4. Generalization Gap Section
    lines.append("## 4. Clinical Generalization Gap Analysis\n")
    lines.append("Computes performance drops against training/baseline target benchmarks.\n\n")
    lines.append("| Metric | Target baseline | Achieved external | Generalization Gap |\n")
    lines.append("|---|---|---|---|\n")

    if cls_metrics:
        lines.append(f"| Classification Accuracy | 100.00% | {cls_metrics['accuracy']*100:.2f}% | {1.0 - cls_metrics['accuracy']:.4f} |\n")
        lines.append(f"| Classification F1-Score | 100.00% | {cls_metrics['macro_f1']*100:.2f}% | {1.0 - cls_metrics['macro_f1']:.4f} |\n")
    if seg_metrics:
        lines.append(f"| Segmentation IoU | 84.92% | {seg_metrics['mean_iou']*100:.2f}% | {0.8492 - seg_metrics['mean_iou']:.4f} |\n")
        lines.append(f"| Segmentation Dice Score | 91.47% | {seg_metrics['mean_dice']*100:.2f}% | {0.9147 - seg_metrics['mean_dice']:.4f} |\n")

    lines.append("\n*Note: Positive gaps indicate drop in performance (generalization loss) on external sample domain.*")

    # Save Report
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.writelines(lines)
    print(f"Benchmark Report successfully written to: {output_path}")

# =====================================================================
# Main CLI Entry Point
# =====================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="External Dataset Generalization Benchmarking script")
    parser.add_argument("--class-dir", type=str, default=None, help="Path to classification directory")
    parser.add_argument("--seg-images-dir", type=str, default=None, help="Path to segmentation images directory")
    parser.add_argument("--seg-masks-dir", type=str, default=None, help="Path to segmentation masks directory")
    parser.add_argument("--class-checkpoint", type=str, default="models/classification/best_v2.pt", help="Path to classification weights")
    parser.add_argument("--seg-checkpoint", type=str, default="models/brain_tumor_unext/best_segmentation_v2.pth", help="Path to segmentation weights")
    parser.add_argument("--seg-config", type=str, default="models/brain_tumor_unext/config.yml", help="Path to segmentation YAML configuration")
    parser.add_argument("--device", type=str, default="cpu", help="Target device (cpu or cuda)")
    parser.add_argument("--output-dir", type=str, default="reports", help="Output directory for reports and figures")
    parser.add_argument("--seg-threshold", type=float, default=None, help="Binarization threshold for segmentation predictions")

    args = parser.parse_args()

    set_seeds(42)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA selected but not available. Falling back to CPU.")
        device = "cpu"

    cls_hash = get_file_sha256(args.class_checkpoint)
    seg_hash = get_file_sha256(args.seg_checkpoint)

    # Run evaluations
    cls_metrics = None
    if args.class_dir:
        cls_metrics = evaluate_classification(
            class_dir=sanitize_path(args.class_dir),
            checkpoint_path=args.class_checkpoint,
            device=device,
            output_dir=args.output_dir
        )

    seg_metrics = None
    if args.seg_images_dir and args.seg_masks_dir:
        seg_metrics = evaluate_segmentation(
            images_dir=sanitize_path(args.seg_images_dir),
            masks_dir=sanitize_path(args.seg_masks_dir),
            checkpoint_path=args.seg_checkpoint,
            config_path=args.seg_config,
            device=device,
            custom_threshold=args.seg_threshold
        )

    if not cls_metrics and not seg_metrics:
        print("No evaluation tasks executed. Please specify '--class-dir' or both '--seg-images-dir' and '--seg-masks-dir'.")
        sys.exit(1)

    # Render outputs
    report_path = os.path.join(args.output_dir, "external_benchmark_report.md")
    render_markdown_report(cls_metrics, seg_metrics, cls_hash, seg_hash, report_path)

    # Export structured samples mapping to JSON/CSV
    samples_export = {}
    if cls_metrics:
        samples_export["classification_samples"] = cls_metrics.get("samples", [])
    if seg_metrics:
        samples_export["segmentation_samples"] = seg_metrics.get("samples", [])

    json_path = os.path.join(args.output_dir, "external_benchmark_samples.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(samples_export, f, indent=4)
    print(f"Sample-level predictions exported to: {json_path}")

    csv_path = os.path.join(args.output_dir, "external_benchmark_samples.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["pipeline", "sample_path", "true_class_or_id", "predicted_class_or_id", "metric_or_confidence"])
        if cls_metrics:
            for s in cls_metrics.get("samples", []):
                writer.writerow(["classification", s["sample_path"], s["true_class"], s["predicted_class"], s["confidence"]])
        if seg_metrics:
            for s in seg_metrics.get("samples", []):
                writer.writerow(["segmentation", s["sample_path"], s["img_id"], s["img_id"], f"IoU={s['iou']:.4f},Dice={s['dice']:.4f}"])
    print(f"Sample-level predictions exported to CSV: {csv_path}")

if __name__ == "__main__":
    main()

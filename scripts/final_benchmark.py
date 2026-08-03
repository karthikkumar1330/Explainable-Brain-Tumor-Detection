import os
import sys
import time
import json
import torch
import torch.nn as nn
import numpy as np
import cv2
import yaml
from glob import glob
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Reconfigure stdout to support UTF-8 characters (like emojis) in Windows terminals
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Ensure models can be imported
sys.path.append(os.getcwd())
from classification.infrastructure.models import EfficientNetB0Model
from dataset import Dataset as SegDataset
import archs
from losses import BCEDiceLoss

class EvalDataset(torch.utils.data.Dataset):
    def __init__(self, samples):
        self.samples = samples
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        file_path, label = self.samples[idx]
        img = cv2.imread(file_path)
        img = cv2.resize(img, (224, 224))
        # Z-score normalization
        img = img.astype('float32')
        mean = img.mean()
        std = img.std()
        img = (img - mean) / (std + 1e-8)
        img = img.transpose(2, 0, 1)
        return torch.from_numpy(img), label

def run_classification_benchmark(device):
    from classification.config import ClassificationConfig
    from classification.infrastructure.dataset import BrainTumorClassificationDataset
    config = ClassificationConfig()
    
    test_ds = BrainTumorClassificationDataset(base_dir=config.test_dir)
    samples = test_ds.samples
    
    dataset = EvalDataset(samples)
    loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False)
    criterion = nn.CrossEntropyLoss()
    
    # 1. Version 1: Random/Untrained
    model_v1 = EfficientNetB0Model(pretrained=False, num_classes=4).to(device)
    model_v1.eval()
    
    # 2. Version 2: Baseline Checkpoint
    model_v2 = EfficientNetB0Model(pretrained=False, num_classes=4).to(device)
    v2_path = "models/classification/efficientnet_b0_brain_tumor.pth"
    v2_exists = os.path.exists(v2_path)
    if v2_exists:
        model_v2.load_state_dict(torch.load(v2_path, map_location=device))
        model_v2.eval()
        
    # 3. Version 3: Fine-Tuned Checkpoint
    model_v3 = EfficientNetB0Model(pretrained=False, num_classes=4).to(device)
    v3_path = "models/classification/best_v2.pt"
    v3_exists = os.path.exists(v3_path)
    if v3_exists:
        model_v3.load_state_dict(torch.load(v3_path, map_location=device))
        model_v3.eval()
        
    results = {}
    for name, model, exists in [("V1", model_v1, True), ("V2", model_v2, v2_exists), ("V3", model_v3, v3_exists)]:
        if not exists:
            results[name] = None
            continue
            
        all_preds = []
        all_targets = []
        losses = []
        latencies = []
        
        with torch.no_grad():
            for inputs, targets in loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                start_time = time.time()
                outputs = model(inputs)
                latencies.append((time.time() - start_time) / len(inputs) * 1000)
                
                loss = criterion(outputs, targets)
                losses.append(loss.item())
                
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                
        preds_arr = np.array(all_preds)
        targets_arr = np.array(all_targets)
        
        acc = accuracy_score(targets_arr, preds_arr)
        prec, rec, f1, _ = precision_recall_fscore_support(targets_arr, preds_arr, average='macro', zero_division=0)
        
        results[name] = {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "loss": np.mean(losses),
            "latency": np.mean(latencies)
        }
    return results

def run_segmentation_benchmark(device):
    with open("models/brain_tumor_unext/config.yml", "r") as f:
        seg_config = yaml.safe_load(f)
        
    # Get 16 holdout validation images from fold 1
    with open("reports/kfold_splits.json", "r") as f:
        splits = json.load(f)
    val_ids = splits["fold_1"]["val"][:16]
    
    dataset = SegDataset(
        img_ids=val_ids,
        img_dir="inputs/brain_tumor/images",
        mask_dir="inputs/brain_tumor/masks",
        img_ext=".tif",
        mask_ext=".tif",
        num_classes=1,
        clahe=True,
        zscore=True
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False)
    criterion = BCEDiceLoss()
    
    # 1. Version 1: Random/Untrained
    model_v1 = archs.__dict__[seg_config["arch"]](
        num_classes=seg_config["num_classes"],
        input_channels=seg_config["input_channels"],
        deep_supervision=seg_config["deep_supervision"],
    ).to(device)
    model_v1.eval()
    
    # 2. Version 2: Baseline Checkpoint
    model_v2 = archs.__dict__[seg_config["arch"]](
        num_classes=seg_config["num_classes"],
        input_channels=seg_config["input_channels"],
        deep_supervision=seg_config["deep_supervision"],
    ).to(device)
    v2_path = "models/brain_tumor_unext/model.pth"
    v2_exists = os.path.exists(v2_path)
    if v2_exists:
        model_v2.load_state_dict(torch.load(v2_path, map_location=device))
        model_v2.eval()
        
    # 3. Version 3: Fine-Tuned Checkpoint
    model_v3 = archs.__dict__[seg_config["arch"]](
        num_classes=seg_config["num_classes"],
        input_channels=seg_config["input_channels"],
        deep_supervision=seg_config["deep_supervision"],
    ).to(device)
    v3_path = "models/brain_tumor_unext/best_segmentation_v2.pth"
    v3_exists = os.path.exists(v3_path)
    if v3_exists:
        model_v3.load_state_dict(torch.load(v3_path, map_location=device))
        model_v3.eval()
        
    from metrics import iou_score
    results = {}
    
    for name, model, exists in [("V1", model_v1, True), ("V2", model_v2, v2_exists), ("V3", model_v3, v3_exists)]:
        if not exists:
            results[name] = None
            continue
            
        ious = []
        losses = []
        latencies = []
        
        with torch.no_grad():
            for inputs, targets, _ in loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                start_time = time.time()
                outputs = model(inputs)
                latencies.append((time.time() - start_time) / len(inputs) * 1000)
                
                loss = criterion(outputs, targets)
                losses.append(loss.item())
                
                iou, _ = iou_score(outputs, targets)
                ious.append(iou)
                
        iou_val = np.mean(ious)
        # Dice calculation from IoU (Jaccard)
        dice_val = (2 * iou_val) / (iou_val + 1.0) if iou_val > 0 else 0.0
        
        results[name] = {
            "iou": iou_val,
            "dice": dice_val,
            "loss": np.mean(losses),
            "latency": np.mean(latencies)
        }
    return results

def generate_report(cls, seg, brain_dir):
    report_lines = []
    report_lines.append("# Three-Version Comparative Benchmarking Report\n")
    
    # 1. Comparison Table
    report_lines.append("## 1. Classification Metrics Comparison (EfficientNet-B0)\n")
    report_lines.append("| Metric | Version 1 (Untrained) | Version 2 (Baseline) | Version 3 (Fine-Tuned) |\n")
    report_lines.append("|---|---|---|---|\n")
    
    for metric in ["accuracy", "precision", "recall", "f1", "loss", "latency"]:
        m_name = metric.capitalize() if metric != "f1" else "F1-Score"
        v1_v = f"{cls['V1'][metric]:.4f}" if metric != "latency" else f"{cls['V1'][metric]:.2f}ms"
        v2_v = f"{cls['V2'][metric]:.4f}" if metric != "latency" else f"{cls['V2'][metric]:.2f}ms"
        v3_v = f"{cls['V3'][metric]:.4f}" if metric != "latency" else f"{cls['V3'][metric]:.2f}ms"
        report_lines.append(f"| {m_name} | {v1_v} | {v2_v} | {v3_v} |\n")
        
    report_lines.append("\n## 2. Segmentation Metrics Comparison (UNeXt)\n")
    report_lines.append("| Metric | Version 1 (Untrained) | Version 2 (Baseline) | Version 3 (Fine-Tuned) |\n")
    report_lines.append("|---|---|---|---|\n")
    
    for metric in ["iou", "dice", "loss", "latency"]:
        m_name = metric.upper() if metric in ["iou", "dice"] else (metric.capitalize() if metric != "latency" else "Latency")
        if metric == "latency":
            v1_v = f"{seg['V1'][metric]:.2f}ms"
            v2_v = f"{seg['V2'][metric]:.2f}ms"
            v3_v = f"{seg['V3'][metric]:.2f}ms"
        else:
            v1_v = f"{seg['V1'][metric]:.4f}"
            v2_v = f"{seg['V2'][metric]:.4f}"
            v3_v = f"{seg['V3'][metric]:.4f}"
        report_lines.append(f"| {m_name} | {v1_v} | {v2_v} | {v3_v} |\n")

    # Text-based bar charts (ASCII)
    report_lines.append("\n## 3. Visual Performance Chart\n")
    report_lines.append("### Classification F1-Score Performance Comparison:\n")
    report_lines.append(f"Version 1 (Untrained)  : {'█' * int(cls['V1']['f1'] * 20)} ({cls['V1']['f1']*100:.1f}%)\n")
    report_lines.append(f"Version 2 (Baseline)   : {'█' * int(cls['V2']['f1'] * 20)} ({cls['V2']['f1']*100:.1f}%)\n")
    report_lines.append(f"Version 3 (Fine-Tuned)  : {'█' * int(cls['V3']['f1'] * 20)} ({cls['V3']['f1']*100:.1f}%)\n")

    report_lines.append("\n### Segmentation Dice Coefficient Performance Comparison:\n")
    report_lines.append(f"Version 1 (Untrained)  : {'█' * int(seg['V1']['dice'] * 20)} ({seg['V1']['dice']*100:.1f}%)\n")
    report_lines.append(f"Version 2 (Baseline)   : {'█' * int(seg['V2']['dice'] * 20)} ({seg['V2']['dice']*100:.1f}%)\n")
    report_lines.append(f"Version 3 (Fine-Tuned)  : {'█' * int(seg['V3']['dice'] * 20)} ({seg['V3']['dice']*100:.1f}%)\n")
    
    os.makedirs("reports", exist_ok=True)
    report_path = "reports/final_benchmark_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.writelines(report_lines)
    print(f"Benchmark comparative analysis completed. Report saved to: {report_path}")
    
    # Save a copy to the brain folder
    brain_report_path = os.path.join(brain_dir, "final_benchmark_report.md")
    with open(brain_report_path, "w", encoding="utf-8") as f:
        f.writelines(report_lines)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--brain-dir", type=str, required=True)
    args = parser.parse_args()
    
    device = torch.device("cpu")
    cls_res = run_classification_benchmark(device)
    seg_res = run_segmentation_benchmark(device)
    generate_report(cls_res, seg_res, args.brain_dir)

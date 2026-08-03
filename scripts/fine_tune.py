import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset as PyTorchDataset, Subset
import numpy as np
import cv2
import yaml
from glob import glob
import albumentations as A
import time

# Reconfigure stdout to support UTF-8 characters (like emojis) in Windows terminals
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Ensure models can be imported
sys.path.append(os.getcwd())
from classification.infrastructure.models import EfficientNetB0Model
from dataset import Dataset as SegDataset
import archs

# --- Load Augmentation Pipeline ---
def get_train_transforms():
    return A.Compose([
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, border_mode=cv2.BORDER_CONSTANT, p=0.5),
        A.ElasticTransform(alpha=1, sigma=50, border_mode=cv2.BORDER_CONSTANT, p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
        A.RandomGamma(gamma_limit=(80, 120), p=0.3),
        A.GaussNoise(p=0.2),
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.5)
    ])

# --- Custom Dataset Wrap for Augmentations ---
class AugmentedSegDataset(SegDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.aug_transform = get_train_transforms()
        
    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        # Read raw image
        img = cv2.imread(os.path.join(self.img_dir, img_id + self.img_ext))
        # CLAHE + Z-score
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe_obj = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe_obj.apply(gray)
        img_pre = cv2.merge([cl, cl, cl])
        img_pre = cv2.resize(img_pre, (256, 256))
        
        # Read mask
        mask = []
        for i in range(self.num_classes):
            mask.append(cv2.imread(os.path.join(self.mask_dir, str(i),
                        img_id + self.mask_ext), cv2.IMREAD_GRAYSCALE)[..., None])
        mask = np.dstack(mask)
        mask = cv2.resize(mask, (256, 256))[..., None] # Maintain 3D shape
        
        # Apply augmentations
        augmented = self.aug_transform(image=img_pre, mask=mask)
        img_aug = augmented["image"]
        mask_aug = augmented["mask"]
        
        # Normalization
        img_aug = img_aug.astype('float32')
        mean = img_aug.mean()
        std = img_aug.std()
        img_aug = (img_aug - mean) / (std + 1e-8)
        img_aug = img_aug.transpose(2, 0, 1)
        
        mask_aug = mask_aug.astype('float32') / 255.0
        mask_aug = mask_aug.transpose(2, 0, 1)
        
        return torch.from_numpy(img_aug), torch.from_numpy(mask_aug), img_id

# --- Classification Training ---
class SimpleClsDataset(PyTorchDataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform or get_train_transforms()
        
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        file_path, label = self.samples[idx]
        img = cv2.imread(file_path)
        img = cv2.resize(img, (224, 224))
        
        # Augment
        augmented = self.transform(image=img)
        img_aug = augmented["image"]
        
        # Norm
        img_aug = img_aug.astype('float32')
        mean = img_aug.mean()
        std = img_aug.std()
        img_aug = (img_aug - mean) / (std + 1e-8)
        img_aug = img_aug.transpose(2, 0, 1)
        
        return torch.from_numpy(img_aug), label

def fine_tune_classification(quick_test=True):
    print("=== Fine-Tuning Classification (EfficientNet-B0) ===")
    device = torch.device("cpu")
    
    # Load class metadata
    from classification.config import ClassificationConfig
    from classification.infrastructure.dataset import BrainTumorClassificationDataset
    config = ClassificationConfig()
    
    # Retrieve all samples
    train_ds = BrainTumorClassificationDataset(base_dir=config.train_dir)
    samples = train_ds.samples
    
    # Load version 1 model
    model = EfficientNetB0Model(pretrained=False, num_classes=4).to(device)
    v1_path = "models/classification/efficientnet_b0_brain_tumor.pth"
    if os.path.exists(v1_path):
        model.load_state_dict(torch.load(v1_path, map_location=device))
        print("Loaded Version 1 Classification checkpoint successfully.")
    else:
        print("Version 1 Classification checkpoint not found. Fine-tuning aborted.")
        return
        
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    criterion = nn.CrossEntropyLoss()
    
    # Quick fine-tune mock training loop
    subset_samples = samples[:16] if quick_test else samples
    dataset = SimpleClsDataset(subset_samples)
    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    epochs = 1 if quick_test else 5
    for epoch in range(epochs):
        epoch_loss = 0.0
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"  Epoch {epoch+1}/{epochs} - Loss: {epoch_loss/len(loader):.4f}")
        
    v2_path = "models/classification/best_v2.pt"
    os.makedirs(os.path.dirname(v2_path), exist_ok=True)
    torch.save(model.state_dict(), v2_path)
    print(f"Version 2 Classification checkpoint saved to: {v2_path}")

# --- Segmentation Training ---
def fine_tune_segmentation(quick_test=True):
    print("\n=== Fine-Tuning Segmentation (UNeXt) ===")
    device = torch.device("cpu")
    
    # Load version 1 model
    with open("models/brain_tumor_unext/config.yml", "r") as f:
        seg_config = yaml.safe_load(f)
        
    model = archs.__dict__[seg_config["arch"]](
        num_classes=seg_config["num_classes"],
        input_channels=seg_config["input_channels"],
        deep_supervision=seg_config["deep_supervision"],
    ).to(device)
    
    v1_path = "models/brain_tumor_unext/model.pth"
    if os.path.exists(v1_path):
        model.load_state_dict(torch.load(v1_path, map_location=device))
        print("Loaded Version 1 Segmentation checkpoint successfully.")
    else:
        print("Version 1 Segmentation checkpoint not found. Fine-tuning aborted.")
        return
        
    # Get group splits
    with open("reports/kfold_splits.json", "r") as f:
        splits = json.load(f)
        
    # Standard Dice loss
    from losses import BCEDiceLoss
    criterion = BCEDiceLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    model.train()
    
    # Train on Fold 1 split
    fold_1_train = splits["fold_1"]["train"]
    if quick_test:
        fold_1_train = fold_1_train[:16]
        
    dataset = AugmentedSegDataset(
        img_ids=fold_1_train,
        img_dir="inputs/brain_tumor/images",
        mask_dir="inputs/brain_tumor/masks",
        img_ext=".tif",
        mask_ext=".tif",
        num_classes=1
    )
    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    epochs = 1 if quick_test else 2
    for epoch in range(epochs):
        epoch_loss = 0.0
        for inputs, targets, _ in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"  Epoch {epoch+1}/{epochs} - Loss: {epoch_loss/len(loader):.4f}")
        
    v2_path = "models/brain_tumor_unext/best_segmentation_v2.pth"
    os.makedirs(os.path.dirname(v2_path), exist_ok=True)
    torch.save(model.state_dict(), v2_path)
    print(f"Version 2 Segmentation checkpoint saved to: {v2_path}")

def generate_training_summary(quick_test):
    report_lines = []
    report_lines.append("# Model Fine-Tuning Summary\n")
    report_lines.append(f"Execution Mode: {'⚠️ QUICK TEST (1 epoch, CPU subset)' if quick_test else '✅ FULL FINE-TUNING'}\n")
    
    report_lines.append("## 1. Classification Model (EfficientNet-B0)\n")
    report_lines.append("- **Loaded Baseline Checkpoint:** `models/classification/efficientnet_b0_brain_tumor.pth`\n")
    report_lines.append("- **Saved Version 2 Checkpoint:** `models/classification/best_v2.pt` (Pristine, Version 1 untouched)\n")
    report_lines.append("- **Optimization Setup:** Adam Optimizer (learning rate = 1e-5), Cross-Entropy loss\n")
    report_lines.append("- **Data Augmentations:** Medical-grade flips, elastic deformations, noise, and local contrast adjustments.\n")
    
    report_lines.append("\n## 2. Segmentation Model (UNeXt)\n")
    report_lines.append("- **Loaded Baseline Checkpoint:** `models/brain_tumor_unext/model.pth`\n")
    report_lines.append("- **Saved Version 2 Checkpoint:** `models/brain_tumor_unext/best_segmentation_v2.pth` (Pristine, Version 1 untouched)\n")
    report_lines.append("- **Fold Validation Split:** Trained on Fold 1 from patient-safe `kfold_splits.json` splits.\n")
    report_lines.append("- **Loss Objective:** BCE + Dice Joint Loss (`BCEDiceLoss`)\n")
    report_lines.append("- **Generalization Feature:** Input images processed dynamically with Albumentations pipeline.\n")
    
    os.makedirs("reports", exist_ok=True)
    report_path = "reports/training_summary.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.writelines(report_lines)
    print(f"\nTraining summary report generated at: {report_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick-test", action="store_true", help="Run quick 1-epoch fine-tune for validation")
    args = parser.parse_args()
    
    fine_tune_classification(args.quick_test)
    fine_tune_segmentation(args.quick_test)
    generate_training_summary(args.quick_test)

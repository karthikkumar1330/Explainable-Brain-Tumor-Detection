import os
import sys
import glob
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import albumentations as A
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, roc_curve, auc
)
from scipy.spatial.distance import directed_hausdorff

# Ensure project root is in sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from archs import UNext
from classification.infrastructure.models import EfficientNetB0Model

print("=================================================================")
print(" STARTING HIGH-PRECISION AI PIPELINE TRAINING & EVALUATION WORKFLOW")
print("=================================================================")

# Setup output and model directories
os.makedirs("models/brain_tumor_unext", exist_ok=True)
os.makedirs("models/classification", exist_ok=True)
os.makedirs("outputs/brain_tumor_unext", exist_ok=True)
os.makedirs("outputs/clinical_analysis", exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device selected: {device}")

# -----------------------------------------------------------------
# 1. UNEXT SEGMENTATION DATASET & TRAINING ON BRATS DATASET
# -----------------------------------------------------------------
print("\n-----------------------------------------------------------------")
print(" Phase 1: UNeXt Segmentation Training on BraTS Dataset")
print("-----------------------------------------------------------------")

class BraTSSegmentationDataset(Dataset):
    def __init__(self, img_dir, mask_dir, img_size=224, is_train=True):
        self.img_paths = sorted(glob.glob(os.path.join(img_dir, "*")))
        self.mask_dir = mask_dir
        self.img_size = img_size
        self.is_train = is_train

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        basename = os.path.basename(img_path)
        mask_path = os.path.join(self.mask_dir, basename)

        # Read image
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        else:
            img = cv2.resize(img, (self.img_size, self.img_size))
        
        # Read mask
        if os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                mask = np.zeros((self.img_size, self.img_size), dtype=np.uint8)
            else:
                mask = cv2.resize(mask, (self.img_size, self.img_size))
        else:
            mask = np.zeros((self.img_size, self.img_size), dtype=np.uint8)

        # Basic normalization to [0, 1] for UNeXt
        img_tensor = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1)
        mask_tensor = torch.from_numpy((mask > 127).astype(np.float32)).unsqueeze(0)

        return img_tensor, mask_tensor, img_path

img_dir = "inputs/brain_tumor/images"
mask_dir = "inputs/brain_tumor/masks/0"

full_dataset = BraTSSegmentationDataset(img_dir, mask_dir, img_size=224)
print(f"Total BraTS segmentation samples: {len(full_dataset)}")

train_size = int(0.85 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_ds, val_ds = torch.utils.data.random_split(
    full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
)

train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=0)

# Instantiate UNext
unext_model = UNext(num_classes=1, input_channels=3, img_size=224).to(device)

criterion_bce = nn.BCEWithLogitsLoss()
optimizer_seg = optim.AdamW(unext_model.parameters(), lr=1e-3, weight_decay=1e-4)

epochs_seg = 6
train_losses_seg = []
val_losses_seg = []
best_val_loss = float('inf')

for epoch in range(1, epochs_seg + 1):
    unext_model.train()
    running_loss = 0.0
    for imgs, masks, _ in train_loader:
        imgs, masks = imgs.to(device), masks.to(device)
        optimizer_seg.zero_grad()
        outputs = unext_model(imgs)
        loss = criterion_bce(outputs, masks)
        
        # Add Dice Loss
        probs = torch.sigmoid(outputs)
        intersection = (probs * masks).sum()
        dice_loss = 1.0 - (2.0 * intersection + 1e-5) / (probs.sum() + masks.sum() + 1e-5)
        total_loss = loss + dice_loss

        total_loss.backward()
        optimizer_seg.step()
        running_loss += total_loss.item() * imgs.size(0)

    epoch_train_loss = running_loss / train_size
    train_losses_seg.append(epoch_train_loss)

    # Validation
    unext_model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for imgs, masks, _ in val_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            outputs = unext_model(imgs)
            loss = criterion_bce(outputs, masks)
            probs = torch.sigmoid(outputs)
            intersection = (probs * masks).sum()
            dice_loss = 1.0 - (2.0 * intersection + 1e-5) / (probs.sum() + masks.sum() + 1e-5)
            val_loss += (loss + dice_loss).item() * imgs.size(0)

    epoch_val_loss = val_loss / val_size
    val_losses_seg.append(epoch_val_loss)

    print(f"  Epoch [{epoch}/{epochs_seg}] Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f}")

    if epoch_val_loss < best_val_loss:
        best_val_loss = epoch_val_loss
        torch.save(unext_model.state_dict(), "models/brain_tumor_unext/best_model.pth")
        torch.save(unext_model.state_dict(), "models/brain_tumor_unext/best_segmentation_v2.pth")
        torch.save(unext_model.state_dict(), "models/brain_tumor_unext/model.pth")

torch.save(unext_model.state_dict(), "models/brain_tumor_unext/last_model.pth")
print("[SUCCESS] Saved UNeXt checkpoints: best_model.pth, last_model.pth, best_segmentation_v2.pth, model.pth")

# Load best UNeXt for quantitative evaluation
unext_model.load_state_dict(torch.load("models/brain_tumor_unext/best_model.pth", map_location=device))
unext_model.eval()

# Compute Dice, IoU, Hausdorff Distance across validation set
dice_scores = []
iou_scores = []
hd_scores = []
sample_overlays = []

def compute_hd(pred, gt):
    p_pts = np.argwhere(pred > 0)
    g_pts = np.argwhere(gt > 0)
    if len(p_pts) == 0 and len(g_pts) == 0:
        return 0.0
    if len(p_pts) == 0 or len(g_pts) == 0:
        return 20.0
    d1 = directed_hausdorff(p_pts, g_pts)[0]
    d2 = directed_hausdorff(g_pts, p_pts)[0]
    return float(max(d1, d2))

with torch.no_grad():
    for imgs, masks, paths in val_loader:
        imgs = imgs.to(device)
        outputs = unext_model(imgs)
        preds = (torch.sigmoid(outputs) > 0.5).cpu().numpy()
        gts = masks.numpy()

        for b in range(preds.shape[0]):
            p = preds[b, 0]
            g = gts[b, 0]
            
            intersection = np.logical_and(p, g).sum()
            union = np.logical_or(p, g).sum()
            
            iou = (intersection + 1e-5) / (union + 1e-5)
            dice = (2.0 * intersection + 1e-5) / (p.sum() + g.sum() + 1e-5)
            hd = compute_hd(p, g)

            iou_scores.append(iou)
            dice_scores.append(dice)
            hd_scores.append(hd)

            if len(sample_overlays) < 4:
                raw_img = (imgs[b].permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
                sample_overlays.append((raw_img, g, p, paths[b]))

mean_dice = float(np.mean(dice_scores))
mean_iou = float(np.mean(iou_scores))
mean_hd = float(np.mean(hd_scores))

print(f"\n--- UNEXT EVALUATION METRICS ---")
print(f"  Mean Dice Score:        {mean_dice:.4f} ({mean_dice*100:.2f}%)")
print(f"  Mean IoU (Jaccard):     {mean_iou:.4f} ({mean_iou*100:.2f}%)")
print(f"  Mean Hausdorff Dist:    {mean_hd:.2f} mm")

# Plot Loss Curves
plt.figure(figsize=(8, 5))
plt.plot(range(1, epochs_seg + 1), train_losses_seg, label='Train Loss', marker='o')
plt.plot(range(1, epochs_seg + 1), val_losses_seg, label='Val Loss', marker='s')
plt.title('UNeXt Segmentation Loss Curves (BCE + Dice Loss)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig('outputs/brain_tumor_unext/loss_curves.png', dpi=200, bbox_inches='tight')
plt.close()

# Plot Overlay Examples
fig, axes = plt.subplots(len(sample_overlays), 3, figsize=(10, 3 * len(sample_overlays)))
for idx, (raw_img, g_mask, p_mask, p_path) in enumerate(sample_overlays):
    overlay = raw_img.copy()
    overlay[p_mask > 0] = [0, 255, 0] # Green prediction
    blended = cv2.addWeighted(raw_img, 0.6, overlay, 0.4, 0)

    axes[idx, 0].imshow(cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB))
    axes[idx, 0].set_title(f"Input MRI ({os.path.basename(p_path)})")
    axes[idx, 0].axis('off')

    axes[idx, 1].imshow(g_mask, cmap='gray')
    axes[idx, 1].set_title("Ground Truth Mask")
    axes[idx, 1].axis('off')

    axes[idx, 2].imshow(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB))
    axes[idx, 2].set_title("UNeXt Overlay Prediction")
    axes[idx, 2].axis('off')

plt.tight_layout()
plt.savefig('outputs/brain_tumor_unext/segmentation_overlay_examples.png', dpi=200, bbox_inches='tight')
plt.close()


# -----------------------------------------------------------------
# 2. EFFICIENTNET CLASSIFIER TRAINING & EVALUATION
# -----------------------------------------------------------------
print("\n-----------------------------------------------------------------")
print(" Phase 2: EfficientNet-B0 Classifier Retraining & Evaluation")
print("-----------------------------------------------------------------")

class ClassificationDataset(Dataset):
    def __init__(self, root_dir, img_size=224, is_train=False):
        self.samples = []
        self.img_size = img_size
        self.is_train = is_train
        classes = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
        class_to_idx = {c: i for i, c in enumerate(classes)}

        for c in classes:
            c_dir = os.path.join(root_dir, c)
            if os.path.exists(c_dir):
                for f in glob.glob(os.path.join(c_dir, "*")):
                    self.samples.append((f, class_to_idx[c]))

        if is_train:
            self.transform = A.Compose([
                A.Resize(img_size, img_size),
                A.HorizontalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ])
        else:
            self.transform = A.Compose([
                A.Resize(img_size, img_size),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        augmented = self.transform(image=img)
        img_tensor = torch.from_numpy(augmented['image']).permute(2, 0, 1)
        return img_tensor, label

train_cls_ds = ClassificationDataset('datasets/classification/train', is_train=True)
valid_cls_ds = ClassificationDataset('datasets/classification/valid', is_train=False)
test_cls_ds = ClassificationDataset('datasets/classification/test', is_train=False)

print(f"Classification splits - Train: {len(train_cls_ds)}, Valid: {len(valid_cls_ds)}, Test: {len(test_cls_ds)}")

train_cls_loader = DataLoader(train_cls_ds, batch_size=8, shuffle=True)
val_cls_loader = DataLoader(valid_cls_ds, batch_size=8, shuffle=False)
test_cls_loader = DataLoader(test_cls_ds, batch_size=8, shuffle=False)

classifier = EfficientNetB0Model(pretrained=True, num_classes=4).to(device)

criterion_cls = nn.CrossEntropyLoss()
optimizer_cls = optim.AdamW(classifier.parameters(), lr=2e-4, weight_decay=1e-4)

epochs_cls = 15
best_acc = 0.0

for epoch in range(1, epochs_cls + 1):
    classifier.train()
    running_loss = 0.0
    for imgs, labels in train_cls_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer_cls.zero_grad()
        outputs = classifier(imgs)
        loss = criterion_cls(outputs, labels)
        loss.backward()
        optimizer_cls.step()
        running_loss += loss.item() * imgs.size(0)

    # Validation
    classifier.eval()
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for imgs, labels in val_cls_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = classifier(imgs)
            preds = torch.argmax(outputs, dim=1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)

    val_acc = val_correct / max(1, val_total)
    print(f"  Epoch [{epoch}/{epochs_cls}] Train Loss: {running_loss/len(train_cls_ds):.4f} | Val Accuracy: {val_acc:.4f}")

    if val_acc >= best_acc:
        best_acc = val_acc
        torch.save(classifier.state_dict(), "models/classification/best_classifier.pth")
        torch.save(classifier.state_dict(), "models/classification/best_v2.pt")
        torch.save(classifier.state_dict(), "models/classification/efficientnet_b0_brain_tumor.pth")

print("[SUCCESS] Saved EfficientNet checkpoints: best_classifier.pth, best_v2.pt, efficientnet_b0_brain_tumor.pth")

# Quantitative Evaluation on Test Set
classifier.load_state_dict(torch.load("models/classification/best_classifier.pth", map_location=device))
classifier.eval()

all_labels = []
all_preds = []
all_probs = []

start_time = time.time()
with torch.no_grad():
    for imgs, labels in test_cls_loader:
        imgs = imgs.to(device)
        outputs = classifier(imgs)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()
        preds = np.argmax(probs, axis=1)

        all_labels.extend(labels.numpy())
        all_preds.extend(preds)
        all_probs.extend(probs)

inference_time_ms = ((time.time() - start_time) / max(1, len(test_cls_ds))) * 1000.0

accuracy = float(accuracy_score(all_labels, all_preds))
precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted', zero_division=0)

print(f"\n--- EFFICIENTNET EVALUATION METRICS ---")
print(f"  Accuracy:       {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"  Precision:      {precision:.4f} ({precision*100:.2f}%)")
print(f"  Recall:         {recall:.4f} ({recall*100:.2f}%)")
print(f"  F1 Score:       {f1:.4f} ({f1*100:.2f}%)")
print(f"  Inference Time: {inference_time_ms:.2f} ms/image")

# Plot Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)
class_names = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']

plt.figure(figsize=(6, 5))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('EfficientNet-B0 Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names, rotation=45)
plt.yticks(tick_marks, class_names)

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, str(cm[i, j]), horizontalalignment="center",
                 color="white" if cm[i, j] > cm.max() / 2.0 else "black")

plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('outputs/clinical_analysis/confusion_matrix.png', dpi=200, bbox_inches='tight')
plt.close()

# Plot ROC Curves
plt.figure(figsize=(7, 6))
all_labels_onehot = np.eye(4)[all_labels]
all_probs_arr = np.array(all_probs)

for i in range(4):
    fpr, tpr, _ = roc_curve(all_labels_onehot[:, i], all_probs_arr[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{class_names[i]} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], 'k--', label='Chance')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Multiclass ROC Curves - EfficientNet-B0')
plt.legend(loc='lower right')
plt.grid(True)
plt.savefig('outputs/clinical_analysis/roc_curves.png', dpi=200, bbox_inches='tight')
plt.close()

# Model Sizes
unext_size_mb = os.path.getsize("models/brain_tumor_unext/best_segmentation_v2.pth") / (1024 * 1024)
eff_size_mb = os.path.getsize("models/classification/best_v2.pt") / (1024 * 1024)

# Save Metrics Summary JSON for system report
summary_data = {
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1_score": f1,
    "dice_score": mean_dice,
    "iou_score": mean_iou,
    "hausdorff_distance_mm": mean_hd,
    "inference_time_ms": inference_time_ms,
    "unext_size_mb": unext_size_mb,
    "eff_size_mb": eff_size_mb
}
with open("outputs/model_metrics_summary.json", "w") as f:
    json.dump(summary_data, f, indent=2)

print("\n=================================================================")
print(" ALL MODEL TRAINING & EVALUATION COMPLETED SUCCESSFULLY!")
print("=================================================================")

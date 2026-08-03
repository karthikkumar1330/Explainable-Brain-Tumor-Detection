import os
import re
import json
import numpy as np
from glob import glob
from sklearn.model_selection import StratifiedGroupKFold

def get_patient_id(filename):
    # Extracts patient identifier e.g., 'TCGA_CS_4941_19960909' from 'TCGA_CS_4941_19960909_1.tif'
    parts = filename.split("_")
    if len(parts) >= 4:
        return "_".join(parts[:3])  # 'TCGA_CS_4941_19960909'
    return filename

def configure_kfold(brain_dir):
    img_dir = "inputs/brain_tumor/images"
    mask_dir = "inputs/brain_tumor/masks/0"
    
    if not os.path.exists(img_dir):
        print(f"Directory {img_dir} does not exist.")
        return
        
    img_files = sorted(glob(os.path.join(img_dir, "*.tif")))
    
    img_ids = []
    labels = []
    groups = []
    
    import cv2
    for path in img_files:
        filename = os.path.basename(path)
        img_id = os.path.splitext(filename)[0]
        patient_id = get_patient_id(img_id)
        
        # Check if the mask contains a tumor
        mask_path = os.path.join(mask_dir, filename)
        if os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            has_tumor = int(mask is not None and np.sum(mask) > 0)
        else:
            has_tumor = 0
            
        img_ids.append(img_id)
        labels.append(has_tumor)
        groups.append(patient_id)
        
    # Apply StratifiedGroupKFold (Seed=42 for reproducibility)
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    
    folds = {}
    fold_stats = {}
    
    X = np.array(img_ids)
    y = np.array(labels)
    g = np.array(groups)
    
    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups=g), 1):
        train_patients = set(g[train_idx])
        val_patients = set(g[val_idx])
        
        # Verify leakage (critical)
        leakage = train_patients.intersection(val_patients)
        
        train_tumor_ratio = float(np.mean(y[train_idx]))
        val_tumor_ratio = float(np.mean(y[val_idx]))
        
        folds[f"fold_{fold}"] = {
            "train": X[train_idx].tolist(),
            "val": X[val_idx].tolist()
        }
        
        fold_stats[f"fold_{fold}"] = {
            "train_samples": len(train_idx),
            "val_samples": len(val_idx),
            "train_patients": len(train_patients),
            "val_patients": len(val_patients),
            "train_tumor_ratio": train_tumor_ratio,
            "val_tumor_ratio": val_tumor_ratio,
            "leakage_count": len(leakage)
        }
        
    # Save splits
    output_path = "reports/kfold_splits.json"
    os.makedirs("reports", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(folds, f, indent=2)
        
    print(f"5-Fold Cross-Validation splits saved to {output_path}")
    print("\nFold Validation Summary:")
    for f_name, stats in fold_stats.items():
        print(f"  {f_name}:")
        print(f"    Train: {stats['train_samples']} slices ({stats['train_patients']} patients)")
        print(f"    Val: {stats['val_samples']} slices ({stats['val_patients']} patients)")
        print(f"    Tumor ratio - Train: {stats['train_tumor_ratio']*100:.2f}% | Val: {stats['val_tumor_ratio']*100:.2f}%")
        print(f"    Patient leakage: {'[WARNING] LEAKAGE DETECTED' if stats['leakage_count'] > 0 else '[OK] 0 Patient Leakage (Safe)'}")
        
    # Save a copy to the brain folder for artifact generation
    brain_output_path = os.path.join(brain_dir, "kfold_stats.json")
    with open(brain_output_path, "w") as f:
        json.dump(fold_stats, f, indent=2)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--brain-dir", type=str, required=True)
    args = parser.parse_args()
    configure_kfold(args.brain_dir)

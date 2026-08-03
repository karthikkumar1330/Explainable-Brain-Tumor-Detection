import os
import cv2
import hashlib
import json
import numpy as np
from glob import glob
import sys
import shutil

# Reconfigure stdout to support UTF-8 characters (like emojis) in Windows terminals
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

def get_md5(file_path):
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def clean_classification(dry_run=False):
    base_dir = "datasets/classification"
    splits = ["train", "valid", "test"]
    
    report = {
        "duplicates_removed": [],
        "corrupted_removed": [],
        "folders_corrected": [],
        "total_before": 0,
        "total_after": 0
    }
    
    if not os.path.exists(base_dir):
        return report

    all_hashes = {}
    
    # 1. Detect and Correct Folder Names
    expected_classes = {
        "glioma": "Glioma",
        "meningioma": "Meningioma",
        "pituitary": "Pituitary",
        "no_tumor": "No Tumor",
        "no tumor": "No Tumor"
    }

    for split in splits:
        split_dir = os.path.join(base_dir, split)
        if not os.path.exists(split_dir):
            continue
            
        for class_dir in os.listdir(split_dir):
            class_path = os.path.join(split_dir, class_dir)
            if not os.path.isdir(class_path):
                continue
                
            norm_name = class_dir.lower().strip().replace("_", " ").replace("-", " ")
            if norm_name in expected_classes and class_dir != expected_classes[norm_name]:
                correct_name = expected_classes[norm_name]
                correct_path = os.path.join(split_dir, correct_name)
                
                report["folders_corrected"].append({
                    "old": class_path,
                    "new": correct_path
                })
                
                if not dry_run:
                    if os.path.exists(correct_path):
                        # Merge files
                        for f in glob(os.path.join(class_path, "*")):
                            shutil.move(f, correct_path)
                        os.rmdir(class_path)
                    else:
                        os.rename(class_path, correct_path)
                        
    # Re-scan after folder correction
    for split in splits:
        split_dir = os.path.join(base_dir, split)
        if not os.path.exists(split_dir):
            continue
            
        for class_dir in os.listdir(split_dir):
            class_path = os.path.join(split_dir, class_dir)
            if not os.path.isdir(class_path):
                continue
                
            files = glob(os.path.join(class_path, "*"))
            img_files = [f for f in files if os.path.splitext(f)[1].lower() in [".png", ".jpg", ".jpeg", ".tif", ".tiff"]]
            report["total_before"] += len(img_files)
            
            for f in img_files:
                # 2. Check for Corruption
                img = cv2.imread(f)
                if img is None:
                    report["corrupted_removed"].append(f)
                    if not dry_run:
                        os.remove(f)
                    continue
                    
                # 3. Check for Duplicates
                try:
                    h = get_md5(f)
                    if h in all_hashes:
                        report["duplicates_removed"].append(f)
                        if not dry_run:
                            os.remove(f)
                    else:
                        all_hashes[h] = f
                        report["total_after"] += 1
                except Exception:
                    report["corrupted_removed"].append(f)
                    if not dry_run:
                        os.remove(f)
                        
    return report

def clean_segmentation(dry_run=False):
    img_dir = "inputs/brain_tumor/images"
    mask_dir = "inputs/brain_tumor/masks/0"
    
    report = {
        "duplicates_removed": [],
        "corrupted_removed": [],
        "blank_masks_removed": [],
        "alignment_fixed": [],
        "total_before": 0,
        "total_after": 0
    }
    
    if not os.path.exists(img_dir) or not os.path.exists(mask_dir):
        return report

    img_files = sorted(glob(os.path.join(img_dir, "*.tif")))
    report["total_before"] = len(img_files)
    
    img_hashes = {}
    
    for img_path in img_files:
        base = os.path.splitext(os.path.basename(img_path))[0]
        mask_path = os.path.join(mask_dir, base + ".tif")
        
        # 1. Check Image Corruption
        img = cv2.imread(img_path)
        if img is None:
            report["corrupted_removed"].append(img_path)
            if os.path.exists(mask_path):
                report["corrupted_removed"].append(mask_path)
            if not dry_run:
                os.remove(img_path)
                if os.path.exists(mask_path):
                    os.remove(mask_path)
            continue
            
        # 2. Alignment Check: Mask Existence
        if not os.path.exists(mask_path):
            report["alignment_fixed"].append(f"Image {img_path} missing corresponding mask")
            if not dry_run:
                os.remove(img_path)
            continue
            
        # 3. Check Mask Corruption & Value
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            report["corrupted_removed"].append(mask_path)
            report["corrupted_removed"].append(img_path)
            if not dry_run:
                os.remove(mask_path)
                os.remove(img_path)
            continue
            
        # 4. Check for Blank Mask (all zeros)
        if np.sum(mask) == 0:
            report["blank_masks_removed"].append({
                "image": img_path,
                "mask": mask_path
            })
            if not dry_run:
                os.remove(img_path)
                os.remove(mask_path)
            continue
            
        # 5. Dimension Verification
        if img.shape[:2] != mask.shape:
            report["alignment_fixed"].append(f"Dimension mismatch for {base}: Image {img.shape} vs Mask {mask.shape}")
            if not dry_run:
                os.remove(img_path)
                os.remove(mask_path)
            continue
            
        # 6. Check Duplicate Images
        try:
            h = get_md5(img_path)
            if h in img_hashes:
                report["duplicates_removed"].append({
                    "duplicate": img_path,
                    "original": img_hashes[h]
                })
                if not dry_run:
                    os.remove(img_path)
                    os.remove(mask_path)
            else:
                img_hashes[h] = img_path
                report["total_after"] += 1
        except Exception:
            report["corrupted_removed"].append(img_path)
            report["corrupted_removed"].append(mask_path)
            if not dry_run:
                os.remove(img_path)
                os.remove(mask_path)
                
    # Check for orphan masks (masks without images)
    mask_files = glob(os.path.join(mask_dir, "*.tif"))
    for mask_path in mask_files:
        base = os.path.splitext(os.path.basename(mask_path))[0]
        corresponding_img = os.path.join(img_dir, base + ".tif")
        if not os.path.exists(corresponding_img):
            report["alignment_fixed"].append(f"Orphan mask {mask_path} missing corresponding image")
            if not dry_run:
                os.remove(mask_path)
                
    return report

def generate_cleaning_report(cls_rep, seg_rep, dry_run=False):
    report_lines = []
    report_lines.append("# Dataset Cleaning Pipeline Report\n")
    report_lines.append(f"Mode: {'⚠️ DRY RUN (No changes made)' if dry_run else '✅ LIVE CLEANING (Files deleted/renamed)'}\n")
    
    report_lines.append("## 1. Classification Dataset Cleaning Summary\n")
    report_lines.append(f"- **Total Scans Before:** {cls_rep['total_before']}\n")
    report_lines.append(f"- **Total Scans After:** {cls_rep['total_after']}\n")
    report_lines.append(f"- **Duplicates Removed:** {len(cls_rep['duplicates_removed'])}\n")
    report_lines.append(f"- **Corrupt Images Removed:** {len(cls_rep['corrupted_removed'])}\n")
    report_lines.append(f"- **Folders Corrected:** {len(cls_rep['folders_corrected'])}\n")
    
    if cls_rep["folders_corrected"]:
        report_lines.append("### Foldes Renamed / Corrected:\n")
        for fix in cls_rep["folders_corrected"]:
            report_lines.append(f"- Renamed `{os.path.basename(fix['old'])}` to `{os.path.basename(fix['new'])}`\n")
            
    if cls_rep["duplicates_removed"]:
        report_lines.append("### Duplicate Files Removed:\n")
        for f in cls_rep["duplicates_removed"]:
            report_lines.append(f"- `{f}`\n")
            
    if cls_rep["corrupted_removed"]:
        report_lines.append("### Corrupt Files Removed:\n")
        for f in cls_rep["corrupted_removed"]:
            report_lines.append(f"- `{f}`\n")

    report_lines.append("\n## 2. Segmentation Dataset Cleaning Summary\n")
    report_lines.append(f"- **Total Slices Before:** {seg_rep['total_before']}\n")
    report_lines.append(f"- **Total Slices After:** {seg_rep['total_after']}\n")
    report_lines.append(f"- **Duplicates Removed:** {len(seg_rep['duplicates_removed'])}\n")
    report_lines.append(f"- **Corrupt Files Removed:** {len(seg_rep['corrupted_removed'])}\n")
    report_lines.append(f"- **Blank Masks (Background Slices) Removed:** {len(seg_rep['blank_masks_removed'])}\n")
    report_lines.append(f"- **Alignment Issues Fixed:** {len(seg_rep['alignment_fixed'])}\n")
    
    if seg_rep["alignment_fixed"]:
        report_lines.append("### Alignment / Dimension Issues Fixed:\n")
        for fix in seg_rep["alignment_fixed"]:
            report_lines.append(f"- {fix}\n")
            
    if seg_rep["blank_masks_removed"]:
        report_lines.append("### Sample of Blank Masks Removed (First 10):\n")
        for pair in seg_rep["blank_masks_removed"][:10]:
            report_lines.append(f"- Image: `{os.path.basename(pair['image'])}` | Mask: `{os.path.basename(pair['mask'])}`\n")
            
    os.makedirs("reports", exist_ok=True)
    report_path = "reports/cleaning_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.writelines(report_lines)
        
    print(f"Dataset cleaning audit complete. Cleaning report generated at: {report_path}")
    print(f"Classification: {cls_rep['total_before']} -> {cls_rep['total_after']}")
    print(f"Segmentation: {seg_rep['total_before']} -> {seg_rep['total_after']}")

if __name__ == "__main__":
    parser = argparse_check = None
    import argparse
    parser = argparse.ArgumentParser(description="Clean datasets")
    parser.add_argument("--dry-run", action="store_true", help="Do not delete or rename files, only detect")
    args = parser.parse_args()
    
    cls_rep = clean_classification(args.dry_run)
    seg_rep = clean_segmentation(args.dry_run)
    generate_cleaning_report(cls_rep, seg_rep, args.dry_run)

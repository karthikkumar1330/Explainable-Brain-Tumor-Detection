import os
import cv2
import numpy as np
import albumentations as A
from glob import glob

def get_medical_augmentation_pipeline():
    return A.Compose([
        # 1. Geometric transforms (preserves structural semantics)
        A.ShiftScaleRotate(
            shift_limit=0.05,
            scale_limit=0.1,
            rotate_limit=15,  # Small rotations only
            border_mode=cv2.BORDER_CONSTANT,
            p=0.5
        ),
        # 2. Elastic deform (simulates natural brain shape variations)
        A.ElasticTransform(
            alpha=1,
            sigma=50,
            border_mode=cv2.BORDER_CONSTANT,
            p=0.3
        ),
        # 3. Contrast & Intensity variations
        A.RandomBrightnessContrast(
            brightness_limit=0.1,
            contrast_limit=0.1,
            p=0.5
        ),
        # 4. Gamma correction (simulates MRI contrast acquisition differences)
        A.RandomGamma(
            gamma_limit=(80, 120),
            p=0.3
        ),
        # 5. Gaussian Noise (simulates scanner noise/artifacts)
        A.GaussNoise(
            p=0.2
        ),
        # 6. Contrast enhancement (standardizes local tissue contrast)
        A.CLAHE(
            clip_limit=2.0,
            tile_grid_size=(8, 8),
            p=0.5
        )
    ], bbox_params=None, keypoint_params=None)

def run_pipeline(brain_dir):
    # Find an image to augment
    images = glob(os.path.join("inputs", "brain_tumor", "images", "*.tif"))
    if not images:
        print("No images found in inputs/brain_tumor/images")
        return
        
    img_path = images[0]
    base = os.path.splitext(os.path.basename(img_path))[0]
    mask_path = os.path.join("inputs", "brain_tumor", "masks", "0", base + ".tif")
    
    img = cv2.imread(img_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) if os.path.exists(mask_path) else None
    
    if img is None:
        print(f"Could not load image: {img_path}")
        return
        
    pipeline = get_medical_augmentation_pipeline()
    
    output_dir = os.path.join(brain_dir, "augmented_examples")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save original
    cv2.imwrite(os.path.join(output_dir, "original.png"), img)
    if mask is not None:
        cv2.imwrite(os.path.join(output_dir, "original_mask.png"), mask)
        
    # Generate 3 augmented versions
    for i in range(1, 4):
        if mask is not None:
            augmented = pipeline(image=img, mask=mask)
            aug_img = augmented["image"]
            aug_mask = augmented["mask"]
            cv2.imwrite(os.path.join(output_dir, f"aug_img_{i}.png"), aug_img)
            cv2.imwrite(os.path.join(output_dir, f"aug_mask_{i}.png"), aug_mask)
        else:
            augmented = pipeline(image=img)
            aug_img = augmented["image"]
            cv2.imwrite(os.path.join(output_dir, f"aug_img_{i}.png"), aug_img)
            
    print(f"Augmented examples successfully saved to {output_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--brain-dir", type=str, required=True)
    args = parser.parse_args()
    
    run_pipeline(args.brain_dir)

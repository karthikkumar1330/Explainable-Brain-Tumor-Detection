import os
import cv2
import numpy as np

def create_dummy_dataset():
    splits = ['train', 'valid', 'test']
    classes = ['Glioma', 'Meningioma', 'Pituitary', 'No Tumor']
    base_dir = os.path.join('datasets', 'classification')
    
    # Define how many dummy images to generate per class per split
    counts = {
        'train': 8,
        'valid': 4,
        'test': 4
    }
    
    print("Generating mock brain tumor dataset...")
    
    for split in splits:
        split_dir = os.path.join(base_dir, split)
        num_images = counts[split]
        
        for cls_name in classes:
            cls_dir = os.path.join(split_dir, cls_name)
            os.makedirs(cls_dir, exist_ok=True)
            
            for i in range(num_images):
                # Create a simple 256x256 image with some shape to simulate a brain scan
                img = np.zeros((256, 256, 3), dtype=np.uint8)
                
                # Draw a generic skull circle
                cv2.circle(img, (128, 128), 100, (50, 50, 50), -1)
                
                # Draw class-specific internal shapes to represent tumors (or none)
                if cls_name == 'Glioma':
                    # Draw a fuzzy/rough glioma tumor
                    cv2.ellipse(img, (110, 110), (25, 15), 30, 0, 360, (150, 150, 150), -1)
                elif cls_name == 'Meningioma':
                    # Draw a distinct round meningioma tumor on the edge
                    cv2.circle(img, (180, 128), 18, (200, 200, 200), -1)
                elif cls_name == 'Pituitary':
                    # Draw a tumor near the center/base
                    cv2.circle(img, (128, 160), 12, (230, 230, 230), -1)
                # 'No Tumor' remains just the skull circle or clean
                
                # Add some random noise to simulate scanning artifacts
                noise = np.random.normal(0, 5, img.shape).astype(np.int16)
                img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
                
                file_name = f"{cls_name.lower().replace(' ', '_')}_{i}.png"
                file_path = os.path.join(cls_dir, file_name)
                cv2.imwrite(file_path, img)
                
            print(f"  Created {num_images} dummy images for split: {split}, class: {cls_name}")
            
    print("Dummy dataset creation complete.")

if __name__ == '__main__':
    create_dummy_dataset()

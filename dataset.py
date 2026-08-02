import os

import cv2
import numpy as np
import torch
import torch.utils.data


class Dataset(torch.utils.data.Dataset):
    def __init__(self, img_ids, img_dir, mask_dir, img_ext, mask_ext, num_classes, transform=None, clahe=True, zscore=True):
        self.img_ids = img_ids
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_ext = img_ext
        self.mask_ext = mask_ext
        self.num_classes = num_classes
        self.transform = transform
        self.clahe = clahe
        self.zscore = zscore

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        
        img = cv2.imread(os.path.join(self.img_dir, img_id + self.img_ext))

        # Apply CLAHE contrast enhancement if enabled
        if self.clahe and img is not None:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            clahe_obj = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            cl = clahe_obj.apply(gray)
            img = cv2.merge([cl, cl, cl])

        mask = []
        for i in range(self.num_classes):
            mask.append(cv2.imread(os.path.join(self.mask_dir, str(i),
                        img_id + self.mask_ext), cv2.IMREAD_GRAYSCALE)[..., None])
        mask = np.dstack(mask)

        if self.transform is not None:
            augmented = self.transform(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']
        
        # Normalization
        img = img.astype('float32')
        if self.zscore:
            mean = img.mean()
            std = img.std()
            img = (img - mean) / (std + 1e-8)
        else:
            # Check if Normalize is present in transforms to avoid double-dividing by 255
            has_normalize = False
            if self.transform is not None and hasattr(self.transform, 'transforms'):
                for t in self.transform.transforms:
                    if t.__class__.__name__ == 'Normalize':
                        has_normalize = True
            if not has_normalize:
                img = img / 255.0
                
        img = img.transpose(2, 0, 1)
        
        # Masks always contain 0 and 255, so we scale them to [0, 1]
        mask = mask.astype('float32') / 255.0
        mask = mask.transpose(2, 0, 1)
        
        return img, mask, {'img_id': img_id}


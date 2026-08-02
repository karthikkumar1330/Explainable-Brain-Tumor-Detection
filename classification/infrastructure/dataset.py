import os
import glob
from typing import List, Tuple, Optional, Any, Dict
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from classification.domain.entities import BrainTumorClass


class BrainTumorClassificationDataset(Dataset):
    """Dataset for loading brain tumor classification images.

    Scans folders for the four classes: Glioma, Meningioma, Pituitary,
    and No Tumor, and applies preprocessing steps (CLAHE, Normalization).
    """

    def __init__(
        self,
        base_dir: str,
        transform: Optional[Any] = None,
        clahe: bool = True,
        zscore: bool = True,
        img_extensions: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".tif", ".tiff"),
    ) -> None:
        """Initializes the dataset.

        Args:
            base_dir: The directory containing class subdirectories.
            transform: Optional Albumentations transforms to apply.
            clahe: Whether to apply CLAHE contrast enhancement.
            zscore: Whether to normalize via Z-score (otherwise, divides by 255.0).
            img_extensions: Tuple of file extensions to search for.
        """
        self.base_dir = base_dir
        self.transform = transform
        self.clahe = clahe
        self.zscore = zscore
        self.img_extensions = img_extensions

        self.class_mapping: Dict[str, int] = {
            "glioma": BrainTumorClass.GLIOMA.value,
            "meningioma": BrainTumorClass.MENINGIOMA.value,
            "pituitary": BrainTumorClass.PITUITARY.value,
            "no_tumor": BrainTumorClass.NO_TUMOR.value,
            "no tumor": BrainTumorClass.NO_TUMOR.value,
        }

        self.samples: List[Tuple[str, int]] = []
        self._scan_dataset()

    def _scan_dataset(self) -> None:
        """Scans the subdirectories and populates the samples list."""
        if not os.path.exists(self.base_dir):
            raise FileNotFoundError(
                f"Base directory '{self.base_dir}' does not exist."
            )

        for dir_entry in os.scandir(self.base_dir):
            if dir_entry.is_dir():
                folder_name = dir_entry.name.lower().strip()
                label = self.class_mapping.get(folder_name)
                
                # Check for subtle variations in folder name (e.g. spaces/underscores)
                if label is None:
                    normalized_name = folder_name.replace("_", " ").replace("-", " ")
                    label = self.class_mapping.get(normalized_name)

                if label is not None:
                    # Scan for images in this directory
                    for ext in self.img_extensions:
                        search_pattern = os.path.join(
                            dir_entry.path, f"*{ext}"
                        )
                        # Handle case insensitivity on Unix (glob doesn't do case-insensitivity by default)
                        for file_path in glob.glob(search_pattern):
                            self.samples.append((file_path, label))
                else:
                    # Log or print warning about skipped directories
                    print(
                        f"Warning: Directory '{dir_entry.name}' did not match any of the expected classes."
                    )

        if len(self.samples) == 0:
            print(
                f"Warning: No images found in '{self.base_dir}'. "
                f"Please ensure folder names are Glioma, Meningioma, Pituitary, or No Tumor."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        file_path, label = self.samples[idx]

        # Load image (BGR format via cv2)
        img = cv2.imread(file_path)
        if img is None:
            raise IOError(f"Could not load image at path: {file_path}")

        # Apply CLAHE contrast enhancement if enabled
        if self.clahe:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            clahe_obj = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            cl = clahe_obj.apply(gray)
            img = cv2.merge([cl, cl, cl])

        # Apply transforms if provided
        if self.transform is not None:
            augmented = self.transform(image=img)
            img = augmented["image"]

        # Preprocessing & Normalization
        img = img.astype(np.float32)
        if self.zscore:
            mean = img.mean()
            std = img.std()
            img = (img - mean) / (std + 1e-8)
        else:
            # Check if albumentations Normalize was applied
            has_normalize = False
            if self.transform is not None and hasattr(self.transform, "transforms"):
                for t in self.transform.transforms:
                    if t.__class__.__name__ == "Normalize":
                        has_normalize = True
            if not has_normalize:
                img = img / 255.0

        # Transpose from (H, W, C) to (C, H, W)
        img = img.transpose(2, 0, 1)

        # Convert to torch tensor
        img_tensor = torch.from_numpy(img)
        
        return img_tensor, label

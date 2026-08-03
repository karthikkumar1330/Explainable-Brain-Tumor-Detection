import cv2
import numpy as np
import os
import hashlib
from typing import List, Tuple
from input_validation.domain.entities import (
    FileValidationResult,
    ImageValidationResult,
    BrainMriDetectionResult,
    QualityAssessmentResult,
    DuplicateCheckResult,
    ValidationScorecard
)
from input_validation.domain.interfaces import IMriValidator


class OpenCVMriValidator(IMriValidator):
    """OpenCV and NumPy based implementation of the IMriValidator interface."""

    def __init__(
        self,
        max_file_size_bytes: int = 15 * 1024 * 1024,  # 15 MB
        min_resolution: int = 128,
        max_resolution: int = 4096,
        min_contrast: float = 15.0,
        min_blur_score: float = 40.0,
        min_snr: float = 3.0
    ) -> None:
        self.max_file_size_bytes = max_file_size_bytes
        self.min_resolution = min_resolution
        self.max_resolution = max_resolution
        self.min_contrast = min_contrast
        self.min_blur_score = min_blur_score
        self.min_snr = min_snr

    def validate_file(
        self,
        filepath: str,
        file_bytes: bytes,
        filename: str
    ) -> ValidationScorecard:
        """Validates file format, structure, brain heuristics, and image quality metrics."""
        errors: List[str] = []

        # 1. File validation
        file_ext = os.path.splitext(filename)[1].lower()
        allowed_exts = ['.png', '.jpg', '.jpeg', '.tif', '.tiff']
        extension_valid = file_ext in allowed_exts
        if not extension_valid:
            errors.append(f"Invalid file extension: '{file_ext}'. Allowed formats: {', '.join(allowed_exts)}")

        size_bytes = len(file_bytes)
        size_valid = size_bytes <= self.max_file_size_bytes
        if not size_valid:
            errors.append(f"File size exceeds limit: {size_bytes / (1024 * 1024):.2f}MB (Max: {self.max_file_size_bytes / (1024 * 1024):.2f}MB)")

        # Verify magic number signatures
        magic_number_valid = self._check_magic_number(file_bytes, file_ext)
        if extension_valid and not magic_number_valid:
            errors.append("File signature (magic number) mismatch. The file content does not match its extension.")

        file_validation = FileValidationResult(
            extension_valid=extension_valid,
            size_valid=size_valid,
            magic_number_valid=magic_number_valid,
            size_bytes=size_bytes,
            file_ext=file_ext
        )

        # Early return if file check fails completely to avoid decoding crash
        if not extension_valid or size_bytes == 0:
            return ValidationScorecard(
                is_valid=False,
                file_validation=file_validation,
                image_validation=ImageValidationResult(False, False, 0, 0, 0),
                brain_detection=BrainMriDetectionResult(False, 0.0, "File check failed. Skipping image decoding."),
                quality_assessment=QualityAssessmentResult(0.0, False, 0.0, False, 0.0, False),
                duplicate_check=DuplicateCheckResult(False, "", None, None),
                errors=errors
            )

        # 2. Image validation (decoding checks)
        try:
            file_bytes_arr = np.frombuffer(file_bytes, dtype=np.uint8)
            img = cv2.imdecode(file_bytes_arr, cv2.IMREAD_COLOR)
        except Exception as e:
            img = None
            errors.append(f"Image parsing error: Failed to decode byte stream ({e})")

        corrupt_check_passed = img is not None
        if img is None:
            return ValidationScorecard(
                is_valid=False,
                file_validation=file_validation,
                image_validation=ImageValidationResult(False, False, 0, 0, 0),
                brain_detection=BrainMriDetectionResult(False, 0.0, "Corrupt image bytes. Decoding failed."),
                quality_assessment=QualityAssessmentResult(0.0, False, 0.0, False, 0.0, False),
                duplicate_check=DuplicateCheckResult(False, "", None, None),
                errors=errors
            )

        h, w, c = img.shape
        dimensions_valid = (self.min_resolution <= w <= self.max_resolution) and \
                           (self.min_resolution <= h <= self.max_resolution)
        if not dimensions_valid:
            errors.append(f"Resolution mismatch: {w}x{h} pixels. Allowed range: {self.min_resolution}x{self.min_resolution} to {self.max_resolution}x{self.max_resolution}")

        image_validation = ImageValidationResult(
            dimensions_valid=dimensions_valid,
            corrupt_check_passed=corrupt_check_passed,
            width=w,
            height=h,
            channels=c
        )

        # 3. Brain MRI Detector (Heuristics Analysis)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Border check (average intensity of outer 5% region should be close to black background)
        border_h = max(1, int(h * 0.05))
        border_w = max(1, int(w * 0.05))
        t_b = gray[0:border_h, :].mean()
        b_b = gray[h-border_h:h, :].mean()
        l_b = gray[:, 0:border_w].mean()
        r_b = gray[:, w-border_w:w].mean()
        border_mean = (t_b + b_b + l_b + r_b) / 4.0
        border_ok = bool(border_mean < 25.0)  # Background black threshold

        # Foreground check (brain tissue ratio)
        fg_mask = gray > 15  # threshold to isolate brain structure
        fg_pixels = fg_mask.sum()
        fg_ratio = fg_pixels / gray.size
        fg_ok = bool(0.12 <= fg_ratio <= 0.80)

        # Centering check (distance between foreground center of mass and slice center)
        if fg_pixels > 0:
            y_indices, x_indices = np.where(fg_mask)
            cy_fg = y_indices.mean()
            cx_fg = x_indices.mean()
            cy_geo = h / 2.0
            cx_geo = w / 2.0
            dist = np.sqrt((cx_fg - cx_geo)**2 + (cy_fg - cy_geo)**2)
            max_dist = np.sqrt(h**2 + w**2)
            centroid_score = max(0.0, 100.0 - (dist / (max_dist * 0.20)) * 100.0)
            centered_ok = bool(dist <= (max_dist * 0.20))
        else:
            centroid_score = 0.0
            centered_ok = False

        # Left-Right Symmetry check
        w_half = w // 2
        left_half = gray[:, :w_half]
        right_half = gray[:, w-w_half:]
        right_flipped = np.fliplr(right_half)
        mae = np.mean(np.abs(left_half.astype(float) - right_flipped.astype(float))) / 255.0
        symmetry_ok = bool(mae < 0.28)

        # Compile Brain MRI Detector score
        border_score = max(0.0, 100.0 - (border_mean / 25.0) * 100.0)
        fg_score = 100.0 if (0.12 <= fg_ratio <= 0.80) else (100.0 - min(abs(fg_ratio - 0.12), abs(fg_ratio - 0.80)) * 200.0)
        fg_score = max(0.0, fg_score)
        sym_score = max(0.0, 100.0 - (mae / 0.28) * 100.0)

        brain_confidence = float(0.3 * border_score + 0.3 * fg_score + 0.2 * centroid_score + 0.2 * sym_score)
        is_brain_mri = bool(border_ok and fg_ok and centered_ok and symmetry_ok and (brain_confidence >= 65.0))

        details = (
            f"Border background mean: {border_mean:.1f} (Threshold: <25.0); "
            f"Foreground tissue ratio: {fg_ratio:.2%} (Threshold: 12%-80%); "
            f"Centering deviation: {centroid_score:.1f}/100; "
            f"Horizontal asymmetry: {mae:.2f} MAE (Threshold: <0.28)"
        )
        
        if not is_brain_mri:
            errors.append(f"Brain MRI Detector failed: Input slice does not resemble standard brain MRI properties ({details}).")

        brain_detection = BrainMriDetectionResult(
            is_brain_mri=is_brain_mri,
            confidence_score=brain_confidence,
            details=details
        )

        # 4. Image Quality Assessment (QA)
        # Blurriness via Laplacian variance
        blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        blur_valid = blur_score >= self.min_blur_score
        if not blur_valid:
            errors.append(f"Image quality: High blur/motion artifacts detected (Variance: {blur_score:.1f}, Min required: {self.min_blur_score})")

        # Contrast via RMS contrast of foreground pixels
        fg_vals = gray[fg_mask]
        contrast_score = float(np.std(fg_vals)) if len(fg_vals) > 0 else 0.0
        contrast_valid = contrast_score >= self.min_contrast
        if not contrast_valid:
            errors.append(f"Image quality: Insufficient contrast variance (RMS Contrast: {contrast_score:.1f}, Min required: {self.min_contrast})")

        # Noise level via Estimated SNR
        bg_mask = gray <= 15
        bg_vals = gray[bg_mask]
        bg_std = np.std(bg_vals) if len(bg_vals) > 0 else 0.0
        fg_mean = np.mean(fg_vals) if len(fg_vals) > 0 else 0.0
        noise_score = float(fg_mean / bg_std) if bg_std > 0.5 else 50.0
        noise_valid = noise_score >= self.min_snr
        if not noise_valid:
            errors.append(f"Image quality: Excess noise level detected (SNR: {noise_score:.2f}, Min required: {self.min_snr})")

        quality_assessment = QualityAssessmentResult(
            contrast_score=contrast_score,
            contrast_valid=contrast_valid,
            blur_score=blur_score,
            blur_valid=blur_valid,
            noise_score=noise_score,
            noise_valid=noise_valid
        )

        # 5. Compile temporary Duplicate result (to be filled by use cases via SHA256 / perceptual average hash)
        duplicate_check = DuplicateCheckResult(
            is_duplicate=False,
            duplicate_hash=self._compute_sha256(file_bytes)
        )

        is_valid = bool(extension_valid and size_valid and magic_number_valid and \
                    dimensions_valid and is_brain_mri and blur_valid and \
                    contrast_valid and noise_valid)

        return ValidationScorecard(
            is_valid=is_valid,
            file_validation=file_validation,
            image_validation=image_validation,
            brain_detection=brain_detection,
            quality_assessment=quality_assessment,
            duplicate_check=duplicate_check,
            errors=errors
        )

    def _check_magic_number(self, file_bytes: bytes, file_ext: str) -> bool:
        ext = file_ext.lower().strip('.')
        if ext in ['png']:
            return file_bytes.startswith(b'\x89PNG\r\n\x1a\n')
        elif ext in ['jpg', 'jpeg']:
            return file_bytes.startswith(b'\xff\xd8\xff')
        elif ext in ['tif', 'tiff']:
            return file_bytes.startswith(b'II*\x00') or file_bytes.startswith(b'MM\x00*')
        return False

    def _compute_sha256(self, file_bytes: bytes) -> str:
        return hashlib.sha256(file_bytes).hexdigest()

    def compute_perceptual_hash(self, file_bytes: bytes) -> str:
        """Computes a 64-bit average perceptual hash (aHash) of decoded grayscale image."""
        try:
            file_bytes_arr = np.frombuffer(file_bytes, dtype=np.uint8)
            img = cv2.imdecode(file_bytes_arr, cv2.IMREAD_GRAYSCALE)
            if img is None:
                return ""
            small = cv2.resize(img, (8, 8), interpolation=cv2.INTER_AREA)
            avg = small.mean()
            bits = small > avg
            phash_val = ""
            for row in bits:
                val = 0
                for bit in row:
                    val = (val << 1) | int(bit)
                phash_val += f"{val:02x}"
            return phash_val
        except Exception:
            return ""

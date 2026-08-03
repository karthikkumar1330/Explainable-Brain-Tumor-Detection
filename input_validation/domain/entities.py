from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any


@dataclass(frozen=True)
class FileValidationResult:
    """Quantitative results of file format, size, and magic numbers verification."""
    extension_valid: bool
    size_valid: bool
    magic_number_valid: bool
    size_bytes: int
    file_ext: str


@dataclass(frozen=True)
class ImageValidationResult:
    """Quantitative results of image parsing integrity, dimension ranges, and corrupt checks."""
    dimensions_valid: bool
    corrupt_check_passed: bool
    width: int
    height: int
    channels: int


@dataclass(frozen=True)
class BrainMriDetectionResult:
    """Heuristic assessment results of whether the input image conforms to a brain MRI slice structure."""
    is_brain_mri: bool
    confidence_score: float  # Score from 0.0 to 100.0
    details: str             # Description of passing/failing criteria (borders, foreground, symmetry, centering)


@dataclass(frozen=True)
class QualityAssessmentResult:
    """Quantitative scores of blurriness, contrast variance, and estimated signal-to-noise ratio."""
    contrast_score: float   # RMS contrast of foreground
    contrast_valid: bool
    blur_score: float       # Laplacian variance
    blur_valid: bool
    noise_score: float      # Estimated SNR (foreground mean / background std)
    noise_valid: bool


@dataclass(frozen=True)
class DuplicateCheckResult:
    """Perceptual and cryptographic duplicate detection results against database history."""
    is_duplicate: bool
    duplicate_hash: str
    duplicate_patient_id: Optional[str] = None
    duplicate_scan_date: Optional[str] = None


@dataclass(frozen=True)
class ValidationScorecard:
    """Aggregated clinical input validation scorecard compiling all QA dimensions."""
    is_valid: bool
    file_validation: FileValidationResult
    image_validation: ImageValidationResult
    brain_detection: BrainMriDetectionResult
    quality_assessment: QualityAssessmentResult
    duplicate_check: DuplicateCheckResult
    errors: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the scorecard into a dictionary for JSON reporting and REST APIs."""
        return asdict(self)

from typing import Dict


class ModelVersionManager:
    """Manages active deep learning model checkpoints, versions, training dates, and configurations."""

    def __init__(self) -> None:
        self.classification_version = "EfficientNet-B0 Classifier (v2.1-calibrated)"
        self.segmentation_version = "UNeXt Segments Extractor (v2.3-postprocessed)"
        self.calibration_version = "Platt Probability Calibration Scaling (v1.0)"
        self.training_date_cls = "2026-08-01"
        self.training_date_seg = "2026-08-02"
        self.checkpoint_version = "v2.6-clinical-qa-monitor"

    def get_version_details(self) -> Dict[str, str]:
        """Returns details about classification, segmentation, calibration, and training cycles."""
        return {
            "classification_version": self.classification_version,
            "segmentation_version": self.segmentation_version,
            "calibration_version": self.calibration_version,
            "classification_training_date": self.training_date_cls,
            "segmentation_training_date": self.training_date_seg,
            "checkpoint_version": self.checkpoint_version,
        }

import os
import unittest
import numpy as np
import cv2
import tempfile
import shutil
from input_validation.infrastructure.validators import OpenCVMriValidator
from input_validation.application.use_cases import ValidateMriUploadUseCase
from persistence.infrastructure.repository import SQLitePersistenceRepository


class TestMriInputValidation(unittest.TestCase):

    def setUp(self):
        # Create temporary workspace directories
        self.test_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.test_dir, "test_clinical_reports.db")
        
        # Initialize DB repository
        self.db_repo = SQLitePersistenceRepository(db_path=self.db_path)
        self.db_repo.initialize_db()
        
        # Setup validator and use case
        self.validator = OpenCVMriValidator()
        self.use_case = ValidateMriUploadUseCase(validator=self.validator, db_path=self.db_path)

    def tearDown(self):
        # Cleanup temporary workspace
        shutil.rmtree(self.test_dir)

    def _create_mock_brain_mri_slice(self) -> bytes:
        """Generates a mock brain MRI slice programmatically (centered ellipse with tissue texture)."""
        # Create black canvas
        img = np.zeros((256, 256, 3), dtype=np.uint8)
        
        # Draw brain-like centered ellipse
        # Center = (128, 128), Axes = (60, 80)
        mask = np.zeros((256, 256), dtype=np.uint8)
        cv2.ellipse(mask, (128, 128), (60, 80), 0, 0, 360, 255, -1)
        
        # Generate some synthetic symmetric texture inside the brain region
        np.random.seed(42)
        half_width = 128
        left_texture = np.random.randint(80, 200, size=(256, half_width), dtype=np.uint8)
        right_texture = np.fliplr(left_texture)
        texture = np.hstack([left_texture, right_texture])
        
        # Apply texture inside mask
        brain_tissue = cv2.bitwise_and(texture, texture, mask=mask)
        
        # Blur the interior slightly to match soft tissues
        brain_tissue_blurred = cv2.GaussianBlur(brain_tissue, (5, 5), 0)
        brain_tissue = np.where(mask > 0, brain_tissue_blurred, 0)
        
        # Convert to BGR color image (standard MRI encoding)
        img[:, :, 0] = brain_tissue  # B
        img[:, :, 1] = brain_tissue  # G
        img[:, :, 2] = brain_tissue  # R
        
        # Add high frequency details inside the brain to pass contrast & blur check
        # e.g., draw ventricles
        cv2.ellipse(img, (128, 110), (15, 8), 0, 0, 360, (40, 40, 40), -1)
        cv2.ellipse(img, (128, 140), (12, 6), 0, 0, 360, (40, 40, 40), -1)
        
        # Encode to PNG bytes
        _, buf = cv2.imencode(".png", img)
        return buf.tobytes()

    def _create_mock_non_brain_image(self) -> bytes:
        """Generates a non-brain natural-like photo (solid white canvas with text)."""
        img = np.ones((256, 256, 3), dtype=np.uint8) * 240
        # Draw some text or boxes to make it non-blank but full of border pixels
        cv2.putText(img, "MEDICAL INC", (50, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        _, buf = cv2.imencode(".png", img)
        return buf.tobytes()

    def test_valid_brain_mri_passes(self):
        """Verifies that a valid brain MRI image passes all scorecard conditions."""
        file_bytes = self._create_mock_brain_mri_slice()
        filepath = os.path.join(self.test_dir, "valid_mri.png")
        with open(filepath, "wb") as f:
            f.write(file_bytes)
            
        scorecard = self.use_case.execute(filepath=filepath, file_bytes=file_bytes, filename="valid_mri.png")
        
        self.assertTrue(scorecard.is_valid, f"Expected validation to pass. Errors: {scorecard.errors}")
        self.assertTrue(scorecard.file_validation.extension_valid)
        self.assertTrue(scorecard.file_validation.magic_number_valid)
        self.assertTrue(scorecard.image_validation.corrupt_check_passed)
        self.assertTrue(scorecard.brain_detection.is_brain_mri)
        self.assertTrue(scorecard.quality_assessment.contrast_valid)
        self.assertTrue(scorecard.quality_assessment.blur_valid)
        self.assertTrue(scorecard.quality_assessment.noise_valid)

    def test_invalid_extension_rejected(self):
        """Verifies that files with unapproved extensions are rejected."""
        file_bytes = b"fake pdf file contents"
        filepath = os.path.join(self.test_dir, "document.pdf")
        with open(filepath, "wb") as f:
            f.write(file_bytes)
            
        scorecard = self.use_case.execute(filepath=filepath, file_bytes=file_bytes, filename="document.pdf")
        self.assertFalse(scorecard.is_valid)
        self.assertFalse(scorecard.file_validation.extension_valid)

    def test_magic_number_mismatch_rejected(self):
        """Verifies that a text file renamed to .png is caught by magic bytes checker."""
        file_bytes = b"Just plain text pretending to be a PNG"
        filepath = os.path.join(self.test_dir, "fake_image.png")
        with open(filepath, "wb") as f:
            f.write(file_bytes)
            
        scorecard = self.use_case.execute(filepath=filepath, file_bytes=file_bytes, filename="fake_image.png")
        self.assertFalse(scorecard.is_valid)
        self.assertFalse(scorecard.file_validation.magic_number_valid)
        self.assertFalse(scorecard.image_validation.corrupt_check_passed)

    def test_non_brain_image_rejected(self):
        """Verifies that non-brain images (e.g. photos/empty canvases) are rejected by Brain MRI Detector."""
        file_bytes = self._create_mock_non_brain_image()
        filepath = os.path.join(self.test_dir, "label_photo.png")
        with open(filepath, "wb") as f:
            f.write(file_bytes)
            
        scorecard = self.use_case.execute(filepath=filepath, file_bytes=file_bytes, filename="label_photo.png")
        self.assertFalse(scorecard.is_valid)
        self.assertFalse(scorecard.brain_detection.is_brain_mri)

    def test_duplicate_check_blocks_inference(self):
        """Verifies that uploading the exact same scan twice sets is_valid=False and flags duplicate."""
        file_bytes = self._create_mock_brain_mri_slice()
        filepath = os.path.join(self.test_dir, "first_upload.png")
        with open(filepath, "wb") as f:
            f.write(file_bytes)
            
        # 1. Run first upload and link to a dummy prediction in the database
        scorecard_1 = self.use_case.execute(filepath=filepath, file_bytes=file_bytes, filename="first_upload.png")
        self.assertTrue(scorecard_1.is_valid)
        
        # Save a fake patient, scan, and prediction to enable database duplicate joins
        conn = self.db_repo._get_connection()
        try:
            with conn:
                conn.execute(
                    "INSERT INTO patients (patient_id, name, age, gender, created_at) VALUES (?, ?, ?, ?, ?)",
                    ("PAT_VAL_1", "Test Subject", 45, "Male", "2026-08-03 12:00:00")
                )
                conn.execute(
                    "INSERT INTO mri_scans (patient_id, image_path, pixel_spacing_mm, ref_physician, scan_date, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                    ("PAT_VAL_1", filepath, 1.0, "Dr. QA", "2026-08-03", "2026-08-03 12:00:00")
                )
                scan_id = conn.execute("SELECT id FROM mri_scans LIMIT 1;").fetchone()[0]
                conn.execute(
                    "INSERT INTO predictions (scan_id, predicted_class, confidence_score, prob_glioma, prob_meningioma, prob_pituitary, prob_no_tumor, tumor_pixel_count, tumor_area_mm2, tumor_percentage_brain, tumor_percentage_image, estimated_brain_pixel_count, rule_based_severity, severity_rule_description, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (scan_id, "Glioma", 0.95, 0.95, 0.0, 0.0, 0.05, 1000, 1000.0, 5.0, 2.0, 20000, "HIGH", "Focal lesion", "2026-08-03 12:00:00")
                )
                pred_id = conn.execute("SELECT id FROM predictions LIMIT 1;").fetchone()[0]
        finally:
            conn.close()

        # Update the scorecard record in database with the created prediction_id
        # (Same logic that app.py and routes.py use after running prediction model)
        p_hash = self.validator.compute_perceptual_hash(file_bytes)
        import json
        self.db_repo.save_validation_scorecard(
            file_hash=scorecard_1.duplicate_check.duplicate_hash,
            p_hash=p_hash,
            is_valid=True,
            scorecard_json=json.dumps(scorecard_1.to_dict()),
            prediction_id=pred_id
        )

        # 2. Run duplicate upload check
        filepath_dup = os.path.join(self.test_dir, "duplicate_upload.png")
        with open(filepath_dup, "wb") as f:
            f.write(file_bytes)
            
        scorecard_2 = self.use_case.execute(filepath=filepath_dup, file_bytes=file_bytes, filename="duplicate_upload.png")
        
        self.assertFalse(scorecard_2.is_valid)
        self.assertTrue(scorecard_2.duplicate_check.is_duplicate)
        self.assertEqual(scorecard_2.duplicate_check.duplicate_patient_id, "PAT_VAL_1")
        self.assertEqual(scorecard_2.duplicate_check.duplicate_scan_date, "2026-08-03")
        self.assertTrue(any("Duplicate scan detected" in e for e in scorecard_2.errors))


if __name__ == "__main__":
    unittest.main()

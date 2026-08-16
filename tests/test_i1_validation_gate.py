import os
import io
import time
import datetime
import shutil
import unittest
import numpy as np
import cv2
import json
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

# Setup database path for test execution
os.environ["DB_PATH"] = os.path.abspath("outputs/test_i1_validation_gate_clean.db")
os.environ["DISABLE_TEST_AUTO_ASSIGN"] = "1"

import pytest

@pytest.fixture(autouse=True)
def manage_auto_assign_env():
    os.environ["DISABLE_TEST_AUTO_ASSIGN"] = "1"
    yield
    os.environ["DISABLE_TEST_AUTO_ASSIGN"] = "1"

from run_api import app
import api.infrastructure.routes as api_routes
import api.routes.auth_routes as auth_routes
from security.domain.entities import Role, User, SecurityAuditLog
from security.infrastructure.jwt_service import JWTService
from security.infrastructure.repository import SQLiteUserRepository
from persistence.infrastructure.repository import SQLitePersistenceRepository
from input_validation.infrastructure.validators import OpenCVMriValidator
from input_validation.application.use_cases import ValidateMriUploadUseCase
from input_validation.domain.entities import ValidationScorecard


class TestMriQualityValidationGate(unittest.TestCase):
    """Phase I1 Quality Validation Gate, central policy, and authorization override tests."""

    def setUp(self):
        self.db_path = os.environ.get("DB_PATH", "outputs/test_i1_validation_gate.db")
        if os.path.exists(self.db_path):
            try:
                os.remove(self.db_path)
            except Exception:
                pass

        # Force routes database paths to point to test database
        api_routes.DEFAULT_DB_PATH = self.db_path
        auth_routes.DEFAULT_DB_PATH = self.db_path

        # Initialize persistence database
        self.db_repo = SQLitePersistenceRepository(db_path=self.db_path)
        self.db_repo.initialize_db()

        # Initialize security tables
        self.user_repo = SQLiteUserRepository(db_path=self.db_path)
        self.user_repo.initialize_security_tables()

        # Drop auto-assignment triggers and clean assignments to test explicit overrides
        conn = sqlite3_connect(self.db_path)
        try:
            with conn:
                conn.execute("DROP TRIGGER IF EXISTS auto_assign_doctor_on_insert;")
                conn.execute("DROP TRIGGER IF EXISTS auto_assign_patient_on_insert;")
                conn.execute("DELETE FROM doctor_patient_assignments;")
        except Exception:
            pass
        finally:
            conn.close()

        # Bootstrap admin user
        self.admin = self.user_repo.bootstrap_admin()

        # Create doctors
        self.doctor = self.user_repo.create_user(User(
            id=None, uuid="dr-gates-uuid", email="dr.gates@aurascan.ai",
            password_hash="hashed_pw_123", full_name="Dr. Validation Gates", role=Role.DOCTOR,
            is_verified=True, is_active=True
        ))
        self.other_doctor = self.user_repo.create_user(User(
            id=None, uuid="dr-stranger-uuid", email="dr.stranger@aurascan.ai",
            password_hash="hashed_pw_123", full_name="Dr. Stranger Danger", role=Role.DOCTOR,
            is_verified=True, is_active=True
        ))

        # Create patient user
        self.patient_user = self.user_repo.create_user(User(
            id=None, uuid="pat-doe-uuid", email="patient.doe@aurascan.ai",
            password_hash="hashed_pw_123", full_name="John Patient Doe", role=Role.PATIENT,
            is_verified=True, is_active=True
        ))

        jwt_svc = JWTService()
        self.doctor_token = jwt_svc.create_access_token(
            user_uuid=self.doctor.uuid,
            user_id=self.doctor.id,
            email=self.doctor.email,
            role=self.doctor.role
        )
        self.other_doctor_token = jwt_svc.create_access_token(
            user_uuid=self.other_doctor.uuid,
            user_id=self.other_doctor.id,
            email=self.other_doctor.email,
            role=self.other_doctor.role
        )
        self.patient_token = jwt_svc.create_access_token(
            user_uuid=self.patient_user.uuid,
            user_id=self.patient_user.id,
            email=self.patient_user.email,
            role=self.patient_user.role
        )

        self.doctor_headers = {"Authorization": f"Bearer {self.doctor_token}"}
        self.other_doctor_headers = {"Authorization": f"Bearer {self.other_doctor_token}"}
        self.patient_headers = {"Authorization": f"Bearer {self.patient_token}"}

        # Set up assignments in DB
        conn = sqlite3_connect(self.db_path)
        try:
            with conn:
                # Add patient
                conn.execute(
                    "INSERT OR IGNORE INTO patients (patient_id, name, age, gender, created_at) VALUES (?, ?, ?, ?, ?)",
                    ("PAT_I1_01", "John Patient Doe", 45, "Male", "2026-08-04 12:00:00")
                )
                # Assign Dr. Gates to patient PAT_I1_01
                conn.execute(
                    "INSERT OR IGNORE INTO doctor_patient_assignments (doctor_id, patient_id, created_at) VALUES (?, ?, ?)",
                    (self.doctor.id, "PAT_I1_01", "2026-08-04 12:00:00")
                )
        finally:
            conn.close()

        # Setup Test Client
        self.test_client_ctx = TestClient(app)
        self.client = self.test_client_ctx.__enter__()

    def tearDown(self):
        self.test_client_ctx.__exit__()
        if os.path.exists(self.db_path):
            try:
                os.remove(self.db_path)
            except Exception:
                pass

        temp_dir = "outputs/temp_uploads"
        if os.path.exists(temp_dir):
            for filename in os.listdir(temp_dir):
                if filename.startswith("upload_") or filename.startswith("batch_"):
                    try:
                        os.remove(os.path.join(temp_dir, filename))
                    except Exception:
                        pass

    def _create_mock_mri_bytes(self, label_text: str = "BRAIN_MRI", seed: int = 42, axes: tuple = (60, 80), brightness_factor: float = 1.0, contrast_val: float = None, blur_kernel: int = None, fill_val: int = None) -> bytes:
        """Generates a mock brain MRI slice with custom attributes to trigger specific quality warnings."""
        img = np.zeros((256, 256, 3), dtype=np.uint8)
        mask = np.zeros((256, 256), dtype=np.uint8)
        cv2.ellipse(mask, (128, 128), axes, 0, 0, 360, 255, -1)

        if fill_val is not None:
            # Solid color interior (low contrast/entropy)
            brain_tissue = np.where(mask > 0, fill_val, 0).astype(np.uint8)
        else:
            np.random.seed(seed)
            texture = np.random.randint(90, 210, size=(256, 256), dtype=np.uint8)
            brain_tissue = cv2.bitwise_and(texture, texture, mask=mask)
            brain_tissue_blurred = cv2.GaussianBlur(brain_tissue, (5, 5), 0)
            brain_tissue = np.where(mask > 0, brain_tissue_blurred, 0).astype(np.uint8)

        if contrast_val is not None:
            # Re-scale contrast manually
            mean_val = brain_tissue[mask > 0].mean() if mask.sum() > 0 else 128
            brain_tissue = np.where(mask > 0, mean_val + (brain_tissue - mean_val) * contrast_val, 0)
            brain_tissue = np.clip(brain_tissue, 0, 255).astype(np.uint8)

        if brightness_factor != 1.0:
            brain_tissue = (brain_tissue * brightness_factor).clip(0, 255).astype(np.uint8)

        img[:, :, 0] = brain_tissue
        img[:, :, 1] = brain_tissue
        img[:, :, 2] = brain_tissue

        # Draw ventricles only if not solid color fill
        if fill_val is None:
            cv2.ellipse(img, (128, 110), (15, 8), 0, 0, 360, (40, 40, 40), -1)
            cv2.ellipse(img, (128, 140), (12, 6), 0, 0, 360, (40, 40, 40), -1)

        if label_text:
            cv2.putText(img, label_text, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        if blur_kernel is not None:
            img = cv2.GaussianBlur(img, (blur_kernel, blur_kernel), 0)

        _, buf = cv2.imencode(".png", img)
        return buf.tobytes()

    def test_valid_mri_passes(self):
        """Verify that a normal valid MRI scan passes validation automatically."""
        file_bytes = self._create_mock_mri_bytes()
        validator = OpenCVMriValidator()
        scorecard = validator.validate_file("mri.png", file_bytes, "mri.png")
        self.assertTrue(scorecard.is_valid)
        self.assertEqual(len(scorecard.errors), 0)
        self.assertEqual(len(scorecard.quality_assessment.warnings), 0)

    def test_empty_file_rejected(self):
        """Verify that an empty file is caught and classified as critical failure."""
        validator = OpenCVMriValidator()
        scorecard = validator.validate_file("empty.png", b"", "empty.png")
        self.assertFalse(scorecard.is_valid)
        self.assertIn("File check failed", scorecard.brain_detection.details)

    def test_corrupt_image_rejected(self):
        """Verify that a corrupted image file fails validation with a critical error."""
        corrupt_bytes = b"\x89PNG\r\n\x1a\n" + b"some corrupted content here"
        validator = OpenCVMriValidator()
        scorecard = validator.validate_file("corrupt.png", corrupt_bytes, "corrupt.png")
        self.assertFalse(scorecard.is_valid)
        self.assertTrue(any("corrupt" in e.lower() or "decoding" in e.lower() for e in scorecard.errors))

    def test_unsupported_format_rejected(self):
        """Verify that files with unapproved extensions or magic numbers are rejected."""
        txt_bytes = b"This is a plain text file pretending to be png."
        validator = OpenCVMriValidator()
        scorecard = validator.validate_file("test.png", txt_bytes, "test.png")
        self.assertFalse(scorecard.is_valid)
        self.assertTrue(any("signature" in e.lower() or "magic" in e.lower() for e in scorecard.errors))

    def test_invalid_dimensions(self):
        """Verify that extremely small resolution scans fail dimensions validation."""
        # Create a tiny 32x32 image
        img = np.zeros((32, 32, 3), dtype=np.uint8)
        _, buf = cv2.imencode(".png", img)
        validator = OpenCVMriValidator()
        scorecard = validator.validate_file("tiny.png", buf.tobytes(), "tiny.png")
        self.assertFalse(scorecard.is_valid)
        self.assertTrue(any("resolution" in e.lower() or "pixel" in e.lower() for e in scorecard.errors))

    def test_extremely_dark_image(self):
        """Verify that extremely dark images trigger mean intensity warning."""
        # Use low brightness factor (0.05) to trigger dark warning
        file_bytes = self._create_mock_mri_bytes(brightness_factor=0.05)
        validator = OpenCVMriValidator(min_mean_intensity=15.0) # configure custom threshold for testing
        scorecard = validator.validate_file("dark.png", file_bytes, "dark.png")
        self.assertFalse(scorecard.is_valid)
        self.assertTrue(any("mean intensity" in w.lower() for w in scorecard.quality_assessment.warnings))

    def test_extremely_bright_image(self):
        """Verify that extremely bright images trigger maximum mean intensity warning."""
        # Draw a white ellipse
        file_bytes = self._create_mock_mri_bytes(fill_val=250)
        validator = OpenCVMriValidator(max_mean_intensity=40.0) # configure custom threshold
        scorecard = validator.validate_file("bright.png", file_bytes, "bright.png")
        self.assertFalse(scorecard.is_valid)
        self.assertTrue(any("mean intensity" in w.lower() for w in scorecard.quality_assessment.warnings))

    def test_low_contrast(self):
        """Verify that low contrast scans trigger contrast warning."""
        file_bytes = self._create_mock_mri_bytes(label_text="", fill_val=140)
        validator = OpenCVMriValidator(min_contrast=15.0) # configure custom threshold
        scorecard = validator.validate_file("low_contrast.png", file_bytes, "low_contrast.png")
        self.assertFalse(scorecard.is_valid)
        self.assertTrue(any("contrast" in w.lower() for w in scorecard.quality_assessment.warnings))

    def test_excessive_blur(self):
        """Verify that blurred images fail blur score threshold check."""
        file_bytes = self._create_mock_mri_bytes(blur_kernel=21)
        validator = OpenCVMriValidator(min_blur_score=50.0) # configure custom threshold
        scorecard = validator.validate_file("blur.png", file_bytes, "blur.png")
        self.assertFalse(scorecard.is_valid)
        self.assertTrue(any("blur" in w.lower() for w in scorecard.quality_assessment.warnings))

    def test_entropy_indicator(self):
        """Verify that flat/low-information images trigger Shannon Entropy warning."""
        file_bytes = self._create_mock_mri_bytes(fill_val=100)
        validator = OpenCVMriValidator(min_entropy=2.0) # configure custom threshold
        scorecard = validator.validate_file("flat.png", file_bytes, "flat.png")
        self.assertFalse(scorecard.is_valid)
        self.assertTrue(any("entropy" in w.lower() for w in scorecard.quality_assessment.warnings))

    def test_multiple_simultaneous_warnings(self):
        """Verify that multiple quality issues are aggregated in warnings scorecard."""
        # Low contrast, flat, dark
        file_bytes = self._create_mock_mri_bytes(fill_val=60)
        validator = OpenCVMriValidator(min_mean_intensity=25.0, min_contrast=5.0, min_entropy=2.0)
        scorecard = validator.validate_file("multi.png", file_bytes, "multi.png")
        self.assertFalse(scorecard.is_valid)
        warnings = scorecard.quality_assessment.warnings
        self.assertTrue(len(warnings) >= 2)

    def test_default_warning_blocks_upload(self):
        """Verify that an upload with quality warnings is blocked by default."""
        file_bytes = self._create_mock_mri_bytes()
        # Set extremely high blur requirement to force warning
        with patch("input_validation.infrastructure.validators.OpenCVMriValidator") as mock_val_cls:
            mock_val = OpenCVMriValidator(min_blur_score=99999.0)
            mock_val_cls.return_value = mock_val

            response = self.client.post(
                "/api/upload",
                files={"file": ("dark_mri.png", file_bytes, "image/png")},
                headers=self.doctor_headers
            )
            self.assertEqual(response.status_code, 400)
            data = response.json()
            self.assertEqual(data["detail"]["message"], "MRI Input Validation Failed")
            self.assertFalse(data["detail"]["scorecard"]["is_valid"])
            self.assertTrue(any("blur" in e for e in data["detail"]["errors"]))

    def test_authorized_doctor_override_succeeds(self):
        """Verify that an authorized doctor override succeeds and bypasses validation warning."""
        file_bytes = self._create_mock_mri_bytes()
        with patch("input_validation.infrastructure.validators.OpenCVMriValidator") as mock_val_cls:
            mock_val = OpenCVMriValidator(min_blur_score=99999.0)
            mock_val_cls.return_value = mock_val

            payload = {
                "confirm_override": "true",
                "override_reason": "Bypassing high blur warning as structures are clearly visible",
                "patient_id": "PAT_I1_01"
            }
            response = self.client.post(
                "/api/upload",
                files={"file": ("dark_mri.png", file_bytes, "image/png")},
                data=payload,
                headers=self.doctor_headers
            )
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertTrue(data["scorecard"]["is_valid"])
            self.assertTrue(data["scorecard"]["override_metadata"]["overridden"])
            self.assertEqual(data["scorecard"]["override_metadata"]["override_reason"], payload["override_reason"])

    def test_unauthorized_override_rejected(self):
        """Verify that a role without override privileges (e.g. PATIENT) is rejected."""
        file_bytes = self._create_mock_mri_bytes()
        payload = {
            "confirm_override": "true",
            "override_reason": "Override attempted by patient",
            "patient_id": "PAT_I1_01"
        }
        # FastAPI Depends(require_roles) handles role restrictions
        response = self.client.post(
            "/api/upload",
            files={"file": ("mri.png", file_bytes, "image/png")},
            data=payload,
            headers=self.patient_headers
        )
        self.assertEqual(response.status_code, 403) # RBAC rejects upload endpoint entirely for patients

    def test_wrong_doctor_patient_assignment_rejected(self):
        """Verify that a doctor trying to override warning for an unassigned patient is rejected (HTTP 403)."""
        file_bytes = self._create_mock_mri_bytes()
        with patch("input_validation.infrastructure.validators.OpenCVMriValidator") as mock_val_cls:
            mock_val = OpenCVMriValidator(min_blur_score=99999.0)
            mock_val_cls.return_value = mock_val

            payload = {
                "confirm_override": "true",
                "override_reason": "Override by unassigned doctor",
                "patient_id": "PAT_I1_01"
            }
            # Other Doctor has no assignment to PAT_I1_01
            response = self.client.post(
                "/api/upload",
                files={"file": ("mri.png", file_bytes, "image/png")},
                data=payload,
                headers=self.other_doctor_headers
            )
            self.assertEqual(response.status_code, 403)
            self.assertIn("Access denied: Doctor not assigned to patient", response.json()["detail"])

    def test_override_reason_required(self):
        """Verify that override requests without a valid or too short reason are rejected."""
        file_bytes = self._create_mock_mri_bytes()
        with patch("input_validation.infrastructure.validators.OpenCVMriValidator") as mock_val_cls:
            mock_val = OpenCVMriValidator(min_blur_score=99999.0)
            mock_val_cls.return_value = mock_val

            payload = {
                "confirm_override": "true",
                "override_reason": "short", # length < 10
                "patient_id": "PAT_I1_01"
            }
            response = self.client.post(
                "/api/upload",
                files={"file": ("mri.png", file_bytes, "image/png")},
                data=payload,
                headers=self.doctor_headers
            )
            self.assertEqual(response.status_code, 400)
            self.assertIn("Override rejected: Valid override reason", response.json()["detail"])

    def test_critical_failure_cannot_be_overridden(self):
        """Verify that critical failures (like invalid format) cannot be bypassed by doctor override."""
        txt_bytes = b"Just plain text file content."
        payload = {
            "confirm_override": "true",
            "override_reason": "Clinical override attempted on invalid format",
            "patient_id": "PAT_I1_01"
        }
        response = self.client.post(
            "/api/upload",
            files={"file": ("test.png", txt_bytes, "image/png")},
            data=payload,
            headers=self.doctor_headers
        )
        self.assertEqual(response.status_code, 400)
        data = response.json()
        self.assertFalse(data["detail"]["scorecard"]["is_valid"])

    def test_audit_event_generated(self):
        """Verify that a successful override generates an audit log in sqlite database."""
        file_bytes = self._create_mock_mri_bytes()
        with patch("input_validation.infrastructure.validators.OpenCVMriValidator") as mock_val_cls:
            mock_val = OpenCVMriValidator(min_blur_score=99999.0)
            mock_val_cls.return_value = mock_val

            payload = {
                "confirm_override": "true",
                "override_reason": "Bypassing high blur warning for auditing verification",
                "patient_id": "PAT_I1_01"
            }
            response = self.client.post(
                "/api/upload",
                files={"file": ("dark_mri.png", file_bytes, "image/png")},
                data=payload,
                headers=self.doctor_headers
            )
            self.assertEqual(response.status_code, 200)

            # Query security audit log directly
            conn = sqlite3_connect(self.db_path)
            try:
                row = conn.execute(
                    "SELECT * FROM security_audit_logs WHERE event_type = ? ORDER BY id DESC LIMIT 1;",
                    ("MRI_QUALITY_OVERRIDE",)
                ).fetchone()
                self.assertIsNotNone(row)
                self.assertEqual(row["status"], "SUCCESS")
                self.assertIn("successfully bypassed quality warning", row["details"])
                self.assertIn("auditing verification", row["details"])
            finally:
                conn.close()

    def test_batch_mixed_quality_images(self):
        """Verify batch ingestion handles mixed quality scans (pass, warning override, blocked error) independently."""
        file1 = self._create_mock_mri_bytes("SLICE_PASS", seed=100) # Valid
        file2 = self._create_mock_mri_bytes("SLICE_WARN", seed=200) # Fails max_mean_intensity if customized
        file3 = b"corrupted file content" # Critical error file

        # We simulate the validation execution for files inside routes
        # file1 -> passes
        # file2 -> warning (mean_intensity out of range, overridden by payload reason)
        # file3 -> critical failure (corruption)

        with patch("input_validation.infrastructure.validators.OpenCVMriValidator") as mock_val_cls:
            # Create a validator instance where file2 triggers warning
            mock_val = OpenCVMriValidator(max_mean_intensity=1.0) # force warning on all except file1/file3 handled specially
            mock_val_cls.return_value = mock_val

            payload = {
                "patient_id": "PAT_I1_01",
                "name": "John Patient Doe",
                "age": 45,
                "gender": "Male",
                "confirm_override": "true",
                "override_reason": "Overriding quality warning for slice 2"
            }
            files = [
                ("files", ("slice1.png", file1, "image/png")),
                ("files", ("slice2.png", file2, "image/png")),
                ("files", ("slice3.png", file3, "image/png"))
            ]

            response = self.client.post(
                "/api/report/batch",
                data=payload,
                files=files,
                headers=self.doctor_headers
            )

            self.assertEqual(response.status_code, 200)
            data = response.json()
            stats = data["stats"]
            self.assertEqual(stats["total_files"], 3)
            # slice1 -> passes validation (valid)
            # slice2 -> overridden validation (valid)
            # slice3 -> corrupted critical block (failed)
            # Total successful = 2
            self.assertEqual(stats["successful"], 2)
            self.assertEqual(stats["failed"], 1)

            results = data["results"]
            failed_item = [r for r in results if r["filename"] == "slice3.png"][0]
            self.assertEqual(failed_item["status"], "FAILED")
            self.assertEqual(failed_item["error_code"], "UNSUPPORTED_FORMAT")
            self.assertTrue("signature" in failed_item["error_message"].lower() or "decoding" in failed_item["error_message"].lower())


def sqlite3_connect(db_path):
    import sqlite3
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


if __name__ == "__main__":
    unittest.main()

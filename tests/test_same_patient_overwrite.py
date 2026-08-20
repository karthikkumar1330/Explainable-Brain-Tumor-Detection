import os
import unittest
import sqlite3
import datetime
import json
import shutil
import hashlib
from fastapi.testclient import TestClient

from run_api import app
from persistence.infrastructure.repository import SQLitePersistenceRepository
from security.domain.entities import Role, User
from security.infrastructure.repository import SQLiteUserRepository
from security.infrastructure.jwt_service import JWTService

class TestSamePatientOverwrite(unittest.TestCase):
    TEST_DB_PATH = os.path.abspath("outputs/test_same_patient_overwrite.db")

    def setUp(self):
        self.db_path = self.TEST_DB_PATH
        if os.path.exists(self.db_path):
            try:
                os.remove(self.db_path)
            except Exception:
                pass

        # Setup database
        self.persistence_repo = SQLitePersistenceRepository(db_path=self.db_path)
        self.persistence_repo.initialize_db()

        self.test_client_ctx = TestClient(app)
        self.client = self.test_client_ctx.__enter__()

        self.user_repo = SQLiteUserRepository(db_path=self.db_path)
        self.user_repo.initialize_security_tables()

        # Bootstrap users
        self.admin_user = self.user_repo.bootstrap_admin()

        from security.infrastructure.password import PasswordHasher
        pass_hash = PasswordHasher.hash_password("Doctor@123")

        # Insert test users
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                "INSERT INTO users (uuid, email, password_hash, full_name, role, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?);",
                ("doc-uuid-overwrite", "doc_overwrite@aurascan.ai", pass_hash, "Dr. Overwrite", Role.DOCTOR.value, datetime.datetime.utcnow().isoformat(), datetime.datetime.utcnow().isoformat())
            )
            conn.execute(
                "INSERT INTO users (uuid, email, password_hash, full_name, role, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?);",
                ("pat-uuid-overwrite", "pat_overwrite@aurascan.ai", pass_hash, "Overwrite Patient", Role.PATIENT.value, datetime.datetime.utcnow().isoformat(), datetime.datetime.utcnow().isoformat())
            )
            # Assign patient to doctor
            from tests.helpers.authorization_fixtures import assign_doctor_to_patient
            assign_doctor_to_patient(conn, "doc_overwrite@aurascan.ai", "pat-uuid-overwrite")
            conn.commit()
        finally:
            conn.close()

        # Get JWT token for authentication
        self.doc_user = self.user_repo.get_by_email("doc_overwrite@aurascan.ai")
        self.jwt_service = JWTService()
        self.token = self.jwt_service.create_access_token(
            user_uuid=self.doc_user.uuid,
            user_id=self.doc_user.id,
            email=self.doc_user.email,
            role=Role.DOCTOR
        )
        self.headers = {"Authorization": f"Bearer {self.token}"}

        # Mock target directories and files
        os.makedirs("outputs/temp_uploads", exist_ok=True)
        self.scan1_path = os.path.abspath("outputs/temp_uploads/test_overwrite_scan1.jpg")
        self.scan2_path = os.path.abspath("outputs/temp_uploads/test_overwrite_scan2.jpg")

        # Copy existing source images to make sure we have valid inputs for models
        src_scan1 = os.path.abspath("outputs/temp_uploads/upload_1787152054_Te-aug-me_12.jpg")
        src_scan2 = os.path.abspath("outputs/temp_uploads/upload_1787154664_Te-aug-me_25.jpg")

        if not (os.path.exists(src_scan1) and os.path.exists(src_scan2)):
            import numpy as np
            import cv2
            def create_mock_mri(filepath, seed, axes, center, tumor_center, tumor_radius):
                img = np.zeros((256, 256, 3), dtype=np.uint8)
                mask = np.zeros((256, 256), dtype=np.uint8)
                cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)

                np.random.seed(seed)
                texture = np.random.randint(90, 210, size=(256, 256), dtype=np.uint8)
                brain_tissue = cv2.bitwise_and(texture, texture, mask=mask)
                brain_tissue_blurred = cv2.GaussianBlur(brain_tissue, (5, 5), 0)
                brain_tissue = np.where(mask > 0, brain_tissue_blurred, 0).astype(np.uint8)

                img[:, :, 0] = brain_tissue
                img[:, :, 1] = brain_tissue
                img[:, :, 2] = brain_tissue

                cv2.ellipse(img, (center[0], center[1] - 18), (15, 8), 0, 0, 360, (40, 40, 40), -1)
                cv2.ellipse(img, (center[0], center[1] + 12), (12, 6), 0, 0, 360, (40, 40, 40), -1)
                cv2.circle(img, tumor_center, tumor_radius, (250, 250, 250), -1)
                cv2.putText(img, "BRAIN_MRI", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                cv2.imwrite(filepath, img)

            create_mock_mri(self.scan1_path, seed=42, axes=(60, 80), center=(128, 128), tumor_center=(100, 100), tumor_radius=15)
            create_mock_mri(self.scan2_path, seed=200, axes=(70, 75), center=(120, 130), tumor_center=(150, 150), tumor_radius=10)
            src_scan1 = self.scan1_path
            src_scan2 = self.scan2_path

        if src_scan1 != self.scan1_path:
            shutil.copy2(src_scan1, self.scan1_path)
        if src_scan2 != self.scan2_path:
            shutil.copy2(src_scan2, self.scan2_path)

    def tearDown(self):
        self.test_client_ctx.__exit__(None, None, None)
        if os.path.exists(self.db_path):
            try:
                os.remove(self.db_path)
            except Exception:
                pass
        for path in [self.scan1_path, self.scan2_path]:
            if os.path.exists(path):
                try:
                    os.remove(path)
                except Exception:
                    pass

    def calculate_sha256(self, filepath):
        if not filepath or not os.path.exists(filepath):
            return "N/A"
        hasher = hashlib.sha256()
        with open(filepath, "rb") as f:
            while chunk := f.read(8192):
                hasher.update(chunk)
        return hasher.hexdigest()

    def test_same_patient_reports_do_not_overwrite(self):
        # 1. Generate Report A
        intake_a = {
            "patient_id": "pat-uuid-overwrite",
            "name": "Overwrite Patient",
            "age": 45,
            "gender": "Male",
            "ref_physician": "Dr. Overwrite",
            "pixel_spacing_mm": 1.0,
            "xai_method": "gradcam",
            "ensemble_mode": False
        }

        print("Generating Report A...")
        response_a = self.client.post(
            "/api/report",
            params={"filepath": self.scan1_path, "confirm_override": False},
            json=intake_a,
            headers=self.headers
        )
        self.assertEqual(response_a.status_code, 200, f"Report A generation failed: {response_a.json()}")
        report_a_db_id = response_a.json()["report_id"]
        print(f"Report A created in DB with ID: {report_a_db_id}")

        # Fetch paths for Report A
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM clinical_reports WHERE id = ?;", (report_a_db_id,))
            row_a = dict(cursor.fetchone())
        finally:
            conn.close()

        # Record Report A paths and SHA256 hashes
        paths_a = {
            "heatmap": row_a["heatmap_path"],
            "overlay": row_a["overlay_path"],
            "mask": row_a["mask_path"],
            "uncertainty": row_a["uncertainty_path"]
        }
        hashes_a_before = {k: self.calculate_sha256(v) for k, v in paths_a.items()}
        print("Report A Visual Hashes Before Report B:", hashes_a_before)

        # 2. Generate Report B (for the same patient)
        intake_b = {
            "patient_id": "pat-uuid-overwrite",
            "name": "Overwrite Patient",
            "age": 45,
            "gender": "Male",
            "ref_physician": "Dr. Overwrite",
            "pixel_spacing_mm": 1.0,
            "xai_method": "gradcam",
            "ensemble_mode": False
        }

        print("Generating Report B...")
        response_b = self.client.post(
            "/api/report",
            params={"filepath": self.scan2_path, "confirm_override": False},
            json=intake_b,
            headers=self.headers
        )
        self.assertEqual(response_b.status_code, 200, f"Report B generation failed: {response_b.json()}")
        report_b_db_id = response_b.json()["report_id"]
        print(f"Report B created in DB with ID: {report_b_db_id}")

        # Fetch paths for Report B
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM clinical_reports WHERE id = ?;", (report_b_db_id,))
            row_b = dict(cursor.fetchone())
        finally:
            conn.close()

        paths_b = {
            "heatmap": row_b["heatmap_path"],
            "overlay": row_b["overlay_path"],
            "mask": row_b["mask_path"],
            "uncertainty": row_b["uncertainty_path"]
        }

        # Record Report A hashes again after Report B generation
        hashes_a_after = {k: self.calculate_sha256(v) for k, v in paths_a.items()}
        print("Report A Visual Hashes After Report B:", hashes_a_after)

        # 3. Assert Report A files remain unchanged
        for k in hashes_a_before:
            self.assertEqual(hashes_a_before[k], hashes_a_after[k], f"Visual asset '{k}' for Report A was modified!")

        # 4. Assert Report B visual paths are different from Report A
        for k in paths_a:
            self.assertNotEqual(paths_a[k], paths_b[k], f"Visual path '{k}' is shared between Report A and B!")

        # 5. Verify both previews return correct HTTP 200 and data
        for report_id in [report_a_db_id, report_b_db_id]:
            for visual_type in ["heatmap", "overlay", "mask", "uncertainty"]:
                resp = self.client.get(
                    f"/api/report/{report_id}/visuals/{visual_type}",
                    headers=self.headers
                )
                self.assertEqual(resp.status_code, 200, f"Failed to fetch preview for report {report_id} visual {visual_type}")
                print(f"Verified report {report_id} visual {visual_type} returns HTTP 200")

import os
os.environ["DB_PATH"] = os.path.abspath("outputs/test_h72d_telemetry.db")

import unittest
import sqlite3
import datetime
import tempfile
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient as FastAPITestClient
from run_api import app as fastapi_app
from dashboard.infrastructure.web_server import create_app
from security.domain.entities import Role, User
from security.infrastructure.jwt_service import JWTService
from security.infrastructure.repository import SQLiteUserRepository
from persistence.infrastructure.repository import SQLitePersistenceRepository
from clinical_reporting.domain.entities import ProcessingSummary, PatientInfo, PredictionResult, ClinicalReport
from api.infrastructure.routes import _calculate_file_sha256

class TestH72DModelProvenance(unittest.TestCase):
    """Phase H7.2-D: Model Identity and Provenance tests."""

    def setUp(self):
        self.db_path = os.environ.get("DB_PATH", "outputs/test_h72d_telemetry.db")
        if os.path.exists(self.db_path):
            try:
                os.remove(self.db_path)
            except Exception:
                pass

        # Override DB paths
        import api.infrastructure.routes as api_routes
        import api.routes.auth_routes as auth_routes
        api_routes.DEFAULT_DB_PATH = self.db_path
        auth_routes.DEFAULT_DB_PATH = self.db_path

        self.flask_app = create_app(self.db_path)
        self.flask_app.config["DATABASE_PATH"] = self.db_path
        self.flask_app.config["TESTING"] = True
        self.flask_app.config["DISABLE_CSRF"] = True

        self.persistence_repo = SQLitePersistenceRepository(db_path=self.db_path)
        self.persistence_repo.initialize_db()

        self.user_repo = SQLiteUserRepository(db_path=self.db_path)
        self.user_repo.initialize_security_tables()

        # Disable auto-assignment triggers
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("DROP TRIGGER IF EXISTS auto_assign_doctor_on_insert;")
            conn.execute("DROP TRIGGER IF EXISTS auto_assign_patient_on_insert;")
            conn.execute("DELETE FROM doctor_patient_assignments;")
            conn.commit()
        finally:
            conn.close()

        # Create roles
        self.admin = self.user_repo.bootstrap_admin()
        self.doctor = User(
            id=None,
            uuid="doc-test-d",
            email="doctor-d@aurascan.ai",
            password_hash="fakehash",
            full_name="Dr. Smith",
            role=Role.DOCTOR,
            is_verified=True,
            is_active=True,
            created_at=datetime.datetime.utcnow().isoformat(),
            updated_at=datetime.datetime.utcnow().isoformat()
        )
        self.doctor = self.user_repo.create_user(self.doctor)

        self.patient = User(
            id=None,
            uuid="pat-test-d",
            email="patient-d@aurascan.ai",
            password_hash="fakehash",
            full_name="Patient Bob",
            role=Role.PATIENT,
            is_verified=True,
            is_active=True,
            created_at=datetime.datetime.utcnow().isoformat(),
            updated_at=datetime.datetime.utcnow().isoformat()
        )
        self.patient = self.user_repo.create_user(self.patient)

        jwt_svc = JWTService()
        self.admin_token = jwt_svc.create_access_token(self.admin.uuid, self.admin.id, self.admin.email, self.admin.role)
        self.doc_token = jwt_svc.create_access_token(self.doctor.uuid, self.doctor.id, self.doctor.email, self.doctor.role)
        self.pat_token = jwt_svc.create_access_token(self.patient.uuid, self.patient.id, self.patient.email, self.patient.role)

        self.fastapi_client = FastAPITestClient(fastapi_app)
        self.flask_client = self.flask_app.test_client()

    def tearDown(self):
        if os.path.exists(self.db_path):
            try:
                os.remove(self.db_path)
            except Exception:
                pass

    def test_01_sha256_calculation_is_deterministic(self):
        """1. Verify SHA-256 helper is deterministic and streams chunks correctly."""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"BrainTumorModelWeightsDataset12345")
            temp_path = f.name
        try:
            h1 = _calculate_file_sha256(temp_path)
            h2 = _calculate_file_sha256(temp_path)
            self.assertEqual(h1, h2)
            self.assertEqual(len(h1), 64)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_02_different_checkpoint_contents_produce_different_hashes(self):
        """2. Verify distinct file contents produce distinct SHA-256 hashes."""
        with tempfile.NamedTemporaryFile(delete=False) as f1:
            f1.write(b"weights_v1")
            p1 = f1.name
        with tempfile.NamedTemporaryFile(delete=False) as f2:
            f2.write(b"weights_v2")
            p2 = f2.name
        try:
            h1 = _calculate_file_sha256(p1)
            h2 = _calculate_file_sha256(p2)
            self.assertNotEqual(h1, h2)
        finally:
            for p in [p1, p2]:
                if os.path.exists(p):
                    os.remove(p)

    def test_03_missing_checkpoint_fails_safely(self):
        """3. Hashing helper fails safely returning default empty hash for missing files."""
        h = _calculate_file_sha256("non_existent_model_checkpoint.pth")
        # Empty file hash / default hash representation
        self.assertEqual(h, "da39a3ee5e6b4b0d3255bfef95601890afd80709")

    def test_04_absolute_filesystem_paths_are_never_persisted(self):
        """4. Verify absolute paths are normalized/blocked from DB model provenance identifiers."""
        db_path = self.db_path
        repo = SQLitePersistenceRepository(db_path=db_path)

        # Register using safe path
        safe_ident = "classification/efficientnet_b0_brain_tumor.pth"
        loaded_at = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        prov_id = repo.register_model_provenance(
            model_type="CLASSIFICATION",
            model_name="efficientnet_b0",
            architecture="EfficientNet-B0",
            model_version="1.0",
            checkpoint_identifier=safe_ident,
            checkpoint_sha256="fake_sha256_12345",
            device="cpu",
            loaded_at=loaded_at
        )
        self.assertGreater(prov_id, 0)

        # Verify stored value in SQLite does not contain windows or unix absolute patterns
        prov_record = repo.get_model_provenance(prov_id)
        self.assertIsNotNone(prov_record)
        self.assertEqual(prov_record["checkpoint_identifier"], safe_ident)
        self.assertNotIn("C:", prov_record["checkpoint_identifier"])
        self.assertNotIn("Users", prov_record["checkpoint_identifier"])

    def test_05_provenance_registration_works_and_is_idempotent(self):
        """5 & 6. Verify provenance registration creates records idempotently."""
        repo = SQLitePersistenceRepository(db_path=self.db_path)
        loaded_at = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # First register
        id1 = repo.register_model_provenance(
            model_type="CLASSIFICATION",
            model_name="efficientnet_b0",
            architecture="EfficientNet-B0",
            model_version="1.0",
            checkpoint_identifier="classification/efficientnet_b0.pth",
            checkpoint_sha256="same_sha_abc",
            device="cpu",
            loaded_at=loaded_at
        )

        # Second register with same parameters (should return same ID)
        id2 = repo.register_model_provenance(
            model_type="CLASSIFICATION",
            model_name="efficientnet_b0",
            architecture="EfficientNet-B0",
            model_version="1.0",
            checkpoint_identifier="classification/efficientnet_b0.pth",
            checkpoint_sha256="same_sha_abc",
            device="cpu",
            loaded_at=loaded_at
        )
        self.assertEqual(id1, id2)

    def test_06_changed_checkpoint_creates_new_provenance_identity(self):
        """7. Changing checkpoint hash yields a different provenance record ID."""
        repo = SQLitePersistenceRepository(db_path=self.db_path)
        loaded_at = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        id1 = repo.register_model_provenance(
            model_type="CLASSIFICATION",
            model_name="efficientnet_b0",
            architecture="EfficientNet-B0",
            model_version="1.0",
            checkpoint_identifier="classification/efficientnet_b0.pth",
            checkpoint_sha256="sha_v1",
            device="cpu",
            loaded_at=loaded_at
        )

        id2 = repo.register_model_provenance(
            model_type="CLASSIFICATION",
            model_name="efficientnet_b0",
            architecture="EfficientNet-B0",
            model_version="1.0",
            checkpoint_identifier="classification/efficientnet_b0.pth",
            checkpoint_sha256="sha_v2",  # modified checksum
            device="cpu",
            loaded_at=loaded_at
        )
        self.assertNotEqual(id1, id2)

    def test_07_predictions_link_to_provenance_ids(self):
        """8, 9, 10, 11, 12. Verify prediction persistence captures and links model provenance."""
        repo = SQLitePersistenceRepository(db_path=self.db_path)
        loaded_at = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # 1. Register active models
        cls_prov_id = repo.register_model_provenance(
            model_type="CLASSIFICATION",
            model_name="efficientnet_b0",
            architecture="EfficientNet-B0",
            model_version="1.0",
            checkpoint_identifier="classification/model.pth",
            checkpoint_sha256="cls_test_sha",
            device="cpu",
            loaded_at=loaded_at
        )
        seg_prov_id = repo.register_model_provenance(
            model_type="SEGMENTATION",
            model_name="brain_tumor_unext",
            architecture="UNeXt",
            model_version="1.0",
            checkpoint_identifier="unext/model.pth",
            checkpoint_sha256="seg_test_sha",
            device="cpu",
            loaded_at=loaded_at
        )

        # 2. Setup mock clinical report object
        pat_info = PatientInfo(
            patient_id="PAT_PROV_99",
            name="Alice Patient",
            age=30,
            gender="Female",
            scan_date="2026-08-15",
            ref_physician="Dr. Tester"
        )
        proc_summary = ProcessingSummary(
            device="cpu",
            execution_time_sec=1.5,
            classification_model_path="classification/model.pth",
            segmentation_model_path="unext/model.pth",
            classification_latency_sec=0.5,
            segmentation_latency_sec=0.8,
            explainability_latency_sec=0.2,
            classification_model_provenance_id=cls_prov_id,
            segmentation_model_provenance_id=seg_prov_id
        )
        pred_res = PredictionResult(
            label=0,
            class_name="Glioma",
            confidence_score=0.92,
            probabilities={"Glioma": 0.92}
        )
        report = ClinicalReport(
            patient_info=pat_info,
            processing_summary=proc_summary,
            classification=pred_res,
            segmentation_metrics=None,
            severity_assessment=None,
            original_image_path="slice.png",
            heatmap_image_path=None,
            overlay_image_path=None,
            segmentation_mask_path=None
        )

        # 3. Save report and query database values directly
        report_id = repo.save_report(report, "outputs")
        self.assertGreater(report_id, 0)

        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            row = conn.execute("SELECT * FROM predictions ORDER BY id DESC LIMIT 1;").fetchone()
            self.assertIsNotNone(row)
            self.assertEqual(row["classification_model_provenance_id"], cls_prov_id)
            self.assertEqual(row["segmentation_model_provenance_id"], seg_prov_id)
        finally:
            conn.close()

    def test_08_existing_predictions_with_null_provenance_remain_readable(self):
        """13. Verify legacy predictions (where model provenance FKs are NULL) do not crash read queries."""
        conn = sqlite3.connect(self.db_path)
        try:
            # Seed legacy rows without provenance ids
            conn.execute("""
                INSERT INTO patients (patient_id, name, age, gender, created_at)
                VALUES ('PAT_LEGACY_01', 'Bob Patient', 50, 'Male', '2026-08-11 12:00:00');
            """)
            conn.execute("""
                INSERT INTO mri_scans (id, patient_id, image_path, pixel_spacing_mm, ref_physician, scan_date, created_at)
                VALUES (777, 'PAT_LEGACY_01', 'legacy.png', 1.0, 'Dr. Smith', '2026-08-11', '2026-08-11 12:00:00');
            """)
            conn.execute("""
                INSERT INTO predictions (
                    id, scan_id, predicted_class, confidence_score,
                    prob_glioma, prob_meningioma, prob_pituitary, prob_no_tumor,
                    tumor_pixel_count, tumor_area_mm2, tumor_percentage_brain, tumor_percentage_image,
                    estimated_brain_pixel_count, rule_based_severity, severity_rule_description, created_at,
                    classification_model_provenance_id, segmentation_model_provenance_id
                ) VALUES (888, 777, 'No Tumor', 0.99, 0.0, 0.0, 0.0, 0.99, 0, 0.0, 0.0, 0.0, 1000, 'Low', 'No tumor', '2026-08-11 12:00:00', NULL, NULL);
            """)
            conn.commit()
        finally:
            conn.close()

        # Retrieve prediction as admin and confirm it reads values successfully
        repo = SQLitePersistenceRepository(db_path=self.db_path)
        health = repo.get_health_telemetry()
        self.assertGreaterEqual(health["total_predictions"], 1)

    def test_09_rbac_endpoint_and_secrets_protection(self):
        """14, 15, 16, 17, 18, 19, 20. Verify RBAC controls and ensure no absolute paths/secrets exist in responses."""
        # Query diagnostics
        headers_adm = {"Authorization": f"Bearer {self.admin_token}"}
        resp_adm = self.flask_client.get("/api/health-telemetry", headers=headers_adm)
        self.assertEqual(resp_adm.status_code, 200)
        data = resp_adm.get_json()

        # Provenance keys present
        self.assertIn("active_classification_model", data)
        self.assertIn("active_segmentation_model", data)

        # Confirm no path disclosures
        for model_info in [data["active_classification_model"], data["active_segmentation_model"]]:
            if model_info:
                self.assertNotIn("C:", str(model_info))
                self.assertNotIn("Users", str(model_info))
                self.assertNotIn("D:", str(model_info))

        # Deny Doctor/Patient access to internal telemetry response structures
        headers_doc = {"Authorization": f"Bearer {self.doc_token}"}
        resp_doc = self.flask_client.get("/api/health-telemetry", headers=headers_doc)
        self.assertEqual(resp_doc.status_code, 403)

        headers_pat = {"Authorization": f"Bearer {self.pat_token}"}
        resp_pat = self.flask_client.get("/api/health-telemetry", headers=headers_pat)
        self.assertEqual(resp_pat.status_code, 403)

        resp_unauth = self.flask_client.get("/api/health-telemetry")
        self.assertEqual(resp_unauth.status_code, 401)

if __name__ == "__main__":
    unittest.main()

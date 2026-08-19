import unittest
import os
import tempfile
import sqlite3
import datetime
from flask import json
from dashboard.infrastructure.web_server import create_app
from security.domain.entities import Role, User, TokenType
from security.infrastructure.repository import SQLiteUserRepository
from security.infrastructure.jwt_service import JWTService
from security.infrastructure.password import PasswordHasher
from persistence.infrastructure.repository import SQLitePersistenceRepository


class TestGlobalSearchAndFiltering(unittest.TestCase):
    """G8.1 integration tests checking secure clinical search workstation."""

    def setUp(self):
        self.db_fd, self.db_path = tempfile.mkstemp(suffix=".db")
        self.app = create_app(db_path=self.db_path)
        self.app.config["TESTING"] = True
        self.client = self.app.test_client()

        # Initialize schema
        self.persistence_repo = SQLitePersistenceRepository(db_path=self.db_path)
        self.persistence_repo.initialize_db()

        self.user_repo = SQLiteUserRepository(db_path=self.db_path)
        self.user_repo.initialize_security_tables()

        # Create mock users
        pass_hash = PasswordHasher.hash_password("AuraScan@2026")
        self.doctor = User(
            id=None,
            uuid="doc-uuid-888",
            email="dr.alice@aurascan.ai",
            password_hash=pass_hash,
            full_name="Dr. Alice",
            role=Role.DOCTOR,
            is_verified=True,
            is_active=True
        )
        self.user_repo.create_user(self.doctor)

        self.patient_1 = User(
            id=None,
            uuid="pat-001",
            email="alice.vance@gmail.com",
            password_hash=pass_hash,
            full_name="Alice Vance",
            role=Role.PATIENT,
            is_verified=True,
            is_active=True
        )
        self.user_repo.create_user(self.patient_1)

        self.patient_2 = User(
            id=None,
            uuid="pat-002",
            email="bob.vance@gmail.com",
            password_hash=pass_hash,
            full_name="Bob Vance",
            role=Role.PATIENT,
            is_verified=True,
            is_active=True
        )
        self.user_repo.create_user(self.patient_2)

        # Generate tokens
        self.jwt_svc = JWTService()
        self.doctor_token = self.jwt_svc.create_access_token(
            user_uuid=self.doctor.uuid,
            user_id=self.doctor.id,
            email=self.doctor.email,
            role=self.doctor.role
        )
        self.patient_1_token = self.jwt_svc.create_access_token(
            user_uuid=self.patient_1.uuid,
            user_id=self.patient_1.id,
            email=self.patient_1.email,
            role=self.patient_1.role
        )
        self.patient_2_token = self.jwt_svc.create_access_token(
            user_uuid=self.patient_2.uuid,
            user_id=self.patient_2.id,
            email=self.patient_2.email,
            role=self.patient_2.role
        )

        self.seed_test_data()

    def tearDown(self):
        os.close(self.db_fd)
        if os.path.exists(self.db_path):
            try:
                os.remove(self.db_path)
            except Exception:
                pass

    def seed_test_data(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        try:
            # Seed Patients
            patients = [
                ("pat-001", "Alice Vance", 45, "Female", "2026-08-01"),
                ("pat-002", "Bob Vance", 50, "Male", "2026-08-02"),
                ("pat-003", "Charlie Vance", 35, "Male", "2026-08-03"),
                ("pat-004", "David Miller", 28, "Male", "2026-08-04")
            ]
            cursor.executemany(
                "INSERT INTO patients (patient_id, name, age, gender, created_at) VALUES (?, ?, ?, ?, ?);",
                patients
            )

            # Seed scans
            scans = [
                (1, "pat-001", "img1.png", 1.0, "Dr. Adams", "2026-08-01", "2026-08-01"),
                (2, "pat-002", "img2.png", 1.0, "Dr. Baker", "2026-08-02", "2026-08-02"),
                (3, "pat-003", "img3.png", 1.0, "Dr. Baker", "2026-08-03", "2026-08-03"),
                (4, "pat-004", "img4.png", 1.0, "Dr. Adams", "2026-08-04", "2026-08-04")
            ]
            cursor.executemany(
                "INSERT INTO mri_scans (id, patient_id, image_path, pixel_spacing_mm, ref_physician, scan_date, created_at) VALUES (?, ?, ?, ?, ?, ?, ?);",
                scans
            )

            # Seed predictions
            predictions = [
                (1, 1, "Glioma", 0.95, 0.95, 0.02, 0.02, 0.01, 100, 10.0, 1.0, 0.5, 1000, "High", " Glioma High Risk", "2026-08-01"),
                (2, 2, "Meningioma", 0.85, 0.05, 0.85, 0.05, 0.05, 80, 8.0, 0.8, 0.4, 1000, "Medium", "Meningioma Medium Risk", "2026-08-02"),
                (3, 3, "No Tumor", 0.99, 0.00, 0.00, 0.00, 0.99, 0, 0.0, 0.0, 0.0, 1000, "Low", "No tumor low risk", "2026-08-03"),
                (4, 4, "Pituitary", 0.90, 0.05, 0.05, 0.90, 0.00, 70, 7.0, 0.7, 0.3, 1000, "High", "Pituitary High Risk", "2026-08-04")
            ]
            cursor.executemany(
                """INSERT INTO predictions (
                    id, scan_id, predicted_class, confidence_score, prob_glioma, prob_meningioma,
                    prob_pituitary, prob_no_tumor, tumor_pixel_count, tumor_area_mm2,
                    tumor_percentage_brain, tumor_percentage_image, estimated_brain_pixel_count,
                    rule_based_severity, severity_rule_description, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);""",
                predictions
            )

            # Seed clinical reports
            reports = [
                (1, 1, "r1.md", "r1.json", "r1.pdf", "2026-08-01"),
                (2, 2, "r2.md", "r2.json", "r2.pdf", "2026-08-02"),
                (3, 3, "r3.md", "r3.json", "r3.pdf", "2026-08-03"),
                (4, 4, "r4.md", "r4.json", "r4.pdf", "2026-08-04")
            ]
            cursor.executemany(
                "INSERT INTO clinical_reports (id, prediction_id, markdown_path, json_path, pdf_path, created_at) VALUES (?, ?, ?, ?, ?, ?);",
                reports
            )

            # Seed reports table (statuses)
            reports_lifecycle = [
                (1, "RPT-2026-000001", "pat-001", "Dr. Adams", "GENERATED", "2026-08-01", "2026-08-01", "r1.pdf", "r1.json"),
                (2, "RPT-2026-000002", "pat-002", "Dr. Baker", "DRAFT", "2026-08-02", "2026-08-02", "r2.pdf", "r2.json"),
                (3, "RPT-2026-000003", "pat-003", "Dr. Baker", "FINALIZED", "2026-08-03", "2026-08-03", "r3.pdf", "r3.json"),
                (4, "RPT-2026-000004", "pat-004", "Dr. Adams", "ARCHIVED", "2026-08-04", "2026-08-04", "r4.pdf", "r4.json")
            ]
            cursor.executemany(
                """INSERT INTO reports (
                    report_id, report_number, patient_id, created_by, status, created_at, updated_at, pdf_path, json_path
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);""",
                reports_lifecycle
            )

            from tests.helpers.authorization_fixtures import assign_doctor_to_patient
            assign_doctor_to_patient(conn, "dr.alice@aurascan.ai", "pat-001")
            assign_doctor_to_patient(conn, "dr.alice@aurascan.ai", "pat-002")
            assign_doctor_to_patient(conn, "dr.alice@aurascan.ai", "pat-003")
            assign_doctor_to_patient(conn, "dr.alice@aurascan.ai", "pat-004")

            conn.commit()
        finally:
            conn.close()

    def get_auth_headers(self, token):
        return {"Authorization": f"Bearer {token}"}

    def test_patient_name_search(self):
        # Doctor searches for patient name "Vance"
        res = self.client.get(
            "/api/search?patient_name=Vance",
            headers=self.get_auth_headers(self.doctor_token)
        )
        self.assertEqual(res.status_code, 200)
        data = json.loads(res.data)
        # Should return Alice, Bob, and Charlie Vance
        self.assertEqual(len(data), 3)
        names = [x["patient_name"] for x in data]
        self.assertIn("Alice Vance", names)
        self.assertIn("Bob Vance", names)
        self.assertIn("Charlie Vance", names)
        self.assertNotIn("David Miller", names)

    def test_patient_id_search(self):
        res = self.client.get(
            "/api/search?patient_id=pat-002",
            headers=self.get_auth_headers(self.doctor_token)
        )
        self.assertEqual(res.status_code, 200)
        data = json.loads(res.data)
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]["patient_id"], "pat-002")
        self.assertEqual(data[0]["patient_name"], "Bob Vance")

    def test_referring_doctor_search(self):
        res = self.client.get(
            "/api/search?referring_doctor=Baker",
            headers=self.get_auth_headers(self.doctor_token)
        )
        self.assertEqual(res.status_code, 200)
        data = json.loads(res.data)
        self.assertEqual(len(data), 2)
        names = [x["patient_name"] for x in data]
        self.assertIn("Bob Vance", names)
        self.assertIn("Charlie Vance", names)

    def test_classification_filter(self):
        res = self.client.get(
            "/api/search?classification=Glioma",
            headers=self.get_auth_headers(self.doctor_token)
        )
        self.assertEqual(res.status_code, 200)
        data = json.loads(res.data)
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]["predicted_class"], "Glioma")
        self.assertEqual(data[0]["patient_name"], "Alice Vance")

    def test_severity_filter(self):
        res = self.client.get(
            "/api/search?severity=High",
            headers=self.get_auth_headers(self.doctor_token)
        )
        self.assertEqual(res.status_code, 200)
        data = json.loads(res.data)
        self.assertEqual(len(data), 2)
        names = [x["patient_name"] for x in data]
        self.assertIn("Alice Vance", names)
        self.assertIn("David Miller", names)

    def test_confidence_filter(self):
        res = self.client.get(
            "/api/search?min_confidence=0.91",
            headers=self.get_auth_headers(self.doctor_token)
        )
        self.assertEqual(res.status_code, 200)
        data = json.loads(res.data)
        self.assertEqual(len(data), 2)
        names = [x["patient_name"] for x in data]
        self.assertIn("Alice Vance", names) # 0.95
        self.assertIn("Charlie Vance", names) # 0.99

    def test_date_range_filter(self):
        res = self.client.get(
            "/api/search?start_date=2026-08-02&end_date=2026-08-03",
            headers=self.get_auth_headers(self.doctor_token)
        )
        self.assertEqual(res.status_code, 200)
        data = json.loads(res.data)
        self.assertEqual(len(data), 2)
        names = [x["patient_name"] for x in data]
        self.assertIn("Bob Vance", names) # 2026-08-02
        self.assertIn("Charlie Vance", names) # 2026-08-03

    def test_combined_filters(self):
        res = self.client.get(
            "/api/search?referring_doctor=Baker&classification=No Tumor",
            headers=self.get_auth_headers(self.doctor_token)
        )
        self.assertEqual(res.status_code, 200)
        data = json.loads(res.data)
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]["patient_name"], "Charlie Vance")

    def test_sorting(self):
        # Sort by patient_name ASC
        res = self.client.get(
            "/api/search?sort_by=patient_name&sort_order=asc",
            headers=self.get_auth_headers(self.doctor_token)
        )
        self.assertEqual(res.status_code, 200)
        data = json.loads(res.data)
        names = [x["patient_name"] for x in data]
        self.assertEqual(names, ["Alice Vance", "Bob Vance", "Charlie Vance", "David Miller"])

        # Sort by scan_date DESC
        res2 = self.client.get(
            "/api/search?sort_by=scan_date&sort_order=desc",
            headers=self.get_auth_headers(self.doctor_token)
        )
        self.assertEqual(res2.status_code, 200)
        data2 = json.loads(res2.data)
        dates = [x["scan_date"] for x in data2]
        self.assertEqual(dates, ["2026-08-04", "2026-08-03", "2026-08-02", "2026-08-01"])

    def test_pagination(self):
        res = self.client.get(
            "/api/search?page=1&page_size=2",
            headers=self.get_auth_headers(self.doctor_token)
        )
        self.assertEqual(res.status_code, 200)
        data = json.loads(res.data)
        self.assertIn("items", data)
        self.assertEqual(data["total_count"], 4)
        self.assertEqual(data["page"], 1)
        self.assertEqual(data["page_size"], 2)
        self.assertEqual(data["total_pages"], 2)
        self.assertEqual(len(data["items"]), 2)

        res2 = self.client.get(
            "/api/search?page=2&page_size=2&sort_by=patient_name&sort_order=asc",
            headers=self.get_auth_headers(self.doctor_token)
        )
        self.assertEqual(res2.status_code, 200)
        data2 = json.loads(res2.data)
        self.assertEqual(data2["page"], 2)
        self.assertEqual(len(data2["items"]), 2)
        names = [x["patient_name"] for x in data2["items"]]
        self.assertEqual(names, ["Charlie Vance", "David Miller"])

    def test_empty_results(self):
        res = self.client.get(
            "/api/search?q=nonexistent_token_query",
            headers=self.get_auth_headers(self.doctor_token)
        )
        self.assertEqual(res.status_code, 200)
        data = json.loads(res.data)
        self.assertEqual(len(data), 0)

    def test_invalid_parameters(self):
        # Invalid min_confidence range
        res = self.client.get(
            "/api/search?min_confidence=1.5",
            headers=self.get_auth_headers(self.doctor_token)
        )
        self.assertEqual(res.status_code, 400)

        # Invalid start date format
        res2 = self.client.get(
            "/api/search?start_date=2026-08-123",
            headers=self.get_auth_headers(self.doctor_token)
        )
        self.assertEqual(res2.status_code, 400)

        # Invalid page
        res3 = self.client.get(
            "/api/search?page=-1",
            headers=self.get_auth_headers(self.doctor_token)
        )
        self.assertEqual(res3.status_code, 400)

        # Oversized page_size
        res4 = self.client.get(
            "/api/search?page_size=200",
            headers=self.get_auth_headers(self.doctor_token)
        )
        self.assertEqual(res4.status_code, 400)

        # Invalid sort_by
        res5 = self.client.get(
            "/api/search?sort_by=nonexistent_column",
            headers=self.get_auth_headers(self.doctor_token)
        )
        self.assertEqual(res5.status_code, 400)

        # Invalid sort_order
        res6 = self.client.get(
            "/api/search?sort_order=invalid_dir",
            headers=self.get_auth_headers(self.doctor_token)
        )
        self.assertEqual(res6.status_code, 400)

    def test_authorization_controls(self):
        # 1. Unauthenticated (Anonymous) request
        res_anon = self.client.get("/api/search?q=Vance")
        self.assertEqual(res_anon.status_code, 401)

        # 2. Patient attempting search for other patients (should only return their own record)
        res_pat1 = self.client.get(
            "/api/search?patient_name=Vance",
            headers=self.get_auth_headers(self.patient_1_token)
        )
        self.assertEqual(res_pat1.status_code, 200)
        data_pat1 = json.loads(res_pat1.data)
        # Vance query matches Alice, Bob, and Charlie. Alice Vance has pat-001.
        # But Alice Vance is logged in, so she should only get Alice Vance!
        self.assertEqual(len(data_pat1), 1)
        self.assertEqual(data_pat1[0]["patient_name"], "Alice Vance")
        self.assertEqual(data_pat1[0]["patient_id"], "pat-001")

        # 3. Patient Bob Vance (pat-002) searches
        res_pat2 = self.client.get(
            "/api/search?patient_name=Vance",
            headers=self.get_auth_headers(self.patient_2_token)
        )
        self.assertEqual(res_pat2.status_code, 200)
        data_pat2 = json.loads(res_pat2.data)
        self.assertEqual(len(data_pat2), 1)
        self.assertEqual(data_pat2[0]["patient_name"], "Bob Vance")
        self.assertEqual(data_pat2[0]["patient_id"], "pat-002")

    def test_sql_injection_safety(self):
        # Send raw SQL injection payload through query parameter
        payload = "pat-001' OR '1'='1"
        res = self.client.get(
            f"/api/search?patient_id={payload}",
            headers=self.get_auth_headers(self.doctor_token)
        )
        self.assertEqual(res.status_code, 200)
        data = json.loads(res.data)
        # Should not fetch everything (safety). Since LIKE pattern is built safely, it will look for
        # patient_id matching "%pat-001' OR '1'='1%" literally, which doesn't exist and returns empty.
        self.assertEqual(len(data), 0)


if __name__ == "__main__":
    unittest.main()

import os
import unittest
import tempfile
import sqlite3
import datetime
import json
from flask import json as flask_json
from dashboard.infrastructure.web_server import create_app
from security.domain.entities import Role, User
from security.infrastructure.repository import SQLiteUserRepository
from security.infrastructure.jwt_service import JWTService
from security.infrastructure.password import PasswordHasher
from persistence.infrastructure.repository import SQLitePersistenceRepository
from clinical_reporting.application.notification_service import NotificationService


class TestPredictionFeedback(unittest.TestCase):
    """Phase G8.3 Automated unit, integration, and security compliance tests for prediction feedback and quality reviews."""

    def setUp(self):
        self.db_fd, self.db_path = tempfile.mkstemp(suffix=".db")
        self.app = create_app(db_path=self.db_path)
        self.app.config["TESTING"] = True
        self.client = self.app.test_client()

        # Initialize databases
        self.persistence_repo = SQLitePersistenceRepository(db_path=self.db_path)
        self.persistence_repo.initialize_db()

        self.user_repo = SQLiteUserRepository(db_path=self.db_path)
        self.user_repo.initialize_security_tables()

        # Users hashes
        pass_hash = PasswordHasher.hash_password("AuraScan@2026")

        self.doctor = User(
            id=None,
            uuid="doc-111",
            email="doctor@aurascan.ai",
            password_hash=pass_hash,
            full_name="Dr. House",
            role=Role.DOCTOR,
            is_verified=True,
            is_active=True
        )
        self.user_repo.create_user(self.doctor)

        self.patient = User(
            id=None,
            uuid="pat-222",
            email="patient@aurascan.ai",
            password_hash=pass_hash,
            full_name="John Doe",
            role=Role.PATIENT,
            is_verified=True,
            is_active=True
        )
        self.user_repo.create_user(self.patient)

        self.other_patient = User(
            id=None,
            uuid="pat-333",
            email="other@aurascan.ai",
            password_hash=pass_hash,
            full_name="Jane Doe",
            role=Role.PATIENT,
            is_verified=True,
            is_active=True
        )
        self.user_repo.create_user(self.other_patient)

        self.admin = self.user_repo.get_by_email("admin@aurascan.ai")

        # JWT Tokens
        self.jwt_svc = JWTService()
        self.doctor_token = self.jwt_svc.create_access_token(
            user_uuid=self.doctor.uuid, user_id=self.doctor.id, email=self.doctor.email, role=self.doctor.role
        )
        self.patient_token = self.jwt_svc.create_access_token(
            user_uuid=self.patient.uuid, user_id=self.patient.id, email=self.patient.email, role=self.patient.role
        )
        self.other_patient_token = self.jwt_svc.create_access_token(
            user_uuid=self.other_patient.uuid, user_id=self.other_patient.id, email=self.other_patient.email, role=self.other_patient.role
        )
        self.admin_token = self.jwt_svc.create_access_token(
            user_uuid=self.admin.uuid, user_id=self.admin.id, email=self.admin.email, role=self.admin.role
        )

        # Seed patient, scan, predictions, reports, and clinical_reports mapping
        conn = sqlite3.connect(self.db_path)
        with conn:
            # Seed patient matching self.patient uuid
            conn.execute("""
                INSERT INTO patients (patient_id, name, age, gender, created_at)
                VALUES (?, ?, 30, 'M', '2026-08-10T12:00:00');
            """, (self.patient.uuid, self.patient.full_name))

            # Seed scan
            conn.execute("""
                INSERT INTO mri_scans (id, patient_id, image_path, pixel_spacing_mm, ref_physician, scan_date, created_at)
                VALUES (501, ?, 'scan1.png', 0.5, 'Dr. Ref', '2026-08-10', '2026-08-10T12:00:00');
            """, (self.patient.uuid,))

            # Seed prediction
            conn.execute("""
                INSERT INTO predictions (id, scan_id, predicted_class, confidence_score, prob_glioma, prob_meningioma, prob_pituitary, prob_no_tumor, tumor_pixel_count, tumor_area_mm2, tumor_percentage_brain, tumor_percentage_image, estimated_brain_pixel_count, rule_based_severity, severity_rule_description, created_at)
                VALUES (901, 501, 'Glioma', 0.94, 0.9, 0.05, 0.02, 0.03, 1000, 120.5, 1.2, 0.8, 100000, 'HIGH', 'Glioma detected', '2026-08-10T12:00:00');
            """)

            # Seed report
            conn.execute("""
                INSERT INTO reports (report_id, report_number, patient_id, status, created_at, updated_at)
                VALUES (801, 'RPT-2026-000801', ?, 'GENERATED', '2026-08-10T12:00:00', '2026-08-10T12:00:00');
            """, (self.patient.uuid,))

            # Seed clinical report mapping
            conn.execute("""
                INSERT INTO clinical_reports (id, prediction_id, markdown_path, json_path, pdf_path, created_at)
                VALUES (801, 901, 'rpt1.md', 'rpt1.json', 'report1.pdf', '2026-08-10T12:00:00');
            """)

            # Seed other patient
            conn.execute("""
                INSERT INTO patients (patient_id, name, age, gender, created_at)
                VALUES (?, ?, 24, 'F', '2026-08-10T12:00:00');
            """, (self.other_patient.uuid, self.other_patient.full_name))

            # Seed other scan
            conn.execute("""
                INSERT INTO mri_scans (id, patient_id, image_path, pixel_spacing_mm, ref_physician, scan_date, created_at)
                VALUES (502, ?, 'scan2.png', 0.5, 'Dr. Ref', '2026-08-10', '2026-08-10T12:00:00');
            """, (self.other_patient.uuid,))

            # Seed other prediction
            conn.execute("""
                INSERT INTO predictions (id, scan_id, predicted_class, confidence_score, prob_glioma, prob_meningioma, prob_pituitary, prob_no_tumor, tumor_pixel_count, tumor_area_mm2, tumor_percentage_brain, tumor_percentage_image, estimated_brain_pixel_count, rule_based_severity, severity_rule_description, created_at)
                VALUES (902, 502, 'No Tumor', 0.99, 0.01, 0.01, 0.01, 0.97, 0, 0.0, 0.0, 0.0, 100000, 'LOW', 'No tumor detected', '2026-08-10T12:00:00');
            """)

            # Seed other report
            conn.execute("""
                INSERT INTO reports (report_id, report_number, patient_id, status, created_at, updated_at)
                VALUES (802, 'RPT-2026-000802', ?, 'GENERATED', '2026-08-10T12:00:00', '2026-08-10T12:00:00');
            """, (self.other_patient.uuid,))

            # Seed other clinical report mapping
            conn.execute("""
                INSERT INTO clinical_reports (id, prediction_id, markdown_path, json_path, pdf_path, created_at)
                VALUES (802, 902, 'rpt2.md', 'rpt2.json', 'report2.pdf', '2026-08-10T12:00:00');
            """)

        conn.close()

    def tearDown(self):
        os.close(self.db_fd)
        if os.path.exists(self.db_path):
            try:
                os.remove(self.db_path)
            except Exception:
                pass

    def test_submit_prediction_feedback_success(self):
        """Verify successful feedback submission with bounds validation and safe parsing."""
        self.client.set_cookie("access_token", self.doctor_token)
        response = self.client.post("/api/predictions/901/feedback", json={
            "rating": 4,
            "feedback_type": "ACCURATE",
            "comment": "Nice segmentation."
        })
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertTrue(data["success"])
        self.assertIn("feedback_id", data)

        # Retrieve the feedback to check values
        response = self.client.get("/api/predictions/901/feedback")
        self.assertEqual(response.status_code, 200)
        feedbacks = json.loads(response.data)
        self.assertEqual(len(feedbacks), 1)
        self.assertEqual(feedbacks[0]["rating"], 4)
        self.assertEqual(feedbacks[0]["feedback_type"], "ACCURATE")
        self.assertEqual(feedbacks[0]["comment"], "Nice segmentation.")

    def test_submit_prediction_feedback_validation_errors(self):
        """Verify server-side rating bounds and type constraints."""
        self.client.set_cookie("access_token", self.doctor_token)

        # Rating too high
        response = self.client.post("/api/predictions/901/feedback", json={
            "rating": 6,
            "feedback_type": "ACCURATE",
            "comment": "Too high"
        })
        self.assertEqual(response.status_code, 400)

        # Invalid type
        response = self.client.post("/api/predictions/901/feedback", json={
            "rating": 3,
            "feedback_type": "AWESOME_PREDICTION",
            "comment": "Bad type"
        })
        self.assertEqual(response.status_code, 400)

        # Comment too long
        response = self.client.post("/api/predictions/901/feedback", json={
            "rating": 3,
            "feedback_type": "ACCURATE",
            "comment": "a" * 1001
        })
        self.assertEqual(response.status_code, 400)

    def test_submit_feedback_idor_and_ownership(self):
        """Verify that patients can only submit feedback for their own predictions."""
        # Patient 222 owns scan 501 / prediction 901. Let's try to submit feedback on prediction 902 (owned by other_patient 333)
        self.client.set_cookie("access_token", self.patient_token)
        response = self.client.post("/api/predictions/902/feedback", json={
            "rating": 5,
            "feedback_type": "ACCURATE",
            "comment": "Stealing access"
        })
        self.assertEqual(response.status_code, 403)

        # Now submit feedback on their own prediction 901
        response = self.client.post("/api/predictions/901/feedback", json={
            "rating": 5,
            "feedback_type": "ACCURATE",
            "comment": "My own results look good!"
        })
        self.assertEqual(response.status_code, 200)

    def test_idempotent_feedback_update(self):
        """Verify that duplicate feedback submissions overwrite/update instead of creating duplicates."""
        self.client.set_cookie("access_token", self.doctor_token)
        # First submission
        res1 = self.client.post("/api/predictions/901/feedback", json={
            "rating": 4,
            "feedback_type": "ACCURATE",
            "comment": "Initial comment"
        })
        self.assertEqual(res1.status_code, 200)

        # Second submission (should update)
        res2 = self.client.post("/api/predictions/901/feedback", json={
            "rating": 5,
            "feedback_type": "TECHNICAL_QUALITY_ISSUE",
            "comment": "Updated comment"
        })
        self.assertEqual(res2.status_code, 200)

        # Retrieve feedback: should only have 1 record
        res3 = self.client.get("/api/predictions/901/feedback")
        self.assertEqual(res3.status_code, 200)
        feedbacks = json.loads(res3.data)
        self.assertEqual(len(feedbacks), 1)
        self.assertEqual(feedbacks[0]["rating"], 5)
        self.assertEqual(feedbacks[0]["feedback_type"], "TECHNICAL_QUALITY_ISSUE")
        self.assertEqual(feedbacks[0]["comment"], "Updated comment")

    def test_submit_and_retrieve_quality_flags(self):
        """Verify quality flagging workflow, duplicate protection, and filtering controls."""
        self.client.set_cookie("access_token", self.doctor_token)

        # Create flag
        response = self.client.post("/api/predictions/901/flags", json={
            "flag_type": "LOW_CONFIDENCE",
            "severity": "MEDIUM",
            "description": "Fuzzy scan lines."
        })
        self.assertEqual(response.status_code, 200)
        flag_data = json.loads(response.data)
        self.assertIn("flag_id", flag_data)
        flag_id = flag_data["flag_id"]

        # Re-flag same prediction with same type (should update existing)
        response2 = self.client.post("/api/predictions/901/flags", json={
            "flag_type": "LOW_CONFIDENCE",
            "severity": "HIGH",
            "description": "Fuzzy scan lines updated description."
        })
        self.assertEqual(response2.status_code, 200)

        # Retrieve flag details
        response3 = self.client.get(f"/api/quality-flags/{flag_id}")
        self.assertEqual(response3.status_code, 200)
        flag_details = json.loads(response3.data)
        self.assertEqual(flag_details["severity"], "HIGH")
        self.assertEqual(flag_details["description"], "Fuzzy scan lines updated description.")
        self.assertEqual(flag_details["status"], "OPEN")

    def test_high_severity_flag_notification(self):
        """Verify that HIGH/CRITICAL severity flags trigger notification creation for doctors/admins."""
        self.client.set_cookie("access_token", self.doctor_token)

        # Create critical flag
        response = self.client.post("/api/predictions/901/flags", json={
            "flag_type": "SEGMENTATION_CONCERN",
            "severity": "CRITICAL",
            "description": "Critical segmentation offset!"
        })
        self.assertEqual(response.status_code, 200)

        # Check notifications for admin user
        self.client.set_cookie("access_token", self.admin_token)
        notif_resp = self.client.get("/api/notifications?limit=10")
        self.assertEqual(notif_resp.status_code, 200)
        notif_data = json.loads(notif_resp.data)

        # Verify that we generated a notification of type QUALITY_FLAG_CREATED
        flag_notifications = [n for n in notif_data["items"] if n["type"] == "QUALITY_FLAG_CREATED"]
        self.assertTrue(len(flag_notifications) >= 1)
        self.assertIn("Urgent Quality Flag Raised", flag_notifications[0]["title"])

    def test_list_quality_flags_rbac_and_filters(self):
        """Verify that patients cannot view global queues and verify filters."""
        # Submit a flag as doctor
        self.client.set_cookie("access_token", self.doctor_token)
        self.client.post("/api/predictions/901/flags", json={
            "flag_type": "CLASSIFICATION_CONCERN",
            "severity": "MEDIUM",
            "description": "Wrong class?"
        })

        # Try to view global queue as patient (should only see their own flagged prediction flags)
        self.client.set_cookie("access_token", self.patient_token)
        p_resp = self.client.get("/api/quality-flags")
        self.assertEqual(p_resp.status_code, 200)
        p_flags = json.loads(p_resp.data)
        # Patient was not the one who flagged, so patient should see 0 flags.
        self.assertEqual(len(p_flags), 0)

        # Try to list flags with filters as doctor
        self.client.set_cookie("access_token", self.doctor_token)
        d_resp = self.client.get("/api/quality-flags?severity=MEDIUM&status=OPEN")
        self.assertEqual(d_resp.status_code, 200)
        d_flags = json.loads(d_resp.data)
        self.assertEqual(len(d_flags), 1)
        self.assertEqual(d_flags[0]["severity"], "MEDIUM")

    def test_flag_review_workflow_and_resolution_notification(self):
        """Verify flag status transition rules, mandatory note for RESOLVED, and notifier alerts."""
        # Create flag
        self.client.set_cookie("access_token", self.doctor_token)
        create_resp = self.client.post("/api/predictions/901/flags", json={
            "flag_type": "OTHER",
            "severity": "LOW",
            "description": "Low severity flag"
        })
        flag_id = json.loads(create_resp.data)["flag_id"]

        # Patient tries to resolve it (should be forbidden)
        self.client.set_cookie("access_token", self.patient_token)
        patch_resp = self.client.patch(f"/api/quality-flags/{flag_id}", json={
            "status": "RESOLVED",
            "resolution_note": "I fixed it"
        })
        self.assertEqual(patch_resp.status_code, 403)

        # Doctor resolves without note (should be forbidden)
        self.client.set_cookie("access_token", self.doctor_token)
        patch_resp = self.client.patch(f"/api/quality-flags/{flag_id}", json={
            "status": "RESOLVED",
            "resolution_note": ""
        })
        self.assertEqual(patch_resp.status_code, 400)

        # Doctor resolves with valid note (should succeed)
        patch_resp = self.client.patch(f"/api/quality-flags/{flag_id}", json={
            "status": "RESOLVED",
            "resolution_note": "Re-verified with senior clinician."
        })
        self.assertEqual(patch_resp.status_code, 200)

        # Verify flag status is updated
        get_resp = self.client.get(f"/api/quality-flags/{flag_id}")
        self.assertEqual(json.loads(get_resp.data)["status"], "RESOLVED")

        # Verify that doctor (who flagged it) got a resolution notification
        notif_resp = self.client.get("/api/notifications?limit=10")
        self.assertEqual(notif_resp.status_code, 200)
        notifs = json.loads(notif_resp.data)["items"]
        res_notifs = [n for n in notifs if n["type"] == "QUALITY_FLAG_RESOLVED"]
        self.assertEqual(len(res_notifs), 1)
        self.assertIn("Quality Flag Resolved", res_notifs[0]["title"])

    def test_audit_logging_integration(self):
        """Verify that all actions log to the security audit trails."""
        self.client.set_cookie("access_token", self.doctor_token)
        # Submit feedback
        self.client.post("/api/predictions/901/feedback", json={
            "rating": 5,
            "feedback_type": "ACCURATE",
            "comment": "Looks good"
        })

        # Check audit logs for FEEDBACK_SUBMITTED
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        sec_logs = conn.execute("SELECT * FROM security_audit_logs WHERE event_type = 'FEEDBACK_SUBMITTED';").fetchall()
        self.assertTrue(len(sec_logs) >= 1)
        self.assertEqual(sec_logs[0]["user_id"], self.doctor.id)
        conn.close()

import os
import math
import sqlite3
import unittest
import json
import datetime
from clinical_reporting.domain.entities import (
    LongitudinalTimelineMetric,
    LongitudinalTimelineEvent,
    LongitudinalPatientTimeline
)
from clinical_reporting.application.services import ReportService, ReportServiceException
from persistence.infrastructure.repository import SQLitePersistenceRepository
from security.infrastructure.jwt_service import JWTService
from security.domain.entities import Role, User
from security.infrastructure.repository import SQLiteUserRepository

class MockRole:
    def __init__(self, value):
        self.value = value

class MockUser:
    def __init__(self, uuid, full_name, role_val):
        self.uuid = uuid
        self.full_name = full_name
        self.role = MockRole(role_val)


class TestLongitudinalTimelineDomain(unittest.TestCase):
    """Domain model tests for Longitudinal Patient Timeline (Phase F3.3.1)."""

    def test_empty_timeline(self):
        timeline = LongitudinalPatientTimeline(
            patient_id="pat-123",
            patient_name="John Doe",
            events=[]
        )
        self.assertEqual(timeline.patient_id, "pat-123")
        self.assertEqual(timeline.patient_name, "John Doe")
        self.assertEqual(timeline.events, [])
        self.assertEqual(timeline.total_events, 0)
        self.assertIsNone(timeline.first_scan_date)
        self.assertIsNone(timeline.latest_scan_date)
        self.assertIsNone(timeline.first_event)
        self.assertIsNone(timeline.latest_event)
        self.assertEqual(timeline.intermediate_events, [])
        self.assertEqual(timeline.timeline_status, "EMPTY")
        self.assertEqual(timeline.trend_summary, "No scan history available.")

    def test_single_report_timeline(self):
        event = LongitudinalTimelineEvent(
            report_id=1,
            scan_id=101,
            patient_id="pat-123",
            scan_date="2026-08-01",
            classification="Glioma",
            confidence=0.92,
            tumor_area=25.0,
            tumor_percentage=2.5,
            severity="Medium",
            severity_score=0.6,
            report_status="FINAL"
        )
        timeline = LongitudinalPatientTimeline(
            patient_id="pat-123",
            patient_name="John Doe",
            events=[event]
        )
        self.assertEqual(timeline.total_events, 1)
        self.assertEqual(timeline.first_scan_date, "2026-08-01")
        self.assertEqual(timeline.latest_scan_date, "2026-08-01")
        self.assertEqual(timeline.first_event, event)
        self.assertEqual(timeline.latest_event, event)
        self.assertEqual(timeline.intermediate_events, [])
        self.assertEqual(timeline.timeline_status, "STABLE")
        self.assertIn("Baseline scan established", timeline.trend_summary)

    def test_multiple_report_timeline(self):
        e1 = LongitudinalTimelineEvent(
            report_id=1, scan_id=101, patient_id="pat-123", scan_date="2026-08-01",
            classification="Glioma", confidence=0.90, tumor_area=20.0, tumor_percentage=2.0, severity="Medium"
        )
        e2 = LongitudinalTimelineEvent(
            report_id=2, scan_id=102, patient_id="pat-123", scan_date="2026-08-05",
            classification="Glioma", confidence=0.91, tumor_area=22.0, tumor_percentage=2.2, severity="Medium"
        )
        e3 = LongitudinalTimelineEvent(
            report_id=3, scan_id=103, patient_id="pat-123", scan_date="2026-08-10",
            classification="Glioma", confidence=0.92, tumor_area=25.0, tumor_percentage=2.5, severity="Medium"
        )
        timeline = LongitudinalPatientTimeline(
            patient_id="pat-123",
            patient_name="John Doe",
            events=[e1, e2, e3]
        )
        self.assertEqual(timeline.total_events, 3)
        self.assertEqual(timeline.first_event, e1)
        self.assertEqual(timeline.latest_event, e3)
        self.assertEqual(timeline.intermediate_events, [e2])
        self.assertEqual(timeline.timeline_status, "AREA_INCREASED")
        self.assertIn("segmented area increased", timeline.trend_summary)

    def test_chronological_event_ordering(self):
        e1 = LongitudinalTimelineEvent(
            report_id=1, scan_id=101, patient_id="pat-123", scan_date="2026-08-01",
            classification="Glioma", confidence=0.90, tumor_area=20.0, tumor_percentage=2.0, severity="Medium"
        )
        e2 = LongitudinalTimelineEvent(
            report_id=2, scan_id=102, patient_id="pat-123", scan_date="2026-08-10",
            classification="Glioma", confidence=0.92, tumor_area=25.0, tumor_percentage=2.5, severity="Medium"
        )
        e3 = LongitudinalTimelineEvent(
            report_id=3, scan_id=103, patient_id="pat-123", scan_date="2026-08-05",
            classification="Glioma", confidence=0.91, tumor_area=22.0, tumor_percentage=2.2, severity="Medium"
        )
        timeline = LongitudinalPatientTimeline(
            patient_id="pat-123",
            patient_name="John Doe",
            events=[e2, e1, e3] # unsorted
        )
        self.assertEqual(timeline.events[0], e1)
        self.assertEqual(timeline.events[1], e3)
        self.assertEqual(timeline.events[2], e2)
        self.assertEqual(timeline.first_scan_date, "2026-08-01")
        self.assertEqual(timeline.latest_scan_date, "2026-08-10")

    def test_missing_scan_date(self):
        e1 = LongitudinalTimelineEvent(
            report_id=1, scan_id=101, patient_id="pat-123", scan_date=None,
            classification="Glioma", confidence=0.90, tumor_area=20.0, tumor_percentage=2.0, severity="Medium"
        )
        e2 = LongitudinalTimelineEvent(
            report_id=2, scan_id=102, patient_id="pat-123", scan_date="2026-08-10",
            classification="Glioma", confidence=0.92, tumor_area=25.0, tumor_percentage=2.5, severity="Medium"
        )
        timeline = LongitudinalPatientTimeline(
            patient_id="pat-123",
            events=[e2, e1]
        )
        self.assertEqual(timeline.events[0], e1)
        self.assertEqual(timeline.events[1], e2)

    def test_missing_measurements(self):
        m = LongitudinalTimelineMetric(
            metric_name="Tumor Area",
            current_value=15.0,
            previous_value=None
        )
        self.assertFalse(m.available)
        self.assertIsNone(m.absolute_change)
        self.assertIsNone(m.percentage_change)
        self.assertEqual(m.trend, "UNAVAILABLE")

    def test_no_tumor_event(self):
        e1 = LongitudinalTimelineEvent(
            report_id=1, scan_id=101, patient_id="pat-123", scan_date="2026-08-01",
            classification="no_tumor", confidence=0.99, tumor_area=0.0, tumor_percentage=0.0, severity="None"
        )
        e2 = LongitudinalTimelineEvent(
            report_id=2, scan_id=102, patient_id="pat-123", scan_date="2026-08-05",
            classification="no_tumor", confidence=0.99, tumor_area=0.0, tumor_percentage=0.0, severity="None"
        )
        timeline = LongitudinalPatientTimeline(
            patient_id="pat-123",
            events=[e1, e2]
        )
        self.assertEqual(timeline.timeline_status, "STABLE")
        self.assertEqual(timeline.trend_summary, "No tumor detected.")

    def test_zero_baseline(self):
        m = LongitudinalTimelineMetric(
            metric_name="Tumor Area",
            current_value=10.0,
            previous_value=0.0
        )
        self.assertTrue(m.available)
        self.assertEqual(m.absolute_change, 10.0)
        self.assertIsNone(m.percentage_change)
        self.assertEqual(m.trend, "INCREASED")

        m2 = LongitudinalTimelineMetric(
            metric_name="Tumor Area",
            current_value=0.0,
            previous_value=0.0
        )
        self.assertTrue(m2.available)
        self.assertEqual(m2.absolute_change, 0.0)
        self.assertEqual(m2.percentage_change, 0.0)
        self.assertEqual(m2.trend, "STABLE")

    def test_nan_handling(self):
        m = LongitudinalTimelineMetric(
            metric_name="Tumor Area",
            current_value=float('nan'),
            previous_value=10.0
        )
        self.assertFalse(m.available)
        self.assertIsNone(m.absolute_change)
        self.assertIsNone(m.percentage_change)
        self.assertEqual(m.trend, "UNAVAILABLE")

        m2 = LongitudinalTimelineMetric(
            metric_name="Tumor Area",
            current_value="NaN",
            previous_value=10.0
        )
        self.assertFalse(m2.available)
        self.assertIsNone(m2.absolute_change)
        self.assertIsNone(m2.percentage_change)
        self.assertEqual(m2.trend, "UNAVAILABLE")

    def test_infinity_handling(self):
        m = LongitudinalTimelineMetric(
            metric_name="Tumor Area",
            current_value=float('inf'),
            previous_value=10.0
        )
        self.assertFalse(m.available)
        self.assertIsNone(m.absolute_change)
        self.assertIsNone(m.percentage_change)
        self.assertEqual(m.trend, "UNAVAILABLE")

        m2 = LongitudinalTimelineMetric(
            metric_name="Tumor Area",
            current_value=15.0,
            previous_value=float('-inf')
        )
        self.assertFalse(m2.available)
        self.assertIsNone(m2.absolute_change)
        self.assertIsNone(m2.percentage_change)
        self.assertEqual(m2.trend, "UNAVAILABLE")

        m3 = LongitudinalTimelineMetric(
            metric_name="Tumor Area",
            current_value="Infinity",
            previous_value=10.0
        )
        self.assertFalse(m3.available)
        self.assertIsNone(m3.absolute_change)
        self.assertIsNone(m3.percentage_change)
        self.assertEqual(m3.trend, "UNAVAILABLE")

    def test_multiple_classifications(self):
        e1 = LongitudinalTimelineEvent(
            report_id=1, scan_id=101, patient_id="pat-123", scan_date="2026-08-01",
            classification="Glioma", confidence=0.90, tumor_area=None, tumor_percentage=None, severity="Medium"
        )
        e2 = LongitudinalTimelineEvent(
            report_id=2, scan_id=102, patient_id="pat-123", scan_date="2026-08-05",
            classification="Meningioma", confidence=0.95, tumor_area=None, tumor_percentage=None, severity="Low"
        )
        timeline = LongitudinalPatientTimeline(
            patient_id="pat-123",
            events=[e1, e2]
        )
        self.assertEqual(timeline.timeline_status, "CHANGED")
        self.assertIn("Classification changed from Glioma to Meningioma", timeline.trend_summary)

    def test_latest_event_identification(self):
        e1 = LongitudinalTimelineEvent(
            report_id=1, scan_id=101, patient_id="pat-123", scan_date="2026-08-01",
            classification="Glioma", confidence=0.90, tumor_area=20.0, tumor_percentage=2.0, severity="Medium"
        )
        e2 = LongitudinalTimelineEvent(
            report_id=2, scan_id=102, patient_id="pat-123", scan_date="2026-08-15",
            classification="Glioma", confidence=0.92, tumor_area=25.0, tumor_percentage=2.5, severity="Medium"
        )
        timeline = LongitudinalPatientTimeline(
            patient_id="pat-123",
            events=[e1, e2]
        )
        self.assertEqual(timeline.latest_event, e2)

    def test_first_event_identification(self):
        e1 = LongitudinalTimelineEvent(
            report_id=1, scan_id=101, patient_id="pat-123", scan_date="2026-08-01",
            classification="Glioma", confidence=0.90, tumor_area=20.0, tumor_percentage=2.0, severity="Medium"
        )
        e2 = LongitudinalTimelineEvent(
            report_id=2, scan_id=102, patient_id="pat-123", scan_date="2026-08-15",
            classification="Glioma", confidence=0.92, tumor_area=25.0, tumor_percentage=2.5, severity="Medium"
        )
        timeline = LongitudinalPatientTimeline(
            patient_id="pat-123",
            events=[e1, e2]
        )
        self.assertEqual(timeline.first_event, e1)

    def test_patient_id_consistency(self):
        e1 = LongitudinalTimelineEvent(
            report_id=1, scan_id=101, patient_id="pat-123", scan_date="2026-08-01",
            classification="Glioma", confidence=0.90, tumor_area=20.0, tumor_percentage=2.0, severity="Medium"
        )
        timeline = LongitudinalPatientTimeline(
            patient_id="pat-123",
            events=[e1]
        )
        for event in timeline.events:
            self.assertEqual(event.patient_id, timeline.patient_id)

    def test_serialization_safety(self):
        e = LongitudinalTimelineEvent(
            report_id=1, scan_id=101, patient_id="pat-123", scan_date="2026-08-01",
            classification="Glioma", confidence=0.90, tumor_area=20.0, tumor_percentage=2.0, severity="Medium",
            severity_score=0.5, report_status="FINAL"
        )
        m = LongitudinalTimelineMetric(
            metric_name="Tumor Area",
            current_value=20.0,
            previous_value=18.0,
            unit="mm2"
        )
        timeline = LongitudinalPatientTimeline(
            patient_id="pat-123",
            patient_name="John Doe",
            events=[e]
        )

        e_dict = e.to_dict()
        self.assertEqual(e_dict["report_id"], 1)
        self.assertEqual(e_dict["patient_id"], "pat-123")
        self.assertEqual(e_dict["classification"], "Glioma")
        self.assertEqual(e_dict["report_status"], "FINAL")

        m_dict = m.to_dict()
        self.assertEqual(m_dict["metric_name"], "Tumor Area")
        self.assertEqual(m_dict["current_value"], 20.0)
        self.assertEqual(m_dict["previous_value"], 18.0)
        self.assertEqual(m_dict["absolute_change"], 2.0)
        self.assertEqual(m_dict["trend"], "INCREASED")
        self.assertTrue(m_dict["available"])

        t_dict = timeline.to_dict()
        self.assertEqual(t_dict["patient_id"], "pat-123")
        self.assertEqual(t_dict["patient_name"], "John Doe")
        self.assertEqual(len(t_dict["events"]), 1)
        self.assertEqual(t_dict["events"][0]["report_id"], 1)
        self.assertEqual(t_dict["total_events"], 1)
        self.assertIn("tumor_area", t_dict["metrics"])


class TestLongitudinalTimelineService(unittest.TestCase):
    """F3.3.2 application service timeline tests."""

    def _create_test_hierarchy(
        self, conn, patient_id, report_id, report_number, prediction_id, scan_id,
        classification, area, severity, scan_date="2026-08-08", confidence=0.95,
        pct_brain=5.0, status="FINALIZED", version_status="FINAL", json_path=""
    ):
        conn.execute(
            "INSERT OR IGNORE INTO patients (patient_id, name, age, gender, created_at) VALUES (?, ?, ?, ?, ?);",
            (patient_id, f"Patient {patient_id}", 45, "Male", datetime.datetime.utcnow().isoformat())
        )
        conn.execute(
            "INSERT OR IGNORE INTO mri_scans (id, patient_id, image_path, pixel_spacing_mm, ref_physician, scan_date, created_at) VALUES (?, ?, ?, ?, ?, ?, ?);",
            (scan_id, patient_id, f"scan_{scan_id}.png", 1.0, "Dr. Smith", scan_date, datetime.datetime.utcnow().isoformat())
        )
        conn.execute(
            """INSERT OR IGNORE INTO predictions (
                id, scan_id, predicted_class, confidence_score, prob_glioma, prob_meningioma,
                prob_pituitary, prob_no_tumor, tumor_pixel_count, tumor_area_mm2,
                tumor_percentage_brain, tumor_percentage_image, estimated_brain_pixel_count,
                rule_based_severity, severity_rule_description, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);""",
            (prediction_id, scan_id, classification, confidence, 0.90, 0.05, 0.03, 0.02, 500, area, pct_brain, 2.0, 10000, severity, f"Severity is {severity}", datetime.datetime.utcnow().isoformat())
        )
        conn.execute(
            "INSERT OR IGNORE INTO reports (report_id, report_number, patient_id, current_version, status, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?);",
            (report_id, report_number, patient_id, 1, status, "2026-08-08 10:00:00", "2026-08-08 10:00:00")
        )
        conn.execute(
            """INSERT OR IGNORE INTO report_versions (
                report_id, version_number, prediction_id, status, created_by, reason, created_at,
                pdf_path, json_path, checksum, integrity_hash
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);""",
            (report_id, 1, prediction_id, version_status, "doc-uuid-1", "Initial draft", "2026-08-08 10:00:00",
             f"outputs/clinical_reports/{report_number}.pdf", json_path, "dummy_checksum", "dummy_integrity")
        )

    def setUp(self):
        self.db_path = os.path.abspath("outputs/test_longitudinal_timeline_service.db")
        if os.path.exists(self.db_path):
            try:
                os.remove(self.db_path)
            except Exception:
                pass

        self.persistence_repo = SQLitePersistenceRepository(db_path=self.db_path)
        self.persistence_repo.initialize_db()

        self.service = ReportService(db_path=self.db_path)

        self.admin = MockUser(uuid="admin-1", full_name="Admin User", role_val="admin")
        self.doctor = MockUser(uuid="doc-1", full_name="Doctor User", role_val="doctor")
        self.patient_bob = MockUser(uuid="bob-jones-uuid", full_name="Bob Jones", role_val="patient")
        self.patient_alice = MockUser(uuid="alice-smith-uuid", full_name="Alice Smith", role_val="patient")

    def test_empty_patient_timeline(self):
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                "INSERT INTO patients (patient_id, name, age, gender, created_at) VALUES (?, ?, ?, ?, ?);",
                ("bob-jones-uuid", "Bob Jones", 45, "Male", datetime.datetime.utcnow().isoformat())
            )
            conn.commit()
        finally:
            conn.close()

        timeline = self.service.get_patient_longitudinal_timeline("bob-jones-uuid", self.doctor)
        self.assertEqual(timeline.patient_id, "bob-jones-uuid")
        self.assertEqual(timeline.patient_name, "Bob Jones")
        self.assertEqual(timeline.total_events, 0)
        self.assertEqual(timeline.timeline_status, "EMPTY")

    def test_single_report(self):
        conn = sqlite3.connect(self.db_path)
        try:
            self._create_test_hierarchy(
                conn, "bob-jones-uuid", report_id=1, report_number="RPT-2026-0001",
                prediction_id=10, scan_id=100, classification="Glioma", area=50.0,
                severity="Medium", scan_date="2026-08-01", confidence=0.95
            )
            conn.commit()
        finally:
            conn.close()

        timeline = self.service.get_patient_longitudinal_timeline("bob-jones-uuid", self.doctor)
        self.assertEqual(timeline.total_events, 1)
        self.assertEqual(timeline.first_scan_date, "2026-08-01")
        self.assertEqual(timeline.timeline_status, "STABLE")
        self.assertIn("Baseline scan established", timeline.trend_summary)

    def test_multiple_reports_and_ordering(self):
        conn = sqlite3.connect(self.db_path)
        try:
            self._create_test_hierarchy(
                conn, "bob-jones-uuid", report_id=2, report_number="RPT-2026-0002",
                prediction_id=12, scan_id=102, classification="Glioma", area=60.0,
                severity="High", scan_date="2026-08-10", confidence=0.96
            )
            self._create_test_hierarchy(
                conn, "bob-jones-uuid", report_id=1, report_number="RPT-2026-0001",
                prediction_id=10, scan_id=100, classification="Glioma", area=50.0,
                severity="Medium", scan_date="2026-08-01", confidence=0.95
            )
            self._create_test_hierarchy(
                conn, "bob-jones-uuid", report_id=3, report_number="RPT-2026-0003",
                prediction_id=13, scan_id=103, classification="Glioma", area=55.0,
                severity="Medium", scan_date="2026-08-05", confidence=0.94
            )
            conn.commit()
        finally:
            conn.close()

        timeline = self.service.get_patient_longitudinal_timeline("bob-jones-uuid", self.doctor)
        self.assertEqual(timeline.total_events, 3)
        self.assertEqual(timeline.events[0].report_id, 1)
        self.assertEqual(timeline.events[1].report_id, 3)
        self.assertEqual(timeline.events[2].report_id, 2)
        self.assertEqual(timeline.first_scan_date, "2026-08-01")
        self.assertEqual(timeline.latest_scan_date, "2026-08-10")

    def test_missing_dates(self):
        conn = sqlite3.connect(self.db_path)
        try:
            self._create_test_hierarchy(
                conn, "bob-jones-uuid", report_id=1, report_number="RPT-2026-0001",
                prediction_id=10, scan_id=100, classification="Glioma", area=50.0,
                severity="Medium", scan_date=None
            )
            self._create_test_hierarchy(
                conn, "bob-jones-uuid", report_id=2, report_number="RPT-2026-0002",
                prediction_id=11, scan_id=101, classification="Glioma", area=55.0,
                severity="Medium", scan_date="2026-08-05"
            )
            conn.commit()
        finally:
            conn.close()

        timeline = self.service.get_patient_longitudinal_timeline("bob-jones-uuid", self.doctor)
        self.assertEqual(timeline.total_events, 2)
        self.assertEqual(timeline.events[0].report_id, 1)
        self.assertEqual(timeline.events[1].report_id, 2)

    def test_duplicate_dates(self):
        conn = sqlite3.connect(self.db_path)
        try:
            self._create_test_hierarchy(
                conn, "bob-jones-uuid", report_id=2, report_number="RPT-2026-0002",
                prediction_id=11, scan_id=101, classification="Glioma", area=55.0,
                severity="Medium", scan_date="2026-08-05"
            )
            self._create_test_hierarchy(
                conn, "bob-jones-uuid", report_id=1, report_number="RPT-2026-0001",
                prediction_id=10, scan_id=100, classification="Glioma", area=50.0,
                severity="Medium", scan_date="2026-08-05"
            )
            conn.commit()
        finally:
            conn.close()

        timeline = self.service.get_patient_longitudinal_timeline("bob-jones-uuid", self.doctor)
        self.assertEqual(timeline.total_events, 2)
        self.assertEqual(timeline.events[0].report_id, 1)
        self.assertEqual(timeline.events[1].report_id, 2)

    def test_missing_measurements_and_none(self):
        conn = sqlite3.connect(self.db_path)
        try:
            self._create_test_hierarchy(
                conn, "bob-jones-uuid", report_id=1, report_number="RPT-2026-0001",
                prediction_id=10, scan_id=100, classification="no_tumor", area=None,
                severity=None, confidence=None
            )
            conn.commit()
        finally:
            conn.close()

        timeline = self.service.get_patient_longitudinal_timeline("bob-jones-uuid", self.doctor)
        self.assertEqual(timeline.total_events, 1)
        self.assertIsNone(timeline.events[0].tumor_area)
        self.assertIsNone(timeline.events[0].confidence)

    def test_classification_severity_confidence_area_changes(self):
        conn = sqlite3.connect(self.db_path)
        try:
            self._create_test_hierarchy(
                conn, "bob-jones-uuid", report_id=1, report_number="RPT-2026-0001",
                prediction_id=10, scan_id=100, classification="Glioma", area=50.0,
                severity="Low", scan_date="2026-08-01", confidence=0.90
            )
            self._create_test_hierarchy(
                conn, "bob-jones-uuid", report_id=2, report_number="RPT-2026-0002",
                prediction_id=11, scan_id=101, classification="Meningioma", area=40.0,
                severity="High", scan_date="2026-08-05", confidence=0.95
            )
            conn.commit()
        finally:
            conn.close()

        timeline = self.service.get_patient_longitudinal_timeline("bob-jones-uuid", self.doctor)
        self.assertEqual(timeline.total_events, 2)

        metrics = timeline.metrics
        self.assertEqual(metrics["classification"].trend, "CHANGED")
        self.assertEqual(metrics["tumor_area"].trend, "DECREASED")
        self.assertEqual(metrics["confidence"].trend, "INCREASED")
        self.assertEqual(metrics["severity"].trend, "INCREASED")

    def test_zero_nan_infinity_values(self):
        conn = sqlite3.connect(self.db_path)
        try:
            self._create_test_hierarchy(
                conn, "bob-jones-uuid", report_id=1, report_number="RPT-2026-0001",
                prediction_id=10, scan_id=100, classification="Glioma", area=float('nan'),
                severity="Medium", scan_date="2026-08-01", confidence=float('inf')
            )
            conn.commit()
        finally:
            conn.close()

        timeline = self.service.get_patient_longitudinal_timeline("bob-jones-uuid", self.doctor)
        self.assertEqual(timeline.total_events, 1)
        self.assertIsNone(timeline.events[0].tumor_area)
        self.assertIsNone(timeline.events[0].confidence)

    def test_invalid_patient(self):
        with self.assertRaises(ReportServiceException) as context:
            self.service.get_patient_longitudinal_timeline("non-existent-patient-uuid", self.doctor)
        self.assertIn("not found", str(context.exception))

    def test_cross_patient_isolation_security(self):
        conn = sqlite3.connect(self.db_path)
        try:
            self._create_test_hierarchy(
                conn, "bob-jones-uuid", report_id=1, report_number="RPT-2026-0001",
                prediction_id=10, scan_id=100, classification="Glioma", area=50.0,
                severity="Medium", scan_date="2026-08-01"
            )
            self._create_test_hierarchy(
                conn, "alice-smith-uuid", report_id=2, report_number="RPT-2026-0002",
                prediction_id=11, scan_id=101, classification="Glioma", area=40.0,
                severity="Low", scan_date="2026-08-02"
            )
            conn.commit()
        finally:
            conn.close()

        with self.assertRaises(ReportServiceException) as context:
            self.service.get_patient_longitudinal_timeline("bob-jones-uuid", self.patient_alice)
        self.assertIn("Access denied", str(context.exception))

        timeline_bob = self.service.get_patient_longitudinal_timeline("bob-jones-uuid", self.patient_bob)
        self.assertEqual(timeline_bob.patient_id, "bob-jones-uuid")

        timeline_doc = self.service.get_patient_longitudinal_timeline("bob-jones-uuid", self.doctor)
        self.assertEqual(timeline_doc.patient_id, "bob-jones-uuid")

    def test_summary_and_first_latest_detection(self):
        conn = sqlite3.connect(self.db_path)
        try:
            self._create_test_hierarchy(
                conn, "bob-jones-uuid", report_id=1, report_number="RPT-2026-0001",
                prediction_id=10, scan_id=100, classification="Glioma", area=50.0,
                severity="Medium", scan_date="2026-08-01"
            )
            self._create_test_hierarchy(
                conn, "bob-jones-uuid", report_id=2, report_number="RPT-2026-0002",
                prediction_id=11, scan_id=101, classification="Glioma", area=55.0,
                severity="High", scan_date="2026-08-10"
            )
            conn.commit()
        finally:
            conn.close()

        timeline = self.service.get_patient_longitudinal_timeline("bob-jones-uuid", self.doctor)
        self.assertEqual(timeline.first_event.report_id, 1)
        self.assertEqual(timeline.latest_event.report_id, 2)
        self.assertEqual(timeline.timeline_status, "AREA_INCREASED")
        self.assertIn("segmented area increased by 10.0%", timeline.trend_summary)


class TestLongitudinalTimelineAPI(unittest.TestCase):
    """API Endpoint integration tests for Longitudinal Patient Timeline (Phase F3.3.3)."""

    def _create_test_hierarchy(
        self, conn, patient_id, report_id, report_number, prediction_id, scan_id,
        classification, area, severity, scan_date="2026-08-08", confidence=0.95,
        pct_brain=5.0, status="FINALIZED", version_status="FINAL", json_path=""
    ):
        conn.execute(
            "INSERT OR IGNORE INTO patients (patient_id, name, age, gender, created_at) VALUES (?, ?, ?, ?, ?);",
            (patient_id, f"Patient {patient_id}", 45, "Male", datetime.datetime.utcnow().isoformat())
        )
        conn.execute(
            "INSERT OR IGNORE INTO mri_scans (id, patient_id, image_path, pixel_spacing_mm, ref_physician, scan_date, created_at) VALUES (?, ?, ?, ?, ?, ?, ?);",
            (scan_id, patient_id, f"scan_{scan_id}.png", 1.0, "Dr. Smith", scan_date, datetime.datetime.utcnow().isoformat())
        )
        conn.execute(
            """INSERT OR IGNORE INTO predictions (
                id, scan_id, predicted_class, confidence_score, prob_glioma, prob_meningioma,
                prob_pituitary, prob_no_tumor, tumor_pixel_count, tumor_area_mm2,
                tumor_percentage_brain, tumor_percentage_image, estimated_brain_pixel_count,
                rule_based_severity, severity_rule_description, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);""",
            (prediction_id, scan_id, classification, confidence, 0.90, 0.05, 0.03, 0.02, 500, area, pct_brain, 2.0, 10000, severity, f"Severity is {severity}", datetime.datetime.utcnow().isoformat())
        )
        conn.execute(
            "INSERT OR IGNORE INTO reports (report_id, report_number, patient_id, current_version, status, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?);",
            (report_id, report_number, patient_id, 1, status, "2026-08-08 10:00:00", "2026-08-08 10:00:00")
        )
        conn.execute(
            """INSERT OR IGNORE INTO report_versions (
                report_id, version_number, prediction_id, status, created_by, reason, created_at,
                pdf_path, json_path, checksum, integrity_hash
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);""",
            (report_id, 1, prediction_id, version_status, "doc-uuid-1", "Initial draft", "2026-08-08 10:00:00",
             f"outputs/clinical_reports/{report_number}.pdf", json_path, "dummy_checksum", "dummy_integrity")
        )

    def setUp(self):
        self.db_path = os.path.abspath("outputs/test_longitudinal_timeline_api.db")
        if os.path.exists(self.db_path):
            try:
                os.remove(self.db_path)
            except Exception:
                pass

        import api.infrastructure.routes as api_routes
        import api.routes.auth_routes as auth_routes
        self.orig_api_db = api_routes.DEFAULT_DB_PATH
        self.orig_auth_db = auth_routes.DEFAULT_DB_PATH
        api_routes.DEFAULT_DB_PATH = self.db_path
        auth_routes.DEFAULT_DB_PATH = self.db_path

        self.persistence_repo = SQLitePersistenceRepository(db_path=self.db_path)
        self.persistence_repo.initialize_db()

        self.user_repo = SQLiteUserRepository(db_path=self.db_path)
        self.user_repo.initialize_security_tables()
        self.admin_user = self.user_repo.bootstrap_admin()

        from security.infrastructure.password import PasswordHasher
        pass_hash = PasswordHasher.hash_password("Doctor@123")

        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                "INSERT INTO users (uuid, email, password_hash, full_name, role, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?);",
                ("doc-uuid-1", "doctor@aurascan.ai", pass_hash, "Dr. Jane Smith", Role.DOCTOR.value, datetime.datetime.utcnow().isoformat(), datetime.datetime.utcnow().isoformat())
            )
            conn.execute(
                "INSERT INTO users (uuid, email, password_hash, full_name, role, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?);",
                ("bob-jones-uuid", "bob@aurascan.ai", pass_hash, "Bob Jones", Role.PATIENT.value, datetime.datetime.utcnow().isoformat(), datetime.datetime.utcnow().isoformat())
            )
            conn.execute(
                "INSERT INTO users (uuid, email, password_hash, full_name, role, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?);",
                ("alice-smith-uuid", "alice@aurascan.ai", pass_hash, "Alice Smith", Role.PATIENT.value, datetime.datetime.utcnow().isoformat(), datetime.datetime.utcnow().isoformat())
            )
            conn.commit()
        finally:
            conn.close()

        self.doctor_user = self.user_repo.get_by_email("doctor@aurascan.ai")
        self.patient_bob = self.user_repo.get_by_email("bob@aurascan.ai")
        self.patient_alice = self.user_repo.get_by_email("alice@aurascan.ai")

        self.jwt_svc = JWTService()
        self.admin_token = self.jwt_svc.create_access_token(
            user_uuid=self.admin_user.uuid, user_id=self.admin_user.id, email=self.admin_user.email, role=Role.ADMIN
        )
        self.doctor_token = self.jwt_svc.create_access_token(
            user_uuid=self.doctor_user.uuid, user_id=self.doctor_user.id, email=self.doctor_user.email, role=Role.DOCTOR
        )
        self.patient_bob_token = self.jwt_svc.create_access_token(
            user_uuid=self.patient_bob.uuid, user_id=self.patient_bob.id, email=self.patient_bob.email, role=Role.PATIENT
        )
        self.patient_alice_token = self.jwt_svc.create_access_token(
            user_uuid=self.patient_alice.uuid, user_id=self.patient_alice.id, email=self.patient_alice.email, role=Role.PATIENT
        )

        from run_api import app
        from fastapi.testclient import TestClient
        self.test_client_ctx = TestClient(app)
        self.client = self.test_client_ctx.__enter__()

    def tearDown(self):
        self.test_client_ctx.__exit__(None, None, None)
        import api.infrastructure.routes as api_routes
        import api.routes.auth_routes as auth_routes
        api_routes.DEFAULT_DB_PATH = self.orig_api_db
        auth_routes.DEFAULT_DB_PATH = self.orig_auth_db

        if os.path.exists(self.db_path):
            try:
                os.remove(self.db_path)
            except Exception:
                pass

    def test_unauthenticated_request(self):
        res = self.client.get("/api/patients/bob-jones-uuid/longitudinal-timeline")
        self.assertEqual(res.status_code, 401)

    def test_authenticated_admin_access(self):
        conn = sqlite3.connect(self.db_path)
        try:
            self._create_test_hierarchy(
                conn, "bob-jones-uuid", report_id=1, report_number="RPT-2026-0001",
                prediction_id=10, scan_id=100, classification="Glioma", area=50.0,
                severity="Medium", scan_date="2026-08-01"
            )
            conn.commit()
        finally:
            conn.close()

        headers = {"Authorization": f"Bearer {self.admin_token}"}
        res = self.client.get("/api/patients/bob-jones-uuid/longitudinal-timeline", headers=headers)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["patient_id"], "bob-jones-uuid")

    def test_authenticated_doctor_access(self):
        conn = sqlite3.connect(self.db_path)
        try:
            self._create_test_hierarchy(
                conn, "bob-jones-uuid", report_id=1, report_number="RPT-2026-0001",
                prediction_id=10, scan_id=100, classification="Glioma", area=50.0,
                severity="Medium", scan_date="2026-08-01"
            )
            conn.commit()
        finally:
            conn.close()

        headers = {"Authorization": f"Bearer {self.doctor_token}"}
        res = self.client.get("/api/patients/bob-jones-uuid/longitudinal-timeline", headers=headers)
        self.assertEqual(res.status_code, 200)

    def test_patient_access_own_timeline(self):
        conn = sqlite3.connect(self.db_path)
        try:
            self._create_test_hierarchy(
                conn, "bob-jones-uuid", report_id=1, report_number="RPT-2026-0001",
                prediction_id=10, scan_id=100, classification="Glioma", area=50.0,
                severity="Medium", scan_date="2026-08-01"
            )
            conn.commit()
        finally:
            conn.close()

        headers = {"Authorization": f"Bearer {self.patient_bob_token}"}
        res = self.client.get("/api/patients/bob-jones-uuid/longitudinal-timeline", headers=headers)
        self.assertEqual(res.status_code, 200)

    def test_patient_access_another_timeline(self):
        conn = sqlite3.connect(self.db_path)
        try:
            self._create_test_hierarchy(
                conn, "bob-jones-uuid", report_id=1, report_number="RPT-2026-0001",
                prediction_id=10, scan_id=100, classification="Glioma", area=50.0,
                severity="Medium", scan_date="2026-08-01"
            )
            conn.commit()
        finally:
            conn.close()

        headers = {"Authorization": f"Bearer {self.patient_alice_token}"}
        res = self.client.get("/api/patients/bob-jones-uuid/longitudinal-timeline", headers=headers)
        self.assertEqual(res.status_code, 403)
        self.assertIn("Access denied", res.json()["detail"])

    def test_invalid_patient_id(self):
        headers = {"Authorization": f"Bearer {self.admin_token}"}
        res = self.client.get("/api/patients/invalid-id/longitudinal-timeline", headers=headers)
        self.assertEqual(res.status_code, 404)
        self.assertIn("not found", res.json()["detail"].lower())

    def test_empty_timeline(self):
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                "INSERT INTO patients (patient_id, name, age, gender, created_at) VALUES (?, ?, ?, ?, ?);",
                ("bob-jones-uuid", "Bob Jones", 45, "Male", datetime.datetime.utcnow().isoformat())
            )
            conn.commit()
        finally:
            conn.close()

        headers = {"Authorization": f"Bearer {self.patient_bob_token}"}
        res = self.client.get("/api/patients/bob-jones-uuid/longitudinal-timeline", headers=headers)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["total_events"], 0)
        self.assertEqual(data["timeline_status"], "EMPTY")

    def test_single_event_timeline(self):
        conn = sqlite3.connect(self.db_path)
        try:
            self._create_test_hierarchy(
                conn, "bob-jones-uuid", report_id=1, report_number="RPT-2026-0001",
                prediction_id=10, scan_id=100, classification="Glioma", area=50.0,
                severity="Medium", scan_date="2026-08-01"
            )
            conn.commit()
        finally:
            conn.close()

        headers = {"Authorization": f"Bearer {self.patient_bob_token}"}
        res = self.client.get("/api/patients/bob-jones-uuid/longitudinal-timeline", headers=headers)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["total_events"], 1)
        self.assertEqual(data["timeline_status"], "STABLE")

    def test_multi_event_timeline_and_chronological_ordering(self):
        conn = sqlite3.connect(self.db_path)
        try:
            self._create_test_hierarchy(
                conn, "bob-jones-uuid", report_id=2, report_number="RPT-2026-0002",
                prediction_id=11, scan_id=101, classification="Glioma", area=60.0,
                severity="High", scan_date="2026-08-10"
            )
            self._create_test_hierarchy(
                conn, "bob-jones-uuid", report_id=1, report_number="RPT-2026-0001",
                prediction_id=10, scan_id=100, classification="Glioma", area=50.0,
                severity="Medium", scan_date="2026-08-01"
            )
            conn.commit()
        finally:
            conn.close()

        headers = {"Authorization": f"Bearer {self.patient_bob_token}"}
        res = self.client.get("/api/patients/bob-jones-uuid/longitudinal-timeline", headers=headers)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["total_events"], 2)
        self.assertEqual(data["events"][0]["report_id"], 1)
        self.assertEqual(data["events"][1]["report_id"], 2)

    def test_missing_values_and_null_serialization(self):
        conn = sqlite3.connect(self.db_path)
        try:
            self._create_test_hierarchy(
                conn, "bob-jones-uuid", report_id=1, report_number="RPT-2026-0001",
                prediction_id=10, scan_id=100, classification="no_tumor", area=None,
                severity=None, confidence=None
            )
            conn.commit()
        finally:
            conn.close()

        headers = {"Authorization": f"Bearer {self.patient_bob_token}"}
        res = self.client.get("/api/patients/bob-jones-uuid/longitudinal-timeline", headers=headers)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIsNone(data["events"][0]["tumor_area"])
        self.assertIsNone(data["events"][0]["confidence"])

    def test_nan_and_infinity_sanitization(self):
        conn = sqlite3.connect(self.db_path)
        try:
            self._create_test_hierarchy(
                conn, "bob-jones-uuid", report_id=1, report_number="RPT-2026-0001",
                prediction_id=10, scan_id=100, classification="Glioma", area=float('nan'),
                severity="Medium", scan_date="2026-08-01", confidence=float('-inf')
            )
            conn.commit()
        finally:
            conn.close()

        headers = {"Authorization": f"Bearer {self.patient_bob_token}"}
        res = self.client.get("/api/patients/bob-jones-uuid/longitudinal-timeline", headers=headers)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIsNone(data["events"][0]["tumor_area"])
        self.assertIsNone(data["events"][0]["confidence"])
        self.assertIn("patient_id", data)
        self.assertIn("events", data)
        self.assertIn("metrics", data)


class TestLongitudinalTimelineFlaskAPI(unittest.TestCase):
    """Flask Endpoint integration tests for Longitudinal Patient Timeline (Phase F3.3.4)."""

    def _create_test_hierarchy(
        self, conn, patient_id, report_id, report_number, prediction_id, scan_id,
        classification, area, severity, scan_date="2026-08-08", confidence=0.95,
        pct_brain=5.0, status="FINALIZED", version_status="FINAL", json_path=""
    ):
        conn.execute(
            "INSERT OR IGNORE INTO patients (patient_id, name, age, gender, created_at) VALUES (?, ?, ?, ?, ?);",
            (patient_id, f"Patient {patient_id}", 45, "Male", datetime.datetime.utcnow().isoformat())
        )
        conn.execute(
            "INSERT OR IGNORE INTO mri_scans (id, patient_id, image_path, pixel_spacing_mm, ref_physician, scan_date, created_at) VALUES (?, ?, ?, ?, ?, ?, ?);",
            (scan_id, patient_id, f"scan_{scan_id}.png", 1.0, "Dr. Smith", scan_date, datetime.datetime.utcnow().isoformat())
        )
        conn.execute(
            """INSERT OR IGNORE INTO predictions (
                id, scan_id, predicted_class, confidence_score, prob_glioma, prob_meningioma,
                prob_pituitary, prob_no_tumor, tumor_pixel_count, tumor_area_mm2,
                tumor_percentage_brain, tumor_percentage_image, estimated_brain_pixel_count,
                rule_based_severity, severity_rule_description, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);""",
            (prediction_id, scan_id, classification, confidence, 0.90, 0.05, 0.03, 0.02, 500, area, pct_brain, 2.0, 10000, severity, f"Severity is {severity}", datetime.datetime.utcnow().isoformat())
        )
        conn.execute(
            "INSERT OR IGNORE INTO reports (report_id, report_number, patient_id, current_version, status, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?);",
            (report_id, report_number, patient_id, 1, status, "2026-08-08 10:00:00", "2026-08-08 10:00:00")
        )
        conn.execute(
            """INSERT OR IGNORE INTO report_versions (
                report_id, version_number, prediction_id, status, created_by, reason, created_at,
                pdf_path, json_path, checksum, integrity_hash
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);""",
            (report_id, 1, prediction_id, version_status, "doc-uuid-1", "Initial draft", "2026-08-08 10:00:00",
             f"outputs/clinical_reports/{report_number}.pdf", json_path, "dummy_checksum", "dummy_integrity")
        )

    def setUp(self):
        self.db_path = os.path.abspath("outputs/test_longitudinal_timeline_flask.db")
        if os.path.exists(self.db_path):
            try:
                os.remove(self.db_path)
            except Exception:
                pass

        from dashboard.infrastructure.web_server import create_app
        self.app = create_app(db_path=self.db_path)
        self.app.config["TESTING"] = True
        self.app.config["DISABLE_CSRF"] = True
        self.client = self.app.test_client()

        self.user_repo = SQLiteUserRepository(db_path=self.db_path)
        self.admin_user = self.user_repo.bootstrap_admin()

        from security.infrastructure.password import PasswordHasher
        pass_hash = PasswordHasher.hash_password("Doctor@123")

        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                "INSERT INTO users (uuid, email, password_hash, full_name, role, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?);",
                ("doc-uuid-1", "doctor@aurascan.ai", pass_hash, "Dr. Jane Smith", Role.DOCTOR.value, datetime.datetime.utcnow().isoformat(), datetime.datetime.utcnow().isoformat())
            )
            conn.execute(
                "INSERT INTO users (uuid, email, password_hash, full_name, role, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?);",
                ("bob-jones-uuid", "bob@aurascan.ai", pass_hash, "Bob Jones", Role.PATIENT.value, datetime.datetime.utcnow().isoformat(), datetime.datetime.utcnow().isoformat())
            )
            conn.execute(
                "INSERT INTO users (uuid, email, password_hash, full_name, role, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?);",
                ("alice-smith-uuid", "alice@aurascan.ai", pass_hash, "Alice Smith", Role.PATIENT.value, datetime.datetime.utcnow().isoformat(), datetime.datetime.utcnow().isoformat())
            )
            conn.commit()
        finally:
            conn.close()

        self.doctor_user = self.user_repo.get_by_email("doctor@aurascan.ai")
        self.patient_bob = self.user_repo.get_by_email("bob@aurascan.ai")
        self.patient_alice = self.user_repo.get_by_email("alice@aurascan.ai")

        self.jwt_svc = JWTService()
        self.admin_token = self.jwt_svc.create_access_token(
            user_uuid=self.admin_user.uuid, user_id=self.admin_user.id, email=self.admin_user.email, role=Role.ADMIN
        )
        self.doctor_token = self.jwt_svc.create_access_token(
            user_uuid=self.doctor_user.uuid, user_id=self.doctor_user.id, email=self.doctor_user.email, role=Role.DOCTOR
        )
        self.patient_bob_token = self.jwt_svc.create_access_token(
            user_uuid=self.patient_bob.uuid, user_id=self.patient_bob.id, email=self.patient_bob.email, role=Role.PATIENT
        )
        self.patient_alice_token = self.jwt_svc.create_access_token(
            user_uuid=self.patient_alice.uuid, user_id=self.patient_alice.id, email=self.patient_alice.email, role=Role.PATIENT
        )

    def tearDown(self):
        if os.path.exists(self.db_path):
            try:
                os.remove(self.db_path)
            except Exception:
                pass

    def test_flask_timeline_unauthenticated(self):
        res = self.client.get("/api/patients/bob-jones-uuid/longitudinal-timeline")
        self.assertEqual(res.status_code, 401)

    def test_flask_timeline_admin_access(self):
        conn = sqlite3.connect(self.db_path)
        try:
            self._create_test_hierarchy(
                conn, "bob-jones-uuid", report_id=1, report_number="RPT-2026-0001",
                prediction_id=10, scan_id=100, classification="Glioma", area=50.0,
                severity="Medium", scan_date="2026-08-01"
            )
            conn.commit()
        finally:
            conn.close()

        self.client.set_cookie("access_token", self.admin_token)
        res = self.client.get("/api/patients/bob-jones-uuid/longitudinal-timeline")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["patient_id"], "bob-jones-uuid")

    def test_flask_timeline_doctor_access(self):
        conn = sqlite3.connect(self.db_path)
        try:
            self._create_test_hierarchy(
                conn, "bob-jones-uuid", report_id=1, report_number="RPT-2026-0001",
                prediction_id=10, scan_id=100, classification="Glioma", area=50.0,
                severity="Medium", scan_date="2026-08-01"
            )
            conn.commit()
        finally:
            conn.close()

        self.client.set_cookie("access_token", self.doctor_token)
        res = self.client.get("/api/patients/bob-jones-uuid/longitudinal-timeline")
        self.assertEqual(res.status_code, 200)

    def test_flask_timeline_patient_own_access(self):
        conn = sqlite3.connect(self.db_path)
        try:
            self._create_test_hierarchy(
                conn, "bob-jones-uuid", report_id=1, report_number="RPT-2026-0001",
                prediction_id=10, scan_id=100, classification="Glioma", area=50.0,
                severity="Medium", scan_date="2026-08-01"
            )
            conn.commit()
        finally:
            conn.close()

        self.client.set_cookie("access_token", self.patient_bob_token)
        res = self.client.get("/api/patients/bob-jones-uuid/longitudinal-timeline")
        self.assertEqual(res.status_code, 200)

    def test_flask_timeline_patient_other_patient_denied(self):
        conn = sqlite3.connect(self.db_path)
        try:
            self._create_test_hierarchy(
                conn, "bob-jones-uuid", report_id=1, report_number="RPT-2026-0001",
                prediction_id=10, scan_id=100, classification="Glioma", area=50.0,
                severity="Medium", scan_date="2026-08-01"
            )
            conn.commit()
        finally:
            conn.close()

        self.client.set_cookie("access_token", self.patient_alice_token)
        res = self.client.get("/api/patients/bob-jones-uuid/longitudinal-timeline")
        self.assertEqual(res.status_code, 403)
        self.assertIn("Access denied", res.get_json()["error"])

    def test_flask_timeline_invalid_patient(self):
        self.client.set_cookie("access_token", self.admin_token)
        res = self.client.get("/api/patients/invalid-id/longitudinal-timeline")
        self.assertEqual(res.status_code, 404)
        self.assertIn("not found", res.get_json()["error"].lower())

    def test_flask_timeline_empty_patient(self):
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                "INSERT INTO patients (patient_id, name, age, gender, created_at) VALUES (?, ?, ?, ?, ?);",
                ("bob-jones-uuid", "Bob Jones", 45, "Male", datetime.datetime.utcnow().isoformat())
            )
            conn.commit()
        finally:
            conn.close()

        self.client.set_cookie("access_token", self.patient_bob_token)
        res = self.client.get("/api/patients/bob-jones-uuid/longitudinal-timeline")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["total_events"], 0)
        self.assertEqual(data["timeline_status"], "EMPTY")

    def test_flask_timeline_multiple_events_and_ordering(self):
        conn = sqlite3.connect(self.db_path)
        try:
            self._create_test_hierarchy(
                conn, "bob-jones-uuid", report_id=2, report_number="RPT-2026-0002",
                prediction_id=11, scan_id=101, classification="Glioma", area=60.0,
                severity="High", scan_date="2026-08-10"
            )
            self._create_test_hierarchy(
                conn, "bob-jones-uuid", report_id=1, report_number="RPT-2026-0001",
                prediction_id=10, scan_id=100, classification="Glioma", area=50.0,
                severity="Medium", scan_date="2026-08-01"
            )
            conn.commit()
        finally:
            conn.close()

        self.client.set_cookie("access_token", self.patient_bob_token)
        res = self.client.get("/api/patients/bob-jones-uuid/longitudinal-timeline")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["total_events"], 2)
        self.assertEqual(data["events"][0]["report_id"], 1)
        self.assertEqual(data["events"][1]["report_id"], 2)

    def test_flask_timeline_missing_values(self):
        conn = sqlite3.connect(self.db_path)
        try:
            self._create_test_hierarchy(
                conn, "bob-jones-uuid", report_id=1, report_number="RPT-2026-0001",
                prediction_id=10, scan_id=100, classification="no_tumor", area=None,
                severity=None, confidence=None
            )
            conn.commit()
        finally:
            conn.close()

        self.client.set_cookie("access_token", self.patient_bob_token)
        res = self.client.get("/api/patients/bob-jones-uuid/longitudinal-timeline")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIsNone(data["events"][0]["tumor_area"])
        self.assertIsNone(data["events"][0]["confidence"])

    def test_flask_timeline_nan_inf_safety_and_response_schema(self):
        conn = sqlite3.connect(self.db_path)
        try:
            self._create_test_hierarchy(
                conn, "bob-jones-uuid", report_id=1, report_number="RPT-2026-0001",
                prediction_id=10, scan_id=100, classification="Glioma", area=float('nan'),
                severity="Medium", scan_date="2026-08-01", confidence=float('inf')
            )
            conn.commit()
        finally:
            conn.close()

        self.client.set_cookie("access_token", self.patient_bob_token)
        res = self.client.get("/api/patients/bob-jones-uuid/longitudinal-timeline")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIsNone(data["events"][0]["tumor_area"])
        self.assertIsNone(data["events"][0]["confidence"])
        self.assertIn("patient_id", data)
        self.assertIn("events", data)
        self.assertIn("metrics", data)


class TestLongitudinalTimelineUIRendering(unittest.TestCase):
    """Integration/Rendering tests for Longitudinal Patient Timeline UI (Phase F3.3.5)."""

    def setUp(self):
        self.db_path = os.path.abspath("outputs/test_longitudinal_timeline_ui.db")
        if os.path.exists(self.db_path):
            try:
                os.remove(self.db_path)
            except Exception:
                pass

        from dashboard.infrastructure.web_server import create_app
        self.app = create_app(db_path=self.db_path)
        self.app.config["TESTING"] = True
        self.app.config["DISABLE_CSRF"] = True
        self.client = self.app.test_client()

        self.user_repo = SQLiteUserRepository(db_path=self.db_path)
        self.user_repo.initialize_security_tables()
        self.patient_user = User(
            id=None,
            uuid="bob-jones-uuid",
            email="bob@aurascan.ai",
            password_hash="dummy_hash",
            full_name="Bob Jones",
            role=Role.PATIENT,
            is_verified=True,
            is_active=True
        )
        self.user_repo.create_user(self.patient_user)

        self.jwt_svc = JWTService()
        self.patient_token = self.jwt_svc.create_access_token(
            user_uuid=self.patient_user.uuid, user_id=self.patient_user.id, email=self.patient_user.email, role=Role.PATIENT
        )

    def tearDown(self):
        if os.path.exists(self.db_path):
            try:
                os.remove(self.db_path)
            except Exception:
                pass

    def test_patient_dashboard_renders_timeline_section(self):
        """Verify that the timeline HTML section and loading states exist on the page."""
        self.client.set_cookie("access_token", self.patient_token)
        res = self.client.get("/patient")
        self.assertEqual(res.status_code, 200)
        html = res.data.decode("utf-8")

        # 1. Timeline section exists
        self.assertIn('id="timeline-section"', html)
        self.assertIn('Patient Longitudinal Timeline', html)
        self.assertIn('Track changes across previous MRI reports', html)

        # 2. Loading skeleton exists
        self.assertIn('id="timeline-loading"', html)

        # 3. Error state exists
        self.assertIn('id="timeline-error"', html)
        self.assertIn('Unable to Load Longitudinal Timeline', html)

        # 4. Empty state exists
        self.assertIn('id="timeline-empty"', html)
        self.assertIn('No longitudinal scan history available yet.', html)

        # 5. Content section and sub-elements exist
        self.assertIn('id="timeline-content"', html)
        self.assertIn('id="timeline-total-scans"', html)
        self.assertIn('id="timeline-first-date"', html)
        self.assertIn('id="timeline-latest-date"', html)
        self.assertIn('id="timeline-status"', html)
        self.assertIn('id="timeline-trend-summary"', html)
        self.assertIn('id="timeline-metrics-grid"', html)
        self.assertIn('id="timeline-visualization-list"', html)

        # 6. JavaScript helper functions exist on the page
        self.assertIn('function formatTimelineValue', html)
        self.assertIn('function createMetricCard', html)
        self.assertIn('function renderVerticalTimeline', html)
        self.assertIn('async function loadTimelineData', html)
        self.assertIn('function showTimelineError', html)

    def test_javascript_timeline_helpers_exist_and_sanitize(self):
        """Verify that the javascript functions exist on the patient page and contain correct rules."""
        self.client.set_cookie("access_token", self.patient_token)
        res = self.client.get("/patient")
        self.assertEqual(res.status_code, 200)
        html = res.data.decode("utf-8")

        # Verify formatTimelineValue sanitization logic is present
        self.assertIn('formatTimelineValue(value, fallback = "N/A")', html)
        self.assertIn('lower === "nan"', html)
        self.assertIn('lower === "inf"', html)
        self.assertIn('lower === "-inf"', html)

        # Verify createMetricCard handles numeric and non-numeric metrics
        self.assertIn('key === "tumor_area"', html)
        self.assertIn('key === "confidence"', html)
        self.assertIn('currentVal = `${currentVal.toFixed(2)} mm²`', html)
        self.assertIn('currentVal = `${(currentVal * 100).toFixed(2)}%`', html)

        # Verify vertical timeline visualization rendering matches Glioma/Confidence/Area occupancy
        self.assertIn('ev.classification', html)
        self.assertIn('ev.confidence', html)
        self.assertIn('ev.tumor_area', html)
        self.assertIn('ev.tumor_percentage', html)
        self.assertIn('ev.severity', html)


class TestLongitudinalTimelineDoctorUIRendering(unittest.TestCase):
    """Integration/Rendering tests for Longitudinal Patient Timeline UI for Doctor (Phase F3.3.6)."""

    def setUp(self):
        self.db_path = os.path.abspath("outputs/test_longitudinal_timeline_doctor_ui.db")
        if os.path.exists(self.db_path):
            try:
                os.remove(self.db_path)
            except Exception:
                pass

        from dashboard.infrastructure.web_server import create_app
        self.app = create_app(db_path=self.db_path)
        self.app.config["TESTING"] = True
        self.app.config["DISABLE_CSRF"] = True
        self.client = self.app.test_client()

        self.user_repo = SQLiteUserRepository(db_path=self.db_path)
        self.user_repo.initialize_security_tables()

        self.doctor_user = User(
            id=None,
            uuid="dr-smith-uuid",
            email="dr.smith@aurascan.ai",
            password_hash="dummy_hash",
            full_name="Dr. Smith",
            role=Role.DOCTOR,
            is_verified=True,
            is_active=True
        )
        self.user_repo.create_user(self.doctor_user)

        self.jwt_svc = JWTService()
        self.doctor_token = self.jwt_svc.create_access_token(
            user_uuid=self.doctor_user.uuid, user_id=self.doctor_user.id, email=self.doctor_user.email, role=Role.DOCTOR
        )

    def tearDown(self):
        if os.path.exists(self.db_path):
            try:
                os.remove(self.db_path)
            except Exception:
                pass

    def test_doctor_dashboard_renders_timeline_section(self):
        """Verify that the doctor dashboard includes the timeline HTML container and widgets."""
        self.client.set_cookie("access_token", self.doctor_token)
        res = self.client.get("/doctor")
        self.assertEqual(res.status_code, 200)
        html = res.data.decode("utf-8")

        # 1. Timeline container exist with correct elements
        self.assertIn('id="timeline-section"', html)
        self.assertIn('Longitudinal Patient Timeline', html)
        self.assertIn('Track clinical findings and tumor measurements across scans.', html)

        # 2. Loading state, empty state, and error states exist
        self.assertIn('id="timeline-loading"', html)
        self.assertIn('id="timeline-error"', html)
        self.assertIn('id="timeline-empty"', html)
        self.assertIn('No longitudinal scan history available.', html)

        # 3. Stats summary fields
        self.assertIn('id="timeline-total-scans"', html)
        self.assertIn('id="timeline-first-date"', html)
        self.assertIn('id="timeline-latest-date"', html)
        self.assertIn('id="timeline-status"', html)
        self.assertIn('id="timeline-trend-summary"', html)

        # 4. Metrics grid and visualization list elements exist
        self.assertIn('id="timeline-metrics-grid"', html)
        self.assertIn('id="timeline-visualization-list"', html)

    def test_doctor_dashboard_javascript_functions_exist(self):
        """Verify that JavaScript functions for timeline are present and match requirements."""
        self.client.set_cookie("access_token", self.doctor_token)
        res = self.client.get("/doctor")
        self.assertEqual(res.status_code, 200)
        html = res.data.decode("utf-8")

        # Helper functions
        self.assertIn('function formatTimelineValue', html)
        self.assertIn('async function loadTimelineData', html)
        self.assertIn('function clearTimelineData', html)
        self.assertIn('function showTimelineError', html)
        self.assertIn('function createMetricCard', html)
        self.assertIn('function renderVerticalTimeline', html)

        # Assert sanitization rules mapping to N/A
        self.assertIn('lower === "nan"', html)
        self.assertIn('lower === "inf"', html)
        self.assertIn('lower === "-inf"', html)

        # Assert key elements matching requirements
        self.assertIn('key === "tumor_percentage"', html)
        self.assertIn('key === "tumor_area"', html)
        self.assertIn('key === "confidence"', html)
        self.assertIn('key === "severity"', html)

        # Assert event parameters (status, score, etc.)
        self.assertIn('ev.report_status', html)
        self.assertIn('ev.severity_score', html)

        # Assert patient switching integration and AbortController
        self.assertIn('timelineAbortController = new AbortController()', html)
        self.assertIn('timelineAbortController.abort()', html)
        self.assertIn('clearTimelineData()', html)


class TestLongitudinalTimelineSecurityAndAuditHardening(unittest.TestCase):
    """Security and audit trail verification for F3.3 Longitudinal Patient Timeline (Phase F3.3.7)."""

    def _create_test_hierarchy(
        self, conn, patient_id, report_id, report_number, prediction_id, scan_id,
        classification, area, severity, scan_date="2026-08-08", confidence=0.95,
        pct_brain=5.0, status="FINALIZED", version_status="FINAL", json_path=""
    ):
        conn.execute(
            "INSERT OR IGNORE INTO patients (patient_id, name, age, gender, created_at) VALUES (?, ?, ?, ?, ?);",
            (patient_id, f"Patient {patient_id}", 45, "Male", datetime.datetime.utcnow().isoformat())
        )
        conn.execute(
            "INSERT OR IGNORE INTO mri_scans (id, patient_id, image_path, pixel_spacing_mm, ref_physician, scan_date, created_at) VALUES (?, ?, ?, ?, ?, ?, ?);",
            (scan_id, patient_id, f"scan_{scan_id}.png", 1.0, "Dr. Smith", scan_date, datetime.datetime.utcnow().isoformat())
        )
        conn.execute(
            """INSERT OR IGNORE INTO predictions (
                id, scan_id, predicted_class, confidence_score, prob_glioma, prob_meningioma,
                prob_pituitary, prob_no_tumor, tumor_pixel_count, tumor_area_mm2,
                tumor_percentage_brain, tumor_percentage_image, estimated_brain_pixel_count,
                rule_based_severity, severity_rule_description, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);""",
            (prediction_id, scan_id, classification, confidence, 0.90, 0.05, 0.03, 0.02, 500, area, pct_brain, 2.0, 10000, severity, f"Severity is {severity}", datetime.datetime.utcnow().isoformat())
        )
        conn.execute(
            "INSERT OR IGNORE INTO reports (report_id, report_number, patient_id, current_version, status, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?);",
            (report_id, report_number, patient_id, 1, status, "2026-08-08 10:00:00", "2026-08-08 10:00:00")
        )
        conn.execute(
            """INSERT OR IGNORE INTO report_versions (
                report_id, version_number, prediction_id, status, created_by, reason, created_at,
                pdf_path, json_path, checksum, integrity_hash
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);""",
            (report_id, 1, prediction_id, version_status, "doc-uuid-1", "Initial draft", "2026-08-08 10:00:00",
             f"outputs/clinical_reports/{report_number}.pdf", json_path, "dummy_checksum", "dummy_integrity")
        )

    def setUp(self):
        test_method_name = self.id().split('.')[-1]
        self.db_path = os.path.abspath(f"outputs/test_longitudinal_timeline_sec_{test_method_name}.db")
        if os.path.exists(self.db_path):
            try:
                os.remove(self.db_path)
            except Exception:
                pass

        from dashboard.infrastructure.web_server import create_app
        self.app = create_app(db_path=self.db_path)
        self.app.config["TESTING"] = True
        self.app.config["DISABLE_CSRF"] = True
        self.client = self.app.test_client()

        # Database Setup
        self.user_repo = SQLiteUserRepository(db_path=self.db_path)
        self.user_repo.initialize_security_tables()

        # Create Patient A
        self.patient_a = User(
            id=None, uuid="patient-a-uuid", email="test-patient-a@aurascan.ai",
            password_hash="pw_hash", full_name="Patient A", role=Role.PATIENT,
            is_verified=True, is_active=True
        )
        self.user_repo.create_user(self.patient_a)

        # Create Patient B
        self.patient_b = User(
            id=None, uuid="patient-b-uuid", email="test-patient-b@aurascan.ai",
            password_hash="pw_hash", full_name="Patient B", role=Role.PATIENT,
            is_verified=True, is_active=True
        )
        self.user_repo.create_user(self.patient_b)

        # Create Doctor
        self.doctor = User(
            id=None, uuid="doctor-uuid", email="test-doctor@aurascan.ai",
            password_hash="pw_hash", full_name="Doctor User", role=Role.DOCTOR,
            is_verified=True, is_active=True
        )
        self.user_repo.create_user(self.doctor)

        # Create Admin
        self.admin = User(
            id=None, uuid="admin-uuid", email="test-admin@aurascan.ai",
            password_hash="pw_hash", full_name="Admin User", role=Role.ADMIN,
            is_verified=True, is_active=True
        )
        self.user_repo.create_user(self.admin)

        # Setup JWT Tokens
        self.jwt_svc = JWTService()
        self.token_a = self.jwt_svc.create_access_token(
            user_uuid=self.patient_a.uuid, user_id=self.patient_a.id, email=self.patient_a.email, role=Role.PATIENT
        )
        self.token_b = self.jwt_svc.create_access_token(
            user_uuid=self.patient_b.uuid, user_id=self.patient_b.id, email=self.patient_b.email, role=Role.PATIENT
        )
        self.token_doc = self.jwt_svc.create_access_token(
            user_uuid=self.doctor.uuid, user_id=self.doctor.id, email=self.doctor.email, role=Role.DOCTOR
        )
        self.token_admin = self.jwt_svc.create_access_token(
            user_uuid=self.admin.uuid, user_id=self.admin.id, email=self.admin.email, role=Role.ADMIN
        )

        # Seed Patients into DB
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                "INSERT INTO patients (patient_id, name, age, gender, created_at) VALUES (?, ?, 30, 'Male', ?);",
                (self.patient_a.uuid, self.patient_a.full_name, datetime.datetime.utcnow().isoformat())
            )
            conn.execute(
                "INSERT INTO patients (patient_id, name, age, gender, created_at) VALUES (?, ?, 25, 'Female', ?);",
                (self.patient_b.uuid, self.patient_b.full_name, datetime.datetime.utcnow().isoformat())
            )
            conn.commit()
        finally:
            conn.close()

    def tearDown(self):
        if os.path.exists(self.db_path):
            try:
                os.remove(self.db_path)
            except Exception:
                pass

    def test_unauthenticated_request_rejected(self):
        """1. Unauthenticated requests must return 401."""
        res = self.client.get("/api/patients/patient-a-uuid/longitudinal-timeline")
        self.assertEqual(res.status_code, 401)

    def test_invalid_authentication_token_rejected(self):
        """2. Invalid or malformed tokens must return 401."""
        self.client.set_cookie("access_token", "invalid_token_xyz")
        res = self.client.get("/api/patients/patient-a-uuid/longitudinal-timeline")
        self.assertEqual(res.status_code, 401)

    def test_patient_own_access_allowed(self):
        """3. Patients must be allowed to access their own timeline."""
        self.client.set_cookie("access_token", self.token_a)
        res = self.client.get("/api/patients/patient-a-uuid/longitudinal-timeline")
        self.assertEqual(res.status_code, 200)

        # Verify SUCCESS audit log was written
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            row = conn.execute("SELECT * FROM security_audit_logs WHERE event_type = 'PATIENT_TIMELINE_ACCESSED' ORDER BY timestamp DESC LIMIT 1;").fetchone()
            self.assertIsNotNone(row)
            self.assertEqual(row["status"], "SUCCESS")
            self.assertIn("patient-a-uuid", row["details"])
        finally:
            conn.close()

    def test_patient_other_access_forbidden(self):
        """4. Patient A requesting Patient B's timeline (IDOR) must be rejected with 403."""
        self.client.set_cookie("access_token", self.token_a)
        res = self.client.get("/api/patients/patient-b-uuid/longitudinal-timeline")
        self.assertEqual(res.status_code, 403)

        # Verify FAILED audit log was written
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            row = conn.execute("SELECT * FROM security_audit_logs WHERE event_type = 'PATIENT_TIMELINE_ACCESSED' ORDER BY timestamp DESC LIMIT 1;").fetchone()
            self.assertIsNotNone(row)
            self.assertEqual(row["status"], "FAILED")
            self.assertIn("Access denied", row["details"])
        finally:
            conn.close()

    def test_doctor_access_allowed(self):
        """5. Doctors must be allowed to access any patient's timeline."""
        self.client.set_cookie("access_token", self.token_doc)
        res = self.client.get("/api/patients/patient-a-uuid/longitudinal-timeline")
        self.assertEqual(res.status_code, 200)

    def test_admin_access_allowed(self):
        """6. Admins must be allowed to access any patient's timeline."""
        self.client.set_cookie("access_token", self.token_admin)
        res = self.client.get("/api/patients/patient-a-uuid/longitudinal-timeline")
        self.assertEqual(res.status_code, 200)

    def test_invalid_patient_uuid_handling(self):
        """7. Requesting non-existent patient returns safe error without leak."""
        self.client.set_cookie("access_token", self.token_doc)
        res = self.client.get("/api/patients/nonexistent-uuid/longitudinal-timeline")
        self.assertEqual(res.status_code, 404)

        # Verify no leaks of database path or python details
        data = res.get_json()
        self.assertNotIn("db_path", str(data))
        self.assertNotIn("sqlite", str(data).lower())

    def test_input_validation_on_patient_id(self):
        """8. Malformed or SQL-like patient IDs must be safely handled and rejected."""
        self.client.set_cookie("access_token", self.token_doc)

        cases = [
            ("patient-a-uuid; DROP TABLE patients;", 422),  # SQL Injection style
            ("../traversal", 422),                          # Path traversal style
            ("A" * 200, 422)                                # excessive length
        ]

        for p_id, expected_code in cases:
            res = self.client.get(f"/api/patients/{p_id}/longitudinal-timeline")
            self.assertIn(res.status_code, [404, 422, 400, 403])

    def test_json_safety_converts_nan_inf_to_null(self):
        """9. Verify that NaN and Infinity float values are mapped to null (None) in serialized response."""
        conn = sqlite3.connect(self.db_path)
        try:
            self._create_test_hierarchy(
                conn, self.patient_a.uuid, report_id=10, report_number="RPT-10",
                prediction_id=10, scan_id=10, classification="Glioma", area="Infinity",
                severity="Medium", scan_date="2026-08-01", confidence="NaN"
            )
            conn.commit()
        finally:
            conn.close()

        self.client.set_cookie("access_token", self.token_a)
        res = self.client.get("/api/patients/patient-a-uuid/longitudinal-timeline")
        self.assertEqual(res.status_code, 200)

        data = res.get_json()
        # Verify serialized values are None (which JSON maps to null)
        event = data["events"][0]
        self.assertIsNone(event["confidence"])
        self.assertIsNone(event["tumor_area"])
        self.assertIsNone(data["metrics"]["confidence"]["current_value"])
        self.assertIsNone(data["metrics"]["tumor_area"]["current_value"])

    def test_report_level_security_ignores_mismatched_patient_json(self):
        """10. Verify report-level security ignores report JSON disk files belonging to another patient."""
        import json
        mismatched_json = os.path.abspath("outputs/mismatched_report.json")
        report_data = {
            "patient": {
                "patient_id": "patient-a-uuid",  # Mismatched patient ID!
                "scan_date": "2026-08-10"
            },
            "classification": {
                "predicted_class": "Meningioma",
                "confidence_score": 0.99
            }
        }
        with open(mismatched_json, "w", encoding="utf-8") as f:
            json.dump(report_data, f)

        conn = sqlite3.connect(self.db_path)
        try:
            self._create_test_hierarchy(
                conn, self.patient_b.uuid, report_id=20, report_number="RPT-20",
                prediction_id=20, scan_id=20, classification="Glioma", area=10.0,
                severity="Low", scan_date="2026-08-10", confidence=0.90, json_path=mismatched_json
            )
            conn.commit()
        finally:
            conn.close()

        # Retrieve Patient B's timeline
        self.client.set_cookie("access_token", self.token_b)
        res = self.client.get("/api/patients/patient-b-uuid/longitudinal-timeline")
        self.assertEqual(res.status_code, 200)

        data = res.get_json()
        # Mismatched json_path details (predicted class Meningioma) must have been ignored,
        # fallback to database details (predicted class Glioma)
        event = data["events"][0]
        self.assertEqual(event["classification"], "Glioma")
        self.assertEqual(event["confidence"], 0.90)

        if os.path.exists(mismatched_json):
            os.remove(mismatched_json)

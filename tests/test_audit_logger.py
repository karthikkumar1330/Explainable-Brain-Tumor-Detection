import os
import sqlite3
import unittest
from monitoring.infrastructure.audit_logger import AuditLogger
from persistence.infrastructure.repository import SQLitePersistenceRepository


class TestAuditLogger(unittest.TestCase):

    def setUp(self):
        self.db_path = "test_audit.db"
        self.db_repo = SQLitePersistenceRepository(db_path=self.db_path)
        self.db_repo.initialize_db()
        self.logger = AuditLogger(db_path=self.db_path)

    def tearDown(self):
        if os.path.exists(self.db_path):
            try:
                os.remove(self.db_path)
            except Exception:
                pass

    def test_log_execution_creates_record(self):
        """Verifies database insertion logic records audit data correctly."""
        self.logger.log_execution(
            patient_id="PATIENT_AUDIT_99",
            user="Dr. Sarah Test",
            model_version_cls="efficientnet_b0_v2.pt",
            model_version_seg="unext_v2.pth",
            runtime_sec=1.45,
            gpu_active=True,
            cpu_threads=8,
            warnings=["Anomaly testing warning"],
            errors=[],
            report_status="Generated",
            database_status="Persisted"
        )
        
        # Verify in DB
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM ai_audit_logs WHERE patient_id = ?", ("PATIENT_AUDIT_99",)).fetchone()
        conn.close()
        
        self.assertIsNotNone(row)
        self.assertEqual(row["user_id"], "Dr. Sarah Test")
        self.assertEqual(row["model_version_cls"], "efficientnet_b0_v2.pt")
        self.assertEqual(row["runtime_sec"], 1.45)
        self.assertEqual(row["gpu_active"], 1)
        self.assertEqual(row["cpu_threads"], 8)
        self.assertIn("Anomaly testing warning", row["warnings_json"])

    def test_get_telemetry_queries_database(self):
        """Verifies query stats aggregator extracts count, averages, and methods."""
        # Insert test audit logs
        self.logger.log_execution(
            patient_id="P_1",
            user="Doc",
            model_version_cls="cls.pt",
            model_version_seg="seg.pth",
            runtime_sec=2.00,
            gpu_active=False,
            cpu_threads=4,
            warnings=[],
            errors=[],
            report_status="Generated",
            database_status="Persisted"
        )
        
        telemetry = self.db_repo.get_health_telemetry()
        self.assertIn("total_predictions", telemetry)
        self.assertEqual(telemetry["avg_runtime"], 2.00)
        self.assertEqual(telemetry["db_healthy"], True)


if __name__ == "__main__":
    unittest.main()

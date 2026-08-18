import os
import sqlite3
from typing import Optional, Any
from security.domain.entities import Role, User

class AuthorizationService:
    """Centralized authorization service to enforce role-based and resource-level access controls.
    
    Prevents IDOR (Insecure Direct Object Reference) and unauthorized cross-patient healthcare record access.
    """

    def __init__(self, db_path: Optional[str] = None) -> None:
        self.db_path = db_path or os.environ.get("DB_PATH", "outputs/clinical_reports.db")

    def _get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn

    def can_access_patient(self, user: Optional[Any], patient_id: str, conn: Optional[sqlite3.Connection] = None) -> bool:
        """Verifies if the authenticated user is authorized to access the patient's records."""
        if not user:
            return False

        role_val = user.role.value if hasattr(user.role, 'value') else str(user.role).lower()

        # Admin policy: admins can access all records
        if role_val == "admin":
            return True

        # Patient policy: patients can only access their own records
        if role_val == "patient":
            # Compare patient_id to user.uuid (primary identity key)
            if str(patient_id).lower() == str(user.uuid).lower():
                return True
            
            return False

        # Doctor policy: doctors can only access assigned patients
        if role_val == "doctor":
            doctor_id = getattr(user, 'id', None)
            if doctor_id is None:
                return False

            local_conn = conn or self._get_connection()
            try:
                # Check if patient exists in the patients table
                patient_exists = local_conn.execute(
                    "SELECT 1 FROM patients WHERE LOWER(patient_id) = LOWER(?);",
                    (patient_id,)
                ).fetchone()
                if not patient_exists:
                    # If patient demographics do not exist yet, we only allow access
                    # if the patient_id corresponds to a registered patient user in the users table.
                    user_exists = local_conn.execute(
                        "SELECT 1 FROM users WHERE LOWER(uuid) = LOWER(?) AND role = 'patient';",
                        (patient_id,)
                    ).fetchone()
                    if user_exists:
                        return True
                    return False

                # Check explicit assignment (case-insensitive query comparison)
                assignment = local_conn.execute(
                    "SELECT 1 FROM doctor_patient_assignments WHERE doctor_id = ? AND LOWER(patient_id) = LOWER(?);",
                    (doctor_id, patient_id)
                ).fetchone()
                if assignment:
                    return True

            except Exception:
                pass
            finally:
                if conn is None:
                    local_conn.close()

            return False

        return False

    def get_authorized_patient_ids(self, user: Optional[Any], conn: Optional[sqlite3.Connection] = None) -> Optional[set]:
        """Returns a set of lowercase patient IDs authorized for the user, or None if authorized for all (Admin)."""
        if not user:
            return set()

        role_val = user.role.value if hasattr(user.role, 'value') else str(user.role).lower()

        if role_val == "admin":
            return None # Admins can access everything

        if role_val == "patient":
            return {str(user.uuid).lower()}

        if role_val == "doctor":
            doctor_id = getattr(user, 'id', None)
            if doctor_id is None:
                return set()

            local_conn = conn or self._get_connection()
            try:
                rows = local_conn.execute(
                    "SELECT patient_id FROM doctor_patient_assignments WHERE doctor_id = ?;",
                    (doctor_id,)
                ).fetchall()
                return {str(row["patient_id"]).lower() for row in rows}
            except Exception:
                return set()
            finally:
                if conn is None:
                    local_conn.close()

        return set()

    def can_access_scan(self, user: Optional[Any], scan_id: int, conn: Optional[sqlite3.Connection] = None) -> bool:
        """Verifies if the authenticated user is authorized to access the given MRI scan."""
        local_conn = conn or self._get_connection()
        try:
            row = local_conn.execute("SELECT patient_id FROM mri_scans WHERE id = ?;", (scan_id,)).fetchone()
            if not row:
                return False
            patient_id = row["patient_id"]
        except Exception:
            return False
        finally:
            if conn is None:
                local_conn.close()

        return self.can_access_patient(user, patient_id, conn=conn)

    def can_access_prediction(self, user: Optional[Any], prediction_id: int, conn: Optional[sqlite3.Connection] = None) -> bool:
        """Verifies authorization for the given prediction by validating the ownership chain."""
        local_conn = conn or self._get_connection()
        try:
            row = local_conn.execute("""
                SELECT s.patient_id 
                FROM predictions p
                JOIN mri_scans s ON p.scan_id = s.id
                WHERE p.id = ?;
            """, (prediction_id,)).fetchone()
            if not row:
                return False
            patient_id = row["patient_id"]
        except Exception:
            return False
        finally:
            if conn is None:
                local_conn.close()

        return self.can_access_patient(user, patient_id, conn=conn)

    def can_access_report(self, user: Optional[Any], report_id: int, conn: Optional[sqlite3.Connection] = None) -> bool:
        """Verifies authorization for the given clinical report ID (mapping reports or clinical_reports)."""
        local_conn = conn or self._get_connection()
        try:
            # Check clinical_reports first (H0-H2 report table)
            row = local_conn.execute("""
                SELECT s.patient_id 
                FROM clinical_reports cr
                JOIN predictions p ON cr.prediction_id = p.id
                JOIN mri_scans s ON p.scan_id = s.id
                WHERE cr.id = ?;
            """, (report_id,)).fetchone()
            if row:
                patient_id = row["patient_id"]
            else:
                # Fallback check reports table (version control table)
                row2 = local_conn.execute("SELECT patient_id FROM reports WHERE report_id = ?;", (report_id,)).fetchone()
                if not row2:
                    return False
                patient_id = row2["patient_id"]
        except Exception:
            return False
        finally:
            if conn is None:
                local_conn.close()

        return self.can_access_patient(user, patient_id, conn=conn)

    def validate_scan_patient_match(self, scan_id: int, patient_id: str) -> bool:
        """Enforces cross-entity consistency. Verifies that the scan belongs to the specified patient."""
        conn = self._get_connection()
        try:
            row = conn.execute("SELECT patient_id FROM mri_scans WHERE id = ?;", (scan_id,)).fetchone()
            if not row:
                return False
            return str(row["patient_id"]).lower() == str(patient_id).lower()
        except Exception:
            return False
        finally:
            conn.close()

    def validate_report_scan_match(self, report_id: int, scan_id: int) -> bool:
        """Enforces cross-entity consistency. Verifies that the report is linked to the specified scan."""
        conn = self._get_connection()
        try:
            row = conn.execute("""
                SELECT p.scan_id 
                FROM clinical_reports cr
                JOIN predictions p ON cr.prediction_id = p.id
                WHERE cr.id = ?;
            """, (report_id,)).fetchone()
            if row:
                return int(row["scan_id"]) == int(scan_id)
            
            # Fallback to reports table
            row2 = conn.execute("""
                SELECT p.scan_id 
                FROM reports r
                JOIN report_versions rv ON r.report_id = rv.report_id
                JOIN predictions p ON rv.prediction_id = p.id
                WHERE r.report_id = ?;
            """, (report_id,)).fetchone()
            if row2:
                return int(row2["scan_id"]) == int(scan_id)
        except Exception:
            return False
        finally:
            conn.close()
        return False

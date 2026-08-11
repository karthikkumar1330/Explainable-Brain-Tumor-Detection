import os
import sqlite3
import datetime
import logging
import math
from typing import Optional, List, Dict, Any

from security.application.authorization_service import AuthorizationService
from security.infrastructure.encryption_service import PIIEncryptionService
from security.domain.entities import Role, User, SecurityAuditLog
from security.infrastructure.repository import SQLiteUserRepository

class MriAnnotationServiceException(Exception):
    """Exception raised for errors in the MriAnnotationService."""
    pass

class MriAnnotationService:
    """Service to manage clinician point annotations on MRI scans."""

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        self.logger = logging.getLogger("mri_annotation_service")
        self.auth_svc = AuthorizationService(db_path=self.db_path)
        self.user_repo = SQLiteUserRepository(db_path=self.db_path)
        try:
            self.encryption_service = PIIEncryptionService()
        except Exception:
            self.encryption_service = None

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _log_audit_event(
        self,
        event_type: str,
        actor: User,
        status: str,
        details: str
    ) -> None:
        try:
            audit_log = SecurityAuditLog(
                id=None,
                timestamp=datetime.datetime.utcnow().isoformat(),
                event_type=event_type,
                user_id=actor.id if actor else None,
                email=actor.email if actor else None,
                ip_address="127.0.0.1",
                status=status,
                details=details,
                user_agent="System"
            )
            self.user_repo.log_security_event(audit_log)
        except Exception as e:
            self.logger.warning(f"Failed to write annotation security audit log: {e}")

    def create_annotation(
        self,
        actor: User,
        scan_id: int,
        x: float,
        y: float,
        label: Optional[str] = None,
        comment: Optional[str] = None
    ) -> Dict[str, Any]:
        """Creates a clinician point annotation on an MRI scan."""
        # 1. Check Authentication and Role
        if not actor:
            raise MriAnnotationServiceException("Authentication required.")
        
        role_val = actor.role.value if hasattr(actor.role, 'value') else str(actor.role).lower()
        if role_val != "doctor" and role_val != "admin":
            raise MriAnnotationServiceException("Unauthorized. Doctor role required.")

        # 2. Input Validation for scan_id
        if scan_id is None or not isinstance(scan_id, int) or scan_id <= 0:
            raise MriAnnotationServiceException("Invalid scan ID.")

        # 3. Input Validation for Coordinates
        try:
            if x is None or y is None:
                raise ValueError()
            if isinstance(x, bool) or isinstance(y, bool):
                raise ValueError()
            x_val = float(x)
            y_val = float(y)
        except (ValueError, TypeError):
            raise MriAnnotationServiceException("Coordinates must be numeric.")
        
        if math.isnan(x_val) or math.isinf(x_val) or math.isnan(y_val) or math.isinf(y_val):
            raise MriAnnotationServiceException("Coordinates cannot be NaN or Infinity.")
            
        if x_val < 0.0 or x_val > 1.0 or y_val < 0.0 or y_val > 1.0:
            raise MriAnnotationServiceException("Coordinates must be normalized between 0.0 and 1.0.")

        # 4. Input Validation for Label
        trimmed_label = ""
        if label is not None:
            if not isinstance(label, str):
                raise MriAnnotationServiceException("Label must be a string.")
            trimmed_label = label.strip()
            if len(trimmed_label) > 100:
                raise MriAnnotationServiceException("Label exceeds maximum length of 100 characters.")

        # 5. Input Validation for Comment
        trimmed_comment = ""
        if comment is not None:
            if not isinstance(comment, str):
                raise MriAnnotationServiceException("Comment must be a string.")
            trimmed_comment = comment.strip()
            if len(trimmed_comment) > 2000:
                raise MriAnnotationServiceException("Comment exceeds maximum length of 2000 characters.")

        # 6. Fetch Scan and Patient ID
        conn = self._get_connection()
        try:
            scan_row = conn.execute("SELECT id, patient_id FROM mri_scans WHERE id = ?;", (scan_id,)).fetchone()
            if not scan_row:
                raise MriAnnotationServiceException("MRI Scan not found.")
            patient_id = scan_row["patient_id"]
        finally:
            conn.close()

        # 7. Enforce Security Access Checks
        if not self.auth_svc.can_access_patient(actor, patient_id):
            self._log_audit_event("POINT_ANNOTATION_CREATED", actor, "FAILED", f"Access denied creating annotation for patient: {patient_id}")
            raise MriAnnotationServiceException("Access denied.")

        if not self.auth_svc.can_access_scan(actor, scan_id):
            self._log_audit_event("POINT_ANNOTATION_CREATED", actor, "FAILED", f"Access denied creating annotation for scan: {scan_id}")
            raise MriAnnotationServiceException("Access denied.")

        if not self.auth_svc.validate_scan_patient_match(scan_id, patient_id):
            raise MriAnnotationServiceException("Mismatched patient and scan.")

        # 8. Encryption
        encrypted_comment = None
        if trimmed_comment:
            if not self.encryption_service:
                raise MriAnnotationServiceException("PII Encryption Service is not initialized.")
            encrypted_comment = self.encryption_service.encrypt(trimmed_comment)

        # 9. Insert
        now = datetime.datetime.utcnow().isoformat()
        conn = self._get_connection()
        try:
            with conn:
                cursor = conn.cursor()
                cursor.execute(
                    """INSERT INTO mri_point_annotations (scan_id, patient_id, doctor_id, x_normalized, y_normalized, label, encrypted_comment, created_at, updated_at, status)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'active');""",
                    (scan_id, patient_id, actor.id, x_val, y_val, trimmed_label, encrypted_comment, now, now)
                )
                annotation_id = cursor.lastrowid
        finally:
            conn.close()

        self._log_audit_event(
            "POINT_ANNOTATION_CREATED",
            actor,
            "SUCCESS",
            f"Created point annotation ID {annotation_id} for scan {scan_id} at ({x_val}, {y_val})."
        )

        return {
            "annotation_id": annotation_id,
            "scan_id": scan_id,
            "patient_id": patient_id,
            "doctor_id": actor.id,
            "x": x_val,
            "y": y_val,
            "label": trimmed_label,
            "comment": trimmed_comment,
            "created_at": now,
            "updated_at": now,
            "status": "active"
        }

    def get_annotations_for_scan(
        self,
        actor: User,
        scan_id: int
    ) -> List[Dict[str, Any]]:
        """Retrieves active point annotations for a given MRI scan."""
        if not actor:
            raise MriAnnotationServiceException("Authentication required.")

        if scan_id is None or not isinstance(scan_id, int) or scan_id <= 0:
            raise MriAnnotationServiceException("Invalid scan ID.")

        conn = self._get_connection()
        try:
            scan_row = conn.execute("SELECT patient_id FROM mri_scans WHERE id = ?;", (scan_id,)).fetchone()
            if not scan_row:
                raise MriAnnotationServiceException("MRI Scan not found.")
            patient_id = scan_row["patient_id"]
        finally:
            conn.close()

        # Enforce security checks
        if not self.auth_svc.can_access_patient(actor, patient_id):
            raise MriAnnotationServiceException("Access denied.")

        if not self.auth_svc.can_access_scan(actor, scan_id):
            raise MriAnnotationServiceException("Access denied.")

        conn = self._get_connection()
        try:
            rows = conn.execute(
                """SELECT a.*, u.full_name as doctor_name, u.email as doctor_email
                   FROM mri_point_annotations a
                   JOIN users u ON a.doctor_id = u.id
                   WHERE a.scan_id = ? AND a.status = 'active'
                   ORDER BY a.created_at ASC;""",
                (scan_id,)
            ).fetchall()
        finally:
            conn.close()

        results = []
        for r in rows:
            decrypted_comment = ""
            if r["encrypted_comment"]:
                if not self.encryption_service:
                    raise MriAnnotationServiceException("PII Encryption Service is not initialized.")
                decrypted_comment = self.encryption_service.decrypt(r["encrypted_comment"])
            
            results.append({
                "annotation_id": r["annotation_id"],
                "scan_id": r["scan_id"],
                "patient_id": r["patient_id"],
                "doctor_id": r["doctor_id"],
                "doctor_name": r["doctor_name"],
                "doctor_email": r["doctor_email"],
                "x": r["x_normalized"],
                "y": r["y_normalized"],
                "label": r["label"],
                "comment": decrypted_comment,
                "created_at": r["created_at"],
                "updated_at": r["updated_at"],
                "status": r["status"]
            })
        return results

    def update_annotation(
        self,
        actor: User,
        annotation_id: int,
        x: float,
        y: float,
        label: Optional[str] = None,
        comment: Optional[str] = None
    ) -> Dict[str, Any]:
        """Updates an existing clinician point annotation."""
        if not actor:
            raise MriAnnotationServiceException("Authentication required.")

        # Input Validation for Coordinates
        try:
            if x is None or y is None:
                raise ValueError()
            if isinstance(x, bool) or isinstance(y, bool):
                raise ValueError()
            x_val = float(x)
            y_val = float(y)
        except (ValueError, TypeError):
            raise MriAnnotationServiceException("Coordinates must be numeric.")
        
        if math.isnan(x_val) or math.isinf(x_val) or math.isnan(y_val) or math.isinf(y_val):
            raise MriAnnotationServiceException("Coordinates cannot be NaN or Infinity.")
            
        if x_val < 0.0 or x_val > 1.0 or y_val < 0.0 or y_val > 1.0:
            raise MriAnnotationServiceException("Coordinates must be normalized between 0.0 and 1.0.")

        # Input Validation for Label
        trimmed_label = ""
        if label is not None:
            if not isinstance(label, str):
                raise MriAnnotationServiceException("Label must be a string.")
            trimmed_label = label.strip()
            if len(trimmed_label) > 100:
                raise MriAnnotationServiceException("Label exceeds maximum length of 100 characters.")

        # Input Validation for Comment
        trimmed_comment = ""
        if comment is not None:
            if not isinstance(comment, str):
                raise MriAnnotationServiceException("Comment must be a string.")
            trimmed_comment = comment.strip()
            if len(trimmed_comment) > 2000:
                raise MriAnnotationServiceException("Comment exceeds maximum length of 2000 characters.")

        # Fetch Annotation
        conn = self._get_connection()
        try:
            row = conn.execute("SELECT * FROM mri_point_annotations WHERE annotation_id = ?;", (annotation_id,)).fetchone()
            if not row:
                raise MriAnnotationServiceException("Annotation not found.")
            
            doctor_id = row["doctor_id"]
            patient_id = row["patient_id"]
            scan_id = row["scan_id"]
        finally:
            conn.close()

        # Ownership enforcement (only creator can update)
        if doctor_id != actor.id:
            self._log_audit_event("POINT_ANNOTATION_UPDATED", actor, "FAILED", f"Unauthorized attempt by user {actor.id} to update annotation ID {annotation_id} authored by {doctor_id}.")
            raise MriAnnotationServiceException("Access denied.")

        # Enforce security checks
        if not self.auth_svc.can_access_patient(actor, patient_id):
            raise MriAnnotationServiceException("Access denied.")

        if not self.auth_svc.can_access_scan(actor, scan_id):
            raise MriAnnotationServiceException("Access denied.")

        # Encryption
        encrypted_comment = None
        if trimmed_comment:
            if not self.encryption_service:
                raise MriAnnotationServiceException("PII Encryption Service is not initialized.")
            encrypted_comment = self.encryption_service.encrypt(trimmed_comment)

        # Update
        now = datetime.datetime.utcnow().isoformat()
        conn = self._get_connection()
        try:
            with conn:
                conn.execute(
                    """UPDATE mri_point_annotations
                       SET x_normalized = ?, y_normalized = ?, label = ?, encrypted_comment = ?, updated_at = ?
                       WHERE annotation_id = ?;""",
                    (x_val, y_val, trimmed_label, encrypted_comment, now, annotation_id)
                )
        finally:
            conn.close()

        self._log_audit_event(
            "POINT_ANNOTATION_UPDATED",
            actor,
            "SUCCESS",
            f"Updated point annotation ID {annotation_id} for scan {scan_id}."
        )

        return {
            "annotation_id": annotation_id,
            "scan_id": scan_id,
            "patient_id": patient_id,
            "doctor_id": actor.id,
            "x": x_val,
            "y": y_val,
            "label": trimmed_label,
            "comment": trimmed_comment,
            "created_at": row["created_at"],
            "updated_at": now,
            "status": row["status"]
        }

    def archive_annotation(
        self,
        actor: User,
        annotation_id: int
    ) -> Dict[str, Any]:
        """Soft-archives/deletes an existing point annotation."""
        if not actor:
            raise MriAnnotationServiceException("Authentication required.")

        # Fetch Annotation
        conn = self._get_connection()
        try:
            row = conn.execute("SELECT * FROM mri_point_annotations WHERE annotation_id = ?;", (annotation_id,)).fetchone()
            if not row:
                raise MriAnnotationServiceException("Annotation not found.")
            
            doctor_id = row["doctor_id"]
            patient_id = row["patient_id"]
            scan_id = row["scan_id"]
        finally:
            conn.close()

        # Ownership check
        if doctor_id != actor.id:
            self._log_audit_event("POINT_ANNOTATION_ARCHIVED", actor, "FAILED", f"Unauthorized attempt by user {actor.id} to archive annotation ID {annotation_id} authored by {doctor_id}.")
            raise MriAnnotationServiceException("Access denied.")

        # Enforce security checks
        if not self.auth_svc.can_access_patient(actor, patient_id):
            raise MriAnnotationServiceException("Access denied.")

        if not self.auth_svc.can_access_scan(actor, scan_id):
            raise MriAnnotationServiceException("Access denied.")

        # Update status
        now = datetime.datetime.utcnow().isoformat()
        conn = self._get_connection()
        try:
            with conn:
                conn.execute(
                    """UPDATE mri_point_annotations
                       SET status = 'archived', updated_at = ?
                       WHERE annotation_id = ?;""",
                    (now, annotation_id)
                )
        finally:
            conn.close()

        self._log_audit_event(
            "POINT_ANNOTATION_ARCHIVED",
            actor,
            "SUCCESS",
            f"Archived point annotation ID {annotation_id}."
        )

        return {
            "annotation_id": annotation_id,
            "scan_id": scan_id,
            "patient_id": patient_id,
            "doctor_id": doctor_id,
            "status": "archived"
        }

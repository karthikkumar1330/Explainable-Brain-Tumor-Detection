import os
import sqlite3
import datetime
import logging
from typing import Optional, List, Dict, Any

from security.application.authorization_service import AuthorizationService
from security.infrastructure.encryption_service import PIIEncryptionService
from security.domain.entities import Role, User, SecurityAuditLog
from security.infrastructure.repository import SQLiteUserRepository

class ClinicianNoteServiceException(Exception):
    """Exception raised for errors in the ClinicianNoteService."""
    pass

class ClinicianNoteService:
    """Service to manage human-authored clinician review notes/observations associated with MRI scans."""

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        self.logger = logging.getLogger("clinician_note_service")
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
            from security.infrastructure.audit_context import audit_context
            ctx = audit_context.get()
            ip_val = ctx.get("client_ip", "127.0.0.1")
            ua_val = ctx.get("user_agent", "System")

            audit_log = SecurityAuditLog(
                id=None,
                timestamp=datetime.datetime.utcnow().isoformat(),
                event_type=event_type,
                user_id=actor.id if actor else None,
                email=actor.email if actor else None,
                ip_address=ip_val,
                status=status,
                details=details,
                user_agent=ua_val
            )
            self.user_repo.log_security_event(audit_log)
        except Exception as e:
            self.logger.warning(f"Failed to write note security audit log: {e}")

    def create_note(
        self,
        actor: User,
        scan_id: int,
        content: str
    ) -> Dict[str, Any]:
        """Creates a clinician note for a given MRI scan."""
        # 1. Check Authentication and Role
        if not actor:
            raise ClinicianNoteServiceException("Authentication required.")
        
        role_val = actor.role.value if hasattr(actor.role, 'value') else str(actor.role).lower()
        if role_val != "doctor" and role_val != "admin":
            raise ClinicianNoteServiceException("Unauthorized. Doctor role required.")

        # 2. Input Validation for scan_id
        if scan_id is None or not isinstance(scan_id, int) or scan_id <= 0:
            raise ClinicianNoteServiceException("Invalid scan ID.")

        # 3. Input Validation for content
        if not content:
            raise ClinicianNoteServiceException("Clinician note content is required.")
        
        trimmed_content = content.strip()
        if not trimmed_content:
            raise ClinicianNoteServiceException("Clinician note content cannot be empty.")

        if len(trimmed_content) > 5000:
            raise ClinicianNoteServiceException("Clinician note exceeds the maximum length of 5000 characters.")

        # 4. Fetch Scan and Patient ID
        conn = self._get_connection()
        try:
            scan_row = conn.execute("SELECT id, patient_id FROM mri_scans WHERE id = ?;", (scan_id,)).fetchone()
            if not scan_row:
                raise ClinicianNoteServiceException("MRI Scan not found.")
            patient_id = scan_row["patient_id"]
        finally:
            conn.close()

        # 5. Enforce Security Access Checks
        if not self.auth_svc.can_access_patient(actor, patient_id):
            self._log_audit_event("CLINICIAN_NOTE_CREATED", actor, "FAILED", f"Access denied creating note for patient: {patient_id}")
            raise ClinicianNoteServiceException("Access denied.")

        if not self.auth_svc.can_access_scan(actor, scan_id):
            self._log_audit_event("CLINICIAN_NOTE_CREATED", actor, "FAILED", f"Access denied creating note for scan: {scan_id}")
            raise ClinicianNoteServiceException("Access denied.")

        if not self.auth_svc.validate_scan_patient_match(scan_id, patient_id):
            raise ClinicianNoteServiceException("Mismatched patient and scan.")

        # 6. Encryption
        if not self.encryption_service:
            raise ClinicianNoteServiceException("PII Encryption Service is not initialized.")
        encrypted_content = self.encryption_service.encrypt(trimmed_content)

        # 7. Insert
        now = datetime.datetime.utcnow().isoformat()
        conn = self._get_connection()
        try:
            with conn:
                cursor = conn.cursor()
                cursor.execute(
                    """INSERT INTO clinician_notes (patient_id, scan_id, doctor_id, encrypted_content, created_at, updated_at, status)
                       VALUES (?, ?, ?, ?, ?, ?, 'active');""",
                    (patient_id, scan_id, actor.id, encrypted_content, now, now)
                )
                note_id = cursor.lastrowid
        finally:
            conn.close()

        self._log_audit_event(
            "CLINICIAN_NOTE_CREATED",
            actor,
            "SUCCESS",
            f"Created clinician note ID {note_id} for patient {patient_id} and scan {scan_id}."
        )

        return {
            "note_id": note_id,
            "scan_id": scan_id,
            "patient_id": patient_id,
            "doctor_id": actor.id,
            "content": trimmed_content,
            "created_at": now,
            "updated_at": now,
            "status": "active"
        }

    def get_notes_for_scan(
        self,
        actor: User,
        scan_id: int
    ) -> List[Dict[str, Any]]:
        """Retrieves active clinician notes for a given MRI scan."""
        if not actor:
            raise ClinicianNoteServiceException("Authentication required.")

        # Validation
        if scan_id is None or not isinstance(scan_id, int) or scan_id <= 0:
            raise ClinicianNoteServiceException("Invalid scan ID.")

        conn = self._get_connection()
        try:
            scan_row = conn.execute("SELECT patient_id FROM mri_scans WHERE id = ?;", (scan_id,)).fetchone()
            if not scan_row:
                raise ClinicianNoteServiceException("MRI Scan not found.")
            patient_id = scan_row["patient_id"]
        finally:
            conn.close()

        # Enforce security checks
        if not self.auth_svc.can_access_patient(actor, patient_id):
            raise ClinicianNoteServiceException("Access denied.")

        if not self.auth_svc.can_access_scan(actor, scan_id):
            raise ClinicianNoteServiceException("Access denied.")

        conn = self._get_connection()
        try:
            rows = conn.execute(
                """SELECT n.*, u.full_name as doctor_name, u.email as doctor_email
                   FROM clinician_notes n
                   JOIN users u ON n.doctor_id = u.id
                   WHERE n.scan_id = ? AND n.status = 'active'
                   ORDER BY n.created_at DESC;""",
                (scan_id,)
            ).fetchall()
        finally:
            conn.close()

        if not self.encryption_service:
            raise ClinicianNoteServiceException("PII Encryption Service is not initialized.")

        results = []
        for r in rows:
            decrypted = self.encryption_service.decrypt(r["encrypted_content"])
            results.append({
                "note_id": r["note_id"],
                "scan_id": r["scan_id"],
                "patient_id": r["patient_id"],
                "doctor_id": r["doctor_id"],
                "doctor_name": r["doctor_name"],
                "doctor_email": r["doctor_email"],
                "content": decrypted,
                "created_at": r["created_at"],
                "updated_at": r["updated_at"],
                "status": r["status"]
            })
        return results

    def update_note(
        self,
        actor: User,
        note_id: int,
        content: str
    ) -> Dict[str, Any]:
        """Updates content of an existing clinician note. Only the authoring doctor may modify it."""
        if not actor:
            raise ClinicianNoteServiceException("Authentication required.")

        # Input Validation for content
        if not content:
            raise ClinicianNoteServiceException("Clinician note content is required.")
        
        trimmed_content = content.strip()
        if not trimmed_content:
            raise ClinicianNoteServiceException("Clinician note content cannot be empty.")

        if len(trimmed_content) > 5000:
            raise ClinicianNoteServiceException("Clinician note exceeds the maximum length of 5000 characters.")

        # Fetch Note
        conn = self._get_connection()
        try:
            note_row = conn.execute("SELECT * FROM clinician_notes WHERE note_id = ?;", (note_id,)).fetchone()
            if not note_row:
                raise ClinicianNoteServiceException("Clinician note not found.")
            
            doctor_id = note_row["doctor_id"]
            patient_id = note_row["patient_id"]
            scan_id = note_row["scan_id"]
        finally:
            conn.close()

        # Enforce ownership check (Only the creator can modify their note)
        if doctor_id != actor.id:
            self._log_audit_event("CLINICIAN_NOTE_UPDATED", actor, "FAILED", f"Unauthorized attempt by user {actor.id} to update note ID {note_id} authored by {doctor_id}.")
            raise ClinicianNoteServiceException("Access denied. You can only modify notes you created.")

        # Enforce standard security checks
        if not self.auth_svc.can_access_patient(actor, patient_id):
            raise ClinicianNoteServiceException("Access denied.")

        if not self.auth_svc.can_access_scan(actor, scan_id):
            raise ClinicianNoteServiceException("Access denied.")

        # Encryption
        if not self.encryption_service:
            raise ClinicianNoteServiceException("PII Encryption Service is not initialized.")
        encrypted_content = self.encryption_service.encrypt(trimmed_content)

        # Update
        now = datetime.datetime.utcnow().isoformat()
        conn = self._get_connection()
        try:
            with conn:
                conn.execute(
                    """UPDATE clinician_notes
                       SET encrypted_content = ?, updated_at = ?
                       WHERE note_id = ?;""",
                    (encrypted_content, now, note_id)
                )
        finally:
            conn.close()

        self._log_audit_event(
            "CLINICIAN_NOTE_UPDATED",
            actor,
            "SUCCESS",
            f"Updated clinician note ID {note_id} for scan {scan_id}."
        )

        return {
            "note_id": note_id,
            "scan_id": scan_id,
            "patient_id": patient_id,
            "doctor_id": actor.id,
            "content": trimmed_content,
            "created_at": note_row["created_at"],
            "updated_at": now,
            "status": note_row["status"]
        }

    def archive_note(
        self,
        actor: User,
        note_id: int
    ) -> Dict[str, Any]:
        """Soft-deletes/archives an existing clinician note."""
        if not actor:
            raise ClinicianNoteServiceException("Authentication required.")

        # Fetch Note
        conn = self._get_connection()
        try:
            note_row = conn.execute("SELECT * FROM clinician_notes WHERE note_id = ?;", (note_id,)).fetchone()
            if not note_row:
                raise ClinicianNoteServiceException("Clinician note not found.")
            
            doctor_id = note_row["doctor_id"]
            patient_id = note_row["patient_id"]
            scan_id = note_row["scan_id"]
        finally:
            conn.close()

        # Enforce ownership check
        if doctor_id != actor.id:
            self._log_audit_event("CLINICIAN_NOTE_ARCHIVED", actor, "FAILED", f"Unauthorized attempt by user {actor.id} to archive note ID {note_id} authored by {doctor_id}.")
            raise ClinicianNoteServiceException("Access denied. You can only archive notes you created.")

        # Enforce security checks
        if not self.auth_svc.can_access_patient(actor, patient_id):
            raise ClinicianNoteServiceException("Access denied.")

        if not self.auth_svc.can_access_scan(actor, scan_id):
            raise ClinicianNoteServiceException("Access denied.")

        # Update
        now = datetime.datetime.utcnow().isoformat()
        conn = self._get_connection()
        try:
            with conn:
                conn.execute(
                    """UPDATE clinician_notes
                       SET status = 'archived', updated_at = ?
                       WHERE note_id = ?;""",
                    (now, note_id)
                )
        finally:
            conn.close()

        self._log_audit_event(
            "CLINICIAN_NOTE_ARCHIVED",
            actor,
            "SUCCESS",
            f"Archived clinician note ID {note_id}."
        )

        return {
            "note_id": note_id,
            "scan_id": scan_id,
            "patient_id": patient_id,
            "doctor_id": doctor_id,
            "status": "archived"
        }

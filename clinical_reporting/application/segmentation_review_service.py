import os
import sqlite3
import datetime
import logging
from typing import Optional, Dict, Any

from security.application.authorization_service import AuthorizationService
from security.infrastructure.encryption_service import PIIEncryptionService
from security.domain.entities import Role, User, SecurityAuditLog
from security.infrastructure.repository import SQLiteUserRepository
from persistence.infrastructure.repository import SQLitePersistenceRepository

class SegmentationReviewServiceException(Exception):
    """Exception raised for errors in the SegmentationReviewService."""
    pass

class SegmentationReviewService:
    """Service to manage clinician reviews and corrections on UNeXt segmentation outputs."""

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        self.logger = logging.getLogger("segmentation_review_service")
        self.auth_svc = AuthorizationService(db_path=self.db_path)
        self.user_repo = SQLiteUserRepository(db_path=self.db_path)
        self.db_repo = SQLitePersistenceRepository(db_path=self.db_path)
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
            self.logger.warning(f"Failed to write review security audit log: {e}")

    def create_or_update_review(
        self,
        actor: User,
        scan_id: int,
        patient_id: str,
        status: str,
        comment: Optional[str] = None
    ) -> Dict[str, Any]:
        """Creates or updates a clinician's segmentation review for an MRI scan."""
        # 1. Authenticated check
        if not actor:
            raise SegmentationReviewServiceException("Authentication required.")

        # 2. Doctor Role validation
        role_val = actor.role.value if hasattr(actor.role, 'value') else str(actor.role).lower()
        if role_val != "doctor" and role_val != "admin":
            raise SegmentationReviewServiceException("Unauthorized. Doctor role required.")

        # 3. Input Validation for scan ID
        if scan_id is None or not isinstance(scan_id, int) or scan_id <= 0:
            raise SegmentationReviewServiceException("Invalid scan ID.")

        # 4. Enforce Security Access Checks (Authorization Gate & IDOR protection)
        if not self.auth_svc.can_access_patient(actor, patient_id):
            self._log_audit_event("SEGMENTATION_REVIEW_SUBMITTED", actor, "FAILED", f"Access denied for patient: {patient_id}")
            raise SegmentationReviewServiceException("Access denied.")

        if not self.auth_svc.can_access_scan(actor, scan_id):
            self._log_audit_event("SEGMENTATION_REVIEW_SUBMITTED", actor, "FAILED", f"Access denied for scan: {scan_id}")
            raise SegmentationReviewServiceException("Access denied.")

        if not self.auth_svc.validate_scan_patient_match(scan_id, patient_id):
            raise SegmentationReviewServiceException("Mismatched patient and scan.")

        # 5. Fetch Prediction associated with scan
        conn = self._get_connection()
        try:
            pred_row = conn.execute(
                "SELECT id FROM predictions WHERE scan_id = ? ORDER BY id DESC LIMIT 1;",
                (scan_id,)
            ).fetchone()
            if not pred_row:
                raise SegmentationReviewServiceException("No prediction found for this scan. Review cannot be saved.")
            prediction_id = pred_row["id"]
        finally:
            conn.close()

        # 6. Status Enlistment checks
        allowed_statuses = ["APPROVED_AS_IS", "APPROVED_WITH_CORRECTIONS", "REJECTED"]
        if status not in allowed_statuses:
            raise SegmentationReviewServiceException(f"Invalid status. Must be one of {allowed_statuses}")

        # 7. APPROVED_WITH_CORRECTIONS safety check (requires at least one active annotation)
        if status == "APPROVED_WITH_CORRECTIONS":
            conn = self._get_connection()
            try:
                p_count = conn.execute(
                    "SELECT COUNT(*) FROM mri_point_annotations WHERE scan_id = ? AND doctor_id = ? AND status = 'active';",
                    (scan_id, actor.id)
                ).fetchone()[0]
                r_count = conn.execute(
                    "SELECT COUNT(*) FROM mri_rectangle_annotations WHERE scan_id = ? AND doctor_id = ? AND status = 'active';",
                    (scan_id, actor.id)
                ).fetchone()[0]
            finally:
                conn.close()

            if (p_count + r_count) == 0:
                raise SegmentationReviewServiceException(
                    "Cannot approve with corrections because no active point or rectangle annotations were found for this doctor on this scan."
                )

        # 8. PII comment encryption
        trimmed_comment = ""
        encrypted_comment = None
        if comment is not None:
            if not isinstance(comment, str):
                raise SegmentationReviewServiceException("Comment must be a string.")
            trimmed_comment = comment.strip()
            if len(trimmed_comment) > 2000:
                raise SegmentationReviewServiceException("Comment exceeds maximum length of 2000 characters.")
            if trimmed_comment:
                if not self.encryption_service:
                    raise SegmentationReviewServiceException("PII Encryption Service is not initialized.")
                encrypted_comment = self.encryption_service.encrypt(trimmed_comment)

        # 9. Persist Review
        # First check if review exists for audit event type (created vs updated)
        conn = self._get_connection()
        try:
            existing_review = conn.execute(
                "SELECT review_id FROM segmentation_reviews WHERE scan_id = ? AND reviewer_id = ?;",
                (scan_id, actor.id)
            ).fetchone()
        finally:
            conn.close()

        review_data = self.db_repo.save_segmentation_review(
            scan_id=scan_id,
            patient_id=patient_id,
            reviewer_id=actor.id,
            prediction_id=prediction_id,
            review_status=status,
            encrypted_comment=encrypted_comment
        )

        event_type = "SEGMENTATION_REVIEW_UPDATED" if existing_review else "SEGMENTATION_REVIEW_SUBMITTED"
        self._log_audit_event(
            event_type,
            actor,
            "SUCCESS",
            f"Review ID {review_data['review_id']} submitted with status {status} for scan {scan_id}."
        )

        # Decrypt comment for response dictionary
        review_data["comment"] = trimmed_comment
        return review_data

    def get_review_for_scan(
        self,
        actor: User,
        scan_id: int
    ) -> Optional[Dict[str, Any]]:
        """Retrieves the clinical review (if any) for a given scan."""
        if not actor:
            raise SegmentationReviewServiceException("Authentication required.")

        if scan_id is None or not isinstance(scan_id, int) or scan_id <= 0:
            raise SegmentationReviewServiceException("Invalid scan ID.")

        # Enforce security access check
        if not self.auth_svc.can_access_scan(actor, scan_id):
            raise SegmentationReviewServiceException("Access denied.")

        review_data = self.db_repo.get_segmentation_review(scan_id)
        if not review_data:
            return None

        # Decrypt comment
        decrypted_comment = ""
        if review_data["encrypted_comment"]:
            if not self.encryption_service:
                raise SegmentationReviewServiceException("PII Encryption Service is not initialized.")
            decrypted_comment = self.encryption_service.decrypt(review_data["encrypted_comment"])

        review_data["comment"] = decrypted_comment
        return review_data

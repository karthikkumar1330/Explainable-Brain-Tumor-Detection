import os
import sqlite3
import datetime
import logging
import json
from typing import Optional, List, Dict, Any

from security.application.authorization_service import AuthorizationService
from security.domain.entities import Role, User, SecurityAuditLog
from security.infrastructure.repository import SQLiteUserRepository
from clinical_reporting.application.notification_service import NotificationService

class FollowupScheduleServiceException(Exception):
    """Exception raised for errors in the FollowupScheduleService."""
    pass

class FollowupScheduleService:
    """Service to manage follow-up schedules for patients."""

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        self.logger = logging.getLogger("followup_schedule_service")
        self.auth_svc = AuthorizationService(db_path=self.db_path)
        self.user_repo = SQLiteUserRepository(db_path=self.db_path)
        self.notif_svc = NotificationService(db_path=self.db_path)

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
            self.logger.warning(f"Failed to write follow-up security audit log: {e}")

    def _notify_affected_parties(
        self,
        event_type: str,
        followup_id: int,
        patient_uuid: str,
        doctor_id: int,
        actor: User,
        title: str,
        message: str
    ) -> None:
        try:
            # Resolve patient user
            pat_user = self.user_repo.get_by_uuid(patient_uuid)
            recipients = set()

            # 1. Patient is always a recipient (if different from actor)
            if pat_user and pat_user.id != actor.id:
                recipients.add((pat_user.id, "patient"))

            # 2. The doctor associated with the follow-up is a recipient (if different from actor)
            if doctor_id and doctor_id != actor.id:
                doc_user = self.user_repo.get_by_id(doctor_id)
                if doc_user:
                    recipients.add((doc_user.id, "doctor"))

            # 3. Any other assigned doctors (if different from actor)
            conn = self._get_connection()
            try:
                rows = conn.execute(
                    "SELECT doctor_id FROM doctor_patient_assignments WHERE patient_id = ?;",
                    (patient_uuid,)
                ).fetchall()
                for r in rows:
                    did = r["doctor_id"]
                    if did != actor.id:
                        recipients.add((did, "doctor"))
            except Exception as dbe:
                self.logger.warning(f"Failed to query assigned doctors for notification: {dbe}")
            finally:
                conn.close()

            # Send notifications
            meta_json = json.dumps({
                "followup_id": followup_id,
                "patient_id": patient_uuid
            })
            for rid, role in recipients:
                try:
                    self.notif_svc.create_notification(
                        user_id=rid,
                        type_=event_type,
                        title=title,
                        message=message,
                        metadata_json=meta_json
                    )
                except Exception as e:
                    self.logger.error(f"Failed to deliver in-app notification to user {rid}: {e}")
        except Exception as ex:
            self.logger.warning(f"Error in notification trigger dispatch: {ex}")

    def _parse_and_normalize_date(self, date_str: str) -> str:
        """Parses, validates, and normalizes a scheduled date string to a consistent format.

        Supports:
        - Date-only: YYYY-MM-DD
        - ISO datetime (naive): YYYY-MM-DDTHH:MM:SS (and variations)
        - Timezone-aware datetime: YYYY-MM-DDTHH:MM:SSZ or with timezone offset
        """
        if not date_str or not isinstance(date_str, str):
            raise ValueError("Scheduled date must be a non-empty string.")

        cleaned = date_str.strip()
        if not cleaned:
            raise ValueError("Scheduled date cannot be blank.")

        # 1. Date-only format: YYYY-MM-DD
        import re
        if re.match(r"^\d{4}-\d{2}-\d{2}$", cleaned):
            try:
                dt = datetime.datetime.strptime(cleaned, "%Y-%m-%d")
                return dt.strftime("%Y-%m-%d")
            except ValueError as e:
                raise ValueError(f"Invalid date components: {e}")

        # 2. Try parsing datetime formats using fromisoformat
        try:
            dt = datetime.datetime.fromisoformat(cleaned)
        except ValueError:
            try:
                # Support space separator instead of T
                dt = datetime.datetime.fromisoformat(cleaned.replace(" ", "T"))
            except ValueError:
                # Fallback to general parsing loop for common formats
                for fmt in ["%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"]:
                    try:
                        dt = datetime.datetime.strptime(cleaned, fmt)
                        break
                    except ValueError:
                        continue
                else:
                    raise ValueError("Invalid date/time format. Supported formats: YYYY-MM-DD, ISO 8601.")

        # 3. Handle timezone conversions
        if dt.tzinfo is not None:
            # Convert to UTC and format with 'Z'
            utc_dt = dt.astimezone(datetime.timezone.utc)
            return utc_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        else:
            # Naive datetime: format as YYYY-MM-DDTHH:MM:SS
            return dt.strftime("%Y-%m-%dT%H:%M:%S")

    def create_followup(
        self,
        actor: User,
        patient_id: str,
        scheduled_date: str,
        reason: Optional[str] = None,
        notes: Optional[str] = None
    ) -> Dict[str, Any]:
        """Creates a follow-up schedule for an assigned patient."""
        # 1. Authenticate Actor
        if not actor:
            raise FollowupScheduleServiceException("Authentication required.")

        # 2. Check Role (only doctor/admin can create)
        role_val = actor.role.value if hasattr(actor.role, 'value') else str(actor.role).lower()
        if role_val not in ["doctor", "admin"]:
            self._log_audit_event("FOLLOWUP_CREATED", actor, "FAILED", f"Access denied: User is not authorized to create follow-ups.")
            raise FollowupScheduleServiceException("Access denied.")

        # 3. Input Validation for patient_id
        if not patient_id or not isinstance(patient_id, str) or not patient_id.strip():
            raise FollowupScheduleServiceException("Patient ID is required.")
        pid_clean = patient_id.strip()

        # 4. Check patient existence, authorization and write using a single connection
        conn = self._get_connection()
        try:
            pat_row = conn.execute("SELECT 1 FROM patients WHERE patient_id = ?;", (pid_clean,)).fetchone()
            if not pat_row:
                raise FollowupScheduleServiceException(f"Patient with ID {pid_clean} not found.")

            # 5. Check Authorization
            if not self.auth_svc.can_access_patient(actor, pid_clean, conn=conn):
                self._log_audit_event("FOLLOWUP_CREATED", actor, "FAILED", f"Access denied: Doctor not assigned to patient: {pid_clean}")
                raise FollowupScheduleServiceException("Access denied.")

            # 6. Validate scheduled_date
            if not scheduled_date or not isinstance(scheduled_date, str):
                raise FollowupScheduleServiceException("Scheduled date is required.")
            try:
                sd_clean = self._parse_and_normalize_date(scheduled_date)
            except ValueError as e:
                raise FollowupScheduleServiceException(f"Invalid scheduled date format. {str(e)}")

            # 7. Validate reason / notes length
            reason_clean = reason.strip() if reason else ""
            if len(reason_clean) > 2000:
                raise FollowupScheduleServiceException("Reason exceeds maximum length of 2000 characters.")
            notes_clean = notes.strip() if notes else ""
            if len(notes_clean) > 2000:
                raise FollowupScheduleServiceException("Notes exceed maximum length of 2000 characters.")

            # 8. Save to DB
            now_str = datetime.datetime.utcnow().isoformat()
            with conn:
                cursor = conn.execute(
                    """
                    INSERT INTO followup_schedules (patient_id, doctor_id, scheduled_date, status, reason, notes, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?);
                    """,
                    (pid_clean, actor.id, sd_clean, "scheduled", reason_clean, notes_clean, now_str, now_str)
                )
                followup_id = cursor.lastrowid

            self._log_audit_event("FOLLOWUP_CREATED", actor, "SUCCESS", f"Created follow-up {followup_id} for patient {pid_clean}")
            self._notify_affected_parties(
                event_type="FOLLOWUP_CREATED",
                followup_id=followup_id,
                patient_uuid=pid_clean,
                doctor_id=actor.id,
                actor=actor,
                title="New Follow-up Scheduled",
                message=f"A new follow-up appointment has been scheduled for {sd_clean}."
            )
            return {
                "followup_id": followup_id,
                "patient_id": pid_clean,
                "doctor_id": actor.id,
                "scheduled_date": sd_clean,
                "status": "scheduled",
                "reason": reason_clean,
                "notes": notes_clean,
                "created_at": now_str,
                "updated_at": now_str,
                "completed_at": None
            }
        except Exception as e:
            if isinstance(e, FollowupScheduleServiceException):
                raise e
            self.logger.error(f"Error creating follow-up: {e}")
            raise FollowupScheduleServiceException("Failed to create follow-up due to an internal database error.")
        finally:
            conn.close()

    def get_followups_for_patient(
        self,
        actor: User,
        patient_id: str
    ) -> List[Dict[str, Any]]:
        """Retrieves all follow-up schedules for a patient."""
        # 1. Authenticate Actor
        if not actor:
            raise FollowupScheduleServiceException("Authentication required.")

        # 2. Input Validation for patient_id
        if not patient_id or not isinstance(patient_id, str) or not patient_id.strip():
            raise FollowupScheduleServiceException("Patient ID is required.")
        pid_clean = patient_id.strip()

        # 3. Check patient existence
        conn = self._get_connection()
        try:
            pat_row = conn.execute("SELECT 1 FROM patients WHERE patient_id = ?;", (pid_clean,)).fetchone()
            if not pat_row:
                raise FollowupScheduleServiceException(f"Patient with ID {pid_clean} not found.")
        finally:
            conn.close()

        # 4. Check Authorization
        if not self.auth_svc.can_access_patient(actor, pid_clean):
            self._log_audit_event("FOLLOWUP_RETRIEVED", actor, "FAILED", f"Access denied: User not authorized to access patient: {pid_clean}")
            raise FollowupScheduleServiceException("Access denied.")

        # 5. Fetch from DB
        conn = self._get_connection()
        try:
            rows = conn.execute(
                """
                SELECT followup_id, patient_id, doctor_id, scheduled_date, status, reason, notes, created_at, updated_at, completed_at
                FROM followup_schedules
                WHERE patient_id = ?
                ORDER BY scheduled_date ASC;
                """,
                (pid_clean,)
            ).fetchall()

            return [dict(row) for row in rows]
        except Exception as e:
            self.logger.error(f"Error retrieving follow-ups: {e}")
            raise FollowupScheduleServiceException("Failed to retrieve follow-ups due to an internal database error.")
        finally:
            conn.close()

    def update_followup(
        self,
        actor: User,
        followup_id: int,
        scheduled_date: Optional[str] = None,
        status: Optional[str] = None,
        reason: Optional[str] = None,
        notes: Optional[str] = None
    ) -> Dict[str, Any]:
        """Updates an existing follow-up schedule."""
        # 1. Authenticate Actor
        if not actor:
            raise FollowupScheduleServiceException("Authentication required.")

        # 2. Check Role (only doctor/admin can update)
        role_val = actor.role.value if hasattr(actor.role, 'value') else str(actor.role).lower()
        if role_val not in ["doctor", "admin"]:
            self._log_audit_event("FOLLOWUP_MODIFIED", actor, "FAILED", f"Access denied: User is not authorized to update follow-up {followup_id}")
            raise FollowupScheduleServiceException("Access denied.")

        # 3. Input Validation for followup_id
        if followup_id is None or not isinstance(followup_id, int) or followup_id <= 0:
            raise FollowupScheduleServiceException("Invalid follow-up ID.")

        # 4. Fetch existing follow-up to retrieve patient_id and verify existence
        # 4. Check existence, authorization and write using a single connection
        conn = self._get_connection()
        try:
            row = conn.execute(
                "SELECT patient_id, status FROM followup_schedules WHERE followup_id = ?;", (followup_id,)
            ).fetchone()
            if not row:
                raise FollowupScheduleServiceException(f"Follow-up schedule with ID {followup_id} not found.")
            patient_id = row["patient_id"]

            # 5. Check patient existence
            pat_row = conn.execute("SELECT 1 FROM patients WHERE patient_id = ?;", (patient_id,)).fetchone()
            if not pat_row:
                raise FollowupScheduleServiceException(f"Patient with ID {patient_id} not found.")

            # 6. Check Authorization using the open connection
            if not self.auth_svc.can_access_patient(actor, patient_id, conn=conn):
                self._log_audit_event("FOLLOWUP_MODIFIED", actor, "FAILED", f"Access denied updating follow-up: Doctor not assigned to patient: {patient_id}")
                raise FollowupScheduleServiceException("Access denied.")

            # 7. Validate scheduled_date if provided
            sd_clean = None
            if scheduled_date is not None:
                try:
                    sd_clean = self._parse_and_normalize_date(scheduled_date)
                except ValueError as e:
                    raise FollowupScheduleServiceException(f"Invalid scheduled date format. {str(e)}")

            # 8. Validate status if provided
            status_clean = None
            if status is not None:
                if not isinstance(status, str):
                    raise FollowupScheduleServiceException("Status must be a string.")
                status_clean = status.strip().lower()
                if status_clean not in ["scheduled", "completed", "cancelled", "overdue"]:
                    raise FollowupScheduleServiceException("Invalid follow-up status.")

            # 9. Update DB
            now_str = datetime.datetime.utcnow().isoformat()
            with conn:
                # Build dynamic query
                fields = []
                params = []
                if sd_clean is not None:
                    fields.append("scheduled_date = ?")
                    params.append(sd_clean)
                if status_clean is not None:
                    fields.append("status = ?")
                    params.append(status_clean)
                    if status_clean in ["completed", "cancelled"]:
                        fields.append("completed_at = ?")
                        params.append(now_str)
                    else:
                        fields.append("completed_at = NULL")
                if reason is not None:
                    reason_clean = reason.strip()
                    if len(reason_clean) > 2000:
                        raise FollowupScheduleServiceException("Reason exceeds maximum length of 2000 characters.")
                    fields.append("reason = ?")
                    params.append(reason_clean)
                if notes is not None:
                    notes_clean = notes.strip()
                    if len(notes_clean) > 2000:
                        raise FollowupScheduleServiceException("Notes exceed maximum length of 2000 characters.")
                    fields.append("notes = ?")
                    params.append(notes_clean)

                if fields:
                    fields.append("updated_at = ?")
                    params.append(now_str)
                    params.append(followup_id)

                    conn.execute(
                        f"UPDATE followup_schedules SET {', '.join(fields)} WHERE followup_id = ?;",
                        tuple(params)
                    )

            # Fetch updated row
            updated_row = conn.execute(
                """
                SELECT followup_id, patient_id, doctor_id, scheduled_date, status, reason, notes, created_at, updated_at, completed_at
                FROM followup_schedules
                WHERE followup_id = ?;
                """,
                (followup_id,)
            ).fetchone()

            self._log_audit_event("FOLLOWUP_MODIFIED", actor, "SUCCESS", f"Updated follow-up {followup_id} status to {status_clean or 'unchanged'}")

            # Send notifications based on what was updated
            if status_clean == "cancelled":
                self._notify_affected_parties(
                    event_type="FOLLOWUP_CANCELLED",
                    followup_id=followup_id,
                    patient_uuid=patient_id,
                    doctor_id=updated_row["doctor_id"],
                    actor=actor,
                    title="Follow-up Cancelled",
                    message=f"The follow-up appointment scheduled for {updated_row['scheduled_date']} has been cancelled."
                )
            elif sd_clean is not None or status_clean is not None or reason is not None or notes is not None:
                self._notify_affected_parties(
                    event_type="FOLLOWUP_UPDATED",
                    followup_id=followup_id,
                    patient_uuid=patient_id,
                    doctor_id=updated_row["doctor_id"],
                    actor=actor,
                    title="Follow-up Updated",
                    message=f"The follow-up appointment has been updated. Scheduled date: {updated_row['scheduled_date']}."
                )

            return dict(updated_row)
        except Exception as e:
            if isinstance(e, FollowupScheduleServiceException):
                raise e
            self.logger.error(f"Error updating follow-up: {e}")
            raise FollowupScheduleServiceException("Failed to update follow-up due to an internal database error.")
        finally:
            conn.close()

    def complete_followup(self, actor: User, followup_id: int) -> Dict[str, Any]:
        """Marks a follow-up schedule as completed."""
        return self.update_followup(actor, followup_id, status="completed")

    def cancel_followup(self, actor: User, followup_id: int) -> Dict[str, Any]:
        """Marks a follow-up schedule as cancelled."""
        return self.update_followup(actor, followup_id, status="cancelled")

"""ReportService application service for clinical report lifecycle and version management."""

import os
import sqlite3
import datetime
import shutil
import logging
import hashlib
from typing import List, Optional, Tuple, Dict, Any

from clinical_reporting.domain.entities import Report, ReportVersion, ReportStatus, can_transition


class ReportServiceException(Exception):
    """Base exception for report service operations."""
    pass


class InvalidTransitionException(ReportServiceException):
    """Exception raised when an invalid report status transition is attempted."""
    pass


class ReportNotFoundException(ReportServiceException):
    """Exception raised when a report cannot be found in the database."""
    pass


class VersionNotFoundException(ReportServiceException):
    """Exception raised when a specific version of a report cannot be found."""
    pass


class FinalizedReportException(ReportServiceException):
    """Exception raised when an operation is disallowed because the report is finalized."""
    pass


class ReportService:
    """Service class orchestrating the lifecycle, versioning, and state transitions of clinical reports."""

    def __init__(self, db_path: str, logger: Optional[logging.Logger] = None) -> None:
        """Initializes the ReportService.

        Args:
            db_path: Path to the SQLite database.
            logger: Optional logger.
        """
        self.db_path = db_path
        self.logger = logger or logging.getLogger("report_service")

    def _get_connection(self) -> sqlite3.Connection:
        """Creates a database connection with performance and foreign key configurations."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn

    def _calculate_checksum(self, filepath: str) -> str:
        """Computes the SHA-256 checksum of a file, falling back to an empty hash on failure."""
        if not filepath or not os.path.exists(filepath):
            return "da39a3ee5e6b4b0d3255bfef95601890afd80709"  # SHA-256 of empty string
        sha255 = hashlib.sha256()
        try:
            with open(filepath, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    sha255.update(chunk)
            return sha255.hexdigest()
        except Exception as e:
            self.logger.warning(f"Failed to calculate checksum for {filepath}: {e}")
            return "da39a3ee5e6b4b0d3255bfef95601890afd80709"

    def _generate_report_number(self, conn: sqlite3.Connection) -> str:
        """Generates the next sequential unique report number from the database."""
        year = datetime.datetime.now().year
        conn.execute("""
            INSERT INTO report_sequence (year, current_val)
            VALUES (?, 1)
            ON CONFLICT(year) DO UPDATE SET current_val = current_val + 1;
        """, (year,))
        row = conn.execute("SELECT current_val FROM report_sequence WHERE year = ?;", (year,)).fetchone()
        seq = row["current_val"]
        return f"RPT-{year}-{seq:06d}"

    def _log_lifecycle_change(
        self,
        conn: sqlite3.Connection,
        report_id: int,
        action: str,
        previous_status: str,
        new_status: str,
        actor: str
    ) -> None:
        """Logs a lifecycle state change into the security audit logs database table."""
        now_str = datetime.datetime.utcnow().isoformat()
        user_id = None
        email = None

        # Resolve user details if actor represents user ID or email
        if isinstance(actor, int) or (isinstance(actor, str) and actor.isdigit()):
            user_id = int(actor)
            row = conn.execute("SELECT email FROM users WHERE id = ?;", (user_id,)).fetchone()
            if row:
                email = row["email"]
        elif isinstance(actor, str) and "@" in actor:
            email = actor
            row = conn.execute("SELECT id FROM users WHERE email = ?;", (email,)).fetchone()
            if row:
                user_id = row["id"]
        else:
            email = actor

        details_str = f"Report ID: {report_id}, Action: {action}, Prev Status: {previous_status}, New Status: {new_status}"
        conn.execute("""
            INSERT INTO security_audit_logs (
                timestamp, event_type, user_id, email, ip_address, status, details, user_agent
            ) VALUES (?, 'REPORT_LIFECYCLE_CHANGE', ?, ?, '127.0.0.1', 'SUCCESS', ?, 'System');
        """, (now_str, user_id, email, details_str))

    def create_report(
        self,
        patient_id: str,
        created_by: Optional[str],
        report_type: str,
        pdf_path: str,
        json_path: str,
        prediction_id: int,
        initial_status: ReportStatus = ReportStatus.GENERATED
    ) -> Report:
        """Creates a new report record and its first version.

        Args:
            patient_id: Patient ID.
            created_by: ID or name of the clinician creator.
            report_type: Type of scan or report.
            pdf_path: Path to compiled PDF report.
            json_path: Path to compiled JSON report.
            prediction_id: ID of the corresponding prediction prediction.
            initial_status: Initial report status.

        Returns:
            The created Report domain entity.
        """
        self.logger.info(f"Creating new clinical report metadata for patient: {patient_id}")
        conn = self._get_connection()
        try:
            with conn:
                # 1. Generate unique report number
                report_number = self._generate_report_number(conn)
                # 2. Compute checksum
                checksum = self._calculate_checksum(pdf_path)
                now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                # 3. Insert report
                cursor = conn.execute("""
                    INSERT INTO reports (
                        report_number, patient_id, created_by, report_type,
                        current_version, status, created_at, updated_at,
                        pdf_path, json_path, checksum
                    ) VALUES (?, ?, ?, ?, 1, ?, ?, ?, ?, ?, ?);
                """, (
                    report_number, patient_id, created_by, report_type,
                    initial_status.value, now_str, now_str, pdf_path, json_path, checksum
                ))
                report_id = cursor.lastrowid

                # 4. Insert version 1
                conn.execute("""
                    INSERT INTO report_versions (
                        report_id, version_number, created_at, created_by,
                        reason, pdf_path, json_path, checksum, status, prediction_id
                    ) VALUES (?, 1, ?, ?, ?, ?, ?, ?, ?, ?);
                """, (
                    report_id, now_str, created_by, "Initial report generation",
                    pdf_path, json_path, checksum, initial_status.value, prediction_id
                ))

                # 5. Log audit trail
                self._log_lifecycle_change(
                    conn=conn,
                    report_id=report_id,
                    action="CREATE_REPORT",
                    previous_status="NONE",
                    new_status=initial_status.value,
                    actor=created_by or "System"
                )

                return Report(
                    report_id=report_id,
                    report_number=report_number,
                    patient_id=patient_id,
                    created_by=created_by,
                    report_type=report_type,
                    current_version=1,
                    status=initial_status,
                    created_at=now_str,
                    updated_at=now_str,
                    finalized_at=None,
                    archived_at=None,
                    pdf_path=pdf_path,
                    json_path=json_path,
                    checksum=checksum
                )
        except Exception as e:
            self.logger.error(f"Failed to create report: {e}")
            raise ReportServiceException(f"Failed to create report: {e}")
        finally:
            conn.close()

    def get_report(self, report_id: int) -> Report:
        """Fetches the current report metadata by ID.

        Args:
            report_id: Target Report ID.

        Returns:
            The Report domain entity.
        """
        conn = self._get_connection()
        try:
            row = conn.execute("SELECT * FROM reports WHERE report_id = ?;", (report_id,)).fetchone()
            if not row:
                raise ReportNotFoundException(f"Report with ID {report_id} not found.")
            return Report(
                report_id=row["report_id"],
                report_number=row["report_number"],
                patient_id=row["patient_id"],
                created_by=row["created_by"],
                report_type=row["report_type"],
                current_version=row["current_version"],
                status=ReportStatus(row["status"]),
                created_at=row["created_at"],
                updated_at=row["updated_at"],
                finalized_at=row["finalized_at"],
                archived_at=row["archived_at"],
                pdf_path=row["pdf_path"],
                json_path=row["json_path"],
                checksum=row["checksum"]
            )
        finally:
            conn.close()

    def get_report_versions(self, report_id: int) -> List[ReportVersion]:
        """Retrieves all versions of a clinical report.

        Args:
            report_id: Target Report ID.

        Returns:
            A list of ReportVersion domain entities.
        """
        conn = self._get_connection()
        try:
            row = conn.execute("SELECT 1 FROM reports WHERE report_id = ?;", (report_id,)).fetchone()
            if not row:
                raise ReportNotFoundException(f"Report with ID {report_id} not found.")

            rows = conn.execute("""
                SELECT * FROM report_versions
                WHERE report_id = ?
                ORDER BY version_number ASC;
            """, (report_id,)).fetchall()

            versions = []
            for r in rows:
                versions.append(ReportVersion(
                    version_id=r["version_id"],
                    report_id=r["report_id"],
                    version_number=r["version_number"],
                    created_at=r["created_at"],
                    created_by=r["created_by"],
                    reason=r["reason"],
                    pdf_path=r["pdf_path"],
                    json_path=r["json_path"],
                    checksum=r["checksum"],
                    status=ReportStatus(r["status"]),
                    prediction_id=r["prediction_id"]
                ))
            return versions
        finally:
            conn.close()

    def create_new_version(
        self,
        report_id: int,
        created_by: Optional[str],
        reason: str,
        pdf_path: Optional[str] = None,
        json_path: Optional[str] = None,
        prediction_id: Optional[int] = None,
        status: ReportStatus = ReportStatus.DRAFT
    ) -> ReportVersion:
        """Increments the report version, creating new files and entries safely.

        Args:
            report_id: Target Report ID.
            created_by: Action user identifier.
            reason: Explanation text for this version.
            pdf_path: Custom PDF file path (optional, will copy current if omitted).
            json_path: Custom JSON file path (optional, will copy current if omitted).
            prediction_id: Link to prediction run (optional).
            status: ReportStatus of the new version.

        Returns:
            The newly created ReportVersion.
        """
        self.logger.info(f"Creating new version for report {report_id}. Initiator: {created_by}")
        conn = self._get_connection()
        try:
            with conn:
                # 1. Fetch current active report
                row = conn.execute("SELECT * FROM reports WHERE report_id = ?;", (report_id,)).fetchone()
                if not row:
                    raise ReportNotFoundException(f"Report with ID {report_id} not found.")

                current_status = ReportStatus(row["status"])
                if current_status == ReportStatus.ARCHIVED:
                    raise FinalizedReportException("Cannot create a new version of an archived report.")

                new_version_num = row["current_version"] + 1

                # 2. Determine and copy paths if not specified
                final_pdf = pdf_path
                final_json = json_path

                if not final_pdf and row["pdf_path"]:
                    old_pdf = row["pdf_path"]
                    base, ext = os.path.splitext(old_pdf)
                    final_pdf = f"{base}_v{new_version_num}{ext}"
                    if os.path.exists(old_pdf):
                        try:
                            shutil.copy2(old_pdf, final_pdf)
                        except Exception as copy_err:
                            self.logger.warning(f"Could not copy PDF file on disk: {copy_err}")
                    else:
                        # Fallback for mock environments
                        final_pdf = old_pdf

                if not final_json and row["json_path"]:
                    old_json = row["json_path"]
                    base, ext = os.path.splitext(old_json)
                    final_json = f"{base}_v{new_version_num}{ext}"
                    if os.path.exists(old_json):
                        try:
                            shutil.copy2(old_json, final_json)
                        except Exception as copy_err:
                            self.logger.warning(f"Could not copy JSON file on disk: {copy_err}")
                    else:
                        final_json = old_json

                # Set placeholders if both old and new are empty
                if not final_pdf:
                    final_pdf = f"outputs/clinical_reports/report_{report_id}_v{new_version_num}.pdf"
                if not final_json:
                    final_json = f"outputs/clinical_reports/report_{report_id}_v{new_version_num}.json"

                # 3. Calculate checksum
                checksum = self._calculate_checksum(final_pdf)
                now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                # 4. Update reports table
                conn.execute("""
                    UPDATE reports
                    SET current_version = ?,
                        status = ?,
                        updated_at = ?,
                        pdf_path = ?,
                        json_path = ?,
                        checksum = ?
                    WHERE report_id = ?;
                """, (new_version_num, status.value, now_str, final_pdf, final_json, checksum, report_id))

                # 5. Maintain backwards compatibility in legacy clinical_reports table
                if prediction_id is not None:
                    conn.execute("""
                        UPDATE clinical_reports
                        SET pdf_path = ?,
                            json_path = ?,
                            prediction_id = ?,
                            created_at = ?
                        WHERE id = ?;
                    """, (final_pdf, final_json, prediction_id, now_str, report_id))
                else:
                    conn.execute("""
                        UPDATE clinical_reports
                        SET pdf_path = ?,
                            json_path = ?,
                            created_at = ?
                        WHERE id = ?;
                    """, (final_pdf, final_json, now_str, report_id))

                # 6. Insert new version record
                cursor = conn.execute("""
                    INSERT INTO report_versions (
                        report_id, version_number, created_at, created_by,
                        reason, pdf_path, json_path, checksum, status, prediction_id
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """, (
                    report_id, new_version_num, now_str, created_by,
                    reason, final_pdf, final_json, checksum, status.value, prediction_id
                ))
                version_id = cursor.lastrowid

                # 7. Log audit change
                self._log_lifecycle_change(
                    conn=conn,
                    report_id=report_id,
                    action="NEW_VERSION",
                    previous_status=current_status.value,
                    new_status=status.value,
                    actor=created_by or "System"
                )

                return ReportVersion(
                    version_id=version_id,
                    report_id=report_id,
                    version_number=new_version_num,
                    created_at=now_str,
                    created_by=created_by,
                    reason=reason,
                    pdf_path=final_pdf,
                    json_path=final_json,
                    checksum=checksum,
                    status=status,
                    prediction_id=prediction_id
                )
        except ReportServiceException:
            raise
        except Exception as e:
            self.logger.error(f"Failed to create new report version: {e}")
            raise ReportServiceException(f"Failed to create new version: {e}")
        finally:
            conn.close()

    def transition_status(
        self,
        report_id: int,
        target_status: ReportStatus,
        actor: str
    ) -> Report:
        """Transitions the report lifecycle status to a new allowed state.

        Args:
            report_id: Target Report ID.
            target_status: Next desired ReportStatus.
            actor: User ID or email initiating transition.

        Returns:
            The updated Report.
        """
        self.logger.info(f"Transitioning report {report_id} status to {target_status} by {actor}")
        conn = self._get_connection()
        try:
            with conn:
                row = conn.execute("SELECT * FROM reports WHERE report_id = ?;", (report_id,)).fetchone()
                if not row:
                    raise ReportNotFoundException(f"Report with ID {report_id} not found.")

                current_status = ReportStatus(row["status"])
                current_version = row["current_version"]

                # Enforce state machine rules
                if not can_transition(current_status, target_status):
                    raise InvalidTransitionException(
                        f"Invalid status transition from {current_status.value} to {target_status.value}."
                    )

                now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                finalized_at = row["finalized_at"]
                archived_at = row["archived_at"]

                if target_status == ReportStatus.FINAL:
                    finalized_at = now_str
                elif target_status == ReportStatus.ARCHIVED:
                    archived_at = now_str

                # Update reports table
                conn.execute("""
                    UPDATE reports
                    SET status = ?,
                        updated_at = ?,
                        finalized_at = ?,
                        archived_at = ?
                    WHERE report_id = ?;
                """, (target_status.value, now_str, finalized_at, archived_at, report_id))

                # Update report_versions record for current version
                conn.execute("""
                    UPDATE report_versions
                    SET status = ?
                    WHERE report_id = ? AND version_number = ?;
                """, (target_status.value, report_id, current_version))

                # Log audit record
                self._log_lifecycle_change(
                    conn=conn,
                    report_id=report_id,
                    action="TRANSITION_STATUS",
                    previous_status=current_status.value,
                    new_status=target_status.value,
                    actor=actor
                )

                return Report(
                    report_id=report_id,
                    report_number=row["report_number"],
                    patient_id=row["patient_id"],
                    created_by=row["created_by"],
                    report_type=row["report_type"],
                    current_version=current_version,
                    status=target_status,
                    created_at=row["created_at"],
                    updated_at=now_str,
                    finalized_at=finalized_at,
                    archived_at=archived_at,
                    pdf_path=row["pdf_path"],
                    json_path=row["json_path"],
                    checksum=row["checksum"]
                )
        except ReportServiceException:
            raise
        except Exception as e:
            self.logger.error(f"Failed to transition status: {e}")
            raise ReportServiceException(f"Failed to transition status: {e}")
        finally:
            conn.close()

    def get_current_version(self, report_id: int) -> ReportVersion:
        """Retrieves details of the currently active version of a report.

        Args:
            report_id: Target Report ID.

        Returns:
            The current active ReportVersion.
        """
        conn = self._get_connection()
        try:
            row = conn.execute("SELECT current_version FROM reports WHERE report_id = ?;", (report_id,)).fetchone()
            if not row:
                raise ReportNotFoundException(f"Report with ID {report_id} not found.")

            v_row = conn.execute("""
                SELECT * FROM report_versions
                WHERE report_id = ? AND version_number = ?;
            """, (report_id, row["current_version"])).fetchone()

            if not v_row:
                raise VersionNotFoundException(
                    f"Active version {row['current_version']} of report {report_id} not found."
                )

            return ReportVersion(
                version_id=v_row["version_id"],
                report_id=v_row["report_id"],
                version_number=v_row["version_number"],
                created_at=v_row["created_at"],
                created_by=v_row["created_by"],
                reason=v_row["reason"],
                pdf_path=v_row["pdf_path"],
                json_path=v_row["json_path"],
                checksum=v_row["checksum"],
                status=ReportStatus(v_row["status"]),
                prediction_id=v_row["prediction_id"]
            )
        finally:
            conn.close()

"""ReportService application service for clinical report lifecycle and version management."""

import os
import sqlite3
import datetime
import shutil
import logging
import hashlib
from typing import List, Optional, Tuple, Dict, Any

from clinical_reporting.domain.entities import Report, ReportVersion, ReportStatus, can_transition, PatientMismatchException, ComparisonMetric, ReportComparison, FollowUpComparisonMetric, FollowUpComparison, LongitudinalTimelineEvent, LongitudinalPatientTimeline, LongitudinalTimelineMetric, ClinicalDistributionAnalytics, PatientAnalytics, PopulationAnalytics, TimeSeriesPoint, TimeSeriesAnalytics


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


class IntegrityFailureException(ReportServiceException):
    """Exception raised when integrity check fails."""
    pass


class PathTraversalException(ReportServiceException):
    """Exception raised when path traversal is detected."""
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
            return "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"  # SHA-256 of empty string
        sha255 = hashlib.sha256()
        try:
            with open(filepath, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    sha255.update(chunk)
            return sha255.hexdigest()
        except Exception as e:
            self.logger.warning(f"Failed to calculate checksum for {filepath}: {e}")
            return "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"

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

                # F2.2: Generate verification token and calculate canonical integrity hash
                import secrets
                verification_token = secrets.token_urlsafe(32)

                from clinical_reporting.domain.entities import generate_integrity_hash
                integrity_hash = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
                if json_path and os.path.exists(json_path):
                    try:
                        import json
                        with open(json_path, "r", encoding="utf-8") as f:
                            payload = json.load(f)
                        integrity_hash = generate_integrity_hash(payload)
                    except Exception:
                        pass

                # 3. Insert report
                cursor = conn.execute("""
                    INSERT INTO reports (
                        report_number, patient_id, created_by, report_type,
                        current_version, status, created_at, updated_at,
                        pdf_path, json_path, checksum, verification_token, integrity_hash
                    ) VALUES (?, ?, ?, ?, 1, ?, ?, ?, ?, ?, ?, ?, ?);
                """, (
                    report_number, patient_id, created_by, report_type,
                    initial_status.value, now_str, now_str, pdf_path, json_path, checksum,
                    verification_token, integrity_hash
                ))
                report_id = cursor.lastrowid

                # 4. Insert version 1
                conn.execute("""
                    INSERT INTO report_versions (
                        report_id, version_number, created_at, created_by,
                        reason, pdf_path, json_path, checksum, status, prediction_id,
                        verification_token, integrity_hash
                    ) VALUES (?, 1, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """, (
                    report_id, now_str, created_by, "Initial report generation",
                    pdf_path, json_path, checksum, initial_status.value, prediction_id,
                    verification_token, integrity_hash
                ))

                # F2.2: Regenerate PDF to bake in the verification token, integrity hash, version and status
                try:
                    self._regenerate_pdf_internal(conn, report_id, 1)
                    checksum = self._calculate_checksum(pdf_path)
                    conn.execute("UPDATE reports SET checksum = ? WHERE report_id = ?;", (checksum, report_id))
                    conn.execute("UPDATE report_versions SET checksum = ? WHERE report_id = ? AND version_number = 1;", (checksum, report_id))
                except Exception as pdf_err:
                    self.logger.warning(f"Failed to bake verification block into PDF: {pdf_err}")

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

                # F2.2: Generate verification token and calculate canonical integrity hash for new version
                import secrets
                verification_token = secrets.token_urlsafe(32)

                from clinical_reporting.domain.entities import generate_integrity_hash
                integrity_hash = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
                if final_json and os.path.exists(final_json):
                    try:
                        import json
                        with open(final_json, "r", encoding="utf-8") as f:
                            payload = json.load(f)
                        integrity_hash = generate_integrity_hash(payload)
                    except Exception:
                        pass

                # 4. Update reports table
                conn.execute("""
                    UPDATE reports
                    SET current_version = ?,
                        status = ?,
                        updated_at = ?,
                        pdf_path = ?,
                        json_path = ?,
                        checksum = ?,
                        verification_token = ?,
                        integrity_hash = ?
                    WHERE report_id = ?;
                """, (new_version_num, status.value, now_str, final_pdf, final_json, checksum, verification_token, integrity_hash, report_id))

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
                        reason, pdf_path, json_path, checksum, status, prediction_id,
                        verification_token, integrity_hash
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """, (
                    report_id, new_version_num, now_str, created_by,
                    reason, final_pdf, final_json, checksum, status.value, prediction_id,
                    verification_token, integrity_hash
                ))
                version_id = cursor.lastrowid

                # F2.2: Regenerate PDF to bake in the verification token, integrity hash, version and status
                try:
                    self._regenerate_pdf_internal(conn, report_id, new_version_num)
                    checksum = self._calculate_checksum(final_pdf)
                    conn.execute("UPDATE reports SET checksum = ? WHERE report_id = ?;", (checksum, report_id))
                    conn.execute("UPDATE report_versions SET checksum = ? WHERE version_id = ?;", (checksum, version_id))
                except Exception as pdf_err:
                    self.logger.warning(f"Failed to bake verification block into PDF: {pdf_err}")

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

                # F2.2: Regenerate PDF to bake in the updated status
                try:
                    self._regenerate_pdf_internal(conn, report_id, current_version)
                    checksum = self._calculate_checksum(row["pdf_path"])
                    conn.execute("UPDATE reports SET checksum = ? WHERE report_id = ?;", (checksum, report_id))
                    conn.execute("UPDATE report_versions SET checksum = ? WHERE report_id = ? AND version_number = ?;", (checksum, report_id, current_version))
                except Exception as pdf_err:
                    self.logger.warning(f"Failed to regenerate PDF on status transition: {pdf_err}")

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

    def verify_report_by_token(self, token: str) -> Dict[str, Any]:
        """Verifies report authenticity and integrity by token."""
        conn = self._get_connection()
        try:
            # 1. Look up report version by token
            row = conn.execute("""
                SELECT rv.*, r.report_number, r.current_version as latest_version, r.finalized_at as r_finalized_at
                FROM report_versions rv
                JOIN reports r ON rv.report_id = r.report_id
                WHERE rv.verification_token = ?;
            """, (token,)).fetchone()

            if not row:
                token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()[:12]
                self.logger.warning(f"Verification failed: invalid token fingerprint: {token_hash}")
                self._log_verification_failure(conn, token, "INVALID_TOKEN")
                return {
                    "verification_state": "INVALID",
                    "reason": "Token not found"
                }

            report_id = row["report_id"]
            version_number = row["version_number"]
            stored_hash = row["integrity_hash"]
            status = row["status"]
            latest_version = row["latest_version"]
            created_at = row["created_at"]
            finalized_at = row["r_finalized_at"]
            json_path = row["json_path"]
            report_number = row["report_number"]

            # 2. Check if tampered by loading the JSON file and comparing its canonical hash
            tampered = False
            calculated_hash = None
            if not json_path or not os.path.exists(json_path):
                tampered = True
                self.logger.warning(f"Verification: JSON file missing at {json_path}")
            else:
                try:
                    import json
                    with open(json_path, "r", encoding="utf-8") as f:
                        payload = json.load(f)
                    from clinical_reporting.domain.entities import generate_integrity_hash
                    calculated_hash = generate_integrity_hash(payload)

                    # Use constant-time comparison
                    import secrets
                    if not secrets.compare_digest(calculated_hash, stored_hash):
                        tampered = True
                        self.logger.warning(f"Verification: hash mismatch. Stored: {stored_hash}, Calculated: {calculated_hash}")
                except Exception as e:
                    self.logger.error(f"Verification: failed to parse JSON to check integrity: {e}")
                    tampered = True

            # 3. Determine verification state
            if tampered:
                state = "TAMPERED"
                self._log_verification_event(conn, "REPORT_INTEGRITY_FAILED", "FAILED", f"Report {report_number} Version {version_number} hash mismatch")
            elif version_number < latest_version:
                state = "SUPERSEDED"
                self._log_verification_event(conn, "REPORT_VERIFIED", "SUCCESS", f"Report {report_number} Version {version_number} verified (SUPERSEDED)")
            elif status == "ARCHIVED":
                state = "ARCHIVED"
                self._log_verification_event(conn, "REPORT_VERIFIED", "SUCCESS", f"Report {report_number} Version {version_number} verified (ARCHIVED)")
            else:
                state = "VALID"
                self._log_verification_event(conn, "REPORT_VERIFIED", "SUCCESS", f"Report {report_number} Version {version_number} verified (VALID)")

            return {
                "report_id": report_id,
                "report_number": report_number,
                "version": version_number,
                "status": status,
                "verification_state": state,
                "integrity_hash": stored_hash,
                "created_at": created_at,
                "finalized_at": finalized_at
            }
        finally:
            conn.close()

    def _log_verification_event(self, conn: sqlite3.Connection, event_type: str, status: str, details: str) -> None:
        now_str = datetime.datetime.utcnow().isoformat()
        conn.execute("""
            INSERT INTO security_audit_logs (
                timestamp, event_type, user_id, email, ip_address, status, details, user_agent
            ) VALUES (?, ?, NULL, NULL, '127.0.0.1', ?, ?, 'System');
        """, (now_str, event_type, status, details))

    def _log_verification_failure(self, conn: sqlite3.Connection, token: str, reason: str) -> None:
        token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()[:12]
        self._log_verification_event(conn, "REPORT_VERIFICATION_FAILED", "FAILED", f"Token fingerprint: {token_hash}, Reason: {reason}")

    def _regenerate_pdf_internal(self, conn: sqlite3.Connection, report_id: int, version_number: int) -> None:
        """Helper to regenerate PDF for a specific version with integrity details."""
        row = conn.execute("""
            SELECT rv.*, r.report_number
            FROM report_versions rv
            JOIN reports r ON rv.report_id = r.report_id
            WHERE rv.report_id = ? AND rv.version_number = ?;
        """, (report_id, version_number)).fetchone()
        if not row:
            return

        pdf_path = row["pdf_path"]
        json_path = row["json_path"]
        token = row["verification_token"]
        integrity_hash = row["integrity_hash"]
        status = row["status"]
        report_number = row["report_number"]

        if not json_path or not os.path.exists(json_path):
            return

        import json
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Reconstruct entities from JSON
        from clinical_reporting.domain.entities import PatientInfo, ProcessingSummary, ClinicalReport
        from classification.domain.entities import PredictionResult
        from tumor_analysis.domain.entities import TumorAnalysisResult, SeverityLevel
        from severity_assessment.domain.entities import SeverityAssessment, SeverityCategory
        from clinical_insight.domain.entities import ClinicalInsight

        p_data = data.get("patient", {})
        patient = PatientInfo(
            patient_id=p_data.get("patient_id", "N/A"),
            name=p_data.get("name", "N/A"),
            age=p_data.get("age", 45),
            gender=p_data.get("gender", "Female"),
            scan_date=p_data.get("scan_date", "N/A"),
            ref_physician=p_data.get("ref_physician", "N/A")
        )

        proc_data = data.get("processing", {})
        lat_data = proc_data.get("latency_sec", {})
        proc = ProcessingSummary(
            device=proc_data.get("device", "CPU"),
            execution_time_sec=proc_data.get("total_execution_time_sec"),
            classification_model_path=proc_data.get("classification_model", ""),
            segmentation_model_path=proc_data.get("segmentation_model", ""),
            classification_latency_sec=lat_data.get("classification"),
            segmentation_latency_sec=lat_data.get("segmentation"),
            explainability_latency_sec=lat_data.get("explainability")
        )

        cls_data = data.get("classification", {})
        classification = PredictionResult(
            label=0,  # placeholder
            class_name=cls_data.get("predicted_class", "No Tumor"),
            confidence_score=cls_data.get("confidence_score", 0.0),
            probabilities=cls_data.get("probabilities", {}),
            is_calibrated=cls_data.get("is_calibrated", False),
            uncalibrated_confidence_score=cls_data.get("uncalibrated_confidence_score"),
            uncalibrated_probabilities=cls_data.get("uncalibrated_probabilities"),
            calibration_method=cls_data.get("calibration_method"),
            calibration_parameters=cls_data.get("calibration_parameters")
        )

        seg_metrics = None
        seg_data = data.get("segmentation")
        if seg_data:
            try:
                seg_metrics = TumorAnalysisResult(
                    pixel_count=seg_data.get("pixel_count", 0),
                    tumor_area_mm2=seg_data.get("tumor_area_mm2", 0.0),
                    tumor_percentage_brain=seg_data.get("tumor_percentage_brain", 0.0),
                    tumor_percentage_image=seg_data.get("tumor_percentage_image", 0.0),
                    estimated_brain_pixel_count=seg_data.get("estimated_brain_pixel_count", 50000),
                    severity_level=SeverityLevel.LOW,  # placeholder
                    post_processing_applied=seg_data.get("post_processing_applied", False),
                    quality_score=seg_data.get("quality_score"),
                    quality_category=seg_data.get("quality_category"),
                    post_processing_metadata=seg_data.get("post_processing_metadata")
                )
            except Exception:
                pass

        severity = None
        sev_data = data.get("severity") or data.get("classification", {})
        if sev_data and "rule_based_severity" in sev_data:
            try:
                cat_str = sev_data.get("rule_based_severity", "LOW")
                severity = SeverityAssessment(
                    category=SeverityCategory(cat_str.upper()) if cat_str.upper() in ["LOW", "MEDIUM", "HIGH"] else SeverityCategory.LOW,
                    rule_description=sev_data.get("severity_rule_description", "")
                )
            except Exception:
                pass

        insight = None
        ins_data = data.get("clinical_insight") or data.get("insight")
        if ins_data:
            try:
                insight = ClinicalInsight(
                    summary_narrative=ins_data.get("summary_narrative", ""),
                    key_findings=ins_data.get("key_findings", []),
                    recommendations=ins_data.get("recommendations", [])
                )
            except Exception:
                pass

        files_data = data.get("files", {})
        orig_p = files_data.get("original_image")
        heat_p = files_data.get("heatmap_image")
        over_p = files_data.get("overlay_image")
        mask_p = files_data.get("segmentation_mask")
        comp_p = files_data.get("comparison_image")

        report = ClinicalReport(
            patient_info=patient,
            processing_summary=proc,
            classification=classification,
            segmentation_metrics=seg_metrics,
            severity_assessment=severity,
            original_image_path=os.path.abspath(orig_p) if orig_p else None,
            heatmap_image_path=os.path.abspath(heat_p) if heat_p else None,
            overlay_image_path=os.path.abspath(over_p) if over_p else None,
            segmentation_mask_path=os.path.abspath(mask_p) if mask_p else None,
            comparison_image_path=os.path.abspath(comp_p) if comp_p else None,
            clinical_insight=insight,
            report_number=report_number,
            version=version_number,
            status=status,
            verification_token=token,
            integrity_hash=integrity_hash
        )

        from clinical_reporting.infrastructure.pdf_generator import ReportLabPDFGenerator
        pdf_gen = ReportLabPDFGenerator()
        pdf_gen.generate_pdf(report, pdf_path)

    def check_report_access(self, report_id: int, user: Optional[Any]) -> str:
        """Verifies user access permissions for a specific report."""
        conn = self._get_connection()
        try:
            row = conn.execute("SELECT patient_id FROM reports WHERE report_id = ?;", (report_id,)).fetchone()
            if not row:
                return "NOT_FOUND"

            if user is None:
                return "UNAUTHORIZED"

            from security.domain.entities import Role
            role_val = user.role.value if hasattr(user.role, 'value') else str(user.role).lower()

            if role_val in ["admin", "doctor"]:
                return "AUTHORIZED"

            if role_val == "patient":
                # Check patient boundaries using name and patient_id
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT p.patient_id, p.name as patient_name
                    FROM clinical_reports cr
                    JOIN predictions pr ON cr.prediction_id = pr.id
                    JOIN mri_scans s ON pr.scan_id = s.id
                    JOIN patients p ON s.patient_id = p.patient_id
                    WHERE cr.id = ?;
                    """,
                    (report_id,)
                )
                row_pat = cursor.fetchone()
                if not row_pat:
                    r_pat_id = row["patient_id"]
                    cursor.execute("SELECT patient_id, name as patient_name FROM patients WHERE patient_id = ?;", (r_pat_id,))
                    row_pat = cursor.fetchone()

                if not row_pat:
                    return "FORBIDDEN"

                pat_name = row_pat["patient_name"].lower()
                pat_id = row_pat["patient_id"].lower()
                user_name = user.full_name.lower() if user.full_name else ""
                user_uuid = user.uuid.lower() if user.uuid else ""

                if pat_name == user_name or pat_id == user_uuid:
                    return "AUTHORIZED"

                return "FORBIDDEN"

            return "FORBIDDEN"
        finally:
            conn.close()

    def resolve_secure_pdf_path(self, report_id: int, version: Optional[int] = None) -> str:
        """Resolves, checks boundaries and verifies integrity of a report's PDF path."""
        conn = self._get_connection()
        try:
            if version is not None:
                row = conn.execute("""
                    SELECT rv.pdf_path, rv.json_path, rv.integrity_hash, rv.status, r.report_number, rv.version_number
                    FROM report_versions rv
                    JOIN reports r ON rv.report_id = r.report_id
                    WHERE rv.report_id = ? AND rv.version_number = ?;
                """, (report_id, version)).fetchone()
                if not row:
                    raise VersionNotFoundException(f"Version {version} not found for report {report_id}.")
            else:
                row = conn.execute("""
                    SELECT rv.pdf_path, rv.json_path, rv.integrity_hash, rv.status, r.report_number, rv.version_number
                    FROM report_versions rv
                    JOIN reports r ON rv.report_id = r.report_id
                    WHERE rv.report_id = ? AND rv.version_number = r.current_version;
                """, (report_id,)).fetchone()
                if not row:
                    raise ReportNotFoundException(f"Report with ID {report_id} not found.")

            pdf_path = row["pdf_path"]
            json_path = row["json_path"]
            stored_hash = row["integrity_hash"]
            version_number = row["version_number"]

            if not pdf_path:
                raise ReportServiceException("PDF path is null in the database.")

            # Path traversal and directory boundary check
            trusted_dir = os.path.abspath("outputs/clinical_reports")
            resolved_pdf = os.path.abspath(pdf_path)
            if not resolved_pdf.startswith(trusted_dir + os.sep) and resolved_pdf != trusted_dir:
                raise PathTraversalException("Path traversal or directory escape detected.")

            # File existence check
            if not os.path.exists(resolved_pdf):
                # Try regenerating on the fly
                try:
                    self._regenerate_pdf_internal(conn, report_id, version_number)
                except Exception:
                    pass
                if not os.path.exists(resolved_pdf):
                    raise FileNotFoundError(f"PDF file does not exist on disk at {resolved_pdf}.")

            # Integrity check if hash is available
            if stored_hash:
                if not json_path or not os.path.exists(json_path):
                    self._log_security_audit_event(
                        conn=conn,
                        event_type="REPORT_INTEGRITY_FAILED",
                        user=None,
                        report_id=report_id,
                        status="FAILED",
                        details=f"Report ID: {report_id}, Version: {version_number}, reason: JSON file missing"
                    )
                    raise IntegrityFailureException("Integrity check failed: JSON file missing.")
                try:
                    import json
                    with open(json_path, "r", encoding="utf-8") as f:
                        payload = json.load(f)
                    from clinical_reporting.domain.entities import generate_integrity_hash
                    calculated_hash = generate_integrity_hash(payload)
                    import secrets
                    if not secrets.compare_digest(calculated_hash, stored_hash):
                        self._log_security_audit_event(
                            conn=conn,
                            event_type="REPORT_INTEGRITY_FAILED",
                            user=None,
                            report_id=report_id,
                            status="FAILED",
                            details=f"Report ID: {report_id}, Version: {version_number}, reason: hash mismatch"
                        )
                        raise IntegrityFailureException("Integrity check failed: hash mismatch.")
                except Exception as e:
                    if isinstance(e, IntegrityFailureException):
                        raise e
                    self._log_security_audit_event(
                        conn=conn,
                        event_type="REPORT_INTEGRITY_FAILED",
                        user=None,
                        report_id=report_id,
                        status="FAILED",
                        details=f"Report ID: {report_id}, Version: {version_number}, reason: {e}"
                    )
                    raise IntegrityFailureException(f"Integrity check failed: {e}")

            return resolved_pdf
        finally:
            conn.close()

    def _sanitize_json_floats(self, val: Any) -> Any:
        """Recursively sanitizes NaN and Infinity float values to None (null in JSON)."""
        import math
        if isinstance(val, dict):
            return {k: self._sanitize_json_floats(v) for k, v in val.items()}
        elif isinstance(val, list):
            return [self._sanitize_json_floats(x) for x in val]
        elif isinstance(val, float):
            if math.isnan(val) or math.isinf(val):
                return None
            return val
        return val

    def get_report_json_for_export(
        self,
        report_id: int,
        version: Optional[int] = None,
        actor: Optional[Any] = None
    ) -> Dict[str, Any]:
        """Retrieves, checks boundaries, and sanitizes report JSON details for secure export."""
        # 1. Enforce access check
        access_status = self.check_report_access(report_id, actor)
        if access_status == "NOT_FOUND":
            self.log_report_access_event("REPORT_NOT_FOUND", actor, report_id, "FAILED", "Requested nonexistent report JSON.")
            raise ReportNotFoundException(f"Report with ID {report_id} not found.")
        elif access_status == "UNAUTHORIZED":
            raise ReportServiceException("Authentication required.")
        elif access_status == "FORBIDDEN":
            self.log_report_access_event("REPORT_ACCESS_DENIED", actor, report_id, "FAILED", "Access denied to patient report JSON.")
            raise ReportServiceException("Access denied to patient report JSON.")

        # 2. Resolve database connection to find path & check boundary safety
        conn = self._get_connection()
        try:
            if version is not None:
                row = conn.execute("""
                    SELECT rv.json_path, rv.version_number, rv.integrity_hash
                    FROM report_versions rv
                    JOIN reports r ON rv.report_id = r.report_id
                    WHERE rv.report_id = ? AND rv.version_number = ?;
                """, (report_id, version)).fetchone()
                if not row:
                    self._log_security_audit_event(conn, "REPORT_NOT_FOUND", actor, report_id, "FAILED", f"Report version {version} not found.")
                    raise VersionNotFoundException(f"Version {version} not found for report {report_id}.")
            else:
                row = conn.execute("""
                    SELECT rv.json_path, rv.version_number, rv.integrity_hash
                    FROM report_versions rv
                    JOIN reports r ON rv.report_id = r.report_id
                    WHERE rv.report_id = ? AND rv.version_number = r.current_version;
                """, (report_id,)).fetchone()
                if not row:
                    self._log_security_audit_event(conn, "REPORT_NOT_FOUND", actor, report_id, "FAILED", "Latest report version not found.")
                    raise ReportNotFoundException(f"Report with ID {report_id} not found.")

            json_path = row["json_path"]
            version_number = row["version_number"]
            stored_hash = row["integrity_hash"]

            if not json_path:
                raise FileNotFoundError("JSON path is empty in database.")

            # Path traversal and boundary security check
            trusted_dir = os.path.abspath("outputs/clinical_reports")
            resolved_json = os.path.abspath(json_path)
            if not resolved_json.startswith(trusted_dir + os.sep) and resolved_json != trusted_dir:
                self._log_security_audit_event(conn, "REPORT_ACCESS_DENIED", actor, report_id, "FAILED", f"Path traversal attempt: {json_path}")
                raise PathTraversalException("Path traversal or directory escape detected.")

            # Missing-file check
            if not os.path.exists(resolved_json):
                # Try regenerating on the fly
                try:
                    self._regenerate_pdf_internal(conn, report_id, version_number)
                except Exception:
                    pass
                if not os.path.exists(resolved_json):
                    raise FileNotFoundError(f"JSON report file not found on server disk at {resolved_json}")

            # Load the JSON
            try:
                import json
                with open(resolved_json, "r", encoding="utf-8") as f:
                    payload = json.load(f)
            except Exception as e:
                raise ValueError(f"Malformed or unreadable JSON file: {e}")

            # Verify integrity if stored_hash is set
            if stored_hash:
                from clinical_reporting.domain.entities import generate_integrity_hash
                calculated_hash = generate_integrity_hash(payload)
                import secrets
                if not secrets.compare_digest(calculated_hash, stored_hash):
                    self._log_security_audit_event(conn, "REPORT_INTEGRITY_FAILED", actor, report_id, "FAILED", f"Report ID: {report_id}, Version: {version_number}, reason: hash mismatch")
                    raise IntegrityFailureException("Report integrity verification failed.")

            # Sanitize NaN/Infinity
            sanitized = self._sanitize_json_floats(payload)

            # Audit success log
            self._log_security_audit_event(conn, "REPORT_JSON_EXPORTED", actor, report_id, "SUCCESS", f"Exported report JSON version {version_number}")
            conn.commit()
            return sanitized
        finally:
            conn.close()

    def get_report_csv_for_export(
        self,
        report_id: int,
        version: Optional[int] = None,
        actor: Optional[Any] = None
    ) -> str:
        """Retrieves and compiles a report's key details and clinical metrics as a CSV string."""
        import csv
        import io
        import math

        # 1. Reuse existing JSON export (which handles RBAC, path traversal, missing files, integrity, sanitization)
        sanitized_json = self.get_report_json_for_export(report_id, version, actor)

        # 2. Extract nested fields to flat representation
        patient_data = sanitized_json.get("patient", {})
        cls_data = sanitized_json.get("classification", {})
        seg_data = sanitized_json.get("segmentation", {})
        sev_data = sanitized_json.get("severity", {})

        # Get metadata from DB
        resolved_version = version
        report_number = ""
        status = ""
        integrity_hash = ""
        created_at = ""
        conn = self._get_connection()
        try:
            if version is not None:
                row = conn.execute("""
                    SELECT r.report_number, rv.version_number, rv.status, rv.integrity_hash, rv.created_at
                    FROM report_versions rv
                    JOIN reports r ON rv.report_id = r.report_id
                    WHERE rv.report_id = ? AND rv.version_number = ?;
                """, (report_id, version)).fetchone()
            else:
                row = conn.execute("""
                    SELECT r.report_number, rv.version_number, rv.status, rv.integrity_hash, rv.created_at
                    FROM report_versions rv
                    JOIN reports r ON rv.report_id = r.report_id
                    WHERE rv.report_id = ? AND rv.version_number = r.current_version;
                """, (report_id,)).fetchone()
            if row:
                report_number = row["report_number"]
                resolved_version = row["version_number"]
                status = row["status"]
                integrity_hash = row["integrity_hash"]
                created_at = row["created_at"]
        finally:
            conn.close()

        # Build tabular data row
        row_data = {
            "Report Number": report_number or "",
            "Version": str(resolved_version) if resolved_version is not None else "",
            "Status": status or "",
            "Created At": created_at or "",
            "Patient ID": patient_data.get("patient_id", ""),
            "Patient Name": patient_data.get("name", ""),
            "Patient Age": str(patient_data.get("age", "")) if patient_data.get("age") is not None else "",
            "Patient Gender": patient_data.get("gender", ""),
            "Scan Date": patient_data.get("scan_date", ""),
            "AI Classification": cls_data.get("predicted_class", ""),
            "Confidence Score": cls_data.get("confidence_score"),
            "Severity": sev_data.get("category", ""),
            "Tumor Area (mm2)": seg_data.get("tumor_area_mm2") if seg_data else "",
            "Tumor Percentage Brain (%)": seg_data.get("tumor_percentage_brain") if seg_data else "",
            "Integrity Hash": integrity_hash or "",
        }

        # Serialize fields safely:
        # Convert float, None to correct string.
        # Prevent CSV injection by escaping any value starting with: =, +, -, @
        # Also serialize NaN/Infinity/None/missing values as empty string
        def sanitize_val(val: Any) -> str:
            if val is None:
                return ""
            if isinstance(val, float):
                if math.isnan(val) or math.isinf(val):
                    return ""
                return f"{val:.6f}".rstrip('0').rstrip('.')
            val_str = str(val)
            # Prevent CSV Formula Injection
            if val_str and val_str[0] in ('=', '+', '-', '@'):
                return "'" + val_str
            return val_str

        headers = list(row_data.keys())
        csv_row = [sanitize_val(row_data[h]) for h in headers]

        dest = io.StringIO()
        writer = csv.writer(dest, lineterminator='\r\n')
        writer.writerow(headers)
        writer.writerow(csv_row)

        # Log successful audit event
        conn = self._get_connection()
        try:
            self._log_security_audit_event(
                conn,
                "REPORT_CSV_EXPORTED",
                actor,
                report_id,
                "SUCCESS",
                f"Exported report CSV version {resolved_version}"
            )
            conn.commit()
        finally:
            conn.close()

        return dest.getvalue()

    def get_report_audit_history(self, user: Any) -> List[Dict[str, Any]]:
        """Retrieves and filters access history logs according to RBAC."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT timestamp, event_type, user_id, email, status, details
                FROM security_audit_logs
                WHERE event_type IN (
                    'REPORT_VIEWED', 'REPORT_DOWNLOADED',
                    'REPORT_ACCESS_DENIED', 'REPORT_NOT_FOUND',
                    'REPORT_INTEGRITY_FAILED', 'REPORT_LIFECYCLE_CHANGE',
                    'REPORT_JSON_EXPORTED', 'REPORT_CSV_EXPORTED'
                )
                ORDER BY id DESC;
            """)
            rows = cursor.fetchall()
            logs = [dict(r) for r in rows]

            role_val = user.role.value if hasattr(user.role, 'value') else str(user.role).lower()
            if role_val in ["admin", "doctor"]:
                return logs

            if role_val == "patient":
                cursor.execute(
                    """
                    SELECT cr.id
                    FROM clinical_reports cr
                    JOIN predictions pr ON cr.prediction_id = pr.id
                    JOIN mri_scans s ON pr.scan_id = s.id
                    JOIN patients p ON s.patient_id = p.patient_id
                    WHERE p.name = ? OR p.patient_id = ?;
                    """,
                    (user.full_name, user.uuid)
                )
                p_rows = cursor.fetchall()
                permitted_ids = {r["id"] for r in p_rows}

                cursor.execute("SELECT report_id FROM reports WHERE patient_id = ?;", (user.uuid,))
                for r in cursor.fetchall():
                    permitted_ids.add(r["report_id"])

                user_email = None
                user_id_val = None
                if user:
                    if isinstance(user, dict):
                        user_email = user.get("email")
                        user_id_val = user.get("id")
                    else:
                        user_email = getattr(user, "email", None)
                        user_id_val = getattr(user, "id", None)

                import re
                filtered = []
                for log in logs:
                    # Let patient see their own actions
                    if user_email and log.get("email") == user_email:
                        filtered.append(log)
                        continue
                    if user_id_val is not None and log.get("user_id") == user_id_val:
                        filtered.append(log)
                        continue

                    details = log.get("details", "")
                    m = re.search(r"Report ID:\s*(\d+)", details)
                    if m:
                        rep_id = int(m.group(1))
                        if rep_id in permitted_ids:
                            filtered.append(log)
                return filtered

            return []
        finally:
            conn.close()

    def log_report_access_event(
        self,
        event_type: str,
        user: Optional[Any],
        report_id: Optional[int],
        status: str,
        details: str
    ) -> None:
        """Public wrapper to log access events."""
        conn = self._get_connection()
        try:
            self._log_security_audit_event(conn, event_type, user, report_id, status, details)
            conn.commit()
        finally:
            conn.close()

    def _log_security_audit_event(
        self,
        conn: sqlite3.Connection,
        event_type: str,
        user: Optional[Any],
        report_id: Optional[int],
        status: str,
        details: str
    ) -> None:
        """Helper to insert structured logs into security_audit_logs."""
        now_str = datetime.datetime.utcnow().isoformat()
        user_id = None
        email = None
        if user:
            if isinstance(user, dict):
                user_id = user.get("id")
                email = user.get("email")
            else:
                user_id = getattr(user, "id", None)
                email = getattr(user, "email", None)

        if report_id is not None and "Report ID:" not in details:
            details = f"Report ID: {report_id}, {details}"

        conn.execute("""
            INSERT INTO security_audit_logs (
                timestamp, event_type, user_id, email, ip_address, status, details, user_agent
            ) VALUES (?, ?, ?, ?, '127.0.0.1', ?, ?, 'System');
        """, (now_str, event_type, user_id, email, status, details))

    def compare_reports(
        self,
        previous_report_id: int,
        current_report_id: int,
        previous_version: Optional[int] = None,
        current_version: Optional[int] = None,
        actor: Optional[Any] = None
    ) -> Dict[str, Any]:
        """Compares two report versions dynamically from persisted JSON and DB metrics."""
        self.logger.info(
            f"Comparing reports: Prev ID={previous_report_id} (v={previous_version}) "
            f"vs Curr ID={current_report_id} (v={current_version}) by actor={actor}"
        )

        # 1. Enforce RBAC access check on both reports
        status_prev = self.check_report_access(previous_report_id, actor)
        if status_prev == "NOT_FOUND":
            self.log_report_access_event("REPORT_NOT_FOUND", actor, previous_report_id, "FAILED", "Requested nonexistent previous report for comparison.")
            raise ReportNotFoundException(f"Previous report with ID {previous_report_id} not found.")
        elif status_prev == "UNAUTHORIZED":
            raise ReportServiceException("Authentication required.")
        elif status_prev == "FORBIDDEN":
            self.log_report_access_event("REPORT_ACCESS_DENIED", actor, previous_report_id, "FAILED", "Access denied to previous report for comparison.")
            raise ReportServiceException("Access denied to previous report.")

        status_curr = self.check_report_access(current_report_id, actor)
        if status_curr == "NOT_FOUND":
            self.log_report_access_event("REPORT_NOT_FOUND", actor, current_report_id, "FAILED", "Requested nonexistent current report for comparison.")
            raise ReportNotFoundException(f"Current report with ID {current_report_id} not found.")
        elif status_curr == "UNAUTHORIZED":
            raise ReportServiceException("Authentication required.")
        elif status_curr == "FORBIDDEN":
            self.log_report_access_event("REPORT_ACCESS_DENIED", actor, current_report_id, "FAILED", "Access denied to current report for comparison.")
            raise ReportServiceException("Access denied to current report.")

        # 2. Fetch both report versions details
        conn = self._get_connection()
        try:
            prev_data = self._fetch_report_version_data(conn, previous_report_id, previous_version)
            curr_data = self._fetch_report_version_data(conn, current_report_id, current_version)
        finally:
            conn.close()

        # 3. Validate same patient
        if str(prev_data["patient_id"]).lower() != str(curr_data["patient_id"]).lower():
            raise PatientMismatchException("Cannot compare reports belonging to different patients.")

        # 4. Calculate comparisons
        metrics = []

        # A. Classification
        prev_cls = prev_data["classification"]
        curr_cls = curr_data["classification"]
        cls_dir = "UNCHANGED" if prev_cls == curr_cls else "CHANGED"
        metrics.append(ComparisonMetric(
            name="classification",
            previous_value=prev_cls,
            current_value=curr_cls,
            direction=cls_dir,
            interpretation_category=cls_dir
        ))

        # B. Classification confidence
        prev_conf = prev_data["confidence"]
        curr_conf = curr_data["confidence"]
        metrics.append(self._compare_numeric_metric("confidence", prev_conf, curr_conf, tolerance=1e-4))

        # C. Tumor area
        prev_area = prev_data["tumor_area_mm2"]
        curr_area = curr_data["tumor_area_mm2"]
        metrics.append(self._compare_numeric_metric("tumor_area_mm2", prev_area, curr_area, tolerance=1e-4))

        # D. Tumor percentage brain (occupancy)
        prev_pct = prev_data["tumor_percentage_brain"]
        curr_pct = curr_data["tumor_percentage_brain"]
        metrics.append(self._compare_numeric_metric("tumor_percentage_brain", prev_pct, curr_pct, tolerance=1e-4))

        # E. Severity
        prev_sev = prev_data["severity"].upper()
        curr_sev = curr_data["severity"].upper()
        metrics.append(self._compare_severity_metric(prev_sev, curr_sev))

        # F. Morphological metrics
        morphology_keys = [
            ("major_axis_mm", "major_axis"),
            ("minor_axis_mm", "minor_axis"),
            ("eccentricity", "eccentricity"),
            ("orientation_deg", "orientation"),
            ("perimeter_mm", "perimeter"),
            ("solidity", "compactness"),
            ("circularity", "circularity")
        ]
        for key_json, key_metric in morphology_keys:
            prev_m = prev_data["morphology"].get(key_json)
            curr_m = curr_data["morphology"].get(key_json)
            if prev_m is not None and curr_m is not None:
                metrics.append(self._compare_numeric_metric(key_metric, prev_m, curr_m, tolerance=1e-4))

        # G. Clinical findings
        prev_findings = prev_data["clinical_findings"]
        curr_findings = curr_data["clinical_findings"]
        findings_dir = "TEXT_UNCHANGED" if prev_findings == curr_findings else "TEXT_CHANGED"
        metrics.append(ComparisonMetric(
            name="clinical_findings",
            previous_value=prev_findings,
            current_value=curr_findings,
            direction=findings_dir,
            interpretation_category=findings_dir
        ))

        # 5. Determine Overall Summary Status & Text
        summary_status, summary_text = self._determine_overall_summary(
            prev_cls, curr_cls,
            prev_area, curr_area,
            prev_sev, curr_sev
        )

        import secrets
        comparison_id = f"CMP-{secrets.token_hex(8).upper()}"
        created_at = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        created_by_str = actor.email if (actor and hasattr(actor, "email")) else (str(actor) if actor else "System")

        disclaimer = (
            "This comparison summarizes differences between stored AI-derived report measurements. "
            "It is not a medical diagnosis and should be interpreted by a qualified healthcare professional."
        )

        comparison = ReportComparison(
            comparison_id=comparison_id,
            patient_id=prev_data["patient_id"],
            previous_report_id=previous_report_id,
            current_report_id=current_report_id,
            previous_version=prev_data["version_number"],
            current_version=curr_data["version_number"],
            created_at=created_at,
            created_by=created_by_str,
            metrics=metrics,
            summary_status=summary_status,
            summary_text=summary_text,
            disclaimer=disclaimer
        )

        # Log comparison event in security audit log
        conn = self._get_connection()
        try:
            self._log_security_audit_event(
                conn,
                "REPORT_COMPARED",
                actor,
                previous_report_id,
                "SUCCESS",
                f"Compared with current report ID: {current_report_id}. Summary status: {summary_status}"
            )
            conn.commit()
        except Exception as audit_err:
            self.logger.warning(f"Failed to log comparison security event: {audit_err}")
        finally:
            conn.close()

        # Convert back to dict
        res_dict = {
            "comparison_id": comparison.comparison_id,
            "patient_id": comparison.patient_id,
            "previous_report": {
                "report_id": comparison.previous_report_id,
                "version": comparison.previous_version
            },
            "current_report": {
                "report_id": comparison.current_report_id,
                "version": comparison.current_version
            },
            "metrics": [
                {
                    "name": m.name,
                    "previous_value": m.previous_value,
                    "current_value": m.current_value,
                    "absolute_difference": m.absolute_difference,
                    "percentage_difference": m.percentage_difference,
                    "direction": m.direction,
                    "interpretation_category": m.interpretation_category
                } for m in comparison.metrics
            ],
            "summary": {
                "status": comparison.summary_status,
                "text": comparison.summary_text
            },
            "disclaimer": comparison.disclaimer
        }

        import math

        def sanitize_non_finite_values(val: Any) -> Any:
            if isinstance(val, dict):
                return {k: sanitize_non_finite_values(v) for k, v in val.items()}
            elif isinstance(val, list):
                return [sanitize_non_finite_values(v) for v in val]
            elif isinstance(val, float):
                if math.isnan(val) or math.isinf(val):
                    return None
                return val
            return val

        return sanitize_non_finite_values(res_dict)

    FOLLOWUP_TOLERANCES = {
        "confidence": 0.01,
        "tumor_area_mm2": 0.1,
        "tumor_percentage_brain": 0.0001,
        "perimeter": 0.1,
        "bbox_w_mm": 0.1,
        "bbox_h_mm": 0.1,
        "solidity": 0.01,
        "circularity": 0.01,
        "eccentricity": 0.01,
        "orientation": 1.0
    }

    def compare_followup_reports(
        self,
        previous_report_id: int,
        current_report_id: int,
        previous_version: Optional[int] = None,
        current_version: Optional[int] = None,
        actor: Optional[Any] = None,
        tolerances: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """Compares two report versions longitudinally, quantifying changes for follow-up evaluation."""
        self.logger.info(
            f"Comparing follow-up reports: Prev ID={previous_report_id} (v={previous_version}) "
            f"vs Curr ID={current_report_id} (v={current_version}) by actor={actor}"
        )

        # 1. Enforce RBAC access check on both reports
        status_prev = self.check_report_access(previous_report_id, actor)
        if status_prev == "NOT_FOUND":
            self.log_report_access_event("REPORT_NOT_FOUND", actor, previous_report_id, "FAILED", "Requested nonexistent previous report for follow-up comparison.")
            raise ReportNotFoundException(f"Previous report with ID {previous_report_id} not found.")
        elif status_prev == "UNAUTHORIZED":
            raise ReportServiceException("Authentication required.")
        elif status_prev == "FORBIDDEN":
            self.log_report_access_event("REPORT_ACCESS_DENIED", actor, previous_report_id, "FAILED", "Access denied to previous report for follow-up comparison.")
            raise ReportServiceException("Access denied to previous report.")

        status_curr = self.check_report_access(current_report_id, actor)
        if status_curr == "NOT_FOUND":
            self.log_report_access_event("REPORT_NOT_FOUND", actor, current_report_id, "FAILED", "Requested nonexistent current report for follow-up comparison.")
            raise ReportNotFoundException(f"Current report with ID {current_report_id} not found.")
        elif status_curr == "UNAUTHORIZED":
            raise ReportServiceException("Authentication required.")
        elif status_curr == "FORBIDDEN":
            self.log_report_access_event("REPORT_ACCESS_DENIED", actor, current_report_id, "FAILED", "Access denied to current report for follow-up comparison.")
            raise ReportServiceException("Access denied to current report.")

        # 2. Fetch both report versions details
        conn = self._get_connection()
        try:
            prev_data = self._fetch_report_version_data(conn, previous_report_id, previous_version)
            curr_data = self._fetch_report_version_data(conn, current_report_id, current_version)
        finally:
            conn.close()

        # 3. Validate same patient
        if str(prev_data["patient_id"]).lower() != str(curr_data["patient_id"]).lower():
            raise PatientMismatchException("Cannot compare reports belonging to different patients.")

        # Resolve tolerances
        tols = dict(self.FOLLOWUP_TOLERANCES)
        if tolerances:
            tols.update(tolerances)

        import math

        def safe_float(val: Any) -> Optional[float]:
            if val is None:
                return None
            try:
                f = float(val)
                if not math.isfinite(f):
                    return None
                return f
            except (ValueError, TypeError):
                return None

        def compare_numeric(name: str, prev_val: Any, curr_val: Any) -> FollowUpComparisonMetric:
            p_f = safe_float(prev_val)
            c_f = safe_float(curr_val)
            if p_f is None or c_f is None:
                return FollowUpComparisonMetric(
                    name=name,
                    previous_value=p_f,
                    current_value=c_f,
                    absolute_change=None,
                    percentage_change=None,
                    status="UNAVAILABLE"
                )
            tol = tols.get(name, 1e-4)
            abs_change = c_f - p_f
            if abs(abs_change) <= tol:
                status = "STABLE"
            elif abs_change > 0:
                status = "INCREASED"
            else:
                status = "DECREASED"

            if p_f == 0.0:
                pct_change = None
            else:
                pct_change = (abs_change / abs(p_f)) * 100.0

            return FollowUpComparisonMetric(
                name=name,
                previous_value=p_f,
                current_value=c_f,
                absolute_change=round(abs_change, 5),
                percentage_change=round(pct_change, 4) if pct_change is not None else None,
                status=status
            )

        metrics = []

        # Classification
        prev_cls = prev_data.get("classification")
        curr_cls = curr_data.get("classification")
        if prev_cls is None or curr_cls is None:
            cls_status = "UNAVAILABLE"
        elif prev_cls == curr_cls:
            cls_status = "STABLE"
        elif prev_cls == "No Tumor":
            cls_status = "INCREASED"
        elif curr_cls == "No Tumor":
            cls_status = "DECREASED"
        else:
            cls_status = "INCREASED"

        metrics.append(FollowUpComparisonMetric(
            name="classification",
            previous_value=prev_cls,
            current_value=curr_cls,
            absolute_change=None,
            percentage_change=None,
            status=cls_status
        ))

        # Confidence
        metrics.append(compare_numeric("confidence", prev_data.get("confidence"), curr_data.get("confidence")))

        # Tumor area
        area_metric = compare_numeric("tumor_area_mm2", prev_data.get("tumor_area_mm2"), curr_data.get("tumor_area_mm2"))
        metrics.append(area_metric)

        # Brain occupancy percentage
        metrics.append(compare_numeric("tumor_percentage_brain", prev_data.get("tumor_percentage_brain"), curr_data.get("tumor_percentage_brain")))

        # Severity
        prev_sev = str(prev_data.get("severity", "LOW")).upper()
        curr_sev = str(curr_data.get("severity", "LOW")).upper()
        severity_order = {"NORMAL": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4, "NORMAL/NO TUMOR": 0, "LOW SEVERITY": 1, "MEDIUM SEVERITY": 2, "HIGH SEVERITY": 3, "CRITICAL SEVERITY": 4}
        prev_idx = severity_order.get(prev_sev, -1)
        curr_idx = severity_order.get(curr_sev, -1)
        if prev_idx != -1 and curr_idx != -1:
            if curr_idx > prev_idx:
                sev_status = "INCREASED"
            elif curr_idx < prev_idx:
                sev_status = "DECREASED"
            else:
                sev_status = "STABLE"
        else:
            sev_status = "UNAVAILABLE"

        metrics.append(FollowUpComparisonMetric(
            name="severity",
            previous_value=prev_sev,
            current_value=curr_sev,
            absolute_change=None,
            percentage_change=None,
            status=sev_status
        ))

        # Perimeter
        prev_p = prev_data.get("morphology", {}).get("perimeter_mm")
        if prev_p is None:
            prev_p = prev_data.get("morphology", {}).get("perimeter")
        curr_p = curr_data.get("morphology", {}).get("perimeter_mm")
        if curr_p is None:
            curr_p = curr_data.get("morphology", {}).get("perimeter")
        metrics.append(compare_numeric("perimeter", prev_p, curr_p))

        # Bounding box dimensions
        prev_bw = prev_data.get("morphology", {}).get("bbox_w_mm")
        curr_bw = curr_data.get("morphology", {}).get("bbox_w_mm")
        metrics.append(compare_numeric("bbox_w_mm", prev_bw, curr_bw))

        prev_bh = prev_data.get("morphology", {}).get("bbox_h_mm")
        curr_bh = curr_data.get("morphology", {}).get("bbox_h_mm")
        metrics.append(compare_numeric("bbox_h_mm", prev_bh, curr_bh))

        # Solidity
        prev_sol = prev_data.get("morphology", {}).get("solidity")
        if prev_sol is None:
            prev_sol = prev_data.get("morphology", {}).get("compactness")
        curr_sol = curr_data.get("morphology", {}).get("solidity")
        if curr_sol is None:
            curr_sol = curr_data.get("morphology", {}).get("compactness")
        metrics.append(compare_numeric("solidity", prev_sol, curr_sol))

        # Circularity
        prev_circ = prev_data.get("morphology", {}).get("circularity")
        curr_circ = curr_data.get("morphology", {}).get("circularity")
        metrics.append(compare_numeric("circularity", prev_circ, curr_circ))

        # Eccentricity
        prev_ecc = prev_data.get("morphology", {}).get("eccentricity")
        curr_ecc = curr_data.get("morphology", {}).get("eccentricity")
        metrics.append(compare_numeric("eccentricity", prev_ecc, curr_ecc))

        # Orientation
        prev_ori = prev_data.get("morphology", {}).get("orientation_deg")
        if prev_ori is None:
            prev_ori = prev_data.get("morphology", {}).get("orientation")
        curr_ori = curr_data.get("morphology", {}).get("orientation_deg")
        if curr_ori is None:
            curr_ori = curr_data.get("morphology", {}).get("orientation")
        metrics.append(compare_numeric("orientation", prev_ori, curr_ori))

        # Clinical findings
        prev_find = prev_data.get("clinical_findings")
        curr_find = curr_data.get("clinical_findings")
        find_status = "UNAVAILABLE"
        if prev_find is not None and curr_find is not None:
            find_status = "STABLE" if prev_find == curr_find else "STABLE"

        metrics.append(FollowUpComparisonMetric(
            name="clinical_findings",
            previous_value=prev_find,
            current_value=curr_find,
            absolute_change=None,
            percentage_change=None,
            status=find_status
        ))

        # 4. Generate deterministic comparison summary based ONLY on actual measured values
        if area_metric.status == "INCREASED":
            if area_metric.percentage_change is not None:
                summary_text = f"Tumor area increased by {area_metric.percentage_change:.1f}%"
            else:
                summary_text = f"Tumor area increased by {area_metric.absolute_change:.2f} mm²"
            summary_status = "MEASUREMENTS_INCREASED"
        elif area_metric.status == "DECREASED":
            if area_metric.percentage_change is not None:
                summary_text = f"Tumor area decreased by {abs(area_metric.percentage_change):.1f}%"
            else:
                summary_text = f"Tumor area decreased by {abs(area_metric.absolute_change):.2f} mm²"
            summary_status = "MEASUREMENTS_DECREASED"
        elif area_metric.status == "STABLE":
            summary_text = "Tumor area remained stable"
            summary_status = "STABLE"
        else:
            summary_text = "Comparison unavailable because previous measurement is missing"
            summary_status = "UNAVAILABLE"

        import secrets
        comparison_id = f"CMP-{secrets.token_hex(8).upper()}"
        created_at = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        created_by_str = actor.email if (actor and hasattr(actor, "email")) else (str(actor) if actor else "System")

        disclaimer = (
            "This comparison summarizes differences between stored AI-derived report measurements. "
            "It is not a medical diagnosis and should be interpreted by a qualified healthcare professional."
        )

        comparison = FollowUpComparison(
            comparison_id=comparison_id,
            patient_id=prev_data["patient_id"],
            previous_report_id=previous_report_id,
            current_report_id=current_report_id,
            previous_version=prev_data["version_number"],
            current_version=curr_data["version_number"],
            previous_scan_date=prev_data.get("scan_date"),
            current_scan_date=curr_data.get("scan_date"),
            created_at=created_at,
            created_by=created_by_str,
            metrics=metrics,
            summary_status=summary_status,
            summary_text=summary_text,
            disclaimer=disclaimer
        )

        # Log comparison event in security audit log
        conn = self._get_connection()
        try:
            self._log_security_audit_event(
                conn,
                "REPORT_COMPARED",
                actor,
                previous_report_id,
                "SUCCESS",
                f"Compared follow-up with current report ID: {current_report_id}. Summary status: {summary_status}"
            )
            conn.commit()
        except Exception as audit_err:
            self.logger.warning(f"Failed to log follow-up comparison security event: {audit_err}")
        finally:
            conn.close()

        # Convert back to dict
        res_dict = {
            "comparison_id": comparison.comparison_id,
            "patient_id": comparison.patient_id,
            "previous_report": {
                "report_id": comparison.previous_report_id,
                "version": comparison.previous_version,
                "scan_date": comparison.previous_scan_date
            },
            "current_report": {
                "report_id": comparison.current_report_id,
                "version": comparison.current_version,
                "scan_date": comparison.current_scan_date
            },
            "metrics": [
                {
                    "name": m.name,
                    "previous_value": m.previous_value,
                    "current_value": m.current_value,
                    "absolute_change": m.absolute_change,
                    "percentage_change": m.percentage_change,
                    "status": m.status
                } for m in comparison.metrics
            ],
            "summary": {
                "status": comparison.summary_status,
                "text": comparison.summary_text
            },
            "disclaimer": comparison.disclaimer
        }

        def sanitize_non_finite_values(val: Any) -> Any:
            if isinstance(val, dict):
                return {k: sanitize_non_finite_values(v) for k, v in val.items()}
            elif isinstance(val, list):
                return [sanitize_non_finite_values(v) for v in val]
            elif isinstance(val, float):
                if math.isnan(val) or math.isinf(val):
                    return None
                return val
            return val

        return sanitize_non_finite_values(res_dict)

    def _fetch_report_version_data(self, conn: sqlite3.Connection, report_id: int, version: Optional[int]) -> Dict[str, Any]:
        """Helper to fetch a specific or latest version and prediction parameters, falling back to JSON file on disk."""
        if version is not None:
            query = """
                SELECT rv.version_number, rv.json_path, rv.prediction_id, r.patient_id, r.report_number,
                       p.predicted_class, p.confidence_score, p.tumor_area_mm2, p.tumor_percentage_brain,
                       p.rule_based_severity, p.severity_rule_description,
                       ms.scan_date
                FROM report_versions rv
                JOIN reports r ON rv.report_id = r.report_id
                LEFT JOIN predictions p ON rv.prediction_id = p.id
                LEFT JOIN mri_scans ms ON p.scan_id = ms.id
                WHERE rv.report_id = ? AND rv.version_number = ?;
            """
            row = conn.execute(query, (report_id, version)).fetchone()
            if not row:
                raise VersionNotFoundException(f"Version {version} not found for report {report_id}.")
        else:
            query = """
                SELECT rv.version_number, rv.json_path, rv.prediction_id, r.patient_id, r.report_number,
                       p.predicted_class, p.confidence_score, p.tumor_area_mm2, p.tumor_percentage_brain,
                       p.rule_based_severity, p.severity_rule_description,
                       ms.scan_date
                FROM report_versions rv
                JOIN reports r ON rv.report_id = r.report_id
                LEFT JOIN predictions p ON rv.prediction_id = p.id
                LEFT JOIN mri_scans ms ON p.scan_id = ms.id
                WHERE rv.report_id = ? AND rv.version_number = r.current_version;
            """
            row = conn.execute(query, (report_id,)).fetchone()
            if not row:
                raise ReportNotFoundException(f"Report with ID {report_id} not found.")

        # Default fallback values from DB row
        res = {
            "version_number": row["version_number"],
            "patient_id": row["patient_id"],
            "report_number": row["report_number"],
            "scan_date": row["scan_date"],
            "classification": row["predicted_class"] or "No Tumor",
            "confidence": row["confidence_score"] or 0.0,
            "tumor_area_mm2": row["tumor_area_mm2"] or 0.0,
            "tumor_percentage_brain": row["tumor_percentage_brain"] or 0.0,
            "severity": row["rule_based_severity"] or "LOW",
            "morphology": {},
            "clinical_findings": row["severity_rule_description"] or ""
        }

        # Try to load and enrich from JSON file
        json_path = row["json_path"]
        if json_path and os.path.exists(json_path):
            try:
                import json
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                # Enrich patient
                if "patient" in data:
                    res["patient_id"] = data["patient"].get("patient_id", res["patient_id"])
                    res["scan_date"] = data["patient"].get("scan_date", res["scan_date"])

                # Enrich classification
                if "classification" in data:
                    c = data["classification"]
                    res["classification"] = c.get("predicted_class", res["classification"])
                    res["confidence"] = c.get("confidence_score", res["confidence"])

                # Enrich segmentation & morphology
                if "segmentation" in data:
                    s = data["segmentation"]
                    res["tumor_area_mm2"] = s.get("tumor_area_mm2", res["tumor_area_mm2"])
                    res["tumor_percentage_brain"] = s.get("tumor_percentage_brain", res["tumor_percentage_brain"])
                    if "shape_statistics" in s:
                        res["morphology"] = s["shape_statistics"]

                # Enrich severity
                if "severity" in data:
                    sev = data["severity"]
                    res["severity"] = sev.get("category", res["severity"])

                # Enrich clinical findings
                if "clinical_insight" in data:
                    ci = data["clinical_insight"]
                    # If key findings are list, join them or use narrative
                    findings_list = ci.get("key_findings", [])
                    if findings_list:
                        res["clinical_findings"] = "; ".join(findings_list)
                    else:
                        res["clinical_findings"] = ci.get("summary_narrative", res["clinical_findings"])
            except Exception as e:
                self.logger.warning(f"Could not load/parse JSON report file at {json_path}: {e}")

        return res

    def _compare_numeric_metric(self, name: str, prev_val: float, curr_val: float, tolerance: float = 1e-4) -> ComparisonMetric:
        """Helper to calculate difference of numerical metrics with float safety tolerance."""
        abs_diff = curr_val - prev_val
        if abs(abs_diff) <= tolerance:
            abs_diff = 0.0
            pct_diff = 0.0
            direction = "UNCHANGED"
            interpretation = "UNCHANGED"
        else:
            if prev_val != 0.0:
                pct_diff = (abs_diff / abs(prev_val)) * 100.0
            else:
                pct_diff = None

            if abs_diff > 0:
                direction = "INCREASED"
                interpretation = "OBSERVED_INCREASE"
            else:
                direction = "DECREASED"
                interpretation = "OBSERVED_DECREASE"

        return ComparisonMetric(
            name=name,
            previous_value=prev_val,
            current_value=curr_val,
            absolute_difference=round(abs_diff, 5) if abs_diff is not None else None,
            percentage_difference=round(pct_diff, 4) if pct_diff is not None else None,
            direction=direction,
            interpretation_category=interpretation
        )

    def _compare_severity_metric(self, prev_sev: str, curr_sev: str) -> ComparisonMetric:
        """Helper to compare ordinal severity levels (LOW, MEDIUM, HIGH)."""
        severity_order = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
        prev_idx = severity_order.get(prev_sev, -1)
        curr_idx = severity_order.get(curr_sev, -1)

        if prev_idx != -1 and curr_idx != -1:
            if curr_idx > prev_idx:
                direction = "INCREASED"
                interpretation = "OBSERVED_INCREASE"
            elif curr_idx < prev_idx:
                direction = "DECREASED"
                interpretation = "OBSERVED_DECREASE"
            else:
                direction = "UNCHANGED"
                interpretation = "UNCHANGED"
        else:
            # Fallback to simple string comparison if not standard severity levels
            if prev_sev == curr_sev:
                direction = "UNCHANGED"
                interpretation = "UNCHANGED"
            else:
                direction = "CHANGED"
                interpretation = "CHANGED"

        return ComparisonMetric(
            name="severity",
            previous_value=prev_sev,
            current_value=curr_sev,
            direction=direction,
            interpretation_category=interpretation
        )

    def _determine_overall_summary(
        self,
        prev_cls: str, curr_cls: str,
        prev_area: float, curr_area: float,
        prev_sev: str, curr_sev: str
    ) -> Tuple[str, str]:
        """Generates deterministic summary category and text based on metric differences."""
        if prev_cls != curr_cls:
            status = "CLASSIFICATION_CHANGED"
            text = f"AI-derived tumor classification changed from {prev_cls} to {curr_cls} between the selected reports."
            return status, text

        # Check severity changes
        severity_order = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
        prev_idx = severity_order.get(prev_sev.upper(), 0)
        curr_idx = severity_order.get(curr_sev.upper(), 0)
        sev_changed = prev_idx != curr_idx

        # Check area changes with tolerance
        tolerance = 1e-4
        area_diff = curr_area - prev_area
        if abs(area_diff) > tolerance:
            if area_diff > 0:
                status = "MEASUREMENTS_INCREASED"
                if prev_area > 0:
                    pct = (area_diff / prev_area) * 100.0
                    text = f"AI-derived tumor area increased by {pct:.1f}% between the selected reports."
                else:
                    text = f"AI-derived tumor area increased by {area_diff:.2f} mm² (from 0 mm²) between the selected reports."
            else:
                status = "MEASUREMENTS_DECREASED"
                if prev_area > 0:
                    pct = (abs(area_diff) / prev_area) * 100.0
                    text = f"AI-derived tumor area decreased by {pct:.1f}% between the selected reports."
                else:
                    text = f"AI-derived tumor area decreased by {abs(area_diff):.2f} mm² between the selected reports."

            # If both severity and area changed, we can mark it as MULTIPLE_CHANGES
            if sev_changed:
                status = "MULTIPLE_CHANGES"
                text += f" Additionally, severity classification changed from {prev_sev} to {curr_sev}."
            return status, text

        # If area is stable but severity changed
        if sev_changed:
            status = "MULTIPLE_CHANGES"
            dir_str = "increased" if curr_idx > prev_idx else "decreased"
            text = f"AI-derived tumor severity classification {dir_str} from {prev_sev} to {curr_sev} while tumor area remained stable."
            return status, text

        # stable cases
        status = "STABLE"
        if prev_cls == "No Tumor":
            text = "No diagnostic changes observed. Tumor measurements remain stable at zero."
        else:
            text = "AI-derived tumor measurements remain stable."
        return status, text

    def get_patient_longitudinal_timeline(
        self,
        patient_id: str,
        actor: Optional[Any] = None
    ) -> LongitudinalPatientTimeline:
        """Retrieves and constructs the chronological patient timeline with strict security checks."""
        import math
        import os
        self.logger.info(f"Retrieving patient timeline for patient_id={patient_id} by actor={actor}")

        # Helper to log audit events safely
        def safe_log_audit(event_type: str, user: Optional[Any], status: str, details: str) -> None:
            try:
                conn_audit = self._get_connection()
                try:
                    self._log_security_audit_event(conn_audit, event_type, user, None, status, details)
                    conn_audit.commit()
                finally:
                    conn_audit.close()
            except Exception as audit_err:
                self.logger.warning(f"Failed to log timeline security audit event: {audit_err}")

        # 1. Input Validation for patient_id
        if not patient_id or not isinstance(patient_id, str):
            safe_log_audit("PATIENT_TIMELINE_ACCESSED", actor, "FAILED", "Timeline request rejected: Invalid patient ID format.")
            raise ReportServiceException("Invalid patient ID format.")

        stripped_id = patient_id.strip()
        if not stripped_id:
            safe_log_audit("PATIENT_TIMELINE_ACCESSED", actor, "FAILED", "Timeline request rejected: Patient ID cannot be empty.")
            raise ReportServiceException("Invalid patient ID format.")

        if len(stripped_id) > 100:
            safe_log_audit("PATIENT_TIMELINE_ACCESSED", actor, "FAILED", f"Timeline request rejected: Patient ID is too long ({len(stripped_id)} chars).")
            raise ReportServiceException("Patient ID is too long.")

        if "/" in stripped_id or "\\" in stripped_id or ".." in stripped_id:
            safe_log_audit("PATIENT_TIMELINE_ACCESSED", actor, "FAILED", "Timeline request rejected: Malformed path traversal characters in Patient ID.")
            raise ReportServiceException("Malformed patient ID.")

        # 2. Enforce security access check
        if actor is None:
            safe_log_audit("PATIENT_TIMELINE_ACCESSED", None, "FAILED", f"Access denied to patient timeline for ID: {stripped_id}. Authentication required.")
            raise ReportServiceException("Authentication required.")

        role_val = actor.role.value if hasattr(actor.role, 'value') else str(actor.role).lower()
        if role_val not in ["admin", "doctor", "patient"]:
            safe_log_audit("PATIENT_TIMELINE_ACCESSED", actor, "FAILED", f"Access denied to patient timeline for ID: {stripped_id}. Unauthorized role: {role_val}.")
            raise ReportServiceException("Access denied.")

        if role_val == "patient":
            authorized = False
            user_uuid = actor.uuid.lower() if actor.uuid else ""
            user_name = actor.full_name.lower() if actor.full_name else ""

            if stripped_id.lower() == user_uuid:
                authorized = True
            else:
                conn = self._get_connection()
                try:
                    row_pat = conn.execute("SELECT name FROM patients WHERE patient_id = ?;", (stripped_id,)).fetchone()
                    if row_pat:
                        pat_name = row_pat["name"].lower()
                        if pat_name == user_name:
                            authorized = True
                finally:
                    conn.close()

            if not authorized:
                safe_log_audit("PATIENT_TIMELINE_ACCESSED", actor, "FAILED", f"Access denied to patient timeline for ID: {stripped_id}. Patient user mismatch (User uuid: {user_uuid}).")
                raise ReportServiceException("Access denied to patient timeline.")

        # 3. Fetch patient name and verify existence
        conn = self._get_connection()
        try:
            row_pat = conn.execute("SELECT name FROM patients WHERE patient_id = ?;", (stripped_id,)).fetchone()
            if not row_pat:
                safe_log_audit("PATIENT_TIMELINE_ACCESSED", actor, "FAILED", f"Timeline request failed. Patient with ID {stripped_id} not found in database.")
                raise ReportServiceException(f"Patient with ID {stripped_id} not found.")
            patient_name = row_pat["name"]

            # 4. Query all reports for patient
            query = """
                SELECT rv.report_id, rv.version_number, rv.prediction_id, rv.json_path, rv.status as version_status,
                       r.report_number, r.status as report_status,
                       p.predicted_class, p.confidence_score, p.tumor_area_mm2, p.tumor_percentage_brain,
                       p.rule_based_severity,
                       ms.id as scan_id, ms.scan_date
                FROM report_versions rv
                JOIN reports r ON rv.report_id = r.report_id
                LEFT JOIN predictions p ON rv.prediction_id = p.id
                LEFT JOIN mri_scans ms ON p.scan_id = ms.id
                WHERE r.patient_id = ? AND rv.version_number = r.current_version;
            """
            rows = conn.execute(query, (stripped_id,)).fetchall()
        finally:
            conn.close()

        def clean_float(val: Any) -> Optional[float]:
            if val is None:
                return None
            try:
                f = float(val)
                if math.isnan(f) or math.isinf(f):
                    return None
                return f
            except (ValueError, TypeError):
                if isinstance(val, str):
                    if val.lower().strip() in ("nan", "inf", "+inf", "-inf", "infinity", "+infinity", "-infinity"):
                        return None
                return None

        events = []
        for row in rows:
            report_id = row["report_id"]
            scan_id = row["scan_id"]
            scan_date = row["scan_date"]
            classification = row["predicted_class"]
            confidence = clean_float(row["confidence_score"])
            tumor_area = clean_float(row["tumor_area_mm2"])
            tumor_percentage = clean_float(row["tumor_percentage_brain"])
            severity = row["rule_based_severity"]
            report_status = row["report_status"]

            # Load and enrich from json file on disk if it exists
            json_path = row["json_path"]
            if json_path and os.path.exists(json_path):
                try:
                    import json
                    with open(json_path, "r", encoding="utf-8") as f:
                        data = json.load(f)

                    # Report-level Security: Verify Patient ID in JSON matches expected patient_id
                    if "patient" in data:
                        file_patient_id = data["patient"].get("patient_id")
                        if file_patient_id and str(file_patient_id).strip() != str(stripped_id).strip():
                            self.logger.warning(
                                f"Report-level Security Violation: Patient ID mismatch in JSON file {json_path}. "
                                f"Expected {stripped_id}, got {file_patient_id}."
                            )
                            # Ignore this JSON file completely to prevent information leakage, but keep DB details
                            data = {}

                        if data:
                            scan_date = data["patient"].get("scan_date", scan_date)
                    if "classification" in data:
                        c = data["classification"]
                        classification = c.get("predicted_class", classification)
                        confidence = clean_float(c.get("confidence_score", confidence))
                    if "segmentation" in data:
                        s = data["segmentation"]
                        tumor_area = clean_float(s.get("tumor_area_mm2", tumor_area))
                        tumor_percentage = clean_float(s.get("tumor_percentage_brain", tumor_percentage))
                    if "severity" in data:
                        severity = data["severity"].get("category", severity)
                except Exception as e:
                    self.logger.warning(f"Could not load/enrich from JSON report file at {json_path}: {e}")

            severity_score = None
            if severity:
                sev_upper = str(severity).upper()
                if sev_upper in ("LOW", "MEDIUM", "HIGH"):
                    if sev_upper == "LOW":
                        severity_score = 0.33
                    elif sev_upper == "MEDIUM":
                        severity_score = 0.66
                    elif sev_upper == "HIGH":
                        severity_score = 1.0

            event = LongitudinalTimelineEvent(
                report_id=report_id,
                scan_id=scan_id,
                patient_id=stripped_id,
                scan_date=scan_date,
                classification=classification,
                confidence=confidence,
                tumor_area=tumor_area,
                tumor_percentage=tumor_percentage,
                severity=severity,
                severity_score=severity_score,
                report_status=report_status
            )
            events.append(event)

        # Build domain timeline (automatically sorts and populates trends)
        timeline = LongitudinalPatientTimeline(
            patient_id=stripped_id,
            patient_name=patient_name,
            events=events
        )

        # Log success audit event
        safe_log_audit(
            "PATIENT_TIMELINE_ACCESSED",
            actor,
            "SUCCESS",
            f"Longitudinal timeline for patient {stripped_id} successfully accessed by user {actor.email if hasattr(actor, 'email') else (actor.get('email') if isinstance(actor, dict) else 'unknown')} with role {role_val}."
        )

        return timeline

    def get_patient_analytics(self, patient_id: str, actor: Any) -> PatientAnalytics:
        """Retrieves and calculates patient-level clinical analytics based on scan history."""
        import math
        # 1. Fetch timeline (this enforces access controls and validation)
        timeline = self.get_patient_longitudinal_timeline(patient_id, actor)

        # 2. Derive analytics from timeline events
        total_scans = timeline.total_events
        first_scan_date = timeline.first_scan_date
        latest_scan_date = timeline.latest_scan_date

        first_event = timeline.first_event
        latest_event = timeline.latest_event

        first_tumor_area = first_event.tumor_area if first_event else None
        latest_tumor_area = latest_event.tumor_area if latest_event else None

        # Absolute and percentage changes
        area_absolute_change = None
        area_percentage_change = None
        if "tumor_area" in timeline.metrics:
            area_absolute_change = timeline.metrics["tumor_area"].absolute_change
            area_percentage_change = timeline.metrics["tumor_area"].percentage_change

        # Occupancy change: absolute difference in tumor percentage
        occupancy_change = None
        if first_event and latest_event and first_event.tumor_percentage is not None and latest_event.tumor_percentage is not None:
            try:
                f_pct = float(first_event.tumor_percentage)
                l_pct = float(latest_event.tumor_percentage)
                if not (math.isnan(f_pct) or math.isinf(f_pct) or math.isnan(l_pct) or math.isinf(l_pct)):
                    occupancy_change = l_pct - f_pct
            except (ValueError, TypeError):
                pass

        # Confidence change
        confidence_change = None
        if "confidence" in timeline.metrics:
            confidence_change = timeline.metrics["confidence"].absolute_change

        # Severity change: absolute difference in severity scores
        severity_change = None
        if "severity" in timeline.metrics:
            severity_change = timeline.metrics["severity"].absolute_change

        # Classification history
        classification_history = [
            {"date": ev.scan_date, "classification": ev.classification}
            for ev in timeline.events
        ]

        progression_status = timeline.timeline_status

        # Create entity (which self-validates)
        return PatientAnalytics(
            patient_id=patient_id,
            patient_name=timeline.patient_name,
            total_scans=total_scans,
            first_scan_date=first_scan_date,
            latest_scan_date=latest_scan_date,
            first_tumor_area=first_tumor_area,
            latest_tumor_area=latest_tumor_area,
            area_absolute_change=area_absolute_change,
            area_percentage_change=area_percentage_change,
            occupancy_change=occupancy_change,
            confidence_change=confidence_change,
            severity_change=severity_change,
            classification_history=classification_history,
            progression_status=progression_status
        )

    def get_patient_timeseries(self, patient_id: str, actor: Any) -> TimeSeriesAnalytics:
        """Retrieves clinical timeseries data points for a patient."""
        # Enforces access control and validation
        timeline = self.get_patient_longitudinal_timeline(patient_id, actor)

        points = []
        for ev in timeline.events:
            if not ev.scan_date:
                continue
            point = TimeSeriesPoint(
                date=ev.scan_date,
                tumor_area=ev.tumor_area,
                occupancy=ev.tumor_percentage,
                confidence=ev.confidence,
                severity_score=ev.severity_score,
                severity=ev.severity,
                classification=ev.classification
            )
            points.append(point)

        return TimeSeriesAnalytics(
            patient_id=patient_id,
            points=points
        )

    def get_population_analytics(self, actor: Any) -> PopulationAnalytics:
        """Calculates population-level aggregated analytics. Restricted to Doctors and Admins."""
        import math
        if actor is None:
            raise ReportServiceException("Authentication required.")

        role_val = actor.role.value if hasattr(actor.role, 'value') else str(actor.role).lower()
        if role_val not in ["admin", "doctor"]:
            raise ReportServiceException("Access denied.")

        conn = self._get_connection()
        try:
            # 1. Base counts
            total_patients = conn.execute("SELECT COUNT(*) FROM patients;").fetchone()[0]
            total_reports = conn.execute("SELECT COUNT(*) FROM reports;").fetchone()[0]
            total_scans = conn.execute("SELECT COUNT(*) FROM mri_scans;").fetchone()[0]

            # 2. Classification and Severity distributions from active reports
            query_dist = """
                SELECT p.predicted_class, p.rule_based_severity, COUNT(*) as cnt
                FROM report_versions rv
                JOIN reports r ON rv.report_id = r.report_id
                JOIN predictions p ON rv.prediction_id = p.id
                WHERE rv.version_number = r.current_version
                GROUP BY p.predicted_class, p.rule_based_severity;
            """
            rows_dist = conn.execute(query_dist).fetchall()

            classification_distribution = {}
            severity_distribution = {}
            for r in rows_dist:
                cls = r["predicted_class"]
                sev = r["rule_based_severity"]
                cnt = r["cnt"]

                classification_distribution[cls] = classification_distribution.get(cls, 0) + cnt
                severity_distribution[sev] = severity_distribution.get(sev, 0) + cnt

            # 3. Average Confidence & Average Tumor Area
            query_avg = """
                SELECT AVG(p.confidence_score) as avg_conf, AVG(p.tumor_area_mm2) as avg_area
                FROM report_versions rv
                JOIN reports r ON rv.report_id = r.report_id
                JOIN predictions p ON rv.prediction_id = p.id
                WHERE rv.version_number = r.current_version;
            """
            row_avg = conn.execute(query_avg).fetchone()

            def clean_avg(val: Any) -> Optional[float]:
                if val is None:
                    return None
                try:
                    f = float(val)
                    if math.isnan(f) or math.isinf(f):
                        return None
                    return round(f, 4)
                except (ValueError, TypeError):
                    return None

            average_confidence = clean_avg(row_avg["avg_conf"]) if row_avg else None
            average_tumor_area = clean_avg(row_avg["avg_area"]) if row_avg else None

            # 4. Activity Over Time (grouped by scan date)
            rows_dates = conn.execute("SELECT scan_date FROM mri_scans;").fetchall()
            activity_dict = {}
            for row in rows_dates:
                s_date = row["scan_date"]
                if s_date:
                    date_str = str(s_date).split()[0].strip()
                    activity_dict[date_str] = activity_dict.get(date_str, 0) + 1

            activity_over_time = sorted(
                [{"date": dt, "count": cnt} for dt, cnt in activity_dict.items()],
                key=lambda x: x["date"]
            )

            # 5. Progression status distribution for all patients in the system
            def clean_float(val: Any) -> Optional[float]:
                if val is None:
                    return None
                try:
                    f = float(val)
                    if math.isnan(f) or math.isinf(f):
                        return None
                    return f
                except (ValueError, TypeError):
                    if isinstance(val, str):
                        if val.lower().strip() in ("nan", "inf", "+inf", "-inf", "infinity", "+infinity", "-infinity"):
                            return None
                    return None

            progression_distribution = {}
            rows_patients = conn.execute("SELECT patient_id, name FROM patients;").fetchall()

            # Execute a single set-based query to retrieve report version metadata and predictions for all patients
            query_all_timelines = """
                SELECT rv.report_id, rv.version_number, rv.prediction_id, rv.json_path, rv.status as version_status,
                       r.patient_id, r.report_number, r.status as report_status,
                       p.predicted_class, p.confidence_score, p.tumor_area_mm2, p.tumor_percentage_brain,
                       p.rule_based_severity,
                       ms.id as scan_id, ms.scan_date
                FROM report_versions rv
                JOIN reports r ON rv.report_id = r.report_id
                LEFT JOIN predictions p ON rv.prediction_id = p.id
                LEFT JOIN mri_scans ms ON p.scan_id = ms.id
                WHERE rv.version_number = r.current_version;
            """
            all_rows = conn.execute(query_all_timelines).fetchall()

            # Group rows by patient_id
            from collections import defaultdict
            patient_timeline_rows = defaultdict(list)
            for row in all_rows:
                pat_id_key = str(row["patient_id"]).strip().lower()
                patient_timeline_rows[pat_id_key].append(row)

            for row_pat in rows_patients:
                pat_id = row_pat["patient_id"]
                pat_name = row_pat["name"]
                stripped_id = str(pat_id).strip()
                lookup_key = stripped_id.lower()

                try:
                    # Enforce name length and traversal checks (matching timeline domain constraints)
                    if not stripped_id or len(stripped_id) > 64:
                        raise ReportServiceException("Patient ID too long or malformed.")
                    if "/" in stripped_id or "\\" in stripped_id or ".." in stripped_id:
                        raise ReportServiceException("Malformed patient ID.")

                    events = []
                    for row in patient_timeline_rows[lookup_key]:
                        report_id = row["report_id"]
                        scan_id = row["scan_id"]
                        scan_date = row["scan_date"]
                        classification = row["predicted_class"]
                        confidence = clean_float(row["confidence_score"])
                        tumor_area = clean_float(row["tumor_area_mm2"])
                        tumor_percentage = clean_float(row["tumor_percentage_brain"])
                        severity = row["rule_based_severity"]
                        report_status = row["report_status"]

                        # JSON file enrichment exactly as get_patient_longitudinal_timeline
                        json_path = row["json_path"]
                        if json_path and os.path.exists(json_path):
                            try:
                                import json
                                with open(json_path, "r", encoding="utf-8") as f:
                                    data = json.load(f)

                                # Report-level Security: Verify Patient ID in JSON matches expected patient_id
                                if "patient" in data:
                                    file_patient_id = data["patient"].get("patient_id")
                                    if file_patient_id and str(file_patient_id).strip() != str(stripped_id).strip():
                                        data = {}

                                    if data:
                                        scan_date = data["patient"].get("scan_date", scan_date)
                                if "classification" in data:
                                    c = data["classification"]
                                    classification = c.get("predicted_class", classification)
                                    confidence = clean_float(c.get("confidence_score", confidence))
                                if "segmentation" in data:
                                    s = data["segmentation"]
                                    tumor_area = clean_float(s.get("tumor_area_mm2", tumor_area))
                                    tumor_percentage = clean_float(s.get("tumor_percentage_brain", tumor_percentage))
                                if "severity" in data:
                                    severity = data["severity"].get("category", severity)
                            except Exception:
                                pass

                        severity_score = None
                        if severity:
                            sev_upper = str(severity).upper()
                            if sev_upper in ("LOW", "MEDIUM", "HIGH"):
                                if sev_upper == "LOW":
                                    severity_score = 0.33
                                elif sev_upper == "MEDIUM":
                                    severity_score = 0.66
                                elif sev_upper == "HIGH":
                                    severity_score = 1.0

                        event = LongitudinalTimelineEvent(
                            report_id=report_id,
                            scan_id=scan_id,
                            patient_id=stripped_id,
                            scan_date=scan_date,
                            classification=classification,
                            confidence=confidence,
                            tumor_area=tumor_area,
                            tumor_percentage=tumor_percentage,
                            severity=severity,
                            severity_score=severity_score,
                            report_status=report_status
                        )
                        events.append(event)

                    # Build domain timeline (automatically sorts and populates trends)
                    timeline = LongitudinalPatientTimeline(
                        patient_id=stripped_id,
                        patient_name=pat_name,
                        events=events
                    )
                    status = timeline.timeline_status
                    progression_distribution[status] = progression_distribution.get(status, 0) + 1
                except Exception:
                    progression_distribution["UNKNOWN"] = progression_distribution.get("UNKNOWN", 0) + 1

            return PopulationAnalytics(
                total_patients=total_patients,
                total_reports=total_reports,
                total_scans=total_scans,
                classification_distribution=classification_distribution,
                severity_distribution=severity_distribution,
                progression_distribution=progression_distribution,
                average_confidence=average_confidence,
                average_tumor_area=average_tumor_area,
                activity_over_time=activity_over_time
            )
        finally:
            conn.close()

    def send_report_email(
        self,
        report_id: int,
        actor: Optional[Any],
        recipient_email: Optional[str] = None,
        version: Optional[int] = None
    ) -> Dict[str, Any]:
        """Orchestrates authorization, template rendering, and delivery of a report PDF."""
        # 1. Authenticate caller
        if actor is None:
            raise ReportServiceException("Authentication required.")

        # 2. Authorize report access
        access_status = self.check_report_access(report_id, actor)
        if access_status == "NOT_FOUND":
            raise ReportNotFoundException(f"Report with ID {report_id} not found.")
        elif access_status == "UNAUTHORIZED":
            raise ReportServiceException("Authentication required.")
        elif access_status == "FORBIDDEN":
            raise PermissionError("Access denied to patient report.")

        # 3. Determine and validate recipient email
        caller_role = actor.role.value if hasattr(actor.role, 'value') else str(actor.role).lower()

        # If no recipient specified, default to caller's email
        if not recipient_email:
            recipient_email = actor.email

        # Standard format and injection validation
        from clinical_reporting.infrastructure.email_service import validate_email_address
        try:
            validate_email_address(recipient_email)
        except Exception as ve:
            raise ValueError(f"Invalid recipient email format: {ve}")

        # Recipient policy check
        if caller_role == "patient":
            # Patient can ONLY email to themselves (IDOR check)
            if recipient_email.lower().strip() != actor.email.lower().strip():
                raise PermissionError("Patients are only authorized to email reports to their own registered email.")
        else:
            # Doctor/Admin can email to themselves or any registered user in the system
            from security.infrastructure.repository import SQLiteUserRepository
            user_repo = SQLiteUserRepository(db_path=self.db_path)
            recipient_user = user_repo.get_by_email(recipient_email)
            if not recipient_user:
                raise PermissionError("Recipient must be a registered user on the AuraScan platform.")

        # 4. Resolve the version number if None
        import datetime
        conn = self._get_connection()
        try:
            if version is None:
                r_row = conn.execute("SELECT current_version FROM reports WHERE report_id = ?;", (report_id,)).fetchone()
                if not r_row:
                    raise ReportNotFoundException(f"Report with ID {report_id} not found.")
                version = r_row["current_version"]
        finally:
            conn.close()

        # Resolve config settings for retry limits
        from clinical_reporting.infrastructure.email_config import EmailConfig
        config = EmailConfig()
        max_attempts = config.email_max_attempts

        # Create Initial EmailDelivery record with 'SENDING' status
        attempted_at = datetime.datetime.utcnow().isoformat()
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO email_deliveries (
                    report_id, actor_user_id, recipient_email, status, attempted_at, created_at, updated_at,
                    attempt_count, max_attempts, report_version
                )
                VALUES (?, ?, ?, 'SENDING', ?, ?, ?, 0, ?, ?);
            """, (report_id, actor.id, recipient_email, attempted_at, attempted_at, attempted_at, max_attempts, version))
            conn.commit()
            delivery_id = cursor.lastrowid
        finally:
            conn.close()

        # 5. Execute the email delivery flow
        return self.execute_email_delivery(
            delivery_id=delivery_id,
            report_id=report_id,
            actor_user_id=actor.id,
            recipient_email=recipient_email,
            version=version,
            attempt_count=0,
            max_attempts=max_attempts
        )

    def execute_email_delivery(
        self,
        delivery_id: int,
        report_id: int,
        actor_user_id: int,
        recipient_email: str,
        version: Optional[int],
        attempt_count: int,
        max_attempts: int
    ) -> Dict[str, Any]:
        """Performs the email send operation for an existing or retried email delivery record."""
        import datetime
        # 1. Fetch actor
        from security.infrastructure.repository import SQLiteUserRepository
        user_repo = SQLiteUserRepository(db_path=self.db_path)
        actor = user_repo.get_by_id(actor_user_id)
        if not actor:
            self._mark_delivery_failed(
                delivery_id=delivery_id,
                reason="INVALID_RECIPIENT",
                actor=None,
                report_id=report_id,
                recipient_email=recipient_email,
                error_msg="Actor user not found"
            )
            return {"success": False, "message": "Actor user not found"}

        # 2. Resolve version details
        conn = self._get_connection()
        try:
            if version is None:
                r_row = conn.execute("SELECT current_version FROM reports WHERE report_id = ?;", (report_id,)).fetchone()
                if not r_row:
                    raise ReportNotFoundException(f"Report with ID {report_id} not found.")
                version = r_row["current_version"]

            # Save version to DB if not set
            conn.execute("UPDATE email_deliveries SET report_version = ? WHERE id = ? AND report_version IS NULL;", (version, delivery_id))
            conn.commit()

            query = """
                SELECT r.report_number, rv.version_number, p.patient_id, p.name as patient_name, rv.created_at
                FROM report_versions rv
                JOIN reports r ON rv.report_id = r.report_id
                JOIN patients p ON r.patient_id = p.patient_id
                WHERE rv.report_id = ? AND rv.version_number = ?;
            """
            row = conn.execute(query, (report_id, version)).fetchone()
            if not row:
                raise VersionNotFoundException(f"Version {version} not found for report {report_id}.")

            report_number = row["report_number"]
            version_number = row["version_number"]
            patient_id = row["patient_id"]
            patient_name = row["patient_name"]
            report_date = row["created_at"]
        except Exception as e:
            self._mark_delivery_failed(
                delivery_id=delivery_id,
                reason="INVALID_REPORT",
                actor=actor,
                report_id=report_id,
                recipient_email=recipient_email,
                error_msg=str(e)
            )
            raise e
        finally:
            conn.close()

        # Update attempt info in DB prior to send
        now = datetime.datetime.utcnow().isoformat()
        conn = self._get_connection()
        try:
            conn.execute("""
                UPDATE email_deliveries
                SET attempt_count = ?, last_attempt_at = ?, updated_at = ?
                WHERE id = ?;
            """, (attempt_count + 1, now, now, delivery_id))
            conn.commit()
        finally:
            conn.close()

        # Resolve secure PDF path (runs integrity and traversal checks)
        try:
            pdf_path = self.resolve_secure_pdf_path(report_id, version)
            if not os.path.exists(pdf_path):
                raise FileNotFoundError(f"PDF file not found at {pdf_path}")
            if os.path.getsize(pdf_path) > 10 * 1024 * 1024:
                raise ValueError("Report PDF size exceeds the limit.")
        except Exception as e:
            self._mark_delivery_failed(
                delivery_id=delivery_id,
                reason="MISSING_PDF" if isinstance(e, FileNotFoundError) else "INVALID_REPORT",
                actor=actor,
                report_id=report_id,
                recipient_email=recipient_email,
                error_msg=str(e)
            )
            raise e

        # 3. Render secure email context and templates
        try:
            from clinical_reporting.presentation.email_templates import EmailTemplateRenderer
            redacted_patient_id = patient_id[:4] + "****" if len(patient_id) > 4 else "****"

            subject = f"AuraScan Clinical Report Notification: {report_number}"
            title = f"Clinical Report: {report_number}"
            message_body = (
                f"Dear user,\n\n"
                f"The clinical report version {version_number} has been generated and is attached to this notification."
            )

            card_items = [
                ("Report Number", report_number),
                ("Version", str(version_number)),
                ("Patient Identifier", redacted_patient_id),
                ("Generated Date", report_date.split("T")[0] if "T" in report_date else report_date)
            ]

            html_body, text_body = EmailTemplateRenderer.render_generic_notification(
                subject=subject,
                title=title,
                message_body=message_body,
                cta_text="Access AuraScan Dashboard",
                cta_url="https://portal.aurascan.ai/dashboard",
                card_items=card_items,
                badge_label="Clinical PDF Attached",
                badge_type="info"
            )
        except Exception as e:
            self._mark_delivery_failed(
                delivery_id=delivery_id,
                reason="TEMPLATE_RENDER_FAILURE",
                actor=actor,
                report_id=report_id,
                recipient_email=recipient_email,
                error_msg=str(e)
            )
            raise e

        # 4. Dispatch email via EmailService
        from clinical_reporting.infrastructure.email_service import EmailService, ConfigurationException
        email_svc = EmailService()
        try:
            email_svc.send(
                to_email=recipient_email,
                subject=subject,
                body_text=text_body,
                body_html=html_body,
                attachment_path=pdf_path
            )
            # Update delivery status to SENT
            now = datetime.datetime.utcnow().isoformat()
            conn = self._get_connection()
            try:
                conn.execute("""
                    UPDATE email_deliveries
                    SET status = 'SENT', sent_at = ?, next_retry_at = NULL, updated_at = ?
                    WHERE id = ?;
                """, (now, now, delivery_id))
                conn.commit()
            finally:
                conn.close()

            # Log security event
            self.log_report_access_event(
                "REPORT_EMAIL_SENT",
                actor,
                report_id,
                "SUCCESS",
                f"Clinical report email sent to {recipient_email}."
            )
            return {"success": True, "message": "Report email sent successfully"}
        except ConfigurationException as e:
            # Handle failure with classification and retry check
            self._handle_delivery_failure(
                delivery_id=delivery_id,
                exception=e,
                actor=actor,
                report_id=report_id,
                recipient_email=recipient_email,
                attempt_count=attempt_count + 1,
                max_attempts=max_attempts
            )
            raise e
        except Exception as e:
            # Handle failure with classification and retry check
            self._handle_delivery_failure(
                delivery_id=delivery_id,
                exception=e,
                actor=actor,
                report_id=report_id,
                recipient_email=recipient_email,
                attempt_count=attempt_count + 1,
                max_attempts=max_attempts
            )
            raise ReportServiceException(f"Failed to send email: {e}")

    def execute_account_email_delivery(
        self,
        delivery_id: int,
        recipient_email: str,
        subject: str,
        body_text: str,
        body_html: Optional[str],
        attempt_count: int,
        max_attempts: int
    ) -> Dict[str, Any]:
        """Performs the email send operation for an account-related email retry/dispatch."""
        import datetime
        now = datetime.datetime.utcnow().isoformat()

        # Update attempt info in DB prior to send
        conn = self._get_connection()
        try:
            conn.execute("""
                UPDATE email_deliveries
                SET attempt_count = ?, last_attempt_at = ?, updated_at = ?
                WHERE id = ?;
            """, (attempt_count + 1, now, now, delivery_id))
            conn.commit()
        finally:
            conn.close()

        # Dispatch email via EmailService
        from clinical_reporting.infrastructure.email_service import EmailService, ConfigurationException
        email_svc = EmailService()
        try:
            email_svc.send(
                to_email=recipient_email,
                subject=subject,
                body_text=body_text,
                body_html=body_html
            )
            # Update delivery status to SENT
            now = datetime.datetime.utcnow().isoformat()
            conn = self._get_connection()
            try:
                conn.execute("""
                    UPDATE email_deliveries
                    SET status = 'SENT', sent_at = ?, next_retry_at = NULL, updated_at = ?
                    WHERE id = ?;
                """, (now, now, delivery_id))
                conn.commit()
            finally:
                conn.close()

            # Log security event
            self.log_report_access_event(
                "ACCOUNT_EMAIL_SENT",
                None,
                None,
                "SUCCESS",
                f"Account email ({subject}) sent to {recipient_email}."
            )
            return {"success": True, "message": "Account email sent successfully"}
        except ConfigurationException as e:
            # Handle failure with classification and retry check
            self._handle_delivery_failure(
                delivery_id=delivery_id,
                exception=e,
                actor=None,
                report_id=None,
                recipient_email=recipient_email,
                attempt_count=attempt_count + 1,
                max_attempts=max_attempts
            )
            raise e
        except Exception as e:
            # Handle failure with classification and retry check
            self._handle_delivery_failure(
                delivery_id=delivery_id,
                exception=e,
                actor=None,
                report_id=None,
                recipient_email=recipient_email,
                attempt_count=attempt_count + 1,
                max_attempts=max_attempts
            )
            raise ReportServiceException(f"Failed to send email: {e}")

    def classify_email_error(self, exception: Exception) -> tuple[str, bool]:
        """Classifies an email exception to standardized failure code and retry status.

        Returns:
            Tuple[str, bool]: (reason_code, is_retryable)
        """
        from clinical_reporting.infrastructure.email_service import (
            InvalidAddressException,
            ConfigurationException,
            AuthenticationException,
            AttachmentException,
            ConnectionException
        )

        err_msg = str(exception).lower()

        # 1. Match specific clean architecture exceptions
        if isinstance(exception, InvalidAddressException):
            return "INVALID_RECIPIENT", False
        if isinstance(exception, ConfigurationException):
            return "INVALID_CONFIGURATION", False
        if isinstance(exception, AuthenticationException):
            return "SMTP_AUTH_FAILED", False
        if isinstance(exception, AttachmentException):
            if "not found" in err_msg or "does not exist" in err_msg:
                return "MISSING_PDF", False
            return "INVALID_REPORT", False
        if isinstance(exception, ConnectionException):
            if "timeout" in err_msg:
                return "SMTP_TIMEOUT", True
            return "SMTP_CONNECTION_FAILED", True
        if isinstance(exception, PermissionError):
            return "UNAUTHORIZED_RECIPIENT", False
        if isinstance(exception, (ReportNotFoundException, VersionNotFoundException)):
            return "INVALID_REPORT", False

        # 2. Fallback to parsing error messages
        if "timeout" in err_msg:
            return "SMTP_TIMEOUT", True
        if "auth" in err_msg or "credential" in err_msg or "login" in err_msg:
            return "SMTP_AUTH_FAILED", False
        if "connection" in err_msg or "refused" in err_msg or "socket" in err_msg or "unreachable" in err_msg:
            return "SMTP_CONNECTION_FAILED", True
        if "temporary" in err_msg or "421" in err_msg or "450" in err_msg or "451" in err_msg or "452" in err_msg:
            return "SMTP_TEMPORARY_UNAVAILABLE", True
        if "recipient" in err_msg or "address" in err_msg or "550" in err_msg or "553" in err_msg:
            return "INVALID_RECIPIENT", False
        if "pdf" in err_msg or "file not found" in err_msg:
            return "MISSING_PDF", False
        if "template" in err_msg or "render" in err_msg:
            return "TEMPLATE_RENDER_FAILURE", False

        # Default fallback
        return "EMAIL_SEND_FAILED", False

    def _mark_delivery_failed(
        self,
        delivery_id: int,
        reason: str,
        actor: Optional[Any],
        report_id: Optional[int],
        recipient_email: str,
        error_msg: str
    ) -> None:
        import datetime
        now = datetime.datetime.utcnow().isoformat()
        conn = self._get_connection()
        try:
            conn.execute("""
                UPDATE email_deliveries
                SET status = 'FAILED', failure_reason = ?, last_failure_code = ?, retryable = 0, next_retry_at = NULL, updated_at = ?
                WHERE id = ?;
            """, (reason, reason, now, delivery_id))
            conn.commit()
        finally:
            conn.close()

        event_name = "REPORT_EMAIL_FAILED" if report_id is not None else "ACCOUNT_EMAIL_FAILED"
        msg_prefix = "clinical report" if report_id is not None else "account email"
        self.log_report_access_event(
            event_name,
            actor,
            report_id,
            "FAILED",
            f"Failed to email {msg_prefix} to {recipient_email} (Permanent: {reason}). Error: {error_msg}"
        )

        try:
            if actor and hasattr(actor, "id") and actor.id:
                from clinical_reporting.application.notification_service import NotificationService
                import json
                notif_svc = NotificationService(db_path=self.db_path)
                notif_svc.create_notification(
                    user_id=actor.id,
                    type_="EMAIL_DELIVERY_FAILED",
                    title="Email Delivery Failed",
                    message=f"Failed to email clinical report to {recipient_email}. Error: {error_msg}",
                    metadata_json=json.dumps({"report_id": report_id, "reason": reason})
                )
        except Exception as notif_err:
            self.logger.error(f"Failed to create email failure notification: {notif_err}")

    def _handle_delivery_failure(
        self,
        delivery_id: int,
        exception: Exception,
        actor: Optional[Any],
        report_id: Optional[int],
        recipient_email: str,
        attempt_count: int,
        max_attempts: int
    ) -> None:
        import datetime
        import random
        from clinical_reporting.infrastructure.email_config import EmailConfig
        config = EmailConfig()

        reason, is_transient = self.classify_email_error(exception)
        now = datetime.datetime.utcnow().isoformat()

        if is_transient and config.email_retry_enabled and attempt_count < max_attempts:
            # Exponential Backoff with upper bound
            base_delay = config.email_retry_base_delay
            max_delay = config.email_retry_max_delay

            exponent = min(attempt_count - 1, 30)
            delay = base_delay * (2 ** exponent)
            delay = min(delay, max_delay)

            # Small bounded jitter
            jitter = random.uniform(0.0, min(5.0, delay * 0.1))
            total_delay = delay + jitter

            next_retry_dt = datetime.datetime.utcnow() + datetime.timedelta(seconds=total_delay)
            next_retry_at = next_retry_dt.isoformat()

            conn = self._get_connection()
            try:
                conn.execute("""
                    UPDATE email_deliveries
                    SET status = 'RETRY_PENDING', failure_reason = ?, last_failure_code = ?, retryable = 1, next_retry_at = ?, updated_at = ?
                    WHERE id = ?;
                """, (reason, reason, next_retry_at, now, delivery_id))
                conn.commit()
            finally:
                conn.close()

            event_name = "REPORT_EMAIL_RETRY_SCHEDULED" if report_id is not None else "ACCOUNT_EMAIL_RETRY_SCHEDULED"
            msg_prefix = "clinical report" if report_id is not None else "account email"
            self.log_report_access_event(
                event_name,
                actor,
                report_id,
                "FAILED",
                f"Failed to email {msg_prefix} to {recipient_email} (Transient: {reason}). Scheduled retry {attempt_count + 1}/{max_attempts} at {next_retry_at}. Error: {exception}"
            )

            try:
                if actor and hasattr(actor, "id") and actor.id:
                    from clinical_reporting.application.notification_service import NotificationService
                    import json
                    notif_svc = NotificationService(db_path=self.db_path)
                    notif_svc.create_notification(
                        user_id=actor.id,
                        type_="EMAIL_RETRY_PENDING",
                        title="Email Retry Scheduled",
                        message=f"Transient email delivery failure to {recipient_email}. Retrying soon.",
                        metadata_json=json.dumps({"report_id": report_id, "reason": reason, "next_retry_at": next_retry_at})
                    )
            except Exception as notif_err:
                self.logger.error(f"Failed to create email retry notification: {notif_err}")
        else:
            conn = self._get_connection()
            try:
                conn.execute("""
                    UPDATE email_deliveries
                    SET status = 'FAILED', failure_reason = ?, last_failure_code = ?, retryable = 0, next_retry_at = NULL, updated_at = ?
                    WHERE id = ?;
                """, (reason, reason, now, delivery_id))
                conn.commit()
            finally:
                conn.close()

            event_name = "REPORT_EMAIL_FAILED" if report_id is not None else "ACCOUNT_EMAIL_FAILED"
            msg_prefix = "clinical report" if report_id is not None else "account email"
            self.log_report_access_event(
                event_name,
                actor,
                report_id,
                "FAILED",
                f"Failed to email {msg_prefix} to {recipient_email} (Permanent or Max Attempts Reached). Error: {exception}"
            )

            try:
                if actor and hasattr(actor, "id") and actor.id:
                    from clinical_reporting.application.notification_service import NotificationService
                    import json
                    notif_svc = NotificationService(db_path=self.db_path)
                    notif_svc.create_notification(
                        user_id=actor.id,
                        type_="EMAIL_DELIVERY_FAILED",
                        title="Email Delivery Failed",
                        message=f"Failed to email report to {recipient_email}. Max attempts reached.",
                        metadata_json=json.dumps({"report_id": report_id, "reason": reason})
                    )
            except Exception as notif_err:
                self.logger.error(f"Failed to create email failure notification: {notif_err}")

    def get_email_history(
        self,
        actor: Optional[Any],
        page: int = 1,
        per_page: int = 10,
        status: Optional[str] = None,
        search: Optional[str] = None,
        report_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """Fetches paginated, authorized email delivery records with optional filters and search."""
        if actor is None:
            raise PermissionError("Authentication required.")

        caller_role = actor.role.value if hasattr(actor.role, 'value') else str(actor.role).lower()

        # Build SQL query dynamically
        where_clauses = []
        params = []

        # Enforce RBAC/IDOR: Patients can only see histories for reports they own
        if caller_role == "patient":
            where_clauses.append("(r.patient_id = ? OR p.name = ?)")
            params.extend([actor.uuid, actor.full_name])

        if status:
            where_clauses.append("ed.status = ?")
            params.append(status)

        if report_id:
            where_clauses.append("ed.report_id = ?")
            params.append(report_id)

        if search:
            where_clauses.append("(ed.recipient_email LIKE ? OR r.report_number LIKE ?)")
            search_param = f"%{search}%"
            params.extend([search_param, search_param])

        where_str = ""
        if where_clauses:
            where_str = "WHERE " + " AND ".join(where_clauses)

        # Count query
        count_query = f"""
            SELECT COUNT(*) as cnt
            FROM email_deliveries ed
            LEFT JOIN reports r ON ed.report_id = r.report_id
            LEFT JOIN patients p ON r.patient_id = p.patient_id
            {where_str}
        """

        # Data query
        offset = (page - 1) * per_page
        data_query = f"""
            SELECT ed.id, ed.report_id, ed.actor_user_id, ed.recipient_email, ed.status,
                   ed.attempted_at, ed.sent_at, ed.failure_reason, ed.email_type,
                   r.report_number, u.full_name as actor_name, u.email as actor_email
            FROM email_deliveries ed
            LEFT JOIN reports r ON ed.report_id = r.report_id
            LEFT JOIN patients p ON r.patient_id = p.patient_id
            LEFT JOIN users u ON ed.actor_user_id = u.id
            {where_str}
            ORDER BY ed.attempted_at DESC
            LIMIT ? OFFSET ?
        """

        conn = self._get_connection()
        try:
            # Execute count
            row_count = conn.execute(count_query, params).fetchone()
            total_items = row_count["cnt"] if row_count else 0

            # Execute data query
            data_params = list(params)
            data_params.extend([per_page, offset])
            rows = conn.execute(data_query, data_params).fetchall()

            items = []
            for r in rows:
                masked_recipient = mask_email(r["recipient_email"])
                actor_email = r["actor_email"]
                if actor_email:
                    masked_actor_email = mask_email(actor_email) if caller_role == "patient" else actor_email
                else:
                    masked_actor_email = "system@aurascan.ai"

                items.append({
                    "id": r["id"],
                    "report_id": r["report_id"],
                    "report_number": r["report_number"] if r["report_number"] else "N/A",
                    "actor_user_id": r["actor_user_id"],
                    "actor_name": r["actor_name"] if r["actor_name"] else "System",
                    "actor_email": masked_actor_email,
                    "recipient_email": masked_recipient,
                    "status": r["status"],
                    "attempted_at": r["attempted_at"],
                    "sent_at": r["sent_at"],
                    "failure_reason": r["failure_reason"],
                    "email_type": r["email_type"] if "email_type" in r.keys() else "REPORT"
                })

            total_pages = (total_items + per_page - 1) // per_page if total_items > 0 else 0

            return {
                "items": items,
                "total_items": total_items,
                "page": page,
                "per_page": per_page,
                "total_pages": total_pages
            }
        finally:
            conn.close()


def mask_email(email: str) -> str:
    """Masks email address characters for confidentiality."""
    if not email or "@" not in email:
        return email
    parts = email.split("@")
    name = parts[0]
    domain = parts[1]
    if len(name) <= 2:
        masked_name = name + "****"
    else:
        masked_name = name[:2] + "****"
    return f"{masked_name}@{domain}"

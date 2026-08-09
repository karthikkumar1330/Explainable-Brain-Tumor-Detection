import os
import time
import logging
import threading
import datetime
import traceback
from typing import Optional, Any, Dict

logger = logging.getLogger("email_retry_scheduler")


class EmailRetryScheduler:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(EmailRetryScheduler, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self, db_path: Optional[str] = None, poll_interval: float = 10.0) -> None:
        if self._initialized:
            return
        self.db_path = db_path or os.environ.get("DB_PATH", "outputs/clinical_reports.db")
        self.poll_interval = poll_interval
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._initialized = True

    def start(self) -> None:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                logger.info("EmailRetryScheduler is already running.")
                return
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._run_loop, name="EmailRetrySchedulerThread", daemon=True)
            self._thread.start()
            logger.info("EmailRetryScheduler started.")

    def stop(self) -> None:
        with self._lock:
            if self._thread is None:
                return
            self._stop_event.set()
            self._thread.join(timeout=5)
            self._thread = None
            logger.info("EmailRetryScheduler stopped.")

    def _run_loop(self) -> None:
        logger.info("Scheduler loop started.")
        while not self._stop_event.is_set():
            try:
                self._process_pending_retries()
            except Exception as e:
                logger.error(f"Error in EmailRetryScheduler loop: {e}\n{traceback.format_exc()}")
            
            # Sleep in small chunks to respond quickly to stop event
            sleep_time = self.poll_interval
            while sleep_time > 0 and not self._stop_event.is_set():
                time.sleep(min(1.0, sleep_time))
                sleep_time -= 1.0
        logger.info("Scheduler loop stopped.")

    def _process_pending_retries(self) -> None:
        # Avoid circular dependencies by importing service inside
        from clinical_reporting.application.services import ReportService
        service = ReportService(db_path=self.db_path)
        
        # Query pending retries
        conn = service._get_connection()
        try:
            now = datetime.datetime.utcnow().isoformat()
            query = """
                SELECT id, report_id, actor_user_id, recipient_email, attempt_count, max_attempts, report_version
                FROM email_deliveries
                WHERE status = 'RETRY_PENDING'
                  AND next_retry_at IS NOT NULL
                  AND next_retry_at <= ?;
            """
            rows = conn.execute(query, (now,)).fetchall()
        finally:
            conn.close()

        for row in rows:
            delivery_id = row["id"]
            report_id = row["report_id"]
            actor_user_id = row["actor_user_id"]
            recipient_email = row["recipient_email"]
            attempt_count = row["attempt_count"]
            max_attempts = row["max_attempts"]
            version = row["report_version"]

            # Claim job to avoid double execution (Idempotency Protection)
            conn = service._get_connection()
            claimed = False
            try:
                cursor = conn.cursor()
                claim_now = datetime.datetime.utcnow().isoformat()
                cursor.execute("""
                    UPDATE email_deliveries
                    SET status = 'SENDING', updated_at = ?
                    WHERE id = ? AND status = 'RETRY_PENDING';
                """, (claim_now, delivery_id))
                conn.commit()
                claimed = cursor.rowcount > 0
            except Exception as claim_err:
                logger.error(f"Failed to claim delivery job {delivery_id}: {claim_err}")
            finally:
                conn.close()

            if not claimed:
                continue

            # Run execution safely
            try:
                logger.info(f"Processing retry for delivery ID {delivery_id} (Attempt {attempt_count + 1}/{max_attempts})")
                service.execute_email_delivery(
                    delivery_id=delivery_id,
                    report_id=report_id,
                    actor_user_id=actor_user_id,
                    recipient_email=recipient_email,
                    version=version,
                    attempt_count=attempt_count,
                    max_attempts=max_attempts
                )
            except Exception as exec_err:
                logger.error(f"Failed to execute retry for delivery ID {delivery_id}: {exec_err}")

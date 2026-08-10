import sqlite3
import datetime
import json
import logging
from typing import Optional, List, Dict, Any

class NotificationService:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.logger = logging.getLogger("notification_service")

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn

    def create_notification(
        self,
        user_id: int,
        type_: str,
        title: str,
        message: str,
        metadata_json: Optional[str] = None
    ) -> Optional[int]:
        # Validate type_ against controlled types
        valid_types = [
            "ANALYSIS_COMPLETED",
            "REPORT_READY",
            "EMAIL_DELIVERY_FAILED",
            "EMAIL_RETRY_PENDING",
            "SECURITY_LOGIN",
            "SECURITY_WARNING",
            "ACCOUNT_UPDATE",
            "SYSTEM_ALERT",
            "QUALITY_FLAG_CREATED",
            "QUALITY_FLAG_RESOLVED"
        ]
        if type_ not in valid_types:
            raise ValueError(f"Invalid notification type: {type_}")

        # Check preferences (ignore check for SECURITY_WARNING or SYSTEM_ALERT)
        pref_category = self._get_pref_category(type_)
        if type_ not in ["SECURITY_WARNING", "SYSTEM_ALERT"]:
            prefs = self.get_preferences(user_id)
            if not prefs.get(pref_category, True):
                self.logger.info(f"Notification of type {type_} suppressed by user {user_id} preferences")
                return None

        # Clean/sanitize metadata (never store sensitive payloads)
        if metadata_json:
            try:
                meta = json.loads(metadata_json)
                # Strip out sensitive keys
                sensitive_keys = ["password", "otp", "token", "jwt", "image", "payload", "access_token", "otp_code", "password_hash"]
                meta = {k: v for k, v in meta.items() if not any(sk in k.lower() for sk in sensitive_keys)}
                metadata_json = json.dumps(meta)
            except Exception:
                metadata_json = None

        conn = self._get_connection()
        try:
            now = datetime.datetime.utcnow().isoformat()
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO notifications (user_id, type, title, message, is_read, created_at, read_at, metadata_json)
                VALUES (?, ?, ?, ?, 0, ?, NULL, ?);
            """, (user_id, type_, title, message, now, metadata_json))
            conn.commit()
            return cursor.lastrowid
        except Exception as e:
            self.logger.error(f"Failed to create notification: {e}")
            return None
        finally:
            conn.close()

    def _get_pref_category(self, type_: str) -> str:
        if type_ == "ANALYSIS_COMPLETED":
            return "analysis"
        elif type_ in ["REPORT_READY", "EMAIL_DELIVERY_FAILED", "EMAIL_RETRY_PENDING", "QUALITY_FLAG_CREATED", "QUALITY_FLAG_RESOLVED"]:
            return "report"
        elif type_ in ["SECURITY_LOGIN", "SECURITY_WARNING"]:
            return "security"
        else:
            return "account"

    def get_preferences(self, user_id: int) -> Dict[str, bool]:
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT analysis_enabled, report_enabled, security_enabled, account_enabled
                FROM notification_preferences WHERE user_id = ?;
            """, (user_id,))
            row = cursor.fetchone()
            if row:
                return {
                    "analysis": bool(row["analysis_enabled"]),
                    "report": bool(row["report_enabled"]),
                    "security": bool(row["security_enabled"]),
                    "account": bool(row["account_enabled"])
                }
            else:
                # Default preferences to True
                return {
                    "analysis": True,
                    "report": True,
                    "security": True,
                    "account": True
                }
        finally:
            conn.close()

    def update_preferences(
        self,
        user_id: int,
        analysis: bool,
        report: bool,
        security: bool,
        account: bool
    ) -> None:
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO notification_preferences (user_id, analysis_enabled, report_enabled, security_enabled, account_enabled)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                    analysis_enabled=excluded.analysis_enabled,
                    report_enabled=excluded.report_enabled,
                    security_enabled=excluded.security_enabled,
                    account_enabled=excluded.account_enabled;
            """, (user_id, int(analysis), int(report), int(security), int(account)))
            conn.commit()
        finally:
            conn.close()

    def list_notifications(
        self,
        user_id: int,
        page: int = 1,
        page_size: int = 10,
        type_filter: Optional[str] = None,
        unread_only: bool = False
    ) -> Dict[str, Any]:
        if page < 1:
            raise ValueError("Page must be >= 1")
        if page_size < 1 or page_size > 100:
            raise ValueError("Page size must be between 1 and 100")

        conn = self._get_connection()
        try:
            cursor = conn.cursor()

            # Build where clause
            where_clauses = ["user_id = ?"]
            params = [user_id]

            if type_filter:
                where_clauses.append("type = ?")
                params.append(type_filter)

            if unread_only:
                where_clauses.append("is_read = 0")

            where_str = " AND ".join(where_clauses)

            # Get total count
            count_query = f"SELECT COUNT(*) as total FROM notifications WHERE {where_str};"
            cursor.execute(count_query, params)
            total_count = cursor.fetchone()["total"]

            # Main query
            offset = (page - 1) * page_size
            query = f"""
                SELECT id, user_id, type, title, message, is_read, created_at, read_at, metadata_json
                FROM notifications
                WHERE {where_str}
                ORDER BY created_at DESC
                LIMIT ? OFFSET ?;
            """
            cursor.execute(query, params + [page_size, offset])
            rows = cursor.fetchall()

            items = []
            for r in rows:
                items.append({
                    "id": r["id"],
                    "user_id": r["user_id"],
                    "type": r["type"],
                    "title": r["title"],
                    "message": r["message"],
                    "is_read": bool(r["is_read"]),
                    "created_at": r["created_at"],
                    "read_at": r["read_at"],
                    "metadata_json": r["metadata_json"]
                })

            import math
            total_pages = math.ceil(total_count / page_size) if total_count > 0 else 1

            return {
                "items": items,
                "total_count": total_count,
                "page": page,
                "page_size": page_size,
                "total_pages": total_pages
            }
        finally:
            conn.close()

    def get_unread_count(self, user_id: int) -> int:
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT COUNT(*) as unread_cnt FROM notifications WHERE user_id = ? AND is_read = 0;
            """, (user_id,))
            return cursor.fetchone()["unread_cnt"]
        finally:
            conn.close()

    def mark_notification_read(self, user_id: int, notification_id: int) -> bool:
        conn = self._get_connection()
        try:
            now = datetime.datetime.utcnow().isoformat()
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE notifications
                SET is_read = 1, read_at = COALESCE(read_at, ?)
                WHERE id = ? AND user_id = ?;
            """, (now, notification_id, user_id))
            conn.commit()
            return cursor.rowcount > 0
        finally:
            conn.close()

    def mark_all_notifications_read(self, user_id: int) -> None:
        conn = self._get_connection()
        try:
            now = datetime.datetime.utcnow().isoformat()
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE notifications
                SET is_read = 1, read_at = COALESCE(read_at, ?)
                WHERE user_id = ? AND is_read = 0;
            """, (now, user_id))
            conn.commit()
        finally:
            conn.close()

    def delete_notification(self, user_id: int, notification_id: int) -> bool:
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                DELETE FROM notifications WHERE id = ? AND user_id = ?;
            """, (notification_id, user_id))
            conn.commit()
            return cursor.rowcount > 0
        finally:
            conn.close()

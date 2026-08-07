import sqlite3
import datetime
import uuid
import logging
from typing import Optional, List, Dict, Any
from security.domain.entities import User, Role, SecurityAuditLog, TokenType
from security.domain.interfaces import IUserRepository
from security.infrastructure.password import PasswordHasher


class SQLiteUserRepository(IUserRepository):
    """SQLite persistence layer for users, auth tokens, session revocations, and audit logs."""

    def __init__(self, db_path: str, logger: Optional[logging.Logger] = None) -> None:
        self.db_path = db_path
        self.logger = logger or logging.getLogger("security_repo")

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn

    def initialize_security_tables(self) -> None:
        """Creates security schema tables if they do not exist. Cleans old schemas if google_id is present."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()

            # Check if old table with google_id column exists
            cursor.execute("PRAGMA table_info(users);")
            columns = [row["name"] for row in cursor.fetchall()]
            if "google_id" in columns or "two_factor_enabled" in columns:
                self.logger.info("Legacy columns detected in users table. Cleaning and recreating security tables...")
                cursor.execute("DROP TABLE IF EXISTS verification_tokens;")
                cursor.execute("DROP TABLE IF EXISTS otp_codes;")
                cursor.execute("DROP TABLE IF EXISTS users;")

            # Also drop token/otp tables if they exist (since they are permanently removed)
            cursor.execute("DROP TABLE IF EXISTS verification_tokens;")
            cursor.execute("DROP TABLE IF EXISTS otp_codes;")

            cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                uuid TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                full_name TEXT NOT NULL,
                role TEXT NOT NULL,
                is_verified INTEGER DEFAULT 1,
                is_active INTEGER DEFAULT 1,
                profile_pic TEXT,
                sessions_revoked_at TEXT,
                failed_login_attempts INTEGER DEFAULT 0,
                lockout_until TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                last_login_at TEXT
            );
            """)

            cursor.execute("""
            CREATE TABLE IF NOT EXISTS revoked_tokens (
                jti TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL,
                revoked_at TEXT NOT NULL,
                expires_at REAL NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            );
            """)

            cursor.execute("""
            CREATE TABLE IF NOT EXISTS security_audit_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                event_type TEXT NOT NULL,
                user_id INTEGER,
                email TEXT,
                ip_address TEXT NOT NULL,
                status TEXT NOT NULL,
                details TEXT NOT NULL,
                user_agent TEXT
            );
            """)

            conn.commit()
            self.logger.info("Security database tables initialized successfully.")
        except Exception as e:
            conn.rollback()
            self.logger.error(f"Failed to initialize security tables: {e}")
            raise
        finally:
            conn.close()

    def bootstrap_admin(self, admin_email: str = "admin@aurascan.ai", admin_pass: str = "Admin@123456") -> User:
        """Bootstraps a default System Administrator account if no admin exists."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM users WHERE role = ? OR email = ?;", (Role.ADMIN.value, admin_email))
            row = cursor.fetchone()
            if row:
                return self._row_to_user(row)

            now = datetime.datetime.utcnow().isoformat()
            user_uuid = str(uuid.uuid4())
            pass_hash = PasswordHasher.hash_password(admin_pass)

            cursor.execute("""
            INSERT INTO users (uuid, email, password_hash, full_name, role, is_verified, is_active, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, 1, 1, ?, ?);
            """, (user_uuid, admin_email, pass_hash, "System Administrator", Role.ADMIN.value, now, now))
            conn.commit()

            new_id = cursor.lastrowid
            self.logger.info(f"Bootstrapped default Admin user ({admin_email}) with ID: {new_id}")
            return User(
                id=new_id,
                uuid=user_uuid,
                email=admin_email,
                password_hash=pass_hash,
                full_name="System Administrator",
                role=Role.ADMIN,
                is_verified=True,
                is_active=True,
                created_at=now,
                updated_at=now,
            )
        finally:
            conn.close()

    def _row_to_user(self, row: sqlite3.Row) -> User:
        return User(
            id=row["id"],
            uuid=row["uuid"],
            email=row["email"],
            password_hash=row["password_hash"],
            full_name=row["full_name"],
            role=Role.from_string(row["role"]),
            is_verified=bool(row["is_verified"]) if "is_verified" in row.keys() else True,
            is_active=bool(row["is_active"]) if "is_active" in row.keys() else True,
            profile_pic=row["profile_pic"] if "profile_pic" in row.keys() else None,
            sessions_revoked_at=row["sessions_revoked_at"] if "sessions_revoked_at" in row.keys() else None,
            failed_login_attempts=row["failed_login_attempts"] if "failed_login_attempts" in row.keys() else 0,
            lockout_until=row["lockout_until"] if "lockout_until" in row.keys() else None,
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            last_login_at=row["last_login_at"],
        )

    def create_user(self, user: User) -> User:
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
            INSERT INTO users (uuid, email, password_hash, full_name, role, is_verified, is_active, profile_pic, sessions_revoked_at, failed_login_attempts, lockout_until, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """, (
                user.uuid,
                user.email.lower().strip(),
                user.password_hash,
                user.full_name.strip(),
                user.role.value if isinstance(user.role, Role) else user.role,
                1 if user.is_verified else 0,
                1 if user.is_active else 0,
                user.profile_pic,
                user.sessions_revoked_at,
                user.failed_login_attempts,
                user.lockout_until,
                user.created_at,
                user.updated_at,
            ))
            conn.commit()
            user.id = cursor.lastrowid
            return user
        finally:
            conn.close()

    def get_by_id(self, user_id: int) -> Optional[User]:
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM users WHERE id = ?;", (user_id,))
            row = cursor.fetchone()
            return self._row_to_user(row) if row else None
        finally:
            conn.close()

    def get_by_uuid(self, uuid_str: str) -> Optional[User]:
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM users WHERE uuid = ?;", (uuid_str,))
            row = cursor.fetchone()
            return self._row_to_user(row) if row else None
        finally:
            conn.close()

    def get_by_email(self, email: str) -> Optional[User]:
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM users WHERE LOWER(email) = ?;", (email.lower().strip(),))
            row = cursor.fetchone()
            return self._row_to_user(row) if row else None
        finally:
            conn.close()

    def update_user(self, user: User) -> User:
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            now = datetime.datetime.utcnow().isoformat()
            user.updated_at = now
            cursor.execute("""
            UPDATE users SET
                email = ?,
                password_hash = ?,
                full_name = ?,
                role = ?,
                is_verified = ?,
                is_active = ?,
                profile_pic = ?,
                sessions_revoked_at = ?,
                failed_login_attempts = ?,
                lockout_until = ?,
                updated_at = ?,
                last_login_at = ?
            WHERE id = ?;
            """, (
                user.email.lower().strip(),
                user.password_hash,
                user.full_name,
                user.role.value if isinstance(user.role, Role) else user.role,
                1 if user.is_verified else 0,
                1 if user.is_active else 0,
                user.profile_pic,
                user.sessions_revoked_at,
                user.failed_login_attempts,
                user.lockout_until,
                user.updated_at,
                user.last_login_at,
                user.id
            ))
            conn.commit()
            return user
        finally:
            conn.close()

    def delete_user(self, user_id: int) -> None:
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM users WHERE id = ?;", (user_id,))
            conn.commit()
        finally:
            conn.close()

    def list_users(self, limit: int = 100, offset: int = 0) -> List[User]:
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM users ORDER BY id ASC LIMIT ? OFFSET ?;", (limit, offset))
            rows = cursor.fetchall()
            return [self._row_to_user(row) for row in rows]
        finally:
            conn.close()

    def revoke_token(self, jti: str, user_id: int, expires_at: float) -> None:
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            now = datetime.datetime.utcnow().isoformat()
            cursor.execute("""
            INSERT OR REPLACE INTO revoked_tokens (jti, user_id, revoked_at, expires_at)
            VALUES (?, ?, ?, ?);
            """, (jti, user_id, now, expires_at))
            conn.commit()
        finally:
            conn.close()

    def is_token_revoked(self, jti: str) -> bool:
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT 1 FROM revoked_tokens WHERE jti = ?;", (jti,))
            return cursor.fetchone() is not None
        finally:
            conn.close()

    def log_security_event(self, event: SecurityAuditLog) -> None:
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
            INSERT INTO security_audit_logs (timestamp, event_type, user_id, email, ip_address, status, details, user_agent)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?);
            """, (
                event.timestamp,
                event.event_type,
                event.user_id,
                event.email,
                event.ip_address,
                event.status,
                event.details,
                event.user_agent,
            ))
            conn.commit()
        except Exception as e:
            self.logger.error(f"Failed to log security event: {e}")
        finally:
            conn.close()

    def get_security_audit_logs(self, limit: int = 100) -> List[Dict[str, Any]]:
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
            SELECT id, timestamp, event_type, user_id, email, ip_address, status, details, user_agent
            FROM security_audit_logs ORDER BY id DESC LIMIT ?;
            """, (limit,))
            rows = cursor.fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

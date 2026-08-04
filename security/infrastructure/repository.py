import sqlite3
import datetime
import uuid
import logging
from typing import Optional, List, Dict, Any
from security.domain.entities import User, Role, VerificationToken, OTPRecord, SecurityAuditLog, TokenType, OtpPurpose
from security.domain.interfaces import IUserRepository
from security.infrastructure.password import PasswordHasher


class SQLiteUserRepository(IUserRepository):
    """SQLite persistence layer for users, auth tokens, OTP codes, session revocations, and audit logs."""

    def __init__(self, db_path: str, logger: Optional[logging.Logger] = None) -> None:
        self.db_path = db_path
        self.logger = logger or logging.getLogger("security_repo")

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn

    def initialize_security_tables(self) -> None:
        """Creates security schema tables if they do not exist."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                uuid TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                full_name TEXT NOT NULL,
                role TEXT NOT NULL,
                is_verified INTEGER DEFAULT 0,
                is_active INTEGER DEFAULT 1,
                two_factor_enabled INTEGER DEFAULT 0,
                two_factor_secret TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                last_login_at TEXT
            );
            """)

            cursor.execute("""
            CREATE TABLE IF NOT EXISTS verification_tokens (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                token TEXT UNIQUE NOT NULL,
                token_type TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                used_at TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            );
            """)

            cursor.execute("""
            CREATE TABLE IF NOT EXISTS otp_codes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                otp_code TEXT NOT NULL,
                purpose TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                used_at TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
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
                details TEXT NOT NULL
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
            INSERT INTO users (uuid, email, password_hash, full_name, role, is_verified, is_active, two_factor_enabled, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, 1, 1, 0, ?, ?);
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
            is_verified=bool(row["is_verified"]),
            is_active=bool(row["is_active"]),
            two_factor_enabled=bool(row["two_factor_enabled"]),
            two_factor_secret=row["two_factor_secret"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            last_login_at=row["last_login_at"],
        )

    def create_user(self, user: User) -> User:
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
            INSERT INTO users (uuid, email, password_hash, full_name, role, is_verified, is_active, two_factor_enabled, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """, (
                user.uuid,
                user.email.lower().strip(),
                user.password_hash,
                user.full_name.strip(),
                user.role.value if isinstance(user.role, Role) else user.role,
                1 if user.is_verified else 0,
                1 if user.is_active else 0,
                1 if user.two_factor_enabled else 0,
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
                two_factor_enabled = ?,
                two_factor_secret = ?,
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
                1 if user.two_factor_enabled else 0,
                user.two_factor_secret,
                user.updated_at,
                user.last_login_at,
                user.id
            ))
            conn.commit()
            return user
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

    def save_verification_token(self, token_record: VerificationToken) -> VerificationToken:
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
            INSERT INTO verification_tokens (user_id, token, token_type, expires_at, used_at, created_at)
            VALUES (?, ?, ?, ?, ?, ?);
            """, (
                token_record.user_id,
                token_record.token,
                token_record.token_type.value if isinstance(token_record.token_type, TokenType) else token_record.token_type,
                token_record.expires_at,
                token_record.used_at,
                token_record.created_at,
            ))
            conn.commit()
            token_record.id = cursor.lastrowid
            return token_record
        finally:
            conn.close()

    def get_verification_token(self, token: str, token_type: TokenType) -> Optional[VerificationToken]:
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            t_type = token_type.value if isinstance(token_type, TokenType) else token_type
            cursor.execute("SELECT * FROM verification_tokens WHERE token = ? AND token_type = ?;", (token, t_type))
            row = cursor.fetchone()
            if not row:
                return None
            return VerificationToken(
                id=row["id"],
                user_id=row["user_id"],
                token=row["token"],
                token_type=TokenType(row["token_type"]),
                expires_at=row["expires_at"],
                used_at=row["used_at"],
                created_at=row["created_at"],
            )
        finally:
            conn.close()

    def mark_token_used(self, token_id: int) -> None:
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            now = datetime.datetime.utcnow().isoformat()
            cursor.execute("UPDATE verification_tokens SET used_at = ? WHERE id = ?;", (now, token_id))
            conn.commit()
        finally:
            conn.close()

    def save_otp(self, otp_record: OTPRecord) -> OTPRecord:
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
            INSERT INTO otp_codes (user_id, otp_code, purpose, expires_at, used_at, created_at)
            VALUES (?, ?, ?, ?, ?, ?);
            """, (
                otp_record.user_id,
                otp_record.otp_code,
                otp_record.purpose.value if isinstance(otp_record.purpose, OtpPurpose) else otp_record.purpose,
                otp_record.expires_at,
                otp_record.used_at,
                otp_record.created_at,
            ))
            conn.commit()
            otp_record.id = cursor.lastrowid
            return otp_record
        finally:
            conn.close()

    def get_latest_otp(self, user_id: int, purpose: OtpPurpose) -> Optional[OTPRecord]:
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            p_str = purpose.value if isinstance(purpose, OtpPurpose) else purpose
            cursor.execute("""
            SELECT * FROM otp_codes WHERE user_id = ? AND purpose = ? AND used_at IS NULL
            ORDER BY id DESC LIMIT 1;
            """, (user_id, p_str))
            row = cursor.fetchone()
            if not row:
                return None
            return OTPRecord(
                id=row["id"],
                user_id=row["user_id"],
                otp_code=row["otp_code"],
                purpose=OtpPurpose(row["purpose"]),
                expires_at=row["expires_at"],
                used_at=row["used_at"],
                created_at=row["created_at"],
            )
        finally:
            conn.close()

    def mark_otp_used(self, otp_id: int) -> None:
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            now = datetime.datetime.utcnow().isoformat()
            cursor.execute("UPDATE otp_codes SET used_at = ? WHERE id = ?;", (now, otp_id))
            conn.commit()
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
            INSERT INTO security_audit_logs (timestamp, event_type, user_id, email, ip_address, status, details)
            VALUES (?, ?, ?, ?, ?, ?, ?);
            """, (
                event.timestamp,
                event.event_type,
                event.user_id,
                event.email,
                event.ip_address,
                event.status,
                event.details,
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
            SELECT id, timestamp, event_type, user_id, email, ip_address, status, details
            FROM security_audit_logs ORDER BY id DESC LIMIT ?;
            """, (limit,))
            rows = cursor.fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

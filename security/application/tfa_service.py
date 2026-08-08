import secrets
import hashlib
import sqlite3
import datetime
import logging

logger = logging.getLogger("tfa_service")

class TFAService:
    """Service to handle Two-Factor Authentication OTP and recovery codes."""

    @staticmethod
    def generate_otp() -> str:
        """Generates a secure 6-digit numeric OTP."""
        # Using secrets module for cryptographically secure random numbers
        return f"{secrets.randbelow(1000000):06d}"

    @staticmethod
    def save_otp(db_path: str, user_id: int, otp_code: str, expires_in_seconds: int = 300) -> None:
        """Saves the OTP code to database, replacing any existing active OTP for the user."""
        now = datetime.datetime.utcnow()
        expires_at = (now + datetime.timedelta(seconds=expires_in_seconds)).isoformat()
        created_at = now.isoformat()

        conn = sqlite3.connect(db_path)
        try:
            with conn:
                conn.execute("""
                    INSERT OR REPLACE INTO otp_codes (user_id, otp_code, expires_at, created_at)
                    VALUES (?, ?, ?, ?);
                """, (user_id, otp_code, expires_at, created_at))
        finally:
            conn.close()

    @staticmethod
    def verify_otp(db_path: str, user_id: int, input_otp_code: str) -> bool:
        """
        Verifies the user's OTP code.
        If valid and not expired, returns True and deletes the code to prevent reuse (single-use constraint).
        """
        if not input_otp_code:
            return False

        conn = sqlite3.connect(db_path)
        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT otp_code, expires_at FROM otp_codes WHERE user_id = ?;
            """, (user_id,))
            row = cursor.fetchone()
            if not row:
                return False

            stored_otp, expires_at_str = row
            
            # Check expiration
            expires_at = datetime.datetime.fromisoformat(expires_at_str)
            if datetime.datetime.utcnow() > expires_at:
                # Delete expired OTP
                with conn:
                    conn.execute("DELETE FROM otp_codes WHERE user_id = ?;", (user_id,))
                return False

            # Constant-time comparison
            is_valid = secrets.compare_digest(stored_otp.strip(), input_otp_code.strip())
            
            if is_valid:
                # Delete on use
                with conn:
                    conn.execute("DELETE FROM otp_codes WHERE user_id = ?;", (user_id,))
                return True
            
            return False
        finally:
            conn.close()

    @staticmethod
    def generate_recovery_codes() -> list[str]:
        """Generates a list of 5 alphanumeric recovery codes."""
        codes = []
        for _ in range(5):
            # Format: XXXX-XXXX
            part1 = secrets.token_hex(2).upper()
            part2 = secrets.token_hex(2).upper()
            codes.append(f"{part1}-{part2}")
        return codes

    @staticmethod
    def hash_recovery_codes(codes: list[str]) -> str:
        """Hashes a list of recovery codes using SHA-256 and returns a comma-separated string."""
        hashed_list = []
        for c in codes:
            # Hash codes cleanly
            hashed_c = hashlib.sha256(c.strip().upper().encode("utf-8")).hexdigest()
            hashed_list.append(hashed_c)
        return ",".join(hashed_list)

    @classmethod
    def verify_and_consume_recovery_code(cls, user_repo, db_user, input_code: str) -> bool:
        """
        Verifies if input_code matches one of the hashed recovery codes for db_user.
        If it matches, consumes (removes) it from the list, updates the DB user record, and returns True.
        """
        if not input_code or not db_user.two_factor_recovery_codes:
            return False

        input_hash = hashlib.sha256(input_code.strip().upper().encode("utf-8")).hexdigest()
        stored_hashes = [h.strip() for h in db_user.two_factor_recovery_codes.split(",") if h.strip()]

        for idx, stored_hash in enumerate(stored_hashes):
            if secrets.compare_digest(stored_hash, input_hash):
                # Consume this code by removing it from the list
                stored_hashes.pop(idx)
                db_user.two_factor_recovery_codes = ",".join(stored_hashes)
                user_repo.update_user(db_user)
                return True

        return False

    @staticmethod
    def send_otp_via_email(email: str, otp_code: str) -> None:
        """Simulates sending the OTP code via email by writing it to secure logs."""
        logger.info(f"[EMAIL SEND SUCCESS] Secure 2FA verification code sent to {email}. Code: {otp_code}")
        # Print for terminal-based validation during manual checks
        print(f"\n>>> [2FA OTP Dispatch] Sent OTP to {email}: {otp_code} <<<\n")

    @staticmethod
    def parse_user_agent(user_agent_str: str) -> tuple[str, str]:
        """Returns (browser, device) from User-Agent string."""
        if not user_agent_str:
            return "Unknown Browser", "Unknown Device"
        
        ua = user_agent_str.lower()
        
        # Parse Browser
        if "edg" in ua:
            browser = "Microsoft Edge"
        elif "chrome" in ua:
            browser = "Google Chrome"
        elif "safari" in ua and "firefox" not in ua:
            browser = "Safari"
        elif "firefox" in ua:
            browser = "Mozilla Firefox"
        elif "opera" in ua or "opr" in ua:
            browser = "Opera"
        elif "msie" in ua or "trident" in ua:
            browser = "Internet Explorer"
        else:
            browser = "Other Browser"
            
        # Parse Device
        if "ipad" in ua:
            device = "Tablet (iPad)"
        elif "android" in ua:
            if "mobile" in ua:
                device = "Mobile (Android)"
            else:
                device = "Tablet (Android)"
        elif "iphone" in ua:
            device = "Mobile (iPhone)"
        elif "mobile" in ua:
            device = "Mobile Device"
        else:
            device = "Desktop"
            
        return browser, device

    @staticmethod
    def get_location_from_ip(ip: str) -> str:
        """Resolves geolocation info for a given IP address, with loopback network checks and offline timeouts."""
        if not ip or ip in ("127.0.0.1", "localhost", "::1"):
            return "Localhost"
        if ip.startswith("192.168.") or ip.startswith("10.") or ip.startswith("172."):
            return "Local Network"
            
        try:
            import urllib.request
            import json
            # Fetch geolocation from public ip-api (free, no key required)
            url = f"http://ip-api.com/json/{ip}?fields=status,country,regionName,city"
            with urllib.request.urlopen(url, timeout=1.0) as response:
                data = json.loads(response.read().decode())
                if data.get("status") == "success":
                    city = data.get("city", "")
                    region = data.get("regionName", "")
                    country = data.get("country", "")
                    parts = [p for p in (city, region, country) if p]
                    return ", ".join(parts)
        except Exception:
            pass
        return "Unknown Location"

from datetime import datetime, timezone
import datetime as dt_module
from zoneinfo import ZoneInfo

def get_current_ist_time():
    """Returns the current datetime in Asia/Kolkata timezone."""
    return datetime.now(ZoneInfo("Asia/Kolkata"))


def convert_utc_to_ist(timestamp_str: str) -> str:
    """Converts a UTC timestamp string (naive, Z, or explicit offset) to an Asia/Kolkata ISO timestamp string.
    If input is empty/None/invalid, returns it as-is (safe fallback).
    """
    if not timestamp_str:
        return timestamp_str

    try:
        ts_str = str(timestamp_str).strip()
        if not ts_str:
            return timestamp_str

        if ts_str.endswith("Z"):
            ts_str = ts_str[:-1] + "+00:00"

        dt = datetime.fromisoformat(ts_str)
        if dt.tzinfo is None:
            # naive timestamp is assumed to be UTC
            dt = dt.replace(tzinfo=timezone.utc)

        # Convert to Asia/Kolkata
        ist_tz = ZoneInfo("Asia/Kolkata")
        ist_dt = dt.astimezone(ist_tz)
        return ist_dt.isoformat()
    except Exception:
        return timestamp_str


def format_last_login(timestamp_str: str) -> str:
    """Formats raw UTC timestamp string to a human friendly relative time format in Asia/Kolkata timezone."""
    if not timestamp_str:
        return "Never"

    try:
        ts_str = str(timestamp_str).strip()
        if not ts_str:
            return "Never"

        if ts_str.endswith("Z"):
            ts_str = ts_str[:-1] + "+00:00"

        dt = datetime.fromisoformat(ts_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)

        ist_dt = dt.astimezone(ZoneInfo("Asia/Kolkata"))

        now_ist = get_current_ist_time()
        today_ist = now_ist.date()
        yesterday_ist = today_ist - dt_module.timedelta(days=1)

        dt_date = ist_dt.date()
        time_part = ist_dt.strftime("%I:%M %p").lstrip('0')

        if dt_date == today_ist:
            return f"Today • {time_part}"
        elif dt_date == yesterday_ist:
            return f"Yesterday • {time_part}"
        else:
            return f"{ist_dt.strftime('%d %b %Y')} • {time_part}"
    except Exception:
        return str(timestamp_str)

import time
from collections import defaultdict
from typing import Dict, List, Tuple


class RateLimiter:
    """Sliding window rate limiter for brute-force protection (OWASP compliant)."""

    def __init__(self):
        # Maps key -> list of timestamps
        self._requests: Dict[str, List[float]] = defaultdict(list)

    def is_rate_limited(self, key: str, max_requests: int = 5, window_seconds: int = 300) -> Tuple[bool, int]:
        """Checks if key (e.g. "login:ip" or "otp:user_id") exceeded rate limit.

        Returns:
            Tuple[is_limited: bool, remaining_seconds_to_wait: int]
        """
        now = time.time()
        window_start = now - window_seconds

        # Clean old records
        timestamps = [t for t in self._requests[key] if t > window_start]
        self._requests[key] = timestamps

        if len(timestamps) >= max_requests:
            oldest_in_window = timestamps[0]
            remaining = int((oldest_in_window + window_seconds) - now) + 1
            return True, max(remaining, 1)

        self._requests[key].append(now)
        return False, 0

    def reset_key(self, key: str) -> None:
        """Resets rate limit counter for a specific key (e.g., after successful login)."""
        if key in self._requests:
            del self._requests[key]


# Global rate limiter instance
global_rate_limiter = RateLimiter()

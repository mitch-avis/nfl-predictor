"""A minimal in-memory login rate limiter."""

from __future__ import annotations

import threading
import time
from collections import deque


class LoginRateLimiter:
    """Allow at most ``max_attempts`` failed logins per key within ``window_seconds``."""

    def __init__(self, max_attempts: int = 10, window_seconds: float = 300.0) -> None:
        """Configure the attempt budget and window."""
        self.max_attempts = max_attempts
        self.window_seconds = window_seconds
        self._attempts: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def _prune(self, key: str, now: float) -> deque[float]:
        """Drop attempts older than the window and return the remaining ones."""
        attempts = self._attempts.setdefault(key, deque())
        while attempts and now - attempts[0] > self.window_seconds:
            attempts.popleft()
        return attempts

    def is_blocked(self, key: str, now: float | None = None) -> bool:
        """Return whether ``key`` has exhausted its budget."""
        now = time.monotonic() if now is None else now
        with self._lock:
            return len(self._prune(key, now)) >= self.max_attempts

    def record_failure(self, key: str, now: float | None = None) -> None:
        """Record a failed attempt for ``key``."""
        now = time.monotonic() if now is None else now
        with self._lock:
            self._prune(key, now).append(now)

    def reset(self, key: str) -> None:
        """Forget every attempt for ``key`` (after a successful login)."""
        with self._lock:
            self._attempts.pop(key, None)

import asyncio
from typing import Any, Dict, Optional


class TokenTracker:
    """Thread-safe and async-safe token usage and API request accumulator."""

    def __init__(self, name: str = "global"):
        self.name = name
        self._prompt_tokens: int = 0
        self._candidate_tokens: int = 0
        self._total_tokens: int = 0
        self._cached_tokens: int = 0
        self._request_count: int = 0
        self._failed_request_count: int = 0
        self._lock = asyncio.Lock()

    async def add_usage(
        self,
        prompt_tokens: int = 0,
        candidate_tokens: int = 0,
        total_tokens: int = 0,
        cached_tokens: int = 0,
    ) -> None:
        """Accumulate token counts from a successful model response."""
        async with self._lock:
            self._prompt_tokens += max(0, prompt_tokens)
            self._candidate_tokens += max(0, candidate_tokens)
            # If total_tokens is provided by API, use it, else sum
            if total_tokens > 0:
                self._total_tokens += total_tokens
            else:
                self._total_tokens += max(0, prompt_tokens) + max(0, candidate_tokens)
            self._cached_tokens += max(0, cached_tokens)
            self._request_count += 1

    def add_usage_sync(
        self,
        prompt_tokens: int = 0,
        candidate_tokens: int = 0,
        total_tokens: int = 0,
        cached_tokens: int = 0,
    ) -> None:
        """Synchronous version for non-async callers."""
        self._prompt_tokens += max(0, prompt_tokens)
        self._candidate_tokens += max(0, candidate_tokens)
        if total_tokens > 0:
            self._total_tokens += total_tokens
        else:
            self._total_tokens += max(0, prompt_tokens) + max(0, candidate_tokens)
        self._cached_tokens += max(0, cached_tokens)
        self._request_count += 1

    async def record_failure(self) -> None:
        """Record a failed API request."""
        async with self._lock:
            self._failed_request_count += 1

    def record_failure_sync(self) -> None:
        """Record a failed API request synchronously."""
        self._failed_request_count += 1

    async def get_summary(self) -> Dict[str, Any]:
        """Return the current token usage summary."""
        async with self._lock:
            return self._build_dict()

    def get_summary_sync(self) -> Dict[str, Any]:
        """Return current summary synchronously."""
        return self._build_dict()

    def _build_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "prompt_tokens": self._prompt_tokens,
            "candidate_tokens": self._candidate_tokens,
            "total_tokens": self._total_tokens,
            "cached_tokens": self._cached_tokens,
            "successful_requests": self._request_count,
            "failed_requests": self._failed_request_count,
            "total_requests": self._request_count + self._failed_request_count,
        }

    async def reset(self) -> None:
        """Reset all tracked counters."""
        async with self._lock:
            self._prompt_tokens = 0
            self._candidate_tokens = 0
            self._total_tokens = 0
            self._cached_tokens = 0
            self._request_count = 0
            self._failed_request_count = 0


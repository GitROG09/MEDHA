import asyncio
import random
from typing import Any, Callable, Coroutine, Optional, Tuple, TypeVar

from config.settings import INITIAL_RETRY_DELAY, MAX_RETRIES, MAX_RETRY_DELAY
from utils.logger import get_logger

logger = get_logger("retry")

T = TypeVar("T")


class LLMError(Exception):
    """Base exception for LLM operations."""

    def __init__(self, message: str, status_code: Optional[int] = None, retryable: bool = False):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.retryable = retryable


class LLMClientError(LLMError):
    """Non-retryable 400 Bad Request error."""

    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message, status_code=status_code, retryable=False)


class LLMAuthError(LLMError):
    """Non-retryable 401/403 Authentication or authorization error."""

    def __init__(self, message: str, status_code: int = 401):
        super().__init__(message, status_code=status_code, retryable=False)


class LLMModelNotFoundError(LLMError):
    """Non-retryable 404 Model not found or unavailable error."""

    def __init__(self, message: str, status_code: int = 404):
        super().__init__(message, status_code=status_code, retryable=False)


class LLMRateLimitError(LLMError):
    """Retryable 429 Rate limit or quota exhaustion error."""

    def __init__(self, message: str, status_code: int = 429):
        super().__init__(message, status_code=status_code, retryable=True)


class LLMServerError(LLMError):
    """Retryable 500/503 Temporary server error."""

    def __init__(self, message: str, status_code: int = 500):
        super().__init__(message, status_code=status_code, retryable=True)


class LLMTimeoutError(LLMError):
    """Retryable network or request timeout error."""

    def __init__(self, message: str):
        super().__init__(message, status_code=408, retryable=True)


def extract_status_code(exc: Exception) -> Optional[int]:
    """Attempt to extract an HTTP status code from various exception types."""
    # Direct status_code or code attribute
    for attr in ("code", "status_code", "status"):
        val = getattr(exc, attr, None)
        if isinstance(val, int):
            return val
        if isinstance(val, str) and val.isdigit():
            return int(val)

    # Check response attribute (e.g., httpx.HTTPStatusError)
    response = getattr(exc, "response", None)
    if response is not None and hasattr(response, "status_code"):
        return getattr(response, "status_code", None)

    # Check error message strings for known status codes
    msg = str(exc)
    for code in [400, 401, 403, 404, 429, 500, 503, 504]:
        if f" {code} " in f" {msg} ":
            return code

    return None


def classify_error(exc: Exception) -> Tuple[bool, LLMError]:
    """Classify an exception into specific LLM error categories and determine retryability."""
    # Check if already classified
    if isinstance(exc, LLMError):
        return exc.retryable, exc

    # Check for timeout exceptions
    if isinstance(exc, (asyncio.TimeoutError, TimeoutError)):
        return True, LLMTimeoutError(f"Request timed out: {exc}")

    exc_name = exc.__class__.__name__.lower()
    if "timeout" in exc_name:
        return True, LLMTimeoutError(f"Network timeout: {exc}")

    code = extract_status_code(exc)
    msg = str(exc)

    if code == 400:
        return False, LLMClientError(f"Invalid request (400): {msg}", status_code=400)
    elif code in (401, 403):
        return False, LLMAuthError(f"Authentication/Permission error ({code}): {msg}", status_code=code)
    elif code == 404:
        return False, LLMModelNotFoundError(f"Model not found or unavailable (404): {msg}", status_code=404)
    elif code == 429:
        return True, LLMRateLimitError(f"Rate limit exceeded (429): {msg}", status_code=429)
    elif code in (500, 502, 503, 504):
        return True, LLMServerError(f"Server failure ({code}): {msg}", status_code=code)

    # Check message for rate limit clues
    lower_msg = msg.lower()
    if "resource exhausted" in lower_msg or "rate limit" in lower_msg or "quota" in lower_msg:
        return True, LLMRateLimitError(f"Rate limit / Quota exhausted: {msg}", status_code=429)

    if "unavailable" in lower_msg or "internal" in lower_msg:
        return True, LLMServerError(f"Server error: {msg}", status_code=500)

    # Default: non-retryable generic error
    return False, LLMError(f"Unclassified LLM error: {msg}", status_code=code, retryable=False)


async def execute_with_retry(
    coro_fn: Callable[[], Coroutine[Any, Any, T]],
    max_retries: int = MAX_RETRIES,
    initial_delay: float = INITIAL_RETRY_DELAY,
    max_delay: float = MAX_RETRY_DELAY,
    timeout_seconds: Optional[float] = None,
) -> T:
    """Execute an async operation with exponential backoff, random jitter, and error classification."""
    attempt = 0

    while True:
        try:
            if timeout_seconds and timeout_seconds > 0:
                return await asyncio.wait_for(coro_fn(), timeout=timeout_seconds)
            else:
                return await coro_fn()

        except Exception as exc:
            attempt += 1
            is_retryable, classified_err = classify_error(exc)

            if not is_retryable or attempt > max_retries:
                logger.error(
                    f"Execution failed (attempt {attempt}/{max_retries + 1}, "
                    f"retryable={is_retryable}): {classified_err}"
                )
                raise classified_err from exc

            # Calculate delay with exponential backoff and full jitter
            base_backoff = min(max_delay, initial_delay * (2 ** (attempt - 1)))
            jitter = random.uniform(0, 0.5 * base_backoff)
            delay = base_backoff + jitter

            logger.warning(
                f"Retryable error on attempt {attempt}/{max_retries + 1} "
                f"({classified_err.__class__.__name__}: {classified_err}). "
                f"Backing off for {delay:.2f}s..."
            )
            await asyncio.sleep(delay)


import asyncio
from unittest.mock import AsyncMock, MagicMock
import pytest
from pydantic import BaseModel

from benchmark.baseline import SingleAgentBaseline
from core.llm_client import GeminiClient
from utils.retry import LLMAuthError, LLMClientError, LLMRateLimitError, LLMServerError
from utils.token_tracker import TokenTracker


class SampleOutput(BaseModel):
    title: str
    score: int


def _create_mock_response(text: str, prompt_tokens: int = 10, candidates_tokens: int = 20):
    response = MagicMock()
    response.text = text
    response.parsed = None
    usage = MagicMock()
    usage.prompt_token_count = prompt_tokens
    usage.candidates_token_count = candidates_tokens
    usage.total_token_count = prompt_tokens + candidates_tokens
    usage.cached_content_token_count = 0
    response.usage_metadata = usage
    return response


@pytest.mark.asyncio
async def test_generate_text_success():
    """Verify generate_text calls client and updates token tracker."""
    tracker = TokenTracker(name="test")
    mock_client = MagicMock()
    mock_client.aio.models.generate_content = AsyncMock(
        return_value=_create_mock_response("Hello from Gemini", prompt_tokens=15, candidates_tokens=25)
    )

    client = GeminiClient(api_key="mock_key", client=mock_client, token_tracker=tracker)
    result = await client.generate_text("Hi")

    assert result == "Hello from Gemini"
    summary = await tracker.get_summary()
    assert summary["prompt_tokens"] == 15
    assert summary["candidate_tokens"] == 25
    assert summary["total_tokens"] == 40
    assert summary["successful_requests"] == 1


@pytest.mark.asyncio
async def test_generate_structured_success():
    """Verify generate_structured parses Pydantic model correctly."""
    tracker = TokenTracker(name="test")
    mock_client = MagicMock()
    mock_client.aio.models.generate_content = AsyncMock(
        return_value=_create_mock_response('{"title": "Test Plan", "score": 95}')
    )

    client = GeminiClient(api_key="mock_key", client=mock_client, token_tracker=tracker)
    result = await client.generate_structured("Generate plan", response_schema=SampleOutput)

    assert isinstance(result, SampleOutput)
    assert result.title == "Test Plan"
    assert result.score == 95


@pytest.mark.asyncio
async def test_non_retryable_auth_error():
    """Verify 401 error is non-retryable and fails immediately."""
    tracker = TokenTracker(name="test")
    mock_client = MagicMock()

    class MockAPIError(Exception):
        code = 401
        message = "Unauthorized"

    mock_client.aio.models.generate_content = AsyncMock(side_effect=MockAPIError("Invalid API key"))

    client = GeminiClient(api_key="bad_key", client=mock_client, token_tracker=tracker)

    with pytest.raises(LLMAuthError) as exc_info:
        await client.generate_text("test", max_retries=3)

    assert exc_info.value.status_code == 401
    assert mock_client.aio.models.generate_content.call_count == 1
    summary = await tracker.get_summary()
    assert summary["failed_requests"] == 1


@pytest.mark.asyncio
async def test_retryable_rate_limit_recovers():
    """Verify 429 error retries and succeeds if subsequent attempt succeeds."""
    tracker = TokenTracker(name="test")
    mock_client = MagicMock()

    class MockRateLimitError(Exception):
        code = 429
        message = "Resource exhausted"

    success_resp = _create_mock_response("Success after retry", prompt_tokens=5, candidates_tokens=5)

    mock_client.aio.models.generate_content = AsyncMock(
        side_effect=[MockRateLimitError("Quota exceeded"), success_resp]
    )

    client = GeminiClient(api_key="mock_key", client=mock_client, token_tracker=tracker)
    result = await client.generate_text("test", max_retries=2)

    assert result == "Success after retry"
    assert mock_client.aio.models.generate_content.call_count == 2
    summary = await tracker.get_summary()
    assert summary["successful_requests"] == 1


@pytest.mark.asyncio
async def test_single_agent_baseline_run():
    """Verify SingleAgentBaseline runs and records output and latency."""
    mock_client = MagicMock()
    mock_client.aio.models.generate_content = AsyncMock(
        return_value=_create_mock_response("Architecture baseline answer", 10, 20)
    )
    gemini_client = GeminiClient(api_key="mock_key", client=mock_client)
    baseline = SingleAgentBaseline(llm_client=gemini_client)

    result = await baseline.run("Design an architecture")

    assert result.success is True
    assert result.answer == "Architecture baseline answer"
    assert result.duration_seconds >= 0.0
    assert result.model is not None


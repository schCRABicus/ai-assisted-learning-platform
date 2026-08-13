"""Tests for the centralized LLM module (retry-with-backoff, rate limiting)."""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from pydantic import BaseModel

from agentic_learning_portal.api.llm import (
    LLMResponseError,
    LLMRetryableError,
    LLMUnavailableError,
    RateLimiter,
    retry,
    ask_ai_for_text_response,
    ask_ai_for_structured_response,
)

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"


class _Answer(BaseModel):
    value: str


def _mock_httpx_client() -> MagicMock:
    """Return a MagicMock standing in for ``httpx.AsyncClient``."""
    mock_client = MagicMock()
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__ = AsyncMock(return_value=False)
    return mock_client


def _json_response(content: str) -> MagicMock:
    fake_response = MagicMock()
    fake_response.status_code = 200
    fake_response.json.return_value = {"choices": [{"message": {"content": content}}]}
    return fake_response


# --- RateLimiter --------------------------------------------------------------


@pytest.mark.asyncio
async def test_rate_limiter_rejects_non_positive_rate() -> None:
    with pytest.raises(ValueError):
        RateLimiter(rate=0)
    with pytest.raises(ValueError):
        RateLimiter(rate=5, period=0)


@pytest.mark.asyncio
async def test_rate_limiter_allows_burst_then_throttles() -> None:
    limiter = RateLimiter(rate=2, period=0.05)
    await limiter.acquire()
    await limiter.acquire()  # burst of 2 consumed immediately

    start = time.monotonic()
    await limiter.acquire()  # must wait ~ (1/2) * 0.05 = 25ms
    elapsed = time.monotonic() - start

    assert elapsed >= 0.01


@pytest.mark.asyncio
async def test_rate_limiter_serializes_concurrent_acquirers() -> None:
    limiter = RateLimiter(rate=2, period=0.05)
    start = time.monotonic()

    async def _acquire() -> None:
        await limiter.acquire()

    # 4 tokens at 2/sec -> the 4th must wait at least one refill interval.
    await asyncio.gather(*(_acquire() for _ in range(4)))
    elapsed = time.monotonic() - start

    assert elapsed >= 0.02


# --- retry --------------------------------------------------------------------


@pytest.mark.asyncio
async def test_retry_retries_then_succeeds() -> None:
    calls = 0

    async def _flaky() -> str:
        nonlocal calls
        calls += 1
        if calls < 3:
            raise httpx.ConnectError("boom")
        return "ok"

    with patch("agentic_learning_portal.api.llm.asyncio.sleep", new=AsyncMock()) as mock_sleep:
        result = await retry(_flaky, retries=5)

    assert result == "ok"
    assert calls == 3
    mock_sleep.assert_awaited()


@pytest.mark.asyncio
async def test_retry_gives_up_after_retries() -> None:
    async def _always_fails() -> str:
        raise LLMRetryableError("HTTP 429")

    with patch("agentic_learning_portal.api.llm.asyncio.sleep", new=AsyncMock()):
        with pytest.raises(LLMRetryableError, match="429"):
            await retry(_always_fails, retries=2)


@pytest.mark.asyncio
async def test_retry_does_not_retry_non_transient() -> None:
    async def _bad() -> str:
        raise ValueError("not transient")

    with patch("agentic_learning_portal.api.llm.asyncio.sleep", new=AsyncMock()) as mock_sleep:
        with pytest.raises(ValueError, match="not transient"):
            await retry(_bad, retries=3)

    mock_sleep.assert_not_awaited()


# --- ask_ai_for_structured_response -----------------------------------------------------------


@pytest.mark.asyncio
async def test_ask_ai_for_structured_response_returns_typed_output() -> None:
    with patch("agentic_learning_portal.api.llm.Agent") as mock_agent:
        mock_agent.return_value.run = AsyncMock(
            return_value=SimpleNamespace(output=_Answer(value="ok"))
        )
        result = await ask_ai_for_structured_response(
            model="google:gemini-3.5-flash",
            output_type=_Answer,
            system_prompt="sys",
            user_prompt="usr",
            limiter=None,
        )

    assert isinstance(result, _Answer)
    assert result.value == "ok"
    mock_agent.assert_called_once_with(
        model="google:gemini-3.5-flash",
        output_type=_Answer,
        system_prompt="sys",
    )
    mock_agent.return_value.run.assert_awaited_once_with("usr")


@pytest.mark.asyncio
async def test_ask_ai_for_structured_response_retries_transient_then_succeeds() -> None:
    with (
        patch("agentic_learning_portal.api.llm.Agent") as mock_agent,
        patch("agentic_learning_portal.api.llm.asyncio.sleep", new=AsyncMock()) as mock_sleep,
    ):
        mock_agent.return_value.run = AsyncMock(
            side_effect=[httpx.ConnectError("boom"), SimpleNamespace(output=_Answer(value="ok"))]
        )
        result = await ask_ai_for_structured_response(
            model="m",
            output_type=_Answer,
            system_prompt="sys",
            user_prompt="usr",
            retries=3,
            limiter=None,
        )

    assert result.value == "ok"
    assert mock_agent.return_value.run.await_count == 2
    mock_sleep.assert_awaited()


@pytest.mark.asyncio
async def test_ask_ai_for_structured_response_acquires_limiter() -> None:
    limiter = SimpleNamespace(acquire=AsyncMock())
    with patch("agentic_learning_portal.api.llm.Agent") as mock_agent:
        mock_agent.return_value.run = AsyncMock(
            return_value=SimpleNamespace(output=_Answer(value="ok"))
        )
        await ask_ai_for_structured_response("m", _Answer, "sys", "usr", limiter=limiter)

    limiter.acquire.assert_awaited_once()


# --- ask_ai_for_text_response ------------------------------------------------------


@pytest.mark.asyncio
async def test_ask_ai_for_text_response_returns_content() -> None:
    with patch("agentic_learning_portal.api.llm.httpx.AsyncClient") as mock_cls:
        mock_client = _mock_httpx_client()
        mock_cls.return_value = mock_client
        mock_client.post = AsyncMock(return_value=_json_response("\n\n12"))

        result = await ask_ai_for_text_response(
            GROQ_URL,
            api_key="key",
            model="qwen/qwen3.6-27b",
            messages=[{"role": "user", "content": "hi"}],
            limiter=None,
        )

    assert result == "12"
    mock_client.post.assert_awaited_once_with(
        GROQ_URL,
        json={
            "model": "qwen/qwen3.6-27b",
            "messages": [{"role": "user", "content": "hi"}],
            "temperature": 0,
        },
        headers={"Authorization": "Bearer key"},
    )


@pytest.mark.asyncio
async def test_ask_ai_for_text_response_retries_rate_limit_then_succeeds() -> None:
    rate_limited = MagicMock()
    rate_limited.status_code = 429

    with (
        patch("agentic_learning_portal.api.llm.httpx.AsyncClient") as mock_cls,
        patch("agentic_learning_portal.api.llm.asyncio.sleep", new=AsyncMock()) as mock_sleep,
    ):
        mock_client = _mock_httpx_client()
        mock_cls.return_value = mock_client
        mock_client.post = AsyncMock(side_effect=[rate_limited, _json_response("12")])

        result = await ask_ai_for_text_response(
            GROQ_URL,
            api_key="key",
            model="m",
            messages=[{"role": "user", "content": "hi"}],
            retries=3,
            limiter=None,
        )

    assert result == "12"
    assert mock_client.post.await_count == 2
    mock_sleep.assert_awaited()


@pytest.mark.asyncio
async def test_ask_ai_for_text_response_raises_unavailable_after_persistent_rate_limit() -> None:
    rate_limited = MagicMock()
    rate_limited.status_code = 429

    with (
        patch("agentic_learning_portal.api.llm.httpx.AsyncClient") as mock_cls,
        patch("agentic_learning_portal.api.llm.asyncio.sleep", new=AsyncMock()),
    ):
        mock_client = _mock_httpx_client()
        mock_cls.return_value = mock_client
        mock_client.post = AsyncMock(return_value=rate_limited)

        with pytest.raises(LLMUnavailableError):
            await ask_ai_for_text_response(
                GROQ_URL,
                api_key="key",
                model="m",
                messages=[{"role": "user", "content": "hi"}],
                retries=2,
                limiter=None,
            )

    assert mock_client.post.await_count == 3  # initial + 2 retries


@pytest.mark.asyncio
async def test_ask_ai_for_text_response_raises_unavailable_after_persistent_transport_error() -> None:
    with (
        patch("agentic_learning_portal.api.llm.httpx.AsyncClient") as mock_cls,
        patch("agentic_learning_portal.api.llm.asyncio.sleep", new=AsyncMock()),
    ):
        mock_client = _mock_httpx_client()
        mock_cls.return_value = mock_client
        mock_client.post = AsyncMock(side_effect=httpx.ConnectError("boom"))

        with pytest.raises(LLMUnavailableError):
            await ask_ai_for_text_response(
                GROQ_URL,
                api_key="key",
                model="m",
                messages=[{"role": "user", "content": "hi"}],
                retries=2,
                limiter=None,
            )


@pytest.mark.asyncio
async def test_ask_ai_for_text_response_raises_response_error_on_malformed_shape() -> None:
    fake_response = MagicMock()
    fake_response.status_code = 200
    fake_response.json.return_value = {"unexpected": "shape"}

    with patch("agentic_learning_portal.api.llm.httpx.AsyncClient") as mock_cls:
        mock_client = _mock_httpx_client()
        mock_cls.return_value = mock_client
        mock_client.post = AsyncMock(return_value=fake_response)

        with pytest.raises(LLMResponseError):
            await ask_ai_for_text_response(
                GROQ_URL,
                api_key="key",
                model="m",
                messages=[{"role": "user", "content": "hi"}],
                limiter=None,
            )


@pytest.mark.asyncio
async def test_ask_ai_for_text_response_propagates_non_retryable_http_error() -> None:
    forbidden = httpx.Response(403, request=httpx.Request("POST", GROQ_URL))

    with patch("agentic_learning_portal.api.llm.httpx.AsyncClient") as mock_cls:
        mock_client = _mock_httpx_client()
        mock_cls.return_value = mock_client
        mock_client.post = AsyncMock(return_value=forbidden)

        with pytest.raises(httpx.HTTPStatusError):
            await ask_ai_for_text_response(
                GROQ_URL,
                api_key="key",
                model="m",
                messages=[{"role": "user", "content": "hi"}],
                retries=2,
                limiter=None,
            )

    assert mock_client.post.await_count == 1  # non-transient -> no retry


@pytest.mark.asyncio
async def test_ask_ai_for_text_response_acquires_limiter() -> None:
    limiter = SimpleNamespace(acquire=AsyncMock())

    with patch("agentic_learning_portal.api.llm.httpx.AsyncClient") as mock_cls:
        mock_client = _mock_httpx_client()
        mock_cls.return_value = mock_client
        mock_client.post = AsyncMock(return_value=_json_response("12"))

        await ask_ai_for_text_response(
            GROQ_URL,
            api_key="key",
            model="m",
            messages=[{"role": "user", "content": "hi"}],
            limiter=limiter,
        )

    limiter.acquire.assert_awaited_once()
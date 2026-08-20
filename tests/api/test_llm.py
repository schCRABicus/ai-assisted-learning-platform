"""Tests for the centralized LLM module (retry-with-backoff, rate limiting)."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from pydantic import BaseModel

from agentic_learning_portal.api.llm import (
    LLMOverloadedError,
    LLMResponseError,
    LLMRetryableError,
    LLMUnavailableError,
    ModelChain,
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

    async def _flaky(model) -> str:
        nonlocal calls
        calls += 1
        if calls < 3:
            raise httpx.ConnectError("boom")
        return "ok"

    with patch("agentic_learning_portal.api.llm.asyncio.sleep", new=AsyncMock()) as mock_sleep:
        result = await retry(_flaky, model=ModelChain(primary_model="primary", secondary_models=None), retries=5)

    assert result == "ok"
    assert calls == 3
    mock_sleep.assert_awaited()


@pytest.mark.asyncio
async def test_retry_gives_up_after_retries() -> None:
    async def _always_fails(model) -> str:
        raise LLMRetryableError("HTTP 429")

    with patch("agentic_learning_portal.api.llm.asyncio.sleep", new=AsyncMock()):
        with pytest.raises(LLMRetryableError, match="429"):
            await retry(_always_fails, model=ModelChain(primary_model="primary", secondary_models=None), retries=2)


@pytest.mark.asyncio
async def test_retry_does_not_retry_non_transient() -> None:
    async def _bad(model) -> str:
        raise ValueError("not transient")

    with patch("agentic_learning_portal.api.llm.asyncio.sleep", new=AsyncMock()) as mock_sleep:
        with pytest.raises(ValueError, match="not transient"):
            await retry(_bad, model=ModelChain(primary_model="primary", secondary_models=None), retries=3)

    mock_sleep.assert_not_awaited()


# --- ask_ai_for_structured_response -----------------------------------------------------------


@pytest.mark.asyncio
async def test_ask_ai_for_structured_response_returns_typed_output() -> None:
    with patch("agentic_learning_portal.api.llm.Agent") as mock_agent:
        mock_agent.return_value.run = AsyncMock(
            return_value=SimpleNamespace(output=_Answer(value="ok"))
        )
        result = await ask_ai_for_structured_response(
            model=ModelChain(primary_model="google:gemini-3.5-flash", secondary_models=None),
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
            model=ModelChain(primary_model="m", secondary_models=None),
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
        await ask_ai_for_structured_response(
            ModelChain(primary_model="m", secondary_models=None), _Answer, "sys", "usr", limiter=limiter
        )

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
            model=ModelChain(primary_model="qwen/qwen3.6-27b", secondary_models=None),
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
            model=ModelChain(primary_model="m", secondary_models=None),
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
                model=ModelChain(primary_model="m", secondary_models=None),
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
                model=ModelChain(primary_model="m", secondary_models=None),
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
                model=ModelChain(primary_model="m", secondary_models=None),
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
                model=ModelChain(primary_model="m", secondary_models=None),
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
            model=ModelChain(primary_model="m", secondary_models=None),
            messages=[{"role": "user", "content": "hi"}],
            limiter=limiter,
        )

    limiter.acquire.assert_awaited_once()


# --- round-robin fallback to secondary models ----------------------------------
#
# On a retryable failure the retry is expected to advance to the next model in
# the chain (round-robin) instead of hammering the same overloaded or
# rate-limited model. A plain ``str`` is a chain of one and keeps retrying the
# same model. These tests cover the three transient error types:
# ``LLMOverloadedError``, ``LLMRetryableError`` and ``LLMUnavailableError``.

_TRANSIENT_ERRORS = [LLMOverloadedError, LLMRetryableError, LLMUnavailableError]


def _recording_agent_factory(models_used: list[str], error: Exception, succeed_after: int) -> Callable:
    """Stand-in for ``Agent`` that records the model of each construction.

    The first ``succeed_after`` constructions get a ``run`` that raises
    ``error``; later ones succeed with a valid structured output.
    """
    calls = 0

    def _factory(**kwargs):
        nonlocal calls
        calls += 1
        models_used.append(kwargs["model"])
        instance = MagicMock()
        if calls <= succeed_after:
            instance.run = AsyncMock(side_effect=error)
        else:
            instance.run = AsyncMock(return_value=SimpleNamespace(output=_Answer(value="ok")))
        return instance

    return _factory


def _recording_post(models_used: list[str], error: Exception, succeed_after: int) -> Callable:
    """Stand-in for ``httpx.AsyncClient.post`` recording each payload model.

    The first ``succeed_after`` calls raise ``error``; later ones return a
    valid chat-completion response.
    """
    calls = 0

    def _post(url, *, json=None, headers=None):
        nonlocal calls
        calls += 1
        models_used.append(json["model"])
        if calls <= succeed_after:
            raise error
        return _json_response("12")

    return _post


@pytest.mark.asyncio
@pytest.mark.parametrize("error_type", _TRANSIENT_ERRORS, ids=lambda t: t.__name__)
async def test_ask_ai_for_structured_response_round_robins_to_secondary_on_transient_error(
    error_type,
) -> None:
    models_used: list[str] = []
    chain = ModelChain(primary_model="primary", secondary_models=["secondary"])

    with (
        patch(
            "agentic_learning_portal.api.llm.Agent",
            side_effect=_recording_agent_factory(models_used, error_type("boom"), succeed_after=1),
        ),
        patch("agentic_learning_portal.api.llm.asyncio.sleep", new=AsyncMock()),
    ):
        result = await ask_ai_for_structured_response(
            model=chain,
            output_type=_Answer,
            system_prompt="sys",
            user_prompt="usr",
            retries=3,
            limiter=None,
        )

    assert result.value == "ok"
    assert models_used == ["primary", "secondary"]


@pytest.mark.asyncio
async def test_ask_ai_for_structured_response_round_robins_through_all_secondary_models() -> None:
    models_used: list[str] = []
    chain = ModelChain(primary_model="primary", secondary_models=["secondary1", "secondary2"])

    with (
        patch(
            "agentic_learning_portal.api.llm.Agent",
            side_effect=_recording_agent_factory(models_used, LLMRetryableError("boom"), succeed_after=2),
        ),
        patch("agentic_learning_portal.api.llm.asyncio.sleep", new=AsyncMock()),
    ):
        result = await ask_ai_for_structured_response(
            model=chain,
            output_type=_Answer,
            system_prompt="sys",
            user_prompt="usr",
            retries=3,
            limiter=None,
        )

    assert result.value == "ok"
    assert models_used == ["primary", "secondary1", "secondary2"]


@pytest.mark.asyncio
async def test_ask_ai_for_structured_response_round_robin_wraps_around_to_primary() -> None:
    models_used: list[str] = []
    chain = ModelChain(primary_model="primary", secondary_models=["secondary"])

    with (
        patch(
            "agentic_learning_portal.api.llm.Agent",
            side_effect=_recording_agent_factory(models_used, LLMRetryableError("boom"), succeed_after=2),
        ),
        patch("agentic_learning_portal.api.llm.asyncio.sleep", new=AsyncMock()),
    ):
        result = await ask_ai_for_structured_response(
            model=chain,
            output_type=_Answer,
            system_prompt="sys",
            user_prompt="usr",
            retries=3,
            limiter=None,
        )

    assert result.value == "ok"
    assert models_used == ["primary", "secondary", "primary"]


@pytest.mark.asyncio
async def test_ask_ai_for_structured_response_reuses_single_model_on_retry() -> None:
    models_used: list[str] = []

    with (
        patch(
            "agentic_learning_portal.api.llm.Agent",
            side_effect=_recording_agent_factory(models_used, LLMRetryableError("boom"), succeed_after=1),
        ),
        patch("agentic_learning_portal.api.llm.asyncio.sleep", new=AsyncMock()),
    ):
        result = await ask_ai_for_structured_response(
            model=ModelChain(primary_model="only-model", secondary_models=None),
            output_type=_Answer,
            system_prompt="sys",
            user_prompt="usr",
            retries=3,
            limiter=None,
        )

    assert result.value == "ok"
    assert models_used == ["only-model", "only-model"]


@pytest.mark.asyncio
@pytest.mark.parametrize("error_type", _TRANSIENT_ERRORS, ids=lambda t: t.__name__)
async def test_ask_ai_for_text_response_round_robins_to_secondary_on_transient_error(
    error_type,
) -> None:
    models_used: list[str] = []
    chain = ModelChain(primary_model="primary", secondary_models=["secondary"])

    with (
        patch("agentic_learning_portal.api.llm.httpx.AsyncClient") as mock_cls,
        patch("agentic_learning_portal.api.llm.asyncio.sleep", new=AsyncMock()),
    ):
        mock_client = _mock_httpx_client()
        mock_cls.return_value = mock_client
        mock_client.post = AsyncMock(
            side_effect=_recording_post(models_used, error_type("boom"), succeed_after=1)
        )

        result = await ask_ai_for_text_response(
            GROQ_URL,
            api_key="key",
            model=chain,
            messages=[{"role": "user", "content": "hi"}],
            retries=3,
            limiter=None,
        )

    assert result == "12"
    assert models_used == ["primary", "secondary"]


@pytest.mark.asyncio
async def test_ask_ai_for_text_response_round_robins_through_all_secondary_models() -> None:
    models_used: list[str] = []
    chain = ModelChain(primary_model="primary", secondary_models=["secondary1", "secondary2"])

    with (
        patch("agentic_learning_portal.api.llm.httpx.AsyncClient") as mock_cls,
        patch("agentic_learning_portal.api.llm.asyncio.sleep", new=AsyncMock()),
    ):
        mock_client = _mock_httpx_client()
        mock_cls.return_value = mock_client
        mock_client.post = AsyncMock(
            side_effect=_recording_post(models_used, LLMRetryableError("boom"), succeed_after=2)
        )

        result = await ask_ai_for_text_response(
            GROQ_URL,
            api_key="key",
            model=chain,
            messages=[{"role": "user", "content": "hi"}],
            retries=3,
            limiter=None,
        )

    assert result == "12"
    assert models_used == ["primary", "secondary1", "secondary2"]


@pytest.mark.asyncio
async def test_ask_ai_for_text_response_round_robin_wraps_around_to_primary() -> None:
    models_used: list[str] = []
    chain = ModelChain(primary_model="primary", secondary_models=["secondary"])

    with (
        patch("agentic_learning_portal.api.llm.httpx.AsyncClient") as mock_cls,
        patch("agentic_learning_portal.api.llm.asyncio.sleep", new=AsyncMock()),
    ):
        mock_client = _mock_httpx_client()
        mock_cls.return_value = mock_client
        mock_client.post = AsyncMock(
            side_effect=_recording_post(models_used, LLMRetryableError("boom"), succeed_after=2)
        )

        result = await ask_ai_for_text_response(
            GROQ_URL,
            api_key="key",
            model=chain,
            messages=[{"role": "user", "content": "hi"}],
            retries=3,
            limiter=None,
        )

    assert result == "12"
    assert models_used == ["primary", "secondary", "primary"]


@pytest.mark.asyncio
async def test_ask_ai_for_text_response_round_robins_to_secondary_on_http_503() -> None:
    models_used: list[str] = []
    overloaded = MagicMock()
    overloaded.status_code = 503

    def _post(url, *, json=None, headers=None):
        models_used.append(json["model"])
        if len(models_used) == 1:
            return overloaded
        return _json_response("12")

    with (
        patch("agentic_learning_portal.api.llm.httpx.AsyncClient") as mock_cls,
        patch("agentic_learning_portal.api.llm.asyncio.sleep", new=AsyncMock()),
    ):
        mock_client = _mock_httpx_client()
        mock_cls.return_value = mock_client
        mock_client.post = AsyncMock(side_effect=_post)

        result = await ask_ai_for_text_response(
            GROQ_URL,
            api_key="key",
            model=ModelChain(primary_model="primary", secondary_models=["secondary"]),
            messages=[{"role": "user", "content": "hi"}],
            retries=3,
            limiter=None,
        )

    assert result == "12"
    assert models_used == ["primary", "secondary"]
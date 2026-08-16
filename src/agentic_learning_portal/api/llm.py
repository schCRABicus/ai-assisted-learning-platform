"""Centralized LLM calls with retry-with-backoff and rate limiting.

Every LLM interaction in the app goes through this module so the failure
handling (retry with exponential backoff + jitter) and the rate limiting
(token-bucket pace limiter) live in one place:

- ``ask_ai_for_structured_response`` — a pydantic-ai ``Agent`` call with a structured
  ``output_type`` (used by the task generator, the Wolfram|Alpha judge's
  translation, and the admin subtopic suggester).
- ``ask_ai_for_text_response`` — a raw OpenAI-compatible chat-completions HTTP call
  (used by the Qwen judge against Groq), which is not a pydantic-ai call and so
  has its own transport path.
- ``MODELS`` — the single registry of which LLM backs each purpose (task
  generation, subtopic suggestion, judge translation / solving); every component
  takes its default model from here and can still be overridden per call.

Both retry only *transient* failures (rate limits, server errors, transport
errors, and pydantic-ai model-behavior hiccups) and share one conservative
default rate limiter. Pass a custom ``RateLimiter`` (or ``None`` to disable)
per call when a provider needs a different pace.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from collections.abc import Awaitable, Callable, Sequence
from typing import Any, TypeVar

import httpx
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.exceptions import UnexpectedModelBehavior

logger = logging.getLogger(__name__)

OutputT = TypeVar("OutputT", bound=BaseModel)
T = TypeVar("T")


class LLMError(Exception):
    """Base error for LLM call failures."""


class LLMRetryableError(LLMError):
    """A transient LLM failure worth retrying (rate limit, 5xx, transport)."""


class LLMUnavailableError(LLMError):
    """The LLM could not complete the call after retries (rate limit / outage)."""


class LLMResponseError(LLMError):
    """The LLM responded but the reply could not be used."""


class RateLimiter:
    """Asyncio-safe token-bucket limiter: ``rate`` acquisitions per ``period`` seconds.

    The bucket starts full, refills continuously, and a single ``asyncio.Lock``
    guards acquisition so concurrent callers cannot overshoot the pace. Use one
    limiter per provider/quota, or the module-level ``DEFAULT_LIMITER`` as a
    coarse backstop.
    """

    def __init__(self, rate: float, period: float = 1.0) -> None:
        if rate <= 0 or period <= 0:
            raise ValueError("rate and period must be positive")
        self._max_tokens = float(rate)
        self._tokens = float(rate)
        self._period = float(period)
        self._updated = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Block until a token is available, then consume it."""
        async with self._lock:
            self._refill()
            while self._tokens < 1:
                self._refill()
                # Seconds until the bucket refills by one token.
                wait = (1 - self._tokens) * self._period / self._max_tokens
                await asyncio.sleep(wait)
            self._tokens -= 1

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self._updated
        self._tokens = min(
            self._max_tokens,
            self._tokens + elapsed * (self._max_tokens / self._period),
        )
        self._updated = now


# Coarse shared backstop: 20 calls/second. Providers have their own quotas; tune
# this or pass a per-call ``RateLimiter`` when a call needs a different pace.
DEFAULT_LIMITER = RateLimiter(rate=20, period=1.0)


#: The model that backs each LLM purpose in the app, kept in one place so
#: retuning which model does what is a one-line change. Components read their
#: default from here by key (still overridable per instance via ``model=``).
MODELS: dict[str, str] = {
    # The authoring call: generates a task — problem text, correct answer, and
    # step-by-step solution — from a prompt input. Also the Wolfram|Alpha
    # judge's translation fallback when no Groq key is set (see
    # ``domains.math.judges.build_judge_ensemble``).
    "task_generation": "google:gemini-3.5-flash",
    # Suggests candidate subtopics for a free-text topic in the admin UI.
    "subtopic_suggestion": "google:gemini-3.5-flash",
    # Translates a task's prose into a bare Wolfram|Alpha-computable expression
    # (``WolframAlphaJudge``). A different provider/family than the generator so
    # the judge doesn't share the authoring model's blind spots. Served by Groq
    # (which now hosts gpt-oss-120b under the ``openai/`` prefix); the ``groq:``
    # prefix is the pydantic-ai provider, the rest is Groq's model id.
    "wolfram_translation": "groq:openai/gpt-oss-120b",
    # Solves the task for an independent numeric answer (``QwenMathJudge``).
    "qwen_solver": "qwen/qwen3.6-27b",
}


def _is_transient(exc: Exception) -> bool:
    """Default retry predicate: transport errors, rate limits, model hiccups."""
    if isinstance(exc, (httpx.TransportError, LLMRetryableError)):
        return True
    return isinstance(exc, UnexpectedModelBehavior)


async def retry(
    fn: Callable[[], Awaitable[T]],
    *,
    retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    retry_on: Callable[[Exception], bool] = _is_transient,
) -> T:
    """Run ``fn``, retrying retryable failures with exponential backoff + jitter.

    The first attempt runs immediately; each subsequent retry waits
    ``base_delay * 2 ** attempt`` seconds (bounded by ``max_delay``) with jitter
    so a burst of callers doesn't all retry at once. Raises the last exception
    once ``retries`` attempts are exhausted.
    """
    for attempt in range(retries + 1):
        try:
            return await fn()
        except Exception as exc:  # noqa: BLE001 - retry decision delegated to retry_on
            last_exc = exc
            if attempt >= retries or not retry_on(exc):
                raise
            delay = min(max_delay, base_delay * (2**attempt)) * random.uniform(0.5, 1.5)
            logger.warning(
                "LLM call failed (attempt %d/%d): %s; retrying in %.2fs",
                attempt + 1,
                retries + 1,
                exc,
                delay,
            )
            await asyncio.sleep(delay)
    raise last_exc  # pragma: no cover - the loop always returns or raises


async def ask_ai_for_structured_response(
    model: str,
    output_type: type[OutputT],
    system_prompt: str,
    user_prompt: str,
    *,
    retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    limiter: RateLimiter | None = DEFAULT_LIMITER,
) -> OutputT:
    """Run a pydantic-ai ``Agent`` with structured output, retrying transient errors.

    Returns ``response.output`` — normally an instance of ``output_type``, but
    a model can still return an unparseable string, so callers should guard the
    result with ``isinstance`` where the shape matters.
    """

    async def _call() -> OutputT:
        agent = Agent(model=model, output_type=output_type, system_prompt=system_prompt)
        response = await agent.run(user_prompt)
        return response.output

    async def _call_limited() -> OutputT:
        if limiter is not None:
            await limiter.acquire()
        return await _call()

    return await retry(_call_limited, retries=retries, base_delay=base_delay, max_delay=max_delay)


async def ask_ai_for_text_response(
    url: str,
    api_key: str,
    model: str,
    messages: Sequence[dict[str, Any]],
    *,
    retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    timeout: float = 60.0,
    max_tokens: int | None = None,
    temperature: float = 0.0,
    limiter: RateLimiter | None = DEFAULT_LIMITER,
) -> str:
    """POST an OpenAI-compatible chat request and return the assistant's message.

    Retries rate limits (429), server errors (5xx) and transport failures with
    backoff; raises ``LLMUnavailableError`` if they persist. Raises
    ``httpx.HTTPStatusError`` for other HTTP errors (e.g. 401) and
    ``LLMResponseError`` when the reply shape is unexpected.
    """
    payload: dict[str, Any] = {"model": model, "messages": list(messages), "temperature": temperature}
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens

    async def _call() -> str:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                url,
                json=payload,
                headers={"Authorization": f"Bearer {api_key}"},
            )
        if response.status_code == 429 or 500 <= response.status_code < 600:
            raise LLMRetryableError(f"HTTP {response.status_code}")
        response.raise_for_status()
        data = response.json()
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as e:
            raise LLMResponseError(f"Unexpected response shape: {data!r}") from e
        return content.strip() if content else ""

    async def _call_limited() -> str:
        if limiter is not None:
            await limiter.acquire()
        return await _call()

    try:
        return await retry(_call_limited, retries=retries, base_delay=base_delay, max_delay=max_delay)
    except (LLMRetryableError, httpx.TransportError) as e:
        raise LLMUnavailableError(f"LLM unavailable after {retries} retries: {e}") from e
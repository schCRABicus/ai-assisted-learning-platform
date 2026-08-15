from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from agentic_learning_portal.api.llm import LLMResponseError, LLMUnavailableError
from agentic_learning_portal.api.model import GeneratedTask
from agentic_learning_portal.api.progress import CollectingProgressListener, ProgressEvent
from agentic_learning_portal.domains.math.qwen_judge import (
    GROQ_CHAT_URL,
    QwenMathJudge,
    SOLVER_SYSTEM_PROMPT,
)

MODEL = "qwen/qwen3.6-27b"


def _task(correct_answer: str | int | float, text: str = "3*4") -> GeneratedTask:
    return GeneratedTask.model_validate(
        {
            "topic": "Arithmetic",
            "text": text,
            "complexity": "easy",
            "correct_answer": correct_answer,
            "solution": "Multiply the two numbers: 3 × 4 = 12.",
        }
    )


# --- answer extraction --------------------------------------------------------


def test_extract_answer_after_think_block() -> None:
    judge = QwenMathJudge(api_key="test-key")

    response = "<think>3 boxes of 4 bricks... 3*4=12</think>\n12"
    assert judge._extract_answer(response) == "12"


def test_extract_answer_ignores_reasoning_numbers() -> None:
    judge = QwenMathJudge(api_key="test-key")

    response = "<think>90-42=48\n48/8=6</think>\n6"
    assert judge._extract_answer(response) == "6"


def test_extract_answer_last_number_without_think_tags() -> None:
    judge = QwenMathJudge(api_key="test-key")

    assert judge._extract_answer("So the total is 12") == "12"


def test_extract_answer_from_boxed() -> None:
    judge = QwenMathJudge(api_key="test-key")

    assert judge._extract_answer("Step by step... \\boxed{9}") == "9"


def test_extract_answer_prefers_boxed_over_last_number() -> None:
    judge = QwenMathJudge(api_key="test-key")

    assert judge._extract_answer("\\boxed{9} (a different 12?)") == "9"


def test_extract_answer_handles_fraction() -> None:
    judge = QwenMathJudge(api_key="test-key")

    assert judge._extract_answer("Answer: 1/2") == "1/2"


def test_extract_answer_empty_for_non_numeric() -> None:
    judge = QwenMathJudge(api_key="test-key")

    assert judge._extract_answer("Sorry, I cannot solve that.") == ""


# --- verify -------------------------------------------------------------------


@pytest.mark.asyncio
async def test_verify_returns_matching_result() -> None:
    judge = QwenMathJudge(api_key="test-key")
    with patch.object(
        judge,
        "_query_groq",
        new_callable=AsyncMock,
        return_value="<think>3*4=12</think>\n12",
    ):
        result = await judge.verify(_task(12))

    assert result.judge == "qwen"
    assert result.verified is True
    assert result.expected_answer == 12
    assert result.judge_answer == "12"


@pytest.mark.asyncio
async def test_verify_reports_mismatch() -> None:
    judge = QwenMathJudge(api_key="test-key")
    with patch.object(
        judge,
        "_query_groq",
        new_callable=AsyncMock,
        return_value="15",
    ):
        result = await judge.verify(_task(12))

    assert result.verified is False
    assert "15" in result.detail
    assert "12" in result.detail


@pytest.mark.asyncio
async def test_verify_unverifiable_when_model_unavailable() -> None:
    judge = QwenMathJudge(api_key="test-key")
    with patch.object(
        judge,
        "_query_groq",
        new_callable=AsyncMock,
        return_value=None,
    ):
        result = await judge.verify(_task(12))

    assert result.verified is False
    assert result.judge_answer is None
    assert "could not produce" in result.detail


@pytest.mark.asyncio
async def test_verify_matches_fraction() -> None:
    judge = QwenMathJudge(api_key="test-key")
    with patch.object(
        judge,
        "_query_groq",
        new_callable=AsyncMock,
        return_value="\\boxed{1/2}",
    ):
        result = await judge.verify(_task(0.5))

    assert result.verified is True


@pytest.mark.asyncio
async def test_verify_unparseable_answer_is_unverified() -> None:
    judge = QwenMathJudge(api_key="test-key")
    with patch.object(
        judge,
        "_query_groq",
        new_callable=AsyncMock,
        return_value="I cannot solve this problem",
    ):
        result = await judge.verify(_task(12))

    assert result.verified is False
    assert result.judge_answer == ""
    assert "no parseable answer" in result.detail


# --- API ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_query_groq_raises_without_api_key() -> None:
    judge = QwenMathJudge(api_key="")

    with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
        await judge._query_groq("3*4")


@pytest.mark.asyncio
async def test_query_groq_posts_to_groq_endpoint() -> None:
    judge = QwenMathJudge(api_key="test-key")

    with patch(
        "agentic_learning_portal.domains.math.qwen_judge.ask_ai_for_text_response",
        new_callable=AsyncMock,
        return_value="12",
    ) as mock_run:
        answer = await judge._query_groq("What is 3 times 4?")

    assert answer == "12"
    mock_run.assert_awaited_once_with(
        url=GROQ_CHAT_URL,
        api_key="test-key",
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": (
                    f"{SOLVER_SYSTEM_PROMPT}\n\nProblem: What is 3 times 4?"
                ),
            }
        ],
        timeout=60.0,
        max_tokens=512,
    )


@pytest.mark.asyncio
async def test_query_groq_returns_none_on_rate_limit() -> None:
    judge = QwenMathJudge(api_key="test-key")

    with patch(
        "agentic_learning_portal.domains.math.qwen_judge.ask_ai_for_text_response",
        new_callable=AsyncMock,
        side_effect=LLMUnavailableError("rate limited"),
    ):
        answer = await judge._query_groq("What is 3 times 4?")

    assert answer is None


@pytest.mark.asyncio
async def test_query_groq_returns_none_on_malformed_response() -> None:
    judge = QwenMathJudge(api_key="test-key")

    with patch(
        "agentic_learning_portal.domains.math.qwen_judge.ask_ai_for_text_response",
        new_callable=AsyncMock,
        side_effect=LLMResponseError("unexpected shape"),
    ):
        answer = await judge._query_groq("What is 3 times 4?")

    assert answer is None


# --- progress reporting -------------------------------------------------------


@pytest.mark.asyncio
async def test_verify_reports_progress_events() -> None:
    judge = QwenMathJudge(api_key="test-key")
    with patch.object(
        judge,
        "_query_groq",
        new_callable=AsyncMock,
        return_value="<think>3*4=12</think>\n12",
    ):
        log: list[ProgressEvent] = []
        result = await judge.verify(_task(12), listener=CollectingProgressListener(log))

    assert result.verified is True
    messages = [e.message for e in log]
    assert any("solving" in m for m in messages)
    assert any("verified" in m for m in messages)


@pytest.mark.asyncio
async def test_verify_reports_unverifiable_when_model_unavailable() -> None:
    judge = QwenMathJudge(api_key="test-key")
    with patch.object(
        judge,
        "_query_groq",
        new_callable=AsyncMock,
        return_value=None,
    ):
        log: list[ProgressEvent] = []
        await judge.verify(_task(12), listener=CollectingProgressListener(log))

    messages = [e.message for e in log]
    assert any("could not produce" in m for m in messages)
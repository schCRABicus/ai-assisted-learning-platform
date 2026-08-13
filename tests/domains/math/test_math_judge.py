from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentic_learning_portal.api.model import GeneratedTask
from agentic_learning_portal.domains.math._comparison import (
    _answers_match,
    _normalize,
    _to_number,
)
from agentic_learning_portal.domains.math.wa_judge import (
    TRANSLATOR_SYSTEM_PROMPT,
    MathQuery,
    WolframAlphaJudge,
)


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


# --- pure helpers ------------------------------------------------------------


def test_normalize_lowercases_and_strips() -> None:
    assert _normalize("  Twelve.  ") == "twelve"


def test_to_number_handles_ints_floats_and_numeric_strings() -> None:
    assert _to_number(12) == 12.0
    assert _to_number(3.5) == 3.5
    assert _to_number("12") == 12.0
    assert _to_number("3.14") == 3.14


def test_to_number_parses_fractions() -> None:
    assert _to_number("1/2") == 0.5


def test_to_number_returns_none_for_non_numeric() -> None:
    assert _to_number("hello") is None
    assert _to_number(True) is None


def test_answers_match_numeric() -> None:
    assert _answers_match(12, "12")
    assert _answers_match(0.1 + 0.2, "0.3")


def test_answers_match_string() -> None:
    assert _answers_match("Lego", "lego")
    assert _answers_match("Twelve", "twelve.")


def test_answers_mismatch() -> None:
    assert not _answers_match(12, "15")
    assert not _answers_match("Lego", "Star Wars")


def test_answers_match_none_wolfram_answer() -> None:
    assert not _answers_match(12, None)


# --- judge -------------------------------------------------------------------


@pytest.mark.asyncio
async def test_verify_returns_matching_result() -> None:
    judge = WolframAlphaJudge(app_id="test-app-id")
    with (
        patch.object(
            judge,
            "_translate_to_query",
            new_callable=AsyncMock,
            return_value="3*4",
        ),
        patch.object(
            judge,
            "_query_wolfram",
            new_callable=AsyncMock,
            return_value="12",
        ),
    ):
        result = await judge.verify(_task(12))

    assert result.judge == "wolframalpha"
    assert result.verified is True
    assert result.expected_answer == 12
    assert result.judge_answer == "12"


@pytest.mark.asyncio
async def test_verify_reports_mismatch() -> None:
    judge = WolframAlphaJudge(app_id="test-app-id")
    with (
        patch.object(
            judge,
            "_translate_to_query",
            new_callable=AsyncMock,
            return_value="3*4",
        ),
        patch.object(
            judge,
            "_query_wolfram",
            new_callable=AsyncMock,
            return_value="15",
        ),
    ):
        result = await judge.verify(_task(12))

    assert result.verified is False
    assert "15" in result.detail
    assert "12" in result.detail


@pytest.mark.asyncio
async def test_verify_queries_translated_expression() -> None:
    judge = WolframAlphaJudge(app_id="test-app-id")
    with (
        patch.object(
            judge,
            "_translate_to_query",
            new_callable=AsyncMock,
            return_value="3*4",
        ) as mock_translate,
        patch.object(
            judge,
            "_query_wolfram",
            new_callable=AsyncMock,
            return_value="12",
        ) as mock_query,
    ):
        await judge.verify(_task(12, text="What is 3 times 4?"))

    mock_translate.assert_awaited_once_with("What is 3 times 4?")
    mock_query.assert_awaited_once_with("3*4")


@pytest.mark.asyncio
async def test_verify_handles_unparseable_wolfram_answer() -> None:
    judge = WolframAlphaJudge(app_id="test-app-id")
    with (
        patch.object(
            judge,
            "_translate_to_query",
            new_callable=AsyncMock,
            return_value="3*4",
        ),
        patch.object(
            judge,
            "_query_wolfram",
            new_callable=AsyncMock,
            return_value=None,
        ),
    ):
        result = await judge.verify(_task(12))

    assert result.verified is False
    assert result.judge_answer is None


@pytest.mark.asyncio
async def test_verify_inconclusive_when_translation_fails() -> None:
    judge = WolframAlphaJudge(app_id="test-app-id")
    with (
        patch.object(
            judge,
            "_translate_to_query",
            new_callable=AsyncMock,
            return_value="",
        ),
        patch.object(
            judge,
            "_query_wolfram",
            new_callable=AsyncMock,
        ) as mock_query,
    ):
        result = await judge.verify(_task(12))

    assert result.verified is False
    assert result.judge_answer is None
    assert "translate" in result.detail.lower()
    mock_query.assert_not_awaited()


@pytest.mark.asyncio
async def test_query_wolfram_raises_without_app_id() -> None:
    judge = WolframAlphaJudge(app_id="")

    with pytest.raises(RuntimeError, match="WOLFRAM_APP_ID"):
        await judge._query_wolfram("3*4")


def _mock_wolfram_client(response: object) -> MagicMock:
    """Return a MagicMock that stands in for ``httpx.AsyncClient``."""
    fake_client = SimpleNamespace(get=AsyncMock(return_value=response))
    mock_client = MagicMock()
    mock_client.__aenter__ = AsyncMock(return_value=fake_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    return mock_client


@pytest.mark.asyncio
async def test_query_wolfram_returns_plain_text_answer() -> None:
    judge = WolframAlphaJudge(app_id="test-app-id")
    fake_response = SimpleNamespace(status_code=200, text=" 12 ")
    fake_response.raise_for_status = lambda: None

    with patch(
        "agentic_learning_portal.domains.math.wa_judge.httpx.AsyncClient",
        return_value=_mock_wolfram_client(fake_response),
    ) as mock_client:
        answer = await judge._query_wolfram("3*4")

    assert answer == "12"
    mock_client.assert_called_once()
    fake_client = mock_client.return_value.__aenter__.return_value
    fake_client.get.assert_awaited_once_with(
        "https://api.wolframalpha.com/v1/result",
        params={"appid": "test-app-id", "i": "3*4"},
    )


@pytest.mark.asyncio
async def test_query_wolfram_returns_none_when_not_understood() -> None:
    judge = WolframAlphaJudge(app_id="test-app-id")
    fake_response = SimpleNamespace(
        status_code=501,
        text="Wolfram|Alpha did not understand your input",
    )

    with patch(
        "agentic_learning_portal.domains.math.wa_judge.httpx.AsyncClient",
        return_value=_mock_wolfram_client(fake_response),
    ):
        answer = await judge._query_wolfram("Crazy Dave has 8 packs of seeds")

    assert answer is None


@pytest.mark.asyncio
async def test_query_wolfram_propagates_other_error_status() -> None:
    judge = WolframAlphaJudge(app_id="test-app-id")

    def _raise() -> None:
        raise RuntimeError("HTTP 403")

    fake_response = SimpleNamespace(status_code=403, text="Forbidden")
    fake_response.raise_for_status = _raise

    with (
        patch(
            "agentic_learning_portal.domains.math.wa_judge.httpx.AsyncClient",
            return_value=_mock_wolfram_client(fake_response),
        ),
        pytest.raises(RuntimeError, match="HTTP 403"),
    ):
        await judge._query_wolfram("3*4")


@pytest.mark.asyncio
async def test_translate_to_query_extracts_expression() -> None:
    judge = WolframAlphaJudge(app_id="test-app-id")

    with patch(
        "agentic_learning_portal.domains.math.wa_judge.ask_ai_for_structured_response",
        new_callable=AsyncMock,
        return_value=MathQuery(query="(7*12-28)/8"),
    ) as mock_run:
        query = await judge._translate_to_query("Crazy Dave has 7 boxes...")

    assert query == "(7*12-28)/8"
    mock_run.assert_awaited_once_with(
        model="groq:llama-3.3-70b-versatile",
        output_type=MathQuery,
        system_prompt=TRANSLATOR_SYSTEM_PROMPT,
        user_prompt="Crazy Dave has 7 boxes...",
    )


@pytest.mark.asyncio
async def test_translate_to_query_returns_empty_on_error() -> None:
    judge = WolframAlphaJudge(app_id="test-app-id")

    with patch(
        "agentic_learning_portal.domains.math.wa_judge.ask_ai_for_structured_response",
        new_callable=AsyncMock,
        side_effect=RuntimeError("boom"),
    ):
        query = await judge._translate_to_query("some problem")

    assert query == ""


@pytest.mark.asyncio
async def test_translate_to_query_retries_when_first_call_fails() -> None:
    judge = WolframAlphaJudge(app_id="test-app-id")

    with patch(
        "agentic_learning_portal.domains.math.wa_judge.ask_ai_for_structured_response",
        new_callable=AsyncMock,
        side_effect=[RuntimeError("boom"), MathQuery(query="3*4")],
    ):
        query = await judge._translate_to_query("What is 3 times 4?", retries=3)

    assert query == "3*4"


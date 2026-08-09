from __future__ import annotations

import xml.etree.ElementTree as ET
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from agentic_learning_portal.api.model import GeneratedTask
from agentic_learning_portal.domains.math.judge import (
    MathQuery,
    WolframAlphaJudge,
    _answers_match,
    _normalize,
    _to_number,
)


def _task(correct_answer: str | int | float, text: str = "3*4") -> GeneratedTask:
    return GeneratedTask.model_validate(
        {
            "topic": "Arithmetic",
            "text": text,
            "complexity": "easy",
            "correct_answer": correct_answer,
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


# --- extraction --------------------------------------------------------------


def test_extract_answer_prefers_result_pod() -> None:
    pods = [
        {"title": "Input interpretation", "text": "3*4"},
        {"title": "Result", "text": "12"},
    ]
    assert WolframAlphaJudge._extract_answer(pods) == "12"


def test_extract_answer_falls_back_to_first_non_empty_pod() -> None:
    pods = [
        {"title": "Input interpretation", "text": "3*4"},
    ]
    assert WolframAlphaJudge._extract_answer(pods) == "3*4"


def test_extract_answer_returns_none_when_all_empty() -> None:
    assert WolframAlphaJudge._extract_answer([{"title": "Result", "text": ""}]) is None
    assert WolframAlphaJudge._extract_answer([]) is None


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
            return_value=[{"title": "Result", "text": "12"}],
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
            return_value=[{"title": "Result", "text": "15"}],
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
            return_value=[{"title": "Result", "text": "12"}],
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
            return_value=[{"title": "No result", "text": ""}],
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


@pytest.mark.asyncio
async def test_translate_to_query_extracts_expression() -> None:
    judge = WolframAlphaJudge(app_id="test-app-id")
    fake_response = SimpleNamespace(output=MathQuery(query="(7*12-28)/8"))

    with patch("agentic_learning_portal.domains.math.judge.Agent") as mock_agent:
        mock_agent.return_value.run = AsyncMock(return_value=fake_response)
        query = await judge._translate_to_query("Crazy Dave has 7 boxes...")

    assert query == "(7*12-28)/8"


@pytest.mark.asyncio
async def test_translate_to_query_returns_empty_on_error() -> None:
    judge = WolframAlphaJudge(app_id="test-app-id")

    with patch("agentic_learning_portal.domains.math.judge.Agent") as mock_agent:
        mock_agent.return_value.run = AsyncMock(side_effect=RuntimeError("boom"))
        query = await judge._translate_to_query("some problem")

    assert query == ""


def _parse(xml_text: str) -> list[dict[str, str]]:
    return WolframAlphaJudge._parse_pods(ET.fromstring(xml_text))


def test_parse_pods_extracts_pod_titles_and_text() -> None:
    xml_text = """
    <queryresult success="true" error="false">
      <pod title="Input interpretation">
        <subpod><plaintext>3*4</plaintext></subpod>
      </pod>
      <pod title="Result">
        <subpod><plaintext>12</plaintext></subpod>
      </pod>
    </queryresult>
    """

    assert _parse(xml_text) == [
        {"title": "Input interpretation", "text": "3*4"},
        {"title": "Result", "text": "12"},
    ]


def test_parse_pods_skips_subpod_without_plaintext() -> None:
    xml_text = """
    <queryresult success="true" error="false">
      <pod title="Visual">
        <subpod><img src="http://example.com/plot.png" /></subpod>
      </pod>
      <pod title="Result">
        <subpod><plaintext>  12  </plaintext></subpod>
      </pod>
    </queryresult>
    """

    assert _parse(xml_text) == [{"title": "Result", "text": "12"}]


def test_parse_pods_returns_empty_for_no_pods() -> None:
    xml_text = '<queryresult success="false" error="false" numpods="0" />'

    assert _parse(xml_text) == []
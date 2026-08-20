from __future__ import annotations

from unittest.mock import AsyncMock, ANY, patch

import pytest

from agentic_learning_portal.admin.subtopic_suggester import (
    SUGGESTOR_SYSTEM_PROMPT,
    SubtopicSuggestions,
    suggest_subtopics,
)


def _suggestions(*subtopics: str) -> SubtopicSuggestions:
    return SubtopicSuggestions(subtopics=list(subtopics))


@pytest.mark.asyncio
async def test_suggest_subtopics_returns_trimmed_deduplicated_list() -> None:
    with patch(
        "agentic_learning_portal.admin.subtopic_suggester.ask_ai_for_structured_response",
        new_callable=AsyncMock,
        return_value=_suggestions(" Fractions ", "fractions", " word problems "),
    ) as mock_run:
        result = await suggest_subtopics("Arithmetic")

    assert result == ["Fractions", "word problems"]
    mock_run.assert_awaited_once_with(
        model=ANY,
        output_type=SubtopicSuggestions,
        system_prompt=SUGGESTOR_SYSTEM_PROMPT,
        user_prompt="Suggest subtopics for the math topic: Arithmetic",
    )


@pytest.mark.asyncio
async def test_suggest_subtopics_respects_max_subtopics() -> None:
    with patch(
        "agentic_learning_portal.admin.subtopic_suggester.ask_ai_for_structured_response",
        new_callable=AsyncMock,
        return_value=_suggestions("a", "b", "c", "d", "e"),
    ):
        result = await suggest_subtopics("Algebra", max_subtopics=3)

    assert result == ["a", "b", "c"]


@pytest.mark.asyncio
async def test_suggest_subtopics_returns_empty_on_provider_error() -> None:
    with patch(
        "agentic_learning_portal.admin.subtopic_suggester.ask_ai_for_structured_response",
        new_callable=AsyncMock,
        side_effect=RuntimeError("boom"),
    ):
        result = await suggest_subtopics("Algebra")

    assert result == []


@pytest.mark.asyncio
async def test_suggest_subtopics_returns_empty_for_blank_topic() -> None:
    assert await suggest_subtopics("   ") == []
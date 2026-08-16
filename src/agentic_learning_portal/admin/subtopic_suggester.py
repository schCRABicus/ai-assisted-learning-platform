"""LLM-based subtopic suggestion for the admin portal."""

from __future__ import annotations

import logging

from pydantic import BaseModel, Field

from agentic_learning_portal.api.llm import MODELS, ask_ai_for_structured_response

logger = logging.getLogger(__name__)

SUGGESTOR_SYSTEM_PROMPT = (
    "You are an experienced math teacher. Given a math topic, propose a "
    "focused list of subtopics that a student could practice. Keep each "
    "subtopic short (1-3 words) and specific."
)


class SubtopicSuggestions(BaseModel):
    """A list of subtopic suggestions for a topic."""

    subtopics: list[str] = Field(
        ...,
        description="Short subtopic names (1-3 words each) for the given topic.",
    )


async def suggest_subtopics(
    topic: str,
    *,
    model: str = MODELS["subtopic_suggestion"],
    max_subtopics: int = 8,
) -> list[str]:
    """Suggest subtopics for ``topic`` with a single LLM call.

    Returns a de-duplicated list of up to ``max_subtopics`` trimmed subtopics.
    This is best-effort: on any failure (blank topic, provider error, or an
    unparseable response) it logs a warning and returns an empty list so the
    admin UI can degrade gracefully instead of crashing.
    """
    if not topic.strip():
        return []

    try:
        result = await ask_ai_for_structured_response(
            model=model,
            output_type=SubtopicSuggestions,
            system_prompt=SUGGESTOR_SYSTEM_PROMPT,
            user_prompt=f"Suggest subtopics for the math topic: {topic}",
        )
        raw = result.subtopics if isinstance(result, SubtopicSuggestions) else []
    except Exception as e:  # noqa: BLE001 - suggestion is best-effort
        logger.warning("Could not suggest subtopics for topic %r: %s", topic, e)
        return []

    result: list[str] = []
    seen: set[str] = set()
    for subtopic in raw:
        clean = subtopic.strip()
        if clean and clean.lower() not in seen:
            seen.add(clean.lower())
            result.append(clean)
        if len(result) >= max_subtopics:
            break
    return result
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from pydantic import ValidationError

from agentic_learning_portal.api.generator import DEFAULT_SYSTEM_PROMPT, Generator
from agentic_learning_portal.domains.math.generator import MathProblemGenerator
from agentic_learning_portal.domains.math.model import MathProblem, MathProblemGenerationPromptInput

VALID_MATH_PROBLEM = {
    "topic": "Lego",
    "text": "You have 3 boxes with 4 bricks each. How many bricks do you have?",
    "complexity": "easy",
    "correct_answer": 12,
}


@pytest.fixture
def generator() -> MathProblemGenerator:
    return MathProblemGenerator(model="test-model")


@pytest.fixture
def prompt_input() -> MathProblemGenerationPromptInput:
    return MathProblemGenerationPromptInput(
        topic="Arithmetic",
        subtopics=["multiplication", "subtraction"],
        context="Lego",
        complexity="easy",
        grade=2,
    )


def test_build_user_prompt_includes_context_topic_grade_and_complexity(
    prompt_input: MathProblemGenerationPromptInput,
) -> None:
    user_prompt = prompt_input.build_user_prompt()

    assert "Lego" in user_prompt
    assert "Arithmetic" in user_prompt
    assert "grade 2" in user_prompt
    assert "easy" in user_prompt
    assert "multiplication, subtraction" in user_prompt
    assert "context" in user_prompt


def test_build_user_prompt_includes_subtopics() -> None:
    prompt_input = MathProblemGenerationPromptInput(
        topic="Arithmetic",
        subtopics=["addition", "fractions"],
        context="Lego",
        complexity="medium",
        grade=3,
    )

    user_prompt = prompt_input.build_user_prompt()

    assert "addition" in user_prompt
    assert "fractions" in user_prompt
    assert "multiplication" not in user_prompt


def test_build_user_prompt_uses_selected_context() -> None:
    prompt_input = MathProblemGenerationPromptInput(
        topic="Algebra",
        subtopics=["linear equations"],
        context="Star Wars",
        complexity="medium",
        grade=5,
    )

    assert "Star Wars" in prompt_input.build_user_prompt()


def test_prompt_input_accepts_any_context() -> None:
    prompt_input = MathProblemGenerationPromptInput(
        topic="Algebra",
        subtopics=["linear equations"],
        context="Harry Potter",
        complexity="easy",
        grade=2,
    )

    assert "Harry Potter" in prompt_input.build_user_prompt()


def test_prompt_input_rejects_grade_below_minimum() -> None:
    with pytest.raises(ValidationError):
        MathProblemGenerationPromptInput(
            topic="Arithmetic",
            subtopics=["multiplication"],
            context="Lego",
            complexity="easy",
            grade=0,
        )


def test_prompt_input_rejects_grade_above_maximum() -> None:
    with pytest.raises(ValidationError):
        MathProblemGenerationPromptInput(
            topic="Arithmetic",
            subtopics=["multiplication"],
            context="Lego",
            complexity="easy",
            grade=12,
        )


@pytest.mark.asyncio
async def test_generate_returns_math_problem(
    generator: MathProblemGenerator,
    prompt_input: MathProblemGenerationPromptInput,
) -> None:
    with patch.object(
        Generator,
        "_call_llm",
        new_callable=AsyncMock,
        return_value=VALID_MATH_PROBLEM,
    ):
        result = await generator.generate(prompt_input)

    assert result == MathProblem.model_validate(VALID_MATH_PROBLEM)


@pytest.mark.asyncio
async def test_generate_passes_built_user_prompt(
    generator: MathProblemGenerator,
    prompt_input: MathProblemGenerationPromptInput,
) -> None:
    with patch.object(
        Generator,
        "_call_llm",
        new_callable=AsyncMock,
        return_value=VALID_MATH_PROBLEM,
    ) as mock_call:
        await generator.generate(prompt_input)

    assert mock_call.await_args.kwargs["user_prompt"] == prompt_input.build_user_prompt()
    assert mock_call.await_args.kwargs["output_type"] is MathProblem


@pytest.mark.asyncio
async def test_generate_uses_custom_system_prompt(
    generator: MathProblemGenerator,
    prompt_input: MathProblemGenerationPromptInput,
) -> None:
    custom_system_prompt = "You are a math teacher for elementary students."

    with patch.object(
        Generator,
        "_call_llm",
        new_callable=AsyncMock,
        return_value=VALID_MATH_PROBLEM,
    ) as mock_call:
        await generator.generate(prompt_input, system_prompt=custom_system_prompt)

    assert mock_call.await_args.kwargs["system_prompt"] == custom_system_prompt


@pytest.mark.asyncio
async def test_generate_uses_default_system_prompt(
    generator: MathProblemGenerator,
    prompt_input: MathProblemGenerationPromptInput,
) -> None:
    with patch.object(
        Generator,
        "_call_llm",
        new_callable=AsyncMock,
        return_value=VALID_MATH_PROBLEM,
    ) as mock_call:
        await generator.generate(prompt_input)

    assert mock_call.await_args.kwargs["system_prompt"] == DEFAULT_SYSTEM_PROMPT

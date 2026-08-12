from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

from agentic_learning_portal.api.generator import DEFAULT_SYSTEM_PROMPT, Generator
from agentic_learning_portal.api.model import GeneratedTask, ProblemGenerationPromptInput

VALID_TASK = {
    "topic": "Lego",
    "text": "You have 3 boxes with 4 bricks each. How many bricks do you have?",
    "complexity": "easy",
    "correct_answer": 12,
    "solution": "Each box holds 4 bricks and there are 3 boxes, so the total is 3 × 4 = 12 bricks.",
}

INVALID_TASK = {
    "topic": "Lego",
    "text": "You have 3 boxes with 4 bricks each. How many bricks do you have?",
    "complexity": "super-hard",
    "correct_answer": 12,
    "solution": "Each box holds 4 bricks and there are 3 boxes, so the total is 3 × 4 = 12 bricks.",
}


class SamplePrompt(ProblemGenerationPromptInput):
    topic: str = "Lego"

    def build_user_prompt(self) -> str:
        return f"Generate a math problem about {self.topic}."


@pytest.fixture
def generator() -> Generator:
    return Generator(model="test-model")


def test_validate_accepts_dict(generator: Generator) -> None:
    validated, error = Generator._validate_response_matches_output_type(
        GeneratedTask,
        VALID_TASK,
    )

    assert error is None
    assert validated == GeneratedTask.model_validate(VALID_TASK)


def test_validate_accepts_json_string(generator: Generator) -> None:
    validated, error = Generator._validate_response_matches_output_type(
        GeneratedTask,
        json.dumps(VALID_TASK),
    )

    assert error is None
    assert validated == GeneratedTask.model_validate(VALID_TASK)


def test_validate_accepts_model_instance(generator: Generator) -> None:
    task = GeneratedTask.model_validate(VALID_TASK)

    validated, error = Generator._validate_response_matches_output_type(
        GeneratedTask,
        task,
    )

    assert error is None
    assert validated == task


def test_validate_rejects_invalid_data(generator: Generator) -> None:
    validated, error = Generator._validate_response_matches_output_type(
        GeneratedTask,
        INVALID_TASK,
    )

    assert validated is None
    assert error is not None
    assert "validation error" in error.lower()


def test_validate_rejects_missing_solution(generator: Generator) -> None:
    task_without_solution = {k: v for k, v in VALID_TASK.items() if k != "solution"}

    validated, error = Generator._validate_response_matches_output_type(
        GeneratedTask,
        task_without_solution,
    )

    assert validated is None
    assert error is not None
    assert "solution" in error


def test_create_retry_prompt_includes_context() -> None:
    retry_prompt = Generator._create_retry_prompt(
        original_prompt="generate a task",
        original_response='{"topic": "Lego"}',
        error_message="missing field",
    )

    assert "generate a task" in retry_prompt
    assert '{"topic": "Lego"}' in retry_prompt
    assert "missing field" in retry_prompt
    assert "Respond ONLY with valid JSON" in retry_prompt


@pytest.mark.asyncio
async def test_feedback_loop_returns_on_first_valid_response(generator: Generator) -> None:
    with patch.object(
        Generator,
        "_call_llm",
        new_callable=AsyncMock,
        return_value=VALID_TASK,
    ) as mock_call:
        result = await generator.call_llm_with_output_feedback_loop(
            system_prompt="system",
            user_prompt="user",
            output_type=GeneratedTask,
            n_retry=2,
        )

    assert result == GeneratedTask.model_validate(VALID_TASK)
    mock_call.assert_awaited_once_with(
        model="test-model",
        system_prompt="system",
        user_prompt="user",
        output_type=GeneratedTask,
    )


@pytest.mark.asyncio
async def test_feedback_loop_retries_then_succeeds(generator: Generator) -> None:
    with patch.object(
        Generator,
        "_call_llm",
        new_callable=AsyncMock,
        side_effect=[INVALID_TASK, VALID_TASK],
    ) as mock_call:
        result = await generator.call_llm_with_output_feedback_loop(
            system_prompt="system",
            user_prompt="user",
            output_type=GeneratedTask,
            n_retry=2,
        )

    assert result == GeneratedTask.model_validate(VALID_TASK)
    assert mock_call.await_count == 2

    retry_call = mock_call.await_args_list[1]
    assert retry_call.kwargs["user_prompt"] != "user"
    assert "validation error" in retry_call.kwargs["user_prompt"].lower()


@pytest.mark.asyncio
async def test_feedback_loop_raises_after_max_retries(generator: Generator) -> None:
    with patch.object(
        Generator,
        "_call_llm",
        new_callable=AsyncMock,
        return_value=INVALID_TASK,
    ) as mock_call:
        with pytest.raises(RuntimeError, match="Max retries reached"):
            await generator.call_llm_with_output_feedback_loop(
                system_prompt="system",
                user_prompt="user",
                output_type=GeneratedTask,
                n_retry=1,
            )

    assert mock_call.await_count == 2


@pytest.mark.asyncio
async def test_generate_returns_validated_task(generator: Generator) -> None:
    prompt = SamplePrompt()

    with patch.object(
        Generator,
        "_call_llm",
        new_callable=AsyncMock,
        return_value=VALID_TASK,
    ):
        result = await generator.generate(prompt, GeneratedTask)

    assert result == GeneratedTask.model_validate(VALID_TASK)


@pytest.mark.asyncio
async def test_generate_uses_custom_system_prompt(generator: Generator) -> None:
    prompt = SamplePrompt()
    custom_system_prompt = "Custom teacher instructions."

    with patch.object(
        Generator,
        "_call_llm",
        new_callable=AsyncMock,
        return_value=VALID_TASK,
    ) as mock_call:
        await generator.generate(
            prompt,
            GeneratedTask,
            system_prompt=custom_system_prompt,
        )

    assert mock_call.await_args.kwargs["system_prompt"] == custom_system_prompt


@pytest.mark.asyncio
async def test_generate_passes_built_user_prompt(generator: Generator) -> None:
    prompt = SamplePrompt(topic="Space")

    with patch.object(
        Generator,
        "_call_llm",
        new_callable=AsyncMock,
        return_value=VALID_TASK,
    ) as mock_call:
        await generator.generate(prompt, GeneratedTask)

    assert mock_call.await_args.kwargs["user_prompt"] == "Generate a math problem about Space."


@pytest.mark.asyncio
async def test_generate_uses_default_system_prompt_when_not_overridden(
    generator: Generator,
) -> None:
    prompt = SamplePrompt()

    with patch.object(
        Generator,
        "_call_llm",
        new_callable=AsyncMock,
        return_value=VALID_TASK,
    ) as mock_call:
        await generator.generate(prompt, GeneratedTask)

    assert mock_call.await_args.kwargs["system_prompt"] == DEFAULT_SYSTEM_PROMPT

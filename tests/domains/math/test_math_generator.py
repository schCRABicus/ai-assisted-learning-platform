from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from pydantic import ValidationError

from agentic_learning_portal.api.generator import DEFAULT_SYSTEM_PROMPT, Generator
from agentic_learning_portal.api.model import GeneratedTask, VerificationResult
from agentic_learning_portal.api.progress import CollectingProgressListener, ProgressEvent
from agentic_learning_portal.domains.math.generator import MathProblemGenerator
from agentic_learning_portal.domains.math.wa_judge import WolframAlphaJudge
from agentic_learning_portal.domains.math.model import MathProblemGenerationPromptInput

VALID_MATH_PROBLEM = {
    "topic": "Lego",
    "text": "You have 3 boxes with 4 bricks each. How many bricks do you have?",
    "complexity": "easy",
    "correct_answer": 12,
    "solution": "Each box holds 4 bricks and there are 3 boxes, so the total is 3 × 4 = 12 bricks.",
}

INVALID_MATH_PROBLEM = {
    "topic": "Lego",
    "text": "You have 3 boxes with 4 bricks each. How many bricks do you have?",
    "complexity": "super-hard",
    "correct_answer": 12,
    "solution": "Each box holds 4 bricks and there are 3 boxes, so the total is 3 × 4 = 12 bricks.",
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

    assert result == GeneratedTask.model_validate(VALID_MATH_PROBLEM)


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
    assert mock_call.await_args.kwargs["output_type"] is GeneratedTask


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


def _verification_result(verified: bool) -> VerificationResult:
    return VerificationResult(
        judge="wolframalpha",
        verified=verified,
        expected_answer=12,
        judge_answer="12" if verified else "15",
        detail="Answers match." if verified else "Expected 12, Wolfram|Alpha returned '15'.",
    )


@pytest.mark.asyncio
async def test_generate_returns_validated_task_without_judge(
    generator: MathProblemGenerator,
    prompt_input: MathProblemGenerationPromptInput,
) -> None:
    with patch.object(
        Generator,
        "_call_llm",
        new_callable=AsyncMock,
        return_value=VALID_MATH_PROBLEM,
    ) as mock_call:
        result = await generator.generate(prompt_input)

    assert result == GeneratedTask.model_validate(VALID_MATH_PROBLEM)
    mock_call.assert_awaited_once()


@pytest.mark.asyncio
async def test_generate_verifies_with_judge(
    generator: MathProblemGenerator,
    prompt_input: MathProblemGenerationPromptInput,
) -> None:
    judge = WolframAlphaJudge(app_id="test-app-id")

    with (
        patch.object(
            Generator,
            "_call_llm",
            new_callable=AsyncMock,
            return_value=VALID_MATH_PROBLEM,
        ),
        patch.object(
            judge,
            "verify",
            new_callable=AsyncMock,
            return_value=_verification_result(True),
        ) as mock_verify,
    ):
        result = await generator.generate(prompt_input, judge=judge)

    assert result == GeneratedTask.model_validate(VALID_MATH_PROBLEM)
    mock_verify.assert_awaited_once()


@pytest.mark.asyncio
async def test_generate_retries_on_judge_mismatch(
    generator: MathProblemGenerator,
    prompt_input: MathProblemGenerationPromptInput,
) -> None:
    judge = WolframAlphaJudge(app_id="test-app-id")

    with (
        patch.object(
            Generator,
            "_call_llm",
            new_callable=AsyncMock,
            return_value=VALID_MATH_PROBLEM,
        ) as mock_call,
        patch.object(
            judge,
            "verify",
            new_callable=AsyncMock,
            side_effect=[_verification_result(False), _verification_result(True)],
        ),
    ):
        result = await generator.generate(prompt_input, judge=judge)

    assert result == GeneratedTask.model_validate(VALID_MATH_PROBLEM)
    assert mock_call.await_count == 2
    retry_prompt = mock_call.await_args_list[1].kwargs["user_prompt"]
    assert "could not confirm" in retry_prompt
    assert "correct_answer" in retry_prompt
    assert "15" not in retry_prompt


@pytest.mark.asyncio
async def test_generate_raises_after_max_retries_on_judge_mismatch(
    generator: MathProblemGenerator,
    prompt_input: MathProblemGenerationPromptInput,
) -> None:
    judge = WolframAlphaJudge(app_id="test-app-id")

    with (
        patch.object(
            Generator,
            "_call_llm",
            new_callable=AsyncMock,
            return_value=VALID_MATH_PROBLEM,
        ) as mock_call,
        patch.object(
            judge,
            "verify",
            new_callable=AsyncMock,
            return_value=_verification_result(False),
        ),
    ):
        with pytest.raises(RuntimeError, match="Max retries reached"):
            await generator.generate(prompt_input, judge=judge, retries=2)

    assert mock_call.await_count == 2


@pytest.mark.asyncio
async def test_generate_raises_after_max_retries_on_schema_failure(
    generator: MathProblemGenerator,
    prompt_input: MathProblemGenerationPromptInput,
) -> None:
    with patch.object(
        Generator,
        "_call_llm",
        new_callable=AsyncMock,
        return_value=INVALID_MATH_PROBLEM,
    ) as mock_call:
        with pytest.raises(RuntimeError, match="Max retries reached"):
            await generator.generate(prompt_input, retries=2)

    assert mock_call.await_count == 2


@pytest.mark.asyncio
async def test_generate_retries_on_schema_failure_then_succeeds(
    generator: MathProblemGenerator,
    prompt_input: MathProblemGenerationPromptInput,
) -> None:
    judge = WolframAlphaJudge(app_id="test-app-id")

    with (
        patch.object(
            Generator,
            "_call_llm",
            new_callable=AsyncMock,
            side_effect=[INVALID_MATH_PROBLEM, VALID_MATH_PROBLEM],
        ),
        patch.object(
            judge,
            "verify",
            new_callable=AsyncMock,
            return_value=_verification_result(True),
        ),
    ):
        result = await generator.generate(prompt_input, judge=judge)

    assert result == GeneratedTask.model_validate(VALID_MATH_PROBLEM)


def test_create_verification_retry_prompt_includes_context() -> None:
    task = GeneratedTask.model_validate(VALID_MATH_PROBLEM)

    retry_prompt = MathProblemGenerator._create_verification_retry_prompt(
        original_prompt="generate a task",
        task=task,
        verification=_verification_result(False),
    )

    assert "generate a task" in retry_prompt
    assert task.model_dump_json() in retry_prompt
    assert "could not confirm" in retry_prompt
    assert "respond ONLY with valid JSON" in retry_prompt


def test_create_verification_retry_prompt_excludes_judge_answer() -> None:
    task = GeneratedTask.model_validate(VALID_MATH_PROBLEM)

    retry_prompt = MathProblemGenerator._create_verification_retry_prompt(
        original_prompt="generate a task",
        task=task,
        verification=_verification_result(False),
    )

    assert "15" not in retry_prompt
    assert "recompute" in retry_prompt.lower()


def test_create_verification_retry_prompt_asks_to_clarify_when_judge_cannot_compute() -> None:
    task = GeneratedTask.model_validate(VALID_MATH_PROBLEM)
    inconclusive = VerificationResult(
        judge="wolframalpha",
        verified=False,
        expected_answer=12,
        judge_answer=None,
        detail="Could not translate the problem into a Wolfram|Alpha query.",
    )

    retry_prompt = MathProblemGenerator._create_verification_retry_prompt(
        original_prompt="generate a task",
        task=task,
        verification=inconclusive,
    )

    assert "could not compute" in retry_prompt
    assert "ambiguous" in retry_prompt


# --- multi-judge --------------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_multi_judge_all_pass(
    generator: MathProblemGenerator,
    prompt_input: MathProblemGenerationPromptInput,
) -> None:
    judge_a = WolframAlphaJudge(app_id="test-app-id")
    judge_b = WolframAlphaJudge(app_id="test-app-id")

    with (
        patch.object(
            Generator,
            "_call_llm",
            new_callable=AsyncMock,
            return_value=VALID_MATH_PROBLEM,
        ),
        patch.object(
            judge_a,
            "verify",
            new_callable=AsyncMock,
            return_value=_verification_result(True),
        ),
        patch.object(
            judge_b,
            "verify",
            new_callable=AsyncMock,
            return_value=_verification_result(True),
        ),
    ):
        result = await generator.generate(prompt_input, judge=[judge_a, judge_b])

    assert result == GeneratedTask.model_validate(VALID_MATH_PROBLEM)


@pytest.mark.asyncio
async def test_generate_multi_judge_runs_both_judges(
    generator: MathProblemGenerator,
    prompt_input: MathProblemGenerationPromptInput,
) -> None:
    judge_a = WolframAlphaJudge(app_id="test-app-id")
    judge_b = WolframAlphaJudge(app_id="test-app-id")

    with (
        patch.object(
            Generator,
            "_call_llm",
            new_callable=AsyncMock,
            return_value=VALID_MATH_PROBLEM,
        ),
        patch.object(
            judge_a,
            "verify",
            new_callable=AsyncMock,
            return_value=_verification_result(True),
        ) as mock_a,
        patch.object(
            judge_b,
            "verify",
            new_callable=AsyncMock,
            return_value=_verification_result(True),
        ) as mock_b,
    ):
        await generator.generate(prompt_input, judge=[judge_a, judge_b])

    mock_a.assert_awaited_once()
    mock_b.assert_awaited_once()


@pytest.mark.asyncio
async def test_generate_multi_judge_retries_until_all_pass(
    generator: MathProblemGenerator,
    prompt_input: MathProblemGenerationPromptInput,
) -> None:
    judge_a = WolframAlphaJudge(app_id="test-app-id")
    judge_b = WolframAlphaJudge(app_id="test-app-id")

    with (
        patch.object(
            Generator,
            "_call_llm",
            new_callable=AsyncMock,
            return_value=VALID_MATH_PROBLEM,
        ) as mock_call,
        patch.object(
            judge_a,
            "verify",
            new_callable=AsyncMock,
            side_effect=[_verification_result(True), _verification_result(True)],
        ),
        patch.object(
            judge_b,
            "verify",
            new_callable=AsyncMock,
            side_effect=[_verification_result(False), _verification_result(True)],
        ),
    ):
        result = await generator.generate(prompt_input, judge=[judge_a, judge_b])

    assert result == GeneratedTask.model_validate(VALID_MATH_PROBLEM)
    assert mock_call.await_count == 2


@pytest.mark.asyncio
async def test_generate_multi_judge_raises_after_max_retries(
    generator: MathProblemGenerator,
    prompt_input: MathProblemGenerationPromptInput,
) -> None:
    judge_a = WolframAlphaJudge(app_id="test-app-id")
    judge_b = WolframAlphaJudge(app_id="test-app-id")

    with (
        patch.object(
            Generator,
            "_call_llm",
            new_callable=AsyncMock,
            return_value=VALID_MATH_PROBLEM,
        ) as mock_call,
        patch.object(
            judge_a,
            "verify",
            new_callable=AsyncMock,
            return_value=_verification_result(False),
        ),
        patch.object(
            judge_b,
            "verify",
            new_callable=AsyncMock,
            return_value=_verification_result(True),
        ),
    ):
        with pytest.raises(RuntimeError, match="Max retries reached"):
            await generator.generate(prompt_input, judge=[judge_a, judge_b], retries=2)

    assert mock_call.await_count == 2


def test_create_verification_retry_prompt_combines_multiple_judges() -> None:
    task = GeneratedTask.model_validate(VALID_MATH_PROBLEM)
    results = [
        _verification_result(False),
        VerificationResult(
            judge="qwen",
            verified=False,
            expected_answer=12,
            judge_answer=None,
            detail="Groq could not produce an answer for the problem.",
        ),
    ]

    retry_prompt = MathProblemGenerator._create_verification_retry_prompt(
        original_prompt="generate a task",
        task=task,
        verification=results,
    )

    assert "wolframalpha" in retry_prompt
    assert "qwen judge" in retry_prompt
    assert "could not confirm" in retry_prompt
    assert "could not compute" in retry_prompt
    assert "15" not in retry_prompt


# --- progress listener --------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_emits_progress_events_without_judge(
    generator: MathProblemGenerator,
    prompt_input: MathProblemGenerationPromptInput,
) -> None:
    log: list[ProgressEvent] = []
    with patch.object(
        Generator,
        "_call_llm",
        new_callable=AsyncMock,
        return_value=VALID_MATH_PROBLEM,
    ):
        await generator.generate(prompt_input, listener=CollectingProgressListener(log))

    stages = [e.stage for e in log]
    assert "generate" in stages
    assert "validate" in stages
    assert stages[-1] == "done"
    generate_event = next(e for e in log if e.stage == "generate")
    assert generate_event.attempt == 1
    assert generate_event.total == 5


@pytest.mark.asyncio
async def test_generate_emits_verify_retry_and_done_with_judge(
    generator: MathProblemGenerator,
    prompt_input: MathProblemGenerationPromptInput,
) -> None:
    judge = WolframAlphaJudge(app_id="test-app-id")
    log: list[ProgressEvent] = []
    with (
        patch.object(
            Generator,
            "_call_llm",
            new_callable=AsyncMock,
            return_value=VALID_MATH_PROBLEM,
        ),
        patch.object(
            judge,
            "verify",
            new_callable=AsyncMock,
            side_effect=[_verification_result(False), _verification_result(True)],
        ),
    ):
        result = await generator.generate(
            prompt_input, judge=judge, listener=CollectingProgressListener(log)
        )

    assert result == GeneratedTask.model_validate(VALID_MATH_PROBLEM)
    stages = [e.stage for e in log]
    assert "verify" in stages
    assert "retry" in stages
    assert stages[-1] == "done"


@pytest.mark.asyncio
async def test_generate_passes_listener_to_judges(
    generator: MathProblemGenerator,
    prompt_input: MathProblemGenerationPromptInput,
) -> None:
    judge = WolframAlphaJudge(app_id="test-app-id")
    log: list[ProgressEvent] = []
    with (
        patch.object(
            Generator,
            "_call_llm",
            new_callable=AsyncMock,
            return_value=VALID_MATH_PROBLEM,
        ),
        patch.object(
            judge,
            "verify",
            new_callable=AsyncMock,
            return_value=_verification_result(True),
        ) as mock_verify,
    ):
        await generator.generate(
            prompt_input, judge=judge, listener=CollectingProgressListener(log)
        )

    mock_verify.assert_awaited_once()
    assert isinstance(
        mock_verify.await_args.kwargs["listener"],
        CollectingProgressListener,
    )


@pytest.mark.asyncio
async def test_generate_emits_error_stage_on_exhaustion(
    generator: MathProblemGenerator,
    prompt_input: MathProblemGenerationPromptInput,
) -> None:
    judge = WolframAlphaJudge(app_id="test-app-id")
    log: list[ProgressEvent] = []
    with (
        patch.object(
            Generator,
            "_call_llm",
            new_callable=AsyncMock,
            return_value=VALID_MATH_PROBLEM,
        ),
        patch.object(
            judge,
            "verify",
            new_callable=AsyncMock,
            return_value=_verification_result(False),
        ),
    ):
        with pytest.raises(RuntimeError, match="Max retries reached"):
            await generator.generate(
                prompt_input,
                judge=judge,
                retries=2,
                listener=CollectingProgressListener(log),
            )

    assert log[-1].stage == "error"

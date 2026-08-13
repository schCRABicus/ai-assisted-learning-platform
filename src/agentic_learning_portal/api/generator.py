from __future__ import annotations

from typing import TypeVar

from pydantic import BaseModel, ValidationError

from agentic_learning_portal.api.llm import ask_ai_for_structured_response
from agentic_learning_portal.api.model import GeneratedTask, ProblemGenerationPromptInput

InputT = TypeVar("InputT", bound=ProblemGenerationPromptInput)
OutputT = TypeVar("OutputT", bound=GeneratedTask)

DEFAULT_SYSTEM_PROMPT = (
    "You are an experienced teacher. "
    "Generate a structured learning task that matches the given parameters. "
    "Include the problem text, the correct answer, and a step-by-step solution "
    "explaining how to arrive at that answer. "
    "Write all math in plain text (for example '3 × 4 = 12'), never in LaTeX — "
    "no dollar signs, backslashes, or LaTeX commands like \\text{} or \\frac{}."
)


class Generator:
    """Async API for generating structured tasks from prompt inputs via an LLM."""

    def __init__(self, model: str = "google:gemini-3.5-flash") -> None:
        self._model = model

    async def generate(
        self,
        prompt_input: InputT,
        output_type: type[OutputT],
        *,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    ) -> OutputT:
        return await self.call_llm_with_output_feedback_loop(
            system_prompt=system_prompt,
            user_prompt=prompt_input.build_user_prompt(),
            output_type=output_type,
        )

    async def call_llm_with_output_feedback_loop(
        self,
        system_prompt: str,
        user_prompt: str,
        output_type: type[OutputT],
        n_retry: int = 5,
    ) -> OutputT:
        response_content = await Generator._call_llm(
            model=self._model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            output_type=output_type,
        )
        current_prompt = user_prompt

        for attempt in range(n_retry + 1):
            validated_data, validation_error = Generator._validate_response_matches_output_type(
                output_type=output_type,
                llm_response=response_content,
            )

            if validation_error is None:
                return validated_data

            if attempt >= n_retry:
                raise RuntimeError(f"Max retries reached. Last error: {validation_error}")

            validation_retry_prompt = Generator._create_retry_prompt(
                original_prompt=current_prompt,
                original_response=response_content,
                error_message=validation_error,
            )
            response_content = await Generator._call_llm(
                model=self._model,
                system_prompt=system_prompt,
                user_prompt=validation_retry_prompt,
                output_type=output_type,
            )
            current_prompt = validation_retry_prompt

        raise RuntimeError("Unexpected end of retry loop")

    @staticmethod
    async def _call_llm(
        model: str,
        system_prompt: str,
        user_prompt: str,
        output_type: type[OutputT],
    ) -> object:
        return await ask_ai_for_structured_response(
            model=model,
            output_type=output_type,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )

    @staticmethod
    def _validate_response_matches_output_type(
        output_type: type[OutputT],
        llm_response: object,
    ) -> tuple[OutputT | None, str | None]:
        try:
            if isinstance(llm_response, str):
                validated_data = output_type.model_validate_json(llm_response)
            elif isinstance(llm_response, BaseModel):
                validated_data = output_type.model_validate_json(llm_response.model_dump_json())
            else:
                validated_data = output_type.model_validate(llm_response)
            return validated_data, None
        except ValidationError as e:
            error_message = f"This response generated a validation error: {e}."
            return None, error_message

    @staticmethod
    def _create_retry_prompt(
        original_prompt: str,
        original_response: object,
        error_message: str,
    ) -> str:
        return f"""
        This is a request to fix an error in the structure of an llm_response.
        Here is the original request:
        <original_prompt>
        {original_prompt}
        </original_prompt>

        Here is the original llm_response:
        <llm_response>
        {original_response}
        </llm_response>

        This response generated an error:
        <error_message>
        {error_message}
        </error_message>

        Compare the error message and the llm_response and identify what
        needs to be fixed or removed in the llm_response to resolve this error.

        Respond ONLY with valid JSON. Do not include any explanations or
        other text or formatting before or after the JSON string.
        """

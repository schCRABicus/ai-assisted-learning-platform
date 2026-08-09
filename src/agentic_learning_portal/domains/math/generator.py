import logging

from agentic_learning_portal.api.generator import DEFAULT_SYSTEM_PROMPT, Generator
from agentic_learning_portal.api.model import GeneratedTask, VerificationResult
from agentic_learning_portal.domains.math.judge import WolframAlphaJudge
from agentic_learning_portal.domains.math.model import MathProblemGenerationPromptInput

logger = logging.getLogger(__name__)


class MathProblemGenerator(Generator):
    """Math Problem Generator class."""

    async def generate(
        self,
        prompt_input: MathProblemGenerationPromptInput,
        *,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        judge: WolframAlphaJudge | None = None,
        retries: int = 5,
    ) -> GeneratedTask:
        """Generate, validate, and (optionally) verify a task in one flow.

        Each attempt calls the LLM and validates the response against
        ``GeneratedTask``. When a ``judge`` is provided, the answer is also
        verified against ground truth. Any failure is fed back to the LLM as a
        corrective prompt and retried, up to ``retries`` attempts total.
        Raises ``RuntimeError`` if no validated task is produced.
        """
        user_prompt = prompt_input.build_user_prompt()

        for attempt in range(1, retries + 1):
            logger.debug("LLM generation attempt %d/%d", attempt, retries)
            response_content = await Generator._call_llm(
                model=self._model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                output_type=GeneratedTask,
            )
            task, validation_error = Generator._validate_response_matches_output_type(
                output_type=GeneratedTask,
                llm_response=response_content,
            )
            if validation_error is not None or task is None:
                logger.warning(
                    "Attempt %d/%d failed validation; retrying: %s",
                    attempt,
                    retries,
                    validation_error or "unknown validation error",
                )
                user_prompt = Generator._create_retry_prompt(
                    original_prompt=user_prompt,
                    original_response=response_content,
                    error_message=validation_error or "unknown validation error",
                )
                continue

            logger.info("LLM generation attempt %d/%d - task generated! %s", attempt, retries, task)
            if judge is not None:
                logger.info("Verifying the generated task's answer correctness by a judge llm call...")
                verification = await judge.verify(task)
                if verification.verified:
                    logger.info(
                        "Attempt %d/%d: task verified by %s",
                        attempt,
                        retries,
                        verification.judge,
                    )
                    return task

                logger.warning(
                    "Attempt %d/%d: judge rejected the task; retrying: %s",
                    attempt,
                    retries,
                    verification.detail,
                )
                user_prompt = MathProblemGenerator._create_verification_retry_prompt(
                    original_prompt=user_prompt,
                    task=task,
                    verification=verification,
                )
            else:
                return task

        logger.error("Could not generate a validated task after %d attempts", retries)
        raise RuntimeError(
            f"Max retries reached. Could not generate a validated task "
            f"after {retries} attempts."
        )

    @staticmethod
    def _create_verification_retry_prompt(
        original_prompt: str,
        task: GeneratedTask,
        verification: VerificationResult,
    ) -> str:
        # Deliberately do NOT include the judge's answer: the LLM must recompute
        # independently, because the judge can be wrong.
        if verification.judge_answer is None:
            judge_feedback = (
                "The judge could not compute an answer from the problem. The "
                "question may be ambiguous or hard to parse — clarify the "
                "wording while keeping the same topic and difficulty."
            )
        else:
            judge_feedback = (
                "The judge could not confirm the answer. Recompute it "
                "independently — the judge may be wrong, so do not copy its "
                "conclusion. Make sure `correct_answer` matches your own "
                "calculation."
            )

        return f"""
        This is a request to regenerate a task that an independent judge could
        not confirm.
        Here is the original request:
        <original_prompt>
        {original_prompt}
        </original_prompt>

        Here is the task you generated:
        <generated_task>
        {task.model_dump_json()}
        </generated_task>

        <judge_feedback>
        {judge_feedback}
        </judge_feedback>

        Regenerate the task and respond ONLY with valid JSON. Do not include
        any explanations or other text or formatting before or after the JSON.
        """
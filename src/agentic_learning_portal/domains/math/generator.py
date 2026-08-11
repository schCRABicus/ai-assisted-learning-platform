import asyncio
import logging
from collections.abc import Sequence

from agentic_learning_portal.api.generator import DEFAULT_SYSTEM_PROMPT, Generator
from agentic_learning_portal.api.model import GeneratedTask, Judge, VerificationResult
from agentic_learning_portal.domains.math.model import MathProblemGenerationPromptInput

logger = logging.getLogger(__name__)


class MathProblemGenerator(Generator):
    """Math Problem Generator class."""

    async def generate(
        self,
        prompt_input: MathProblemGenerationPromptInput,
        *,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        judge: Judge | Sequence[Judge] | None = None,
        retries: int = 5,
    ) -> GeneratedTask:
        """Generate, validate, and (optionally) verify a task in one flow.

        Task generation — the LLM call and schema-validation retry — is
        delegated to the base class; only judge verification is this domain's
        responsibility. When one or more ``judge`` objects are provided (pass a
        single judge or a sequence of judges), the task is accepted only when
        *every* judge verifies it — a judge-ensemble consensus. A judge
        rejection is fed back to the LLM as a corrective prompt and
        re-generated, up to ``retries`` attempts total. Raises ``RuntimeError``
        if no judge-verified task is produced.
        """
        user_prompt = prompt_input.build_user_prompt()

        for attempt in range(1, retries + 1):
            logger.debug("LLM generation attempt %d/%d", attempt, retries)
            # Delegate the LLM call + schema validation to the base class.
            # ``Generator.generate()`` only wraps this method, but it rebuilds
            # the prompt from ``prompt_input``, so call the lower-level method
            # to send the corrective prompt after a judge rejection.
            task = await super().call_llm_with_output_feedback_loop(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                output_type=GeneratedTask,
                n_retry=1,
            )

            logger.info("LLM generation attempt %d/%d - task generated! %s", attempt, retries, task)
            if judge is None:
                return task

            judges = list(judge) if isinstance(judge, Sequence) else [judge]
            logger.info(
                "Verifying the generated task's answer correctness by %d judge(s)...",
                len(judges),
            )
            verifications = await asyncio.gather(
                *(each.verify(task) for each in judges)
            )
            if all(v.verified for v in verifications):
                logger.info(
                    "Attempt %d/%d: task verified by all judges",
                    attempt,
                    retries,
                )
                return task

            failing = [v for v in verifications if not v.verified]
            logger.warning(
                "Attempt %d/%d: %d/%d judges rejected the task; retrying",
                attempt,
                retries,
                len(failing),
                len(verifications),
            )
            user_prompt = self._create_verification_retry_prompt(
                original_prompt=user_prompt,
                task=task,
                verification=failing[0] if len(failing) == 1 else failing,
            )

        logger.error("Could not generate a validated task after %d attempts", retries)
        raise RuntimeError(
            f"Max retries reached. Could not generate a validated task "
            f"after {retries} attempts."
        )

    @staticmethod
    def _create_verification_retry_prompt(
        original_prompt: str,
        task: GeneratedTask,
        verification: VerificationResult | Sequence[VerificationResult],
    ) -> str:
        # Deliberately do NOT include any judge's answer: the LLM must recompute
        # independently, because a judge can be wrong.
        results = (
            [verification]
            if isinstance(verification, VerificationResult)
            else list(verification)
        )
        judge_feedback = "\n\n".join(
            MathProblemGenerator._verification_feedback(result)
            for result in results
        )

        return f"""
        This is a request to regenerate a task that independent judge(s) could
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

    @staticmethod
    def _verification_feedback(verification: VerificationResult) -> str:
        """Feedback for one judge's rejection, without leaking its answer."""
        if verification.judge_answer is None:
            return (
                f"The {verification.judge} judge could not compute an answer "
                "from the problem. The question may be ambiguous or hard to "
                "parse — clarify the wording while keeping the same topic and "
                "difficulty."
            )
        return (
            f"The {verification.judge} judge could not confirm the answer. "
            "Recompute it independently — the judge may be wrong, so do not "
            "copy its conclusion. Make sure `correct_answer` matches your own "
            "calculation."
        )

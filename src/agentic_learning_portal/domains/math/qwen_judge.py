"""LLM-based judge for generated math tasks using Qwen on Groq."""

from __future__ import annotations

import logging
import os
import re

from agentic_learning_portal.api.llm import LLMError, ask_ai_for_text_response
from agentic_learning_portal.api.model import GeneratedTask, Judge, VerificationResult
from agentic_learning_portal.api.progress import ProgressListener, report_progress
from agentic_learning_portal.domains.math._comparison import _answers_match, _to_number

logger = logging.getLogger(__name__)

GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"

# Qwen models conventionally close their final answer with \boxed{...}.
BOXED_ANSWER_PATTERN = re.compile(r"\\boxed\{([^}]*)\}")
# Integers, decimals, and fractions (e.g. "42", "3.14", "1/2").
NUMBER_PATTERN = re.compile(r"-?\d+(?:\.\d+)?(?:/\d+)?")

SOLVER_SYSTEM_PROMPT = (
    "You are a math solver. Read the following math problem and output ONLY "
    "the final numeric answer. Do not include reasoning, explanations, or "
    "step-by-step solutions. Output just the answer as a single number or "
    "expression."
)


class QwenMathJudge(Judge):
    """Judge generated tasks with a Qwen model served by Groq.

    Queries ``qwen/qwen3.6-27b`` through Groq's OpenAI-compatible chat API:
    the problem's ``text`` is sent as the user message, and the model's reply
    is mined for a single numeric answer. The model is a reasoning model — it
    wraps chain-of-thought in ``<think>...</think>`` and answers afterwards —
    so answer extraction ignores everything up to the last ``</think>``.

    This is an LLM-based judge — distinct from Wolfram|Alpha's symbolic ground
    truth. Qwen runs on Groq, a different provider from the Gemini task
    generator, so it provides an independent second opinion for multi-judge
    verification. Override the model via ``model``.
    """

    def __init__(
        self,
        api_key: str | None = None,
        *,
        model: str = "qwen/qwen3.6-27b",
        timeout: float = 60.0,
        max_tokens: int = 512,
    ) -> None:
        self._api_key = (
            api_key if api_key is not None else os.environ.get("GROQ_API_KEY", "")
        )
        self._model = model
        self._timeout = timeout
        self._max_tokens = max_tokens

    async def verify(
        self,
        task: GeneratedTask,
        *,
        listener: ProgressListener | None = None,
    ) -> VerificationResult:
        logger.info("Verifying task with Qwen judge...")
        await report_progress(listener, "verify", "Qwen: solving the problem...")
        response_text = await self._query_groq(task.text)
        if not response_text:
            await report_progress(
                listener,
                "verify",
                "Qwen: could not produce an answer.",
            )
            return VerificationResult(
                judge="qwen",
                verified=False,
                expected_answer=task.correct_answer,
                judge_answer=None,
                detail="Groq could not produce an answer for the problem.",
            )

        answer = self._extract_answer(response_text)
        verified = bool(answer) and _answers_match(task.correct_answer, answer)
        await report_progress(
            listener,
            "verify",
            "Qwen: "
            + ("verified" if verified else "could not verify")
            + " the answer.",
        )

        logger.info(
            "Qwen verdict: verified=%s (expected=%r, answer=%r)",
            verified,
            task.correct_answer,
            answer,
        )

        if not answer:
            detail = f"Qwen produced no parseable answer: {response_text!r}"
        elif verified:
            detail = "Answers match."
        else:
            detail = (
                f"Expected {task.correct_answer!r}, "
                f"Qwen returned {answer!r}."
            )

        return VerificationResult(
            judge="qwen",
            verified=verified,
            expected_answer=task.correct_answer,
            judge_answer=answer,
            detail=detail,
        )

    async def _query_groq(self, task_text: str) -> str | None:
        """Ask Qwen via Groq's chat API and return the assistant's message.

        Returns ``None`` when the model could not answer — a rate limit
        (HTTP 429), a server/transport failure, or a malformed reply — so the
        caller can treat the task as unverifiable rather than crash the
        generation pipeline. The HTTP call itself runs through
        ``ask_ai_for_text_response``, which retries transient failures with backoff
        before giving up. Split out so tests can mock it without a network call.
        """
        if not self._api_key:
            raise RuntimeError(
                "GROQ_API_KEY not set. Add it to .env or pass api_key "
                "to QwenMathJudge()."
            )

        try:
            return await ask_ai_for_text_response(
                url=GROQ_CHAT_URL,
                api_key=self._api_key,
                model=self._model,
                messages=[
                    {
                        "role": "user",
                        "content": f"{SOLVER_SYSTEM_PROMPT}\n\nProblem: {task_text}",
                    }
                ],
                timeout=self._timeout,
                max_tokens=self._max_tokens,
            )
        except LLMError as e:
            logger.warning("Groq chat completion failed: %s", e)
            return None

    def _extract_answer(self, response_text: str) -> str:
        """Extract a single numeric answer from Qwen's reply.

        A reasoning model wraps chain-of-thought in ``<think>...</think>`` and
        answers after the closing tag, so everything up to the last
        ``</think>`` is ignored to keep a stray number inside the reasoning
        from being mistaken for the answer. Then tries, in order: the
        ``\\boxed{...}`` answer (Qwen's convention), then the last number-like
        token anywhere in the remaining text. Returns ``""`` if nothing parses
        as a number.
        """
        text = response_text
        think_end = text.rfind("</think>")
        if think_end != -1:
            text = text[think_end + len("</think>") :]

        for candidate in reversed(BOXED_ANSWER_PATTERN.findall(text)):
            answer = self._parse_number_token(candidate)
            if answer:
                return answer

        for candidate in reversed(NUMBER_PATTERN.findall(text)):
            answer = self._parse_number_token(candidate)
            if answer:
                return answer

        return ""

    @staticmethod
    def _parse_number_token(token: str) -> str:
        """Return ``token`` stripped if it parses as a number, else ``""``."""
        token = token.strip().strip(".,;:!?()")
        if _to_number(token) is not None:
            return token
        return ""
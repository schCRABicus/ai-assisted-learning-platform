"""Ground-truth judge for generated math tasks using Wolfram|Alpha."""

from __future__ import annotations

import logging
import os

import httpx
from pydantic import BaseModel, Field

from agentic_learning_portal.api.llm import MODELS, ask_ai_for_structured_response
from agentic_learning_portal.api.model import GeneratedTask, Judge, VerificationResult
from agentic_learning_portal.api.progress import ProgressListener, report_progress
from agentic_learning_portal.domains.math._comparison import _answers_match

logger = logging.getLogger(__name__)


class MathQuery(BaseModel):
    """A bare arithmetic expression extracted from a word problem."""

    query: str = Field(
        ...,
        description="A single Wolfram|Alpha-compatible expression of the problem's computation.",
    )


TRANSLATOR_SYSTEM_PROMPT = (
    "You are a math parser. Given a word problem, extract the computation and "
    "express it as a single arithmetic expression that Wolfram|Alpha can "
    'evaluate directly. Example: "7 boxes of 12, 28 destroyed, split among 8 '
    'lines" becomes "(7*12-28)/8". Respond with a JSON object containing only '
    'the "query" field.'
)


class WolframAlphaJudge(Judge):
    """Judge generated tasks against Wolfram|Alpha as ground truth.

    Wolfram|Alpha can't parse themed narrative, so the task's ``text`` is first
    translated into a bare arithmetic expression by a small LLM call. The
    translation uses the judge's own ``model``, which defaults to the
    ``wolfram_translation`` entry in ``api.llm.MODELS`` — a different model
    family on a different provider than the Gemini task generator, so the
    judge does not share the authoring LLM's systematic blind spots. Override
    via ``model``.
    """

    def __init__(
        self,
        app_id: str | None = None,
        *,
        timeout: float = 30.0,
        model: str = MODELS["wolfram_translation"],
    ) -> None:
        self._app_id = app_id if app_id is not None else os.environ.get("WOLFRAM_APP_ID", "")
        self._timeout = timeout
        # The judge's own LLM, used for translation. Defaults to a Groq-hosted
        # gpt-oss-120b — independent of the Gemini task generator — so the
        # judge doesn't share its blind spots.
        self._model = model

    async def verify(
        self,
        task: GeneratedTask,
        *,
        listener: ProgressListener | None = None,
    ) -> VerificationResult:
        logger.info("Verifying task...")
        await report_progress(
            listener,
            "verify",
            "Wolfram|Alpha: translating the problem into an expression...",
        )
        query = await self._translate_to_query(task.text)
        logger.info("Task translated into query = %s", query)
        if not query:
            await report_progress(
                listener,
                "verify",
                "Wolfram|Alpha: could not translate the problem into an expression.",
            )
            return VerificationResult(
                judge="wolframalpha",
                verified=False,
                expected_answer=task.correct_answer,
                judge_answer=None,
                detail="Could not translate the problem into a Wolfram|Alpha query.",
            )

        logger.info("Querying Wolfram|Alpha with: %r", query)
        await report_progress(
            listener,
            "verify",
            "Wolfram|Alpha: computing the answer...",
        )
        wolfram_answer = await self._query_wolfram(query)
        verified = _answers_match(task.correct_answer, wolfram_answer)
        await report_progress(
            listener,
            "verify",
            "Wolfram|Alpha: "
            + ("verified" if verified else "could not verify")
            + " the answer.",
        )

        logger.info(
            "Wolfram|Alpha verdict: verified=%s (expected=%r, wolfram=%r)",
            verified,
            task.correct_answer,
            wolfram_answer,
        )

        if verified:
            detail = "Answers match."
        elif wolfram_answer is None:
            detail = "Wolfram|Alpha could not produce an answer for the problem."
        else:
            detail = (
                f"Expected {task.correct_answer!r}, "
                f"Wolfram|Alpha returned {wolfram_answer!r}."
            )

        return VerificationResult(
            judge="wolframalpha",
            verified=verified,
            expected_answer=task.correct_answer,
            judge_answer=wolfram_answer,
            detail=detail,
        )

    async def _translate_to_query(self, text: str, *, retries: int = 3) -> str:
        """Extract a Wolfram|Alpha-computable expression from a word problem.

        The translation is a single best-effort LLM call, retried only when it
        errors or returns nothing (a transient provider failure). If no query is
        produced the method returns ``""`` and the task is treated as
        unverifiable rather than crash the generation pipeline.
        """
        for attempt in range(1, retries + 1):
            query = await self._translate_once(text)
            if query:
                return query
            logger.warning(
                "Translation produced no query (attempt %d/%d)",
                attempt,
                retries,
            )
        return ""

    async def _translate_once(self, text: str) -> str:
        """Single best-effort LLM translation of a word problem into an expression.

        Returns ``""`` on any failure so the caller can treat the task as
        unverifiable rather than crash the generation pipeline.
        """
        try:
            result = await ask_ai_for_structured_response(
                model=self._model,
                output_type=MathQuery,
                system_prompt=TRANSLATOR_SYSTEM_PROMPT,
                user_prompt=text,
            )
            query = result.query if isinstance(result, MathQuery) else ""
            return query.strip()
        except Exception as e:  # noqa: BLE001 - translation is best-effort
            logger.warning("Could not translate problem into a query: %s", e)
            return ""

    async def _query_wolfram(self, query: str) -> str | None:
        """Compute ``query`` with Wolfram|Alpha and return its plain-text answer.

        Uses the Short Answers endpoint (``/v1/result``), which returns exactly
        one plain-text answer for a bare arithmetic expression — no pod XML to
        parse. Returns ``None`` when Wolfram|Alpha did not understand the query
        (HTTP 501), so the caller can treat the task as unverifiable. Uses
        httpx directly rather than the ``wolframalpha`` package, whose client
        hardcodes a 5s timeout that parses routinely exceed. Split out so tests
        can mock it without a network call.
        """
        if not self._app_id:
            raise RuntimeError(
                "WOLFRAM_APP_ID not set. Add it to .env or pass app_id "
                "to WolframAlphaJudge()."
            )

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.get(
                "https://api.wolframalpha.com/v1/result",
                params={"appid": self._app_id, "i": query},
            )
        if response.status_code == 501:
            # "Wolfram|Alpha did not understand your input"
            return None
        response.raise_for_status()
        text = response.text.strip()
        return text or None

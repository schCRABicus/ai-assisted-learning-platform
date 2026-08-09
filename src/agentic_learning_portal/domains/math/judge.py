"""Ground-truth judge for generated math tasks using Wolfram|Alpha."""

from __future__ import annotations

import logging
import math
import os
import xml.etree.ElementTree as ET
from fractions import Fraction

import httpx
from pydantic import BaseModel, Field
from pydantic_ai import Agent

from agentic_learning_portal.api.model import GeneratedTask, VerificationResult

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


def _normalize(text: str) -> str:
    """Lowercase, strip surrounding punctuation and collapse whitespace."""
    return " ".join(text.strip().strip(".,;:!?").lower().split())


def _to_number(value: str | int | float) -> float | None:
    """Parse a value into a float for numeric comparison, else None."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = value.strip()
    try:
        return float(text)
    except ValueError:
        pass
    try:
        return float(Fraction(text))
    except (ValueError, ZeroDivisionError):
        return None


def _answers_match(generated: str | int | float, wolfram: str | None) -> bool:
    """Compare the generated answer against Wolfram|Alpha's answer."""
    if wolfram is None:
        return False

    generated_number = _to_number(generated)
    wolfram_number = _to_number(wolfram)
    if generated_number is not None and wolfram_number is not None:
        return math.isclose(generated_number, wolfram_number, rel_tol=1e-6, abs_tol=1e-9)

    return _normalize(str(generated)) == _normalize(wolfram)


class WolframAlphaJudge:
    """Judge generated tasks against Wolfram|Alpha as ground truth.

    The task's ``text`` is first translated into a bare arithmetic expression
    by a small LLM call (Wolfram|Alpha can't parse themed narrative), that
    expression is sent to Wolfram|Alpha, and its answer is compared with the
    task's ``correct_answer``.
    """

    def __init__(
        self,
        app_id: str | None = None,
        *,
        timeout: float = 30.0,
        model: str = "google:gemini-3.5-flash",
    ) -> None:
        self._app_id = app_id if app_id is not None else os.environ.get("WOLFRAM_APP_ID", "")
        self._timeout = timeout
        self._model = model

    async def verify(self, task: GeneratedTask) -> VerificationResult:
        logger.info("Verifying task...")
        query = await self._translate_to_query(task.text)
        logger.info("Task translated into query = %s", query)
        if not query:
            return VerificationResult(
                judge="wolframalpha",
                verified=False,
                expected_answer=task.correct_answer,
                judge_answer=None,
                detail="Could not translate the problem into a Wolfram|Alpha query.",
            )

        logger.info("Querying Wolfram|Alpha with: %r", query)
        pods = await self._query_wolfram(query)
        wolfram_answer = self._extract_answer(pods)
        verified = _answers_match(task.correct_answer, wolfram_answer)

        logger.info(
            "Wolfram|Alpha verdict: verified=%s (expected=%r, wolfram=%r)",
            verified,
            task.correct_answer,
            wolfram_answer,
        )

        if verified:
            detail = "Answers match."
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

    async def _translate_to_query(self, text: str) -> str:
        """Extract a Wolfram|Alpha-computable expression from a word problem.

        Best-effort: returns ``""`` on any failure so the caller can treat the
        task as unverifiable rather than crash the generation pipeline.
        """
        try:
            agent = Agent(
                model=self._model,
                output_type=MathQuery,
                system_prompt=TRANSLATOR_SYSTEM_PROMPT,
            )
            response = await agent.run(text)
            query = response.output.query if isinstance(response.output, MathQuery) else ""
            return query.strip()
        except Exception as e:  # noqa: BLE001 - translation is best-effort
            logger.warning("Could not translate problem into a query: %s", e)
            return ""

    async def _query_wolfram(self, query: str) -> list[dict[str, str]]:
        """Query Wolfram|Alpha and return pod titles/text as plain dicts.

        Uses httpx directly rather than the ``wolframalpha`` package, whose
        client hardcodes a 5s timeout that natural-language parses routinely
        exceed. Split out so tests can mock it without a network call.
        """
        if not self._app_id:
            raise RuntimeError(
                "WOLFRAM_APP_ID not set. Add it to .env or pass app_id "
                "to WolframAlphaJudge()."
            )

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.get(
                "https://api.wolframalpha.com/v2/query",
                params={"appid": self._app_id, "input": query},
            )
        response.raise_for_status()

        root = ET.fromstring(response.text)
        if root.get("error") == "true":
            raise RuntimeError(f"Wolfram|Alpha query failed for: {query!r}")
        return WolframAlphaJudge._parse_pods(root)

    @staticmethod
    def _parse_pods(root: ET.Element) -> list[dict[str, str]]:
        """Extract pod title/plaintext pairs from a Wolfram|Alpha response."""
        pods: list[dict[str, str]] = []
        for pod in root.findall("pod"):
            subpod = pod.find("subpod")
            if subpod is None:
                continue
            text = subpod.findtext("plaintext")
            if text is not None and text.strip():
                pods.append({"title": pod.get("title") or "", "text": text.strip()})
        return pods

    @staticmethod
    def _extract_answer(pods: list[dict[str, str]]) -> str | None:
        """Pick the answer text: the 'Result' pod, else the first non-empty pod."""
        for pod in pods:
            if pod["title"] == "Result":
                text = pod["text"].strip()
                return text or None

        for pod in pods:
            text = pod["text"].strip()
            if text:
                return text
        return None
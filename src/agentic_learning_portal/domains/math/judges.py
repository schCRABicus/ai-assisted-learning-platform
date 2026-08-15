"""Building the default judge ensemble from environment configuration."""

from __future__ import annotations

import os

from agentic_learning_portal.api.model import Judge
from agentic_learning_portal.domains.math.qwen_judge import QwenMathJudge
from agentic_learning_portal.domains.math.wa_judge import WolframAlphaJudge


def build_judge_ensemble() -> list[Judge]:
    """Build the judge ensemble enabled by the current environment.

    Mirrors the demo CLI: Wolfram|Alpha is added when ``WOLFRAM_APP_ID`` is
    set — using its default Groq-hosted Llama translator when ``GROQ_API_KEY``
    is present, else falling back to the Gemini translator — and Qwen is added
    when ``GROQ_API_KEY`` is set. Judges are skipped individually when their
    key is missing; without any judge keys the ensemble is empty and generation
    falls back to plain (unverified) generation.
    """
    judges: list[Judge] = []
    if os.environ.get("WOLFRAM_APP_ID"):
        judges.append(
            WolframAlphaJudge()
            if os.environ.get("GROQ_API_KEY")
            else WolframAlphaJudge(model="google:gemini-3.5-flash")
        )
    if os.environ.get("GROQ_API_KEY"):
        judges.append(QwenMathJudge())
    return judges
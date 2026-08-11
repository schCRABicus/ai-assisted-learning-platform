"""Answer-comparison helpers shared by the math domain's judges."""

from __future__ import annotations

import math
from fractions import Fraction


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


def _answers_match(generated: str | int | float, judge: str | None) -> bool:
    """Compare a generated answer against a judge's answer."""
    if judge is None:
        return False

    generated_number = _to_number(generated)
    judge_number = _to_number(judge)
    if generated_number is not None and judge_number is not None:
        return math.isclose(generated_number, judge_number, rel_tol=1e-6, abs_tol=1e-9)

    return _normalize(str(generated)) == _normalize(judge)

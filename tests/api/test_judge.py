from __future__ import annotations

from abc import ABC

import pytest

from agentic_learning_portal.api.model import Judge
from agentic_learning_portal.domains.math import QwenMathJudge, WolframAlphaJudge


def test_judge_is_abstract_base_class() -> None:
    assert issubclass(Judge, ABC)


def test_judge_cannot_be_instantiated_without_verify() -> None:
    class IncompleteJudge(Judge):
        """Deliberately omits ``verify``."""

    with pytest.raises(TypeError, match="abstract"):
        IncompleteJudge()


def test_math_judges_implement_the_contract() -> None:
    assert issubclass(WolframAlphaJudge, Judge)
    assert issubclass(QwenMathJudge, Judge)
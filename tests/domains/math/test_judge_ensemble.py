from __future__ import annotations

from agentic_learning_portal.domains.math import build_judge_ensemble
from agentic_learning_portal.domains.math.qwen_judge import QwenMathJudge
from agentic_learning_portal.domains.math.wa_judge import WolframAlphaJudge


def test_build_judge_ensemble_empty_without_keys(monkeypatch) -> None:
    monkeypatch.delenv("WOLFRAM_APP_ID", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    assert build_judge_ensemble() == []


def test_build_judge_ensemble_includes_wolfram_with_app_id_only(monkeypatch) -> None:
    monkeypatch.setenv("WOLFRAM_APP_ID", "app-id")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    judges = build_judge_ensemble()

    assert len(judges) == 1
    assert isinstance(judges[0], WolframAlphaJudge)
    # No Groq key -> the Wolfram translator falls back to the Gemini model.
    assert judges[0]._model == "google:gemini-3.5-flash"


def test_build_judge_ensemble_includes_qwen_with_groq_key_only(monkeypatch) -> None:
    monkeypatch.delenv("WOLFRAM_APP_ID", raising=False)
    monkeypatch.setenv("GROQ_API_KEY", "key")

    judges = build_judge_ensemble()

    assert len(judges) == 1
    assert isinstance(judges[0], QwenMathJudge)


def test_build_judge_ensemble_includes_both_with_all_keys(monkeypatch) -> None:
    monkeypatch.setenv("WOLFRAM_APP_ID", "app-id")
    monkeypatch.setenv("GROQ_API_KEY", "key")

    judges = build_judge_ensemble()

    assert len(judges) == 2
    assert isinstance(judges[0], WolframAlphaJudge)
    # With Groq available, the Wolfram judge uses its default Groq translator.
    assert judges[0]._model == "groq:openai/gpt-oss-120b"
    assert isinstance(judges[1], QwenMathJudge)
# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

Requires Python 3.14+ and [uv](https://docs.astral.sh/uv/).

- Install dependencies: `uv sync`
- Run all tests: `uv run pytest`
- Run a single test: `uv run pytest tests/domains/math/test_math_generator.py::test_build_user_prompt_includes_subtopics`
- Run the demo CLI: `uv run run-app` — needs a `GOOGLE_API_KEY` in `.env` (`echo 'GOOGLE_API_KEY=your-key-here' > .env`). The demo also verifies with Wolfram|Alpha, which needs a `WOLFRAM_APP_ID` in `.env`; without it the demo falls back to plain generation.

`pytest` runs in `asyncio_mode = "auto"` (configured in `pyproject.toml`), so async tests need no `@pytest.mark.asyncio`.

## Architecture

The package is split into a generic **api layer** and domain-specific **domains** that plug into it. New problem domains follow the same pattern as `math`.

### api layer (domain-agnostic)

- `api/model.py` — `ProblemGenerationPromptInput`, the abstract base for prompt inputs; `build_user_prompt()` is the one required extension point. Also defines `GeneratedTask` (the base output schema) and `VerificationResult` (a judge's verdict on a task).
- `api/generator.py` — `Generator`: turns a `ProblemGenerationPromptInput` into a user prompt, calls a `pydantic_ai` `Agent`, and validates the response against an output class. `DEFAULT_SYSTEM_PROMPT` is the fallback system prompt.

### domain layer (e.g. `domains/math/`)

- `domains/math/model.py` — `MathProblemGenerationPromptInput(ProblemGenerationPromptInput)` adds domain fields (`topic`, `subtopics`, `context`, `complexity`, `grade`) and implements `build_user_prompt()`. The typed LLM output is the api layer's generic `GeneratedTask`.
- `domains/math/generator.py` — `MathProblemGenerator(Generator)` pins the output type to `GeneratedTask`. `generate(prompt_input, *, judge=None, retries=5)` runs the full flow in one loop: generate → validate → (optionally) judge → retry. It raises `RuntimeError` if no validated task is produced after `retries` attempts.
- `domains/math/judge.py` — `WolframAlphaJudge`: ground-truth judge for generated tasks. It first translates `GeneratedTask.text` into a bare arithmetic expression with a small LLM call (`_translate_to_query`, best-effort; returns `""` on failure so the task is treated as unverifiable), then sends that expression to Wolfram|Alpha (`WOLFRAM_APP_ID` from `.env`), extracts the answer (the `Result` pod, else the first non-empty pod), and compares it with `correct_answer` numerically (tolerance via `math.isclose`) or by normalized string. Uses `httpx` directly — the `wolframalpha` package hardcodes a 5s timeout that natural-language parses routinely exceed. `_translate_to_query` and `_query_wolfram` are the mockable seams for tests.
- `domains/math/__init__.py` — re-exports the domain's public API.

### The generation flow

`MathProblemGenerator.generate(prompt_input)` → `prompt_input.build_user_prompt()` → LLM call → response validated as `GeneratedTask`. Each attempt calls `Generator._call_llm` and validates via `_validate_response_matches_output_type`; a validation failure is wrapped into a corrective retry prompt (`Generator._create_retry_prompt`) and re-sent. When a `judge` is passed, the validated task is also checked with `WolframAlphaJudge.verify()`; a mismatch is folded into `_create_verification_retry_prompt`. Both failure paths share one loop bounded by `retries` (default 5), after which it raises `RuntimeError("Max retries reached...")`. Tests mock `Generator._call_llm` (and `_query_wolfram` for judge tests) with an `AsyncMock` to avoid real model calls.

Default model is `google:gemini-3.5-flash`, overridable via `Generator(model=...)`.

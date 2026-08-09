# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

Requires Python 3.14+ and [uv](https://docs.astral.sh/uv/).

- Install dependencies: `uv sync`
- Run all tests: `uv run pytest`
- Run a single test: `uv run pytest tests/domains/math/test_math_generator.py::test_build_user_prompt_includes_subtopics`
- Run the demo CLI: `uv run run-app` — needs a `GOOGLE_API_KEY` in `.env` (`echo 'GOOGLE_API_KEY=your-key-here' > .env`)

`pytest` runs in `asyncio_mode = "auto"` (configured in `pyproject.toml`), so async tests need no `@pytest.mark.asyncio`.

## Architecture

The package is split into a generic **api layer** and domain-specific **domains** that plug into it. New problem domains follow the same pattern as `math`.

### api layer (domain-agnostic)

- `api/model.py` — `ProblemGenerationPromptInput`, the abstract base for prompt inputs; `build_user_prompt()` is the one required extension point. Also defines `GeneratedTask`, the base output schema.
- `api/generator.py` — `Generator`: turns a `ProblemGenerationPromptInput` into a user prompt, calls a `pydantic_ai` `Agent`, and validates the response against an output class. `DEFAULT_SYSTEM_PROMPT` is the fallback system prompt.

### domain layer (e.g. `domains/math/`)

- `domains/math/model.py` — `MathProblemGenerationPromptInput(ProblemGenerationPromptInput)` adds domain fields (`topic`, `subtopics`, `context`, `complexity`, `grade`) and implements `build_user_prompt()`. The typed LLM output is the api layer's generic `GeneratedTask`.
- `domains/math/generator.py` — `MathProblemGenerator(Generator)` pins the output type to `GeneratedTask`.
- `domains/math/__init__.py` — re-exports the domain's public API.

### The generation flow

`MathProblemGenerator.generate(prompt_input)` → `prompt_input.build_user_prompt()` → LLM call → response validated as `MathProblem`. The LLM call happens in `Generator.call_llm_with_output_feedback_loop`: if the response fails output validation, the raw response and error are wrapped into a corrective retry prompt and re-sent, up to `n_retry` (default 5); if it never validates, it raises `RuntimeError("Max retries reached...")`. Tests mock `Generator._call_llm` with an `AsyncMock` to avoid real model calls.

Default model is `google:gemini-3.5-flash`, overridable via `Generator(model=...)`.

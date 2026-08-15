# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

Requires Python 3.14+ and [uv](https://docs.astral.sh/uv/).

- Install dependencies: `uv sync`
- Run all tests: `uv run pytest`
- Run a single test: `uv run pytest tests/domains/math/test_math_generator.py::test_build_user_prompt_includes_subtopics`
- Run the demo CLI: `uv run run-app` — needs a `GOOGLE_API_KEY` in `.env` (`echo 'GOOGLE_API_KEY=your-key-here' > .env`). The demo also verifies with a judge ensemble: Wolfram|Alpha (needs `WOLFRAM_APP_ID` and `GROQ_API_KEY` — the judge's translation runs on `groq:llama-3.3-70b-versatile`; without a Groq key the judge falls back to the Gemini model) and Qwen (needs `GROQ_API_KEY` — `qwen/qwen3.6-27b`, served by Groq). Judges are skipped individually when their key is missing; without any judge keys the demo falls back to plain generation.

`pytest` runs in `asyncio_mode = "auto"` (configured in `pyproject.toml`), so async tests need no `@pytest.mark.asyncio`.

- Run the portal: `uv run run-portal` (or `uv run streamlit run src/agentic_learning_portal/app.py`) — a single Streamlit app exposing each page as an endpoint via `st.navigation`: the admin page at `/admin` (task generation) and a student placeholder at `/student`. Needs a `GOOGLE_API_KEY` in `.env` for the subtopic-suggestion call and for generation.

## Architecture

The package is split into a generic **api layer** and domain-specific **domains** that plug into it. New problem domains follow the same pattern as `math`.

### api layer (domain-agnostic)

- `api/model.py` — `ProblemGenerationPromptInput`, the abstract base for prompt inputs; `build_user_prompt()` is the one required extension point. Also defines `GeneratedTask` (the base output schema), `VerificationResult` (a judge's verdict on a task), and `Judge` (the interface every verification judge implements — subclasses must provide `verify(task, *, listener=None)`).
- `api/llm.py` — the single home for every LLM call in the app: `ask_ai_for_structured_response` (a `pydantic_ai` `Agent` call with an `output_type`) and `ask_ai_for_text_response` (a raw OpenAI-compatible chat-completions HTTP call). Both retry only transient failures — rate limits, 5xx, transport errors, and `pydantic_ai` model-behavior hiccups — with exponential backoff + jitter (`retry`), and acquire a token from a `RateLimiter` (token bucket; module-level `DEFAULT_LIMITER` of 20 calls/s is the shared backstop, overridable per call). Exceptions: `LLMError`, `LLMRetryableError`, `LLMUnavailableError` (raised after retries on 429/5xx/transport), `LLMResponseError` (unexpected reply shape). Note `ask_ai_for_text_response` does **not** retry other 4xx errors (e.g. 401) — those raise `httpx.HTTPStatusError`.
- `api/generator.py` — `Generator`: turns a `ProblemGenerationPromptInput` into a user prompt, runs the LLM call through `api/llm.ask_ai_for_structured_response` (`_call_llm`), and validates the response against an output class. `DEFAULT_SYSTEM_PROMPT` is the fallback system prompt.
- `api/progress.py` — progress reporting for the pipeline. `ProgressEvent(stage, message, attempt, total)` is one discrete step; `ProgressListener` is the (no-op by default) sink with `async on_progress(event)`; `CollectingProgressListener(events)` appends to a list (used by the admin page and tests); `LoggingProgressListener` logs each event; `report_progress(listener, stage, message, ...)` forwards an event, skipping a `None` listener. The generator and each judge accept an optional `listener=` kwarg and emit events at each stage.

### domain layer (e.g. `domains/math/`)

- `domains/math/model.py` — `MathProblemGenerationPromptInput(ProblemGenerationPromptInput)` adds domain fields (`topic`, `subtopics`, `context`, `complexity`, `grade`) and implements `build_user_prompt()`. The typed LLM output is the api layer's generic `GeneratedTask`.
- `domains/math/generator.py` — `MathProblemGenerator(Generator)` pins the output type to `GeneratedTask`. `generate(prompt_input, *, judge=None, retries=5, listener=None)` delegates task generation (LLM call + schema validation) to the base class and adds judge verification on top. `judge` accepts a single `Judge` or a `Sequence[Judge]`; with multiple, the task is accepted only when *all* verify it (judge-ensemble consensus; judges run concurrently via `asyncio.gather`). It raises `RuntimeError` if no judge-verified task is produced after `retries` attempts. When `listener` is provided, the flow reports each stage — generation attempt, schema validation, judge verification (forwarded to the judges themselves), retries, and the final outcome — through `api/progress.report_progress`.
- `domains/math/wa_judge.py` — `WolframAlphaJudge(Judge)`: ground-truth judge for generated tasks. It translates `GeneratedTask.text` into a bare arithmetic expression with a small LLM call (`_translate_once`, which runs through `api/llm.ask_ai_for_structured_response` and so retries transient errors with backoff), retried only when it yields nothing (`_translate_to_query(retries=3)`, returning `""` if none succeeds so the task is treated as unverifiable). There is no faithfulness pass — the same model evaluating its own translation added no reliability. The expression is sent to Wolfram|Alpha's Short Answers endpoint (`v1/result`), which returns a single plain-text answer (`None` on HTTP 501 "did not understand" → task treated as unverifiable), compared with `correct_answer` numerically (tolerance via `math.isclose`) or by normalized string. The Wolfram query uses `httpx` directly — the `wolframalpha` package hardcodes a 5s timeout that natural-language parses routinely exceed (this is a symbolic-computation API, not an LLM call, so it stays here). The judge's translation uses its own `model` (default `groq:llama-3.3-70b-versatile`), independent of the Gemini generator; override via `WolframAlphaJudge(model=...)`. `_translate_to_query`, `_translate_once`, and `_query_wolfram` are the mockable seams for tests. `verify(task, *, listener=None)` reports translation → computation → verdict as progress events.
- `domains/math/qwen_judge.py` — `QwenMathJudge`: LLM judge using `qwen/qwen3.6-27b` (a math-capable reasoning model on Groq — a different family/provider than both the Gemini generator and the Groq Llama translator). Feeds `GeneratedTask.text` to Groq's OpenAI-compatible chat API (`POST https://api.groq.com/openai/v1/chat/completions`, `GROQ_API_KEY` from `.env`) through `api/llm.ask_ai_for_text_response` — which retries rate limits (429), 5xx, and transport failures with backoff before giving up — extracts a single answer from the completion — stripping `<think>...</think>` reasoning first, then `\boxed{...}`, then the last number — and compares it with `correct_answer` via the shared helpers. Any `LLMError` (unavailable after retries, malformed reply) is treated as unverifiable (`_query_groq` returns `None` → `VerificationResult(verified=False)`). Override the model via `QwenMathJudge(model=...)`. `_query_groq` and `_extract_answer` are the mockable seams for tests. `verify(task, *, listener=None)` reports solve → verdict as progress events.
- `domains/math/judges.py` — `build_judge_ensemble()`: builds the judge ensemble from the environment (`WOLFRAM_APP_ID`, `GROQ_API_KEY`), mirroring the demo CLI — Wolfram|Alpha first (Gemini translator when no Groq key), then Qwen. Used by the demo CLI and the admin page; empty means plain (unverified) generation.
- `domains/math/_comparison.py` — shared answer-comparison helpers (`_normalize`, `_to_number`, `_answers_match`) used by both judges.
- `domains/math/__init__.py` — re-exports the domain's public API.

### Streamlit portal (single app, one endpoint per page)

- `app.py` — the single Streamlit entry point. `st.navigation` maps each page
  to a distinct URL: `pages/admin.py` → `/admin` (default), and a
  `pages/student.py` placeholder → `/student`. Launch with `uv run run-portal`.
- `pages/admin.py` — the admin endpoint. As soon as a free-text topic is
  entered, an LLM call populates a multi-select of subtopic suggestions
  (session state records the topic they were generated for so it fires once
  per topic change); grade/complexity dropdowns plus a context field drive
  generation via `MathProblemGenerator`, and a single prominent "Generate
  task" button presents the resulting task (problem, correct answer,
  solution). Generation runs in a background daemon thread: the worker emits
  `ProgressEvent` objects through a `CollectingProgressListener` into a plain
  `gen_state` dict in session state. The page renders the progress in a *single*
  script run — it drains the tail of the log every `POLL_INTERVAL_S` and appends
  each new step (`st.markdown`) to an `st.status` panel once, never re-rendering
  previous lines; Streamlit streams those deltas live, so the panel ticks
  generation attempt → schema validation → each judge's verification → retries
  as they land with no per-poll rerun (which would make the page jump). When the
  thread finishes the page reruns once to re-enable the button and show the
  result, and a collapsed "Generation steps" expander keeps the history.
  Verification uses `build_judge_ensemble()` (so the admin page mirrors the demo
  CLI: judges only when their env keys are set, plain generation otherwise). The
  button disables itself while generating, so a double-click can't start a
  second generation. A comma-separated field lets the admin add subtopics
  manually. The problem text and solution pass through
  `admin/formatting.latex_to_plain_text` before rendering, because the model
  occasionally emits LaTeX (`$$...$$`, `\text{}`, `\frac{}{}`, a stray `\558`)
  despite prompt instructions to write math in plain text.
- `admin/subtopic_suggester.py` — `suggest_subtopics(topic)`: a best-effort
  structured LLM call through `api/llm.ask_ai_for_structured_response` (default
  `google:gemini-3.5-flash`) that returns trimmed, de-duplicated subtopic
  suggestions for a topic, or `[]` on any failure so the UI degrades
  gracefully. Tested in `tests/admin/`.

### The generation flow

`MathProblemGenerator.generate(prompt_input, *, judge=None, retries=5, listener=None)` → `prompt_input.build_user_prompt()` → the base class's `call_llm_with_output_feedback_loop` runs the LLM call (via `api/llm.ask_ai_for_structured_response`) and schema-validation retry, returning a `GeneratedTask`. When a `judge` is passed (single or a list), the validated task is checked with each judge's `verify()` (`WolframAlphaJudge`: translate → Wolfram compare; `QwenMathJudge`: Qwen compare); the task is accepted only when every judge verifies it, and a rejection is folded into `_create_verification_retry_prompt` (which never leaks a judge's answer) and re-sent. Judge rejections are retried up to `retries` (default 5) times, after which it raises `RuntimeError("Max retries reached...")`. When `listener` is passed, the generator reports generate → validate → verify → (retry →) done/error events through it and forwards it to each judge. Tests mock `Generator._call_llm`, the `api/llm` entry points (`ask_ai_for_structured_response`, `ask_ai_for_text_response`), and judge seams like `_translate_to_query` / `_query_wolfram` / `_query_groq` with an `AsyncMock` to avoid real model calls.

Default model is `google:gemini-3.5-flash`, overridable via `Generator(model=...)`. The Wolfram judge's translation defaults to `groq:llama-3.3-70b-versatile` (independent of the generator), overridable via `WolframAlphaJudge(model=...)`. The Qwen judge defaults to `qwen/qwen3.6-27b`, overridable via `QwenMathJudge(model=...)`.

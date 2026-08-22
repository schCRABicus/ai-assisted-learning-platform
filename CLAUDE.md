# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Claude Code Automation Rules

### System Instruction
You are the Master Architect. Your sole responsibility is reasoning, architectural planning, and workflow orchestration. You do not write final application code blocks.

OPERATIONAL PROTOCOL:
1. PHASE 1 (REASONING): When a task is assigned, use your thinking budget to analyze the request. Map out the code architecture, state changes, file dependencies, and potential edge cases.
2. PHASE 2 (STRATEGY): Break the massive task down into highly isolated, atomic, file-specific or function-specific tasks.
3. PHASE 3 (DELEGATION): For every single atomic task identified, spawn a dedicated subagent. 

SUBAGENT SPAWNING RULES:
- You must call your subagent tool (e.g., `ask_follow_up_agent` or your environment's equivalent agent-spawning command).
- For each subagent request, clearly specify the target model (e.g., `google/gemma-4-31b-it:free` or `cohere/north-mini-code:free`).
- Provide the subagent with:
  a) The exact file path to modify or create.
  b) The structural blueprint you designed in Phase 1.
  c) The specific constraints, inputs, and expected outputs for that file alone.
- You must wait for the subagent to report back with its work before spawning the next subagent in the sequence.

CRITICAL CONSTRAINT: 
Do not output code blocks inside your main chat window. If a subagent fails, do not fix the code yourself; instead, re-analyze the error and send corrective instructions to a new subagent instance.

### Read permissions
Read only files specifically mentioned in prompt via `@`. To read any other file out of scope, always ask for permission first.

Spawn subagents and pick the cheapest model that can handle the job:

- Haiku: bulk mechanical tasks, no judgment needed
- Sonnet: scoped research, code exploration, synthesis
- Opus: only when real planning or tradeoffs are involved

### CRITICAL WORKFLOW RULES:
1. When a task requires editing, refactoring, or generating a specific single file, do NOT write the code yourself.
2. Instead, use your `ask_follow_up_agent` or tool-spawning capability to invoke a subagent.
3. Explicitly override the subagent's target model.
4. Provide the subagent with exactly ONE file context, the system requirements, and the isolated task.
5. Once the subagent finishes modifying the file, review their output, run compilation/test tools, and proceed with the next file or file-chunk.

## Commands

Requires Python 3.14+ and [uv](https://docs.astral.sh/uv/).

- Install dependencies: `uv sync`
- Run all tests: `uv run pytest`
- Run a single test: `uv run pytest tests/domains/math/test_math_generator.py::test_build_user_prompt_includes_subtopics`
- Run the demo CLI: `uv run run-app` — needs a `GOOGLE_API_KEY` in `.env` (`echo 'GOOGLE_API_KEY=your-key-here' > .env`). The demo also verifies with a judge ensemble: Wolfram|Alpha (needs `WOLFRAM_APP_ID` and `GROQ_API_KEY` — the judge's translation runs on `groq:openai/gpt-oss-120b`; without a Groq key the judge falls back to the Gemini model) and Qwen (needs `GROQ_API_KEY` — `qwen/qwen3.6-27b`, served by Groq). Judges are skipped individually when their key is missing; without any judge keys the demo falls back to plain generation.
- Storage bootstraps from `.env` when the portal (or a script) calls `load_dotenv()`: if `ADMIN_USERNAME` and `ADMIN_PASSWORD` are both set, the storage backend seeds an admin (roles `admin` + `teacher`) on construction — idempotent, and the password is stored as a salted scrypt hash, never plaintext. `verify_credentials(username, password)` is the login check. Two backends implement the same `Storage` contract: `SqliteStorage` (file-backed at `PORTAL_DB_PATH`, default `portal.db` in the working dir, git-ignored) and `InMemoryStorage` (dict-backed, instant, nothing persisted). `auth.get_storage()` picks one per process via `STORAGE_BACKEND` (`"sqlite"` default | `"memory"`) — see *Authentication & authorization* below.

`pytest` runs in `asyncio_mode = "auto"` (configured in `pyproject.toml`), so async tests need no `@pytest.mark.asyncio`.

- Run the portal: `uv run run-portal` (or `uv run streamlit run src/agentic_learning_portal/app.py`) — a single Streamlit app exposing each page as an endpoint via `st.navigation`: the admin page at `/admin` (task generation) and a student placeholder at `/student`. Needs a `GOOGLE_API_KEY` in `.env` for the subtopic-suggestion call and for generation. Both endpoints are auth-gated (see *Authentication & authorization*); set `ADMIN_USERNAME`/`ADMIN_PASSWORD` in `.env` for the initial admin account.

## Architecture

The package is split into a generic **api layer** and domain-specific **domains** that plug into it. New problem domains follow the same pattern as `math`.

### api layer (domain-agnostic)

- `api/model.py` — `ProblemGenerationPromptInput`, the abstract base for prompt inputs; `build_user_prompt()` is the one required extension point. Also defines `GeneratedTask` (the base output schema), `VerificationResult` (a judge's verdict on a task), and `Judge` (the interface every verification judge implements — subclasses must provide `verify(task, *, listener=None)`).
- `api/llm.py` — the single home for every LLM call in the app: `ask_ai_for_structured_response` (a `pydantic_ai` `Agent` call with an `output_type`) and `ask_ai_for_text_response` (a raw OpenAI-compatible chat-completions HTTP call). Both retry only transient failures — rate limits, 5xx, transport errors, and `pydantic_ai` model-behavior hiccups — with exponential backoff + jitter (`retry`), and acquire a token from a `RateLimiter` (token bucket; module-level `DEFAULT_LIMITER` of 20 calls/s is the shared backstop, overridable per call). Exceptions: `LLMError`, `LLMRetryableError`, `LLMUnavailableError` (raised after retries on 429/5xx/transport), `LLMResponseError` (unexpected reply shape). Note `ask_ai_for_text_response` does **not** retry other 4xx errors (e.g. 401) — those raise `httpx.HTTPStatusError`. `MODELS` is the single dict mapping each LLM purpose to its model (`task_generation`, `subtopic_suggestion`, `wolfram_translation`, `qwen_solver`); every component takes its default from here by key and can still override per instance via `model=`.
- `api/generator.py` — `Generator`: turns a `ProblemGenerationPromptInput` into a user prompt, runs the LLM call through `api/llm.ask_ai_for_structured_response` (`_call_llm`), and validates the response against an output class. `DEFAULT_SYSTEM_PROMPT` is the fallback system prompt.
- `api/progress.py` — progress reporting for the pipeline. `ProgressEvent(stage, message, attempt, total)` is one discrete step; `ProgressListener` is the (no-op by default) sink with `async on_progress(event)`; `CollectingProgressListener(events)` appends to a list (used by the admin page and tests); `LoggingProgressListener` logs each event; `report_progress(listener, stage, message, ...)` forwards an event, skipping a `None` listener. The generator and each judge accept an optional `listener=` kwarg and emit events at each stage.

### domain layer (e.g. `domains/math/`)

- `domains/math/model.py` — `MathProblemGenerationPromptInput(ProblemGenerationPromptInput)` adds domain fields (`topic`, `subtopics`, `context`, `complexity`, `grade`) and implements `build_user_prompt()`. The typed LLM output is the api layer's generic `GeneratedTask`.
- `domains/math/generator.py` — `MathProblemGenerator(Generator)` pins the output type to `GeneratedTask`. `generate(prompt_input, *, judge=None, retries=5, listener=None)` delegates task generation (LLM call + schema validation) to the base class and adds judge verification on top. `judge` accepts a single `Judge` or a `Sequence[Judge]`; with multiple, the task is accepted only when *all* verify it (judge-ensemble consensus; judges run concurrently via `asyncio.gather`). It raises `RuntimeError` if no judge-verified task is produced after `retries` attempts. When `listener` is provided, the flow reports each stage — generation attempt, schema validation, judge verification (forwarded to the judges themselves), retries, and the final outcome — through `api/progress.report_progress`.
- `domains/math/wa_judge.py` — `WolframAlphaJudge(Judge)`: ground-truth judge for generated tasks. It translates `GeneratedTask.text` into a bare arithmetic expression with a small LLM call (`_translate_once`, which runs through `api/llm.ask_ai_for_structured_response` and so retries transient errors with backoff), retried only when it yields nothing (`_translate_to_query(retries=3)`, returning `""` if none succeeds so the task is treated as unverifiable). There is no faithfulness pass — the same model evaluating its own translation added no reliability. The expression is sent to Wolfram|Alpha's Short Answers endpoint (`v1/result`), which returns a single plain-text answer (`None` on HTTP 501 "did not understand" → task treated as unverifiable), compared with `correct_answer` numerically (tolerance via `math.isclose`) or by normalized string. The Wolfram query uses `httpx` directly — the `wolframalpha` package hardcodes a 5s timeout that natural-language parses routinely exceed (this is a symbolic-computation API, not an LLM call, so it stays here). The judge's translation uses its own `model` (default `groq:openai/gpt-oss-120b`, served by Groq), independent of the Gemini generator; override via `WolframAlphaJudge(model=...)`. `_translate_to_query`, `_translate_once`, and `_query_wolfram` are the mockable seams for tests. `verify(task, *, listener=None)` reports translation → computation → verdict as progress events.
- `domains/math/qwen_judge.py` — `QwenMathJudge`: LLM judge using `qwen/qwen3.6-27b` (a math-capable reasoning model on Groq — a different family/provider than both the Gemini generator and the Groq Llama translator). Feeds `GeneratedTask.text` to Groq's OpenAI-compatible chat API (`POST https://api.groq.com/openai/v1/chat/completions`, `GROQ_API_KEY` from `.env`) through `api/llm.ask_ai_for_text_response` — which retries rate limits (429), 5xx, and transport failures with backoff before giving up — extracts a single answer from the completion — stripping `<think>...</think>` reasoning first, then `\boxed{...}`, then the last number — and compares it with `correct_answer` via the shared helpers. Any `LLMError` (unavailable after retries, malformed reply) is treated as unverifiable (`_query_groq` returns `None` → `VerificationResult(verified=False)`). Override the model via `QwenMathJudge(model=...)`. `_query_groq` and `_extract_answer` are the mockable seams for tests. `verify(task, *, listener=None)` reports solve → verdict as progress events.
- `domains/math/judges.py` — `build_judge_ensemble()`: builds the judge ensemble from the environment (`WOLFRAM_APP_ID`, `GROQ_API_KEY`), mirroring the demo CLI — Wolfram|Alpha first (Gemini translator when no Groq key), then Qwen. Used by the demo CLI and the admin page; empty means plain (unverified) generation.
- `domains/math/_comparison.py` — shared answer-comparison helpers (`_normalize`, `_to_number`, `_answers_match`) used by both judges.
- `domains/math/__init__.py` — re-exports the domain's public API.

### Streamlit portal (single app, one endpoint per page)

- `app.py` — the single Streamlit entry point. `st.navigation` maps each page
  to a distinct URL: `pages/admin.py` → `/admin` (default), and a
  `pages/student.py` placeholder → `/student`. Launch with `uv run run-portal`.
- `pages/admin.py` — the admin endpoint, which is assignment-centric. The
  landing view is just an assignment-creation form: a title input + "➕ Add
  assignment" button. On click the assignment is persisted immediately via
  `storage.create_assignment(title, created_by=user.id)` (so it has a name, an
  id, and storage presence before any task exists) and the page swaps to a
  *native* Streamlit carousel (no third-party component). The carousel renders
  the current task slot as a bordered card (one per assignment task) flanked by
  ◀/▶ arrow buttons that are part of the carousel itself — there is no separate
  navigation row — with a "➕ Add task" button to its right that appends a slot
  and navigates to it, and a "Task X of Y" counter + ○●○ dots below. A slide's
  content depends on its slot's state (all in `carousel_*` session-state keys):
  an empty slot shows just a "✨ Generate task" button; clicking it reveals the
  generation form *embedded in the slide* (the form is never shown otherwise);
  while generating, a live `st.status` progress panel replaces the form; once
  generated, the slide shows the finished task card. "❌ Cancel assignment"
  clears the session state (the stored assignment and any generated tasks
  survive) and "🏁 Finish" clears it after showing a summary. The embedded form
  reuses the same generation controls with unique `carousel_*` keys per slot: as
  soon as a free-text topic is entered, an LLM call populates a multi-select of
  subtopic suggestions (fires once per topic change, degrades to a warning +
  manual comma-separated entry when the suggestion fails); grade/complexity
  dropdowns plus a context field drive generation via `MathProblemGenerator`.
  Generation runs in a background daemon thread: the worker emits
  `ProgressEvent` objects through a `CollectingProgressListener` into the slot's
  `carousel_gen_state_{i}` dict. The page shows live progress by *re-running
  every `POLL_INTERVAL_S`*: each run re-renders the slot's `st.status` panel
  with the log accumulated so far, then sleeps and reruns until the thread
  sets `result`/`error`, so the panel ticks generation attempt → schema
  validation → each judge's verification → retries as they land. When the thread
  finishes, the page persists the task via `storage.create_task(task)` +
  `storage.add_task_to_assignment(assignment_id, task_id)` and shows the card.
  Each rerun also calls `_sync_assignment_tasks_from_storage()`, which backfills
  any slot whose in-session task was lost from the assignment's persisted tasks
  (storage is the source of truth), so a generated task can't disappear after
  navigation. Verification uses `build_judge_ensemble()` (so the admin page
  mirrors the demo CLI: judges only when their env keys are set, plain
  generation otherwise). A comma-separated field lets the admin add subtopics
  manually. The problem text and solution pass through
  `admin/formatting.latex_to_plain_text` before rendering, because the model
  occasionally emits LaTeX (`$$...$$`, `\text{}`, `\frac{}{}`, a stray `\558`)
  despite prompt instructions to write math in
  plain text.
- `admin/subtopic_suggester.py` — `suggest_subtopics(topic)`: a best-effort
  structured LLM call through `api/llm.ask_ai_for_structured_response` (default
  `google:gemini-3.5-flash`) that returns trimmed, de-duplicated subtopic
  suggestions for a topic, or `[]` on any failure so the UI degrades
  gracefully. Tested in `tests/admin/`.

### Authentication & authorization (local login/password)

Both endpoints are gated by `auth.py` via `require_roles(...)`, called at the
top of each page before any widget: `/admin` requires the `admin` or `teacher`
role, `/student` requires any authenticated user. An unauthenticated visitor gets
an inline login form (`auth.render_login_form`) and `st.stop()` — so no LLM call
fires before sign-in; a signed-in user lacking a required role gets an
access-denied message. `app.py` renders a "signed in as" badge + Log out in the
sidebar (`auth.render_sidebar_user`).

Identity lives in `st.session_state["user"]` and is checked against the
configured storage backend via `verify_credentials` (salted scrypt hashes,
never plaintext). The `.env`-seeded admin (roles `admin` + `teacher`) is the
first account. `auth.get_storage()` selects the backend by `STORAGE_BACKEND`
(`"sqlite"` default | `"memory"`): sqlite keeps one `SqliteStorage` per thread
(the class forbids cross-thread use) all pointing at the same `PORTAL_DB_PATH`
file; memory hands out one shared `InMemoryStorage` for the whole process (no
per-thread connection concern, and AppTest's page-script thread must see users
created on the test's main thread). `auth.reset_storage()` drops every instance
(test isolation) and re-seeds the admin on the next construction.

Auth is **local username/password by design, not OIDC**: hashing, roles, and the
admin seed already exist, and there is no identity provider to integrate in a
single-process Streamlit sandbox. OIDC would only pay off with an existing IdP
(SSO, managed MFA/password policy, external users); the page guards are role-only,
so swapping in OIDC later means replacing `auth.login` / `verify_credentials`
without touching any page. User provisioning beyond the seeded admin isn't
implemented yet (storage's `create_user`/`list_users` already exist for it). The
guards are exercised in `tests/portal/test_auth.py` (AppTest) and
`tests/admin/test_admin_page.py` logs in through the form.

### The generation flow

`MathProblemGenerator.generate(prompt_input, *, judge=None, retries=5, listener=None)` → `prompt_input.build_user_prompt()` → the base class's `call_llm_with_output_feedback_loop` runs the LLM call (via `api/llm.ask_ai_for_structured_response`) and schema-validation retry, returning a `GeneratedTask`. When a `judge` is passed (single or a list), the validated task is checked with each judge's `verify()` (`WolframAlphaJudge`: translate → Wolfram compare; `QwenMathJudge`: Qwen compare); the task is accepted only when every judge verifies it, and a rejection is folded into `_create_verification_retry_prompt` (which never leaks a judge's answer) and re-sent. Judge rejections are retried up to `retries` (default 5) times, after which it raises `RuntimeError("Max retries reached...")`. When `listener` is passed, the generator reports generate → validate → verify → (retry →) done/error events through it and forwards it to each judge. Tests mock `Generator._call_llm`, the `api/llm` entry points (`ask_ai_for_structured_response`, `ask_ai_for_text_response`), and judge seams like `_translate_to_query` / `_query_wolfram` / `_query_groq` with an `AsyncMock` to avoid real model calls. An autouse `_forbid_real_llm_calls` fixture in `tests/conftest.py` patches `api/llm.Agent` and `httpx.AsyncClient` to raise `AssertionError`, so any test that lets a real LLM/HTTP call escape fails immediately (tests that exercise those paths patch the seams — their `with patch(...)` overrides the guard). Only `tests/storage/test_sqlite_storage.py` exercises `SqliteStorage`; the portal tests run against the in-memory backend (`STORAGE_BACKEND=memory` via the `portal_env` fixture), with `tests/storage/test_memory_storage.py` covering `InMemoryStorage` itself.

Default models all live in `api/llm.MODELS`, keyed by purpose: task generation (`google:gemini-3.5-flash`, overridable via `Generator(model=...)`), subtopic suggestion (`google:gemini-3.5-flash`), the Wolfram judge's translation (`groq:openai/gpt-oss-120b` — served by Groq, independent of the generator, overridable via `WolframAlphaJudge(model=...)`), and the Qwen judge's solver (`qwen/qwen3.6-27b`, overridable via `QwenMathJudge(model=...)`).

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

[//]: # (## Claude Code Automation Rules)

[//]: # ()
[//]: # (### System Instruction)

[//]: # (You are the Master Architect. Your sole responsibility is reasoning, architectural planning, and workflow orchestration. You do not write final application code blocks.)

[//]: # ()
[//]: # (OPERATIONAL PROTOCOL:)

[//]: # (1. PHASE 1 &#40;REASONING&#41;: When a task is assigned, use your thinking budget to analyze the request. Map out the code architecture, state changes, file dependencies, and potential edge cases.)

[//]: # (2. PHASE 2 &#40;STRATEGY&#41;: Break the massive task down into highly isolated, atomic, file-specific or function-specific tasks.)

[//]: # (3. PHASE 3 &#40;DELEGATION&#41;: For every single atomic task identified, spawn a dedicated subagent. If you spawn a subagent to write or run tests, pass the exact file contents or relevant code snippets directly into the subagent's prompt context as well as required references. Do not let the subagent re-read the files from disk. If you find yourself reading the same file more than twice without making edits, halt immediately and ask me for clarification.)

[//]: # ()
[//]: # ()
[//]: # (SUBAGENT SPAWNING RULES:)

[//]: # (- You must call your subagent tool &#40;e.g., `ask_follow_up_agent` or your environment's equivalent agent-spawning command&#41;.)

[//]: # (- For each subagent request, clearly specify the target model &#40;e.g., `google/gemma-4-31b-it:free` or `cohere/north-mini-code:free`&#41;.)

[//]: # (- Provide the subagent with:)

[//]: # (  a&#41; The exact file path to modify or create.)

[//]: # (  b&#41; The structural blueprint you designed in Phase 1.)

[//]: # (  c&#41; The specific constraints, inputs, and expected outputs for that file alone.)

[//]: # (  d&#41; The exact file contents or relevant code snippets directly into the subagent's prompt context. Do not let the subagent re-read the files from disk.)

[//]: # (- If you find yourself reading the same file more than twice without making edits, halt immediately and ask me for clarification.)

[//]: # (- You must wait for the subagent to report back with its work before spawning the next subagent in the sequence.)

[//]: # ()
[//]: # (CRITICAL CONSTRAINT: )

[//]: # (Do not output code blocks inside your main chat window. If a subagent fails, do not fix the code yourself; instead, re-analyze the error and send corrective instructions to a new subagent instance.)

[//]: # (If you find yourself reading the same file more than twice without making edits, halt immediately and ask me for clarification.)

[//]: # ()
[//]: # (### Read permissions)

[//]: # (Read only files specifically mentioned in prompt via `@`. To read any other file out of scope, always ask for permission first.)

[//]: # ()
[//]: # (Spawn subagents and pick the cheapest model that can handle the job:)

[//]: # ()
[//]: # (- Haiku: bulk mechanical tasks, no judgment needed)

[//]: # (- Sonnet: scoped research, code exploration, synthesis)

[//]: # (- Opus: only when real planning or tradeoffs are involved)

[//]: # ()
[//]: # (### CRITICAL WORKFLOW RULES:)

[//]: # (1. When a task requires editing, refactoring, or generating a specific single file, do NOT write the code yourself.)

[//]: # (2. Instead, use your `ask_follow_up_agent` or tool-spawning capability to invoke a subagent.)

[//]: # (3. Explicitly override the subagent's target model.)

[//]: # (4. Provide the subagent with exactly ONE file context, the system requirements, and the isolated task.)

[//]: # (5. Once the subagent finishes modifying the file, review their output, run compilation/test tools, and proceed with the next file or file-chunk.)

## Commands

Requires Python 3.14+ and [uv](https://docs.astral.sh/uv/).

- Install dependencies: `uv sync`
- Run all tests: `uv run pytest`
- Run a single test: `uv run pytest tests/domains/math/test_math_generator.py::test_build_user_prompt_includes_subtopics`
- Run the demo CLI: `uv run run-app` — needs a `GOOGLE_API_KEY` in `.env` (`echo 'GOOGLE_API_KEY=your-key-here' > .env`). The demo also verifies with a judge ensemble: Wolfram|Alpha (needs `WOLFRAM_APP_ID` and `GROQ_API_KEY` — the judge's translation runs on `groq:openai/gpt-oss-120b`; without a Groq key the judge falls back to the Gemini model) and Qwen (needs `GROQ_API_KEY` — `qwen/qwen3.6-27b`, served by Groq). Judges are skipped individually when their key is missing; without any judge keys the demo falls back to plain generation.
- Storage bootstraps from `.env` when the portal (or a script) calls `load_dotenv()`: if `ADMIN_USERNAME` and `ADMIN_PASSWORD` are both set, the storage backend seeds an admin (roles `admin` + `teacher`) on construction — idempotent, and the password is stored as a salted scrypt hash, never plaintext. `verify_credentials(username, password)` is the login check. Two backends implement the same `Storage` contract: `SqliteStorage` (file-backed at `PORTAL_DB_PATH`, default `portal.db` in the working dir, git-ignored) and `InMemoryStorage` (dict-backed, instant, nothing persisted). `auth.get_storage()` picks one per process via `STORAGE_BACKEND` (`"sqlite"` default | `"memory"`) — see *Authentication & authorization* below. The contract also carries the remember-me sessions: `create_session(user_id, ttl_seconds=...)` returns a raw `secrets.token_urlsafe` token and stores only its SHA-256 (table `sessions`, migration 0006), `get_user_by_session_token` resolves a live one and rejects an expired/revoked/unknown token alike, `delete_session` revokes one, and `delete_sessions_for_user` revokes every session a user holds (wired into `set_password`, so a password reset can't leave a session opened with the old password alive).

`pytest` runs in `asyncio_mode = "auto"` (configured in `pyproject.toml`), so async tests need no `@pytest.mark.asyncio`.

- Run the portal: `uv run run-portal` (or `uv run streamlit run src/agentic_learning_portal/app.py`) — a single Streamlit app exposing each page as an endpoint via `st.navigation`: the public sign-in page at `/` (the only `default=True` page — holding that flag is what strips a page's named URL, so it must sit on the page that never needs one), the admin pages at `/admin`, `/assignments` and `/users`, the student page at `/student`, and the public `/verify` linked from the invitation email. Needs a `GOOGLE_API_KEY` in `.env` for the subtopic-suggestion call and for generation. The admin and student pages are auth-gated (see *Authentication & authorization*); set `ADMIN_USERNAME`/`ADMIN_PASSWORD` in `.env` for the initial admin account.

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

- `app.py` — the single Streamlit entry point. `st.navigation` maps each page to
  a distinct URL and builds the sidebar from the signed-in user's roles: an
  admin or teacher sees only the **Admin** section (`views/admin.py` →
  `/admin`, `views/admin/01_assignments.py` → `/assignments`,
  `views/admin/03_users.py` → `/users`), a student only **Student**
  (`views/student.py`). A user holding both sees both sections, Admin first —
  the same precedence `auth.landing_page_path` uses. Signed out, the only entry
  is **Sign in** (`views/signin.py`), which is also the one `default=True` page
  and therefore the app's front door at `/`.
  **Every page is registered on every run** — `visibility` only hides a page
  from the sidebar, and the hidden ones (`views/signin.py`,
  `01_assignment_editor.py`, `views/student_take.py`, `views/verify.py`) ride
  along inside whichever section is showing so no header is left empty. That
  matters because a page dropped from the navigation stops resolving outright:
  its URL would 404 and any `st.switch_page` aimed at it would raise for
  whoever is signed in at the time. The pages gate themselves
  (`auth.require_roles`), so registering one is not a grant. `flush_remember_me()`
  runs here, before `st.navigation`, since a signed-out visitor's sign-in page
  is public and no guard would run for them. Launch with `uv run run-portal`.
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

Pages are gated by `auth.py` via `require_roles(...)`, called at the top of each
page before any widget: the admin pages require the `admin` or `teacher` role
(`auth.ADMIN_ROLES`, shared with the nav so the two can't drift), `/student`
requires any authenticated user. An unauthenticated visitor gets an inline login
form (`auth.render_login_form`) and `st.stop()` — so no LLM call fires before
sign-in; a signed-in user lacking a required role gets an access-denied message.
Two pages are deliberately **public**: `views/signin.py` (at `/`) and
`views/verify.py`, whose visitor has an emailed token but no account access yet.
`app.py` renders a "signed in as" badge + Log out in the sidebar
(`auth.render_sidebar_user`).

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

**"Remember me" is an opaque session token in a cookie**
(`remember_me_logged_in_user`), never a user id — a stored id would be forgeable
by hand-editing the cookie into a full sign-in as that user. `login(...,
remember_me=True)` opens a session (`create_session`) and queues its raw token;
`current_user()` resolves the cookie through `get_user_by_session_token`, and
`logout()` deletes the session server-side before clearing state, so a copy of
the cookie left on disk is dead. The session is resolved from
`st.context.cookies` — the cookies of the request that opened the session — which
is synchronous and available on the very first run. **Writing** the cookie is
deferred to a run that completes without an `st.rerun()` (`auth.flush_remember_me`,
called from `app.py` on every run and idempotently from `require_roles`): doing it
inline races the post-login rerun, which tears the `<script>` element down before
the browser executes it. The write itself is a `<script>` emitted through
`st.html(..., unsafe_allow_javascript=True)` — the JS runs in the main document,
and values go through `json.dumps`. This replaced `streamlit_cookies_controller`,
whose Python layer passed `expires` as an ISO-8601 string while its bundled JS
required a real `Date` and threw before `document.cookie` was ever assigned, so
every write failed silently. `HttpOnly` is impossible from JavaScript, which is
precisely why the value is an opaque, revocable token rather than an identity.
A per-session `_FORCE_LOGGED_OUT` flag keeps logout sticky, since the session's
`st.context.cookies` are immutable and would otherwise re-authenticate the user
on the next run.

Auth is **local username/password by design, not OIDC**: hashing, roles, and the
admin seed already exist, and there is no identity provider to integrate in a
single-process Streamlit sandbox. OIDC would only pay off with an existing IdP
(SSO, managed MFA/password policy, external users); the page guards are role-only,
so swapping in OIDC later means replacing `auth.login` / `verify_credentials`
without touching any page. `tests/portal/test_auth.py` (AppTest) exercises the
guards and the session cookie, `tests/portal/test_nav.py` asserts the per-role
sidebar, and `tests/admin/test_admin_page.py` logs in through the form.

### The generation flow

`MathProblemGenerator.generate(prompt_input, *, judge=None, retries=5, listener=None)` → `prompt_input.build_user_prompt()` → the base class's `call_llm_with_output_feedback_loop` runs the LLM call (via `api/llm.ask_ai_for_structured_response`) and schema-validation retry, returning a `GeneratedTask`. When a `judge` is passed (single or a list), the validated task is checked with each judge's `verify()` (`WolframAlphaJudge`: translate → Wolfram compare; `QwenMathJudge`: Qwen compare); the task is accepted only when every judge verifies it, and a rejection is folded into `_create_verification_retry_prompt` (which never leaks a judge's answer) and re-sent. Judge rejections are retried up to `retries` (default 5) times, after which it raises `RuntimeError("Max retries reached...")`. When `listener` is passed, the generator reports generate → validate → verify → (retry →) done/error events through it and forwards it to each judge. Tests mock `Generator._call_llm`, the `api/llm` entry points (`ask_ai_for_structured_response`, `ask_ai_for_text_response`), and judge seams like `_translate_to_query` / `_query_wolfram` / `_query_groq` with an `AsyncMock` to avoid real model calls. An autouse `_forbid_real_llm_calls` fixture in `tests/conftest.py` patches `api/llm.Agent` and `httpx.AsyncClient` to raise `AssertionError`, so any test that lets a real LLM/HTTP call escape fails immediately (tests that exercise those paths patch the seams — their `with patch(...)` overrides the guard). Only `tests/storage/test_sqlite_storage.py` exercises `SqliteStorage`; the portal tests run against the in-memory backend (`STORAGE_BACKEND=memory` via the `portal_env` fixture), with `tests/storage/test_memory_storage.py` covering `InMemoryStorage` itself.

Default models all live in `api/llm.MODELS`, keyed by purpose: task generation (`google:gemini-3.5-flash`, overridable via `Generator(model=...)`), subtopic suggestion (`google:gemini-3.5-flash`), the Wolfram judge's translation (`groq:openai/gpt-oss-120b` — served by Groq, independent of the generator, overridable via `WolframAlphaJudge(model=...)`), and the Qwen judge's solver (`qwen/qwen3.6-27b`, overridable via `QwenMathJudge(model=...)`).

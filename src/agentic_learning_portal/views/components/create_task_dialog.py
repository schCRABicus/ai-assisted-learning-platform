"""The task-authoring dialog (views/components/create_task_dialog.py).

:func:`create_task_dialog` is a Streamlit ``st.dialog`` opened by a page through
the ``create_task_open`` session-state flag: the page sets the flag to the
assignment id and re-calls the dialog on every run while it's set, so the dialog
survives the ``st.rerun()`` calls it makes (e.g. the live progress poll loop).
Used by the Admin page (``views/admin.py``) and the assignment editor
(``views/admin/02_assignment_editor.py``).

The dialog is a small 3-step wizard over a single session-state dict
(``create_task_state``):

1. **Describe** — the authoring form (topic, subtopics, context, grade,
   complexity). Clicking **Generate** clears any previous progress and starts
   generation in a background thread.
2. **Generate** — the live progress panel. The dialog re-runs every poll
   interval and renders the accumulated ``ProgressEvent`` log (a
   :class:`CollectingProgressListener` appends events to the state's ``log``).
   The form is not reachable while generation is in flight; when the thread
   finishes, the dialog advances to step 3.
3. **Review** — the finished task card with Save, or the error with a retry.
   From here every step is reachable again (edit the form, re-read the log, or
   save).

Steps 2 and 3 are disabled until a generation has produced a result or an error,
so the wizard opens on the form alone. Saving persists the task via storage and
links it to the assignment.
"""

from __future__ import annotations

import asyncio
import threading
import time

import streamlit as st

from agentic_learning_portal.admin.formatting import latex_to_plain_text
from agentic_learning_portal.admin.subtopic_suggester import suggest_subtopics
from agentic_learning_portal.api.llm import MODELS, ModelChain
from agentic_learning_portal.api.progress import CollectingProgressListener, ProgressEvent
from agentic_learning_portal.domains.math import (
    MathProblemGenerationPromptInput,
    MathProblemGenerator,
    build_judge_ensemble,
)
from agentic_learning_portal.storage import Storage
from agentic_learning_portal.storage.factory import StorageFactory
from agentic_learning_portal.api.progress import CallbackProgressListener

STORAGE_FACTORY = StorageFactory()

DEFAULT_TASK_CONTEXT = "Plants vs Zombie"
DEFAULT_SUBTOPIC_SUGGESTION_MODEL = MODELS["subtopic_suggestion"]

# How long the task-creation dialog sleeps between polls of the worker thread
# that runs generation (see ``_render_generation_progress``).
POLL_INTERVAL_S = 0.5

# Icon per progress stage, shown in the live status panel while a task is generated.
_EVENT_ICONS = {
    "generate": "📝",
    "validate": "✅",
    "verify": "🔎",
    "retry": "🔁",
    "done": "🎉",
    "error": "❌",
}


def _event_icon(stage: str) -> str:
    return _EVENT_ICONS.get(stage, "•")


def get_storage() -> Storage:
    """Return the :class:`Storage` the portal should use.

    The backend is chosen by ``STORAGE_BACKEND`` (``"sqlite"`` or ``"memory"``),
    read at call time so tests and ``.env`` can switch it after import. The
    sqlite backend keeps one instance per thread (``seed_admin_from_env`` runs
    on construction, so the ``.env``-configured admin is available from the
    start); the memory backend is a single shared instance.
    """
    return STORAGE_FACTORY.get_storage()


@st.cache_data
def get_subtopic_suggestions(topic: str, _model: ModelChain) -> list[str]:
    # ``_model`` is underscore-prefixed so ``st.cache_data`` skips hashing it
    # (Streamlit's hasher walks object internals and ``ModelChain`` is not
    # hashable by it). The model is a stable module-level constant, so the
    # cache key only needs the topic.
    return ["Addition", "Subtraction"]
    # return asyncio.run(suggest_subtopics(topic, model=_model))


# def _run_generation_in_thread(
def _run_generation(
    prompt_input: MathProblemGenerationPromptInput,
    gen_state: dict,
) -> None:
    """Generate a task in a daemon thread, writing progress + outcome to ``gen_state``.

    ``gen_state`` is a plain dict shared with the main script: the worker
    appends ``ProgressEvent`` objects to ``gen_state["log"]`` through a
    :class:`CallbackProgressListener` and stores the finished ``GeneratedTask``
    in ``gen_state["result"]`` (or an error string in ``gen_state["error"]``).
    The main script polls it on every rerun and renders the accumulated log.
    If the dialog was dismissed mid-flight (``gen_state["cancel"]`` set by the
    ``on_dismiss`` callback), the finished outcome is discarded so a stale
    result/error can't leak into a later run. Nothing here touches
    ``st.session_state``.
    """
    def progress_log_collector(event: ProgressEvent) -> None:
        gen_state["log"].append(event)

    listener = CallbackProgressListener(progress_log_collector)

    async def _run() -> None:
        try:
            task = await MathProblemGenerator().generate(
                prompt_input,
                judge=build_judge_ensemble() or None,
                listener=listener,
            )
            if not gen_state.get("cancel", False):
                gen_state["result"] = task
        except RuntimeError as e:
            if not gen_state.get("cancel", False):
                gen_state["error"] = str(e)
        except Exception as e:  # noqa: BLE001 - surface any failure in the UI
            if not gen_state.get("cancel", False):
                gen_state["error"] = f"{type(e).__name__}: {e}"

        st.rerun()

    asyncio.run(_run())

def _await_generation_completion(gen_state: dict):
    async def _run() -> None:
        if gen_state.get("cancel", False) or gen_state.get("result", False):
            return

        time.sleep(POLL_INTERVAL_S)

    asyncio.run(_run())


def _create_task_state(assignment_id: int) -> dict:
    """Return the task-creation dialog's session state, reset for a new assignment."""
    state = st.session_state.setdefault("create_task_state", {})
    if state.get("assignment_id") != assignment_id:
        state = st.session_state["create_task_state"] = {
            "assignment_id": assignment_id,
            "step": 1,
            "log": [],
            "result": None,
            "error": None,
            "generating": False,
            "task_id": None,
            "prompt_input": None,
            "form": {},
        }
    return state


def _close_create_task_dialog() -> None:
    """Close the task-creation dialog (drops its open flag + state) and rerun."""
    st.session_state.pop("create_task_open", None)
    st.session_state.pop("create_task_state", None)
    st.rerun()


def _cancel_create_task_dialog() -> None:
    """Cancel an in-flight generation and drop the dialog when it is dismissed.

    Registered as the ``on_dismiss`` callback of the dialog, so it runs whenever
    the user dismisses it with the ✕ button, ESC, or an outside click. Dismissal
    only hides the dialog on the frontend — Streamlit does not rerun the app and
    ``create_task_open`` stays set — so without this hook the page would keep
    re-opening the dialog on every progress-poll rerun. Clearing
    ``create_task_open`` stops that re-open; the ``cancel`` flag tells the worker
    thread to discard its (possibly still running) outcome. Unlike
    :func:`_close_create_task_dialog`, no ``st.rerun()`` is needed here: the
    dismissal triggers the app rerun itself.
    """
    state = st.session_state.get("create_task_state")
    if state is not None:
        state["cancel"] = True
    st.session_state.pop("create_task_open", None)
    st.session_state.pop("create_task_state", None)


@st.fragment(run_every=POLL_INTERVAL_S)
def _render_progress_log_loop(state: dict) -> None:
    """Render the progress-log and handle navigation clicks."""
    _render_progress_log(state["log"])

def _render_progress_log(events: list) -> None:
    """Render the collected ``ProgressEvent`` log as icon + message lines."""
    for event in events:
        st.markdown(f"{_event_icon(event.stage)} {event.message}")


# --- wizard navigation ----------------------------------------------------------


def _render_step_navigation(state: dict) -> None:
    """Render the 1-2-3 wizard bar and handle navigation clicks.

    Step 1 (Describe) is always reachable except while generation is in flight —
    once **Generate** is clicked you're committed until the thread finishes.
    Steps 2 (Generate) and 3 (Review) are only reachable once a generation has
    produced a result or an error (or is in flight, for step 2); with no
    generated state they stay disabled. The active step is highlighted as the
    primary button.
    """
    generating = state.get("generating", False)
    has_generated = state.get("result") is not None or state.get("error") is not None
    current = state.get("step", 1)

    col1, col2, col3 = st.columns(3)
    if col1.button(
        "1️⃣ Describe",
        key="step_nav_1",
        type="primary" if current == 1 else "secondary",
        disabled=generating,
    ):
        state["step"] = 1
        st.rerun()
    if col2.button(
        "2️⃣ Generate",
        key="step_nav_2",
        type="primary" if current == 2 else "secondary",
        disabled=not (generating or has_generated),
    ):
        state["step"] = 2
        st.rerun()
    if col3.button(
        "3️⃣ Review",
        key="step_nav_3",
        type="primary" if current == 3 else "secondary",
        disabled=not has_generated,
    ):
        state["step"] = 3
        st.rerun()


# --- wizard steps ----------------------------------------------------------------


def _render_task_settings_step(state: dict) -> None:
    content = st.empty()
    with content.container(key="task_settings_container", border=True):
        """The authoring form phase (wizard step 1)."""
        st.caption("Describe the task below, then Generate.")

        # The form's widget values can be pruned when the wizard leaves the form
        # (Streamlit only retains values for widgets rendered in the most recent
        # run), so they are mirrored into ``state["form"]`` and used to rehydrate
        # the widgets when the user navigates back to edit.
        form = state.setdefault("form", {})
        form.setdefault("topic", "")
        form.setdefault("selected_subtopics", [])
        form.setdefault("manual_subtopics", "")
        form.setdefault("context", DEFAULT_TASK_CONTEXT)
        form.setdefault("grade", 5)
        form.setdefault("complexity", "easy")

        # --- topic + dynamically suggested subtopics -----------------------------------

        topic = st.text_input(
            "📚 Topic",
            value=form["topic"],
            placeholder="e.g. Arithmetic, Algebra, Geometry",
            help="The math topic to generate a task for.",
            key="create_task_topic",
        )

        # ``get_subtopic_suggestions`` is memoized, so re-runs (widget interaction,
        # closing and reopening the form) hit the cache instead of firing another
        # LLM call.
        suggestions: list[str] = []
        if topic.strip():
            with st.spinner("🔍 Suggesting subtopics..."):
                suggestions = get_subtopic_suggestions(
                    topic, DEFAULT_SUBTOPIC_SUGGESTION_MODEL
                )
            if not suggestions:
                st.warning(
                    "The LLM could not suggest subtopics for this topic. Add them "
                    "manually below."
                )

        selected_subtopics = st.multiselect(
            "🧩 Subtopics",
            options=suggestions,
            default=[s for s in form["selected_subtopics"] if s in suggestions],
            key="create_task_subtopics",
            help="LLM-suggested subtopics; select the ones to focus on.",
        )
        manual = st.text_input(
            "✍️ Add your own subtopics (comma-separated)",
            value=form["manual_subtopics"],
            placeholder="e.g. fractions, word problems",
            key="create_task_manual_subtopics",
        )
        manual_subtopics = [s.strip() for s in manual.split(",") if s.strip()]
        subtopics = list(selected_subtopics) + [
            s for s in manual_subtopics if s not in selected_subtopics
        ]

        if not suggestions:
            st.caption("💡 Enter a topic to get LLM-suggested subtopics.")
        elif not selected_subtopics:
            st.caption("💡 Select the subtopics to focus on (or add your own below).")

        # --- other parameters ----------------------------------------------------------

        context = st.text_input(
            "🎭 Context",
            value=form["context"],
            key="create_task_context",
            help="Thematic setting for the problem (e.g. Lego, Plants vs Zombies).",
        )

        col_grade, col_complexity = st.columns(2)
        grade = col_grade.selectbox(
            "🎓 Grade",
            options=list(range(1, 12)),
            index=form["grade"] - 1,
            key="create_task_grade",
            help="Student grade level (1-11).",
        )
        complexity = col_complexity.selectbox(
            "📊 Complexity",
            options=["easy", "medium", "hard"],
            index=["easy", "medium", "hard"].index(form["complexity"]),
            key="create_task_complexity",
        )

        # Mirror the form's current values back into the state so they survive a
        # round trip through steps 2 and 3 (see the rehydration comment above).
        form["topic"] = topic
        form["selected_subtopics"] = list(selected_subtopics)
        form["manual_subtopics"] = manual
        form["context"] = context
        form["grade"] = grade
        form["complexity"] = complexity

        # --- generate / discard ---------------------------------------------------------

        # def trigger_generate():


        generate_button = st.button(
            "🎯 Generate",
            type="primary",
            disabled=not (topic.strip() and subtopics),
            key="create_task_generate",
        )
        if generate_button:
            prompt_input = MathProblemGenerationPromptInput(
                topic=topic.strip(),
                subtopics=subtopics,
                context=context.strip() or DEFAULT_TASK_CONTEXT,
                complexity=complexity,
                grade=grade,
            )
            # Restart the wizard from scratch: drop any previous generation outcome,
            # kick off a fresh background thread, and advance to the live step.
            state["log"] = []
            state["result"] = None
            state["error"] = None
            state["prompt_input"] = prompt_input
            state["generating"] = True
            state["step"] = 2
            threading.Thread(
                target=_run_generation,
                args=(prompt_input, state),
                daemon=True,
            ).start()
            st.rerun()

        st.button("✖ Cancel", key="create_task_cancel", on_click=_close_create_task_dialog)

def _render_generation_progress_step(state: dict) -> None:
    """Live progress panel while a task is generated (wizard step 2).

    While ``generating``, the script re-runs every poll interval: each run
    renders the status panel with the log accumulated so far, then sleeps and
    reruns until the worker thread sets ``result``/``error``, at which point the
    dialog advances to step 3. If the user navigates back here after generation
    finished (e.g. from step 3), the finished log is shown without the poll
    loop.
    """
    content = st.empty()
    with content.container(key="generation_progress_container", border=True):
        if state.get("generating", False):
            status = st.status("✨ Generating task...", expanded=True)
            with st.expander("⚙️ Generation progress", expanded=True):
                _render_progress_log_loop(state)

            _await_generation_completion(state)

            if state.get("result") is not None:
                status.update(label="✅ Task generated and verified", state="complete")
                state["generating"] = False
                state["step"] = 3
            elif state.get("error") is not None:
                status.update(label="❌ Task generation failed", state="error")
                state["generating"] = False
                state["step"] = 3

            st.rerun()

        # Generation already finished; show the log without re-running the worker.
        if not state.get("generating", False):
            if state.get("result") is not None:
                st.status("✅ Task generated and verified", state="complete", expanded=False)
            elif state.get("error") is not None:
                st.status("❌ Task generation failed", state="error", expanded=False)
            with st.expander("⚙️ Generation progress", expanded=True):
                _render_progress_log(state["log"])


def _render_task_generation_result_step(assignment_id: int, state: dict) -> None:
    """The finished-task-card phase (wizard step 3, success): preview + Save / Cancel."""
    content = st.empty()
    with content.container(key="generation_result_container", border=True):
        task = state["result"]
        # The prompt is only available in-session; storage-synced tasks would have
        # none, so degrade to "—".
        prompt_input = state.get("prompt_input")
        grade = prompt_input.grade if prompt_input is not None else "—"
        subtopics = (
            ", ".join(prompt_input.subtopics) if prompt_input is not None else "—"
        )

        st.success("Task generated and verified.")
        with st.expander("⚙️ Generation progress", expanded=False):
            _render_progress_log(state["log"])

        st.markdown(f"**📚 Topic:** {task.topic}")
        meta = st.columns(4)
        meta[0].markdown(f"**🎓 Grade:** {grade}")
        meta[1].markdown(f"**📊 Complexity:** {task.complexity}")
        meta[2].markdown(f"**🧩 Subtopics:** {subtopics}")

        st.markdown("---")
        st.markdown(f"**📝 Problem**\n\n{latex_to_plain_text(task.text)}")
        st.markdown(f"**✅ Correct answer:** `{task.correct_answer}`")
        with st.expander("💡 Solution", expanded=False):
            st.markdown(latex_to_plain_text(task.solution))
        st.markdown("---")

        col_save, col_cancel = st.columns(2)
        if col_save.button("💾 Save", type="primary", key="create_task_save"):
            storage = get_storage()
            persisted = storage.create_task(task)
            storage.add_task_to_assignment(assignment_id, persisted.id)
            state["task_id"] = persisted.id
            st.success(f"Task #{persisted.id} saved to the assignment.")
            _close_create_task_dialog()
        if col_cancel.button("✖ Cancel", key="create_task_cancel"):
            _close_create_task_dialog()


def _render_create_task_error(state: dict) -> None:
    """The failure phase (wizard step 3, failure): show the error + log, offer a retry."""
    with st.container(border=True):
        st.error(f"Task generation failed: {state['error']}")
        with st.expander("⚙️ Generation progress", expanded=True):
            _render_progress_log(state["log"])

        col_retry, col_cancel = st.columns(2)
        if col_retry.button("↻ Try again", key="create_task_retry"):
            state["generating"] = False
            state["result"] = None
            state["error"] = None
            state["step"] = 1
            st.rerun()
        if col_cancel.button("✖ Cancel", key="create_task_cancel"):
            _close_create_task_dialog()


@st.dialog(
    "Task Creation",
    dismissible=True,
    on_dismiss=_cancel_create_task_dialog,
)
def create_task_dialog(assignment_id: int) -> None:
    """Author and save a task for ``assignment_id`` in a modal wizard.

    Walks through the three steps driven by ``create_task_state`` (see the
    module docstring): the authoring form, the live progress panel while
    generation runs in a background thread, then the finished task card with a
    Save button or an error with a retry. Steps 2 and 3 are disabled until a
    generation has produced a result or an error. Saving persists the task via
    storage and links it to the assignment.

    Dismissing the dialog (✕ / ESC / outside click) while generation is in
    flight runs the ``on_dismiss`` callback (:func:`_cancel_create_task_dialog`),
    which cancels the background worker and drops the dialog's flags so the
    progress-poll loop can't re-open it.
    """
    state = _create_task_state(assignment_id)

    # With no generation in flight or finished, the wizard can only sit on the
    # authoring form (steps 2 and 3 are locked).
    step = state.get("step", 1)
    if step > 1 and not state.get("generating", False):
        if state.get("result") is None and state.get("error") is None:
            step = state["step"] = 1

    _render_step_navigation(state)
    st.divider()

    if step == 1:
        _render_task_settings_step(state)
    elif step == 2:
        _render_generation_progress_step(state)
    elif state.get("result") is not None:
        _render_task_generation_result_step(assignment_id, state)
    else:
        _render_create_task_error(state)
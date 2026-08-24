"""Shared dialogs for the portal pages (views/components/modals.py).

Both dialogs are Streamlit ``st.dialog`` functions opened by a page through a
session-state flag: the delete confirmation dialog and the task-authoring
:func:`create_task_dialog`. The opener sets the flag to the relevant id and
re-calls the dialog function on every run while the flag is set, so the dialog
survives the ``st.rerun()`` calls it makes (e.g. the live progress poll loop).

:func:`create_task_dialog` is a small phase machine over a single session-state
dict (``create_task_state``): the authoring form, a live progress panel while
generation runs in a background thread (a :class:`CollectingProgressListener`
appends events to the state's ``log``), then either the finished task card with
a Save button or an error with a retry. Saving persists the task via storage and
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
from agentic_learning_portal.api.progress import CollectingProgressListener
from agentic_learning_portal.domains.math import (
    MathProblemGenerationPromptInput,
    MathProblemGenerator,
    build_judge_ensemble,
)
from agentic_learning_portal.storage import Storage
from agentic_learning_portal.storage.factory import StorageFactory

STORAGE_FACTORY = StorageFactory()

DEFAULT_TASK_CONTEXT = "Plants vs Zombie"
DEFAULT_SUBTOPIC_SUGGESTION_MODEL = MODELS["subtopic_suggestion"]

# How long the task-creation dialog sleeps between polls of the worker thread
# that runs generation (see ``_render_generation_progress``).
POLL_INTERVAL_S = 0.25

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


# Define the confirmation pop-up dialog
@st.dialog("Confirm Assignment Deletion", dismissible=True)
def delete_assignment_dialog(assignment_id: int) -> None:
    st.write("Are you sure you want to delete this assignment? This cannot be undone.")

    col1, col2 = st.columns(2)

    if col1.button("Yes, Delete", type="primary"):
      get_storage().delete_assignment(assignment_id)
      st.success("Item deleted successfully!")
      # st.session_state.show_dialog = False
      st.rerun()

    if col2.button("Cancel"):
      # st.session_state.show_dialog = False
      st.rerun()


@st.cache_data
def get_subtopic_suggestions(topic: str, _model: ModelChain) -> list[str]:
    # ``_model`` is underscore-prefixed so ``st.cache_data`` skips hashing it
    # (Streamlit's hasher walks object internals and ``ModelChain`` is not
    # hashable by it). The model is a stable module-level constant, so the
    # cache key only needs the topic.
    return asyncio.run(suggest_subtopics(topic, model=_model))


def _run_generation_in_thread(
    prompt_input: MathProblemGenerationPromptInput,
    gen_state: dict,
) -> None:
    """Generate a task in a daemon thread, writing progress + outcome to ``gen_state``.

    ``gen_state`` is a plain dict shared with the main script: the worker
    appends ``ProgressEvent`` objects to ``gen_state["log"]`` through a
    :class:`CollectingProgressListener` and stores the finished ``GeneratedTask``
    in ``gen_state["result"]`` (or an error string in ``gen_state["error"]``).
    The main script polls it on every rerun and renders the accumulated log.
    Nothing here touches ``st.session_state``.
    """
    listener = CollectingProgressListener(gen_state["log"])

    async def _run() -> None:
        try:
            task = await MathProblemGenerator().generate(
                prompt_input,
                judge=build_judge_ensemble() or None,
                listener=listener,
            )
            gen_state["result"] = task
        except RuntimeError as e:
            gen_state["error"] = str(e)
        except Exception as e:  # noqa: BLE001 - surface any failure in the UI
            gen_state["error"] = f"{type(e).__name__}: {e}"

    asyncio.run(_run())


def _create_task_state(assignment_id: int) -> dict:
    """Return the task-creation dialog's session state, reset for a new assignment."""
    state = st.session_state.setdefault("create_task_state", {})
    if state.get("assignment_id") != assignment_id:
        state = st.session_state["create_task_state"] = {
            "assignment_id": assignment_id,
            "log": [],
            "result": None,
            "error": None,
            "generating": False,
            "task_id": None,
            "prompt_input": None,
        }
    return state


def _close_create_task_dialog() -> None:
    """Close the task-creation dialog (drops its open flag + state) and rerun."""
    st.session_state.pop("create_task_open", None)
    st.session_state.pop("create_task_state", None)
    st.rerun()


def _render_progress_log(events: list) -> None:
    """Render the collected ``ProgressEvent`` log as icon + message lines."""
    for event in events:
        st.markdown(f"{_event_icon(event.stage)} {event.message}")


def _render_generation_progress(state: dict) -> None:
    """Live progress panel for an in-flight generation.

    The script re-runs every poll interval: each run renders the status panel
    with the log accumulated so far, then sleeps and reruns until the worker
    thread finishes. Because every rerun is a fresh render, the generation steps
    (attempt → validation → judge checks) update live instead of appearing all
    at once at the end.
    """
    status = st.status("✨ Generating task...", expanded=True)
    _render_progress_log(state["log"])

    if state.get("result") is not None:
        status.update(label="✅ Task generated and verified", state="complete")
        state["generating"] = False
        st.rerun()
        return
    if state.get("error") is not None:
        status.update(label="❌ Task generation failed", state="error")
        state["generating"] = False
        st.rerun()
        return
    time.sleep(POLL_INTERVAL_S)
    st.rerun()


def _render_create_task_form(assignment_id: int, state: dict) -> None:
    """The authoring form phase of the task-creation dialog."""
    st.caption("Describe the task below, then Generate.")

    # --- topic + dynamically suggested subtopics -----------------------------------

    topic = st.text_input(
        "📚 Topic",
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
        key="create_task_subtopics",
        help="LLM-suggested subtopics; select the ones to focus on.",
    )
    manual = st.text_input(
        "✍️ Add your own subtopics (comma-separated)",
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
        value=DEFAULT_TASK_CONTEXT,
        key="create_task_context",
        help="Thematic setting for the problem (e.g. Lego, Plants vs Zombies).",
    )

    col_grade, col_complexity = st.columns(2)
    grade = col_grade.selectbox(
        "🎓 Grade",
        options=list(range(1, 12)),
        index=4,
        key="create_task_grade",
        help="Student grade level (1-11).",
    )
    complexity = col_complexity.selectbox(
        "📊 Complexity",
        options=["easy", "medium", "hard"],
        index=0,
        key="create_task_complexity",
    )

    # --- generate / discard ---------------------------------------------------------

    generate_clicked = st.button(
        "🎯 Generate",
        type="primary",
        disabled=not (topic.strip() and subtopics),
        key="create_task_generate",
    )

    if generate_clicked:
        prompt_input = MathProblemGenerationPromptInput(
            topic=topic.strip(),
            subtopics=subtopics,
            context=context.strip() or DEFAULT_TASK_CONTEXT,
            complexity=complexity,
            grade=grade,
        )
        state["log"] = []
        state["result"] = None
        state["error"] = None
        state["generating"] = True
        state["prompt_input"] = prompt_input
        threading.Thread(
            target=_run_generation_in_thread,
            args=(prompt_input, state),
            daemon=True,
        ).start()
        st.rerun()

    if st.button("✖ Cancel", key="create_task_cancel"):
        _close_create_task_dialog()


def _render_create_task_result(assignment_id: int, state: dict) -> None:
    """The finished-task-card phase: preview + Save / Cancel."""
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
    """The failure phase: show the error + the progress log, offer a retry."""
    st.error(f"Task generation failed: {state['error']}")
    with st.expander("⚙️ Generation progress", expanded=True):
        _render_progress_log(state["log"])

    col_retry, col_cancel = st.columns(2)
    if col_retry.button("↻ Try again", key="create_task_retry"):
        state["generating"] = False
        state["result"] = None
        state["error"] = None
        st.rerun()
    if col_cancel.button("✖ Cancel", key="create_task_cancel"):
        _close_create_task_dialog()


@st.dialog("Task Creation", dismissible=True)
def create_task_dialog(assignment_id: int) -> None:
    """Author and save a task for ``assignment_id`` in a modal dialog.

    Walks through phases driven by ``create_task_state``: the authoring form, a
    live progress panel while generation runs in a background thread (the
    :class:`CollectingProgressListener` appends to the state's ``log``), then
    either the finished task card with a Save button or an error with a retry.
    Saving persists the task via storage and links it to the assignment.
    """
    state = _create_task_state(assignment_id)

    if state["generating"]:
        _render_generation_progress(state)
        return
    if state.get("result") is not None:
        _render_create_task_result(assignment_id, state)
        return
    if state.get("error") is not None:
        _render_create_task_error(state)
        return
    _render_create_task_form(assignment_id, state)
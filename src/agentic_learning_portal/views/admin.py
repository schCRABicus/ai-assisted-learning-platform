"""Admin endpoint of the Agentic Learning Portal.

Run the whole portal with ``uv run run-portal``; this page is served at
``/admin``.

The page is assignment-centric: you first create an assignment (title + id,
persisted immediately), then author its tasks one slot at a time in a native
Streamlit carousel. Each slide's content depends on the slot's state — an empty
slot shows a "Generate task" button, clicking it reveals the generation form
embedded in the slide, generation runs with a live progress panel, and a
generated slot shows the finished task card. A "➕" button to the right of the
carousel adds another task slot. Every generated task is persisted and linked
to the assignment.
"""

from __future__ import annotations

import asyncio
import threading
import time

import streamlit as st

from agentic_learning_portal.admin.formatting import latex_to_plain_text
from agentic_learning_portal.admin.subtopic_suggester import suggest_subtopics
from agentic_learning_portal.api.llm import MODELS
from agentic_learning_portal.api.progress import CollectingProgressListener
from agentic_learning_portal.auth import current_user, get_storage, require_roles
from agentic_learning_portal.domains.math import (
    MathProblemGenerationPromptInput,
    MathProblemGenerator,
    build_judge_ensemble,
)

# Gate the page before any widget (the topic input below fires an LLM call for
# subtopic suggestions, so it must never run for unauthenticated visitors).
require_roles("admin", "teacher")

st.title("🎓 Task Generation Admin")

DEFAULT_CONTEXT = "Everyday life"
DEFAULT_MODEL = MODELS["subtopic_suggestion"]
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


def _run_generation_in_thread(
    prompt_input: MathProblemGenerationPromptInput,
    gen_state: dict,
) -> None:
    """Generate a task in a daemon thread, writing progress + outcome to ``gen_state``.

    ``gen_state`` is a plain dict shared with the main script: the worker
    appends ``ProgressEvent`` objects to ``gen_state["log"]`` and stores the
    finished ``GeneratedTask`` in ``gen_state["result"]`` (or an error string in
    ``gen_state["error"]``). The main script polls it on every rerun and renders
    the accumulated log. Nothing here touches ``st.session_state``.
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


def _run_suggest_subtopics(topic: str, model: str) -> list[str]:
    return asyncio.run(suggest_subtopics(topic, model=model))


# --- assignment carousel helpers ---------------------------------------------------


def _clear_assignment_state() -> None:
    """Clear all assignment-related session state (cancel/finish)."""
    st.session_state["assignment_id"] = None
    st.session_state["assignment_title"] = ""
    st.session_state["assignment_active"] = False
    st.session_state["assignment_task_count"] = 0
    st.session_state["assignment_current_index"] = 0
    # Clear per-slot carousel state so a new assignment starts fresh.
    for key in list(st.session_state.keys()):
        if key.startswith("carousel_"):
            del st.session_state[key]


def _persist_generated_task(index: int, task: object) -> None:
    """Persist ``task`` and link it to the active assignment (idempotent)."""
    if st.session_state.get(f"carousel_task_id_{index}") is not None:
        return
    storage = get_storage()
    persisted = storage.create_task(task)
    storage.add_task_to_assignment(st.session_state["assignment_id"], persisted.id)
    st.session_state[f"carousel_task_id_{index}"] = persisted.id


def _sync_assignment_tasks_from_storage() -> None:
    """Backfill per-slot session state from the assignment's persisted tasks.

    Storage is the source of truth: any task already linked to the active
    assignment (e.g. generated earlier in this session) is restored into its
    slot so the card can never be lost to session-state churn.
    """
    assignment_id = st.session_state.get("assignment_id")
    if assignment_id is None:
        return
    try:
        tasks = get_storage().list_assignment_tasks(assignment_id)
    except Exception:  # noqa: BLE001 - best-effort backfill
        return
    for i, task in enumerate(tasks):
        # Don't clobber a task that is already in session state.
        if st.session_state.get(f"carousel_last_task_{i}") is None:
            st.session_state[f"carousel_last_task_{i}"] = task


# --- per-slot renderers ------------------------------------------------------------


def _render_empty_slot(index: int) -> None:
    """An authored-but-not-generated slot: just the Generate button."""
    st.markdown(f"**Task {index + 1}**")
    st.caption("No task generated for this slot yet.")
    if st.button(
        "✨ Generate task",
        key=f"carousel_open_form_{index}",
        type="primary",
        help="Open the authoring form for this task slot.",
    ):
        st.session_state[f"carousel_form_open_{index}"] = True
        st.rerun()


def _render_generation_form(index: int) -> None:
    """The authoring form for one slot, embedded in the carousel slide.

    This reuses the same patterns as a task-generation form but with unique
    session-state keys per slot so the widgets don't collide across slots.
    """
    suggested_for_key = f"carousel_suggested_for_{index}"
    suggested_subtopics_key = f"carousel_suggested_subtopics_{index}"
    selected_subtopics_key = f"carousel_selected_subtopics_{index}"

    st.markdown(f"**Task {index + 1}**")
    st.caption("Describe the task below, then Generate.")

    # --- topic + dynamically suggested subtopics -----------------------------------

    topic = st.text_input(
        "📚 Topic",
        placeholder="e.g. Arithmetic, Algebra, Geometry",
        help="The math topic to generate a task for.",
        key=f"carousel_topic_{index}",
    )

    if topic.strip() and topic.strip().lower() != st.session_state.get(
        suggested_for_key, ""
    ).lower():
        with st.spinner("🔍 Suggesting subtopics..."):
            suggestions = _run_suggest_subtopics(topic, DEFAULT_MODEL)
        st.session_state[suggested_subtopics_key] = list(suggestions)
        st.session_state[suggested_for_key] = topic.strip()
        st.session_state[selected_subtopics_key] = []
        if not suggestions:
            st.warning(
                "The LLM could not suggest subtopics for this topic. Add them "
                "manually below."
            )

    suggested = st.session_state.get(suggested_subtopics_key, [])
    selected_subtopics = st.multiselect(
        "🧩 Subtopics",
        options=suggested,
        key=selected_subtopics_key,
        help="LLM-suggested subtopics; select the ones to focus on.",
    )
    manual = st.text_input(
        "✍️ Add your own subtopics (comma-separated)",
        placeholder="e.g. fractions, word problems",
        key=f"carousel_manual_subtopics_{index}",
    )
    manual_subtopics = [s.strip() for s in manual.split(",") if s.strip()]
    subtopics = list(selected_subtopics) + [
        s for s in manual_subtopics if s not in selected_subtopics
    ]

    if not suggested:
        st.caption("💡 Enter a topic to get LLM-suggested subtopics.")
    elif not selected_subtopics:
        st.caption("💡 Select the subtopics to focus on (or add your own below).")

    # --- other parameters ----------------------------------------------------------

    context = st.text_input(
        "🎭 Context",
        value=DEFAULT_CONTEXT,
        key=f"carousel_context_{index}",
        help="Thematic setting for the problem (e.g. Lego, Plants vs Zombies).",
    )

    col_grade, col_complexity = st.columns(2)
    grade = col_grade.selectbox(
        "🎓 Grade",
        options=list(range(1, 12)),
        index=4,
        key=f"carousel_grade_{index}",
        help="Student grade level (1-11).",
    )
    complexity = col_complexity.selectbox(
        "📊 Complexity",
        options=["easy", "medium", "hard"],
        index=0,
        key=f"carousel_complexity_{index}",
    )

    # --- generate / discard ---------------------------------------------------------

    generate_clicked = st.button(
        "🎯 Generate",
        type="primary",
        disabled=not (topic.strip() and subtopics),
        key=f"carousel_generate_{index}",
    )

    if generate_clicked:
        prompt_input = MathProblemGenerationPromptInput(
            topic=topic.strip(),
            subtopics=subtopics,
            context=context.strip() or DEFAULT_CONTEXT,
            complexity=complexity,
            grade=grade,
        )
        st.session_state[f"carousel_last_prompt_{index}"] = prompt_input
        st.session_state[f"carousel_gen_state_{index}"] = {
            "log": [],
            "result": None,
            "error": None,
        }
        st.session_state[f"carousel_generating_{index}"] = True
        threading.Thread(
            target=_run_generation_in_thread,
            args=(
                prompt_input,
                st.session_state[f"carousel_gen_state_{index}"],
            ),
            daemon=True,
        ).start()
        st.rerun()

    if st.button(
        "✖ Discard",
        key=f"carousel_discard_{index}",
        help="Close this form without generating a task.",
    ):
        st.session_state[f"carousel_form_open_{index}"] = False
        st.rerun()


def _render_progress(index: int) -> None:
    """Live progress panel for an in-flight generation in this slot.

    The script re-runs every poll interval: each run renders the status panel
    with the log accumulated so far, then sleeps and reruns until the worker
    thread finishes. Because every rerun is a fresh render, the generation steps
    (attempt → validation → judge checks) update live instead of appearing all
    at once at the end.
    """
    gen_state = st.session_state.get(f"carousel_gen_state_{index}", {})
    status = st.status(f"✨ Generating task {index + 1}...", expanded=True)
    for event in gen_state.get("log", []):
        st.markdown(f"{_event_icon(event.stage)} {event.message}")

    if gen_state.get("result") is not None or gen_state.get("error") is not None:
        if gen_state.get("error") is not None:
            status.update(label="❌ Task generation failed", state="error")
            st.session_state[f"carousel_generation_status_{index}"] = (
                "error",
                f"Could not generate a validated task: {gen_state['error']}",
            )
        else:
            status.update(label="✅ Task generated and verified", state="complete")
            st.session_state[f"carousel_last_task_{index}"] = gen_state["result"]
            st.session_state[f"carousel_generation_status_{index}"] = (
                "success",
                "Task generated and schema-validated.",
            )
            # Persist the generated task and link it to the assignment.
            _persist_generated_task(index, gen_state["result"])
        st.session_state[f"carousel_generating_{index}"] = False
        st.session_state[f"carousel_form_open_{index}"] = False
        st.rerun()
    else:
        time.sleep(POLL_INTERVAL_S)
        st.rerun()


def _render_generated_task(index: int) -> None:
    """The finished task card for a slot whose generation completed."""
    task = st.session_state.get(f"carousel_last_task_{index}")
    # The prompt is only available in-session; storage-synced tasks (backfilled
    # by ``_sync_assignment_tasks_from_storage``) have none, so degrade to "—".
    prompt_input = st.session_state.get(f"carousel_last_prompt_{index}")
    grade = prompt_input.grade if prompt_input is not None else "—"
    subtopics = (
        ", ".join(prompt_input.subtopics) if prompt_input is not None else "—"
    )

    status = st.session_state.pop(f"carousel_generation_status_{index}", None)
    if status is not None:
        kind, message = status
        if kind == "success":
            st.success(message)
        else:
            st.error(message)

    st.markdown(f"**Task {index + 1}**")
    meta = st.columns(4)
    meta[0].markdown(f"**📚 Topic:** {task.topic}")
    meta[1].markdown(f"**🎓 Grade:** {grade}")
    meta[2].markdown(f"**📊 Complexity:** {task.complexity}")
    meta[3].markdown(f"**🧩 Subtopics:** {subtopics}")

    st.markdown("---")
    st.markdown(f"**📝 Problem**\n\n{latex_to_plain_text(task.text)}")
    st.markdown(f"**✅ Correct answer:** `{task.correct_answer}`")
    with st.expander("💡 Solution", expanded=False):
        st.markdown(latex_to_plain_text(task.solution))
    st.markdown("---")


def _render_carousel_slide(index: int) -> None:
    """Render one carousel slide; content depends on the slot's state."""
    if st.session_state.get(f"carousel_last_task_{index}") is not None:
        _render_generated_task(index)
    elif st.session_state.get(f"carousel_generating_{index}", False):
        _render_progress(index)
    elif st.session_state.get(f"carousel_form_open_{index}", False):
        _render_generation_form(index)
    else:
        _render_empty_slot(index)


# --- persistent state -------------------------------------------------------------

if "assignment_id" not in st.session_state:
    st.session_state["assignment_id"] = None
if "assignment_title" not in st.session_state:
    st.session_state["assignment_title"] = ""
if "assignment_active" not in st.session_state:
    st.session_state["assignment_active"] = False
if "assignment_task_count" not in st.session_state:
    st.session_state["assignment_task_count"] = 0
if "assignment_current_index" not in st.session_state:
    st.session_state["assignment_current_index"] = 0

user = current_user()

# --- landing: create an assignment ------------------------------------------------

if not st.session_state["assignment_active"]:
    st.subheader("🎯 Create an assignment")
    st.caption(
        "Name an assignment, then author its tasks one at a time in a carousel — "
        "each generated task is saved to the assignment automatically."
    )
    col_title, col_add = st.columns([3, 1], vertical_alignment="center")
    with col_title:
        assignment_title_input = st.text_input(
            "Assignment title",
            key="assignment_title_input",
            placeholder="e.g. Week 3 Algebra Practice",
            help="Group generated tasks into an assignment.",
        )
    with col_add:
        if st.button(
            "➕ Add assignment",
            type="primary",
            disabled=not assignment_title_input.strip(),
            use_container_width=True,
        ):
            storage = get_storage()
            assignment = storage.create_assignment(
                title=assignment_title_input.strip(),
                created_by=user.id,
            )
            st.session_state["assignment_id"] = assignment.id
            st.session_state["assignment_title"] = assignment.title
            st.session_state["assignment_active"] = True
            st.session_state["assignment_task_count"] = 1
            st.session_state["assignment_current_index"] = 0
            st.rerun()
    st.stop()

# --- carousel mode (an assignment is active) --------------------------------------

st.markdown(f"### 📝 {st.session_state['assignment_title']}")

top_cols = st.columns([3, 1, 1])
with top_cols[1]:
    if st.button(
        "❌ Cancel assignment",
        key="carousel_cancel",
        use_container_width=True,
    ):
        st.warning("Assignment canceled — its saved tasks stay in storage.")
        _clear_assignment_state()
        st.rerun()
with top_cols[2]:
    if st.button(
        "🏁 Finish",
        key="carousel_finish",
        type="primary",
        use_container_width=True,
    ):
        st.balloons()
        st.success(
            f"Assignment **{st.session_state['assignment_title']}** "
            f"(id={st.session_state['assignment_id']}) complete."
        )
        _clear_assignment_state()
        st.rerun()

# Storage is the source of truth: restore any persisted task into its slot
# (e.g. one generated earlier in this session) before rendering the slide.
_sync_assignment_tasks_from_storage()

total = st.session_state["assignment_task_count"]
idx = st.session_state["assignment_current_index"]

# Native carousel: the current slide (its content depends on slot state) flanked
# by ◀/▶ arrows that are part of the carousel itself, with the "+" to the right.
car_col, add_col = st.columns([6, 1], vertical_alignment="center")
with car_col:
    carousel_inner = st.columns([1, 8, 1], vertical_alignment="center")
    with carousel_inner[0]:
        if idx > 0:
            if st.button(
                "◀",
                key="carousel_prev",
                help="Previous task",
                use_container_width=True,
            ):
                st.session_state["assignment_current_index"] = idx - 1
                st.rerun()
    with carousel_inner[1]:
        with st.container(border=True):
            _render_carousel_slide(idx)
    with carousel_inner[2]:
        if idx < total - 1:
            if st.button(
                "▶",
                key="carousel_next",
                help="Next task",
                use_container_width=True,
            ):
                st.session_state["assignment_current_index"] = idx + 1
                st.rerun()
with add_col:
    if st.button(
        "➕ Add task",
        key="carousel_add",
        help="Add another task slot to this assignment.",
        use_container_width=True,
    ):
        st.session_state["assignment_task_count"] = total + 1
        st.session_state["assignment_current_index"] = total
        st.rerun()

# Position indicator below the carousel: counter + clickable-free dots.
st.markdown(f"**Task {idx + 1} of {total}**")
dots = "  ".join("●" if i == idx else "○" for i in range(total))
st.caption(dots)

st.stop()  # Nothing below: the page is assignment-only.
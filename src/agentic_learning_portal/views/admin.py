"""Admin endpoint of the Agentic Learning Portal.

Run the whole portal with ``uv run run-portal``; this page is served at
``/admin``.

The page is assignment-centric: you first create an assignment (title + id,
persisted immediately), then author its tasks one slot at a time in a native
Streamlit carousel. Each slide's content depends on the slot's state — an empty
slot shows a "Generate task" button that opens the shared task-creation dialog
(``views/components/create_task_dialog.create_task_dialog``); a slot whose task was saved to
storage shows the finished task card. A "➕" button to the right of the carousel
adds another task slot. All task authoring happens through the dialog, which
persists the task to storage and links it to the assignment; slots then backfill
their cards from storage.
"""

from __future__ import annotations

import streamlit as st

from agentic_learning_portal.admin.formatting import latex_to_plain_text
from agentic_learning_portal.auth import current_user, get_storage, require_roles
from agentic_learning_portal.views.components.create_task_dialog import (
    create_task_dialog,
)


# --- assignment carousel helpers ---------------------------------------------------


def _clear_assignment_state() -> None:
    """Clear all assignment-related session state (cancel/finish)."""
    st.session_state["assignment_id"] = None
    st.session_state["assignment_title"] = ""
    st.session_state["assignment_active"] = False
    st.session_state["assignment_task_count"] = 0
    st.session_state["assignment_current_index"] = 0
    # Close any task-creation dialog opened for this assignment.
    st.session_state.pop("create_task_open", None)
    st.session_state.pop("create_task_state", None)
    # Clear per-slot carousel state so a new assignment starts fresh.
    for key in list(st.session_state.keys()):
        if key.startswith("carousel_"):
            del st.session_state[key]


def _sync_assignment_tasks_from_storage() -> None:
    """Backfill per-slot session state from the assignment's persisted tasks.

    Storage is the source of truth: any task already linked to the active
    assignment (e.g. saved through the task-creation dialog) is restored into
    its slot so the card can never be lost to session-state churn.
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
    """An authored-but-not-generated slot: just the button to open the dialog."""
    st.markdown(f"**Task {index + 1}**")
    st.caption("No task generated for this slot yet.")
    if st.button(
        "✨ Generate task",
        key=f"carousel_open_form_{index}",
        type="primary",
        help="Open the task-creation dialog for this assignment.",
    ):
        st.session_state["create_task_open"] = st.session_state["assignment_id"]
        st.rerun()


def _render_generated_task(index: int) -> None:
    """The finished task card for a slot whose task was saved to storage."""
    task = st.session_state.get(f"carousel_last_task_{index}")
    # The prompt is only available in-session; storage-synced tasks (backfilled
    # by ``_sync_assignment_tasks_from_storage``) have none, so degrade to "—".
    prompt_input = st.session_state.get(f"carousel_last_prompt_{index}")
    grade = prompt_input.grade if prompt_input is not None else "—"
    subtopics = (
        ", ".join(prompt_input.subtopics) if prompt_input is not None else "—"
    )

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

# Gate the page before any widget (task generation must never run for
# unauthenticated visitors).
@require_roles("admin", "teacher")
def render_page() -> None:
    st.title("🎓 Task Generation Admin")

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
    # (e.g. one saved through the dialog earlier in this session) before rendering
    # the slide.
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

    # The task-creation dialog overlays the page while its open flag is set. It is
    # called on every run so it survives its own poll-loop reruns; the dialog closes
    # itself by clearing the flag when the task is saved or canceled.
    if st.session_state.get("create_task_open") is not None:
        create_task_dialog(st.session_state["create_task_open"])

    st.stop()  # Nothing below: the page is assignment-only.

render_page()
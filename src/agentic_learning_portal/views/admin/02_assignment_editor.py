"""Assignment editor endpoint of the Agentic Learning Portal.

Served at ``/assignment_editor`` (via ``st.navigation`` in ``app.py``), reachable
from the assignments list by clicking **Edit** on an assignment. The page renders
a native Streamlit carousel over the assignment's tasks — each slide shows the
problem text, the correct answer, and the solution — with per-slide **Edit** and
**Remove** buttons that open modal dialogs
(``views/components/edit_task_dialog.edit_task_dialog`` /
``views/components/remove_task_dialog.remove_task_dialog``).
**➕ Add task** opens the shared task-authoring dialog
(``views/components/create_task_dialog.create_task_dialog``); when it saves, the
carousel jumps to the new task.

The assignment being edited is passed from the assignments page through
``st.session_state["edit_assignment_id"]`` (session state survives
``st.switch_page``), so a direct visit without a selection shows a "no
assignment selected" callout with a way back.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from agentic_learning_portal.admin.formatting import latex_to_plain_text
from agentic_learning_portal.auth import get_storage, require_roles
from agentic_learning_portal.storage import Task
from agentic_learning_portal.views.components.create_task_dialog import (
    create_task_dialog,
)
from agentic_learning_portal.views.components.edit_task_dialog import edit_task_dialog
from agentic_learning_portal.views.components.remove_task_dialog import (
    remove_task_dialog,
)

ASSIGNMENTS_PAGE = str(Path(__file__).parent / "01_assignments.py")


def _clear_task_dialogs() -> None:
    """Close every task dialog (Streamlit allows only one ``st.dialog`` per run).

    Drops the open flags for the create, edit, and remove dialogs plus the
    post-save jump marker, so a dialog can't leak across pages or survive a
    change of assignment.
    """
    for flag in (
        "create_task_open",
        "edit_task_open",
        "remove_task_open",
        "editor_jump_to_last",
    ):
        st.session_state.pop(flag, None)


def _open_task_dialog(flag: str, value: int) -> None:
    """Open exactly one task dialog, closing any other open one."""
    _clear_task_dialogs()
    st.session_state[flag] = value
    st.rerun()


def _render_task_slide(task: Task, index: int, total: int) -> None:
    """Render one carousel slide: the task's problem, answer, and solution."""
    st.markdown(f"**Task {index + 1} of {total}**")
    meta = st.columns(3)
    meta[0].markdown(f"**📚 Topic:** {task.topic}")
    meta[1].markdown(f"**📊 Complexity:** {task.complexity}")
    meta[2].markdown(f"**🆔 id:** {task.id}")

    st.markdown("---")
    st.markdown(f"**📝 Problem**\n\n{latex_to_plain_text(task.text)}")
    st.markdown(f"**✅ Correct answer:** `{task.correct_answer}`")
    st.markdown(f"**💡 Solution**\n\n{latex_to_plain_text(task.solution)}")
    st.markdown("---")

    col_edit, col_remove = st.columns(2)
    if col_edit.button(
        "✏️ Edit task",
        key=f"edit_task_{task.id}",
        type="primary",
        help="Open the edit dialog for this task.",
    ):
        _open_task_dialog("edit_task_open", task.id)
    if col_remove.button(
        "🗑 Remove task",
        key=f"remove_task_{task.id}",
        help="Remove this task from the assignment.",
    ):
        _open_task_dialog("remove_task_open", task.id)


@require_roles("admin", "teacher")
def render_assignment_editor_page() -> None:
    st.set_page_config(page_title="✏️ Assignment Editor", page_icon="✏️")

    st.title("✏️ Assignment Editor")

    assignment_id = st.session_state.get("edit_assignment_id")
    if assignment_id is None:
        st.info("No assignment selected. Choose one from the Assignments page.")
        if st.button("← Back to Assignments", key="editor_back_home"):
            st.switch_page(ASSIGNMENTS_PAGE)
        st.stop()

    storage = get_storage()
    assignment = storage.get_assignment(assignment_id)
    if assignment is None:
        st.error(f"Assignment #{assignment_id} not found.")
        if st.button("← Back to Assignments", key="editor_back_home"):
            st.switch_page(ASSIGNMENTS_PAGE)
        st.stop()

    # Editing a different assignment resets the carousel position and closes any
    # task dialog left over from a previous assignment.
    if st.session_state.get("editor_assignment_id") != assignment_id:
        st.session_state["editor_assignment_id"] = assignment_id
        st.session_state["editor_current_index"] = 0
        _clear_task_dialogs()

    col_title, col_add, col_back = st.columns([6, 1, 1], vertical_alignment="center")
    with col_title:
        st.markdown(f"### {assignment.title}")
    with col_add:
        if st.button(
            "➕ Add task",
            key="editor_add_task",
            use_container_width=True,
            help="Add a new task to this assignment.",
        ):
            # Set both flags before ``st.rerun()`` (which immediately stops the
            # run): ``create_task_open`` opens the dialog, ``editor_jump_to_last``
            # tells the page to jump the carousel to the new task once it saves.
            _clear_task_dialogs()
            st.session_state["create_task_open"] = assignment_id
            st.session_state["editor_jump_to_last"] = True
            st.rerun()
    with col_back:
        # if st.button("← Back", key="editor_back", use_container_width=True):
        #     _clear_task_dialogs()
        #     st.switch_page(ASSIGNMENTS_PAGE)
        st.button("← Back", key="editor_back", use_container_width=True, on_click=lambda :st.switch_page(ASSIGNMENTS_PAGE))

    tasks = storage.list_assignment_tasks(assignment_id)

    # The create-task dialog just closed: jump to the newly added (last) task.
    if (
        st.session_state.get("editor_jump_to_last")
        and st.session_state.get("create_task_open") is None
    ):
        st.session_state.pop("editor_jump_to_last", None)
        if tasks:
            st.session_state["editor_current_index"] = len(tasks) - 1

    if not tasks:
        st.info("No tasks in this assignment yet — use ➕ Add task to create one.")
        # The first task is added through the create dialog; dispatch it here too,
        # since ``st.stop()`` below would otherwise cut the page off before the
        # dialog dispatch at the end of this function.
        if st.session_state.get("create_task_open") is not None:
            create_task_dialog(st.session_state["create_task_open"])
        st.stop()

    total = len(tasks)
    idx = min(st.session_state["editor_current_index"], total - 1)

    # Native carousel: the current task slide flanked by ◀/▶ arrows.
    carousel_inner = st.columns([1, 8, 1], vertical_alignment="center")
    with carousel_inner[0]:
        if idx > 0:
            if st.button(
                "◀",
                key="editor_prev",
                help="Previous task",
                use_container_width=True,
            ):
                st.session_state["editor_current_index"] = idx - 1
                st.rerun()
    with carousel_inner[1]:
        with st.container(border=True):
            _render_task_slide(tasks[idx], idx, total)
    with carousel_inner[2]:
        if idx < total - 1:
            if st.button(
                "▶",
                key="editor_next",
                help="Next task",
                use_container_width=True,
            ):
                st.session_state["editor_current_index"] = idx + 1
                st.rerun()

    # Position indicator below the carousel: counter + dots.
    st.markdown(f"**Task {idx + 1} of {total}**")
    dots = "  ".join("●" if i == idx else "○" for i in range(total))
    st.caption(dots)

    # Task dialogs overlay the page while their open flag is set. Only one may
    # be open per run (Streamlit allows a single ``st.dialog``), so they are
    # dispatched mutually-exclusively; each dialog closes itself by clearing its
    # flag when done.
    if st.session_state.get("create_task_open") is not None:
        create_task_dialog(st.session_state["create_task_open"])
    elif st.session_state.get("edit_task_open") is not None:
        edit_task_dialog(st.session_state["edit_task_open"])
    elif st.session_state.get("remove_task_open") is not None:
        remove_task_dialog(assignment_id, st.session_state["remove_task_open"])


render_assignment_editor_page()

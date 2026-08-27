"""The edit-task dialog (views/components/edit_task_dialog.py).

:func:`edit_task_dialog` is a Streamlit ``st.dialog`` opened by the assignment
editor page (``views/admin/02_assignment_editor.py``) through the
``edit_task_open`` session-state flag: the page sets the flag to the task id and
re-calls the dialog on every run while it's set, so it survives the
``st.rerun()`` calls it makes. It reads the task fresh from storage on every
open and shows its current fields pre-populated; Save persists the changes via
``storage.update_task``, Cancel leaves the task untouched.
"""

from __future__ import annotations

import streamlit as st

from agentic_learning_portal.storage import Storage
from agentic_learning_portal.storage.factory import StorageFactory

STORAGE_FACTORY = StorageFactory()


def get_storage() -> Storage:
    """Return the :class:`Storage` the portal should use.

    The backend is chosen by ``STORAGE_BACKEND`` (``"sqlite"`` or ``"memory"``),
    read at call time so tests and ``.env`` can switch it after import. The
    sqlite backend keeps one instance per thread (``seed_admin_from_env`` runs
    on construction, so the ``.env``-configured admin is available from the
    start); the memory backend is a single shared instance.
    """
    return STORAGE_FACTORY.get_storage()


def _close_edit_task_dialog() -> None:
    """Close the edit-task dialog (drops its open flag) and rerun."""
    st.session_state.pop("edit_task_open", None)
    st.rerun()


@st.dialog("Edit Task", dismissible=True)
def edit_task_dialog(task_id: int) -> None:
    """Edit an existing task's fields in a modal dialog.

    Reads the task fresh from storage on every open and shows its current
    fields pre-populated; Save persists the changes via
    ``storage.update_task``, Cancel leaves the task untouched. Opened by a
    page through the ``edit_task_open`` session-state flag.
    """
    storage = get_storage()
    task = storage.get_task(task_id)
    if task is None:
        st.error(f"Task #{task_id} not found.")
        if st.button("✖ Close", key="edit_task_close"):
            _close_edit_task_dialog()
        return

    st.caption("Update the task fields below.")

    topic = st.text_input(
        "📚 Topic",
        value=task.topic,
        key="edit_task_topic",
    )
    complexity = st.selectbox(
        "📊 Complexity",
        options=["easy", "medium", "hard"],
        index=["easy", "medium", "hard"].index(task.complexity),
        key="edit_task_complexity",
    )
    correct_answer = st.text_input(
        "✅ Correct answer",
        value=str(task.correct_answer),
        key="edit_task_answer",
    )
    text = st.text_area(
        "📝 Problem",
        value=task.text,
        key="edit_task_text",
    )
    solution = st.text_area(
        "💡 Solution",
        value=task.solution,
        key="edit_task_solution",
    )

    col_save, col_cancel = st.columns(2)
    if col_save.button(
        "💾 Save",
        type="primary",
        key="edit_task_save",
        disabled=not topic.strip(),
    ):
        storage.update_task(
            task_id,
            topic=topic.strip(),
            text=text.strip(),
            complexity=complexity,
            correct_answer=correct_answer.strip(),
            solution=solution.strip(),
        )
        st.success("Task updated.")
        _close_edit_task_dialog()
    if col_cancel.button("✖ Cancel", key="edit_task_cancel"):
        _close_edit_task_dialog()

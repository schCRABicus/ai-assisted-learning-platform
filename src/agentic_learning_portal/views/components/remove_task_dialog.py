"""The remove-task dialog (views/components/remove_task_dialog.py).

:func:`remove_task_dialog` is a Streamlit ``st.dialog`` opened by the assignment
editor page (``views/admin/02_assignment_editor.py``) through the
``remove_task_open`` session-state flag: the page sets the flag to the task id
and re-calls the dialog on every run while it's set, so it survives the
``st.rerun()`` calls it makes. Confirming unlinks the task from the assignment
(and deletes it when no other assignment references it); Cancel leaves the
assignment untouched.
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


def _close_remove_task_dialog() -> None:
    """Close the remove-task dialog (drops its open flag) and rerun."""
    st.session_state.pop("remove_task_open", None)
    st.rerun()


@st.dialog("Remove Task", dismissible=True)
def remove_task_dialog(assignment_id: int, task_id: int) -> None:
    """Confirm removing ``task_id`` from ``assignment_id`` in a modal dialog.

    Confirming calls ``storage.remove_task_from_assignment`` (which unlinks the
    task and deletes it only when no other assignment references it); Cancel
    leaves the assignment untouched. Opened by a page through the
    ``remove_task_open`` session-state flag.
    """
    storage = get_storage()
    task = storage.get_task(task_id)
    label = f"'{task.topic}'" if task is not None else f"task #{task_id}"
    st.write(
        f"Remove **{label}** from this assignment? "
        "The task is deleted when no other assignment uses it."
    )

    col_remove, col_cancel = st.columns(2)
    if col_remove.button("🗑 Remove", type="primary", key="remove_task_confirm"):
        storage.remove_task_from_assignment(assignment_id, task_id)
        st.success("Task removed.")
        _close_remove_task_dialog()
    if col_cancel.button("✖ Cancel", key="remove_task_cancel"):
        _close_remove_task_dialog()

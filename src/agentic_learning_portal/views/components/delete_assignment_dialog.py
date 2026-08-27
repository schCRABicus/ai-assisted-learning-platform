"""The delete-assignment confirmation dialog (views/components/delete_assignment_dialog.py).

:func:`delete_assignment_dialog` is a Streamlit ``st.dialog`` opened by the
assignments page (``views/admin/01_assignments.py``) from its **Delete** button
callback. Confirming deletes the assignment from storage; Cancel (or dismissing
the dialog) leaves it untouched.
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


@st.dialog("Confirm Assignment Deletion", dismissible=True)
def delete_assignment_dialog(assignment_id: int) -> None:
    """Confirm the deletion of ``assignment_id`` in a modal dialog.

    Confirming calls ``storage.delete_assignment`` (irreversible); Cancel leaves
    the assignment untouched. Opened by the assignments page through its Delete
    button callback.
    """
    st.write("Are you sure you want to delete this assignment? This cannot be undone.")

    col1, col2 = st.columns(2)

    if col1.button("Yes, Delete", type="primary"):
        get_storage().delete_assignment(assignment_id)
        st.success("Item deleted successfully!")
        st.rerun()

    if col2.button("Cancel"):
        st.rerun()

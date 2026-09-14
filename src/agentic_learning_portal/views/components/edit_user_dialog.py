"""The edit-user dialog (views/components/edit_user_dialog.py).

:func:`edit_user_dialog` is a Streamlit ``st.dialog`` opened by the users page
(``views/admin/03_users.py``) through the ``edit_user_open`` session-state flag
(its value is the user id). It reads the user fresh from storage and shows their
username, email, and roles pre-populated; Save persists via ``storage.update_user``,
Cancel leaves the user untouched.

An email change re-triggers verification: the new address can't be signed in
against until it's confirmed, so the admin should re-issue an invite after a
change. Leaving the email blank leaves it unchanged (``update_user`` treats
``None`` as "don't touch").
"""

from __future__ import annotations

import streamlit as st

from agentic_learning_portal.auth import get_storage


def _close_edit_user_dialog() -> None:
    """Close the dialog (drop its open flag) and rerun."""
    st.session_state.pop("edit_user_open", None)
    st.rerun()


@st.dialog("Edit user", dismissible=True)
def edit_user_dialog(user_id: int) -> None:
    """Edit an existing user's username, email, and roles in a modal dialog."""
    storage = get_storage()
    user = storage.get_user(user_id)
    if user is None:
        st.error(f"User #{user_id} not found.")
        if st.button("Close", key="edit_user_close"):
            _close_edit_user_dialog()
        return

    role_options = [r.name for r in storage.list_roles()]

    username = st.text_input("Username", value=user.username, key="edit_user_username")
    email = st.text_input(
        "Email",
        value=user.email or "",
        key="edit_user_email",
        help="Leave blank to keep the current email. Changing it re-verifies the account.",
    )
    selected_roles = st.multiselect(
        "Roles",
        options=role_options,
        default=[r for r in user.roles],
        key="edit_user_roles",
    )

    col_save, col_cancel = st.columns(2)
    if col_save.button(
        "💾 Save",
        type="primary",
        key="edit_user_save",
        disabled=not username.strip() or not selected_roles,
    ):
        try:
            storage.update_user(
                user_id,
                username=username.strip(),
                email=email.strip() or None,
                roles=selected_roles,
            )
        except ValueError as exc:
            st.error(str(exc))
        else:
            st.success("User updated.")
            _close_edit_user_dialog()
    if col_cancel.button("✖ Cancel", key="edit_user_cancel"):
        _close_edit_user_dialog()
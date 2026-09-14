"""The invite-a-user dialog (views/components/create_user_dialog.py).

:func:`create_user_dialog` is a Streamlit ``st.dialog`` opened by the users page
(``views/admin/03_users.py``) through the ``create_user_open`` session-state
flag. It collects a username, email, and roles; on submit it creates the user
*unverified* (no password, so they can't sign in), issues a one-time
verification token, and hands it to the mailer — whose dev stub just logs and
returns the link, which the dialog then surfaces for the admin to copy.
"""

from __future__ import annotations

import streamlit as st

from agentic_learning_portal.auth import get_storage
from agentic_learning_portal.mailer import build_verify_url, send_invite_email

# Holds the just-created invite's username + link while the success pane shows,
# so the dialog can render the copyable URL without re-reading storage.
_INVITE_RESULT = "_invite_result"


def _close_create_user_dialog() -> None:
    """Close the dialog (drop its open flag and any pending result) and rerun."""
    st.session_state.pop("create_user_open", None)
    st.session_state.pop(_INVITE_RESULT, None)
    st.rerun()


@st.dialog("Invite a user", dismissible=True)
def create_user_dialog() -> None:
    """Collect a new user's details, create them, and show the invite link.

    The user is created with ``email_verified=False`` and no password; the
    verification link (sent via the mailer stub) is what lets them set a
    password and finish signing up.
    """
    storage = get_storage()

    result = st.session_state.get(_INVITE_RESULT)
    if result is not None:
        st.success(f"Invited **{result['username']}**!")
        st.write("Verification link (dev stub — no real email is sent):")
        st.code(result["url"])
        st.caption(
            "Copy this link and send it to the user. They'll set a password there."
        )
        if st.button("Done", type="primary", key="create_user_done"):
            _close_create_user_dialog()
        return

    role_options = [r.name for r in storage.list_roles()]
    st.caption("The invited user can't sign in until they open the link and set a password.")

    username = st.text_input("Username", key="create_user_username")
    email = st.text_input("Email", key="create_user_email")
    selected_roles = st.multiselect(
        "Roles",
        options=role_options,
        default=["student"],
        key="create_user_roles",
    )

    col_create, col_cancel = st.columns(2)
    if col_create.button(
        "Create & send invite",
        type="primary",
        key="create_user_submit",
        disabled=not username.strip() or not email.strip() or not selected_roles,
    ):
        try:
            user = storage.create_user(
                username.strip(),
                selected_roles,
                email=email.strip(),
                email_verified=False,
            )
            token = storage.issue_verification_token(user.id)
            url = send_invite_email(email.strip(), build_verify_url(token))
        except ValueError as exc:
            st.error(str(exc))
        else:
            st.session_state[_INVITE_RESULT] = {"username": user.username, "url": url}
            st.rerun()
    if col_cancel.button("Cancel", key="create_user_cancel"):
        _close_create_user_dialog()
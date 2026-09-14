"""Account-verification endpoint of the Agentic Learning Portal.

Served at ``/verify`` (via ``st.navigation`` in ``app.py``). This is the one
*public* page — it is not role-gated, because the person who opens it is by
definition not yet signed in (they clicked an emailed link).

Flow: the URL carries a one-time token (``?token=…``). The page hashes it and
asks storage for the pending user; on a hit it shows a "choose a password" form,
and on submit it sets the password and marks the email verified (which clears
the token, so the link can't be replayed). A missing/unknown/expired token shows
an error and nothing else.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from agentic_learning_portal.auth import get_storage

STUDENT_PAGE = str(Path(__file__).parent / "student.py")

# Set once the password is saved and the email marked verified, so the success
# screen survives the post-submit rerun even though the token is now gone.
_VERIFY_DONE = "_verify_done"


def _finish_verification(user_id: int, password: str) -> None:
    """Set the user's password and mark their email verified (single-use token)."""
    storage = get_storage()
    storage.set_password(user_id, password)
    storage.complete_email_verification(user_id)


def render_verify_page() -> None:
    st.title("🔐 Verify your account")

    # Success branch runs on the rerun after a good submit, when the token has
    # already been consumed — so check it before reading the (now-cleared) token.
    if st.session_state.get(_VERIFY_DONE):
        st.success("Account verified! You can now sign in.")
        st.balloons()
        if st.button("← Go to sign in"):
            st.switch_page(STUDENT_PAGE)
        st.stop()

    token = st.query_params.get("token")
    if not token:
        st.error("This verification link is missing its token.")
        st.caption("Ask the admin who invited you to resend the invite.")
        st.stop()

    storage = get_storage()
    user = storage.get_user_by_verification_token(token)
    if user is None:
        st.error("This verification link is invalid or has expired.")
        st.caption("Ask the admin who invited you to send a new invite.")
        st.stop()

    st.success(
        f"Welcome, **{user.username}**"
        + (f" ({user.email})" if user.email else "")
        + "! Choose a password to finish setting up your account."
    )

    with st.form("verify_form"):
        password = st.text_input("Password", type="password")
        confirm = st.text_input("Confirm password", type="password")
        submitted = st.form_submit_button("Set password & verify", type="primary")
        if submitted:
            if not password or not confirm:
                st.error("Enter a password and confirm it.")
            elif password != confirm:
                st.error("Passwords do not match.")
            else:
                _finish_verification(user.id, password)
                st.session_state[_VERIFY_DONE] = True
                st.rerun()


render_verify_page()
"""Sign-in endpoint of the Agentic Learning Portal.

Served at ``/``: this is the app's front door and the **only** page marked
``default=True``. That matters beyond routing — a ``st.Page`` with
``default=True`` reports an empty ``url_path``, so whichever page holds the flag
loses its named URL (``/admin`` would stop resolving). Keeping the flag on this
always-public page means every other page keeps a real, linkable URL.

The page is public on purpose: whoever opens it is by definition not signed in
yet. Signed out, it renders the username/password form; already signed in,
landing on a login form would be absurd, so it forwards to the page the user's
roles land on.
"""

from __future__ import annotations

import streamlit as st

from agentic_learning_portal.auth import (
    current_user,
    landing_page_path,
    render_login_form,
)


def render_signin_page() -> None:
    """Render the login form, or forward a signed-in visitor to their landing page."""
    user = current_user()
    if user is not None:
        # ``switch_page`` ends this script run, so nothing below renders.
        st.switch_page(landing_page_path(user))

    st.title("🎓 Agentic Learning Portal")
    if render_login_form() is not None:
        # Re-run so the page renders as the now-authorized user. The queued
        # remember-me cookie is written on that run — ``auth.flush_remember_me``
        # runs in ``app.py`` before anything else.
        st.rerun()


render_signin_page()

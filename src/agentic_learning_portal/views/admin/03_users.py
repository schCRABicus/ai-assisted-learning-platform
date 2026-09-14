"""Users management endpoint of the Agentic Learning Portal.

Served at ``/users`` (via ``st.navigation`` in ``app.py``). The page lists every
user — username, roles, email, and verification status — newest first, each in a
bordered card with an **✏️ Edit** button. **➕ Invite user** opens the invite
dialog (``views/components/create_user_dialog.create_user_dialog``), which
creates an unverified user and surfaces a one-time verification link (dev-stub
mailer). Edit opens
``views/components/edit_user_dialog.edit_user_dialog``.

No LLM calls, no threading — just storage reads/writes.
"""

from __future__ import annotations

import streamlit as st

from agentic_learning_portal.auth import get_storage, require_roles
from agentic_learning_portal.views.components.create_user_dialog import (
    create_user_dialog,
)
from agentic_learning_portal.views.components.edit_user_dialog import edit_user_dialog


def _clear_user_dialogs() -> None:
    """Close any open user dialog (Streamlit allows only one ``st.dialog``/run)."""
    st.session_state.pop("create_user_open", None)
    st.session_state.pop("edit_user_open", None)


@require_roles("admin", "teacher")
def render_users_page() -> None:
    st.title("👥 Users")
    st.caption(
        "Manage portal users. Invited users must verify their email before signing in."
    )

    storage = get_storage()

    col_title, col_add = st.columns([5, 1], vertical_alignment="center")
    with col_add:
        if st.button(
            "➕ Invite user",
            type="primary",
            key="users_invite",
            use_container_width=True,
        ):
            _clear_user_dialogs()
            st.session_state["create_user_open"] = True
            st.rerun()

    users = storage.list_users()
    if not users:
        st.info("No users yet. Use ➕ Invite user to add one.")
    else:
        for user in users:
            with st.container(border=True):
                col_name, col_edit = st.columns([5, 1], vertical_alignment="center")
                with col_name:
                    st.markdown(f"### {user.username}")
                    verified = "✅ verified" if user.email_verified else "⏳ pending verification"
                    st.markdown(f"**Roles:** {', '.join(user.roles)}")
                    st.markdown(f"**Email:** {user.email or '—'} · {verified}")
                    st.caption(f"id {user.id} · created {user.created_at}")
                with col_edit:
                    if st.button(
                        "✏️ Edit",
                        key=f"edit_user_{user.id}",
                        use_container_width=True,
                    ):
                        _clear_user_dialogs()
                        st.session_state["edit_user_open"] = user.id
                        st.rerun()

    # User dialogs overlay the page while their open flag is set. Only one may
    # be open per run; each closes itself by clearing its flag when done.
    if st.session_state.get("create_user_open"):
        create_user_dialog()
    elif st.session_state.get("edit_user_open") is not None:
        edit_user_dialog(st.session_state["edit_user_open"])


render_users_page()
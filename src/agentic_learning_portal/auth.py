"""Authentication and authorization for the Streamlit portal pages.

All identity lives in ``st.session_state`` (there is no server-side session),
so the guards are safe across Streamlit's stateless script reruns. Credentials
are checked against a :class:`Storage` backend (salted scrypt hashes, see
``storage/security.py``), whose admin is seeded from ``ADMIN_USERNAME`` /
``ADMIN_PASSWORD`` on construction.

Which backend ``get_storage`` builds is selected by ``STORAGE_BACKEND``:
``"sqlite"`` (default; file-backed, shared across threads and runs) or
``"memory"`` (one shared :class:`InMemoryStorage` for the whole process —
instant, nothing persisted, for tests and throwaway use).

Page guards:
- ``require_roles("admin", "teacher")`` gates admin pages (task generation).
- ``require_roles()`` with no roles gates student pages for any authenticated user.

Role-based on purpose: the guards never inspect *how* a user was authenticated,
so the login mechanism is a swap-in seam. Today it's local username/password;
an OIDC provider could replace ``login``/``verify_credentials`` later without
touching any page.
"""

from __future__ import annotations

import os
import threading

import streamlit as st

from agentic_learning_portal.storage import InMemoryStorage, RoleName, SqliteStorage, Storage, User

# Where the portal's SQLite database lives (used when ``STORAGE_BACKEND`` is
# ``"sqlite"``). File-backed so users, the seeded admin, and password hashes
# survive Streamlit's per-run process, and so a script rerun logs in against
# the same data. Override with PORTAL_DB_PATH.
DEFAULT_DB_PATH = "portal.db"

# A ``SqliteStorage`` must be used on the thread that constructed it (Python
# 3.14 removed the post-connect ``check_same_thread`` switch), and Streamlit may
# run a session's script runs on different threads — so keep one storage per
# thread, all backed by the same ``PORTAL_DB_PATH`` file. ``_STORAGES`` maps
# thread id → instance so :func:`reset_storage` can drop every instance.
_STORAGES: dict[int, Storage] = {}

# In memory mode the portal hands out one process-wide ``InMemoryStorage``:
# unlike SQLite there is no per-thread connection to worry about, and the store
# must be shared so a user created on one thread (a test's main thread) is
# visible to a page script running on another (AppTest's). ``reset_storage``
# drops it for test isolation.
_MEMORY_STORAGE: InMemoryStorage | None = None


def get_storage() -> Storage:
    """Return the :class:`Storage` the portal should use.

    The backend is chosen by ``STORAGE_BACKEND`` (``"sqlite"`` or ``"memory"``),
    read at call time so tests and ``.env`` can switch it after import. The
    sqlite backend keeps one instance per thread (``seed_admin_from_env`` runs
    on construction, so the ``.env``-configured admin is available from the
    start); the memory backend is a single shared instance.
    """
    # Read the env var at call time so tests (and .env) can switch backends
    # after ``auth`` is imported.
    if os.getenv("STORAGE_BACKEND", "sqlite") == "memory":
        global _MEMORY_STORAGE
        if _MEMORY_STORAGE is None:
            _MEMORY_STORAGE = InMemoryStorage()
        return _MEMORY_STORAGE

    ident = threading.get_ident()
    storage = _STORAGES.get(ident)
    if storage is None:
        storage = SqliteStorage(os.getenv("PORTAL_DB_PATH", DEFAULT_DB_PATH))
        _STORAGES[ident] = storage
    return storage


def reset_storage() -> None:
    """Forget every storage instance (tests use this for isolation).

    The in-memory backend is dropped wholesale. For sqlite, only the instance
    bound to the calling thread is closed explicitly; the others are dropped and
    closed by garbage collection when their thread ends (a connection can only
    be closed from the thread that created it). The next ``get_storage``
    re-opens the DB, re-seeding the admin from env.
    """
    global _MEMORY_STORAGE
    _MEMORY_STORAGE = None
    ident = threading.get_ident()
    current = _STORAGES.pop(ident, None)
    if current is not None:
        current.close()
    _STORAGES.clear()


def current_user() -> User | None:
    """Return the logged-in user for this browser session, or ``None``."""
    return st.session_state.get("user")


def login(username: str, password: str) -> User | None:
    """Verify credentials against storage and remember the user in the session.

    Returns the ``User`` on success (already stored in session state), or
    ``None`` when the credentials don't match any user.
    """
    user = get_storage().verify_credentials(username, password)
    if user is not None:
        st.session_state["user"] = user
    return user


def logout() -> None:
    """Forget the current user and reload the page."""
    st.session_state.pop("user", None)
    st.rerun()


def render_login_form() -> User | None:
    """Render the username/password sign-in form.

    Returns the authenticated ``User`` on a successful submit (the caller is
    expected to ``st.rerun()`` so the page re-renders as authorized), or
    ``None`` when the form wasn't submitted or the credentials were wrong.
    """
    st.subheader("Sign in")
    st.caption("Log in to access the portal.")
    with st.form("portal_login"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Sign in", type="primary")
        if submitted:
            if not username or not password:
                st.error("Enter a username and password.")
                return None
            user = login(username, password)
            if user is None:
                st.error("Invalid username or password.")
                return None
            return user
    return None


def render_sidebar_user() -> None:
    """Show the signed-in user's name and a Log out button in the sidebar."""
    user = current_user()
    if user is None:
        return
    with st.sidebar:
        st.caption(f"Signed in as **{user.username}**")
        if st.button("Log out"):
            logout()


def require_roles(*roles: RoleName) -> User:
    """Gate the current page behind authentication and role membership.

    With no ``roles``, any authenticated user passes (student pages). With
    roles given (e.g. ``"admin", "teacher"``), the user must hold at least one.
    When not signed in, renders the login form; when signed in but lacking a
    required role, renders an access-denied message. Both cases stop the page,
    so nothing below the guard runs. Returns the authenticated ``User`` on
    success so pages can greet/identify them.
    """
    user = current_user()
    if user is None:
        if render_login_form() is not None:
            st.rerun()
        st.stop()
    if roles and not any(role in user.roles for role in roles):
        st.error(
            "Access denied — you need one of these roles to view this page: "
            + ", ".join(roles)
            + "."
        )
        st.stop()
    return user

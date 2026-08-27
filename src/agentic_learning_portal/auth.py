"""Authentication and authorization for the Streamlit portal views.

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
- ``require_roles("admin", "teacher")`` gates admin views (task generation).
- ``require_roles()`` with no roles gates student views for any authenticated user.

Role-based on purpose: the guards never inspect *how* a user was authenticated,
so the login mechanism is a swap-in seam. Today it's local username/password;
an OIDC provider could replace ``login``/``verify_credentials`` later without
touching any page.

Remember-me ("stay signed in") works through a browser cookie
(``REMEMBER_ME_COOKIE_NAME``) that auto-logs-in a fresh session:

- **Reading** goes through ``st.context.cookies`` — the cookies the browser
  actually sent in the request that opened the session. This is synchronous and
  available on the very first run, so a returning visitor is signed in before
  anything renders. (The old approach asked a custom component to round-trip
  the cookie to Python, which never resolved on a fresh session.)
- **Writing** (and clearing) the cookie goes through
  ``streamlit_cookies_controller.CookieController`` — its JS writes
  ``document.cookie`` inside a same-origin iframe. Because the iframe executes
  asynchronously, the write is *deferred* to a run that completes without an
  ``st.rerun()`` (see ``_flush_remember_me``); doing it on the login run would
  race the post-login rerun and tear the iframe down before the cookie lands.

A per-session ``_FORCE_LOGGED_OUT`` flag keeps the logout stick: once a user
logs out, the remember-me cookie must not re-authenticate them on a later run
of the same session (``st.context.cookies`` is immutable for the whole session,
so without the flag logout would immediately sign them back in).
"""

from __future__ import annotations

import functools
from typing import Callable

import streamlit as st

from streamlit_cookies_controller import CookieController

from agentic_learning_portal.storage import RoleName, Storage, User
from agentic_learning_portal.storage.factory import StorageFactory

STORAGE_FACTORY = StorageFactory()

REMEMBER_ME_COOKIE_NAME = "remember_me_logged_in_user"
REMEMBER_ME_MAX_AGE = 7 * 24 * 60 * 60  # 7 days, in seconds.

# Session-state flags that hand the browser-cookie write off to a later run:
# ``_PENDING_REMEMBER_ME`` carries the user id to persist, ``_CLEAR_REMEMBER_ME``
# requests the cookie be removed, and ``_FORCE_LOGGED_OUT`` blocks auto-login
# for the rest of the session after an explicit logout.
_PENDING_REMEMBER_ME = "_remember_me_pending"
_CLEAR_REMEMBER_ME = "_remember_me_clear"
_FORCE_LOGGED_OUT = "_force_logged_out"


def get_storage() -> Storage:
    """Return the :class:`Storage` the portal should use.

    The backend is chosen by ``STORAGE_BACKEND`` (``"sqlite"`` or ``"memory"``),
    read at call time so tests and ``.env`` can switch it after import. The
    sqlite backend keeps one instance per thread (``seed_admin_from_env`` runs
    on construction, so the ``.env``-configured admin is available from the
    start); the memory backend is a single shared instance.
    """
    return STORAGE_FACTORY.get_storage()


def _remember_me_user() -> User | None:
    """Return the user a browser remember-me cookie points to, or ``None``.

    Reads the cookie from ``st.context.cookies`` — the cookies the browser sent
    in the request that opened this session — so it is available synchronously
    on the very first run, before any widget renders. Returns ``None`` (never
    raises) when there is no cookie, it isn't a valid user id, or the user no
    longer exists.
    """
    user_id_str = st.context.cookies.get(REMEMBER_ME_COOKIE_NAME)
    # A real browser only ever sends ``str`` (or no cookie). Reject anything
    # else outright: under AppTest the runtime is replaced with a ``MagicMock``,
    # so ``st.context.cookies.get(...)`` hands back a truthy Mock and
    # ``int(Mock) == 1`` — which would auto-sign-in the very first user.
    if not isinstance(user_id_str, str):
        return None
    try:
        user_id = int(user_id_str)
    except ValueError:
        return None
    return get_storage().get_user(user_id)


def current_user() -> User | None:
    """Return the logged-in user for this browser session, or ``None``.

    The user is looked up in this order: an explicit sign-in in
    ``st.session_state``, then the remember-me cookie from the request that
    opened the session, then ``None``. Once found the user is cached in
    session state. An explicit logout within the same session wins over the
    (immutable) cookie, so it never signs the user back in on a later run.
    """
    if st.session_state.get(_FORCE_LOGGED_OUT):
        return None
    if st.session_state.get("user") is not None:
        return st.session_state.get("user")

    user = _remember_me_user()
    if user is not None:
        st.session_state["user"] = user
        return user
    return None


def login(username: str, password: str) -> User | None:
    """Verify credentials against storage and remember the user in the session.

    Returns the ``User`` on success (already stored in session state), or
    ``None`` when the credentials don't match any user. Success also clears the
    logged-out flag so a remember-me sign-in after a logout takes effect.
    """
    user = get_storage().verify_credentials(username, password)
    if user is not None:
        st.session_state["user"] = user
        st.session_state[_FORCE_LOGGED_OUT] = False
    return user


def _cookie_writer() -> CookieController:
    """Build the controller that writes cookies, without its read-only component.

    ``CookieController.__init__`` renders a ``getAll`` custom component (unless
    ``st.session_state["cookies"]`` already exists) whose frontend round-trips a
    value back and triggers an extra rerun — which would race the ``set`` /
    ``remove`` iframe this flush is about to render. Seeding the session key
    first skips that component; the read path is ``st.context.cookies`` anyway,
    so we never use the library's cached reads.
    """
    st.session_state.setdefault("cookies", {})
    return CookieController()


def _set_remember_me_cookie(user_id: int) -> None:
    _cookie_writer().set(
        REMEMBER_ME_COOKIE_NAME, user_id, max_age=REMEMBER_ME_MAX_AGE
    )


def _clear_remember_me_cookie() -> None:
    _cookie_writer().remove(REMEMBER_ME_COOKIE_NAME)


def _flush_remember_me() -> None:
    """Apply pending remember-me cookie writes/clears on a run that completes.

    The cookie is written (or removed) by a JS iframe — a custom component whose
    frontend executes asynchronously. Creating that element *here*, on a run
    that finishes without an ``st.rerun()``, gives the browser time to execute
    the write before the element is torn down. Calling it inside the login or
    logout run would race the guard's immediate ``st.rerun()`` and the cookie
    would never land in the browser (the bug this fixes).

    Called at the top of every guarded page's run, so the pending action is
    applied on the first clean run after it was queued.
    """
    pending_id = st.session_state.pop(_PENDING_REMEMBER_ME, None)
    if pending_id is not None:
        _set_remember_me_cookie(pending_id)
    if st.session_state.pop(_CLEAR_REMEMBER_ME, False):
        _clear_remember_me_cookie()


def logout() -> None:
    """Forget the current user, clear the remember-me cookie, and reload."""
    st.session_state.pop("user", None)
    st.session_state[_PENDING_REMEMBER_ME] = None
    st.session_state[_CLEAR_REMEMBER_ME] = True
    st.session_state[_FORCE_LOGGED_OUT] = True
    st.rerun()


def render_login_form() -> User | None:
    """Render the username/password sign-in form.

    Returns the authenticated ``User`` on a successful submit (the caller is
    expected to ``st.rerun()`` so the page re-renders as authorized), or
    ``None`` when the form wasn't submitted or the credentials were wrong.

    With "Remember me" checked, the browser cookie is *not* written here — it
    would race the post-login ``st.rerun()``. Instead the user id is queued in
    session state and ``_flush_remember_me`` writes it on the next clean run.
    """
    st.subheader("Sign in")
    st.caption("Log in to access the portal.")
    with st.form("portal_login"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        remember_me = st.checkbox("Remember me")
        submitted = st.form_submit_button("Sign in", type="primary")
        if submitted:
            if not username or not password:
                st.error("Enter a username and password.")
                return None
            user = login(username, password)
            if user is None:
                st.error("Invalid username or password.")
                return None
            if remember_me:
                # Deferred to the post-login run (see ``_flush_remember_me``).
                st.session_state[_PENDING_REMEMBER_ME] = user.id
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


def require_roles(*roles: RoleName) -> Callable:
    """Gate the current page behind authentication and role membership.

    With no ``roles``, any authenticated user passes (student views). With
    roles given (e.g. ``"admin", "teacher"``), the user must hold at least one.
    When not signed in, renders the login form; when signed in but lacking a
    required role, renders an access-denied message. Both cases stop the page,
    so nothing below the guard runs. Returns the authenticated ``User`` on
    success so views can greet/identify them.
    """
    def require_roles_decorator(func: Callable) -> Callable:

        @functools.wraps(func)
        def func_wrapper(*args, **kwargs):
            # Apply any queued remember-me cookie write/clear before the guard
            # runs, so the browser gets the cookie on a run that completes.
            _flush_remember_me()

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

            result = func(*args, **kwargs)

            return result

        return func_wrapper

    return require_roles_decorator

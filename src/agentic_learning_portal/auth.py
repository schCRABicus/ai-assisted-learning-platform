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

Remember-me ("stay signed in") is a browser cookie
(``REMEMBER_ME_COOKIE_NAME``) holding an **opaque session token** — never a
user id:

- **The token** is minted by ``Storage.create_session`` and only its SHA-256 is
  stored, so the cookie is not a credential anyone can guess or forge. Reading
  it goes through ``st.context.cookies`` — the cookies the browser sent in the
  request that opened the session — which is synchronous and available on the
  very first run, so a returning visitor is signed in before anything renders.
- **Writing** (and clearing) it goes through a ``<script>`` emitted by
  ``st.html(..., unsafe_allow_javascript=True)``, which executes in the main
  document. This replaces ``streamlit_cookies_controller``, whose bundled JS
  serializer required a real ``Date`` for ``expires`` while its Python layer
  always passed it an ISO-8601 string — so every write threw before
  ``document.cookie`` was assigned and the cookie was silently never stored.

Because the browser can only be asked to store the cookie once a run has been
sent, and because the token has to outlive the rerun that follows a login or
logout, the write is *deferred* to a run that completes without an
``st.rerun()`` (see ``_flush_remember_me``).

A per-session ``_FORCE_LOGGED_OUT`` flag keeps the logout stick: once a user
logs out, the remember-me cookie must not re-authenticate them on a later run
of the same session (``st.context.cookies`` is immutable for the whole session,
so without the flag logout would immediately sign them back in).
"""

from __future__ import annotations

import functools
import json
from pathlib import Path
from typing import Callable

import streamlit as st

from agentic_learning_portal.storage import RoleName, Storage, User
from agentic_learning_portal.storage.factory import StorageFactory

STORAGE_FACTORY = StorageFactory()

REMEMBER_ME_COOKIE_NAME = "remember_me_logged_in_user"
REMEMBER_ME_MAX_AGE = 7 * 24 * 60 * 60  # 7 days, in seconds.

# Roles that see the admin side of the portal (kept here so the nav and the page
# guards can't drift apart).
ADMIN_ROLES: tuple[RoleName, ...] = ("admin", "teacher")

# Session-state flags that hand the browser-cookie write off to a later run:
# ``_PENDING_REMEMBER_ME`` carries the session token to persist, ``_CLEAR_REMEMBER_ME``
# requests the cookie be removed, and ``_FORCE_LOGGED_OUT`` blocks auto-login
# for the rest of the session after an explicit logout. ``_SESSION_TOKEN`` holds
# the raw token of the session this browser is using, so ``logout`` can revoke it.
_PENDING_REMEMBER_ME = "_remember_me_pending"
_CLEAR_REMEMBER_ME = "_remember_me_clear"
_FORCE_LOGGED_OUT = "_force_logged_out"
_SESSION_TOKEN = "session_token"


def get_storage() -> Storage:
    """Return the :class:`Storage` the portal should use.

    The backend is chosen by ``STORAGE_BACKEND`` (``"sqlite"`` or ``"memory"``),
    read at call time so tests and ``.env`` can switch it after import. The
    sqlite backend keeps one instance per thread (``seed_admin_from_env`` runs
    on construction, so the ``.env``-configured admin is available from the
    start); the memory backend is a single shared instance.
    """
    return STORAGE_FACTORY.get_storage()


def _remember_me_session() -> tuple[User, str] | None:
    """Resolve the remember-me cookie to a user and the token that proved it.

    Reads the cookie from ``st.context.cookies`` — the cookies the browser sent
    in the request that opened this session — so it is available synchronously
    on the very first run, before any widget renders, and returns ``None``
    (never raises) when there is no cookie or nothing live matches it. An
    unknown, revoked, and expired token are indistinguishable to the caller on
    purpose: all three mean "sign in again".
    """
    token = st.context.cookies.get(REMEMBER_ME_COOKIE_NAME)
    # A real browser only ever sends ``str`` (or no cookie). Reject anything
    # else outright: under AppTest the runtime is replaced with a ``MagicMock``,
    # so ``st.context.cookies.get(...)`` hands back a truthy Mock rather than
    # ``None``, and that Mock would flow on into ``hash_token``.
    if not isinstance(token, str) or not token:
        return None
    user = get_storage().get_user_by_session_token(token)
    return None if user is None else (user, token)


def current_user() -> User | None:
    """Return the logged-in user for this browser session, or ``None``.

    The user is looked up in this order: an explicit sign-in in
    ``st.session_state``, then the remember-me session from the request that
    opened the session, then ``None``. Once found the user is cached in
    session state, along with the token that proves it so ``logout`` can revoke
    it. An explicit logout within the same session wins over the (immutable)
    cookie, so it never signs the user back in on a later run.
    """
    if st.session_state.get(_FORCE_LOGGED_OUT):
        return None
    if st.session_state.get("user") is not None:
        return st.session_state.get("user")

    session = _remember_me_session()
    if session is not None:
        user, token = session
        st.session_state["user"] = user
        st.session_state[_SESSION_TOKEN] = token
        return user
    return None


def login(username: str, password: str, *, remember_me: bool = False) -> User | None:
    """Verify credentials against storage and remember the user in the session.

    Returns the ``User`` on success (already stored in session state), or
    ``None`` when the credentials don't match any user. Success also clears the
    logged-out flag so a sign-in after a logout takes effect.

    With ``remember_me``, a session is opened and its token queued for the
    browser cookie. See ``_flush_remember_me`` for why the cookie write itself
    waits for a later run.
    """
    user = get_storage().verify_credentials(username, password)
    if user is None:
        return None

    st.session_state["user"] = user
    st.session_state[_FORCE_LOGGED_OUT] = False
    if remember_me:
        # Mint the token now — so a failed cookie write still leaves the session
        # revocable — but let ``_flush_remember_me`` hand it to the browser on
        # the next run that completes.
        token = get_storage().create_session(user.id, ttl_seconds=REMEMBER_ME_MAX_AGE)
        st.session_state[_PENDING_REMEMBER_ME] = token
        st.session_state[_SESSION_TOKEN] = token
    return user


def _render_cookie_write(name: str, value: str, *, max_age: int) -> None:
    """Ask the browser to store ``name=value`` for ``max_age`` seconds.

    Emitted as a ``<script>`` through ``st.html(..., unsafe_allow_javascript=True)``,
    which the frontend runs in the **main document** rather than inside a
    component iframe — so the assignment doesn't depend on an iframe finishing
    its load, and ``streamlit.components.v1.html`` (past its removal date) is
    avoided. The flag is only safe because nothing here is user-supplied: the
    value is a session token we minted, and both strings go through
    ``json.dumps`` so neither can break out of the literal.

    ``max_age=0`` clears the cookie. There is no way to set ``HttpOnly`` from
    JavaScript — the cookie is readable by any script on the page — which is
    exactly why the value is an opaque, revocable token and not an identity.
    """
    st.html(
        f"<script>document.cookie = {json.dumps(name)} + '=' + {json.dumps(value)}"
        f" + '; path=/; max-age={int(max_age)}; SameSite=Lax';</script>",
        unsafe_allow_javascript=True,
    )


def flush_remember_me() -> None:
    """Apply a pending remember-me cookie write or clear.

    The cookie can only be handed to the browser as part of a run's output, so
    the action is queued by ``login``/``logout`` and emitted here on a run that
    finishes without an ``st.rerun()``. Doing it inline would race that rerun:
    the rerun replaces the page's contents, tearing the ``<script>`` element
    down before the browser has run it — which is why the cookie silently never
    landed before.

    Called from ``app.py`` on every run (so it also fires on the public sign-in
    page, where no guard runs) and again from ``require_roles``, where the
    second call is a no-op because the pending flags were already popped.
    """
    token = st.session_state.pop(_PENDING_REMEMBER_ME, None)
    if token:
        _render_cookie_write(
            REMEMBER_ME_COOKIE_NAME, token, max_age=REMEMBER_ME_MAX_AGE
        )
    if st.session_state.pop(_CLEAR_REMEMBER_ME, False):
        _render_cookie_write(REMEMBER_ME_COOKIE_NAME, "", max_age=0)


def logout() -> None:
    """Forget the current user, revoke their session, and reload.

    The session is deleted server-side, not merely cleared from the browser: a
    copy of the cookie left on disk (or in another tab) is dead the moment the
    token is gone, so logout actually ends the session rather than just hiding
    it from this browser.
    """
    token = st.session_state.pop(_SESSION_TOKEN, None)
    if token is not None:
        get_storage().delete_session(token)
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

    With "Remember me" checked, ``login`` opens a session and queues its token;
    the browser cookie itself is written on the next clean run (see
    ``_flush_remember_me``).
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
            user = login(username, password, remember_me=remember_me)
            if user is None:
                st.error("Invalid username or password.")
                return None
            return user
    return None


def landing_page_path(user: User) -> str:
    """Path of the page ``user`` should land on after signing in.

    Admins and teachers land on the admin page, everyone else on the student
    one. A user holding both an admin role and ``student`` lands on the admin
    side, matching the nav's precedence.
    """
    views = Path(__file__).parent / "views"
    if any(role in user.roles for role in ADMIN_ROLES):
        return str(views / "admin.py")
    return str(views / "student.py")


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
            # ``app.py`` already flushed this run; the second call is a no-op
            # because the pending flags were popped there. Kept so a page driven
            # directly (an AppTest guard page, say) still applies the cookie.
            flush_remember_me()

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

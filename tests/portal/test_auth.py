"""AppTest-driven tests for the portal's auth guards.

Each test drives a tiny guard-only page through ``streamlit.testing.v1.AppTest``
rather than the full ``app.py``, so the assertions target ``require_roles``
behaviour directly. The storage is pointed at a fresh file DB (``PORTAL_DB_PATH``)
and seeded with an admin from env vars, so login runs against real hashed
credentials end to end.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from streamlit.testing.v1 import AppTest

from agentic_learning_portal import auth
from agentic_learning_portal.auth import (
    REMEMBER_ME_COOKIE_NAME,
    REMEMBER_ME_MAX_AGE,
    get_storage,
)


def _guarded_page(*roles: str) -> None:
    """AppTest page: guard with ``require_roles(*roles)`` then render.

    ``AppTest.from_function`` copies just this function's source into a temp
    script, so everything it needs — including ``st`` and ``roles`` — must be
    imported locally or passed via ``from_function(args=...)``.
    """

    from agentic_learning_portal.auth import require_roles, current_user

    @require_roles(*roles)
    def page() -> None:
        import streamlit as st

        user = current_user()
        st.markdown(f"PAGE_RENDERED as {user.username}")

    return page()


def _page_with_logout() -> None:
    """AppTest page with a guarded page plus the sidebar Log out button."""

    from agentic_learning_portal.auth import require_roles, render_sidebar_user, current_user
    import streamlit as st

    @require_roles()
    def page() -> None:
        render_sidebar_user()
        user = current_user()
        st.markdown(f"PAGE_RENDERED as {user.username}")

    return page()


def _admin_page() -> AppTest:
    return AppTest.from_function(_guarded_page, args=("admin", "teacher"))


def _student_page() -> AppTest:
    return AppTest.from_function(_guarded_page, args=())


def _logout_page() -> AppTest:
    return AppTest.from_function(_page_with_logout)


def _login_remember_me(at: AppTest, username: str, password: str) -> AppTest:
    at.checkbox[0].check()
    at.text_input[0].set_value(username)
    at.text_input[1].set_value(password)
    at.button[0].click().run()
    return at


def _login(at: AppTest, username: str, password: str) -> AppTest:
    at.text_input[0].set_value(username)
    at.text_input[1].set_value(password)
    at.button[0].click().run()
    return at


def _rendered(at: AppTest) -> bool:
    return any("PAGE_RENDERED" in m.value for m in at.markdown)


def test_anonymous_sees_login_form_not_content(portal_env) -> None:
    at = _admin_page().run()

    assert any(t.label == "Username" for t in at.text_input)
    assert any(t.label == "Password" for t in at.text_input)
    assert not _rendered(at)


def test_wrong_password_is_rejected(portal_env) -> None:
    at = _admin_page().run()
    _login(at, "boss", "not-the-password")

    assert any("Invalid username or password" in e.value for e in at.error)
    assert not _rendered(at)


def test_admin_can_access_admin_page(portal_env) -> None:
    at = _admin_page().run()
    _login(at, "boss", "hunter2")

    assert _rendered(at)


def test_teacher_can_access_admin_page(portal_env) -> None:
    get_storage().create_user("tess", "teacher", password="pw123")
    at = _admin_page().run()
    _login(at, "tess", "pw123")

    assert _rendered(at)


def test_student_cannot_access_admin_page(portal_env) -> None:
    get_storage().create_user("pupil", "student", password="pw123")
    at = _admin_page().run()
    _login(at, "pupil", "pw123")

    assert any("Access denied" in e.value for e in at.error)
    assert not _rendered(at)


def test_student_can_access_student_page(portal_env) -> None:
    get_storage().create_user("pupil", "student", password="pw123")
    at = _student_page().run()
    _login(at, "pupil", "pw123")

    assert _rendered(at)


# --- remember-me sessions ----------------------------------------------------
#
# The cookie holds an opaque session token, never a user id: the id would be
# forgeable (writing ``remember_me_logged_in_user=1`` signed you in as the
# admin), and a token can be revoked server-side when the user logs out.


def _with_cookie(value: object) -> SimpleNamespace:
    """A stand-in for ``st.context`` whose only cookie is the remember-me one."""
    return SimpleNamespace(cookies={REMEMBER_ME_COOKIE_NAME: value})


def _mint_session(
    username: str = "boss", *, ttl_seconds: int = REMEMBER_ME_MAX_AGE
) -> str:
    """Open a real session for ``username`` and return its raw token."""
    user = get_storage().get_user_by_username(username)
    return get_storage().create_session(user.id, ttl_seconds=ttl_seconds)


def test_remember_me_session_resolves_a_live_token(portal_env) -> None:
    """A live session token from the request's cookies resolves to its user."""
    admin = get_storage().get_user_by_username("boss")
    token = _mint_session()

    with patch.object(auth.st, "context", _with_cookie(token)):
        assert auth._remember_me_session() == (admin, token)


def test_forged_user_id_cookie_signs_nobody_in(portal_env) -> None:
    """Regression: the cookie used to *be* a user id, so it was forgeable.

    Setting ``remember_me_logged_in_user=<id>`` by hand signed you in as that
    user — admin included — because the read path fed the value straight to
    ``int()`` and ``get_user``. Now the value only ever matches a stored session
    token hash, so an id resolves to nothing.
    """
    admin = get_storage().get_user_by_username("boss")

    with patch.object(auth.st, "context", _with_cookie(str(admin.id))):
        assert auth._remember_me_session() is None

    with patch.object(auth.st, "context", _with_cookie("1")):
        assert auth._remember_me_session() is None


def test_remember_me_session_ignores_a_non_string_cookie(portal_env) -> None:
    """Anything that isn't a ``str`` is rejected before it reaches storage.

    Under AppTest the runtime is a ``MagicMock``, so ``st.context.cookies.get``
    hands back a truthy Mock rather than ``None``; letting one through would
    send a Mock on to ``hash_token`` (and, under the old integer scheme,
    ``int(Mock) == 1`` auto-signed-in the first user).
    """
    for value in (MagicMock(), 1, None, ""):
        with patch.object(auth.st, "context", _with_cookie(value)):
            assert auth._remember_me_session() is None


def test_remember_me_session_returns_none_without_cookie(portal_env) -> None:
    """No remember-me cookie in the request -> no auto-login."""
    with patch.object(auth.st, "context", SimpleNamespace(cookies={})):
        assert auth._remember_me_session() is None


def test_expired_session_token_signs_nobody_in(portal_env) -> None:
    """A token past its expiry is refused even though the row still exists."""
    token = _mint_session(ttl_seconds=-1)

    with patch.object(auth.st, "context", _with_cookie(token)):
        assert auth._remember_me_session() is None


def test_revoked_session_token_signs_nobody_in(portal_env) -> None:
    """Deleting the session kills the cookie that carried it."""
    token = _mint_session()
    get_storage().delete_session(token)

    with patch.object(auth.st, "context", _with_cookie(token)):
        assert auth._remember_me_session() is None


def test_remember_me_auto_logs_in_on_fresh_session(portal_env) -> None:
    """A returning session with a live cookie is signed in before anything renders."""
    token = _mint_session()

    with patch.object(auth.st, "context", _with_cookie(token)):
        at = _admin_page().run()

    assert _rendered(at)
    assert not any(t.label == "Username" for t in at.text_input)


def test_remember_me_login_mints_a_session(portal_env) -> None:
    """Checking "Remember me" opens a session and keeps its raw token in session.

    The token must be the opaque one storage minted — it is what ``logout``
    needs in order to revoke the session server-side.
    """
    admin = get_storage().get_user_by_username("boss")
    at = _admin_page().run()
    _login_remember_me(at, "boss", "hunter2")

    assert _rendered(at)
    token = at.session_state["session_token"]
    assert token != str(admin.id)
    assert get_storage().get_user_by_session_token(token) == admin


def test_remember_me_login_writes_the_cookie_on_a_clean_run(portal_env) -> None:
    """The cookie is handed to the browser once, on the run after the login.

    Not during the login run itself: that run ends in ``st.rerun()``, which
    replaces the page and would tear the ``<script>`` element down before the
    browser executed it — the bug that made remember-me silently never work.
    """
    at = _admin_page().run()
    with patch.object(auth, "_render_cookie_write") as write:
        _login_remember_me(at, "boss", "hunter2")

    assert _rendered(at)
    write.assert_called_once()
    name, token = write.call_args.args
    assert name == REMEMBER_ME_COOKIE_NAME
    assert write.call_args.kwargs == {"max_age": REMEMBER_ME_MAX_AGE}
    # What was written is a live session token, not an identity.
    assert get_storage().get_user_by_session_token(token) is not None


def test_remember_me_not_checked_writes_no_cookie(portal_env) -> None:
    """Without the checkbox, no remember-me cookie is written and no session opens."""
    at = _admin_page().run()
    with patch.object(auth, "_render_cookie_write") as write:
        _login(at, "boss", "hunter2")

    assert _rendered(at)
    write.assert_not_called()
    assert "session_token" not in at.session_state


def test_logout_revokes_the_session_server_side(portal_env) -> None:
    """Logout deletes the session, so the cookie still on disk is dead."""
    at = _logout_page().run()
    _login_remember_me(at, "boss", "hunter2")
    assert _rendered(at)
    token = at.session_state["session_token"]
    assert get_storage().get_user_by_session_token(token) is not None

    with patch.object(auth, "_render_cookie_write") as write:
        at.button[0].click().run()

    assert get_storage().get_user_by_session_token(token) is None
    name, value = write.call_args.args
    assert (name, value) == (REMEMBER_ME_COOKIE_NAME, "")
    assert write.call_args.kwargs == {"max_age": 0}
    assert not _rendered(at)
    assert any(t.label == "Username" for t in at.text_input)


def test_logout_stays_logged_out_despite_the_still_present_cookie(portal_env) -> None:
    """``st.context.cookies`` is immutable for the session, so logout needs its flag.

    Clearing the cookie is asynchronous and the browser keeps sending the old
    one until it lands — without ``_FORCE_LOGGED_OUT`` the very next run would
    auto-sign the user back in.
    """
    token = _mint_session()
    at = _logout_page().run()
    _login(at, "boss", "hunter2")
    assert _rendered(at)

    with patch.object(auth.st, "context", _with_cookie(token)):
        at.button[0].click().run()

    assert not _rendered(at)
    assert any(t.label == "Username" for t in at.text_input)

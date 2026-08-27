"""AppTest-driven tests for the portal's auth guards.

Each test drives a tiny guard-only page through ``streamlit.testing.v1.AppTest``
rather than the full ``app.py``, so the assertions target ``require_roles``
behaviour directly. The storage is pointed at a fresh file DB (``PORTAL_DB_PATH``)
and seeded with an admin from env vars, so login runs against real hashed
credentials end to end.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from streamlit.testing.v1 import AppTest
from streamlit_cookies_controller import CookieController

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


# --- remember-me cookie ------------------------------------------------------


def test_remember_me_user_reads_user_id_cookie(portal_env) -> None:
    """``_remember_me_user`` resolves a cookie user id to a real user."""
    admin = get_storage().get_user_by_username("boss")
    fake_context = SimpleNamespace(cookies={REMEMBER_ME_COOKIE_NAME: str(admin.id)})
    with patch.object(auth.st, "context", fake_context):
        assert auth._remember_me_user() == admin


def test_remember_me_user_returns_none_without_cookie(portal_env) -> None:
    """No remember-me cookie in the request -> no auto-login."""
    with patch.object(auth.st, "context", SimpleNamespace(cookies={})):
        assert auth._remember_me_user() is None


def test_remember_me_user_ignores_garbage_cookie(portal_env) -> None:
    """A non-numeric cookie value never crashes auto-login."""
    fake_context = SimpleNamespace(cookies={REMEMBER_ME_COOKIE_NAME: "not-an-int"})
    with patch.object(auth.st, "context", fake_context):
        assert auth._remember_me_user() is None


def test_remember_me_user_ignores_cookie_for_unknown_user(portal_env) -> None:
    """A cookie for a deleted user falls back to anonymous."""
    fake_context = SimpleNamespace(cookies={REMEMBER_ME_COOKIE_NAME: "99999"})
    with patch.object(auth.st, "context", fake_context):
        assert auth._remember_me_user() is None


def test_remember_me_login_writes_cookie_on_clean_run(portal_env) -> None:
    """Remember-me defers the cookie write to the post-login run, then renders.

    The write must happen on a run that completes without ``st.rerun()`` (so the
    JS iframe executes before teardown) — hence exactly one ``set`` after the
    login chain settles, not during the login run itself.
    """
    admin = get_storage().get_user_by_username("boss")
    at = _admin_page().run()
    with patch.object(CookieController, "set") as mock_set:
        _login_remember_me(at, "boss", "hunter2")

    assert _rendered(at)
    mock_set.assert_called_once_with(
        REMEMBER_ME_COOKIE_NAME, admin.id, max_age=REMEMBER_ME_MAX_AGE
    )


def test_remember_me_not_checked_does_not_set_cookie(portal_env) -> None:
    """Without the checkbox, no remember-me cookie is written."""
    at = _admin_page().run()
    with patch.object(CookieController, "set") as mock_set:
        _login(at, "boss", "hunter2")

    assert _rendered(at)
    mock_set.assert_not_called()


def test_remember_me_auto_logs_in_on_fresh_session(portal_env) -> None:
    """A returning session with the cookie is signed in before anything renders."""
    admin = get_storage().get_user_by_username("boss")
    with patch("agentic_learning_portal.auth._remember_me_user", return_value=admin):
        at = _admin_page().run()

    assert _rendered(at)
    assert not any(t.label == "Username" for t in at.text_input)


def test_logout_clears_cookie_and_stays_logged_out(portal_env) -> None:
    """Logout removes the cookie and is not undone by auto-login this session."""
    admin = get_storage().get_user_by_username("boss")
    at = _logout_page().run()
    _login(at, "boss", "hunter2")
    assert _rendered(at)

    # The remember-me cookie is present for the rest of this session; logout
    # must still clear it and stay logged out despite the (immutable) cookie.
    with patch.object(CookieController, "remove") as mock_remove, patch(
            "agentic_learning_portal.auth._remember_me_user", return_value=admin
    ):
        at.button[0].click().run()

    mock_remove.assert_called_once_with(REMEMBER_ME_COOKIE_NAME)
    assert not _rendered(at)
    assert any(t.label == "Username" for t in at.text_input)

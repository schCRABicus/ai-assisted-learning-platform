"""AppTest-driven tests for the portal's auth guards.

Each test drives a tiny guard-only page through ``streamlit.testing.v1.AppTest``
rather than the full ``app.py``, so the assertions target ``require_roles``
behaviour directly. The storage is pointed at a fresh file DB (``PORTAL_DB_PATH``)
and seeded with an admin from env vars, so login runs against real hashed
credentials end to end.
"""

from __future__ import annotations

from streamlit.testing.v1 import AppTest

from agentic_learning_portal.auth import get_storage


def _guarded_page(*roles: str) -> None:
    """AppTest page: guard with ``require_roles(*roles)`` then render.

    ``AppTest.from_function`` copies just this function's source into a temp
    script, so everything it needs — including ``st`` and ``roles`` — must be
    imported locally or passed via ``from_function(args=...)``.
    """
    import streamlit as st

    from agentic_learning_portal.auth import require_roles

    user = require_roles(*roles)
    st.markdown(f"PAGE_RENDERED as {user.username}")


def _admin_page() -> AppTest:
    return AppTest.from_function(_guarded_page, args=("admin", "teacher"))


def _student_page() -> AppTest:
    return AppTest.from_function(_guarded_page, args=())


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

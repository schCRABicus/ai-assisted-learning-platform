"""AppTest-driven tests for the users management page (views/admin/03_users.py).

The page lists every user (username, roles, email, verification status) and
offers an **➕ Invite user** button plus per-user **✏️ Edit**. It is gated behind
``require_roles("admin", "teacher")``, so tests sign in with the env-seeded admin
(``portal_env``) through the login form first.
"""

from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest

from agentic_learning_portal.auth import get_storage

USERS_PAGE = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "agentic_learning_portal"
    / "views"
    / "admin"
    / "03_users.py"
)


def _login(at: AppTest) -> None:
    at.text_input[0].set_value("boss")
    at.text_input[1].set_value("hunter2")
    at.button[0].click().run()


def _find_button(at: AppTest, label: str) -> AppTest | None:
    needle = label.lower()
    for button in at.button:
        if needle in button.label.lower():
            return button
    return None


def test_users_page_requires_auth(portal_env) -> None:
    at = AppTest.from_file(str(USERS_PAGE), default_timeout=10).run()

    assert not at.exception
    assert any(t.label == "Username" for t in at.text_input)
    assert any(t.label == "Password" for t in at.text_input)


def test_users_page_lists_seeded_admin(portal_env) -> None:
    at = AppTest.from_file(str(USERS_PAGE), default_timeout=10).run()
    _login(at)

    assert not at.exception
    assert at.title[0].value == "👥 Users"
    # The env-seeded admin appears in the list.
    assert any("boss" in m.value for m in at.markdown)


def test_users_page_lists_created_users(portal_env) -> None:
    get_storage().create_user("pupil", "student", email="pupil@example.com")

    at = AppTest.from_file(str(USERS_PAGE), default_timeout=10).run()
    _login(at)

    assert not at.exception
    assert any("boss" in m.value for m in at.markdown)
    assert any("pupil" in m.value for m in at.markdown)


def test_invite_button_opens_create_dialog(portal_env) -> None:
    at = AppTest.from_file(str(USERS_PAGE), default_timeout=10).run()
    _login(at)

    invite = _find_button(at, "Invite user")
    assert invite is not None
    invite.click().run()

    assert not at.exception
    # The create-user dialog is now open, exposing its form controls.
    labels = {t.label for t in at.text_input}
    assert "Username" in labels
    assert "Email" in labels
    assert any("Create & send invite" in b.label for b in at.button)